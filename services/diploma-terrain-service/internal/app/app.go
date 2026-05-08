package app

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/TwiLightDM/diploma-terrain-service/internal/config"
	"github.com/TwiLightDM/diploma-terrain-service/internal/entities"
	"github.com/TwiLightDM/diploma-terrain-service/internal/kafka"
	terrain_service "github.com/TwiLightDM/diploma-terrain-service/internal/terrain-service"
	"github.com/TwiLightDM/diploma-terrain-service/package/databases"
	"github.com/TwiLightDM/diploma-terrain-service/proto/terrainservicepb"
	"google.golang.org/grpc"
)

func Run(cfg *config.Config) error {
	db, err := databases.Connect(cfg.Postgres.Host, cfg.Postgres.Port, cfg.Postgres.User, cfg.Postgres.Password, cfg.Postgres.DB)
	if err != nil {
		return fmt.Errorf("db connect: %w", err)
	}

	if err := db.AutoMigrate(&entities.TerrainJob{}); err != nil {
		return fmt.Errorf("automigrate: %w", err)
	}

	seed := `
		INSERT INTO generation_limits (id, user_id, daily_limit, used_today, reset_at)
		SELECT gen_random_uuid(), id, 1000000, 0, NOW() + INTERVAL '1 day'
		FROM users
		WHERE role = 'admin'
		ON CONFLICT (user_id) DO UPDATE SET daily_limit = 1000000
	`
	if err := db.Exec(seed).Error; err != nil {
		log.Printf("admin limit seed: %v", err)
	}

	producer := kafka.NewProducer(cfg.Kafka.Brokers, cfg.Kafka.RequestsTopic)
	defer producer.Close()

	repo := terrain_service.NewRepository(db)
	svc := terrain_service.NewService(repo, producer)
	handler := terrain_service.NewHandler(svc)

	consumer := kafka.NewConsumer(cfg.Kafka.Brokers, cfg.Kafka.ResultsTopic, cfg.Kafka.ConsumerGroup, svc.HandleResult)
	defer consumer.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go consumer.Run(ctx)

	lis, err := net.Listen("tcp", ":"+cfg.GRPCPort)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	srv := grpc.NewServer()
	terrainservicepb.RegisterTerrainServiceServer(srv, handler)

	go func() {
		log.Printf("terrain-service gRPC on :%s", cfg.GRPCPort)
		if err := srv.Serve(lis); err != nil {
			log.Printf("grpc serve error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("shutting down terrain-service...")
	srv.GracefulStop()
	return nil
}

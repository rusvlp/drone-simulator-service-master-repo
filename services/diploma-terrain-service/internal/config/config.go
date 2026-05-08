package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	GRPCPort string

	Postgres struct {
		Host     string
		Port     string
		User     string
		Password string
		DB       string
	}

	Kafka struct {
		Brokers       string
		RequestsTopic string
		ResultsTopic  string
		ConsumerGroup string
	}
}

func Load() *Config {
	if err := godotenv.Load(); err != nil {
		log.Println(".env not found, using environment variables")
	}

	cfg := &Config{}

	cfg.GRPCPort = getEnv("GRPC_PORT", "50053")

	cfg.Postgres.Host = getEnv("POSTGRES_HOST", "localhost")
	cfg.Postgres.Port = getEnv("POSTGRES_PORT", "5432")
	cfg.Postgres.User = getEnv("POSTGRES_USER", "postgres_user")
	cfg.Postgres.Password = getEnv("POSTGRES_PASSWORD", "")
	cfg.Postgres.DB = getEnv("POSTGRES_DB", "diploma")

	cfg.Kafka.Brokers = getEnv("KAFKA_BROKERS", "localhost:9092")
	cfg.Kafka.RequestsTopic = getEnv("KAFKA_REQUESTS_TOPIC", "terrain.requests")
	cfg.Kafka.ResultsTopic = getEnv("KAFKA_RESULTS_TOPIC", "terrain.results")
	cfg.Kafka.ConsumerGroup = getEnv("KAFKA_CONSUMER_GROUP", "terrain-service")

	return cfg
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

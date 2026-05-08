package terrain_service

import (
	"context"
	"encoding/base64"
	"errors"
	"time"

	"github.com/TwiLightDM/diploma-terrain-service/internal/entities"
	"github.com/TwiLightDM/diploma-terrain-service/internal/kafka"
	"github.com/google/uuid"
)

type Service interface {
	SubmitJob(ctx context.Context, userID string, image []byte, scaleZ float32, yUp bool, textureMode string) (string, error)
	GetJobStatus(ctx context.Context, jobID string) (*entities.TerrainJob, error)
	GetUserJobs(ctx context.Context, userID string) ([]*entities.TerrainJob, error)
	GetLimit(ctx context.Context, userID string) (*entities.GenerationLimit, error)
	SetLimit(ctx context.Context, userID string, dailyLimit int) (*entities.GenerationLimit, error)
	HandleResult(ctx context.Context, result *kafka.GenerationResult) error
}

type service struct {
	repo     Repository
	producer *kafka.Producer
}

func NewService(repo Repository, producer *kafka.Producer) Service {
	return &service{repo: repo, producer: producer}
}

func (s *service) SubmitJob(ctx context.Context, userID string, image []byte, scaleZ float32, yUp bool, textureMode string) (string, error) {
	limit, err := s.repo.GetLimit(ctx, userID)
	if err != nil {
		return "", err
	}

	now := time.Now().UTC()
	if limit != nil && !limit.ResetAt.Before(now) && limit.UsedToday >= limit.DailyLimit {
		return "", errors.New("daily generation limit reached")
	}

	jobID := uuid.NewString()
	job := &entities.TerrainJob{
		ID:     jobID,
		UserID: userID,
		Status: entities.JobStatusPending,
	}
	if err := s.repo.CreateJob(ctx, job); err != nil {
		return "", err
	}

	if err := s.repo.IncrementUsed(ctx, userID); err != nil {
		return "", err
	}

	req := &kafka.GenerationRequest{
		JobID:       jobID,
		UserID:      userID,
		ImageBase64: base64.StdEncoding.EncodeToString(image),
		ScaleZ:      scaleZ,
		YUp:         yUp,
		TextureMode: textureMode,
	}
	if err := s.producer.SendRequest(ctx, req); err != nil {
		return "", err
	}

	return jobID, nil
}

func (s *service) GetJobStatus(ctx context.Context, jobID string) (*entities.TerrainJob, error) {
	return s.repo.GetJob(ctx, jobID)
}

func (s *service) GetUserJobs(ctx context.Context, userID string) ([]*entities.TerrainJob, error) {
	return s.repo.GetJobsByUser(ctx, userID)
}

func (s *service) GetLimit(ctx context.Context, userID string) (*entities.GenerationLimit, error) {
	limit, err := s.repo.GetLimit(ctx, userID)
	if err != nil {
		return nil, err
	}
	if limit == nil {
		return &entities.GenerationLimit{
			UserID:     userID,
			DailyLimit: 10,
			UsedToday:  0,
			ResetAt:    time.Now().UTC().Add(24 * time.Hour),
		}, nil
	}
	return limit, nil
}

func (s *service) SetLimit(ctx context.Context, userID string, dailyLimit int) (*entities.GenerationLimit, error) {
	limit, err := s.repo.GetLimit(ctx, userID)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	if limit == nil {
		limit = &entities.GenerationLimit{
			ID:         uuid.NewString(),
			UserID:     userID,
			DailyLimit: dailyLimit,
			UsedToday:  0,
			ResetAt:    time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC),
		}
	} else {
		limit.DailyLimit = dailyLimit
	}
	if err := s.repo.UpsertLimit(ctx, limit); err != nil {
		return nil, err
	}
	return limit, nil
}

func (s *service) HandleResult(ctx context.Context, result *kafka.GenerationResult) error {
	job, err := s.repo.GetJob(ctx, result.JobID)
	if err != nil || job == nil {
		return err
	}
	job.Status = result.Status
	job.ModelURL = result.ModelURL
	job.TextureURL = result.TextureURL
	job.TreesURL = result.TreesURL
	job.Error = result.Error
	return s.repo.UpdateJob(ctx, job)
}

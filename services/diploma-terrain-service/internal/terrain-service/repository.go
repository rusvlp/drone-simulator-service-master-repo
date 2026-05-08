package terrain_service

import (
	"context"
	"errors"
	"time"

	"github.com/TwiLightDM/diploma-terrain-service/internal/entities"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Repository interface {
	CreateJob(ctx context.Context, job *entities.TerrainJob) error
	UpdateJob(ctx context.Context, job *entities.TerrainJob) error
	GetJob(ctx context.Context, jobID string) (*entities.TerrainJob, error)
	GetJobsByUser(ctx context.Context, userID string) ([]*entities.TerrainJob, error)

	GetLimit(ctx context.Context, userID string) (*entities.GenerationLimit, error)
	UpsertLimit(ctx context.Context, limit *entities.GenerationLimit) error
	IncrementUsed(ctx context.Context, userID string) error
}

type repository struct {
	db *gorm.DB
}

func NewRepository(db *gorm.DB) Repository {
	return &repository{db: db}
}

func (r *repository) CreateJob(ctx context.Context, job *entities.TerrainJob) error {
	return r.db.WithContext(ctx).Create(job).Error
}

func (r *repository) UpdateJob(ctx context.Context, job *entities.TerrainJob) error {
	return r.db.WithContext(ctx).Save(job).Error
}

func (r *repository) GetJob(ctx context.Context, jobID string) (*entities.TerrainJob, error) {
	var job entities.TerrainJob
	err := r.db.WithContext(ctx).Where("id = ?", jobID).First(&job).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &job, err
}

func (r *repository) GetJobsByUser(ctx context.Context, userID string) ([]*entities.TerrainJob, error) {
	var jobs []*entities.TerrainJob
	err := r.db.WithContext(ctx).Where("user_id = ?", userID).Order("created_at DESC").Limit(50).Find(&jobs).Error
	return jobs, err
}

func (r *repository) GetLimit(ctx context.Context, userID string) (*entities.GenerationLimit, error) {
	var limit entities.GenerationLimit
	err := r.db.WithContext(ctx).Where("user_id = ?", userID).First(&limit).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &limit, err
}

func (r *repository) UpsertLimit(ctx context.Context, limit *entities.GenerationLimit) error {
	return r.db.WithContext(ctx).Save(limit).Error
}

func (r *repository) IncrementUsed(ctx context.Context, userID string) error {
	now := time.Now().UTC()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)

	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var limit entities.GenerationLimit
		if err := tx.Where("user_id = ?", userID).First(&limit).Error; err != nil {
			if !errors.Is(err, gorm.ErrRecordNotFound) {
				return err
			}
			limit = entities.GenerationLimit{
				ID:         uuid.New().String(),
				UserID:     userID,
				DailyLimit: 10,
				UsedToday:  0,
				ResetAt:    today.Add(24 * time.Hour),
			}
		}

		if limit.ResetAt.Before(now) {
			limit.UsedToday = 0
			limit.ResetAt = today.Add(24 * time.Hour)
		}

		limit.UsedToday++
		return tx.Save(&limit).Error
	})
}

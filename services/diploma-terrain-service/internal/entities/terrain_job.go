package entities

import "time"

const (
	JobStatusPending    = "pending"
	JobStatusProcessing = "processing"
	JobStatusDone       = "done"
	JobStatusFailed     = "failed"
)

type TerrainJob struct {
	ID         string    `gorm:"primaryKey;type:uuid"`
	UserID     string    `gorm:"not null;type:uuid;index"`
	Status     string    `gorm:"not null;default:'pending'"`
	ModelURL   string    `gorm:"default:''"`
	TextureURL string    `gorm:"default:''"`
	TreesURL   string    `gorm:"default:''"`
	Error      string    `gorm:"default:''"`
	CreatedAt  time.Time `gorm:"autoCreateTime"`
}

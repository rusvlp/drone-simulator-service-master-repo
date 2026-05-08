package entities

import "time"

type GenerationLimit struct {
	ID         string    `gorm:"primaryKey;type:uuid"`
	UserID     string    `gorm:"uniqueIndex;not null;type:uuid"`
	DailyLimit int       `gorm:"not null;default:10"`
	UsedToday  int       `gorm:"not null;default:0"`
	ResetAt    time.Time `gorm:"not null"`
}

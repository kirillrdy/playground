package models

import (
	"time"

	"gorm.io/gorm"
)

type File struct {
	gorm.Model
	Name       string    `json:"name"`
	Path       string    `json:"path"`
	Size       int64     `json:"size"`
	RecordedAt time.Time `gorm:"column:recorded_at" json:"recorded_at"`
	UUID       string    `json:"uuid"`
	Duration   int64     `json:"duration"`
}

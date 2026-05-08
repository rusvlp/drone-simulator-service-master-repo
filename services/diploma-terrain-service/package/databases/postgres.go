package databases

import (
	"fmt"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func Connect(host, port, user, password, dbname string) (*gorm.DB, error) {
	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)
	return gorm.Open(postgres.Open(dsn), &gorm.Config{})
}

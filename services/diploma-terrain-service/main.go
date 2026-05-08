package main

import (
	"log"

	"github.com/TwiLightDM/diploma-terrain-service/internal/app"
	"github.com/TwiLightDM/diploma-terrain-service/internal/config"
)

func main() {
	cfg := config.Load()
	if err := app.Run(cfg); err != nil {
		log.Fatal(err)
	}
}

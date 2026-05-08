package kafka

import (
	"context"
	"encoding/json"
	"log"

	kafkago "github.com/segmentio/kafka-go"
)

type ResultHandler func(ctx context.Context, result *GenerationResult) error

type Consumer struct {
	reader  *kafkago.Reader
	handler ResultHandler
}

func NewConsumer(brokers, topic, group string, handler ResultHandler) *Consumer {
	r := kafkago.NewReader(kafkago.ReaderConfig{
		Brokers: []string{brokers},
		Topic:   topic,
		GroupID: group,
	})
	return &Consumer{reader: r, handler: handler}
}

func (c *Consumer) Run(ctx context.Context) {
	log.Println("[kafka consumer] started")
	for {
		msg, err := c.reader.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				log.Println("[kafka consumer] shutting down")
				return
			}
			log.Printf("[kafka consumer] read error: %v", err)
			continue
		}

		var result GenerationResult
		if err := json.Unmarshal(msg.Value, &result); err != nil {
			log.Printf("[kafka consumer] unmarshal error: %v", err)
			continue
		}

		if err := c.handler(ctx, &result); err != nil {
			log.Printf("[kafka consumer] handler error job=%s: %v", result.JobID, err)
		}
	}
}

func (c *Consumer) Close() error {
	return c.reader.Close()
}

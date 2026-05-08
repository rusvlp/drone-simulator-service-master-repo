package kafka

import (
	"context"
	"encoding/json"
	"log"

	kafkago "github.com/segmentio/kafka-go"
)

type Producer struct {
	writer *kafkago.Writer
}

func NewProducer(brokers, topic string) *Producer {
	w := &kafkago.Writer{
		Addr:      kafkago.TCP(brokers),
		Topic:     topic,
		Balancer:  &kafkago.LeastBytes{},
		BatchBytes: 20 * 1024 * 1024,
	}
	return &Producer{writer: w}
}

func (p *Producer) SendRequest(ctx context.Context, req *GenerationRequest) error {
	data, err := json.Marshal(req)
	if err != nil {
		return err
	}
	err = p.writer.WriteMessages(ctx, kafkago.Message{
		Key:   []byte(req.JobID),
		Value: data,
	})
	if err != nil {
		log.Printf("[kafka producer] send error: %v", err)
	}
	return err
}

func (p *Producer) Close() error {
	return p.writer.Close()
}

FROM golang:1.24-bookworm

WORKDIR /app

COPY . .

ENV MIGRATIONS_PATH=/app/migrations

CMD ["go", "run", "-mod=vendor", "main.go"]

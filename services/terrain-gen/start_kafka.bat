@echo off
cd /d "%~dp0"
python kafka_worker.py ^
  --brokers localhost:9093 ^
  --requests-topic terrain.requests ^
  --results-topic terrain.results ^
  --group-id terrain-gen-worker ^
  --minio-endpoint localhost:9000 ^
  --minio-access-key minio_user ^
  --minio-secret-key superStrongPassword ^
  --bucket diploma ^
  --minio-public-url http://localhost:9000

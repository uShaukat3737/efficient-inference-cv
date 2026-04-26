#!/bin/bash
set -e

cleanup() {
  echo ""
  echo "Shutting down services..."
  kill "$WORKER_PID" "$API_PID" 2>/dev/null || true
  redis-cli shutdown 2>/dev/null || true
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting Redis..."
redis-server --daemonize yes

echo "Starting worker pool (4 workers)..."
python3 -m src.serving.worker_pool --workers 4 &
WORKER_PID=$!

echo "Starting FastAPI..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

#wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "API failed to start after 30s. Check logs."
    cleanup
  fi
done

echo ""
echo "=========================================="
echo "All services running."
echo "  API:    http://127.0.0.1:8000"
echo "  Health: http://127.0.0.1:8000/health"
echo "  Press Ctrl+C to stop."
echo "=========================================="

wait "$WORKER_PID" "$API_PID"

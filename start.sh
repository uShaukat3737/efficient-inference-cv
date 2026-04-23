#!/bin/bash

echo "Starting Redis server..."
redis-server &
REDIS_PID=$!

# Give Redis a second to initialize
sleep 1

echo "Starting Inference Worker..."
python3 -m src.serving.worker &
WORKER_PID=$!

echo "Starting FastAPI Server..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo ""
echo "========================================================="
echo "All services are running!"
echo "FastAPI endpoint: http://127.0.0.1:8000"
echo "Press Ctrl+C to shut everything down gracefully."
echo "========================================================="

# Trap Ctrl+C (SIGINT) to kill all background processes
trap "echo -e '\nShutting down services...'; kill $WORKER_PID $API_PID $REDIS_PID; exit" SIGINT SIGTERM

# Wait indefinitely so the script doesn't exit, allowing the trap to catch Ctrl+C
wait $WORKER_PID $API_PID $REDIS_PID

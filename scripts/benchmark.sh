#!/bin/bash
set -e

WORKER_PID=""
API_PID=""

cleanup() {
  if [ -n "$WORKER_PID" ] || [ -n "$API_PID" ]; then
    echo ""
    echo "Stopping serving stack..."
    kill "$WORKER_PID" "$API_PID" 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    sleep 1
  fi
}

trap cleanup EXIT SIGINT SIGTERM

echo "=========================================="
echo "Efficient Inference CV — Full Benchmark"
echo "=========================================="
echo ""

#kill any leftover processes from a previous run
pkill -f "worker_pool" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
sleep 1

#export models if missing
if [ ! -f "models/exported/model.pt" ] || [ ! -f "models/exported/model.onnx" ]; then
  echo "Exported models not found — running export..."
  python3 -m scripts.export_model
  echo ""
fi

# ---- standalone benchmarks (no server needed) ----
echo "--- [1/5] Format latency (PT / TorchScript / ONNX) ---"
python3 -m benchmarks.latency_test
echo ""

echo "--- [2/5] Batch-size sweep (all formats, sizes 1-64) ---"
python3 -m benchmarks.batch_experiments
echo ""

echo "--- [3/5] CPU vs MPS device benchmark ---"
python3 -m benchmarks.device_benchmark
echo ""

# ---- start the serving stack ----
echo "Starting serving stack for live-traffic benchmarks..."
redis-server --daemonize yes

python3 -m src.serving.worker_pool --workers 4 &
WORKER_PID=$!

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

#wait for API health
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "API is ready."
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "ERROR: API did not become healthy after 30s."
    exit 1
  fi
done
echo ""

# ---- server-dependent benchmarks ----
echo "--- [4/5] Throughput test (100 concurrent async requests) ---"
python3 -m benchmarks.throughput_test
echo ""

echo "--- [5/5] Worker scaling study (1 / 2 / 4 / 8 workers) ---"
python3 -m benchmarks.worker_experiments
echo ""

# ---- tear down stack ----
echo "Stopping serving stack..."
kill "$WORKER_PID" "$API_PID" 2>/dev/null || true
WORKER_PID=""
API_PID=""
redis-cli shutdown 2>/dev/null || true
sleep 1
echo ""

# ---- visualize ----
echo "--- Generating plots ---"
python3 -m benchmarks.visualize
echo ""

echo "=========================================="
echo "All benchmarks complete."
echo "  Results: benchmarks/results/"
echo "  Plots:   benchmarks/plots/"
echo "=========================================="

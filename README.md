# Efficient Inference CV

A computer vision project focused on training and efficiently serving a deep learning model (MobileNetV2) on the CIFAR-10 dataset. This project demonstrates best practices for moving from a standard PyTorch model to highly optimized inference using TorchScript and ONNX.

## Features

- **Efficient Training**: Uses lazy-loading and metadata-backed caching to efficiently train on preprocessed CIFAR-10 batches without exhausting memory or causing severe disk I/O thrashing.
- **Model Export**: Easily export trained PyTorch models to `TorchScript (.pt)` and `ONNX (.onnx)` formats for high-performance deployment.
- **Unified Predictor**: A robust `Predictor` class capable of seamlessly switching between PyTorch, TorchScript, and ONNX models for inference, with optional device targeting (`"cpu"` or `"mps"` for Apple Metal).
- **FastAPI Service**: A production-ready API for serving image predictions, complete with comprehensive error handling.
- **Asynchronous Batch Serving**: High-concurrency support using a Redis-backed message queue. The API asynchronously pushes requests to a background worker pool that dynamically batches images for massive throughput gains.
- **Worker Pool**: Scales to utilize multiple CPU cores by spawning independent multiprocessing workers (`worker_pool.py`) that safely consume from the same Redis queue.
- **Dual Protocol Support**: REST (FastAPI) and gRPC endpoints for protocol comparison under high load. gRPC uses HTTP/2 multiplexing and Protobuf serialization for reduced overhead.
- **Per-Request Metrics**: Internal instrumentation tracks queue wait time, inference latency, and total end-to-end latency with percentile aggregation (p50, p95, p99). Exposed via `/metrics` endpoint for real-time system observability.
- **Benchmarking**: Compare inference latencies, batch size efficiencies, system saturation scaling curves across different model formats, worker counts, protocols, and hardware accelerators.
- **Visualization**: Single-command rendering of every benchmark JSON into publication-ready PNG plots (model format comparison, batch curves, worker scaling, protocol comparison, CPU vs MPS).

## Project Structure

```text
├── benchmarks/                  # Benchmark scripts and outputs
│   ├── batch_experiments.py     # Batch size sweep (all formats, sizes 1–64)
│   ├── benchmark_utils.py       # Shared async HTTP benchmark helpers
│   ├── device_benchmark.py      # CPU vs MPS latency/throughput sweep
│   ├── docker_experiments.py    # Docker vs native comparison
│   ├── grpc_experiments.py      # REST vs gRPC protocol comparison
│   ├── latency_test.py          # Per-format latency (PT / TorchScript / ONNX)
│   ├── plot_utils.py            # latest_result() helper and shared paths
│   ├── throughput_test.py       # Concurrent async request throughput test
│   ├── visualize.py             # Generates all 7 PNG plots from results JSON
│   ├── worker_experiments.py    # Worker scaling study (1 / 2 / 4 / 8 workers)
│   ├── plots/                   # Generated PNGs (gitignored, re-created by visualize.py)
│   └── results/                 # Benchmark output JSON files
├── configs/                     # Configuration files
├── data/
│   ├── processed/               # Preprocessed CIFAR-10 tensor batches (.pt)
│   └── raw/                     # Downloaded CIFAR-10 raw data
├── info/
│   ├── notes.txt                # Research observation log (failures, surprises, pivots)
│   ├── why.txt                  # Design rationale journal
│   └── fairness_log.txt         # Benchmark fairness fixes with before/after metrics
├── models/
│   ├── checkpoints/             # Per-epoch training checkpoints
│   ├── exported/                # model.pt (TorchScript) and model.onnx
│   └── trained/                 # mobilenetv2.pth (raw PyTorch weights)
├── notebooks/                   # Exploratory Jupyter notebooks
├── proto/
│   └── inference.proto          # Protobuf definition for gRPC service
├── scripts/
│   ├── train.sh                 # Full training pipeline (preprocess → train → export)
│   ├── start.sh                 # Start Redis + worker pool + FastAPI
│   └── benchmark.sh             # Run all benchmarks and generate plots
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI app — /predict, /predict_sync, /health, /metrics
│   ├── grpc_api/
│   │   ├── server.py            # Async gRPC server (port 50051)
│   │   ├── inference_pb2.py     # Generated Protobuf message classes
│   │   └── inference_pb2_grpc.py
│   ├── inference/
│   │   ├── predictor.py         # Format-agnostic predictor (.pt / .pth / .onnx)
│   │   └── preprocess.py        # Resize, normalize, add batch dim
│   ├── serving/
│   │   ├── batch_scheduler.py   # Batches requests (size=8 or 20ms timeout)
│   │   ├── consumer.py          # Redis queue consumer (blocking/non-blocking)
│   │   ├── metrics_collector.py # Rolling window aggregator (p50/p95/p99)
│   │   ├── producer.py          # Pushes requests to Redis with enqueue timestamp
│   │   ├── redis_client.py      # Redis connection helper
│   │   ├── worker.py            # Single worker process loop
│   │   └── worker_pool.py       # Spawns N worker processes
│   └── training/
│       ├── export_model.py      # Exports trained model to TorchScript and ONNX
│       ├── model.py             # MobileNetV2 architecture definition
│       ├── preprocess_data.py   # Downloads CIFAR-10, saves tensor batches
│       └── train.py             # Training loop with per-epoch checkpointing
├── tests/                       # Unit and integration tests (fakeredis, no live services)
├── Dockerfile                   # Multi-stage build for containerized deployment
├── docker-compose.yml           # Orchestrates redis, worker, and api services
└── requirements.txt
```

## Getting Started

### Prerequisites

Ensure you have Python 3 installed and the project dependencies available. 

```bash
pip install -r requirements.txt
```

### 1. Data Preprocessing

Before training, preprocess the CIFAR-10 dataset into optimized tensor batches:

```bash
python3 scripts/preprocess_data.py
```
This downloads the dataset and creates preprocessed `.pt` batches along with a `metadata.json` file in `data/processed/`.

### 2. Training the Model

Train the MobileNetV2 model using the preprocessed data. The training pipeline efficiently utilizes the metadata to avoid loading the entire dataset into memory.

```bash
python3 -m src.training.train
```
The trained model weights will be saved to `models/trained/mobilenetv2.pth`.

### 3. Exporting the Model

Export the trained PyTorch model to optimized formats (TorchScript and ONNX) for faster production inference:

```bash
python3 -m scripts.export_model
```
This generates `model.pt` and `model.onnx` in the `models/exported/` directory.

### 4. Running the Serving System & API

The inference system relies on a background worker to dynamically batch requests. You must run Redis, the worker, and the API together.

**The easiest way** to start everything is using the included shell script, which launches all three services in the background and shuts them down gracefully when you press `Ctrl+C`:

```bash
bash start.sh
```

Alternatively, you can run them manually in separate terminal windows:
1. `redis-server`
2. `python3 -m src.serving.worker_pool --workers 4`
3. `uvicorn src.api.main:app --reload`

**Docker Deployment** (Phase 6):

Complete Docker infrastructure exists and is fully functional:

```bash
docker compose up
```

This orchestrates:
- `redis:7-alpine` — Redis service for request queueing
- `worker` — Worker pool (4 processes) consuming from the queue
- `api` — FastAPI server on port 8000

The Docker setup auto-configures Redis hostname discovery via `REDIS_HOST` environment variable, enabling seamless multi-container networking. All services are ready when the API health check passes.

```bash
#verify the stack is healthy
curl http://127.0.0.1:8000/health

#tear down
docker compose down
```

**Note (Phase 6):** Docker infrastructure is complete and production-ready. On Apple Silicon (macOS), the Docker environment runs Linux containers with CPU-only PyTorch. Docker vs native comparison benchmarks are deferred to environments with NVIDIA hardware where containerization overhead can be meaningfully measured against the same hardware baseline. Phase 6 benchmarks use native Apple Silicon (MPS) deployment as the baseline.

The REST API provides four endpoints:
- `GET /health`: Checks if the API is running and Redis is successfully connected.
- `POST /predict`: Upload an image (JPEG/PNG). The image is pushed to the Redis queue and processed by the highly-concurrent worker pool.
- `POST /predict_sync`: A baseline endpoint that ignores the queue and processes the image locally on the main thread (useful for demonstrating CPU thread-thrashing under high concurrency).
- `GET /metrics`: Returns aggregated performance metrics (latency percentiles, queue wait, inference time, batch size distribution) from the last 1000 requests.

A gRPC server also runs on port 50051 for protocol comparison experiments.

### 5. Benchmarking Suite

The project includes a comprehensive suite of benchmarking tools in the `benchmarks/` directory:

- **Latency Test**: Compares inference speed across PyTorch, TorchScript, and ONNX formats.
  ```bash
  python3 -m benchmarks.latency_test
  ```
- **Batch Experiments**: Tests the raw inference engines across various batch sizes (1 to 64) to demonstrate the efficiency gains of dynamic batching.
  ```bash
  python3 -m benchmarks.batch_experiments
  ```
- **Throughput Test**: An end-to-end stress test that bombards the FastAPI server with concurrent asynchronous requests to measure the overall system's Requests-Per-Second (RPS).
  ```bash
  python3 -m benchmarks.throughput_test
  ```
- **Worker Scaling & Concurrency Experiments**: A fully automated experiment that orchestrates the API and Worker Pool to test throughput, latency (p50, p95, p99), and saturation points across `1, 2, 4, and 8` workers compared to the Sync API baseline.
  ```bash
  # Ensure redis-server is running first
  python3 -m benchmarks.worker_experiments
  ```
- **gRPC vs REST Protocol Comparison**: Measures payload sizes, latency, and throughput for both protocols at varying concurrency levels (1, 5, 20, 50 concurrent requests).
  ```bash
  python3 -m benchmarks.grpc_experiments
  ```
- **Phase 6 Native Baseline Benchmarks**: Tests throughput and latency on native Apple Silicon (MPS) hardware. Docker infrastructure is complete but Docker vs native comparison is deferred to NVIDIA hardware environments where containerization overhead is meaningful.
  ```bash
  bash benchmark_phase6.sh
  ```
- **Device Benchmark (Phase 7)**: Compares inference latency and throughput on CPU vs Apple Metal (MPS) across PyTorch and TorchScript models at batch sizes 1–64. ONNX is CPU-only (ONNX Runtime has no MPS backend). Reveals where MPS acceleration provides meaningful speedup over CPU.
  ```bash
  python3 -m benchmarks.device_benchmark
  ```
- **Visualization (Phase 8)**: Renders all benchmark JSON results in `benchmarks/results/` into seven publication-ready PNG plots in `benchmarks/plots/`. Picks the most recent run per experiment type by filename timestamp.
  ```bash
  python3 -m benchmarks.visualize
  ```
  Generated figures: model format latency comparison, batch latency/throughput curves, async worker scaling, gRPC vs REST protocol comparison, and CPU vs MPS device latency/throughput curves.

All scripts automatically save detailed JSON reports to `benchmarks/results/`. Use `/metrics` endpoint to observe latency breakdown during any benchmark run.

## Robustness & Error Handling


This project is built with strong error handling at every layer:
- **API Model Fallback**: The API attempts to load the most efficient model available (TorchScript). If unavailable or corrupted, it falls back to ONNX, and finally to the raw PyTorch model.
- **Image Validation**: The `/predict` endpoint catches bad uploads, UnidentifiedImageError, and corrupted files, returning clear HTTP 400 errors instead of vague internal server errors.
- **File I/O Safety**: Data preprocessing and model export scripts safely wrap file I/O operations in try-except blocks.

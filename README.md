# Efficient Inference CV

A computer vision project focused on training and efficiently serving a deep learning model (MobileNetV2) on the CIFAR-10 dataset. This project demonstrates best practices for moving from a standard PyTorch model to highly optimized inference using TorchScript and ONNX.

## Features

- **Efficient Training**: Uses lazy-loading and metadata-backed caching to efficiently train on preprocessed CIFAR-10 batches without exhausting memory or causing severe disk I/O thrashing.
- **Model Export**: Easily export trained PyTorch models to `TorchScript (.pt)` and `ONNX (.onnx)` formats for high-performance deployment.
- **Unified Predictor**: A robust `Predictor` class capable of seamlessly switching between PyTorch, TorchScript, and ONNX models for inference.
- **FastAPI Service**: A production-ready API for serving image predictions, complete with comprehensive error handling.
- **Asynchronous Batch Serving**: High-concurrency support using a Redis-backed message queue. The API asynchronously pushes requests to a background worker pool that dynamically batches images for massive throughput gains.
- **Worker Pool**: Scales to utilize multiple CPU cores by spawning independent multiprocessing workers (`worker_pool.py`) that safely consume from the same Redis queue.
- **Dual Protocol Support**: REST (FastAPI) and gRPC endpoints for protocol comparison under high load. gRPC uses HTTP/2 multiplexing and Protobuf serialization for reduced overhead.
- **Per-Request Metrics**: Internal instrumentation tracks queue wait time, inference latency, and total end-to-end latency with percentile aggregation (p50, p95, p99). Exposed via `/metrics` endpoint for real-time system observability.
- **Benchmarking**: Compare inference latencies, batch size efficiencies, system saturation scaling curves across different model formats, worker counts, protocols, and hardware accelerators.

## Project Structure

```text
├── benchmarks/         # Inference latency benchmarking scripts
├── configs/            # Configuration files
├── data/               # Raw and preprocessed CIFAR-10 data
├── models/             # Checkpoints, trained models, and exported formats
├── notebooks/          # Exploratory Jupyter notebooks
├── scripts/            # Utility scripts for data preprocessing and model export
│   ├── preprocess_data.py
│   └── export_model.py
├── src/                # Core source code
│   ├── api/            # FastAPI application (main.py)
│   ├── inference/      # Prediction and inference preprocessing logic
│   ├── serving/        # Background workers, redis integration, and worker_pool.py
│   ├── training/       # Training loops, datasets, and model architecture
│   └── utils/          # Shared utilities
└── tests/              # Unit and integration tests
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

Alternatively, deploy the entire stack using Docker Compose:

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
- **Docker vs Native Deployment Comparison** (Phase 6): Tests throughput and latency of the containerized stack (via docker-compose) vs native deployment on the same hardware.
  ```bash
  python3 -m benchmarks.docker_experiments
  ```

All scripts automatically save detailed JSON reports to `benchmarks/results/`. Use `/metrics` endpoint to observe latency breakdown during any benchmark run.

## Robustness & Error Handling


This project is built with strong error handling at every layer:
- **API Model Fallback**: The API attempts to load the most efficient model available (TorchScript). If unavailable or corrupted, it falls back to ONNX, and finally to the raw PyTorch model.
- **Image Validation**: The `/predict` endpoint catches bad uploads, UnidentifiedImageError, and corrupted files, returning clear HTTP 400 errors instead of vague internal server errors.
- **File I/O Safety**: Data preprocessing and model export scripts safely wrap file I/O operations in try-except blocks.

# Efficient Inference CV

A computer vision project focused on training and efficiently serving a deep learning model (MobileNetV2) on the CIFAR-10 dataset. This project demonstrates best practices for moving from a standard PyTorch model to highly optimized inference using TorchScript and ONNX.

## Features

- **Efficient Training**: Uses lazy-loading and metadata-backed caching to efficiently train on preprocessed CIFAR-10 batches without exhausting memory or causing severe disk I/O thrashing.
- **Model Export**: Easily export trained PyTorch models to `TorchScript (.pt)` and `ONNX (.onnx)` formats for high-performance deployment.
- **Unified Predictor**: A robust `Predictor` class capable of seamlessly switching between PyTorch, TorchScript, and ONNX models for inference.
- **FastAPI Service**: A production-ready API for serving image predictions, complete with comprehensive error handling.
- **Asynchronous Batch Serving**: High-concurrency support using a Redis-backed message queue. The API asynchronously pushes requests to a background worker that dynamically batches images for massive throughput gains.
- **Benchmarking**: Compare inference latencies, batch size efficiencies, and overall system throughput across different model formats.

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
│   ├── serving/        # Background workers and redis integration
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

**Terminal 1 (Redis):**
Start your local Redis server:
```bash
redis-server
```

**Terminal 2 (Background Worker):**
Start the background worker to consume the inference queue:
```bash
python3 -m src.serving.worker
```

**Terminal 3 (FastAPI Server):**
Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

The API provides two endpoints:
- `GET /health`: Checks if the API is running and Redis is successfully connected.
- `POST /predict`: Upload an image (JPEG/PNG) to receive a class prediction (0-9 corresponding to CIFAR-10 classes).

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
- **Throughput Test**: An end-to-end stress test that bombards the FastAPI server with concurrent asynchronous requests to measure the overall system's Requests-Per-Second (RPS). *(Requires the API, Worker, and Redis to be running)*.
  ```bash
  python3 -m benchmarks.throughput_test
  ```

All scripts automatically save detailed JSON reports to `benchmarks/results/`.

## Robustness & Error Handling


This project is built with strong error handling at every layer:
- **API Model Fallback**: The API attempts to load the most efficient model available (TorchScript). If unavailable or corrupted, it falls back to ONNX, and finally to the raw PyTorch model.
- **Image Validation**: The `/predict` endpoint catches bad uploads, UnidentifiedImageError, and corrupted files, returning clear HTTP 400 errors instead of vague internal server errors.
- **File I/O Safety**: Data preprocessing and model export scripts safely wrap file I/O operations in try-except blocks.

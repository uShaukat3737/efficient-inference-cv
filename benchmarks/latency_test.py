import time
import torch
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from datetime import datetime
from src.training.model import get_model

# set number of threads to 1 for fair comparison between models (avoid unfair speedups)
torch.set_num_threads(1)

# dummy input for testing (batch size 1, 3 color channels, 32x32 pixels)
dummy_np = np.random.randn(1,3,32,32).astype(np.float32)
dummy_torch = torch.from_numpy(dummy_np)

# results directory
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def print_stats(times):
  print(f"Avg Latency:  {round(np.mean(times), 2)} ms")
  print(f"Min Latency:  {round(np.min(times), 2)} ms")
  print(f"Max Latency:  {round(np.max(times), 2)} ms")
  print(f"Std Dev:      {round(np.std(times), 2)} ms")

def compute_stats(times):
  return {
    "mean": round(np.mean(times), 2),
    "min": round(np.min(times), 2),
    "max": round(np.max(times), 2),
    "std": round(np.std(times), 2)
  }

# TorchScript Benchmark
def benchmark_torchscript():
  print(f"\n{'='*50}")
  print(f"Benchmarking: TorchScript Model")
  print(f"{'='*50}")

  # load TorchScript model (optimized version of PyTorch model for inference)
  model = torch.jit.load("models/exported/model.pt")
  model.eval()

  times = []

  # warmup runs (important to remove first-run overhead like caching, initialization)
  for _ in range(10):
    model(dummy_torch)

  # run inference 100 times and record time taken for each prediction
  for _ in range(100):
    start = time.time()
    model(dummy_torch)
    end = time.time()
    times.append((end-start)*1000)  # convert to milliseconds

  stats = compute_stats(times)
  print_stats(times)
  return stats

# Raw PyTorch Benchmark
def benchmark_pytorch():
  print(f"\n{'='*50}")
  print(f"Benchmarking: Raw PyTorch Model")
  print(f"{'='*50}")

  # load original PyTorch model architecture and trained weights
  model = get_model()
  model.load_state_dict(torch.load("models/trained/mobilenetv2.pth"))
  model.eval()

  times = []

  # warmup runs
  for _ in range(10):
    model(dummy_torch)

  # run inference 100 times and record time taken
  for _ in range(100):
    start = time.time()
    model(dummy_torch)
    end = time.time()
    times.append((end-start)*1000)

  stats = compute_stats(times)
  print_stats(times)
  return stats


# ONNX Benchmark
def benchmark_onnx():
  print(f"\n{'='*50}")
  print(f"Benchmarking: ONNX Model")
  print(f"{'='*50}")

  # create ONNX runtime session with controlled threading (for fair comparison)
  sess_options = ort.SessionOptions()
  sess_options.intra_op_num_threads = 1

  # load ONNX model into inference session
  session = ort.InferenceSession("models/exported/model.onnx", sess_options)

  # get input name required by ONNX runtime
  input_name = session.get_inputs()[0].name

  times = []

  # warmup runs
  for _ in range(10):
    session.run(None, {input_name: dummy_np})

  # run inference 100 times and record time taken
  for _ in range(100):
    start = time.time()
    session.run(None, {input_name: dummy_np})
    end = time.time()
    times.append((end-start)*1000)

  stats = compute_stats(times)
  print_stats(times)
  return stats

# --------------------------
# Run all benchmarks
# --------------------------
if __name__ == "__main__":
  results = {
    "timestamp": datetime.now().isoformat(),
    "models": {
      "torchscript": benchmark_torchscript(),
      "pytorch": benchmark_pytorch(),
      "onnx": benchmark_onnx()
    }
  }
  
  # Save results to JSON file
  results_file = RESULTS_DIR / f"latency_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
  with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
  
  print(f"\n{'='*50}")
  print(f"Results saved to: {results_file}")
  print(f"{'='*50}")
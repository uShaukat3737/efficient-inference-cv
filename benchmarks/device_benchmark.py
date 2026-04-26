import time
import platform
import torch
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

#add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference.predictor import Predictor

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MODELS = {
    "torchscript": "models/exported/model.pt",
    "pytorch": "models/trained/mobilenetv2.pth",
    "onnx": "models/exported/model.onnx",
}

def get_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices

def run_experiment(predictor, batch_size, device_str, num_runs=50):
    dummy_input = torch.randn(batch_size, 3, 32, 32)

    #warmup runs
    for _ in range(5):
        predictor.predict_batch(dummy_input)
        if device_str == "mps":
            #MPS dispatches ops asynchronously; synchronize to ensure warmup completes
            torch.mps.synchronize()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictor.predict_batch(dummy_input)
        if device_str == "mps":
            #required: block until MPS kernel finishes; without this, timer captures dispatch only
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  #milliseconds

    mean_latency = np.mean(times)
    latency_per_image = mean_latency / batch_size
    throughput = batch_size / (mean_latency / 1000)

    return {
        "batch_size": batch_size,
        "total_latency_ms": round(mean_latency, 2),
        "latency_per_image_ms": round(latency_per_image, 4),
        "throughput_img_per_sec": round(throughput, 1),
    }

def main():
    devices = get_devices()

    print(f"{'='*70}")
    print("Phase 7: CPU vs MPS Device Benchmark")
    print(f"Platform: {platform.platform()}")
    print(f"Devices: {devices}")
    print(f"{'='*70}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "devices_available": devices,
        "results": {},
    }

    for device_str in devices:
        print(f"\n--- Device: {device_str.upper()} ---")
        results["results"][device_str] = {}

        for model_name, path in MODELS.items():
            if model_name == "onnx" and device_str == "mps":
                print(f"  Skipping ONNX on MPS (ONNX Runtime has no MPS backend)")
                continue

            print(f"\n  Loading {model_name} on {device_str}...")
            try:
                predictor = Predictor(path, device=device_str)
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue

            print(f"  {'Batch':<8} | {'Total (ms)':<12} | {'Per-img (ms)':<14} | {'Throughput (img/s)':<20}")
            print(f"  {'-'*60}")

            model_results = []
            for batch_size in BATCH_SIZES:
                res = run_experiment(predictor, batch_size, device_str)
                model_results.append(res)
                print(
                    f"  {res['batch_size']:<8} | "
                    f"{res['total_latency_ms']:<12} | "
                    f"{res['latency_per_image_ms']:<14} | "
                    f"{res['throughput_img_per_sec']:<20}"
                )

            results["results"][device_str][model_name] = model_results

    results_file = RESULTS_DIR / f"device_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

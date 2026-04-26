import time
import torch
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

#add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference.predictor import Predictor

#match worker pool conditions: one thread per process, same as latency_test.py
torch.set_num_threads(1)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#test configurations
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MODELS = {
    "onnx": "models/exported/model.onnx",
    "torchscript": "models/exported/model.pt",
    "pytorch": "models/trained/mobilenetv2.pth"
}

def run_experiment(predictor, batch_size, num_runs=50):
    #dummy tensor for the given batch size
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    
    #warmup
    for _ in range(5):
        predictor.predict_batch(dummy_input)
        
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictor.predict_batch(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000) #milliseconds
        
    mean_latency = np.mean(times)
    latency_per_image = mean_latency / batch_size
    
    return {
        "batch_size": batch_size,
        "total_latency_ms": round(mean_latency, 2),
        "latency_per_image_ms": round(latency_per_image, 2)
    }

def main():
    print(f"{'='*60}")
    print("Starting Batch Size Experiments")
    print(f"{'='*60}")
    
    results = {}
    
    for model_name, path in MODELS.items():
        print(f"\nLoading {model_name}...")
        try:
            predictor = Predictor(path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
            
        model_results = []
        print(f"{'Batch Size':<15} | {'Total Latency (ms)':<20} | {'Latency/Image (ms)':<20}")
        print("-" * 60)
        
        for batch_size in BATCH_SIZES:
            res = run_experiment(predictor, batch_size)
            model_results.append(res)
            print(f"{batch_size:<15} | {res['total_latency_ms']:<20} | {res['latency_per_image_ms']:<20}")
            
        results[model_name] = model_results

    #save results
    results_file = RESULTS_DIR / f"batch_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"Experiments completed! Results saved to: {results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

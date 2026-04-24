#import necessary modules for async requests and benchmarking
import asyncio
import aiohttp
import time
import io
import json
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

#setup results directory
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

URL_ASYNC = "http://127.0.0.1:8000/predict"
URL_SYNC = "http://127.0.0.1:8000/predict_sync"

#create a dummy jpeg image in memory for testing
def create_dummy_image():
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

#send a single async multipart form request and return status and latency
async def send_request(session, url, image_data):
    data = aiohttp.FormData()
    data.add_field('file', image_data, filename='dummy.jpg', content_type='image/jpeg')
    
    start_time = time.time()
    try:
        async with session.post(url, data=data) as response:
            status = response.status
            await response.read()
            latency = (time.time() - start_time) * 1000
            return status == 200, latency
    except Exception:
        return False, 0

#run bounded concurrent requests using a semaphore and collect metrics
async def run_load_test(url, concurrency, total_requests):
    image_data = create_dummy_image()
    
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with sem:
                return await send_request(session, url, image_data)
                
        start_time = time.time()
        tasks = [bounded_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
    total_time = end_time - start_time
    successful = sum(1 for r in results if r[0])
    failed = total_requests - successful
    latencies = [r[1] for r in results if r[0]]
    
    rps = total_requests / total_time
    
    #calculate latency percentiles
    if latencies:
        metrics = {
            "avg_latency": round(np.mean(latencies), 2),
            "p50_latency": round(np.percentile(latencies, 50), 2),
            "p95_latency": round(np.percentile(latencies, 95), 2),
            "p99_latency": round(np.percentile(latencies, 99), 2)
        }
    else:
        metrics = {"avg_latency": 0, "p50_latency": 0, "p95_latency": 0, "p99_latency": 0}
        
    print(f"    RPS: {rps:.1f} | Avg Latency: {metrics['avg_latency']}ms | P95: {metrics['p95_latency']}ms")
    return {
        "concurrency": concurrency,
        "rps": round(rps, 2),
        "metrics": metrics,
        "successful": successful,
        "failed": failed
    }

#poll the health endpoint until the server is ready
async def wait_for_server(url):
    for _ in range(30):
        try:
            async with aiohttp.ClientSession() as session:
                #hit health endpoint
                health_url = "http://127.0.0.1:8000/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return True
        except:
            pass
        await asyncio.sleep(0.5)
    return False

async def main():
    print(f"{'='*60}")
    print("Worker Pool Scaling & Concurrency Experiments")
    print("Note: Ensure redis-server is running before starting!")
    print(f"{'='*60}")
    
    #start the fastapi server in the background
    print("\nStarting FastAPI server in background...")
    api_proc = subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--port", "8000"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    server_up = await wait_for_server(URL_SYNC)
    if not server_up:
        print("Failed to start FastAPI server. Is port 8000 already in use?")
        api_proc.terminate()
        return

    results = {
        "sync_api_baseline": [],
        "async_queue_scaling": {}
    }
    
    try:
        #test sync api baseline no queue
        print("\n--- Testing Sync API Baseline (No Queue) ---")
        sync_concurrencies = [1, 5, 20, 50]
        for c in sync_concurrencies:
            print(f"Testing Concurrency {c}...")
            res = await run_load_test(URL_SYNC, c, max(50, c * 5))
            results["sync_api_baseline"].append(res)
            await asyncio.sleep(1)
            
        #test async api with worker scaling
        worker_counts = [1, 2, 4, 8]
        async_concurrencies = [1, 10, 50]
        
        for w in worker_counts:
            print(f"\n--- Testing Async API with {w} Worker(s) ---")
            worker_proc = subprocess.Popen(
                ["python3", "-m", "src.serving.worker_pool", "--workers", str(w)], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print("Giving workers 5 seconds to load models into memory...")
            await asyncio.sleep(5) 
            
            w_results = []
            for c in async_concurrencies:
                print(f"Testing Concurrency {c}...")
                res = await run_load_test(URL_ASYNC, c, max(50, c * 5))
                w_results.append(res)
                await asyncio.sleep(1)
                
            results["async_queue_scaling"][f"{w}_workers"] = w_results
            
            print(f"Shutting down {w} workers...")
            worker_proc.terminate()
            worker_proc.wait()
            
    finally:
        #clean up resources and shut down server
        print("\nShutting down FastAPI server...")
        api_proc.terminate()
        api_proc.wait()
    
    #save results to json file
    out_file = RESULTS_DIR / f"worker_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"Experiments completed! Results saved to: {out_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())

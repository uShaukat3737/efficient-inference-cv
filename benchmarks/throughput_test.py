import asyncio
import aiohttp
import time
import io
import json
from PIL import Image
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://127.0.0.1:8000/predict"

#create a valid dummy image in memory
def create_dummy_image():
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

async def send_request(session, image_data):
    data = aiohttp.FormData()
    data.add_field('file', image_data, filename='dummy.jpg', content_type='image/jpeg')
    
    start_time = time.time()
    try:
        async with session.post(URL, data=data) as response:
            status = response.status
            await response.read()
            latency = (time.time() - start_time) * 1000
            return status == 200, latency
    except Exception:
        return False, 0

async def run_throughput_test(concurrency, total_requests):
    image_data = create_dummy_image()
    
    print(f"\nStarting test: {concurrency} concurrent workers, {total_requests} total requests...")
    
    async with aiohttp.ClientSession() as session:
        #create a semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with sem:
                return await send_request(session, image_data)
                
        start_time = time.time()
        
        #fire off all requests
        tasks = [bounded_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
    total_time = end_time - start_time
    successful = sum(1 for r in results if r[0])
    failed = total_requests - successful
    latencies = [r[1] for r in results if r[0]]
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    rps = total_requests / total_time
    
    print(f"Results for Concurrency {concurrency}:")
    print(f"  Time taken:   {total_time:.2f} seconds")
    print(f"  Throughput:   {rps:.2f} requests/second")
    print(f"  Success rate: {successful}/{total_requests} ({successful/total_requests*100:.1f}%)")
    print(f"  Avg latency:  {avg_latency:.2f} ms")
    
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time_seconds": round(total_time, 2),
        "requests_per_second": round(rps, 2),
        "successful_requests": successful,
        "failed_requests": failed,
        "avg_latency_ms": round(avg_latency, 2)
    }

async def main():
    print(f"{'='*60}")
    print("API Throughput Testing (Requires API + Worker + Redis to be running)")
    print(f"{'='*60}")
    
    #test scenarios: [concurrency, total_requests]
    scenarios = [
        (1, 50),
        (5, 100),
        (20, 200),
        (50, 500)
    ]
    
    all_results = []
    
    #warmup
    print("Warming up server...")
    await run_throughput_test(1, 10)
    
    for concurrency, total in scenarios:
        res = await run_throughput_test(concurrency, total)
        all_results.append(res)
        await asyncio.sleep(2) #cool down between tests
        
    #save results
    results_file = RESULTS_DIR / f"throughput_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"Testing completed! Results saved to: {results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    #check if aiohttp is installed
    try:
        import aiohttp
    except ImportError:
        print("Please install aiohttp to run this script: pip install aiohttp")
        sys.exit(1)
        
    asyncio.run(main())

import asyncio
import io
import time
from pathlib import Path
from typing import Dict, Any

import aiohttp
import numpy as np
from PIL import Image

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_dummy_jpeg(size=(224, 224)) -> bytes:
    #creates a dummy JPEG image for testing
    img = Image.new("RGB", size, color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def run_rest_benchmark(
    url: str,
    concurrency: int,
    num_requests: int = None
) -> Dict[str, Any]:
    #benchmark REST API with async requests
    if num_requests is None:
        num_requests = max(100, concurrency * 5)

    image_data = create_dummy_jpeg()
    latencies = []
    failed = 0

    async def make_request(session, semaphore):
        nonlocal failed
        async with semaphore:
            try:
                form_data = aiohttp.FormData()
                form_data.add_field("file", io.BytesIO(image_data), filename="test.jpg", content_type="image/jpeg")

                start = time.monotonic()
                async with session.post(url, data=form_data, timeout=30) as resp:
                    await resp.json()
                    elapsed = (time.monotonic() - start) * 1000
                    latencies.append(elapsed)
            except Exception as e:
                failed += 1

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [make_request(session, semaphore) for _ in range(num_requests)]
        await asyncio.gather(*tasks)

    if not latencies:
        return {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "failed": failed,
            "rps": 0,
            "avg_latency_ms": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0
        }

    latencies_sorted = sorted(latencies)
    total_time_s = max(latencies) / 1000
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "failed": failed,
        "rps": round(len(latencies) / total_time_s, 2) if total_time_s > 0 else 0,
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p50": float(np.percentile(latencies_sorted, 50)),
        "p95": float(np.percentile(latencies_sorted, 95)),
        "p99": float(np.percentile(latencies_sorted, 99))
    }

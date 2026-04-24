import asyncio
import io
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from grpc import aio
from PIL import Image

from src.grpc_api import inference_pb2, inference_pb2_grpc

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRPC_PORT = 50051
REST_URL_ASYNC = "http://127.0.0.1:8000/predict"
REST_HEALTH_URL = "http://127.0.0.1:8000/health"


def create_dummy_jpeg(size=(224, 224)):
    img = Image.new("RGB", size, color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def measure_payload_sizes(image_bytes):
    """Compare binary payload size of gRPC protobuf vs HTTP multipart."""
    grpc_size = inference_pb2.PredictRequest(image_data=image_bytes).ByteSize()

    boundary = b"----WebKitFormBoundary7MA4YWxkTrZu0gW"
    multipart = (
        b"--" + boundary + b"\r\n"
        + b'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
        + b"Content-Type: image/jpeg\r\n\r\n"
        + image_bytes
        + b"\r\n--" + boundary + b"--\r\n"
    )
    rest_size = len(multipart)

    return {
        "grpc_protobuf_bytes": grpc_size,
        "rest_multipart_bytes": rest_size,
        "rest_overhead_pct": round((rest_size - grpc_size) / grpc_size * 100, 2),
    }


def _build_metrics(protocol, concurrency, total_requests, duration, latencies, errors):
    def pct(arr, p):
        return round(float(np.percentile(arr, p)), 2) if arr else 0

    return {
        "protocol": protocol,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(latencies),
        "failed": errors,
        "rps": round(total_requests / duration, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0,
        "p50_latency_ms": pct(latencies, 50),
        "p95_latency_ms": pct(latencies, 95),
        "p99_latency_ms": pct(latencies, 99),
    }


async def run_grpc_benchmark(stub, image_bytes, concurrency, total_requests):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def single():
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                await stub.Predict(inference_pb2.PredictRequest(image_data=image_bytes))
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[single() for _ in range(total_requests)])
    duration = time.perf_counter() - t_start

    return _build_metrics("grpc", concurrency, total_requests, duration, latencies, errors)


async def run_rest_benchmark(session, image_bytes, concurrency, total_requests):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def single():
        nonlocal errors
        async with sem:
            data = aiohttp.FormData()
            data.add_field("file", image_bytes, filename="image.jpg", content_type="image/jpeg")
            t0 = time.perf_counter()
            try:
                async with session.post(REST_URL_ASYNC, data=data) as resp:
                    await resp.read()
                    if resp.status == 200:
                        latencies.append((time.perf_counter() - t0) * 1000)
                    else:
                        errors += 1
            except Exception:
                errors += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[single() for _ in range(total_requests)])
    duration = time.perf_counter() - t_start

    return _build_metrics("rest", concurrency, total_requests, duration, latencies, errors)


async def wait_for_rest(timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(REST_HEALTH_URL) as r:
                    if r.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def wait_for_grpc(channel, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            await asyncio.wait_for(channel.channel_ready(), timeout=1)
            return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def main():
    print("=" * 60)
    print("REST vs gRPC Protocol Comparison Benchmark")
    print("Note: Ensure redis-server and worker pool are running!")
    print("=" * 60)

    image_bytes = create_dummy_jpeg()
    payload_info = measure_payload_sizes(image_bytes)
    print(
        f"\nPayload size analysis (224x224 JPEG):\n"
        f"  gRPC (protobuf) : {payload_info['grpc_protobuf_bytes']:,} bytes\n"
        f"  REST (multipart): {payload_info['rest_multipart_bytes']:,} bytes\n"
        f"  REST overhead   : +{payload_info['rest_overhead_pct']}%"
    )

    print("\nStarting FastAPI server...")
    api_proc = subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not await wait_for_rest():
        print("FastAPI server failed to start. Is port 8000 already in use?")
        api_proc.terminate()
        return

    print("Starting gRPC server on port 50051...")
    grpc_proc = subprocess.Popen(
        ["python3", "-m", "src.grpc_api.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    channel = aio.insecure_channel(f"localhost:{GRPC_PORT}")
    if not await wait_for_grpc(channel):
        print("gRPC server failed to start.")
        grpc_proc.terminate()
        api_proc.terminate()
        return

    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # Concurrency levels and request counts match worker_experiments.py conventions
    concurrencies = [1, 5, 20, 50]
    requests_per_run = 100

    results = {
        "payload_sizes": payload_info,
        "grpc_results": [],
        "rest_async_results": [],
    }

    try:
        print("\n--- gRPC Benchmarks ---")
        for c in concurrencies:
            n = max(requests_per_run, c * 5)
            print(f"  Concurrency {c:>2} ({n} requests)...", end=" ", flush=True)
            res = await run_grpc_benchmark(stub, image_bytes, c, n)
            results["grpc_results"].append(res)
            print(f"RPS: {res['rps']:>8.1f} | P50: {res['p50_latency_ms']}ms | P95: {res['p95_latency_ms']}ms | P99: {res['p99_latency_ms']}ms")
            await asyncio.sleep(1)

        print("\n--- REST Async Benchmarks ---")
        async with aiohttp.ClientSession() as session:
            for c in concurrencies:
                n = max(requests_per_run, c * 5)
                print(f"  Concurrency {c:>2} ({n} requests)...", end=" ", flush=True)
                res = await run_rest_benchmark(session, image_bytes, c, n)
                results["rest_async_results"].append(res)
                print(f"RPS: {res['rps']:>8.1f} | P50: {res['p50_latency_ms']}ms | P95: {res['p95_latency_ms']}ms | P99: {res['p99_latency_ms']}ms")
                await asyncio.sleep(1)

    finally:
        await channel.close()
        grpc_proc.terminate()
        grpc_proc.wait()
        api_proc.terminate()
        api_proc.wait()

    out_file = RESULTS_DIR / f"grpc_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete. Results saved to: {out_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

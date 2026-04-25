import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from benchmarks.benchmark_utils import run_rest_benchmark

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REST_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

CONCURRENCY_LEVELS = [1, 5, 20, 50]


async def wait_for_health_check(max_retries=30, delay=1):
    #wait for the API to be healthy
    import aiohttp

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(HEALTH_URL, timeout=5) as resp:
                    if resp.status == 200:
                        print("API is healthy")
                        return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Health check attempt {attempt + 1}/{max_retries} failed, retrying...")
                await asyncio.sleep(delay)
            else:
                print(f"Health check failed after {max_retries} attempts")
                return False
    return False


async def run_docker_benchmark():
    #run benchmark against Docker containerized stack
    print("Starting Docker stack via docker-compose...")

    try:
        #bring up the stack
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"docker-compose up failed: {result.stderr}")
            return None

        print("Waiting for API to become healthy...")
        if not await wait_for_health_check():
            print("API failed to become healthy")
            return None

        print("Running benchmark sweep...")
        results = {
            "timestamp": datetime.now().isoformat(),
            "deployment": "docker",
            "concurrency_results": []
        }

        #run benchmark at each concurrency level
        for concurrency in CONCURRENCY_LEVELS:
            print(f"  Running benchmark at concurrency {concurrency}...")
            num_requests = max(100, concurrency * 5)

            try:
                result = await run_rest_benchmark(
                    REST_URL,
                    concurrency=concurrency,
                    num_requests=num_requests
                )
                results["concurrency_results"].append(result)
                print(f"    Concurrency {concurrency}: {result['rps']:.2f} RPS, {result['avg_latency_ms']:.2f}ms avg")
            except Exception as e:
                print(f"    Benchmark failed at concurrency {concurrency}: {e}")
                return None

        return results

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

    finally:
        print("Tearing down Docker stack...")
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=".",
            capture_output=True,
            timeout=30
        )


def main():
    print("Phase 6B: Docker vs Native Benchmark - Docker Deployment")
    print("=" * 60)

    #run the benchmark
    results = asyncio.run(run_docker_benchmark())

    if results:
        #save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"docker_experiments_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {result_file}")
        print(json.dumps(results, indent=2))
    else:
        print("\nBenchmark failed")


if __name__ == "__main__":
    main()

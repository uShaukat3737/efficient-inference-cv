#import necessary modules
import argparse
import multiprocessing
import time
from src.serving.worker import main as worker_main

#define worker wrapper to catch keyboard interrupts gracefully
def run_worker():
    try:
        worker_main()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    #setup argument parser for number of workers
    parser = argparse.ArgumentParser(description="Start a pool of inference workers.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to spawn")
    args = parser.parse_args()

    print(f"Starting worker pool with {args.workers} processes...")
    
    #spawn the requested number of worker processes
    processes = []
    for i in range(args.workers):
        p = multiprocessing.Process(target=run_worker, name=f"worker-{i+1}")
        p.start()
        processes.append(p)
        print(f"Started {p.name} (PID: {p.pid})")
        
    #wait for all processes to finish or terminate them on interrupt
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down worker pool...")
        for p in processes:
            p.terminate()
            p.join()
        print("All workers stopped.")

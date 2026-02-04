import time
import numpy as np

def benchmark(infer_fn, runs=100):
    times = []
    for _ in range(runs):
        start = time.time()
        infer_fn()
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)

print("Average Latency (ms):", benchmark(lambda: None))

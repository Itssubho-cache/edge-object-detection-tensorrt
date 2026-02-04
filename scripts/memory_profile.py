import psutil
import os

process = psutil.Process(os.getpid())
print("CPU Memory (MB):", process.memory_info().rss / 1024 / 1024)

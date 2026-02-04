# Edge-Optimized Object Detection Using TensorRT

## Overview
This project focuses on optimizing deep learning–based object detection models
for real-time inference on edge and embedded platforms. The system emphasizes
low latency, high throughput, and efficient memory usage using TensorRT.

## Motivation
Deploying AI models on edge devices introduces strict constraints on latency,
memory, and compute resources. This project explores system-level and model-level
optimizations to meet real-time requirements.

## Tech Stack
- C++, Python
- PyTorch → ONNX → TensorRT
- OpenCV
- Linux

## Project Structure
edge-object-detection-tensorrt/
├── data/
├── models/
├── src/
├── scripts/
├── experiments/
└── docs/


## Key Features
- TensorRT-optimized inference (FP32 / FP16 / INT8)
- High-performance C++ inference pipeline
- Latency, FPS, and memory benchmarking
- Modular and extensible architecture

## Getting Started
Refer to:
- `docs/architecture.md`
- `docs/evaluation.md`

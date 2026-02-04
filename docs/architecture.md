## System Architecture

The system follows a modular inference pipeline designed for
real-time and edge deployment scenarios.

### Pipeline Stages
1. Input acquisition
2. Preprocessing
3. TensorRT inference
4. Postprocessing
5. Output handling

Each stage is decoupled to enable independent optimization,
profiling, and future extensions such as real-time streaming
using GStreamer and RTP.

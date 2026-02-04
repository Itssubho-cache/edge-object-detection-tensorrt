#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace nvinfer1;

/* ---------------- Logger ---------------- */
class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

/* ---------------- Utility: Load Engine ---------------- */
ICudaEngine* loadEngine(const std::string& enginePath, IRuntime*& runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error loading engine file!" << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime = createInferRuntime(gLogger);
    return runtime->deserializeCudaEngine(engineData.data(), size);
}

/* ---------------- Main ---------------- */
int main() {
    const std::string enginePath =
        "models/tensorrt/detector_fp16.engine";

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = loadEngine(enginePath, runtime);

    if (!engine) {
        std::cerr << "Failed to load TensorRT engine." << std::endl;
        return -1;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return -1;
    }

    /* -------- Allocate GPU buffers -------- */
    void* buffers[2];

    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");

    size_t inputSize  = 3 * 224 * 224 * sizeof(float);
    size_t outputSize = 1000 * sizeof(float); // example output

    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* -------- Run inference -------- */
    context->enqueueV2(buffers, stream, nullptr);
    cudaStreamSynchronize(stream);

    std::cout << "Inference executed successfully." << std::endl;

    /* -------- Cleanup -------- */
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}



// TODO:
// - Integrate OpenCV preprocessing
// - Add batch support
// - Add performance timing (latency measurement)
// - Integrate with GStreamer for live video inference

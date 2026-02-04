#include <iostream>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg
                  << ": " << cudaGetErrorString(result)
                  << std::endl;
    }
}

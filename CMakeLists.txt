cmake_minimum_required(VERSION 3.10)
project(edge_object_detection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include_directories(
    /usr/include
    /usr/include/x86_64-linux-gnu
)

link_directories(
    /usr/lib/x86_64-linux-gnu
)

add_executable(trt_infer
    src/inference/trt_infer.cpp
)

target_link_libraries(trt_infer
    nvinfer
    cudart
)

cmake_minimum_required(VERSION 3.28)
project(TinyINR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# CUDA kernel library:
add_library(kernels STATIC
    src/kernels/coordinate_embedding.cu
)
target_include_directories(kernels PUBLIC src/kernels)

# tests:
add_executable(test_fourier_embedding_cuda
    tests/test_fourier_embedding_cuda.cu
)
target_link_libraries(test_fourier_embedding_cuda PRIVATE kernels)
target_include_directories(test_fourier_embedding_cuda PRIVATE src/kernels)

add_executable(test_cpu_cuda_parity
    tests/test_cpu_cuda_parity.cu
)
target_link_libraries(test_cpu_cuda_parity PRIVATE kernels)
target_include_directories(test_cpu_cuda_parity PRIVATE src/kernels)

# benchmark target:
add_executable(bench_coordinate_embedding
    benchmarks/bench_coordinate_embedding.cu
)
target_link_libraries(bench_coordinate_embedding PRIVATE kernels)
target_include_directories(bench_coordinate_embedding PRIVATE src/kernels)

add_executable(benchmark_cpu_embedding
    benchmarks/benchmark_cpu_embedding.cpp
)
target_link_libraries(benchmark_cpu_embedding PRIVATE kernels)
target_include_directories(benchmark_cpu_embedding PRIVATE src/kernels)
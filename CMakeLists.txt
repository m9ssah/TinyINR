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
add_executable(test_coordinate_embedding
    tests/test_coordinate_embedding.cu
)
target_link_libraries(test_coordinate_embedding PRIVATE kernels)
target_include_directories(test_coordinate_embedding PRIVATE src/kernels)
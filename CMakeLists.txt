cmake_minimum_required(VERSION 3.28)
project(TinyINR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


add_library(tinyinr_core STATIC
    src/tensor.cpp
    src/model/mlp.cpp
    src/ops/coordinate_batch.cpp
    src/ops/fourier_embedding_raw.cpp
    src/ops/fourier_embedding.cpp
    src/training/cicfm.cpp
    src/training/loss.cpp
    src/training/train_debug.cpp
    src/training/train_step.cpp
)
target_include_directories(tinyinr_core PUBLIC include)

add_executable(test_fourier_embedding_tensor
    tests/test_fourier_embedding_tensor.cpp
)
target_link_libraries(test_fourier_embedding_tensor PRIVATE tinyinr_core)

add_executable(test_mlp_forward
    tests/test_mlp_forward.cpp
)
target_link_libraries(test_mlp_forward PRIVATE tinyinr_core)

add_executable(test_loss
    tests/test_loss.cpp
)
target_link_libraries(test_loss PRIVATE tinyinr_core)

add_executable(test_cicfm
    tests/test_cicfm.cpp
)
target_link_libraries(test_cicfm PRIVATE tinyinr_core)

add_executable(test_mlp_backward
    tests/test_mlp_backward.cpp
)
target_link_libraries(test_mlp_backward PRIVATE tinyinr_core)

add_executable(test_training_debug
    tests/test_training_debug.cpp
)
target_link_libraries(test_training_debug PRIVATE tinyinr_core)

add_executable(test_train_step
    tests/test_train_step.cpp
)
target_link_libraries(test_train_step PRIVATE tinyinr_core)

add_executable(test_full_size_baseline
    tests/test_full_size_baseline.cpp
)
target_link_libraries(test_full_size_baseline PRIVATE tinyinr_core)

if(CMAKE_CUDA_COMPILER)
    # CUDA kernel library:
    add_library(kernels STATIC
        src/kernels/coordinate_embedding.cu
        src/kernels/silu.cu
    )
    target_include_directories(kernels PUBLIC src/kernels)

    # CUDA tests:
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

    add_executable(test_silu_cuda
        tests/test_silu.cu
    )
    target_link_libraries(test_silu_cuda PRIVATE kernels tinyinr_core)
    target_include_directories(test_silu_cuda PRIVATE src/kernels)

    # CUDA benchmark target:
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
endif()

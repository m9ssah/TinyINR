The **dependency** of the benchmark files should be like this: 

coordinate_embedding.cu
        |
        v
CPU function + CUDA kernel
        |
        v
tests prove correctness
        |
        v
benchmarks measure speed
        |
        v
CSV stores results
        |
        v
plot script visualizes results
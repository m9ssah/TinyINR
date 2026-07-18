# Project Milestones

## Week 3
- [x] Repo structure setup
- [x] Fourier embedding implementation
- [x] Coordinate batch implementation
- [x] Tensor implementation
- [x] Integrate all 3 modules
- [x] Create website to host the blog
- [x] Write blog post about flow matching
- [ ] Write Unit tests for all 3 modules
 
```
Image
↓
CoordinateBatch
↓
Fourier Embedding
↓
MLP
↓
Predicted RGB
↓
Loss
↓
Backpropagation
↓
Gradient Descent
```

## Week 4
- [ ] port fourier coordinate embedding to CUDA
- [ ] prove kernel accelerates performance
- [ ] 

```
Your tasks:

Write CUDA allocation/copy helpers:
- cudaMalloc
- host → device copy
- device → host copy
- cudaFree

Implement the first kernel:
- one thread per output embedding element, or
- preferably one thread per coordinate dimension/frequency pair
- Flatten indexing from [B, N, D, F, sin_cos] into 1D memory.
- Add CUDA error checking after kernel launches.

Time:
- H2D transfer
- kernel execution
- D2H transfer
- total GPU pipeline
- Test output against the CPU embedding implementation.
```

```
CoordinateBatch coordinates
        ↓
FourierEmbedding CPU implementation
        ↓
CUDA coordinate embedding kernel
        ↓
Correctness comparison + benchmark report
```


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
- [x] port fourier coordinate embedding to CUDA
- [X] prove kernel accelerates performance

# Week 5 (MASSAH)
- [x] wire CoordinateBatch into CPU minibatch training
- [x] implement seeded uniform coordinate sampling with replacement
- [x] keep coordinates and targets aligned after sampling
- [x] write the CPU training loop driver
- [x] log loss, coverage, and sample counts
- [x] record CPU training-step timings, including sampling and priority feedback


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

## Week 6 — CPU adaptive sampling

- [x] sample using priorities while retaining uniform exploration
- [x] update original-point priorities from MSE/CICFM residuals
- [x] aggregate duplicate draws before applying each priority update
- [x] reproduce seeded training and fixed full-grid evaluation
- [x] pass all eleven Debug CPU tests and CLI regression checks
- [x] save matched uniform/adaptive runs and width-256 smoke evidence
- [x] document commands, measured results, and limitations in the
  [acceptance report](week6_adaptive_sampling_results.md)

These completed gates cover the CPU reference. CUDA sampling, attention
integration, and multi-seed performance experiments remain deferred.

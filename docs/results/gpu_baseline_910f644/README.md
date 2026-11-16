# NVIDIA baseline evidence

Source: `910f64406178b36e9c3e35a27f2bf604a5756619`, tested on Vast GPU 0,
2026-11-16 UTC. See [baseline report](../../gpu_adaptive_sampling_baseline.md)
for environment, source archive checksum, test inventory and build commands.

`transcript.txt` contains exported environment, configure/build, individual test,
CLI and compiler/sanitizer logs. The C++14 probe failure is intentional evidence
of installed CUB's C++17 requirement, not a failure of the existing project.
`toolkit_probe.cu` is a standalone baseline probe, not a production sampling kernel.

Reproduce the probe from this directory on the recorded NVIDIA toolchain:

```sh
export CUDA_VISIBLE_DEVICES=0
nvcc -std=c++14 -arch=sm_89 toolkit_probe.cu -o /tmp/tinyinr-probe14
# Expected compilation failure: CUB/libcu++ require C++17.
nvcc -std=c++17 -arch=sm_89 toolkit_probe.cu -o /tmp/tinyinr-probe17
/tmp/tinyinr-probe17
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/tinyinr-probe17
```

The probe verifies double inclusive scan results and Philox range with runtime
checks, then releases all allocations. Memcheck applies only to this probe;
full adaptive-kernel sanitizer coverage belongs to milestone 8.

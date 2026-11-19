#pragma once
#include <stdint.h>

struct uint4_rng {
  uint32_t x, y, z, w;
};

struct uint2_rng {
  uint32_t x, y;
};

__device__ __forceinline__ uint4_rng philox4x32(uint4_rng ctr, uint2_rng key,
                                                int round) {
  // magic constants
  constexpr uint32_t M0 = 0xD2511F53;
  constexpr uint32_t M1 = 0xCD9E8D57;
  constexpr uint32_t W0 = 0x9E3779B9;
  constexpr uint32_t W1 = 0xBB67AE85;

#pragma unroll
  for (int i = 0; i < round; i++) {
    uint32_t hi0 = __umulhi(M0, ctr.x);
    uint32_t lo0 = M0 * ctr.x;
    uint32_t hi1 = __umulhi(M1, ctr.z);
    uint32_t lo1 = M1 * ctr.z;

    ctr = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    key.x += W0;
    key.y += W1;
  }
  return ctr;
}

__device__ __forceinline__ uint4_rng philox_sample(uint64_t seed,
                                                   uint64_t index,
                                                   uint64_t step, int round) {
  uint2_rng key = {static_cast<uint32_t>(seed),
                   static_cast<uint32_t>(seed >> 32)};

  uint4_rng ctr = {
      static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32),
      static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32)};

  return philox4x32(ctr, key, round);
}

__device__ __forceinline__ float uint_to_uniform_float(uint32_t x) {
  return (x >> 8) * (1.0f / 16777216.0f); // uses top 24 bits of x
}
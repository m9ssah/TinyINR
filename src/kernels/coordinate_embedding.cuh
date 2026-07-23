#pragma once

__global__ void
coordinate_embedding(const float *d_input, float *d_output,
                     int B, // batch size
                     int N, // number of points per batch
                     int D, // coordinate dimension (e.g. 2 for 2D images)
                     int F  // number of frequency bands
);

void cpu_coordinate_embedding(const float *input, float *output, int B, int N,
                              int D, int F);

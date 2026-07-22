#include <cmath>
#include <iostream>
#include <vector>

#include "../src/kernels/coordinate_embedding.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float expected_value(float x, int f, int trig)
{
    float angle = std::pow(2.0f, static_cast<float>(f)) * static_cast<float>(M_PI) * x;
    return trig == 0 ? std::sin(angle) : std::cos(angle);
}
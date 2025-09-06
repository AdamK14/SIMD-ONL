#include "simd_lib.h"
#include <cmath>

namespace simd_lib {

void vector_add_scalar(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_multiply_scalar(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

float dot_product_scalar(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_subtract_scalar(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] - b[i];
    }
}

void vector_scale_scalar(const float* a, float scale, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * scale;
    }
}

float vector_norm_scalar(const float* a, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

float vector_norm_squared_scalar(const float* a, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i] * a[i];
    }
    return sum;
}

void vector_normalize_scalar(const float* a, float* result, size_t count) {
    float norm = vector_norm_scalar(a, count);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        vector_scale_scalar(a, inv_norm, result, count);
    } else {
        // Handle zero vector
        for (size_t i = 0; i < count; ++i) {
            result[i] = 0.0f;
        }
    }
}

} // namespace simd_lib

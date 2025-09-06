#include "simd_lib.h"

namespace simd_lib {

void matrix_multiply_4x4_scalar(const float* a, const float* b, float* result) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
}

void matrix_multiply_3x3_scalar(const float* a, const float* b, float* result) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
}

void matrix_vector_multiply_4x4_scalar(const float* matrix, const float* vector, float* result) {
    for (int i = 0; i < 4; ++i) {
        result[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            result[i] += matrix[i * 4 + j] * vector[j];
        }
    }
}

void matrix_vector_multiply_3x3_scalar(const float* matrix, const float* vector, float* result) {
    for (int i = 0; i < 3; ++i) {
        result[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i * 3 + j] * vector[j];
        }
    }
}

} // namespace simd_lib

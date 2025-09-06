#pragma once

#include <cstddef>
#include <cstdint>

namespace simd_lib {

// CPU feature detection
struct CPUFeatures {
    bool has_sse4_1 = false;
    bool has_sse4_2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
};

// Initialize CPU feature detection
void init_cpu_features();
const CPUFeatures& get_cpu_features();

// Vector addition functions
void vector_add(const float* a, const float* b, float* result, size_t count);
void vector_add_scalar(const float* a, const float* b, float* result, size_t count);
void vector_add_sse4(const float* a, const float* b, float* result, size_t count);
void vector_add_avx2(const float* a, const float* b, float* result, size_t count);

// Vector multiplication functions
void vector_multiply(const float* a, const float* b, float* result, size_t count);
void vector_multiply_scalar(const float* a, const float* b, float* result, size_t count);
void vector_multiply_avx2(const float* a, const float* b, float* result, size_t count);

// Dot product functions
float dot_product(const float* a, const float* b, size_t count);
float dot_product_scalar(const float* a, const float* b, size_t count);
float dot_product_avx2(const float* a, const float* b, size_t count);

// Vector operations
void vector_subtract(const float* a, const float* b, float* result, size_t count);
void vector_subtract_scalar(const float* a, const float* b, float* result, size_t count);
void vector_subtract_avx2(const float* a, const float* b, float* result, size_t count);

void vector_scale(const float* a, float scale, float* result, size_t count);
void vector_scale_scalar(const float* a, float scale, float* result, size_t count);
void vector_scale_avx2(const float* a, float scale, float* result, size_t count);

float vector_norm(const float* a, size_t count);
float vector_norm_scalar(const float* a, size_t count);
float vector_norm_avx2(const float* a, size_t count);

float vector_norm_squared(const float* a, size_t count);
float vector_norm_squared_scalar(const float* a, size_t count);
float vector_norm_squared_avx2(const float* a, size_t count);

void vector_normalize(const float* a, float* result, size_t count);
void vector_normalize_scalar(const float* a, float* result, size_t count);
void vector_normalize_avx2(const float* a, float* result, size_t count);

// Matrix operations
void matrix_multiply_4x4(const float* a, const float* b, float* result);
void matrix_multiply_4x4_scalar(const float* a, const float* b, float* result);
void matrix_multiply_3x3(const float* a, const float* b, float* result);
void matrix_multiply_3x3_scalar(const float* a, const float* b, float* result);
void matrix_vector_multiply_4x4(const float* matrix, const float* vector, float* result);
void matrix_vector_multiply_4x4_scalar(const float* matrix, const float* vector, float* result);
void matrix_vector_multiply_3x3(const float* matrix, const float* vector, float* result);
void matrix_vector_multiply_3x3_scalar(const float* matrix, const float* vector, float* result);

// FFT operations (basic implementation)
void fft_radix2(float* real, float* imag, size_t n, bool inverse = false);
void fft_forward(float* real, float* imag, size_t n);
void fft_inverse(float* real, float* imag, size_t n);

// Utility functions
void print_cpu_features();
const char* get_simd_version();

} // namespace simd_lib

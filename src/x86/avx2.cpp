#include "simd_lib.h"
#include <immintrin.h>
#include <cstring>
#include <cmath>

namespace simd_lib {

void vector_add_avx2(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    
    // Process 8 floats at a time (256-bit AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_multiply_avx2(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    
    // Process 8 floats at a time (256-bit AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

float dot_product_avx2(const float* a, const float* b, size_t count) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 floats at a time
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 mul_vec = _mm256_mul_ps(a_vec, b_vec);
        sum_vec = _mm256_add_ps(sum_vec, mul_vec);
    }
    
    // Horizontal sum of the 8 elements in sum_vec - more accurate approach
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    
    // Sum the two 128-bit vectors
    __m128 sum = _mm_add_ps(sum_high, sum_low);
    
    // Extract individual elements and sum them manually for better accuracy
    float result = 0.0f;
    float temp[4];
    _mm_storeu_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void vector_subtract_avx2(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    
    // Process 8 floats at a time (256-bit AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result[i] = a[i] - b[i];
    }
}

void vector_scale_avx2(const float* a, float scale, float* result, size_t count) {
    __m256 scale_vec = _mm256_set1_ps(scale);
    size_t i = 0;
    
    // Process 8 floats at a time (256-bit AVX2)
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 result_vec = _mm256_mul_ps(a_vec, scale_vec);
        _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result[i] = a[i] * scale;
    }
}

float vector_norm_squared_avx2(const float* a, size_t count) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 floats at a time
    for (; i + 8 <= count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 mul_vec = _mm256_mul_ps(a_vec, a_vec);
        sum_vec = _mm256_add_ps(sum_vec, mul_vec);
    }
    
    // Horizontal sum of the 8 elements in sum_vec
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    
    // Sum the two 128-bit vectors
    __m128 sum = _mm_add_ps(sum_high, sum_low);
    
    // Extract individual elements and sum them manually for better accuracy
    float result = 0.0f;
    float temp[4];
    _mm_storeu_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result += a[i] * a[i];
    }
    
    return result;
}

float vector_norm_avx2(const float* a, size_t count) {
    return std::sqrt(vector_norm_squared_avx2(a, count));
}

void vector_normalize_avx2(const float* a, float* result, size_t count) {
    float norm = vector_norm_avx2(a, count);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        vector_scale_avx2(a, inv_norm, result, count);
    } else {
        // Handle zero vector
        for (size_t i = 0; i < count; ++i) {
            result[i] = 0.0f;
        }
    }
}

} // namespace simd_lib

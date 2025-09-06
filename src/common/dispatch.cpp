#include "simd_lib.h"

namespace simd_lib {

void vector_add(const float* a, const float* b, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        vector_add_avx2(a, b, result, count);
    } else if (features.has_sse4_1) {
        vector_add_sse4(a, b, result, count);
    } else {
        vector_add_scalar(a, b, result, count);
    }
}

void vector_multiply(const float* a, const float* b, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        vector_multiply_avx2(a, b, result, count);
    } else {
        vector_multiply_scalar(a, b, result, count);
    }
}

float dot_product(const float* a, const float* b, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        return dot_product_avx2(a, b, count);
    } else {
        return dot_product_scalar(a, b, count);
    }
}

void vector_subtract(const float* a, const float* b, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        vector_subtract_avx2(a, b, result, count);
    } else {
        vector_subtract_scalar(a, b, result, count);
    }
}

void vector_scale(const float* a, float scale, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        vector_scale_avx2(a, scale, result, count);
    } else {
        vector_scale_scalar(a, scale, result, count);
    }
}

float vector_norm(const float* a, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        return vector_norm_avx2(a, count);
    } else {
        return vector_norm_scalar(a, count);
    }
}

float vector_norm_squared(const float* a, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        return vector_norm_squared_avx2(a, count);
    } else {
        return vector_norm_squared_scalar(a, count);
    }
}

void vector_normalize(const float* a, float* result, size_t count) {
    const auto& features = get_cpu_features();
    
    if (features.has_avx2) {
        vector_normalize_avx2(a, result, count);
    } else {
        vector_normalize_scalar(a, result, count);
    }
}

} // namespace simd_lib

#include "simd_lib.h"
#include <smmintrin.h>

namespace simd_lib {

void vector_add_sse4(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    
    // Process 4 floats at a time (128-bit SSE)
    for (; i + 4 <= count; i += 4) {
        __m128 a_vec = _mm_loadu_ps(&a[i]);
        __m128 b_vec = _mm_loadu_ps(&b[i]);
        __m128 result_vec = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(&result[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

} // namespace simd_lib

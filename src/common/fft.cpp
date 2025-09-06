#include "simd_lib.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace simd_lib {

// Helper function to check if n is a power of 2
bool is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Helper function to reverse bits for FFT
size_t reverse_bits(size_t x, size_t bits) {
    size_t result = 0;
    for (size_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void fft_radix2(float* real, float* imag, size_t n, bool inverse) {
    if (!is_power_of_2(n)) {
        // For simplicity, we only support power-of-2 sizes
        return;
    }
    
    size_t bits = 0;
    size_t temp = n;
    while (temp >>= 1) ++bits;
    
    // Bit-reverse the arrays
    for (size_t i = 0; i < n; ++i) {
        size_t j = reverse_bits(i, bits);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    
    // FFT computation
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * M_PI / len;
        float w_real = std::cos(angle);
        float w_imag = std::sin(angle);
        
        for (size_t i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            
            for (size_t j = 0; j < len / 2; ++j) {
                size_t u = i + j;
                size_t v = i + j + len / 2;
                
                float t_real = w_r * real[v] - w_i * imag[v];
                float t_imag = w_r * imag[v] + w_i * real[v];
                
                real[v] = real[u] - t_real;
                imag[v] = imag[u] - t_imag;
                real[u] += t_real;
                imag[u] += t_imag;
                
                float next_w_r = w_r * w_real - w_i * w_imag;
                float next_w_i = w_r * w_imag + w_i * w_real;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (inverse) {
        for (size_t i = 0; i < n; ++i) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

void fft_forward(float* real, float* imag, size_t n) {
    fft_radix2(real, imag, n, false);
}

void fft_inverse(float* real, float* imag, size_t n) {
    fft_radix2(real, imag, n, true);
}

} // namespace simd_lib

#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_fft_operations() {
    std::cout << "=== FFT Operations Test ===\n";
    
    const size_t n = 1024;  // Power of 2
    std::vector<float> real(n), imag(n);
    
    // Create a test signal: sum of sine waves
    for (size_t i = 0; i < n; ++i) {
        float t = (float)i / n;
        real[i] = std::sin(2.0f * M_PI * 10.0f * t) + 0.5f * std::sin(2.0f * M_PI * 25.0f * t);
        imag[i] = 0.0f;
    }
    
    // Test forward FFT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        // Reset data for each iteration
        for (size_t j = 0; j < n; ++j) {
            float t = (float)j / n;
            real[j] = std::sin(2.0f * M_PI * 10.0f * t) + 0.5f * std::sin(2.0f * M_PI * 25.0f * t);
            imag[j] = 0.0f;
        }
        simd_lib::fft_forward(real.data(), imag.data(), n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto fft_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Forward FFT (1024 points):\n";
    std::cout << "  Time: " << fft_time.count() << " us\n";
    std::cout << "  Time per FFT: " << fft_time.count() / 100.0f << " us\n\n";
    
    // Test inverse FFT
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        // Reset data for each iteration
        for (size_t j = 0; j < n; ++j) {
            float t = (float)j / n;
            real[j] = std::sin(2.0f * M_PI * 10.0f * t) + 0.5f * std::sin(2.0f * M_PI * 25.0f * t);
            imag[j] = 0.0f;
        }
        simd_lib::fft_forward(real.data(), imag.data(), n);
        simd_lib::fft_inverse(real.data(), imag.data(), n);
    }
    end = std::chrono::high_resolution_clock::now();
    auto ifft_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Inverse FFT (1024 points):\n";
    std::cout << "  Time: " << ifft_time.count() << " us\n";
    std::cout << "  Time per IFFT: " << ifft_time.count() / 100.0f << " us\n\n";
    
    // Test round-trip accuracy
    std::vector<float> original_real(n), original_imag(n);
    for (size_t i = 0; i < n; ++i) {
        original_real[i] = real[i];
        original_imag[i] = imag[i];
    }
    
    // Forward then inverse
    simd_lib::fft_forward(real.data(), imag.data(), n);
    simd_lib::fft_inverse(real.data(), imag.data(), n);
    
    // Check accuracy
    float max_error = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float error = std::abs(real[i] - original_real[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Round-trip accuracy test:\n";
    std::cout << "  Max error: " << std::scientific << std::setprecision(2) << max_error << "\n";
    std::cout << "  Accuracy: " << (max_error < 1e-5f ? "Good" : "Poor") << "\n\n";
    
    // Show frequency domain results
    std::cout << "Frequency domain (first 20 bins):\n";
    for (size_t i = 0; i < 20; ++i) {
        float magnitude = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);
        std::cout << "  Bin " << i << ": " << std::fixed << std::setprecision(3) << magnitude << "\n";
    }
}

int main() {
    std::cout << simd_lib::get_simd_version() << " - FFT Test\n\n";
    
    // Initialize CPU features
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    test_fft_operations();
    
    return 0;
}

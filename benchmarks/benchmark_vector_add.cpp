#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

void benchmark_vector_add(size_t count, int iterations = 100) {
    std::vector<float> a(count), b(count), result(count);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    
    for (size_t i = 0; i < count; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        simd_lib::vector_add(a.data(), b.data(), result.data(), count);
    }
    
    // Benchmark scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        simd_lib::vector_add_scalar(a.data(), b.data(), result.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Benchmark SIMD
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        simd_lib::vector_add(a.data(), b.data(), result.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double scalar_avg = (double)scalar_time.count() / iterations;
    double simd_avg = (double)simd_time.count() / iterations;
    double speedup = scalar_avg / simd_avg;
    
    std::cout << std::setw(10) << count 
              << std::setw(15) << std::fixed << std::setprecision(2) << scalar_avg / 1000.0
              << std::setw(15) << std::fixed << std::setprecision(2) << simd_avg / 1000.0
              << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x\n";
}

int main() {
    std::cout << simd_lib::get_simd_version() << " - Vector Addition Benchmark\n\n";
    
    // Initialize CPU features
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    std::cout << std::setw(10) << "Elements"
              << std::setw(15) << "Scalar (μs)"
              << std::setw(15) << "SIMD (μs)"
              << std::setw(10) << "Speedup\n";
    std::cout << std::string(50, '-') << "\n";
    
    // Test different sizes
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    for (size_t size : sizes) {
        benchmark_vector_add(size);
    }
    
    return 0;
}

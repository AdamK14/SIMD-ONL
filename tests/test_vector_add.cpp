#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

void test_vector_add() {
    const size_t count = 1000000;
    std::vector<float> a(count), b(count), result_scalar(count), result_simd(count);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    
    for (size_t i = 0; i < count; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // Test scalar implementation
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_add_scalar(a.data(), b.data(), result_scalar.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test SIMD implementation
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_add(a.data(), b.data(), result_simd.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify results are the same
    bool correct = true;
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(result_scalar[i] - result_simd[i]) > 1e-6f) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Vector Addition Test:\n";
    std::cout << "  Elements: " << count << "\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n";
    std::cout << "  Correct:     " << (correct ? "Yes" : "No") << "\n\n";
}

void test_dot_product() {
    const size_t count = 1000000;
    std::vector<float> a(count), b(count);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < count; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // Test scalar implementation
    auto start = std::chrono::high_resolution_clock::now();
    float scalar_result = 0.0f;
    for (int i = 0; i < 100; ++i) {
        scalar_result = simd_lib::dot_product_scalar(a.data(), b.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test SIMD implementation
    start = std::chrono::high_resolution_clock::now();
    float simd_result = 0.0f;
    for (int i = 0; i < 100; ++i) {
        simd_result = simd_lib::dot_product(a.data(), b.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify results are the same
    bool correct = std::abs(scalar_result - simd_result) < 1e-5f;
    
    std::cout << "Dot Product Test:\n";
    std::cout << "  Elements: " << count << "\n";
    std::cout << "  Scalar result: " << scalar_result << "\n";
    std::cout << "  SIMD result:   " << simd_result << "\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n";
    std::cout << "  Correct:     " << (correct ? "Yes" : "No") << "\n\n";
}

int main() {
    std::cout << simd_lib::get_simd_version() << "\n\n";
    
    // Initialize CPU features
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    // Run tests
    test_vector_add();
    test_dot_product();
    
    return 0;
}

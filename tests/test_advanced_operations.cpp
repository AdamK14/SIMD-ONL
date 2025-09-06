#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

void test_vector_operations() {
    std::cout << "=== Vector Operations Test ===\n";
    
    const size_t count = 1000000;
    std::vector<float> a(count), b(count), result(count);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (size_t i = 0; i < count; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // Test vector subtraction
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_subtract_scalar(a.data(), b.data(), result.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_subtract(a.data(), b.data(), result.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Subtraction:\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n\n";
    
    // Test vector scaling
    float scale = 2.5f;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_scale_scalar(a.data(), scale, result.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        simd_lib::vector_scale(a.data(), scale, result.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Scaling:\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n\n";
    
    // Test vector norm
    start = std::chrono::high_resolution_clock::now();
    float scalar_norm = 0.0f;
    for (int i = 0; i < 100; ++i) {
        scalar_norm = simd_lib::vector_norm_scalar(a.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    float simd_norm = 0.0f;
    for (int i = 0; i < 100; ++i) {
        simd_norm = simd_lib::vector_norm(a.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Norm:\n";
    std::cout << "  Scalar result: " << std::fixed << std::setprecision(6) << scalar_norm << "\n";
    std::cout << "  SIMD result:   " << std::fixed << std::setprecision(6) << simd_norm << "\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n";
    std::cout << "  Error:       " << std::fabs(scalar_norm - simd_norm) / scalar_norm * 100.0f << "%\n\n";
}

void test_matrix_operations() {
    std::cout << "=== Matrix Operations Test ===\n";
    
    // Test 4x4 matrix multiplication
    std::vector<float> a(16), b(16), result(16);
    
    // Fill with simple values for testing
    for (int i = 0; i < 16; ++i) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 1) * 0.1f;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        simd_lib::matrix_multiply_4x4_scalar(a.data(), b.data(), result.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        simd_lib::matrix_multiply_4x4(a.data(), b.data(), result.data());
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "4x4 Matrix Multiplication:\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n\n";
    
    // Test 4x4 matrix-vector multiplication
    std::vector<float> matrix(16), vector(4), result_vec(4);
    
    for (int i = 0; i < 16; ++i) {
        matrix[i] = (float)(i + 1) * 0.1f;
    }
    for (int i = 0; i < 4; ++i) {
        vector[i] = (float)(i + 1);
    }
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        simd_lib::matrix_vector_multiply_4x4_scalar(matrix.data(), vector.data(), result_vec.data());
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        simd_lib::matrix_vector_multiply_4x4(matrix.data(), vector.data(), result_vec.data());
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "4x4 Matrix-Vector Multiplication:\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " us\n";
    std::cout << "  SIMD time:   " << simd_time.count() << " us\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) 
              << (double)scalar_time.count() / simd_time.count() << "x\n";
    std::cout << "  Result: [" << result_vec[0] << ", " << result_vec[1] 
              << ", " << result_vec[2] << ", " << result_vec[3] << "]\n\n";
}

int main() {
    std::cout << simd_lib::get_simd_version() << " - Advanced Operations Test\n\n";
    
    // Initialize CPU features
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    test_vector_operations();
    test_matrix_operations();
    
    return 0;
}

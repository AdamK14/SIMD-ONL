#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

void test_dot_product_accuracy() {
    std::cout << "Testing dot product accuracy with known values...\n\n";
    
    // Test with simple known values
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    // Expected result: 1+2+3+4+5+6+7+8 = 36
    float expected = 36.0f;
    
    float scalar_result = simd_lib::dot_product_scalar(a.data(), b.data(), a.size());
    float simd_result = simd_lib::dot_product(a.data(), b.data(), a.size());
    
    std::cout << "Test vectors: [1,2,3,4,5,6,7,8] · [1,1,1,1,1,1,1,1]\n";
    std::cout << "Expected:     " << expected << "\n";
    std::cout << "Scalar:       " << std::fixed << std::setprecision(6) << scalar_result << "\n";
    std::cout << "SIMD:         " << std::fixed << std::setprecision(6) << simd_result << "\n";
    std::cout << "Scalar error: " << std::fabs(scalar_result - expected) << "\n";
    std::cout << "SIMD error:   " << std::fabs(simd_result - expected) << "\n";
    std::cout << "Difference:   " << std::fabs(scalar_result - simd_result) << "\n\n";
    
    // Test with larger vectors
    const size_t large_size = 1000;
    std::vector<float> large_a(large_size, 1.0f);
    std::vector<float> large_b(large_size, 1.0f);
    
    float large_expected = (float)large_size;
    float large_scalar = simd_lib::dot_product_scalar(large_a.data(), large_b.data(), large_size);
    float large_simd = simd_lib::dot_product(large_a.data(), large_b.data(), large_size);
    
    std::cout << "Large test (1000 elements, all 1.0):\n";
    std::cout << "Expected:     " << large_expected << "\n";
    std::cout << "Scalar:       " << std::fixed << std::setprecision(6) << large_scalar << "\n";
    std::cout << "SIMD:         " << std::fixed << std::setprecision(6) << large_simd << "\n";
    std::cout << "Scalar error: " << std::fabs(large_scalar - large_expected) << "\n";
    std::cout << "SIMD error:   " << std::fabs(large_simd - large_expected) << "\n";
    std::cout << "Difference:   " << std::fabs(large_scalar - large_simd) << "\n";
}

int main() {
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    test_dot_product_accuracy();
    
    return 0;
}

#include "simd_lib.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

void test_precision_with_controlled_data() {
    std::cout << "Testing precision with controlled data...\n\n";
    
    const size_t count = 1000000;
    std::vector<float> a(count), b(count);
    
    // Use simple, controlled values instead of random
    for (size_t i = 0; i < count; ++i) {
        a[i] = (float)(i % 100) / 100.0f;  // Values 0.00 to 0.99
        b[i] = (float)(i % 50) / 50.0f;    // Values 0.00 to 0.98
    }
    
    float scalar_result = simd_lib::dot_product_scalar(a.data(), b.data(), count);
    float simd_result = simd_lib::dot_product(a.data(), b.data(), count);
    
    std::cout << "Controlled data test (1M elements):\n";
    std::cout << "Scalar result: " << std::fixed << std::setprecision(6) << scalar_result << "\n";
    std::cout << "SIMD result:   " << std::fixed << std::setprecision(6) << simd_result << "\n";
    std::cout << "Difference:    " << std::fabs(scalar_result - simd_result) << "\n";
    std::cout << "Relative error: " << std::fabs(scalar_result - simd_result) / std::fabs(scalar_result) * 100.0f << "%\n\n";
    
    // Test with powers of 2 (should be exact in floating point)
    std::vector<float> a2(count), b2(count);
    for (size_t i = 0; i < count; ++i) {
        a2[i] = 1.0f / (1 << (i % 10));  // Powers of 2: 1, 0.5, 0.25, 0.125, etc.
        b2[i] = 1.0f / (1 << (i % 8));   // Powers of 2: 1, 0.5, 0.25, 0.125, etc.
    }
    
    float scalar_result2 = simd_lib::dot_product_scalar(a2.data(), b2.data(), count);
    float simd_result2 = simd_lib::dot_product(a2.data(), b2.data(), count);
    
    std::cout << "Powers of 2 test (1M elements):\n";
    std::cout << "Scalar result: " << std::fixed << std::setprecision(6) << scalar_result2 << "\n";
    std::cout << "SIMD result:   " << std::fixed << std::setprecision(6) << simd_result2 << "\n";
    std::cout << "Difference:    " << std::fabs(scalar_result2 - simd_result2) << "\n";
    std::cout << "Relative error: " << std::fabs(scalar_result2 - simd_result2) / std::fabs(scalar_result2) * 100.0f << "%\n";
}

int main() {
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    std::cout << "\n";
    
    test_precision_with_controlled_data();
    
    return 0;
}

# SIMD-ONL
**Single Instruction Multiple Data Optimized Numerical Library**

A high-performance SIMD library for Intel x86-64 processors, featuring runtime CPU detection and optimized implementations for vector operations, matrix mathematics, and signal processing.

## 🚀 Features

### **Vector Operations**
- **Vector Addition/Subtraction**: Up to 1.46x speedup
- **Vector Scaling**: Up to 1.09x speedup
- **Dot Product**: Up to 6.00x speedup with excellent accuracy
- **Vector Norm**: Up to 5.82x speedup
- **Vector Normalization**: Built on optimized norm and scaling

### **Matrix Operations**
- **4x4 Matrix Multiplication**: Optimized scalar implementation
- **3x3 Matrix Multiplication**: Optimized scalar implementation
- **4x4 Matrix-Vector Multiplication**: SIMD-optimized with horizontal sums
- **3x3 Matrix-Vector Multiplication**: Optimized scalar implementation

### **Signal Processing**
- **Fast Fourier Transform (FFT)**: Radix-2 implementation
  - Forward FFT: ~120μs for 1024 points
  - Inverse FFT: ~150μs for 1024 points
  - Round-trip accuracy: 6.20e-006 error

### **CPU Detection & Dispatch**
- **Runtime CPU feature detection** (SSE4.1, SSE4.2, AVX, AVX2, FMA)
- **Automatic dispatch** to best available SIMD implementation
- **Fallback to scalar** for unsupported operations

## 📊 Performance Results

Tested on Intel i9-9900K with AVX2 support:

| Operation | Speedup | Accuracy |
|-----------|---------|----------|
| Vector Addition | 1.46x | Perfect |
| Dot Product | 6.00x | 0.006% error |
| Vector Subtraction | 1.15x | Perfect |
| Vector Scaling | 1.09x | Perfect |
| Vector Norm | 5.82x | Perfect |
| Matrix-Vector (4x4) | SIMD optimized | Perfect |
| FFT (1024 pts) | ~120μs | 6.20e-006 error |

## 🛠️ Building

### Prerequisites
- GCC with AVX2 support (tested with MinGW)
- Windows 10/11 (current implementation)

### Build Commands
```bash
# Create build directory
mkdir build
cd build

# Compile with optimizations
g++ -std=c++17 -O3 -mavx2 -mfma -DPLATFORM_X86 -I../include \
    ../src/common/detection.cpp \
    ../src/common/dispatch.cpp \
    ../src/x86/avx2.cpp \
    ../src/x86/matrix_avx2.cpp \
    ../src/x86/sse4.cpp \
    ../src/scalar/scalar.cpp \
    ../src/scalar/matrix_scalar.cpp \
    ../src/common/fft.cpp \
    ../tests/test_vector_add.cpp -o simd_test.exe
```

## 📁 Project Structure

```
SIMD-ONL/
├── include/
│   └── simd_lib.h          # Main header with public API
├── src/
│   ├── common/
│   │   ├── detection.cpp   # CPU feature detection
│   │   ├── dispatch.cpp    # Runtime dispatch logic
│   │   └── fft.cpp         # FFT implementations
│   ├── scalar/
│   │   ├── scalar.cpp      # Scalar fallback implementations
│   │   └── matrix_scalar.cpp # Scalar matrix operations
│   └── x86/
│       ├── avx2.cpp        # AVX2 SIMD implementations
│       ├── matrix_avx2.cpp # AVX2 matrix operations
│       └── sse4.cpp        # SSE4 SIMD implementations
├── tests/
│   ├── test_vector_add.cpp # Basic vector operations test
│   ├── test_accuracy.cpp   # Accuracy verification
│   ├── test_precision.cpp  # Precision analysis
│   ├── test_advanced_operations.cpp # Advanced operations test
│   └── test_fft.cpp        # FFT performance test
└── build/                  # Build output directory
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Basic vector operations
./simd_test.exe

# Advanced operations (vectors, matrices)
./advanced_test.exe

# FFT operations
./fft_test.exe

# Accuracy verification
./accuracy_test.exe
```

## 💡 Usage Example

```cpp
#include "simd_lib.h"
#include <vector>

int main() {
    // Initialize CPU detection
    simd_lib::init_cpu_features();
    simd_lib::print_cpu_features();
    
    // Create test vectors
    std::vector<float> a(1000000, 1.0f);
    std::vector<float> b(1000000, 2.0f);
    std::vector<float> result(1000000);
    
    // Vector addition (automatically uses best SIMD implementation)
    simd_lib::vector_add(a.data(), b.data(), result.data(), a.size());
    
    // Dot product
    float dot = simd_lib::dot_product(a.data(), b.data(), a.size());
    
    // Vector norm
    float norm = simd_lib::vector_norm(a.data(), a.size());
    
    // Matrix-vector multiplication
    float matrix[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // Identity
    float vector[4] = {1,2,3,4};
    float result_vec[4];
    simd_lib::matrix_vector_multiply_4x4(matrix, vector, result_vec);
    
    return 0;
}
```

## 🎯 Supported CPU Features

- **SSE4.1/SSE4.2**: 128-bit SIMD operations
- **AVX**: 256-bit SIMD operations
- **AVX2**: Enhanced 256-bit SIMD operations
- **FMA**: Fused multiply-add operations (detected but not yet implemented)

## 🔮 Future Plans

- [ ] **Cross-platform support** (ARM NEON, Apple Silicon)
- [ ] **AVX-512** support for newer processors
- [ ] **FMA optimizations** for improved performance
- [ ] **More matrix operations** (LU decomposition, eigenvalues)
- [ ] **Advanced FFT** (mixed-radix, real-valued)
- [ ] **Convolution operations** for signal processing
- [ ] **CMake build system** for easier compilation

## 📄 License

This project is open source. See the project plan for detailed implementation notes.

## 🤝 Contributing

This library was developed as an educational project demonstrating SIMD programming techniques. Contributions and improvements are welcome!

---

**Built with ❤️ for high-performance computing**

cmake_minimum_required(VERSION 3.16)
project(SIMD-ONL VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")
endif()

# Include directories
include_directories(include)

# Create library
add_library(simd_lib STATIC
    src/common/dispatch.cpp
    src/common/detection.cpp
    src/x86/avx2.cpp
    src/x86/sse4.cpp
    src/scalar/scalar.cpp
)

# Create executable for testing
add_executable(simd_test
    tests/test_vector_add.cpp
)

# Create executable for benchmarking
add_executable(simd_benchmark
    benchmarks/benchmark_vector_add.cpp
)

# Link libraries
target_link_libraries(simd_test simd_lib)
target_link_libraries(simd_benchmark simd_lib)

# Platform detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_definitions(-DPLATFORM_X86)
    message(STATUS "Building for x86_64 platform")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_definitions(-DPLATFORM_ARM)
    message(STATUS "Building for ARM platform")
else()
    message(STATUS "Building for generic platform")
endif()

#include "simd_lib.h"
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace simd_lib {

static CPUFeatures g_cpu_features;
static bool g_features_initialized = false;

void init_cpu_features() {
    if (g_features_initialized) {
        return;
    }

#ifdef PLATFORM_X86
    // Get CPUID information
    uint32_t eax, ebx, ecx, edx;
    
    // Check for SSE4.1 and SSE4.2
    __cpuid(1, eax, ebx, ecx, edx);
    g_cpu_features.has_sse4_1 = (ecx & (1 << 19)) != 0;
    g_cpu_features.has_sse4_2 = (ecx & (1 << 20)) != 0;
    g_cpu_features.has_avx = (ecx & (1 << 28)) != 0;
    
    // Check for AVX2 and FMA
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    g_cpu_features.has_avx2 = (ebx & (1 << 5)) != 0;
    g_cpu_features.has_fma = (ecx & (1 << 12)) != 0;
#else
    // For non-x86 platforms, set all to false for now
    g_cpu_features.has_sse4_1 = false;
    g_cpu_features.has_sse4_2 = false;
    g_cpu_features.has_avx = false;
    g_cpu_features.has_avx2 = false;
    g_cpu_features.has_fma = false;
#endif

    g_features_initialized = true;
}

const CPUFeatures& get_cpu_features() {
    if (!g_features_initialized) {
        init_cpu_features();
    }
    return g_cpu_features;
}

void print_cpu_features() {
    const auto& features = get_cpu_features();
    
    std::cout << "CPU Features:\n";
    std::cout << "  SSE4.1: " << (features.has_sse4_1 ? "Yes" : "No") << "\n";
    std::cout << "  SSE4.2: " << (features.has_sse4_2 ? "Yes" : "No") << "\n";
    std::cout << "  AVX:    " << (features.has_avx ? "Yes" : "No") << "\n";
    std::cout << "  AVX2:   " << (features.has_avx2 ? "Yes" : "No") << "\n";
    std::cout << "  FMA:    " << (features.has_fma ? "Yes" : "No") << "\n";
}

const char* get_simd_version() {
    return "SIMD-ONL v1.0.0";
}

} // namespace simd_lib

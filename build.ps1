Write-Host "Building SIMD-ONL..." -ForegroundColor Green

# Create build directory
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Name "build"
}
Set-Location "build"

# Compile with GCC
$compileArgs = @(
    "-std=c++17", "-O3", "-mavx2", "-mfma", "-I../include",
    "../src/common/detection.cpp",
    "../src/common/dispatch.cpp", 
    "../src/x86/avx2.cpp",
    "../src/x86/sse4.cpp",
    "../src/scalar/scalar.cpp",
    "../tests/test_vector_add.cpp",
    "-o", "simd_test.exe"
)

Write-Host "Compiling..." -ForegroundColor Yellow
& g++ @compileArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful! Running tests..." -ForegroundColor Green
    Write-Host ""
    & .\simd_test.exe
} else {
    Write-Host "Build failed!" -ForegroundColor Red
}

Set-Location ".."

@echo off
echo Building SIMD-ONL...

REM Create build directory
if not exist build mkdir build
cd build

REM Compile with GCC
g++ -std=c++17 -O3 -mavx2 -mfma -DPLATFORM_X86 -I../include ^
    ../src/common/detection.cpp ^
    ../src/common/dispatch.cpp ^
    ../src/x86/avx2.cpp ^
    ../src/x86/sse4.cpp ^
    ../src/scalar/scalar.cpp ^
    ../tests/test_vector_add.cpp ^
    -o simd_test.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Running tests...
    echo.
    simd_test.exe
) else (
    echo Build failed!
)

cd ..
pause

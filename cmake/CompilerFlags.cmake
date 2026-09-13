# CompilerFlags.cmake - Compiler and CUDA flags for IMP

# Warning flags (-Wall -Wextra -Wpedantic) intentionally not global here: they leak onto
# FetchContent deps (gtest/cutlass). Live on the imp_warnings INTERFACE target (CMakeLists.txt),
# linked PRIVATE, scoped to CXX only (never nvcc).
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DIMP_DEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -DNDEBUG")
# RelWithDebInfo: keep Release-grade optimizer (-O3, host vectorization), add -g
# for host stack frames so profilers/asserts get source mapping. Without -O3 the
# host launch loop becomes a measurable perf floor for decode-heavy workloads.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=x86-64-v3 -g -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3 -DNDEBUG")
# RelWithDebInfo: full Release-grade device optimizer (-O3, fast-math, PTX -O3,
# extra-device-vectorization) plus -lineinfo for source-line mapping in Nsight Compute/Systems
# and CUDA error reports. -lineinfo is code-gen-neutral.
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3 -lineinfo -DNDEBUG")

# Suppress noisy CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# CUTLASS sm100/sm103 headers use [=] lambdas that implicitly capture 'this' (deprecated in C++20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=2908 --diag-suppress=177")

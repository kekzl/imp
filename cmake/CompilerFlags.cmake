# CompilerFlags.cmake - Compiler and CUDA flags for IMP

# C++ flags
# NOTE: warning flags (-Wall -Wextra -Wpedantic) are NOT set globally here — that
# leaked them onto FetchContent deps (gtest/cutlass) as un-fixable noise. They now
# live on the `imp_warnings` INTERFACE target in CMakeLists.txt, linked PRIVATE by
# every first-party target and scoped to $<COMPILE_LANGUAGE:CXX> (host TUs only,
# never nvcc), which preserves the previous behavior for our own code.
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
# extra-device-vectorization), plus -lineinfo so profilers (Nsight Compute,
# Nsight Systems) and CUDA error reports get source-line mapping. -lineinfo is
# code-gen-neutral; the previous "-O2 -g" (CMake default) cost ~2x decode and
# ~4x prefill on Qwen3-8B Q8_0 vs Release because --use_fast_math and PTX -O3
# were missing.
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3 -lineinfo -DNDEBUG")

# Suppress noisy CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# CUTLASS sm100/sm103 headers use [=] lambdas that implicitly capture 'this' (deprecated in C++20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=2908 --diag-suppress=177")

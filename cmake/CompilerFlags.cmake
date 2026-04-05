# CompilerFlags.cmake - Compiler and CUDA flags for IMP

# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DIMP_DEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo -DNDEBUG")

# Suppress noisy CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# CUTLASS sm100/sm103 headers use [=] lambdas that implicitly capture 'this' (deprecated in C++20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=2908")

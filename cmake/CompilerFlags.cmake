# CompilerFlags.cmake - Compiler and CUDA flags for IMP

# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DIMP_DEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
# Note: --use_fast_math is replaced by its constituent flags to allow per-file
# override via --ftz=false etc. (CUDA 13.2 ptxas bug on sm_120a for TurboQuant).
# --use_fast_math = --ftz=true --prec-div=false --prec-sqrt=false --fmad=true
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --extra-device-vectorization -Xptxas -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo -DNDEBUG")

# Suppress noisy CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

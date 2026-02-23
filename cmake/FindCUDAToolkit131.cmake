# FindCUDAToolkit131.cmake
# Extended CUDA Toolkit detection for CUDA 13.1 specific features
#
# Sets:
#   CUDA_131_GREEN_CONTEXTS  - Green Contexts available
#   CUDA_131_GROUPED_GEMM    - cuBLASLt Grouped GEMM available
#   CUDA_131_PDL             - Programmatic Dependent Launch available

include(CheckCXXSourceCompiles)

if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.1")
    set(CUDA_131_GREEN_CONTEXTS TRUE)
    set(CUDA_131_GROUPED_GEMM TRUE)
    set(CUDA_131_PDL TRUE)
    message(STATUS "CUDA 13.1 features: Green Contexts=ON, Grouped GEMM=ON, PDL=ON")
else()
    set(CUDA_131_GREEN_CONTEXTS FALSE)
    set(CUDA_131_GROUPED_GEMM FALSE)
    set(CUDA_131_PDL FALSE)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA ${CUDAToolkit_VERSION}: 13.1 features not available")
    endif()
endif()

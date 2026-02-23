cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(imp LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerFlags)

# --- CUDA Toolkit ---
find_package(CUDAToolkit 13.1 REQUIRED)
message(STATUS "CUDA ${CUDAToolkit_VERSION} detected")
add_compile_definitions(IMP_CUDA_13_1=1)

# --- CUDA Architecture ---
# sm_90a = Hopper, sm_100 = Blackwell B200, sm_120 = Blackwell B300
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90a;100;120")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# --- Options ---
option(IMP_BUILD_TESTS "Build tests" ON)
option(IMP_BUILD_TOOLS "Build CLI and benchmark tools" ON)
option(IMP_BUILD_BENCH "Build benchmarks" ON)

# --- Third-party dependencies ---
include(FetchContent)

if(IMP_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# --- Core library sources ---
set(IMP_CORE_SOURCES
    src/core/tensor.cpp
    src/core/buffer.cpp
    src/core/allocator.cpp
    src/core/logging.cpp
    src/core/threading.cpp
)

set(IMP_MEMORY_SOURCES
    src/memory/device_allocator.cu
    src/memory/pinned_allocator.cpp
    src/memory/kv_cache.cu
    src/memory/kv_cache_manager.cpp
)

set(IMP_MODEL_SOURCES
    src/model/model.cpp
    src/model/gguf_loader.cpp
    src/model/safetensors_loader.cpp
    src/model/weight_map.cpp
    src/model/weight_upload.cu
    src/model/tokenizer.cpp
)

set(IMP_QUANT_SOURCES
    src/quant/dequant_fp16.cu
    src/quant/dequant_int8.cu
    src/quant/fp8_utils.cu
    src/quant/quant_gemm.cu
    src/quant/fp8_quant.cu
    src/quant/nvfp4_quant.cu
    src/quant/nvfp4_gemm.cu
)

set(IMP_COMPUTE_SOURCES
    src/compute/gemm.cu
    src/compute/gemm_grouped.cu
    src/compute/attention.cu
    src/compute/attention_paged.cu
    src/compute/attention_tc.cu
    src/compute/attention_blackwell.cu
    src/compute/attention_dispatch.cu
    src/compute/rope.cu
    src/compute/layernorm.cu
    src/compute/activation.cu
    src/compute/embedding.cu
    src/compute/sampling.cu
    src/compute/softmax.cu
    src/compute/moe_routing.cu
    src/compute/reduce.cu
)

set(IMP_RUNTIME_SOURCES
    src/runtime/green_ctx.cu
    src/runtime/scheduler.cpp
    src/runtime/request.cpp
    src/runtime/batch.cpp
    src/runtime/engine.cpp
    src/runtime/cuda_graph.cu
    src/runtime/pdl.cu
    src/runtime/speculative.cpp
)

set(IMP_GRAPH_SOURCES
    src/graph/op.cpp
    src/graph/graph.cpp
    src/graph/executor.cu
)

set(IMP_API_SOURCES
    src/api/imp_api.cpp
)

# --- Main library target ---
add_library(imp STATIC
    ${IMP_CORE_SOURCES}
    ${IMP_MEMORY_SOURCES}
    ${IMP_MODEL_SOURCES}
    ${IMP_QUANT_SOURCES}
    ${IMP_COMPUTE_SOURCES}
    ${IMP_RUNTIME_SOURCES}
    ${IMP_GRAPH_SOURCES}
    ${IMP_API_SOURCES}
)

target_include_directories(imp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(imp
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    PRIVATE
        pthread
)

# Enable separable compilation for device-side launches (PDL)
set_target_properties(imp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# --- Tools ---
if(IMP_BUILD_TOOLS)
    # imp-cli
    add_executable(imp-cli
        tools/imp-cli/main.cpp
        tools/imp-cli/args.cpp
    )
    target_link_libraries(imp-cli PRIVATE imp)
    target_include_directories(imp-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # imp-bench
    if(IMP_BUILD_BENCH)
        add_executable(imp-bench
            tools/imp-bench/main.cpp
            tools/imp-bench/bench_gemm.cu
            tools/imp-bench/bench_attention.cu
            tools/imp-bench/bench_e2e.cpp
        )
        target_link_libraries(imp-bench PRIVATE imp)
        target_include_directories(imp-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    endif()
endif()

# --- Tests ---
if(IMP_BUILD_TESTS)
    enable_testing()

    set(IMP_TEST_SOURCES
        tests/test_tensor.cpp
        tests/test_gguf_loader.cpp
        tests/test_kv_cache.cpp
        tests/test_attention.cu
        tests/test_rope.cu
        tests/test_layernorm.cu
        tests/test_moe.cu
        tests/test_quant.cu
        tests/test_green_ctx.cu
        tests/test_moe_executor.cu
        tests/test_quant_integration.cu
        tests/test_e2e.cpp
        tests/test_continuous_batching.cpp
        tests/test_fp8_gemm.cu
        tests/test_attention_tc.cu
        tests/test_nvfp4_quant.cu
        tests/test_speculative.cpp
    )

    add_executable(imp-tests ${IMP_TEST_SOURCES})
    target_link_libraries(imp-tests PRIVATE imp GTest::gtest GTest::gtest_main)
    target_include_directories(imp-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    include(GoogleTest)
    gtest_discover_tests(imp-tests)
endif()

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
# When upgrading to CUDA 13.2: uncomment for MXFP8 Grouped GEMM + cub::DeviceTopK
# add_compile_definitions(IMP_CUDA_13_2=1)

# --- CUDA Architecture ---
# sm_90a = Hopper, sm_100 = Blackwell B200, sm_120a = Blackwell B300/RTX 5090
option(IMP_NATIVE_SM120 "Build exclusively for sm_120 (RTX 5090) — enables native Blackwell SASS" OFF)
if(IMP_NATIVE_SM120)
    set(CMAKE_CUDA_ARCHITECTURES "120a")
    message(STATUS "CUDA architectures: 120a (native RTX 5090 build)")
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90a;100;120a")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# --- Options ---
option(IMP_BUILD_TESTS "Build tests" ON)
option(IMP_BUILD_TOOLS "Build CLI and benchmark tools" ON)
option(IMP_BUILD_BENCH "Build benchmarks" ON)
option(IMP_BUILD_SERVER "Build imp-server (OpenAI-compatible HTTP server)" ON)

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

option(IMP_USE_CUTLASS "Use CUTLASS for MoE GEMM (requires sm_90+)" ON)
if(IMP_USE_CUTLASS)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.4.1
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    # CUTLASS is header-only for our use — disable all build targets
    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cutlass)
endif()

# --- Core library sources ---
set(IMP_CORE_SOURCES
    src/core/tensor.cpp
    src/core/buffer.cpp
    src/core/allocator.cpp
    src/core/logging.cpp
    src/core/threading.cpp
    src/core/toml_parser.cpp
)

set(IMP_MEMORY_SOURCES
    src/memory/device_allocator.cu
    src/memory/pinned_allocator.cpp
    src/memory/kv_cache.cu
    src/memory/kv_cache_manager.cpp
    src/memory/ssm_state.cu
    src/memory/layer_offload.cu
)

set(IMP_MODEL_SOURCES
    src/model/model.cpp
    src/model/gguf_loader.cpp
    src/model/safetensors_loader.cpp
    src/model/weight_map.cpp
    src/model/weight_upload.cu
    src/model/tokenizer.cpp
    src/model/chat_template.cpp
)

set(IMP_QUANT_SOURCES
    src/quant/dequant_fp16.cu
    src/quant/dequant_int8.cu
    src/quant/dequant_gpu.cu
    src/quant/fp8_utils.cu
    src/quant/quant_gemm.cu
    src/quant/fp8_quant.cu
    src/quant/nvfp4_quant.cu
    src/quant/nvfp4_gemm.cu
)

set(IMP_COMPUTE_SOURCES
    src/compute/gemm.cu
    src/compute/gemm_dp4a.cu
    src/compute/gemm_grouped.cu
    src/compute/gemm_moe_fused.cu
    src/compute/gemm_moe_fused_tc.cu
    src/compute/gemm_q6k.cu
    src/compute/attention.cu
    src/compute/attention_paged.cu
    src/compute/attention_paged_fp8.cu
    src/compute/attention_paged_int8.cu
    src/compute/attention_paged_int4.cu
    src/compute/attention_tc.cu
    src/compute/attention_blackwell.cu
    src/compute/attention_cublas.cu
    src/compute/attention_dispatch.cu
    src/compute/rope.cu
    src/compute/layernorm.cu
    src/compute/activation.cu
    src/compute/embedding.cu
    src/compute/sampling.cu
    src/compute/softmax.cu
    src/compute/json_constrain.cu
    src/compute/json_schema.cpp
    src/compute/schema_constrain.cu
    src/compute/moe_routing.cu
    src/compute/reduce.cu
    src/compute/ssm.cu
    src/compute/gemm_cublaslt_nvfp4.cu
    src/compute/hadamard.cu
)

if(IMP_USE_CUTLASS)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass_mxfp4_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_cutlass_fmha.cu)
endif()

set(IMP_RUNTIME_SOURCES
    src/runtime/green_ctx.cu
    src/runtime/scheduler.cpp
    src/runtime/request.cpp
    src/runtime/batch.cpp
    src/runtime/engine.cpp
    src/runtime/cuda_graph.cu
    src/runtime/pdl.cu
    src/runtime/speculative.cpp
    src/runtime/self_speculative.cpp
    src/runtime/ngram_spec.cpp
    src/runtime/presets.cpp
)

set(IMP_GRAPH_SOURCES
    src/graph/op.cpp
    src/graph/graph.cpp
    src/graph/executor.cu
    src/graph/executor_kernels.cu
    src/graph/executor_workspace.cu
    src/graph/executor_forward.cu
    src/graph/expert_cache.cu
)

set(IMP_VISION_SOURCES
    src/vision/vision_model.cpp
    src/vision/vision_loader.cpp
    src/vision/image_processor.cpp
    src/vision/vision_encoder.cu
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
    ${IMP_VISION_SOURCES}
    ${IMP_API_SOURCES}
)

target_include_directories(imp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
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

if(IMP_USE_CUTLASS)
    target_include_directories(imp PRIVATE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
        ${cutlass_SOURCE_DIR}/examples/88_hopper_fmha
    )
    target_compile_definitions(imp PRIVATE IMP_USE_CUTLASS=1)
endif()

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
        set_target_properties(imp-bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
endif()

# --- Server ---
if(IMP_BUILD_SERVER)
    FetchContent_Declare(httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.18.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(httplib)

    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)

    add_executable(imp-server
        tools/imp-server/main.cpp
        tools/imp-server/args.cpp
        tools/imp-server/utils.cpp
        tools/imp-server/tool_call.cpp
        tools/imp-server/handlers.cpp
        tools/imp-server/batching_engine.cpp
    )
    target_link_libraries(imp-server PRIVATE imp httplib::httplib nlohmann_json::nlohmann_json)
    target_include_directories(imp-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
        tests/test_fp8_kv_cache.cu
        tests/test_speculative.cpp
        tests/test_sampling.cu
        tests/test_activation.cu
        tests/test_embedding.cu
        tests/test_gemm.cu
        tests/test_paged_attention.cu
        tests/test_reduce.cu
        tests/test_tokenizer.cpp
        tests/test_chat_template.cpp
        tests/test_hadamard.cu
    )

    add_executable(imp-tests ${IMP_TEST_SOURCES})
    target_link_libraries(imp-tests PRIVATE imp GTest::gtest GTest::gtest_main)
    target_include_directories(imp-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(imp-tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    include(GoogleTest)
    gtest_discover_tests(imp-tests)
endif()

# --- Install ---
include(GNUInstallDirs)

install(TARGETS imp
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/imp/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/imp
    FILES_MATCHING PATTERN "*.h"
)

if(IMP_BUILD_TOOLS)
    install(TARGETS imp-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(IMP_BUILD_BENCH)
        install(TARGETS imp-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()

if(IMP_BUILD_SERVER)
    install(TARGETS imp-server RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

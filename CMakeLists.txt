cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(imp LANGUAGES C CXX CUDA VERSION 0.18.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake's NVIDIA-CUDA module (through 4.3.x) tops out at c++20 — it defines no
# CUDA23 dialect flag, so `CMAKE_CUDA_STANDARD 23` fails with "CMake does not know
# the flags to enable it". nvcc 13.3 does accept `-std=c++23`, so teach CMake the
# flag directly. Drop this once the module ships a native CUDA23 mapping.
if(NOT DEFINED CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION  "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerFlags)
include(imp-deps)  # third-party dependency pins (single source; mirrored by Dockerfile)

# First-party warning flags as an INTERFACE target. Linked PRIVATE by every
# imp target (lib, tools, server, tests) so warnings apply to OUR host C++ TUs
# only — never nvcc, never FetchContent deps. (Was a global CMAKE_CXX_FLAGS.)
add_library(imp_warnings INTERFACE)
target_compile_options(imp_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

# --- CUDA Toolkit ---
find_package(CUDAToolkit 13.2 REQUIRED)
message(STATUS "CUDA ${CUDAToolkit_VERSION} detected")

# --- CUDA Architecture ---
# Primary target: sm_120a (RTX 5090, RTX PRO 6000 — GB202).
# 'a' suffix = architecture-specific (FP8 MMA .kind::f8f6f4, NVFP4 block-scaled
# MMA, TMA warp-specialized grouped GEMM tactics).
#
# Fallback PTX: compute_120f (family-portable). JIT-compiles on load for GB203
# (RTX 5080, 5070 Ti). Loses TMA-WS grouped GEMM tactics → CUTLASS NVFP4
# fast-path on Mamba2 shapes is degraded. Set IMP_DISABLE_120F_FALLBACK=ON to
# strip the PTX entry and shrink the fatbin.
#
# CMake < 3.31 doesn't parse the 'a'/'f' suffix in CMAKE_CUDA_ARCHITECTURES,
# so we pass gencode directly.
option(IMP_DISABLE_120F_FALLBACK "Skip compute_120f PTX fallback (RTX 5090-only fatbin)" OFF)
set(IMP_SM120_FLAGS "--generate-code=arch=compute_120a,code=sm_120a")
if(NOT IMP_DISABLE_120F_FALLBACK)
    set(IMP_SM120_FLAGS "${IMP_SM120_FLAGS} --generate-code=arch=compute_120f,code=compute_120f")
    message(STATUS "CUDA architectures: sm_120a (SASS, RTX 5090 / RTX PRO 6000) + compute_120f (PTX, RTX 5080 / 5070 Ti)")
else()
    message(STATUS "CUDA architectures: sm_120a (RTX 5090 only — fallback disabled)")
endif()
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${IMP_SM120_FLAGS}")

# --- Options ---
option(IMP_BUILD_TESTS "Build tests" ON)
option(IMP_BUILD_TOOLS "Build CLI and benchmark tools" ON)
option(IMP_BUILD_BENCH "Build benchmarks" ON)
option(IMP_BUILD_SERVER "Build imp-server (OpenAI-compatible HTTP server)" ON)
option(IMP_SANITIZERS "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

if(IMP_SANITIZERS)
    set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    # Note: CUDA code is NOT sanitized (nvcc doesn't support ASAN).
    # ASAN catches host-side memory errors in the C++ runtime/API layer.
    message(STATUS "Sanitizers enabled: ASAN + UBSAN (host code only)")
endif()

# --- Third-party dependencies ---
include(FetchContent)

if(IMP_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG ${IMP_DEP_GOOGLETEST_TAG}
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

option(IMP_USE_CUTLASS "Use CUTLASS for MoE GEMM (requires sm_90+)" ON)
if(IMP_USE_CUTLASS)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG ${IMP_DEP_CUTLASS_TAG}
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
    src/core/qtype.cpp
    src/core/buffer.cpp
    src/core/allocator.cpp
    src/core/logging.cpp
    src/core/threading.cpp
)

set(IMP_MEMORY_SOURCES
    src/memory/vram_allocator.cu
    src/memory/mem_account.cu
    src/memory/kv_cache.cu
    src/memory/kv_cache_manager.cpp
    src/memory/recurrent_snapshot_store.cpp
    src/memory/vram_query.cpp
    src/memory/ssm_state.cu
    src/memory/layer_offload.cu
)

set(IMP_MODEL_SOURCES
    src/model/model.cpp
    src/model/model_profile.cpp
    src/model/gguf_loader.cpp
    src/model/gguf_parse.cpp
    src/model/gguf_tensor_assign.cpp
    src/model/safetensors_loader.cpp
    src/model/llm_compressor_loader.cpp
    src/model/sentencepiece_loader.cpp
    src/model/json_util.cpp
    src/model/hf_config_loader.cpp
    src/model/hf_hub.cpp
    src/model/weight_map.cpp
    src/model/weight_upload.cu
    src/model/tokenizer.cpp
    src/model/tensor_kind_name.cpp
    src/model/tensor_kind_table.cu
    src/model/tensor_kind_matcher.cpp
    src/model/chat_template.cpp
    src/model/chat_template_families.cpp
    src/model/jinja.cpp
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
    src/quant/nvfp4_gemv_dense.cu
    src/quant/nvfp4_gemv_fused.cu
    src/quant/nvfp4_gemv_moe.cu
    src/quant/mxfp4_gemm.cu
    # src/quant/turboquant.cu removed (TurboQuant retired Phase 5, 2026-05-17)
    src/quant/dequant_gptq.cu
)

set(IMP_COMPUTE_SOURCES
    src/compute/gemm.cu
    src/compute/gemm_capture_fp16_sm120.cu
    src/compute/ffn_sparsity_probe.cu
    src/compute/ffn_sparsity_mask.cu
    src/compute/weight_dispatch.cu
    src/compute/gemm_dp4a.cu
    src/compute/gemm_grouped.cu
    src/compute/gemv_ggml_compat.cu
    src/compute/ggml_mmvq.cu
    src/compute/gemm_moe_fused.cu
    src/compute/gemm_moe_fused_tc.cu
    src/compute/gemm_q6k.cu
    src/compute/gemm_q4k.cu
    src/compute/kv_gather.cu
    src/compute/attention_paged.cu
    src/compute/attention_paged_fp8.cu
    src/compute/attention_paged_fp8_tile.cu
    src/compute/attention_paged_int8.cu
    src/compute/attention_paged_int4.cu
    src/compute/attention_paged_nvfp4.cu
    src/compute/attention_paged_nvfp4_tc.cu
    # src/compute/attention_paged_turboquant.cu removed (TurboQuant retired Phase 5, 2026-05-17)
    src/compute/attention_blackwell.cu
    src/compute/attention_cublas.cu
    src/compute/attention_dispatch.cu
    src/compute/rope.cu
    src/compute/layernorm.cu
    src/compute/mla_kv_assemble.cu
    src/compute/mla_absorb.cu
    src/compute/activation.cu
    src/compute/embedding.cu
    src/compute/sampling.cu
    src/compute/sampling_topk_topp.cu
    src/compute/sampling_penalties.cu
    src/compute/sampling_filters.cu
    src/compute/gdn_scan.cu
    src/compute/gdn_scan_tc.cu
    src/compute/gemm_gemv_dtype.cu
    src/compute/gemm_moe_gemv.cu
    src/compute/gemm_batched.cu
    src/compute/mmq_q8_imma_q4k.cu
    src/compute/mmq_q8_imma_q6k.cu
    src/compute/mmq_q8_imma_q51.cu
    src/compute/moe_routing_permute.cu
    src/compute/softmax.cu
    src/compute/json_constrain.cu
    src/compute/json_schema.cpp
    src/compute/schema_constrain.cu
    src/compute/moe_routing.cu
    src/compute/reduce.cu
    src/compute/ssm.cu
    src/compute/gdn.cu
    src/compute/hadamard.cu
    src/compute/mmq_q4k_imma_layout.cu
    src/compute/mmq_q4k_imma_tile.cu
    src/compute/mmq_q8_imma.cu
    src/compute/mmq_q4k_hmma.cu
)

if(IMP_USE_CUTLASS)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass_mxfp4_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_cutlass_grouped_3x.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/gemm_grouped_nvfp4_smallM.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/quantize_fp16_nvfp4_moe_native.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_mxfp4_sm120.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_mxfp4_prefill.cu)
    list(APPEND IMP_COMPUTE_SOURCES src/compute/nvfp4_quant_hw.cu)
    # Bench/probe TUs were relocated to tests/bench/ (P3 §10 #12, P5 §3).
    # Validate/ref TUs that still serve internal call sites stay in src/compute/.
    if(IMP_BUILD_TESTS OR IMP_BUILD_BENCH)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/attention_mxf4nvf4_probe.cu)
        list(APPEND IMP_COMPUTE_SOURCES src/compute/nvfp4_quant_ref.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/mxf4nvf4_mma_bench.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/mma_peak_saturated.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/mxf4nvf4_mma_variants_bench.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/fp4_pv_bench.cu)
        list(APPEND IMP_COMPUTE_SOURCES src/compute/mxf4nvf4_qkt_validate.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/tma_block_scale_bench.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/fmha_v_load_bench.cu)
        list(APPEND IMP_COMPUTE_SOURCES tests/bench/mmq_q4k_imma_bench.cu)
    endif()
    if(IMP_BUILD_TESTS)
        set_source_files_properties(src/compute/gemm_grouped_nvfp4_smallM.cu
            PROPERTIES COMPILE_DEFINITIONS "SMALLM_TEST_HOOKS=1")
    endif()
endif()

# Note: TurboQuant paged attention previously required a fast-math override
# (ptxas C7600 on sm_120a). This is resolved by using sm_120f instead.

set(IMP_RUNTIME_SOURCES
    src/runtime/config.cpp
    src/runtime/process_diag.cpp
    src/runtime/green_ctx.cu
    src/runtime/scheduler.cpp
    src/runtime/request.cpp
    src/runtime/batch.cpp
    src/runtime/engine.cpp
    src/runtime/engine_init_resolver.cpp
    src/runtime/engine_weight_upload.cpp
    src/runtime/engine_kv_cache_init.cpp
    src/runtime/engine_workspace_warmup.cpp
    src/runtime/engine_graph_decode.cpp
    src/runtime/engine_scheduler.cpp
    src/runtime/engine_spec_ngram.cpp
    src/runtime/engine_spec_capture.cpp
    src/runtime/engine_spec_mtp.cpp
    src/runtime/engine_sampling_stop.cpp
    src/runtime/ngram_draft.cpp
    src/runtime/suffix_draft.cpp
    src/vision/vision_pipeline.cpp
    src/runtime/constraint_manager.cpp
    src/runtime/vram_budget.cpp
    src/runtime/cuda_graph.cu
    src/runtime/pdl.cu
    src/runtime/presets.cpp
    src/runtime/storage_planner.cpp
    src/compute/mtp_forward.cu
    src/compute/encoder_forward.cu
)

set(IMP_EXEC_SOURCES
    src/exec/executor.cu
    src/exec/executor_kernels.cu
    src/exec/executor_kv_write_kernels.cu
    src/exec/executor_elementwise.cu
    src/exec/pre_dequant_phase3_fp8.cu
    src/exec/pre_dequant_phase3_cutlass.cu
    src/exec/pre_dequant_phase3_moe.cu
    src/exec/executor_gemm_dispatch.cu
    src/exec/gemm_kernel_registry.cu
    src/exec/gemm_kernel_fp8.cu
    src/exec/gemm_kernel_nvfp4_gemv.cu
    src/exec/gemm_kernel_nvfp4_gemm.cu
    src/exec/gemm_kernel_cutlass_nvfp4.cu
    src/exec/gemm_kernel_mxfp4.cu
    src/exec/gemm_kernel_gguf.cu
    src/exec/gemm_kernel_generic_dequant.cu
    src/exec/gemm_kernel_q4k_imma.cu
    src/exec/gemm_kernel_q4k_hmma.cu
    src/exec/gemm_scratch.cu
    src/exec/executor_workspace.cu
    src/exec/executor_workspace_buffers.cu
    src/exec/executor_kv_write.cu
    src/exec/executor_pre_dequant.cu
    src/exec/pre_dequant_phase0_nvfp4_loader.cu
    src/exec/pre_dequant_phase1_fp16_cache.cu
    src/exec/pre_dequant_phase2_fp8_cache.cu
    src/exec/pre_dequant_phase3_nvfp4_decode.cu
    src/exec/pre_dequant_phase3c_mxfp4.cu
    src/exec/pre_dequant_phase4_tensor_registry.cu
    src/exec/executor_workspace_config.cu
    src/exec/executor_attention.cu
    src/exec/executor_forward.cu
    src/exec/executor_perplexity.cu
    src/exec/executor_lora.cu
    src/lora/lora_adapter.cpp
    src/quant/gpt_oss_mxfp4_convert.cu
    src/exec/executor_ffn.cu
    src/exec/executor_ssm_gdn.cu
    src/exec/executor_forward_moe.cu
    src/exec/executor_forward_moe_cutlass.cu
    src/exec/executor_forward_moe_batch.cu
    src/exec/executor_forward_moe_legacy.cu
    src/exec/expert_cache.cu
    src/exec/weight_handle.cu
    src/exec/moe_workspace.cu
    src/exec/quant_scratch.cu
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
    ${IMP_EXEC_SOURCES}
    ${IMP_VISION_SOURCES}
    ${IMP_API_SOURCES}
)

target_include_directories(imp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests  # for tests/bench/ headers (test/bench TUs only)
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Single-source the runtime version string from the project version so
# imp_version() / `imp-cli --version` can never drift from CMakeLists (#760).
target_compile_definitions(imp PRIVATE IMP_VERSION_STRING="${PROJECT_VERSION}")

target_link_libraries(imp
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    PRIVATE
        pthread
        imp_warnings
)

if(IMP_USE_CUTLASS)
    target_include_directories(imp PRIVATE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
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
    target_link_libraries(imp-cli PRIVATE imp imp_warnings)
    target_include_directories(imp-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # imp-bench
    if(IMP_BUILD_BENCH)
        add_executable(imp-bench
            tools/imp-bench/main.cpp
            tools/imp-bench/bench_gemm.cu
            tools/imp-bench/bench_attention.cu
            tools/imp-bench/bench_e2e.cpp
        )
        target_link_libraries(imp-bench PRIVATE imp imp_warnings)
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
        GIT_TAG ${IMP_DEP_HTTPLIB_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(httplib)

    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG ${IMP_DEP_NLOHMANN_JSON_TAG}
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)

    add_executable(imp-server
        tools/imp-server/main.cpp
        tools/imp-server/args.cpp
        tools/imp-server/utils.cpp
        tools/imp-server/tool_call.cpp
        tools/imp-server/anthropic.cpp
        tools/imp-server/responses.cpp
        tools/imp-server/handlers.cpp
        tools/imp-server/handlers_chat.cpp
        tools/imp-server/handlers_chat_core.cpp
        tools/imp-server/handlers_chat_stream.cpp
        tools/imp-server/stream_driver.cpp
        tools/imp-server/handlers_messages.cpp
        tools/imp-server/handlers_responses.cpp
        tools/imp-server/handlers_misc.cpp
        tools/imp-server/batching_engine.cpp
    )
    target_link_libraries(imp-server PRIVATE imp httplib::httplib nlohmann_json::nlohmann_json imp_warnings)
    target_include_directories(imp-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # Optional gcov instrumentation of the server TUs only (the CUDA imp lib is
    # left uninstrumented — nvcc can't take --coverage). Measures real line
    # coverage of tools/imp-server/ over an E2E run. Off by default.
    option(IMP_COVERAGE "Instrument imp-server with gcov" OFF)
    if(IMP_COVERAGE)
        target_compile_options(imp-server PRIVATE --coverage -O0 -g)
        target_link_options(imp-server PRIVATE --coverage)
    endif()
endif()

# --- Tests ---
# Tests are split into per-module binaries so a kernel change only relinks the
# affected binary instead of one monolithic 60-file imp-tests target.
# A wrapper shell script `imp-tests` is generated for backwards-compat with
# Makefile, CI, and smoke_test.sh (forwards args to every binary in sequence;
# gtest_filter mismatches return success, so only matching tests run).
if(IMP_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)

    function(imp_add_test_module target)
        cmake_parse_arguments(ARG "WITH_STUB" "" "SOURCES" ${ARGN})
        set(_sources ${ARG_SOURCES})
        if(ARG_WITH_STUB)
            list(APPEND _sources tests/gguf_stub.cpp)
        endif()
        add_executable(${target} ${_sources})
        target_link_libraries(${target} PRIVATE imp GTest::gtest GTest::gtest_main imp_warnings)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        # NOTE: deliberately NOT gtest_discover_tests() — it would register every
        # test individually IN ADDITION to the unit/gpu/perf label aggregates
        # below, so a plain `ctest` (no label) ran the whole suite twice (R5/#580).
        # The label aggregates are the single ctest registration; discovery would
        # also require a GPU at configure-time (the binaries enumerate CUDA tests).
    endfunction()

    imp_add_test_module(test-core SOURCES
        tests/test_tensor.cpp
        tests/test_tensor_kind_table.cpp
        tests/test_tensor_kind_matcher.cpp
        tests/test_tensor_kind_coverage.cpp
        tests/test_kv_cache.cpp
        tests/test_gguf_loader.cpp
        tests/test_gguf_fault_injection.cpp
        tests/test_gguf_half.cpp
        tests/test_llm_compressor_loader.cpp
        tests/test_hf_config_loader.cpp
        tests/test_safetensors_loader.cpp
        tests/test_sentencepiece_loader.cpp
        tests/test_config.cpp
        tests/test_rope_override.cpp
        tests/test_ngram_draft.cpp
        tests/test_suffix_draft.cpp
        tests/test_routing_decision.cpp
        tests/test_think_stop_logic.cpp
        tests/test_stream_pipeline.cpp
        tests/test_vision_preprocess.cpp
        tests/test_mla.cpp
    )
    if(IMP_BUILD_SERVER)
        # Anthropic /v1/messages transform tests reuse the server's
        # self-contained anthropic.cpp (json-only dependencies).
        # test_sse_stream_utils.cpp covers the SSE chunk envelopes +
        # reasoning/channel splits in tools/imp-server/utils.cpp (issue #557).
        target_sources(test-core PRIVATE
            tools/imp-server/anthropic.cpp
            tools/imp-server/responses.cpp
            tools/imp-server/utils.cpp
            tools/imp-server/tool_call.cpp
            tools/imp-server/args.cpp
            tests/test_anthropic_transform.cpp
            tests/test_responses_transform.cpp
            tests/test_sse_stream_utils.cpp
            tests/test_tool_call.cpp
            tests/test_tool_stream_filter.cpp
            tests/test_reasoning_reconcile.cpp
            tests/test_server_auth.cpp
            tests/test_server_args.cpp
            tests/test_base64.cpp)
        target_include_directories(test-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/imp-server)
        target_link_libraries(test-core PRIVATE nlohmann_json::nlohmann_json httplib::httplib)
    endif()
    # test_stream_pipeline.cpp includes the self-contained pure header
    # tools/imp-server/stream_pipeline.h (no httplib/json deps).
    target_include_directories(test-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/imp-server)

    imp_add_test_module(test-text SOURCES
        tests/test_tokenizer.cpp
        tests/test_tokenizer_robustness.cpp
        tests/test_tokenizer_compat.cpp
        tests/test_chat_template.cpp
        tests/test_gpt_oss_harmony_golden.cpp
        tests/test_jinja.cpp
    )

    imp_add_test_module(test-compute SOURCES
        tests/test_rope.cu
        tests/test_gpt_oss_yarn_ref.cu
        tests/test_layernorm.cu
        tests/test_mla.cu
        tests/test_activation.cu
        tests/test_embedding.cu
        tests/test_gemm.cu
        tests/test_gemm_capture_fp16_sm120.cu
        tests/test_gemm_dp4a.cu
        tests/test_gemm_q4k_fused_prefill.cu
        tests/test_gemm_kernel_registry.cu
        tests/test_cluster_launch.cu
        tests/test_fp8_gemm.cu
        tests/test_mmvq.cu
        tests/test_reduce.cu
        tests/test_softmax.cu
        tests/test_sampling.cu
        tests/test_executor_kernels.cu
        tests/test_hadamard.cu
        tests/test_mmq_q4k_imma_layout.cu
        tests/test_ffn_sparsity.cu
    )

    imp_add_test_module(test-attention SOURCES
        tests/test_attention_fmha_sm120.cu
        tests/test_attention_crosspath.cu
        tests/test_gpt_oss_sinks_ref.cu
        tests/test_fmha_fp8.cu
        tests/test_attention_mxfp4.cu
        tests/test_attention_fmha_mxfp4.cu
        tests/test_paged_attention.cu
        tests/test_attention_paged_oracle.cu
        tests/test_attention_paged_nvfp4_tc.cu
        tests/test_attention_paged_nvfp4_tc_residual.cu
        tests/test_attention_chunked.cu
    )

    imp_add_test_module(test-quant SOURCES
        tests/test_quant.cu
        tests/test_quant_integration.cu
        tests/test_gguf_dequant_ref.cu
        tests/test_gpt_oss_mxfp4_convert_ref.cu
        tests/test_nvfp4_quant.cu
        tests/test_nvfp4_quant_ref.cu
        tests/test_nvfp4_quant_hw.cu
        tests/test_nvfp4_compressed_tensors_ref.cu
        tests/test_nvfp4_outlier_ref.cu
        tests/test_nvfp4_gemv_kpar_loop.cu
        tests/test_nvfp4_gemm_graph_capture.cu
        # tests/test_turboquant.cu removed (TurboQuant retired Phase 5, 2026-05-17)
        tests/test_cutlass_grouped_3x_nvfp4.cu
        tests/test_cutlass_grouped_ref.cu
        tests/test_cutlass_nvfp4_alpha.cu
        tests/test_gemm_grouped_nvfp4_smallM.cu
        tests/test_quantize_fp16_nvfp4_moe_native.cu
        tests/test_mxf4nvf4_probe.cu
        tests/test_mxf4nvf4_mma_bench.cu
        tests/test_mma_peak_saturated.cu
        tests/test_mxf4nvf4_mma_variants_bench.cu
        tests/test_fp4_pv_bench.cu
        tests/test_mxf4nvf4_qkt_validate.cu
        tests/test_tma_block_scale_bench.cu
        tests/test_fmha_v_load_bench.cu
        tests/test_mmq_q4k_imma_bench.cu
        tests/test_mmq_q4k_imma_tile_bench.cu
        tests/test_mmq_q4k_imma_gemm.cu
        tests/test_mmq_q8_imma.cu
        tests/test_mmq_q4k_hmma.cu
        tests/test_weight_dispatch.cu
    )

    imp_add_test_module(test-kv SOURCES
        tests/test_fp8_kv_cache.cu
        tests/test_kv_cache_write.cu
        tests/test_kv_gather.cu
        tests/test_green_ctx.cu
        tests/test_capture_abort.cu
        tests/test_prefix_cache_equiv.cpp
        tests/test_recurrent_snapshot_store.cpp
        tests/test_vram_query.cpp
    )

    imp_add_test_module(test-moe-gdn SOURCES
        tests/test_moe.cu
        tests/test_moe_executor.cu
        tests/test_expert_cache.cu
        tests/test_gdn.cu
        tests/test_ssm.cu
        tests/test_json_constrain.cu
        tests/test_schema_constrain.cu
    )

    imp_add_test_module(test-e2e WITH_STUB SOURCES
        tests/test_forward_pass.cu
        tests/test_engine_integration.cu
        tests/test_e2e.cpp
        tests/test_e2e_models.cpp
        tests/test_e2e_greedy_lock.cpp
        tests/test_continuous_batching.cpp
        tests/test_degeneration.cpp
        tests/test_quant_pipeline.cpp
        tests/test_weight_registry_preservation.cpp
        tests/test_vram_budget_reserve.cpp
        tests/test_e2e_llm_compressor.cpp
        tests/test_chunked_prefill.cu
        tests/test_mtp_forward.cpp
        tests/test_encoder_embed.cpp
        tests/test_api_generate.cpp
        tests/test_engine_relaunch.cpp
        tests/test_lora.cpp
        tests/test_prefix_cache_e2e.cpp
        tests/test_determinism_e2e.cpp
        tests/test_vision_golden.cu
    )

    set(IMP_TEST_BINARIES
        test-core test-text test-compute test-attention
        test-quant test-kv test-moe-gdn test-e2e
    )

    # --- CTest label aggregates ---
    # ctest -L unit  — CPU-only tests (no GPU required)
    # ctest -L gpu   — GPU tests (require NVIDIA GPU)
    # ctest -L perf  — Performance regression tests
    #
    # Source of truth for the unit/gpu split. The gtest filter only applies to
    # test-e2e (which contains both stub-only unit tests and CUDA-backed E2E
    # tests in the same binary). All other binaries are clean unit-or-GPU.
    #
    # The CPU and GPU tests are interleaved inside test_e2e.cpp / test_continuous_
    # batching.cpp (e.g. StubModelTest has both CPU and GPU subtests, sharing one
    # fixture), so they cannot be split into a separate binary without duplicating
    # fixtures. Instead the split stays a filter, but `guard_e2e_lane_split` below
    # asserts the filter still resolves to the exact expected CPU-only set — a
    # rename can no longer silently shift a test into the wrong lane (R5/#580).
    # NOTE: scripts/verify.sh carries a literal copy of this filter (container
    # path has no build/ dir) — keep both in sync when changing it.
    set(_unit_e2e_filter "BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub")

    add_test(NAME unit_core       COMMAND test-core)
    add_test(NAME unit_text       COMMAND test-text)
    add_test(NAME unit_e2e_subset COMMAND test-e2e "--gtest_filter=${_unit_e2e_filter}")
    add_test(NAME guard_e2e_lane_split
        COMMAND ${CMAKE_COMMAND} -E env
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_e2e_lane_split.sh
                $<TARGET_FILE:test-e2e> "${_unit_e2e_filter}")
    set_tests_properties(unit_core unit_text unit_e2e_subset guard_e2e_lane_split
        PROPERTIES LABELS "unit")

    add_test(NAME gpu_compute   COMMAND test-compute)
    add_test(NAME gpu_attention COMMAND test-attention)
    add_test(NAME gpu_quant     COMMAND test-quant)
    add_test(NAME gpu_kv        COMMAND test-kv)
    add_test(NAME gpu_moe_gdn   COMMAND test-moe-gdn)
    add_test(NAME gpu_e2e       COMMAND test-e2e "--gtest_filter=-${_unit_e2e_filter}")
    set_tests_properties(gpu_compute gpu_attention gpu_quant gpu_kv gpu_moe_gdn gpu_e2e
        PROPERTIES LABELS "gpu")

    set(_perf_filter "*Perf*:*Bench*:*Throughput*")
    add_test(NAME perf_compute   COMMAND test-compute   "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_attention COMMAND test-attention "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_quant     COMMAND test-quant     "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_e2e       COMMAND test-e2e       "--gtest_filter=${_perf_filter}")
    set_tests_properties(perf_compute perf_attention perf_quant perf_e2e
        PROPERTIES LABELS "perf")

    # --- Convenience wrappers for environments without the build tree ---
    # The Docker runtime image only carries the test binaries (no
    # CTestTestfile.cmake), so `ctest -L unit` cannot be used there. These
    # shell wrappers run the same partitions inside the runtime container.
    # On a host build, prefer `ctest -L unit|gpu|perf` over the wrappers.

    # imp-tests: run all GTest binaries sequentially. Matches `ctest` (no label).
    file(WRITE ${CMAKE_BINARY_DIR}/imp-tests
"#!/bin/sh
DIR=$(cd \"$(dirname \"$0\")\" && pwd)
status=0
for bin in test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e; do
    if [ -x \"$DIR/$bin\" ]; then
        \"$DIR/$bin\" \"$@\" || status=$?
    fi
done
exit $status
")

    # imp-tests-unit: CPU-only subset. Matches `ctest -L unit`. Sourced from the
    # same _unit_e2e_filter as the unit_e2e_subset ctest entry above.
    file(WRITE ${CMAKE_BINARY_DIR}/imp-tests-unit
"#!/bin/sh
DIR=$(cd \"$(dirname \"$0\")\" && pwd)
status=0
if [ -x \"$DIR/test-core\" ]; then \"$DIR/test-core\" \"$@\" || status=$?; fi
if [ -x \"$DIR/test-text\" ]; then \"$DIR/test-text\" \"$@\" || status=$?; fi
if [ -x \"$DIR/test-e2e\" ];  then \"$DIR/test-e2e\" '--gtest_filter=${_unit_e2e_filter}' \"$@\" || status=$?; fi
exit $status
")

    foreach(_w imp-tests imp-tests-unit)
        file(CHMOD ${CMAKE_BINARY_DIR}/${_w}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
    endforeach()
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

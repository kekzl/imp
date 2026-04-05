# Performance Tuning — Fastest on Market

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize decode throughput (tok/s) and server throughput across all supported models, targeting #1 performance on single RTX 5090.

**Architecture:** Five sequential optimization phases: (1) Gemma-3 decode bug fix via compute-sanitizer, (2) host-sync-free MoE grouped GEMM, (3) batched GEMV for multi-request decode, (4) MXFP4 dual-path weights (FP8 attention + MXFP4 FFN), (5) PTX 9.2 KV cache quantization intrinsics.

**Tech Stack:** CUDA 13.2, C++20, cuBLASLt, CUTLASS 4.4.2, PTX ISA 9.2, inline PTX asm

---

## Phase 1: Gemma-3 Decode Bug Fix (15 → 129 tok/s)

### Task 1.1: Reproduce and Diagnose with compute-sanitizer

**Files:**
- Debug: `src/compute/attention_paged.cu:1350-1602`
- Debug: `src/graph/executor_attention.cu:505-601`
- Test: `tests/test_paged_attention.cu`

The bug manifests as "invalid argument" CUDA errors at every Gemma-3 decode step, causing 15 tok/s instead of 129. Gemma-3 is unique: `head_dim=256`, `sliding_window_pattern=6`, `attn_logit_softcap=50.0`.

- [ ] **Step 1: Run compute-sanitizer on Gemma-3 decode**

Run inside Docker container with host CUDA toolkit mounted (see `memory/host_cuda_toolkit.md`):

```bash
compute-sanitizer --tool memcheck ./imp-cli \
  --model /models/gemma-3-12b-it-Q8_0.gguf \
  --prompt "Hello world" --max-tokens 16
```

Capture the exact kernel name, grid/block dims, and error location.

- [ ] **Step 2: Narrow to specific kernel + parameters**

Based on compute-sanitizer output, identify which of these paths fails for HD=256:
1. `paged_attention_splitk_pipeline_kernel<256>` — split-K with cp.async (sm_90+ path, line 1423)
2. `paged_attention_gqa_kernel` — GQA kernel (line 1552)
3. `paged_attention_decode_kernel<256>` — MHA fallback (line 1572)

Check these specific suspects:
- **Split-K smem**: `pipe_smem = 8 * 3 * 256 * 2 = 12288` — verify this doesn't exceed `cudaFuncAttributeMaxDynamicSharedMemorySize` for the kernel
- **Split-K scratch sizing**: `executor_workspace.cu:580-599` allocates `max_batch * n_heads * max_splits * (2 + head_dim) * 4` — verify `head_dim` is read correctly from Gemma-3 config
- **GQA kernel smem**: With HD=256, warps_per_q=4, n_q_per_kv=2 (Gemma-3 has 16 heads / 8 KV heads = 2): verify smem fits
- **Block tables stride**: `max_num_blocks` calculation for sliding_window layers

- [ ] **Step 3: Fix the root cause**

Apply the fix based on diagnosis. Common suspects for HD=256:
- Shared memory exceeding per-SM limit without `cudaFuncSetAttribute` call
- Incorrect `max_blocks_per_seq` for sliding window + HD=256 combination
- Register pressure causing launch failure on sm_120

- [ ] **Step 4: Verify the fix**

```bash
# Run tests
./imp-tests --gtest_filter="PagedAttention*"

# Benchmark Gemma-3 decode
./imp-cli --model /models/gemma-3-12b-it-Q8_0.gguf \
  --prompt "Explain quantum computing" --max-tokens 256
```

Expected: ~129 tok/s decode (matching previous baseline), no CUDA errors.

- [ ] **Step 5: Commit**

```bash
git add src/compute/attention_paged.cu
git commit -m "fix: Gemma-3 decode — resolve invalid argument for head_dim=256"
```

---

## Phase 2: Host-Sync-Free MoE Grouped GEMM

### Task 2.1: Implement device-side cublasLtGroupedMatrixLayout

**Files:**
- Modify: `src/compute/gemm_grouped.cu:569-766` (`gemm_moe_device_grouped()`)
- Modify: `src/compute/gemm_cutlass_grouped_sm120.cu:115-131`
- Test: `tests/test_moe.cu`

Currently, all MoE GEMM paths require D2H sync for expert offsets (`cudaStreamSynchronize` at `gemm_grouped.cu:729` and `gemm_cutlass_grouped_sm120.cu:117,129`). CUDA 13.2 cuBLASLt provides `cublasLtGroupedMatrixLayoutCreate` for device-resident problem shapes.

- [ ] **Step 1: Write a test for device-side grouped GEMM**

Add to `tests/test_moe.cu`:

```cpp
TEST(MoETest, DeviceGroupedGEMM_NoHostSync) {
    // Setup: 8 experts, random token routing, device-resident offsets
    const int n_experts = 8, K = 4096, N = 4096;
    std::vector<int> h_tokens_per_expert = {12, 0, 5, 8, 3, 0, 15, 7};  // 50 total

    // Allocate device arrays for M values, pointers, offsets
    int* d_M; cudaMalloc(&d_M, n_experts * sizeof(int));
    cudaMemcpy(d_M, h_tokens_per_expert.data(), n_experts * sizeof(int), cudaMemcpyHostToDevice);

    // ... allocate weight/activation/output buffers per expert ...

    // Call new device-grouped API — must NOT call cudaStreamSynchronize internally
    gemm_moe_device_grouped_v2(d_A_ptrs, d_B_ptrs, d_C_ptrs, d_M, K, N,
                                n_experts, DType::FP16, stream);

    // Verify correctness against reference
    // ...
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./imp-tests --gtest_filter="MoETest.DeviceGroupedGEMM_NoHostSync"
```

Expected: FAIL — `gemm_moe_device_grouped_v2` doesn't exist yet.

- [ ] **Step 3: Implement cublasLtGroupedMatrixLayout path**

In `gemm_grouped.cu`, add a new function that uses the CUDA 13.2 grouped layout API:

```cpp
void gemm_moe_device_grouped_v2(
    const void** d_A_ptrs, const void** d_B_ptrs, void** d_C_ptrs,
    const int* d_M_values,  // device-resident per-expert M
    int K, int N, int n_experts, DType dtype, cudaStream_t stream)
{
    cublasLtHandle_t handle = get_cublaslt_handle();

    // Create grouped matrix layouts with device-side shapes
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    // A: [sum(M_i), K] with per-group row counts from d_M_values
    cublasLtGroupedMatrixLayoutCreate(&layoutA, CUDA_R_16F,
        /* total_rows */ 0,  // computed from d_M_values
        K, K,                // cols, leading dim
        n_experts,
        d_M_values,          // device pointer to per-group row counts
        CUBLASLT_GROUPED_MATRIX_LAYOUT_FORMAT_VARIABLE_M);

    // B: [n_experts, N, K] uniform (same K, N for all experts)
    cublasLtGroupedMatrixLayoutCreate(&layoutB, CUDA_R_16F,
        K, N, K,
        n_experts,
        nullptr,  // uniform B across groups
        CUBLASLT_GROUPED_MATRIX_LAYOUT_FORMAT_UNIFORM);

    // C: [sum(M_i), N]
    cublasLtGroupedMatrixLayoutCreate(&layoutC, CUDA_R_16F,
        0, N, N,
        n_experts,
        d_M_values,
        CUBLASLT_GROUPED_MATRIX_LAYOUT_FORMAT_VARIABLE_M);

    // Matmul descriptor + algorithm selection
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(handle, desc,
                   &alpha, d_A_ptrs, layoutA, d_B_ptrs, layoutB,
                   &beta, d_C_ptrs, layoutC, d_C_ptrs, layoutC,
                   nullptr, nullptr, 0, stream);

    // Cleanup
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(desc);
}
```

**IMPORTANT**: The exact `cublasLtGroupedMatrixLayoutCreate` API may differ from the pseudocode above. Check `cublasLt.h` in CUDA 13.2 headers for the real signature. The dead_ends memory notes that `cuBLASLt grouped layout (sm_120)` returned ZERO algorithms — **this may still be blocked**. If so, fall back to approach 2.1b.

- [ ] **Step 4: If cuBLASLt grouped is still blocked — use CUTLASS 3.x device-side shapes**

The CUTLASS 4.4.2 `GroupedGemmKernel` supports device-resident problem shapes. Replace the CUTLASS 2.x `GemmGrouped` path in `gemm_cutlass_grouped_sm120.cu`:

```cpp
// Replace lines 115-131 (D2H sync) with device-resident CUTLASS 3.x path
// CUTLASS 3.x GroupedGemmKernel reads problem sizes from device memory directly
using GroupedGemm = cutlass::gemm::device::GroupedGemm<...>;
typename GroupedGemm::Arguments args{
    n_problems,
    d_problem_sizes,  // device pointer: int3[n_problems] = {M, N, K}
    d_A_ptrs, d_B_ptrs, d_C_ptrs, d_D_ptrs,
    // ...
};
GroupedGemm gemm_op;
gemm_op.initialize(args, workspace);
gemm_op.run(stream);  // No host sync needed
```

- [ ] **Step 5: Run tests and verify no D2H syncs**

```bash
./imp-tests --gtest_filter="MoE*"

# Profile to verify no cudaStreamSynchronize in MoE path
nsys profile --filter-kernel "moe" ./imp-cli \
  --model /models/qwen3-coder-30b-Q8_0.gguf \
  --prompt "Write a hello world" --max-tokens 32
```

- [ ] **Step 6: Integrate into executor**

Update `executor_forward_moe.cu` to call the new path:
- Lines 689-751 (FP16 batch path): Replace `h_offsets` D2H copy with device-grouped call
- Lines 754-848 (FP8 batch path): Same treatment
- Lines 918-1243 (legacy fallback): Same treatment

- [ ] **Step 7: Commit**

```bash
git add src/compute/gemm_grouped.cu src/compute/gemm_cutlass_grouped_sm120.cu \
        src/graph/executor_forward_moe.cu tests/test_moe.cu
git commit -m "perf: host-sync-free MoE grouped GEMM via device-resident shapes"
```

---

## Phase 3: Batched GEMV for Multi-Request Decode

### Task 3.1: GEMV→GEMM crossover at batch_size > 1

**Files:**
- Modify: `src/graph/executor_kernels.cu:1820-1960` (gemm_dispatch)
- Modify: `src/quant/nvfp4_gemm.cu` (optional batched kernel)
- Test: `tests/test_gemm.cu`

Currently, decode with batch_size=B launches B separate GEMV kernels (one per row). For B>1, a single cuBLAS GEMM call is faster because it amortizes weight loads across all rows.

- [ ] **Step 1: Write benchmark test for GEMV vs GEMM at small M**

Add to `tests/test_gemm.cu`:

```cpp
TEST(GEMMTest, BatchedDecodeGEMV_vs_GEMM) {
    // Compare: M sequential GEMV launches vs 1 GEMM for M=2,4,8
    const int K = 4096, N = 4096;
    for (int M : {1, 2, 4, 8}) {
        // Time M × gemv_nvfp4_kpar
        auto t_gemv = benchmark([&]() {
            for (int i = 0; i < M; i++) {
                gemv_nvfp4_kpar(weight, input_row_i, output_row_i, N, K, stream);
            }
        });

        // Time 1 × gemm_nvfp4 (CUTLASS or dequant path)
        auto t_gemm = benchmark([&]() {
            gemm_nvfp4(weight, input_batch, output_batch, stream);
        });

        printf("M=%d: GEMV=%.1f us, GEMM=%.1f us, winner=%s\n",
               M, t_gemv, t_gemm, t_gemv < t_gemm ? "GEMV" : "GEMM");
    }
}
```

- [ ] **Step 2: Run benchmark to find crossover point**

```bash
./imp-tests --gtest_filter="GEMMTest.BatchedDecodeGEMV_vs_GEMM"
```

Expected: GEMM wins at M≥2 for NVFP4, M≥4 for dp4a (due to cuBLAS setup overhead).

- [ ] **Step 3: Add M-threshold dispatch in gemm_dispatch**

In `executor_kernels.cu`, modify the dispatch logic around line 1863:

```cpp
// Current: always GEMV for M=1
if (input.shape[0] == 1) {
    gemv_nvfp4_kpar(...);
}

// New: GEMV for M=1, GEMM for M>1
int M = static_cast<int>(input.shape[0]);
if (M == 1) {
    gemv_nvfp4_kpar(...);
} else {
    // M>1: single GEMM is faster than M sequential GEMVs
    gemm_nvfp4(it->second, input, output, stream);
}
```

Apply same pattern to:
- MXFP4 dispatch (line 1830): `gemv_mxfp4_kpar` → `gemm_mxfp4` for M>1
- dp4a dispatch (line 1920): `dispatch_dp4a_gemv` → `gemm(cuBLAS)` for M>threshold

- [ ] **Step 4: Handle fused QKV/gate_up for batched decode**

The fused GEMV paths (`gemv_nvfp4_qkv_fused`, `gemv_nvfp4_gate_up_fused`) also need M>1 handling. In `executor_attention.cu` and `executor_ffn.cu`:

```cpp
// Current: fused GEMV for n=1
if (n == 1 && nvfp4_qkv) {
    gemv_nvfp4_qkv_fused(...);
}

// New: fused GEMV for n=1, separate GEMM for n>1
if (n == 1 && nvfp4_qkv) {
    gemv_nvfp4_qkv_fused(...);
} else if (n > 1 && nvfp4_qkv) {
    // Fall through to 3× separate gemm_dispatch for Q, K, V
    gemm_dispatch(no, ly.wq, ly.wq_qtype, q_target, ctx);
    gemm_dispatch(no, ly.wk, ly.wk_qtype, kk, ctx);
    gemm_dispatch(no, ly.wv, ly.wv_qtype, vv, ctx);
}
```

- [ ] **Step 5: Extend CUDA Graph pool for batch>8**

In `engine.cpp`, the graph pool is capped at `kMaxGraphPoolSize = 8`:

```cpp
static constexpr int kMaxGraphPoolSize = 8;
CudaGraphRunner decode_graph_pool_[kMaxGraphPoolSize];
```

For higher batch sizes (continuous batching server), extend to 16 or 32:

```cpp
static constexpr int kMaxGraphPoolSize = 32;
```

Note: Each cached graph uses ~1-2 MB metadata. 32 graphs = ~64 MB — acceptable on 32 GB VRAM.

- [ ] **Step 6: Run server benchmark with concurrent requests**

```bash
# Start server
./imp-server --model /models/qwen3-8b-Q8_0.gguf --port 8080 &

# Benchmark with 1, 2, 4, 8 concurrent requests
for C in 1 2 4 8; do
    echo "=== Concurrent=$C ==="
    seq $C | xargs -P $C -I {} curl -s http://localhost:8080/v1/completions \
      -d '{"prompt":"Hello","max_tokens":128}' | jq .usage
done
```

Expected: Near-linear throughput scaling up to ~4 concurrent requests (weight loads amortized).

- [ ] **Step 7: Commit**

```bash
git add src/graph/executor_kernels.cu src/graph/executor_attention.cu \
        src/graph/executor_ffn.cu src/runtime/engine.cpp tests/test_gemm.cu
git commit -m "perf: GEMM dispatch for batched decode (M>1), extend CUDA graph pool"
```

---

## Phase 4: MXFP4 Dual-Path Weights

### Task 4.1: Implement per-layer-type quantization strategy

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:98-389` (budget allocation)
- Modify: `src/graph/executor_kernels.cu:2003-2026` (GemmContext)
- Modify: `src/model/model_config.h` (layer type classification)
- Test: `tests/test_quant_integration.cu`

The idea: attention projections (WQ, WK, WV, WO) stay at FP8 for quality, FFN weights (gate, up, down) use MXFP4 for 2x bandwidth reduction. Per the MXFP4 memory: "Hybrid: MXFP4 for FFN weights (tolerant), FP8 for attention (sensitive) — the TRT-LLM dual-path approach."

- [ ] **Step 1: Add layer-type classification**

In `model_config.h`, add a helper:

```cpp
enum class WeightRole { ATTENTION, FFN, EMBEDDING, LM_HEAD };

inline WeightRole classify_weight(const std::string& name) {
    if (name.find(".attn.") != std::string::npos ||
        name.find("wq") != std::string::npos ||
        name.find("wk") != std::string::npos ||
        name.find("wv") != std::string::npos ||
        name.find("wo") != std::string::npos)
        return WeightRole::ATTENTION;
    if (name.find("gate") != std::string::npos ||
        name.find("up") != std::string::npos ||
        name.find("down") != std::string::npos ||
        name.find("ffn") != std::string::npos)
        return WeightRole::FFN;
    if (name.find("embed") != std::string::npos)
        return WeightRole::EMBEDDING;
    return WeightRole::LM_HEAD;
}
```

- [ ] **Step 2: Modify budget allocation for dual-path**

In `executor_pre_dequant.cu`, modify the Phase 2 (FP8 cache) and Phase 3 (NVFP4 cache) allocation to be role-aware:

```cpp
// Phase 2: FP8 cache — ONLY for attention weights
for (auto& [ptr, entry] : weights_to_cache) {
    if (entry.role != WeightRole::ATTENTION) continue;
    // ... existing FP8 allocation logic ...
}

// Phase 3: NVFP4/MXFP4 cache — ONLY for FFN weights
for (auto& [ptr, entry] : weights_to_cache) {
    if (entry.role != WeightRole::FFN) continue;
    // ... existing NVFP4 allocation logic ...
}
```

- [ ] **Step 3: Add CLI flag for dual-path mode**

In `imp.h` / `config.h`:

```cpp
struct ImpConfig {
    // ...
    bool dual_path_quant = false;  // FP8 attention + MXFP4 FFN
};
```

Wire through CLI: `--dual-path-quant` in `imp-cli/main.cpp`.

- [ ] **Step 4: Test quality on Qwen3-8B**

```bash
# Baseline (Q8_0)
./imp-cli --model /models/qwen3-8b-Q8_0.gguf \
  --prompt "The capital of France is" --max-tokens 64 --seed 42

# Dual-path
./imp-cli --model /models/qwen3-8b-Q8_0.gguf \
  --prompt "The capital of France is" --max-tokens 64 --seed 42 \
  --dual-path-quant
```

Compare output coherence. FFN layers are known to be more tolerant to 4-bit quantization.

- [ ] **Step 5: Benchmark decode throughput**

```bash
./imp-cli --model /models/qwen3-8b-Q8_0.gguf --bench --dual-path-quant
```

Expected: ~15-25% decode throughput improvement (FFN weights are ~67% of total, at 50% less bandwidth).

- [ ] **Step 6: Commit**

```bash
git add src/graph/executor_pre_dequant.cu src/graph/executor_kernels.cu \
        src/model/model_config.h include/imp/config.h tools/imp-cli/main.cpp \
        tests/test_quant_integration.cu
git commit -m "perf: MXFP4 dual-path — FP8 attention + MXFP4 FFN for lower weight bandwidth"
```

---

## Phase 5: PTX 9.2 KV Cache Quantization

### Task 5.1: Implement packed bf16↔e2m1 conversion via inline PTX

**Files:**
- Modify: `src/quant/nvfp4_quant.cu` (quantize kernel)
- Modify: `src/memory/kv_cache.cu` (KV write path)
- Create: `src/quant/ptx_convert.cuh` (inline PTX wrappers)
- Test: `tests/test_nvfp4_quant.cu`

PTX 9.2 provides `cvt.rn.satfinite.e2m1x2.f16x2` for packed FP16→FP4 conversion (2 elements per instruction). Currently the NVFP4 quantization uses scalar float→FP4 mapping. **Note from dead_ends**: this instruction was REJECTED by ptxas in CUDA 13.2.0 — verify with current toolkit version first.

- [ ] **Step 1: Test PTX instruction availability**

Create `src/quant/ptx_convert.cuh`:

```cpp
#pragma once
#include <cuda_fp16.h>

// PTX 9.2: packed FP16x2 → FP4 E2M1x2 conversion
// Returns: packed byte with two E2M1 nibbles
__device__ __forceinline__ uint8_t cvt_f16x2_to_e2m1x2(half2 val) {
    uint8_t result;
    uint32_t src = *reinterpret_cast<const uint32_t*>(&val);
    asm("cvt.rn.satfinite.e2m1x2.f16x2 %0, %1;" : "=h"(result) : "r"(src));
    return result;
}

// PTX 9.2: packed FP4 E2M1x2 → FP16x2 conversion
__device__ __forceinline__ half2 cvt_e2m1x2_to_f16x2(uint8_t val) {
    uint32_t result;
    asm("cvt.rn.f16x2.e2m1x2 %0, %1;" : "=r"(result) : "h"(val));
    return *reinterpret_cast<const half2*>(&result);
}
```

- [ ] **Step 2: Compile test to check ptxas acceptance**

```bash
# Compile just the header with a minimal kernel
nvcc -arch=sm_120f -ptx src/quant/ptx_convert.cuh -o /dev/null 2>&1
```

If ptxas rejects the instruction (as noted in dead_ends for CUDA 13.2.0), this task is **blocked until CUDA 13.3+**. In that case, skip to the fallback in Step 3b.

- [ ] **Step 3a: If PTX works — integrate into KV cache write path**

Replace the scalar FP16→FP4 quantization loop in `kv_cache.cu` with the packed conversion:

```cpp
// Current: scalar loop
for (int i = 0; i < head_dim; i += 2) {
    uint8_t lo = fp16_to_e2m1(k_ptr[i]);
    uint8_t hi = fp16_to_e2m1(k_ptr[i+1]);
    packed[i/2] = lo | (hi << 4);
}

// New: packed PTX (2x throughput)
for (int i = 0; i < head_dim; i += 2) {
    half2 val = *reinterpret_cast<const half2*>(&k_ptr[i]);
    packed[i/2] = cvt_f16x2_to_e2m1x2(val);
}
```

- [ ] **Step 3b: Fallback — optimize scalar path with prmt**

If PTX cvt is rejected, optimize the existing scalar path:

```cpp
// Use prmt.b32 for parallel nibble packing (same technique as NVFP4 GEMV)
__device__ __forceinline__ uint32_t pack_4_e2m1(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    uint32_t ab = a | (b << 4);
    uint32_t cd = c | (d << 4);
    return ab | (cd << 8);
}
```

- [ ] **Step 4: Benchmark KV cache write throughput**

```bash
./imp-tests --gtest_filter="NVFP4Quant*"
# Compare cycles/element for old vs new path
```

Expected: 1.5-2x speedup on KV cache quantization kernel.

- [ ] **Step 5: Commit**

```bash
git add src/quant/ptx_convert.cuh src/quant/nvfp4_quant.cu src/memory/kv_cache.cu \
        tests/test_nvfp4_quant.cu
git commit -m "perf: PTX 9.2 packed FP16→FP4 conversion for KV cache quantization"
```

---

## Verification Protocol

After all phases, run the full verification suite per CLAUDE.md:

```bash
# 1. All tests pass
cd build && ctest --output-on-failure

# 2. Benchmark all models — no regressions
./imp-cli --model /models/qwen3-4b-Q8_0.gguf --bench     # baseline: tg256=375
./imp-cli --model /models/qwen3-8b-Q8_0.gguf --bench     # baseline: tg256=255
./imp-cli --model /models/qwen35-4b-Q8_0.gguf --bench    # baseline: tg256=308
./imp-cli --model /models/gemma-3-12b-Q8_0.gguf --bench  # baseline: tg256=129 (after fix)

# 3. Real prompts — coherent output
./imp-cli --model /models/qwen3-8b-Q8_0.gguf \
  --prompt "Write a Python function to find prime numbers" --max-tokens 256
./imp-cli --model /models/gemma-3-12b-Q8_0.gguf \
  --prompt "Explain the theory of relativity" --max-tokens 256
```

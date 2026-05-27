# Phase 1 — Long Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the ~4-6k context ceiling imposed by the 1 GiB cuBLAS S-matrix, enabling 32k+ context for agent workloads on NVFP4 KV.

**Architecture:** Add `q_offset` to the existing FMHA kernels (1-line causal mask fix), route chunked prefill through FMHA instead of cuBLAS, auto-derive the cuBLAS→FMHA crossover from S-matrix capacity, and shrink the S-matrix allocation to free VRAM for KV cache. No new kernels — the existing `fmha_sm120_prefill` and `flash_attention_blackwell` already produce correct output and bench at cuBLAS parity (A/B 2026-05-20/22).

**Tech Stack:** CUDA 13.2, sm_120a, C++20, GTest.

**Prior art:** Track E tiled-streaming-softmax plan (`2026-05-21-track-e-tiled-streaming-softmax.md`) was a 10-15 day hand-written FA2 kernel. This plan takes the pragmatic path instead: reuse existing FMHA kernels that are proven correct and at parity perf. Track E remains a future optimization for prefill throughput.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/compute/attention_paged_common.cuh` | Modify | Add `q_offset` param to `apply_score_masks` |
| `src/compute/attention_fmha_sm120.h` | Modify | Add `q_offset` to public API |
| `src/compute/attention_fmha_sm120.cu` | Modify | Thread `q_offset` through kernel launch |
| `src/compute/attention_blackwell.cu` | Modify | Thread `q_offset` through kernel + launcher |
| `src/compute/attention_dispatch.cu` | Modify | Pass `q_offset` through FMHA dispatch chain |
| `src/exec/executor_attention.cu` | Modify | Route chunked prefill through FMHA; pass `q_offset` |
| `src/exec/executor_workspace_buffers.cu` | Modify | Reduce `kMaxAttnScoresMiB` from 1024 to 256 |
| `src/runtime/config.h` | Modify | Default `fmha_prefill_threshold` to -1 (auto) |
| `tests/test_attention_fmha_sm120.cu` | Modify | Add rectangular Q/KV + q_offset tests |
| `tests/test_long_context.cu` | Create | End-to-end 16k+ context correctness test |

---

## Task 1: Add q_offset to apply_score_masks

**Files:**
- Modify: `src/compute/attention_paged_common.cuh:310-334`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_attention_fmha_sm120.cu`:

```cpp
TEST(FMHA_SM120, ChunkedCausalQOffset) {
    // Chunked prefill: Q has 64 rows at position 448 in a 512-token sequence
    // K/V has 512 rows (full context)
    // Causal mask: Q row i (global pos 448+i) can attend to K rows [0..448+i]
    const int seq_q = 64, seq_kv = 512, q_offset = 448;
    const int nh = 8, nkv = 8, hd = 128;

    auto Q = make_random_fp16({1, seq_q, nh, hd});
    auto K = make_random_fp16({1, seq_kv, nkv, hd});
    auto V = make_random_fp16({1, seq_kv, nkv, hd});
    auto O = make_zeros_fp16({1, seq_q, nh, hd});
    auto O_ref = make_zeros_fp16({1, seq_q, nh, hd});

    // Reference: cuBLAS with q_offset
    attention_cublas_prefill(Q, K, V, O_ref, attn_scores_,
                            nh, nkv, hd, 1.0f / sqrtf(hd),
                            /*causal=*/true, /*softcap=*/0.0f,
                            q_offset, stream_, /*sliding_window=*/0);

    // FMHA with q_offset
    bool ok = fmha_sm120_prefill(Q, K, V, O, 1.0f / sqrtf(hd),
                                  /*causal=*/true, /*sliding_window=*/0,
                                  /*softcap=*/0.0f, q_offset, stream_);
    ASSERT_TRUE(ok);
    cudaStreamSynchronize(stream_);

    float max_diff = max_abs_diff(O, O_ref);
    EXPECT_LT(max_diff, 0.05f) << "FMHA q_offset output diverges from cuBLAS reference";
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test-gpu TEST_FILTER="FMHA_SM120.ChunkedCausalQOffset"`
Expected: FAIL — `fmha_sm120_prefill` doesn't accept `q_offset` parameter yet.

- [ ] **Step 3: Add q_offset to apply_score_masks**

Modify `src/compute/attention_paged_common.cuh:310`:

```cpp
__device__ __forceinline__ void apply_score_masks(float* S_tile, int Br, int Bc,
                                                  int block_threads, int tid,
                                                  int q_start, int kv_start,
                                                  int seq_q, int seq_kv,
                                                  float scale, float softcap,
                                                  bool causal, int sliding_window,
                                                  int q_offset = 0) {
    const int total = Br * Bc;
    for (int i = tid; i < total; i += block_threads) {
        int r = i / Bc;
        int c = i % Bc;
        int gq = q_offset + q_start + r;
        int gk = kv_start + c;

        if ((q_start + r) < seq_q && gk < seq_kv) {
            float val = S_tile[i] * scale;
            if (softcap > 0.0f)
                val = apply_softcap(val, softcap);
            if (causal && gq < gk)
                val = -FLT_MAX;
            if (sliding_window > 0 && (gq - gk) >= sliding_window)
                val = -FLT_MAX;
            S_tile[i] = val;
        } else {
            S_tile[i] = -FLT_MAX;
        }
    }
}
```

Key changes:
- Added `int q_offset = 0` parameter (default 0 = backward compatible)
- `gq` now includes `q_offset`: `int gq = q_offset + q_start + r`
- Bounds check uses local `q_start + r < seq_q` (not offset-shifted)
- Causal mask uses global position `gq` vs `gk`

- [ ] **Step 4: Commit**

```bash
git add src/compute/attention_paged_common.cuh tests/test_attention_fmha_sm120.cu
git commit -m "feat: add q_offset param to apply_score_masks for chunked prefill"
```

---

## Task 2: Thread q_offset through FMHA kernel interfaces

**Files:**
- Modify: `src/compute/attention_fmha_sm120.h:23-30`
- Modify: `src/compute/attention_fmha_sm120.cu` (kernel launch + __global__ signature)
- Modify: `src/compute/attention_blackwell.cu` (kernel + launcher)

- [ ] **Step 1: Update public headers**

Modify `src/compute/attention_fmha_sm120.h`:

```cpp
bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream,
                        int q_offset = 0);

bool fmha_sm120_fp8_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset = 0);
```

- [ ] **Step 2: Update FP16 FMHA kernel in attention_fmha_sm120.cu**

Add `int q_offset` to the `__global__` kernel signature. Pass it to `apply_score_masks`:

```cpp
// In the kernel template, at the apply_score_masks call (~line 223):
apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS, tid, q_start, kv_start,
                  seq_q, seq_kv, scale, softcap, causal, sliding_window, q_offset);
```

Update the host launcher to pass `q_offset` through to the kernel.

- [ ] **Step 3: Update flash_attention_blackwell in attention_blackwell.cu**

Same pattern: add `int q_offset = 0` to `flash_attention_blackwell()` signature and thread through to the kernel's `apply_score_masks` call at ~line 203.

Update the header declaration (in `attention_blackwell.h` or wherever it's declared).

- [ ] **Step 4: Update dispatch chain in attention_dispatch.cu**

Modify `attention_prefill_dispatch` (the FMHA fallback entry point) to accept and forward `q_offset`:

```cpp
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                float scale, bool causal, int sliding_window, float softcap,
                                cudaStream_t stream, const RuntimeConfig& rcfg,
                                int q_offset = 0) {
    // ... existing dispatch chain, passing q_offset to each candidate:
    if (fmha_sm120_mxfp4_prefill(..., q_offset)) return;
    if (fmha_sm120_fp8_prefill(..., q_offset)) return;
    if (fmha_sm120_prefill(..., q_offset)) return;
    flash_attention_blackwell(..., q_offset);
}
```

- [ ] **Step 5: Run the test from Task 1**

Run: `make test-gpu TEST_FILTER="FMHA_SM120.ChunkedCausalQOffset"`
Expected: PASS — FMHA with q_offset matches cuBLAS reference.

- [ ] **Step 6: Commit**

```bash
git add src/compute/attention_fmha_sm120.h src/compute/attention_fmha_sm120.cu \
        src/compute/attention_blackwell.cu src/compute/attention_dispatch.cu
git commit -m "feat: thread q_offset through all FMHA kernels and dispatch chain"
```

---

## Task 3: Expanded FMHA correctness tests

**Files:**
- Modify: `tests/test_attention_fmha_sm120.cu`

- [ ] **Step 1: Add test matrix for rectangular Q/KV with q_offset**

```cpp
TEST(FMHA_SM120, ChunkedRectangularSweep) {
    struct Case { int seq_q; int seq_kv; int q_offset; int nh; int nkv; int hd; int sw; };
    std::vector<Case> cases = {
        // chunk at end of sequence
        {64,  512,  448, 8, 8, 128, 0},
        // chunk at middle
        {128, 1024, 256, 8, 8, 128, 0},
        // first chunk (q_offset=0, rectangular because seq_q < seq_kv via padding)
        {512, 512,  0,   8, 8, 128, 0},
        // GQA (4:1)
        {64,  512,  448, 32, 8, 128, 0},
        // HD=256 (Gemma-4 SWA layers)
        {64,  512,  448, 16, 8, 256, 0},
        // With sliding window
        {64,  512,  448, 8, 8, 128, 128},
        // Large context
        {512, 4096, 3584, 8, 8, 128, 0},
    };

    for (auto& c : cases) {
        auto Q = make_random_fp16({1, c.seq_q, c.nh, c.hd});
        auto K = make_random_fp16({1, c.seq_kv, c.nkv, c.hd});
        auto V = make_random_fp16({1, c.seq_kv, c.nkv, c.hd});
        auto O = make_zeros_fp16({1, c.seq_q, c.nh, c.hd});
        auto O_ref = make_zeros_fp16({1, c.seq_q, c.nh, c.hd});

        attention_cublas_prefill(Q, K, V, O_ref, attn_scores_, c.nh, c.nkv, c.hd,
                                1.0f / sqrtf(c.hd), true, 0.0f, c.q_offset, stream_, c.sw);
        bool ok = fmha_sm120_prefill(Q, K, V, O, 1.0f / sqrtf(c.hd),
                                      true, c.sw, 0.0f, stream_, c.q_offset);
        ASSERT_TRUE(ok) << "FMHA unsupported for hd=" << c.hd;
        cudaStreamSynchronize(stream_);

        float max_diff = max_abs_diff(O, O_ref);
        EXPECT_LT(max_diff, 0.1f)
            << "Mismatch: seq_q=" << c.seq_q << " seq_kv=" << c.seq_kv
            << " q_offset=" << c.q_offset << " hd=" << c.hd << " sw=" << c.sw;
    }
}
```

- [ ] **Step 2: Run tests**

Run: `make test-gpu TEST_FILTER="FMHA_SM120.Chunked*"`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_attention_fmha_sm120.cu
git commit -m "test: rectangular Q/KV + q_offset sweep for FMHA chunked prefill"
```

---

## Task 4: Route chunked prefill through FMHA

**Files:**
- Modify: `src/exec/executor_attention.cu:705-823` (chunked prefill path)

This is the core change. Currently, chunked prefill (q_offset > 0) always gathers KV from the paged cache into contiguous buffers and calls cuBLAS. We add an FMHA branch.

- [ ] **Step 1: Write an integration test**

Add to `tests/test_attention_fmha_sm120.cu`:

```cpp
TEST(FMHA_SM120, ChunkedPrefillIntegration) {
    // Verify that the executor_attention chunked path produces identical output
    // when routed through FMHA vs cuBLAS.
    // This test sets fmha_prefill_threshold=1 to force FMHA, then compares
    // against a run with threshold=0 (cuBLAS).
    // Requires a loaded model — skip if unavailable.
    if (!test_model_available("Qwen3-4B-Q8_0")) GTEST_SKIP();

    // ... (model-dependent integration test)
}
```

- [ ] **Step 2: Add FMHA branch to chunked prefill path**

In `src/exec/executor_attention.cu`, after the KV gather (around line 804), replace the unconditional cuBLAS call with a dispatch:

```cpp
// After KV gather into k_full_t, v_full_t (existing code, unchanged):
// ...

// NEW: choose FMHA or cuBLAS for the actual attention computation
const bool chunked_use_fmha =
    !force_cublas_attn &&
    (runtime_config().attention.fmha_prefill_threshold > 0 &&
     ctx_len >= runtime_config().attention.fmha_prefill_threshold);

if (chunked_use_fmha) {
    // FMHA path: no S-matrix needed, O(n) memory
    attention_prefill_dispatch(qv, k_full_t, v_full_t, ao, scale,
                               /*causal=*/true, layer_sliding_window,
                               cfg.attn_logit_softcap, stream,
                               runtime_config(), q_offset);
} else {
    // cuBLAS path (existing): needs S-matrix for [n_chunk × ctx_len]
    attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_,
                            nh, nkv, hd, scale, /*causal=*/true,
                            cfg.attn_logit_softcap, q_offset, stream,
                            layer_sliding_window);
}
```

- [ ] **Step 3: Also add q_offset to the non-chunked FMHA path**

At the existing dispatch (~line 846), pass `q_offset = 0`:

```cpp
} else {
    attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true,
                               layer_sliding_window, cfg.attn_logit_softcap,
                               stream, runtime_config(), /*q_offset=*/0);
}
```

- [ ] **Step 4: Run tests**

Run: `make test-gpu`
Expected: All existing tests pass (q_offset=0 is the default, no behavior change).

- [ ] **Step 5: Commit**

```bash
git add src/exec/executor_attention.cu
git commit -m "feat: route chunked prefill through FMHA when threshold is set"
```

---

## Task 5: Auto-derive fmha_prefill_threshold

**Files:**
- Modify: `src/runtime/config.h:82`
- Modify: `src/exec/executor_workspace_buffers.cu:227`
- Modify: `src/exec/executor_attention.cu:830`

- [ ] **Step 1: Change default to -1 (auto)**

In `src/runtime/config.h`:

```cpp
int fmha_prefill_threshold = -1;  // -1 = auto (derived from S-matrix capacity)
```

- [ ] **Step 2: Compute auto threshold after S-matrix allocation**

In `src/exec/executor_workspace_buffers.cu`, after allocating `attn_scores_buf_`, store the computed `attn_seq` as the auto threshold:

```cpp
// After S-matrix allocation (~line 255):
if (runtime_config().attention.fmha_prefill_threshold == -1) {
    auto& cfg_mut = const_cast<RuntimeConfig::Attention&>(runtime_config().attention);
    cfg_mut.fmha_prefill_threshold = attn_seq > 0 ? attn_seq : 1;
    IMP_LOG_INFO("auto fmha_prefill_threshold = %d (S-matrix cap)", cfg_mut.fmha_prefill_threshold);
}
```

- [ ] **Step 3: Verify the non-chunked dispatch uses the threshold**

The existing code at `executor_attention.cu:830` already checks:

```cpp
const bool prefer_fmha = (n >= runtime_config().attention.fmha_prefill_threshold) &&
                         !force_cublas_attn;
```

With `fmha_prefill_threshold` auto-set to `attn_seq`, any sequence longer than the S-matrix cap auto-routes to FMHA. No change needed here.

- [ ] **Step 4: Run tests**

Run: `make test-gpu && make verify-fast`
Expected: All pass. Short sequences still cuBLAS (threshold = attn_seq ≈ 2896), long sequences now FMHA.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/config.h src/exec/executor_workspace_buffers.cu
git commit -m "feat: auto-derive fmha_prefill_threshold from S-matrix capacity"
```

---

## Task 6: Shrink S-matrix, free VRAM for KV

**Files:**
- Modify: `src/exec/executor_workspace_buffers.cu:227`

- [ ] **Step 1: Reduce kMaxAttnScoresMiB from 1024 to 256**

```cpp
constexpr size_t kMaxAttnScoresMiB = 256;  // was 1024; FMHA handles overflow
```

This frees ~768 MiB VRAM for KV cache. Short sequences (up to ~1448 tokens for 32-head models at 256 MiB) still use cuBLAS. Everything above routes to FMHA via the auto threshold.

- [ ] **Step 2: Run perf baseline to confirm no regression**

Run: `make verify-fast`
Expected: All perf gates pass. The pp512 benchmarks use 512 tokens which fits in 256 MiB for all tested models.

- [ ] **Step 3: Commit**

```bash
git add src/exec/executor_workspace_buffers.cu
git commit -m "perf: shrink S-matrix cap 1024→256 MiB, free 768 MiB for KV cache"
```

---

## Task 7: End-to-end long context test

**Files:**
- Create: `tests/test_long_context.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a long-context correctness test**

```cpp
#include <gtest/gtest.h>
#include "imp/imp.h"

TEST(LongContext, Qwen3_8k_Coherent) {
    // Load Qwen3-4B Q8_0, generate at 8192 context
    // Verify output is coherent (not garbage / NaN / repetition)
    const char* model_path = "models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    if (access(model_path, F_OK) != 0) GTEST_SKIP() << "Model not available";

    ImpModel model;
    ASSERT_EQ(imp_model_load(model_path, IMP_FORMAT_GGUF, &model), IMP_OK);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 16384;
    ImpContext ctx;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_OK);

    // Build a long prompt (~6k tokens): repeat a paragraph
    std::string prompt;
    const char* para = "The quick brown fox jumps over the lazy dog. ";
    while (prompt.size() < 24000) prompt += para;  // ~6k tokens
    prompt += "Summarize everything above in one sentence:";

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;

    char output[4096];
    size_t output_len;
    ASSERT_EQ(imp_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len), IMP_OK);

    // Basic coherence check: output is not empty, not all same token
    EXPECT_GT(output_len, 10) << "Output too short — likely degenerate";
    std::string out_str(output, output_len);
    EXPECT_NE(out_str.find(' '), std::string::npos) << "Output has no spaces — likely garbage";

    imp_context_free(ctx);
    imp_model_free(model);
}

TEST(LongContext, Qwen3_16k_NoOOM) {
    // Same model, 16k context — verify no OOM or crash
    const char* model_path = "models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    if (access(model_path, F_OK) != 0) GTEST_SKIP() << "Model not available";

    ImpModel model;
    ASSERT_EQ(imp_model_load(model_path, IMP_FORMAT_GGUF, &model), IMP_OK);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 16384;
    cfg.kv_dtype = IMP_DTYPE_NVFP4;  // NVFP4 KV for max context
    ImpContext ctx;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_OK);

    std::string prompt;
    const char* para = "The quick brown fox jumps over the lazy dog. ";
    while (prompt.size() < 60000) prompt += para;  // ~16k tokens
    prompt += "What animal was mentioned?";

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;

    char output[2048];
    size_t output_len;
    auto status = imp_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len);
    EXPECT_EQ(status, IMP_OK) << "Long context generation failed";

    if (status == IMP_OK) {
        EXPECT_GT(output_len, 0);
    }

    imp_context_free(ctx);
    imp_model_free(model);
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add `tests/test_long_context.cu` to the test sources.

- [ ] **Step 3: Run tests**

Run: `make test-gpu TEST_FILTER="LongContext.*"`
Expected: Both pass (or skip if model not available).

- [ ] **Step 4: Commit**

```bash
git add tests/test_long_context.cu CMakeLists.txt
git commit -m "test: end-to-end long context tests at 8k and 16k"
```

---

## Task 8: A/B benchmark and perf baseline refresh

**Files:**
- Modify: `tests/perf_baseline.json` (if thresholds need updating)

- [ ] **Step 1: Run decode benchmark (regression check)**

```bash
./build/imp-cli --model models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --max-tokens 128 --bench-reps 5
```

Expected: tg128 ≈ 270 (no regression from S-matrix shrink — decode doesn't use S-matrix).

- [ ] **Step 2: Run prefill benchmark at short context**

```bash
./build/imp-cli --model models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --max-tokens 128 --bench-reps 5
```

Expected: pp512 within ±5% of baseline (still cuBLAS path at 512 tokens).

- [ ] **Step 3: Run prefill benchmark at long context (new capability)**

```bash
./build/imp-cli --model models/Qwen3-4B-Instruct-2507-Q8_0.gguf --bench --bench-pp 4096 --max-tokens 32 --bench-reps 3
./build/imp-cli --model models/Qwen3-4B-Instruct-2507-Q8_0.gguf --bench --bench-pp 8192 --max-tokens 32 --bench-reps 3
```

Expected: pp4096 and pp8192 complete without error. Record numbers as new baseline.

- [ ] **Step 4: Run verify-fast**

```bash
make verify-fast
```

Expected: All gates pass.

- [ ] **Step 5: Refresh perf baseline if needed**

```bash
scripts/gen_perf_baseline.sh
```

- [ ] **Step 6: Commit**

```bash
git add tests/perf_baseline.json
git commit -m "perf: refresh baseline after Phase 1 long-context changes"
```

---

## Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| FMHA regresses short-seq prefill | Low | cuBLAS still handles short seqs via auto threshold |
| Causal mask off-by-one with q_offset | Medium | Task 3 correctness sweep against cuBLAS reference |
| Gemma-4 hd=512 breaks with FMHA | Low | `force_cublas_attn` gate unchanged, Gemma-4 stays cuBLAS |
| FMHA OOMs on very long chunked KV gather | Low | Gather allocates `ctx_len × nkv × hd × 2` bytes — at 32k ctx, 8 kv_heads, hd=128 = 64 MiB |
| Perf baseline gate fails after S-matrix shrink | Low | pp512 fits in 256 MiB for all tested models |

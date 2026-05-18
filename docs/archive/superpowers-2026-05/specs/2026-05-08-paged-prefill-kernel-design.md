# Paged Prefill — Chunked Attention Correctness Fix

**Status**: design accepted, implementation pending
**Roadmap link**: `docs/roadmap.md` known-limitation "Chunked prefill: missing past-KV in attention (paged-prefill kernel pending)" (L2)
**Mitigation in place**: PR #114 (`engine.cpp:1644` default `prefill_chunk_size=0` clamped to executor max_tokens)

## Problem

`src/graph/executor_attention.cu:688+` (`if (state.is_prefill)` branch) computes attention scores from the chunk-local Q/K/V buffers. When chunked prefill is active and `offset > 0`, queries from chunk N are NOT attended against keys/values from chunks `[0..N)` — those past tokens have been written to the paged KV cache by `write_kv_cache` but the prefill attention path never reads them back.

Cross-engine A/B (2026-05-07): same `gemma-4-26B-A4B-it-Q8_0.gguf`, llama.cpp v9049 vs imp at chunk=512: imp drops long-context recall ≥1024 tokens, llama.cpp passes. For full-attention models (Qwen3, Llama) decode partially recovers via paged attention on the first generated token; for Gemma-4's 5:1 SWA:full architecture the propagated hidden states are corrupted enough that decode cannot recover.

The PR #114 mitigation (default `chunk_size=0` → single-chunk) avoids triggering the bug for typical prompts but blocks decode during long single-shot prefills and disables chunked-prefill's multi-tenant decode-latency benefit.

## Scope

In scope:
- Full-attention models (Qwen3, Llama, Mistral) — uniform head_dim, no SWA
- KV cache dtypes FP16 (default) and FP8_E4M3 (`--kv-fp8` opt-in)
- Default `prefill_chunk_size = 512` re-enabled for in-scope models

Out of scope (stay at `prefill_chunk_size=0` via per-arch default):
- Gemma-4 (SWA + dual head_dim 256/512)
- Hybrid models with non-attention layers (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)
- Sub-byte KV cache (INT4, NVFP4, TurboQuant variants)

Each excluded class is a separate larger work item. The fix in this spec is correctness-only for full-attention + FP16/FP8 KV.

## Approach

**Gather → dense FMHA/cuBLAS** (Option B from brainstorming):

1. When `state.is_prefill && state.prefill_offset > 0`, read past chunks' K/V (positions `[0, offset)`) from the paged cache into a contiguous flat buffer, dequanting FP8 → FP16 if applicable.
2. Concatenate the current chunk's K/V (already contiguous in `kk` / `vv`) onto the flat buffer at offset `q_offset`.
3. Call a generalized rectangular-shape `attention_cublas_prefill` with `Q[q_len, nh*hd]`, `K_full[ctx_len, nkv*hd]`, `V_full[ctx_len, nkv*hd]`, `q_offset`, with offset-aware causal mask `K[j]` visible iff `j ≤ q_offset + i`.
4. `write_kv_cache` for the current chunk runs after attention as today.

**Why not a custom paged-prefill kernel** (Option A): correctness-first ships faster; gather bandwidth is negligible vs prefill compute (16 MiB read+write per layer per chunk on Qwen3-4B ctx=8192). Custom FA2-style chunked-Q kernel is a follow-up perf opportunity, not a correctness prerequisite.

## Components

| Component | File | LoC est. |
|---|---|---|
| Gather FP16 paged → flat | `src/compute/kv_gather.cu` (new) | ~60 |
| Gather FP8 → FP16 paged → flat | `src/compute/kv_gather.cu` | ~80 |
| Header | `src/compute/kv_gather.h` (new) | ~20 |
| Generalized `causal_softmax_inplace_kernel(S, q_len, kv_len, q_offset)` | `src/compute/attention_cublas.cu` (refactor) | ~30 (delta) |
| Generalized `attention_cublas_prefill(..., q_offset)` | `src/compute/attention_cublas.cu` (refactor) | ~80 (delta, rectangular GEMM dims) |
| Chunked-prefill dispatch in `run_attention` | `src/graph/executor_attention.cu` (extend) | ~80 |
| `InferenceState::prefill_offset` field | `src/graph/executor.h` | ~3 |
| Per-arch default `prefill_chunk_size` | `src/runtime/engine.cpp` | ~20 |
| Unit tests `test_kv_gather.cu` | `tests/test_kv_gather.cu` (new) | ~150 |
| Unit tests `test_attention_chunked.cu` | `tests/test_attention_chunked.cu` (new) | ~150 |
| E2E logits-equality `test_chunked_prefill.cu` | `tests/test_chunked_prefill.cu` (new) | ~250 |
| `verify-fast` flag pin (`--prefill-chunk-size 0`) | scripts | ~5 |
| Roadmap close + CHANGELOG | `docs/roadmap.md` + `CHANGELOG.md` | ~15 |

## Detailed design

### 1. Gather kernels

Header `src/compute/kv_gather.h`:

```cpp
namespace imp {

// FP16 paged KV → flat FP16. dst layout [n_past, nkv, hd] contiguous.
// src layout [num_blocks, block_size, nkv, hd] paged via block_table.
void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table,
                          int n_past, int block_size, int nkv, int hd, cudaStream_t stream);

// FP8 E4M3 paged → FP16 flat with per-tensor scalar dequant: dst = src * kv_scale.
void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream);

}  // namespace imp
```

Kernel mapping: 2D grid `(n_past, nkv)`, block 128 threads each handling `hd / 128` head_dim elements (vectorized half2). Streaming load via `__ldcs` so KV bytes don't pollute L2 (same hint as `paged_attention_decode`). Edge: last block may be partial — guard `if (pos < n_past)` per thread.

Gather targets the *destination buffer at offset 0*: caller allocates `K_full[ctx_len, nkv*hd]` upfront and gather writes `K_full[0..q_offset, :, :]`. Avoids a temporary `k_past` buffer + memcpy.

### 2. Generalized `causal_softmax`

Existing `causal_softmax_inplace_kernel(half* S, int seq_len, bool causal)` is the special case `q_len == kv_len, q_offset == 0`. Refactor to:

```cpp
__global__ void causal_softmax_inplace_kernel(half* __restrict__ S,
                                               int q_len, int kv_len, int q_offset);
__global__ void causal_softmax_fp32_inplace_kernel(float* __restrict__ S,
                                                    int q_len, int kv_len, int q_offset);
```

Mask: `j > q_offset + row` → `-FLT_MAX`. Grid `(q_len, n_heads)`. Square callers use `q_len=kv_len=seq_len, q_offset=0`. The `bool causal` flag is preserved — `executor_attention.cu:842` (debug `force_cublas_decode` path) calls with `causal=false`. When `causal == false`, the mask is skipped entirely (no `-FLT_MAX` writes); `q_offset` is ignored.

### 3. Generalized `attention_cublas_prefill`

```cpp
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V,
                              Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim,
                              float scale, bool causal, float softcap,
                              int q_offset,           // NEW, default 0 in inline header wrapper
                              cudaStream_t stream);
```

Internal change: `seq_len` (used for both M and N of QK^T and PV GEMMs) splits into `q_len = Q.shape[0]` and `kv_len = K.shape[0]`. cuBLAS handles asymmetric M/N natively; `cublasGemmStridedBatchedEx` calls become:

- QK^T: `M=kv_len, N=q_len, K=hd` (was `seq_len, seq_len, hd`)
- PV: `M=hd, N=q_len, K=kv_len` (was `hd, seq_len, seq_len`)

`strideS = q_len * kv_len`. FP32-vs-FP16 S-buffer heuristic recomputed against `n_heads * q_len * kv_len * 4 ≤ s_buf_fp16_elems * 2`. Softcap kernel iterates `n_heads * q_len * kv_len` elements.

Existing square callers wrap as `attention_cublas_prefill(..., q_offset=0)`.

### 4. Dispatch in `run_attention`

In the `if (state.is_prefill)` branch, before the existing dispatch:

```cpp
const int q_offset = state.prefill_offset;
const bool chunked_prefill = (q_offset > 0);

if (chunked_prefill) {
    KVCache* cache = state.kv_cache;
    QType kvt = cache->qtype();
    // Defense-in-depth: engine-side default already gates these out.
    if ((kvt != QType::F16 && kvt != QType::FP8_E4M3) || sliding_active || per_layer_shapes) {
        IMP_LOG_ERROR("chunked_prefill: unsupported config (kv=%d, swa=%d, per_layer=%d) at L%d",
                      (int)kvt, (int)sliding_active, (int)per_layer_shapes, layer);
        std::abort();
    }

    int kv_layer = get_kv_layer(kv_layer_map_, layer);
    int ctx_len = q_offset + n;
    int kv_bs = cache->block_size();

    half* k_full = nullptr;
    half* v_full = nullptr;
    size_t full_bytes = (size_t)ctx_len * nkv * hd * sizeof(half);
    cudaMallocAsync(&k_full, full_bytes, stream);
    cudaMallocAsync(&v_full, full_bytes, stream);

    // Gather past [0, q_offset) directly into k_full / v_full at offset 0.
    if (kvt == QType::F16) {
        paged_kv_gather_fp16(k_full, (const half*)cache->k_ptr(kv_layer, 0),
                             state.block_tables, q_offset, kv_bs, nkv, hd, stream);
        paged_kv_gather_fp16(v_full, (const half*)cache->v_ptr(kv_layer, 0),
                             state.block_tables, q_offset, kv_bs, nkv, hd, stream);
    } else {  // FP8_E4M3
        // Per-layer FP32 scale lives on the executor (matches paged_attention_decode_fp8 path
        // at executor_attention.cu:928).
        float kv_scale = (!kv_scales_.empty() && kv_layer < (int)kv_scales_.size())
                             ? kv_scales_[kv_layer] : 1.0f;
        paged_kv_gather_fp8_to_fp16(k_full,
                                    (const __nv_fp8_e4m3*)cache->k_ptr(kv_layer, 0),
                                    state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
        paged_kv_gather_fp8_to_fp16(v_full,
                                    (const __nv_fp8_e4m3*)cache->v_ptr(kv_layer, 0),
                                    state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
    }

    // Append current chunk at offset q_offset.
    cudaMemcpyAsync(k_full + (size_t)q_offset * nkv * hd, kk.data,
                    (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_full + (size_t)q_offset * nkv * hd, vv.data,
                    (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    int64_t kv_full_shape[2] = {(int64_t)ctx_len, (int64_t)(nkv * hd)};
    Tensor k_full_t(k_full, QType::F16, 2, kv_full_shape, true);
    Tensor v_full_t(v_full, QType::F16, 2, kv_full_shape, true);

    attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_,
                             nh, nkv, hd, scale, /*causal=*/true,
                             cfg.attn_logit_softcap, q_offset, stream);

    cudaFreeAsync(k_full, stream);
    cudaFreeAsync(v_full, stream);
} else {
    // Existing path unchanged: cuBLAS-vs-FMHA dispatch with sliding_active /
    // gemma4_global_too_long / use_naive_for_swa branches.
}

write_kv_cache(layer, state, stream);
```

### 5. State plumbing

`src/graph/executor.h` line ~137:

```cpp
bool is_prefill = true;
int prefill_offset = 0;  // absolute pos of state.positions[0]; 0 means single-chunk / first chunk
```

`src/runtime/engine.cpp:1838` (in `step_prefill_one`, after `state.is_prefill = true;`):

```cpp
state.prefill_offset = offset;  // already a local `int` in the function
```

### 6. Per-arch default `prefill_chunk_size`

Sentinel: `imp_config.prefill_chunk_size == -1` means "use model-derived default". `0` means "explicit single-chunk". `>0` means "explicit chunk size, validated against constraints".

`src/runtime/engine.cpp` near config init / planning:

```cpp
// Returns true if the (arch, kv_dtype) pair is in the chunked-prefill scope.
bool Engine::supports_chunked_prefill() const {
    const auto& cfg = model_->config();
    if (cfg.arch == ModelArch::GEMMA4) return false;
    if (cfg.arch == ModelArch::QWEN35) return false;       // GDN hybrid
    if (cfg.arch == ModelArch::QWEN35_MOE) return false;   // GDN hybrid
    if (cfg.arch == ModelArch::QWEN36_MOE) return false;   // GDN hybrid
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE) return false;  // Mamba2 hybrid
    KVCache* cache = kv_cache_raw_;
    if (cache && cache->qtype() != QType::F16 && cache->qtype() != QType::FP8_E4M3) return false;
    return true;
}

int Engine::resolve_prefill_chunk_size() const {
    int explicit_val = config_.prefill_chunk_size;
    if (explicit_val < 0) {
        // Sentinel: use per-arch default
        return supports_chunked_prefill() ? 512 : 0;
    }
    if (explicit_val == 0) return 0;  // user pinned single-chunk
    // explicit_val > 0: user wants chunking. If unsupported, refuse + clamp to 0.
    if (!supports_chunked_prefill()) {
        IMP_LOG_WARN("prefill_chunk_size=%d ignored: arch=%d / kv_dtype=%d not in chunked-prefill scope; using 0",
                     explicit_val, (int)model_->config().arch,
                     kv_cache_raw_ ? (int)kv_cache_raw_->qtype() : -1);
        return 0;
    }
    return explicit_val;
}
```

`step_prefill` reads via `resolve_prefill_chunk_size()` instead of `config_.prefill_chunk_size`. Single source of truth for the (arch, KV-dtype) gate; no contradiction between explicit-respect and out-of-scope-clamp because the resolution is centralised.

## Testing

### Unit tests (no model file)

`tests/test_kv_gather.cu`:
- `FP16_PagedToFlat_RoundTrip` — synthesized random FP16 paged → gather → byte-equal vs CPU reference indexing
- `FP8_PagedToFlat_DequantMatchesReference` — FP8 + scale → CPU reference dequant, max-abs-diff ≤ 1e-3 in FP16
- `FP16_PartialLastBlock` — `n_past = block_size + 1` → kernel writes only the 1 valid slot in the last block

`tests/test_attention_chunked.cu`:
- `RectangularEqualsSquareAtZeroOffset` — `q_len=kv_len=128, q_offset=0` → byte-equal vs existing path on identical input
- `OffsetAwareCausalMask` — synthesized one-hot K at position j; verify Q[i] (abs `q_offset + i`) attends to `K[0..q_offset+i]` only
- `GQA_Ratio4` — `nh=16, nkv=4` rectangular path
- `GQA_Ratio8` — `nh=32, nkv=4`
- `FP32_S_Buffer_Path` — large `q_len*kv_len` triggering FP32 S
- `Softcap_Applied` — softcap > 0 produces tanh-bounded scores

### E2E logits-equality (model-dependent, skipped if files absent)

`tests/test_chunked_prefill.cu`:
- `Qwen3_4B_Q8_0_FP16_KV_LogitsEqual` — prompt 2049 tok, chunk ∈ {0, 64, 128, 512, 1024}, all chunked variants must match `chunk=0` last-token logits at max-abs-diff ≤ 1e-2
- `Qwen3_4B_Q8_0_FP8_KV_LogitsEqual` — same with `--kv-fp8`
- `Llama_3_2_3B_Chunk_64_LogitsEqual` — small chunk, non-block-aligned boundary
- `Qwen3_4B_ChunkLargerThanPrompt` — `chunk=4096`, prompt=128 → behaves as single-chunk
- `Qwen3_4B_Chunk0_MatchesPrePR` — pre-PR baseline logits saved as a small JSON; ensure refactor (steps 3–4) doesn't drift the square path
- `Qwen3_4B_GenerationCoherent` — generate 50 tokens chunked vs single, both produce same prefix (greedy, temp=0)
- `Gemma4_DefaultsToZero` — `Engine::resolve_prefill_chunk_size()` returns 0 regardless of explicit config when arch==GEMMA4

### Performance gate

- Existing `tests/perf_baseline.json`: unchanged. `make verify-fast` pinned to `--prefill-chunk-size 0` so baseline stays apples-to-apples.
- New `tests/perf_baseline_chunked.json`: `tg256` and `pp512` for Qwen3-4B Q8_0, Qwen3-8B Q8_0, Llama-3.2-3B Q8_0 with `--prefill-chunk-size 512`. Gate: 5% decode, 8% prefill regression. Looser than main gate to account for the per-chunk gather + rect-attn overhead.

## Rollout (commit sequence)

1. `feat(state): add prefill_offset field to InferenceState`
2. `feat(compute): paged_kv_gather kernels (FP16 + FP8→FP16) with unit tests`
3. `refactor(attention): generalize causal_softmax_inplace_kernel to (q_len, kv_len, q_offset)`
4. `refactor(attention): generalize attention_cublas_prefill with q_offset` — square callers wrapped, tests stay green
5. `feat(graph): chunked-prefill dispatch in run_attention` — wires gather + rect-attn for q_offset > 0
6. `feat(engine): per-arch default prefill_chunk_size with -1 sentinel`
7. `test(prefill): chunked-vs-single logits-equality battery`
8. `bench: pin verify-fast to --prefill-chunk-size 0; add perf_baseline_chunked.json`
9. `docs(roadmap): close L2 paged-prefill, document chunked default + scope`

Each commit independently buildable + test-green. Steps 3–4 are byte-equivalence refactors. Step 5 is the only correctness-changing commit.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Refactor (3–4) breaks square path silently | Low | Existing test suite covers square; perf-baseline gate; byte-equivalence test in unit suite |
| Regression on `pp512` for in-scope models | Medium | New chunked-baseline file with 5%/8% gate; if violated, defer default-512 flip |
| FP8 dequant during gather drifts | Low | Unit test vs CPU reference at 1e-3 |
| `cudaMallocAsync` pool fragmentation | Low | Pool already pre-warmed by engine; observed to handle per-request alloc fine in PR #114 era |
| User explicitly forces `--prefill-chunk-size 512` for an unsupported arch | Medium | `Engine::resolve_prefill_chunk_size()` clamps to 0 + logs WARN; never reaches `run_attention`'s defensive abort |
| New gather kernels miscount partial last block | Medium | Unit test `FP16_PartialLastBlock` |

## Acceptance criteria

- All existing tests (`make test-gpu`) green
- New unit tests green: `test_kv_gather` + `test_attention_chunked`
- New e2e tests green when model files present: 7 cases in `test_chunked_prefill`
- `make verify-fast` smoke pass with pinned `--prefill-chunk-size 0`
- New `perf_baseline_chunked.json` measurements within 5% decode / 8% prefill of single-chunk baseline
- `docs/roadmap.md` L2 entry moved to CHANGELOG (closed)
- Chunked prefill default `512` active for Qwen3 / Llama / Mistral with FP16 or FP8 KV; `0` for Gemma-4 / hybrid / sub-byte KV

## Out of scope (follow-up work items)

- Custom paged-prefill kernel (FA2-style chunked-Q) for higher prefill throughput on long chunks
- Gemma-4 SWA + dual head_dim chunked prefill
- Sub-byte KV (NVFP4 / INT4) chunked prefill
- Hybrid model (SSM/GDN) chunked prefill
- Multi-tenant decode-latency benchmarks proving the chunked-prefill latency benefit

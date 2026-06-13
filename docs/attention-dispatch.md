# Attention dispatch

Companion doc to [`architecture.md`](architecture.md) — covers exactly which attention kernel runs for each (phase × dtype × layer) combination.

If this doc and the code disagree, the code wins. Source of truth is `src/exec/executor_attention.cu` for the gate, `src/compute/attention_dispatch.cu` for the FMHA chain.

> **Measured coverage (2026-06-07, [`docs/audit/roofline_2026_06_07.md`](audit/roofline_2026_06_07.md)):**
> on hd=128 models (Qwen3 dense/MoE — Q8_0, Q4_K_M, NVFP4) the legacy
> materialized cuBLAS+softmax path is **0.0% of prefill time** at pp512–pp4096
> — FP16-QK FA2 (#525) covers the short range, FA2/FP8-FMHA the long range.
> Only hd≠128 models (gemma-3-12b, hd=256) still hit it: 3.6–6.9% of prefill
> time = 92–99% of their attention time.

## Prefill — gate (post #525: FA2 f16-QK first, cuBLAS as hd≠128 fallback)

The prefill dispatch in `src/exec/executor_attention.cu` (~line 942) tries the
FP16-QK FA2 drop-in before the materialized cuBLAS path:

```cpp
const bool force_cublas_attn = per_layer_shapes || attn_sinks != nullptr;
const bool s_matrix_fits     = attn_scores_buf_ != nullptr &&
                               n <= attn_scores_.shape[1];
const bool prefer_fmha       = !force_cublas_attn &&
                               (n >= attention.fmha_prefill_threshold);

if (s_matrix_fits && !prefer_fmha) {
    if (!force_cublas_attn && try_fa2_fp16qk_prefill(...)) { /* FA2 f16-QK */ }
    else attention_cublas_prefill(...);            // legacy materialized
} else {
    attention_prefill_dispatch(...);               // FMHA chain
}
```

| Condition | Path | Why |
|---|---|---|
| short seq (n < `fmha_prefill_threshold`, auto ≈ S-matrix cap + 1), **hd=128**, `fa2_fp16qk != "never"` | `fmha_sm120_fa2_kernel<…,FP16QK=1>` | FP16-QK FA2 drop-in (#525): cuBLAS-FP16-quality without the materialized S-matrix. |
| short seq, **hd≠128** (gemma hd=256) or `attn_sinks`/per-layer shapes | `attention_cublas_prefill` | Legacy materialized path: cuBLAS QK^T → [nh, n, n] FP16 S-matrix (`attn_scores_mib`, default 384 MiB) → causal softmax → cuBLAS PV. The only remaining production user. |
| long seq (n ≥ threshold) | FMHA chain below | Tiled O(n) memory; no materialized S. |
| chunked continuation (`q_offset > 0`) with cumulative ctx < threshold | `attention_cublas_prefill` | FA2 f16-QK still declines `q_offset > 0` (conservative blanket gate post-#548); the FMHA chain needs ctx ≥ threshold. |

### FMHA chain (`src/compute/attention_dispatch.cu`, host model: `attention_dispatch_decision.h`)

Tried in order, first hit wins:

1. **`fmha_sm120_mxfp4_prefill`** — opt-in (`attention_mxfp4_available()`), hd%32==0
2. **`fmha_sm120_fa2_prefill`** — register-resident FA2 (#477/#478, `fmha_fa2 == "on"` default), **hd==128 only**. f16-QK mode unless the fp8-QK pair is explicitly opted in (`fa2_fp16qk=never` AND `fp8_fmha=on`).
3. **`fmha_sm120_fp8_prefill`** — strictly opt-in (`attention.fp8_fmha == "on"`), hd%32==0. Raw e4m3 Q/K conversion compounds per-layer score error on real activations (#511): teacher-forced PPL gemma-3-12b 16.6→549 / Qwen3-8B 40.5→4506 when it served prefill. Off by default.
4. **`fmha_sm120_prefill`** — FP16 WMMA, hd%16==0. Default server for hd≠128 long prefill (gemma-3 hd=256: PPL-identical to cuBLAS, 15.53 both at n=3441 incl. sliding window).
5. **`flash_attention_blackwell`** — WMMA 128×64 tiles, last tier. Declines hd ∉ {64,96,128,256} and smem-over-limit configs (hd=256 needs ~176 KB at Br=64 vs the 99 KB sm_120 opt-in).
6. **Chain exhausted → `std::runtime_error`** (#654). The old silent fallback to `flash_attention_prefill_tc` swallowed launch failures at hd=256 (smem over limit, unchecked `cudaGetLastError`) and produced garbage logits (teacher-forced PPL ~1e10); tc also lacks `q_offset`, so chunked continuations would mask wrongly even when it launches. Reaching this tier means a config override disabled the FP16 WMMA tier or an unsupported head_dim — both error loudly now.

The previously-archived variants (`attention_fmha_sm120_cluster.cu`, `attention_fmha_mxf4nvf4_sm120.cu`, `attention_naive.cu`) are summarized in [`archive/README.md`](archive/README.md); their full source is in git history.

### Chunked prefill carve-out (default for most archs)

Per-arch default `prefill_chunk_size = 2048` (512 until 2026-06-11; larger chunks halve/quarter per-chunk weight re-reads — NVFP4-MoE pp4096 +77%) for full-attention models (Qwen3, Llama, Mistral), hybrid GDN+MoE / Mamba2+MoE (Qwen3.5/3.6, Nemotron-H), and Gemma-4. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk; the result then hits the dispatch gate above with `q_offset`-aware causal masking. See `src/exec/executor_attention.cu` chunked-prefill branch, `Engine::resolve_prefill_chunk_size_()`, and the "Chunked prefill scope" entry in `docs/roadmap.md`.

## Decode — switch on cache_dtype

The decode dispatch (further down in `executor_attention.cu`) is a single `switch` on the KV cache dtype:

| `cache_dtype` | Kernel | Notes |
|---|---|---|
| FP16 | `paged_attention_decode_fp16` | Default. WMMA 16×16 tiles. |
| FP8 (E4M3) | `paged_attention_decode_fp8` | Per-token activation quant; bit-identical to FP16 within ~0.5% perplexity. |
| INT8 | `paged_attention_decode_int8` | Per-head INT8 scale; rarely chosen. |
| INT4 | `paged_attention_decode_int4` | Symmetric 4-bit + per-head FP16 scale. Long-context quality regression vs FP16. |
| NVFP4 | `paged_attention_decode_nvfp4` or `_nvfp4_tc` | TC variant for SM120 mma.sync; falls back to non-TC for unsupported shapes. |
| MXFP4 KV | `paged_attention_decode_mxfp4_kv` | MXFP4-quantized K/V with UE8M0 block-scale (Phase 3 of TurboQuant/MXFP4-KV slice; see `mxfp4_kv_slice3_findings_2026_05_17`). |

### BitDecoding residual cache

When `kv_cache.bitdecoding_qk` is true AND `kv_cache.dtype = nvfp4`, the newest `kv_cache.bitdecoding_residual_tokens` tokens are kept in a residual FP16 buffer and combined with the quantized older blocks at attention time. Used by NVFP4-decode for higher fidelity on the recent context. See `src/compute/attention_paged_nvfp4_tc.cu`.

## Sliding-window mask

cuBLAS path uses `attention_cublas_prefill`'s `sliding_window` parameter (Gemma-4 SWA layers). FMHA path uses `fmha_sm120_prefill`'s `sliding_window` argument (every FMHA variant accepts it). Naive FP32 SWA path is archived (see [`archive/README.md`](archive/README.md); source in git history).

## Soft-cap (logit cap)

cuBLAS path passes `cfg.attn_logit_softcap` through `attention_cublas_prefill`. FMHA path threads `softcap` through every kernel. Soft-cap is a Gemma-3/Gemma-4 feature; other archs default to 0 (off).

## Known wounds

- **1 GiB cuBLAS S-matrix workspace** (`executor_workspace_buffers.cu`). Caps maximum context length. Phase 5 Track E (tiled streaming softmax) deferred as ~10-15-day perf-sensitive work. Reopens when a regression specifically attributes to the workspace cap.

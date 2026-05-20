# Attention dispatch

Companion doc to [`architecture.md`](architecture.md) — covers exactly which attention kernel runs for each (phase × dtype × layer) combination.

If this doc and the code disagree, the code wins. Source of truth is `src/exec/executor_attention.cu` for the gate, `src/compute/attention_dispatch.cu` for the FMHA chain.

## Prefill — two-branch gate

Post-Phase-2 (PR #344), the prefill dispatch in `src/exec/executor_attention.cu` is a flat two-branch switch:

```cpp
const bool force_cublas_attn = per_layer_shapes;          // Gemma-4 hd=512
const bool s_matrix_fits     = attn_scores_buf_ != nullptr &&
                               n <= attn_scores_.shape[1];
const bool non_gemma4_sliding = !force_cublas_attn && sliding_active;

if (s_matrix_fits && !non_gemma4_sliding) {
    attention_cublas_prefill(...);
} else {
    attention_prefill_dispatch(...);   // FMHA fallback
}
```

| Condition | Path | Why |
|---|---|---|
| S-matrix fits **and** (not non-Gemma-4 sliding) | `attention_cublas_prefill` | Default for typical Qwen3 / Gemma-4 configs. cuBLAS QK^T → materialized [nh, n, n] FP16 S-matrix (capped at ~1 GiB) → causal+sliding softmax → cuBLAS PV. |
| S-matrix doesn't fit (long context past cap) | FMHA fallback chain | Tiled O(n) memory; no materialized S. |
| `non_gemma4_sliding` (non-Gemma-4 model with sliding window) | FMHA fallback chain | cuBLAS's sliding-mask path is Gemma-4-optimized; FMHA is faster for other archs. |
| `force_cublas_attn` (Gemma-4 hd=512 global layers) | `attention_cublas_prefill` always | FMHA OOMs the 100 KiB smem cap at hd=512. cuBLAS handles arbitrary head_dim. |

### FMHA fallback chain (`src/compute/attention_dispatch.cu`)

Tried in order, first hit wins. After Phase 2 archival the chain is:

1. **`fmha_sm120_mxfp4_prefill`** — when `attention.mxfp4` enabled and the kernel accepts the head_dim
2. **`fmha_sm120_fp8_prefill`** — when `attention.fp8_fmha != "never"` (default auto)
3. **`fmha_sm120_prefill`** — FP16, the surviving non-cluster variant
4. **`flash_attention_blackwell`** — WMMA 128×64 tiles as the last resort

The previously-archived variants (`attention_fmha_sm120_cluster.cu`, `attention_fmha_mxf4nvf4_sm120.cu`, `attention_naive.cu`) live in `docs/archive/` with resurrection memos.

### Chunked prefill carve-out (default for most archs)

Per-arch default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral), hybrid GDN+MoE / Mamba2+MoE (Qwen3.5/3.6, Nemotron-H), and Gemma-4. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk; the result then hits the dispatch gate above with `q_offset`-aware causal masking. See `src/exec/executor_attention.cu` chunked-prefill branch, `Engine::resolve_prefill_chunk_size_()`, and the "Chunked prefill scope" entry in `docs/roadmap.md`.

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

cuBLAS path uses `attention_cublas_prefill`'s `sliding_window` parameter (Gemma-4 SWA layers). FMHA path uses `fmha_sm120_prefill`'s `sliding_window` argument (every FMHA variant accepts it). Naive FP32 SWA path is archived (`docs/archive/attention_naive/`).

## Soft-cap (logit cap)

cuBLAS path passes `cfg.attn_logit_softcap` through `attention_cublas_prefill`. FMHA path threads `softcap` through every kernel. Soft-cap is a Gemma-3/Gemma-4 feature; other archs default to 0 (off).

## Known wounds

- **1 GiB cuBLAS S-matrix workspace** (`executor_workspace_buffers.cu`). Caps maximum context length. Phase 5 Track E (tiled streaming softmax) deferred as ~10-15-day perf-sensitive work. Reopens when a regression specifically attributes to the workspace cap.

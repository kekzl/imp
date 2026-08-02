# Attention dispatch

Companion doc to [`architecture.md`](architecture.md) — covers exactly which attention kernel runs for each (phase × dtype × layer) combination.

If this doc and the code disagree, the code wins. Source of truth is `src/exec/executor_attention.cu` for the gate, `src/compute/attention_dispatch.cu` for the FMHA chain.

> **Measured coverage (2026-06-07, [`docs/audit/roofline_2026_06_07.md`](audit/roofline_2026_06_07.md)):**
> on hd=128 models (Qwen3 dense/MoE — Q8_0, Q4_K_M, NVFP4) the legacy
> materialized cuBLAS+softmax path is **0.0% of prefill time** at pp512–pp4096
> — FP16-QK FA2 (#525) covers the short range, FA2/FP8-FMHA the long range.
> Since #930/#932 hd=256 rides the FA2 port too (`attention.fa2_hd256`, default on).
>
> **This does NOT generalise to Gemma-4** (2026-08-02 audit, F-16). Its global
> layers are hd=512 and take `attention_cublas_prefill` — or, when the S-matrix
> overflows, `attention_cublas_prefill_sliced` (#1036) — *by design and by
> measurement*: both beat the SMEM-capped fused hd=512 kernel
> ([`docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md`](audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md)).
> So on an advertised model the "legacy" path is the *default* prefill path for
> half the layer stack, and the 384 MiB `attn_scores` workspace is retained for it.

## Prefill — the gate

**Read the code, not a snippet.** This section used to inline the dispatch
source; the variables it named (`force_cublas_attn`, `s_matrix_fits`,
`prefer_fmha`) no longer all exist, and the quoted logic was wrong about
Gemma-4 for six weeks. Source of truth:
`src/exec/executor_attention_prefill.cu` for the outer gate (two blocks —
chunked and non-chunked) and `src/compute/attention_dispatch.cu` for the FMHA
chain.

The outer gate is decided **per layer**, not per model — that is the part the
old snippet got wrong. A Gemma-4 request takes FA2 on its hd=256 SWA layers and
cuBLAS on its hd=512 global layers, in the same forward pass:

| Condition (per layer) | Path |
|---|---|
| no learned sinks, and `hd == 128` or (`hd == 256` and `attention.fa2_hd256`), and `fa2_fp16qk != "never"` | `try_fa2_fp16qk_prefill` — FP16-QK FA2, O(n) memory, primary path |
| S-matrix fits and below the FMHA threshold | `attention_cublas_prefill` — materialized `[nh, n, ctx]` FP16 S-matrix |
| `hd == 512`, S-matrix too small for the whole chunk | `attention_cublas_prefill_sliced` (#1036) — cuBLAS in workspace-sized q-row slices; 3.4–3.9x faster than the fused hd=512 FMHA at Skv 8k/16k |
| otherwise | `attention_prefill_dispatch` → the FMHA chain below |

Learned sinks (gpt-oss) are pre-gated at `attention_dispatch.cu:45`: they route
straight to the FP16 WMMA FMHA — the only sink-capable tier — and **throw** on
decline rather than falling through to a sink-blind kernel (#992).

Since #1205 the resolved path is also **observable at runtime**: the engine logs
one `Resolved dispatch: attn_prefill=… attn_decode=… moe_prefill=…` line after
the first step that has seen both a prefill and a decode, recorded from inside
the real dispatch rather than predicted.

### FMHA chain (`src/compute/attention_dispatch.cu`, host model: `attention_dispatch_decision.h`)

Tried in order, first hit wins:

1. **`fmha_sm120_mxfp4_prefill`** — opt-in (`attention_mxfp4_available()`), hd%32==0
2. **`fmha_sm120_fa2_prefill`** — register-resident FA2 (#477/#478, `fmha_fa2 == "on"` default), **hd 128 and 256** (hd=256 via `attention.fa2_hd256`, default on since #932; Bq=64/TWOSLOT instance). f16-QK mode unless the fp8-QK pair is explicitly opted in (`fa2_fp16qk=never` AND `fp8_fmha=on`).
3. **`fmha_sm120_fp8_prefill`** — strictly opt-in (`attention.fp8_fmha == "on"`), hd%32==0. Raw e4m3 Q/K conversion compounds per-layer score error on real activations (#511): teacher-forced PPL gemma-3-12b 16.6→549 / Qwen3-8B 40.5→4506 when it served prefill. Off by default.
4. **`fmha_sm120_prefill`** — FP16 WMMA, hd%16==0. Fallback for the configs FA2 declines: hd=256 with `fa2_hd256=false`, FA2-declined chunk continuations (`q_offset > 0`), other head dims (gemma-3 hd=256: PPL-identical to cuBLAS, 15.53 both at n=3441 incl. sliding window).
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

## MLA (Multi-head Latent Attention)

DeepSeek-V2/V3 checkpoints take a different route entirely and were missing from
this doc until the 2026-08-02 audit. `ModelProfile::AttnVariant::MLA` marks them;
the compressed KV latent is expanded by `src/compute/mla_kv_assemble.cu` before
the assembled K/V reach the gate above, so from the dispatch's point of view an
MLA layer looks like a normal attention layer with the assembled shapes. There is
no standard RoPE on the latent path — the YaRN `mscale` ratio bug fixed
2026-07-07 lived here.

## Sliding-window mask

cuBLAS path uses `attention_cublas_prefill`'s `sliding_window` parameter (Gemma-4 SWA layers). FMHA path uses `fmha_sm120_prefill`'s `sliding_window` argument (every FMHA variant accepts it). Naive FP32 SWA path is archived (see [`archive/README.md`](archive/README.md); source in git history).

## Soft-cap (logit cap)

cuBLAS path passes `cfg.attn_logit_softcap` through `attention_cublas_prefill`. FMHA path threads `softcap` through every kernel. Soft-cap is a Gemma-3/Gemma-4 feature; other archs default to 0 (off).

## Known wounds

- **~384 MiB cuBLAS S-matrix workspace** (default `attention.attn_scores_mib`, `executor_workspace_buffers.cu`). Caps maximum context length for the legacy cuBLAS fallback (FA2 is the primary path for hd 128 since #687 and hd 256 since #932; the S-matrix is skipped entirely at init on FA2-served configs). Phase 5 Track E (the six-variant tiled streaming softmax) was **closed**, not deferred — code removed in PR #358. Reopens only if a regression specifically attributes to the workspace cap.

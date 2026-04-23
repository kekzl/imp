# Changelog

All notable changes since v0.6. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] - 2026-04-23

Big correctness + platform release: the long-context dispatch cliff is gone,
Gemma-4 and the Qwen 3.5/3.6 GDN family now produce clean output on Blackwell,
CUDA 13.2.1 with stream priorities and mem-sync domains is live, and the
StreamingLLM smart-KV mode is available.

### Fixed

- **FP8 FMHA long-context cliff at n>1024** (#33) — `fmha_sm120_fp8_kernel` placed
  `S_tile` only `Bkv*head_dim` bytes past `KV_fp8`, but the K-as-FP8 / V-as-half
  slot is reserved for the full `Bkv*head_dim*sizeof(half)` bytes. V row `Bkv/2+`
  overwrote the P values the PV MMA was about to read → NaN on every attention
  layer above n=1024 (cuBLAS dispatch boundary). Invisible to prior benchmarks
  because `pp=512/1024` always stayed on cuBLAS and decode uses paged attention.
  All tested models (Qwen3-4B/8B, Qwen3.5-4B/9B, Llama-3.2-3B, Mistral-24B,
  Qwen3-32B) now coherent at n≥1025.
- **Qwen 3.5/3.6 GDN fused-kernel launch_bounds** (#30) — `__launch_bounds__(HD, 2)`
  miscompiled at HD=128 (register pressure with `H_reg[128]` and 2 blocks/SM).
  Dropping to `(HD, 1)` fixes Qwen3.5-4B/9B Q8_0 coherence and improves
  Qwen 3.6 tg256 from 36 → 57 tok/s.
- **Qwen 3.5 partial-RoPE pair offset** (#30) — sister fix: partial-RoPE pair index
  was `pair_idx + head_dim/2` instead of `pair_idx + rope_pairs`. Both fixes are
  needed for correct output.
- **Qwen 3.6 h_state FP32 preservation** (#28) — engine auto-downgraded
  `ssm_state_dtype` from FP32 to FP16 for all SSM models, but the GDN scan writes
  FP32. Each layer's scan overflowed 1 MB into the next layer's `conv_state` /
  `h_state` region, producing NaN at L38 on Qwen 3.6. Also switched L2 norm to
  PyTorch-style `rsqrtf(fmaxf(sum_sq, 1e-12))` for near-zero-head stability.
- **Gemma-4 SWA long-context degeneration** (#21) — fixed regression where prompts
  >1024 tokens on global layers emitted garbage via the broken FMHA fallback.
- **Gemma-4 rope_freqs** (#20) — per-layer `rope_freqs` were ignored on global layers;
  llama.cpp uses them with `n_rot=hd`. Fix cuts L13/L14 drift from 11-15 % to <2 %.
- **Gemma-4 host-resident MoE** (e879bcd) — fused gate_up split on host, batch buffer
  preserved. Fixes silent output corruption when experts are CPU-offloaded.
- **Gemma-4 Q4_K_M CUDA-graph decode** (873f1d7) — split-K pipeline kernel only issued
  one 16 B `cp.async` per load, missing half the data for HEAD_DIM=512 on global
  layers. Loops `cp_async_ca_16` in 8-half chunks. tg256 Q4_K_M 55 → 183 tok/s
  (×1.21 vs llama.cpp 151). Also +12 % on Qwen3-4B MXFP4.
- **Qwen 3.5 GDN L2-window CUDA errors** (275807c) — `cudaStreamAttributeAccessPolicyWindow`
  with `num_bytes > cudaDevAttrMaxAccessPolicyWindowSize` (128 MiB on RTX 5090)
  silently poisoned the stream. Clamped in `set_l2_streaming` +
  `set_l2_persist_kv`.
- **Gemma-4 ≥3120 token limit was VRAM, not architecture** (2026-04-20) — default
  ceiling lifted 3120 → ~7881 tok; `--min-kv-tokens 14000` reaches 11242 tok. Root
  cause: max_seq_len ordering bug + defensive 80 % cap.
- **Gemma-4 decode FP32 router + half rope_dim on global layers** (5a1e844) —
  MoE routing FP16 accumulation caused expert mis-pick at L29. Also fixed full
  rope rotation being applied instead of the partial-RoPE schedule.
- **Async decode loop correctness** (3b766bc) — four latent bugs in the async
  decode path that only surfaced with real long generations.

### Added

- **StreamingLLM smart KV cache** (#26) — attention sinks + sliding window; keeps
  long-conversation coherence without unbounded VRAM growth.
- **Weight-storage refactor** (#27) — `TensorKind` + `StoragePlanner` +
  `gemm_dispatch` (phases 0-5). Collapses 21-param dispatch to 5 params, legacy
  overload retired, `beta=1` supported. No functional change, -1200 LoC
  churn absorbed cleanly.
- **CUTLASS 3.x NVFP4 Grouped GEMM scaffold** (#22) — path for sm_100+ FP4 grouped
  with fused MoE quantize; default ON for all batch sizes after the gate+up
  shared-quantize opt (decode 51 > legacy 37 on Qwen3-Coder-30B-A3B NVFP4).
- **CUDA 13.2.1 base images** (#16) + **stream priorities, mem-sync domains,
  cluster spread** (#17).
- **Qwen 3.6 `ModelArch::QWEN36_MOE` scaffold** (#23) — GDN + MoE hybrid.
- **GDN reference infrastructure + Qwen 3.6 cache preservation** (#25) — shared
  helpers for GDN debug dumps, multi-turn state preservation.
- **IMP_DEBUG_RAW meta-flag** (#29) — single switch that turns off CUDA graphs,
  PDL, host-MoE, and other sources of non-determinism for reference-diff runs.
- **IMP_EXPERT_OVERHEAD_PCT hint** (#32) — runtime emits the right env-var
  suggestion when it disables CUDA graphs due to insufficient VRAM headroom.
- **IMP_GEMMA4_CUDA_GRAPHS, IMP_FORCE_HOST_EXPERTS=N, IMP_NO_MMVQ,
  IMP_NO_MMVQ_Q8_0** — debug overrides surfaced during the Gemma-4 stabilization.
- **`tools/analysis/layer_diff.py`** (#20) — .npy-based per-layer tensor diff
  between imp and llama.cpp for drift analysis.
- **CUDA graph diagnostics** (#11-#14) — `IMP_GRAPH_DIAG` / `IMP_GRAPH_DUMP`,
  device-side stop-reason trace in `post_decode_step_kernel`,
  `cudaDeviceGraphMemTrim` on capture lifecycle.
- **Regression tests** — Gemma-4 e2e suite (7633e1a), `Gemma4GraphsTest` for
  the CUDA graph path (dd10244), `FmhaFP8Test.Qwen35LikeHD256_GQA41_SeqMultiTile`
  for the long-context fix (#33).
- **cpp-httplib 0.40.0 → 0.42.0** (4295c05).

### Changed

- `imp_version()` and `project(… VERSION)` now return **0.7.0**.
- Gemma-4 CUDA graphs enabled by default for decode fast-path (no more D2H in
  the routing path on that arch).
- Gemma-4 benchmark docs refreshed (#24) with the quality caveat for Q4_K_M on
  complex code-gen prompts — Q5_K_M / Q8_0 recommended when output quality
  matters.

### Deprecated / Removed

- Nothing user-visible. Internal legacy `_inline_quant` GEMV and stale
  `gen_reference` stub removed (e558f10).

### Known Issues

- Qwen3-Coder-30B-A3B NVFP4: `--no-cuda-graphs` still required for coherence on
  the MoE routing path (general-MoE D2H routing memcpy is incompatible with
  capture; Gemma-4 is excepted via the decode fast-path).
- Prefill throughput shows up to 2.6× variance between container restarts due
  to cuBLAS autotuning algorithm selection — compare decode-only for reliable
  A/B testing.
- FP8 FMHA path (n>1024 prefill) is ~30 % slower than cuBLAS at the dispatch
  boundary on small dense models (Qwen3-4B: 27 k → 19 k tok/s at 1024→2048).
  Output is correct; optimization is future work.
- MXFP4 GGUFs use the imp-proprietary tensor-type 31, which llama.cpp reads as
  the removed `Q4_0_4_4`. Cross-tool perplexity comparison is therefore not
  possible without a standard-format MXFP4 export.

---

## [0.6] - 2026-04

Previous release. Highlights that shipped under this tag:

- NVIDIA Model Optimizer NVFP4 prequant SafeTensors loading
  (Qwen3-Coder-30B-A3B-FP4 verified).
- imp-server SafeTensors support + `resolve_model_auto()` format detection.
- Chat-template array format (HuggingFace convention).
- Jinja2 macro support — fixed Qwen 3.5 "ignores prompt" symptom.

## [0.5.1], [0.4.1], [0.4], [0.2]

Pre-0.6 tags retained for reference. See `git log` for details.

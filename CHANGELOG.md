# Changelog

All notable changes since v0.6. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — post-v0.7.0 (2026-04-24 → current)

41 PRs since v0.7.0. Highlights below; full list under each section.

### Fixed

- **NVFP4 prequant MoE decode fast-path** (#85) — Qwen3.6-NVFP4 went 8.34 →
  117–142 tok/s (~14–17×); Gemma-4-NVFP4 went ~42 → 157–180 tok/s (~4×).
  Three bugs: `can_decode_fast` whitelist did not include NVFP4-prequant
  models; `cache_moe_native_nvfp4` had to be added to build the contiguous
  per-expert NVFP4 buffer for SafeTensors per-expert layouts; per-layer
  free of per-expert allocations (32 GiB VRAM ceiling on 35B-A3B).
  Memos: `qwen36_nvfp4_decode_underutil_2026_04_30.md`,
  `gemma4_nvfp4_decode_fastpath_2026_05_01.md`.
- **Six Qwen3.5/3.6-NVFP4 SafeTensors loader bugs** (#81) blocking coherent
  decode (head layout, RMSNorm `1+W` convention sidecars, etc.).
- **Qwen3.5 GDN Q8_0 α/β qtype mismatch** (#59) — `upload_weight` pre-dequanted
  Q8 → FP16 without updating `qtype`. Dispatcher mis-interpreted bytes →
  state collapse (` my my my…`). Memo: `qwen35_q8_alpha_beta_qtype_bug_2026_04_25.md`.
- **MXFP4 GDN-fallback dequant** (#58) — replaced buggy CPU path with GPU kernel.
- **MXFP4 FP16-fallback VRAM oversubscription diagnostic** (#60) — clear error
  message for the Qwen3.5-27B-MXFP4 IMA-on-load case (was silent).
- **Qwen3.5-MXFP4 `A_log` from `blk.X.ssm_dt.weight`** (#61).
- **MoE expert-offload auto-pick** (#54) — defaults try 10 % overhead first
  before falling back to 30 %. Qwen3-Coder-30B Q6_K 77 → 234 tok/s.
  Memo: `moe_expert_offload_fix_2026_04_24.md`.
- **Mistral-3.2-NVFP4 `use_default_system_prompt`** (#78) — honour the
  tokenizer-config flag and skip the 600-token jinja default system prompt.
  "I am the capital of France?" → "Paris". Memo:
  `mistral_3_2_nvfp4_use_default_system_2026_04_28.md`.
- **Server `<channel|>` swallowing answer body on Gemma-4** (#39).
- **Gemma-4 byte-fallback on common names** (#37).
- **Server `reasoning_content` for chat-template-injected `<think>`** (#86).
- **`verify` auto re-execs in `imp:test` when host CMake is missing** (#70) —
  unblocks `make verify-fast` for clean-host workflows.

### Added

- **KV-cache safety default flip** (#51) — default KV dtype is now FP16; FP8
  is opt-in via `--kv-fp8` / `imp.conf:kv_cache.dtype="fp8"`. Fixes Mistral,
  DeepSeek, and Qwen3.5-GDN out of the box on first decode.
- **Auto-deterministic cuBLAS when FP8 KV active** (#52) — pins cuBLAS algo
  selection to avoid quant-dequant noise → softmax NaN. Necessary fix; not
  sufficient for all archs (see TODO.md "FP8 KV stride bug").
- **CUDA Graph coverage expansion** (#53) — speculative-verify graphs, SigLIP
  vision graph, default mem-pool retain, `cudaGraphExecUpdate` re-capture.
  Memo: `cuda_graph_expansion_2026_04_24.md`.
- **SM120 FMHA optimisation pass — Project B Stage 4** (#55, #56) — float4
  tile loads + HW FP4 conversion. **+11–13 % prefill** on Qwen3-4B Q8_0 at
  pp=8192. Stage 5 (`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) layouts
  verified byte-exact, integration is the next open Project B item.
- **NVFP4 SafeTensors loader from llm-compressor** (Phase 1, #63; Phase 2
  Item 1 Mistral3, #64; Phase 2 Item 2 Gemma-4 partial, #65). Mistral3-NVFP4
  decode tg ≈ 81 tok/s post Phase 2 Item 1. Gemma-4-NVFP4 from
  llm-compressor still incoherent (use Model Optimizer NVFP4 instead).
- **Qwen3.6-NVFP4 SafeTensors plumbing** (Phase 1 #71) — load-only.
  Decode lit up later via #85.
- **JSON config plumbing** (#74, #77) — `generation_config.json` sampling
  defaults, `special_tokens_map.json`, Mistral V3 tokenizer-config flags.
- **Tokenizer-config `use_default_system_prompt=false` honoured** (#78) — see
  Fixed.
- **Type-system + config refactor** (#72) — unified `QType`, `Tensor` sidecars,
  `imp.conf` (TOML, ~50 former `IMP_*` env vars now keys). New top-level
  `imp.conf.example`. CLI `--set kv_cache.dtype=fp8` for per-run overrides.
- **NVFP4 collapsed load-time scratch** (#73) — single `Model` map.
- **FP32 attention S-matrix + Qwen3.5 QK-norm split** (#66) — improves
  numerical headroom on long-context attention.
- **Diagnostic env vars for NVFP4 + attention** (#79) — reproducer support
  for the long-context NVFP4 bug.
- **Anthropic `/v1/messages` endpoint** (Phase 1 non-streaming #35,
  Phase 2 streaming #36) — synthetic SSE stream over the OpenAI handler.
- **Storage-planner enumerates shared-expert FFN + top-level embeddings/LM
  head** (#38, #40) — fixes silent miss in MoE memory budget.
- **Strengthened GDN coherence test** (#48) — detects recurrent-state collapse.
- **Strengthened Gemma-4 NVFP4 e2e gate** (#68) — Paris coherence assertion.
- **Synthetic `gemv_kpar` M>1 per-row-loop bug repro test** (#69).
- **Split `imp-tests` into 8 per-module binaries** (#57) — speeds up filtered
  test runs.
- **`tools/analysis/` PTX survey scripts** (#67) — re-runnable cvt / MMA /
  async-TMA / atomics / SFU / cluster surveys for `sm_120f` after CUDA upgrades.

### Changed

- **Default KV dtype is FP16** (#51) — see Added. Was implicit auto-FP8.
- **`imp.conf` is now the configuration interface** (#72) — ~50 `IMP_*` env
  vars retired; sectioned TOML keys (`runtime.cuda_graphs`, `kv_cache.dtype`,
  `attention.fp8_fmha`, …). Loading precedence: `--config` → `$IMP_CONFIG` →
  `./imp.conf` → `~/.config/imp/imp.conf` → embedded defaults. CLI overrides
  via `--set section.key=value`.

### Repository / build hygiene

- **Untracked `build-docker/` and `bringup_artifacts/`** (#82) — debug dumps
  no longer in VCS.
- **Removed tracked binaries + stale Gemma-4 debug snapshots** (#83).
- **Removed obsolete top-level docs + stale benchmarks dir** (#84).

### Performance baseline refresh (2026-04-29)

`tests/perf_baseline.json` refreshed during the autonomous bringup mission
(#80). Drift was visible **before** the bringup branch's two code commits —
i.e. not a regression introduced by the mission. Numbers reflect the
RelWithDebInfo build with full GPU boost engaged (P1, 2880 MHz, 456 W).
Refresh mechanism documented in CLAUDE.md memory:
`scripts/gen_perf_baseline.sh`.

### Known issues (carry-over)

- **FP8 KV cache** still breaks Llama-3.2 / Mistral-Small-3.1 / DeepSeek-R1-Distill
  out of the box even with the determ-cuBLAS gate. Default is FP16; opt-in
  per model after testing. See TODO.md.
- **NVFP4 long-context regression** on Mistral-3.2-NVFP4 at ~500+ raw tokens
  remains open. PR #79 ships diagnostics; PR #78 ships the
  `use_default_system_prompt=false` workaround for the most common trigger.
- **Qwen3-Coder-30B-A3B NVFP4** still requires `--no-cuda-graphs` for
  coherence on the MoE routing path; Gemma-4 + NVFP4-prequant MoE excepted
  via the decode fast-path post #85.
- **Prefill throughput** shows up to 2.6× variance between container restarts
  due to cuBLAS autotuning. Compare decode-only for reliable A/B.

---

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

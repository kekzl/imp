# TODO

Open work and tracked bugs. Completed work lives in [CHANGELOG.md](CHANGELOG.md);
known historical dead ends in `memory/dead_ends.md`.

---

## Open Bugs

### FP8 KV cache stride / numerical instability — root cause unknown
**Workaround live since PR #51**: default KV dtype is FP16; `--kv-fp8` (or
`kv_cache.dtype=fp8` in `imp.conf`) opts in.

PR #52 added an auto-deterministic-cuBLAS gate that fixes the symptom on
**Qwen3** and **Qwen3.5/3.6 GDN** with FP8 KV active. **Empirically retested
2026-05-01 across 6 models with the gate engaged**: Llama-3.2-3B, Mistral-Small-3.1
Q6_K, and DeepSeek-R1-Distill-14B Q6_K still break (degenerate output / illegal
memory access in `sampling.cu:971` / `<think><｜User｜>` collapse). Therefore
the "flip default to FP8" recommendation in older notes is refuted.

Per-arch behaviour at decode with default-FP8 KV (hypothetical):

| Family | Status | Notes |
|---|---|---|
| Qwen3 dense | ✓ coherent | gate sufficient |
| Qwen3.5/3.6 GDN | ✓ coherent | needs `α/β qtype` fix from PR #59 |
| Gemma-4 | force-FP16 carve-out at `engine.cpp:547` | Real fix needs per-layer head_dim awareness in KV write/read |
| Llama-3.2 | ✗ degraded | "is." instead of "is Paris." |
| Mistral-Small-3.1 | ✗ illegal memory access | `sampling.cu:971` |
| DeepSeek-R1-Distill | ✗ degenerate | tg=2, instant `<think><｜User｜>` |

Root cause is suspected to be a stride mismatch in the FP8 KV write/read path
that interacts with specific head_dim / num_kv_heads layouts. The Gemma-4
carve-out at `engine.cpp:547` is the existing escape hatch; generalising it
needs storage-planner work, not a one-line fix.

### NVFP4 long-context regression (Mistral-3.2-NVFP4) — partial fix landed
Originally numerical-hash garbage at ~95+ tokens with Lorem ipsum prefixes,
~130 with `[SYSTEM_PROMPT]` markers, ~250+ with English prose.

**Partial fix shipped via PR #88**: `executor_pre_dequant.cu` now registers
prequant-promoted NVFP4 weights in `wcache_.cutlass_nvfp4`, lighting up the
native CUTLASS NVFP4×NVFP4 prefill path that previously fell through to
`gemm_nvfp4` dequant→cuBLAS. Mistral-3.2-NVFP4 prefill 283 → 3122 tok/s
(11×); Lorem×11 went from `a long established in 1999999999` (numerical
garbage) to `a dolor sit amet, consectetur adipiscing elit, Quis...`
(coherent Latin continuing the prefix). Memos:
`memory/nvfp4_long_context_regression_2026_04_28.md`,
`memory/nvfp4_prequant_cutlass_cache_2026_05_01.md`.

**Still open**: long English prose ≥250 tokens doesn't always reach the
"Paris" answer — the model picks contextually-attracted continuations
instead. Root cause is the SmoothQuant 0.9 + NVFP4 + FP16-activation
mismatch (recipe expects dynamic NVFP4 input act-quant; imp uses FP16).
The CUTLASS NVFP4×NVFP4 path quantizes activations dynamically per-block,
which reduces but doesn't eliminate the noise. Final fix would
implement the recipe-intended path: load the per-Linear `input_scale`
from the SafeTensors and use it for static activation NVFP4 quant on top
of the dynamic per-block scales (~1-2 days). PR #78
(`use_default_system_prompt=false`) remains the practical workaround for
typical chat prompts. Diagnostic env vars added in PR #79.

### Qwen3.5-27B MXFP4 illegal memory access at load
12 GiB MXFP4 weights + 48 GiB FP16 fallback oversubscribes VRAM on the
32 GiB RTX 5090. PR #60 added a clear diagnostic ("MXFP4 FP16-fallback VRAM
oversubscription"). Real fix needs host-dequant + StoragePlanner — ~1-2
days of work. Workarounds: 9B Q8_0, 35B-A3B Q4_K_M.

### Gemma-4 Q4_K_M output quality on complex prompts
Q4_K_M decodes coherent for chat but degenerates on complex code-gen prompts
(Fibonacci → backtick loop). Accumulated FP16 drift over 30 layers.
FP32-router / FP32-expert-down env-var stacks insufficient. Practical fix:
**use Q5_K_M or Q8_0** when output quality matters. Memo:
`memory/gemma4_q4km_vs_q8_2026_04_19.md`.

### General MoE D2H routing graph-incompatible
Non-Gemma-4 / non-NVFP4-prequant MoE decode falls through the legacy
expert-routing path with a D2H sync per layer per token. CUDA Graphs are
disabled for these models. Gemma-4 and NVFP4-prequant MoE (Qwen3.6, Gemma-4
llm-compressor) capture cleanly via the decode fast-path. Generalising the
fast-path to GGUF MoE = open work item.

---

## Open Performance Work

### MXFP4 native GGUF weight format
Native MXFP4 weights would feed directly into Blackwell tensor cores via
CUTLASS — zero dequant overhead, expected 2-4× prefill speedup vs Q4_K_M.

CUTLASS MXFP4 GEMM is fully implemented (`attention.mxfp4 = "always"`
prefill path). Only works today when an NVFP4 cache exists as source data.
Native MXFP4 GGUF would eliminate that dependency.

**Quality caveat (2026-04-23 measurement)**: Qwen3-4B wikitext-2 PPL is
Q8_0 = 8.48, Q4_K_M = 8.67 (+2.2 %). MXFP4 round-to-nearest literature sits
at +5–15 %  — **worse than Q4_K_M** unless MR-GPTQ calibrated. The
calibration step is effectively required to make this worth shipping.

Remaining work:
1. GGUF type extension + loader (~50 lines)
2. Python converter SafeTensors → block-Hadamard → MXFP4 → GGUF (~200 lines)
3. Weight upload mmap → GPU-ready format (~30 lines)
4. MXFP4 GEMV for decode (~100 lines CUDA)
5. **Required** for competitive quality: MR-GPTQ calibration (~150 lines Python)

### TurboQuant — close the gap to FP8
Current gap vs FP8 baseline: **-23% decode** (191 vs 248 tok/s on Qwen3-8B
Q8_0). Algorithm-inherent — QJL sketch computation adds per-token overhead.

Already optimised: warp-cooperative Q sketch, warp-parallel QJL XNOR+popcount,
INT4 dequant prefetch, PolarQuant symmetric clamp.

Only way to close the gap further: remove QJL entirely and use MXFP4 K
directions with group micro-scales instead.

### 1024→2048 prefill cliff on small dense models
Qwen3-4B Q8_0 drops 27k → 19k tok/s at the dispatch boundary where cuBLAS
attention hands off to FP8 FMHA. Output is correct; the kernel is just less
tuned. Options: raise the cuBLAS cap past 1024, or tune FP8-FMHA occupancy /
Bq for the small-model regime.

### `pp=512` on large dense models
Qwen3-32B Q4_K_M, Mistral-24B Q6_K: ~0.5–0.6× llama.cpp at pp=512.
Suspected cuBLAS autotuning variance + launch-overhead-bound regime.
Output correct; not gating any user.

### Speculative decoding — abandoned options
- **EAGLE-3**: 56 tok/s vs 306 baseline on single GPU. Draft model shares
  weights → 78% cost per layer.
- **Self-speculative**: 50% of baseline. Memory-bound decode can't amortise.
- **DFlash**: No draft model for Qwen3-32B; training requires datacenter GPUs.
- **TurboDraft (PPM + classifier)**: PPM 0% acceptance on real text; SVD
  classifier too lossy.
- **N-gram speculation**: Implemented, default off. CLI: `--ngram-spec`.
  +10% on repetitive content, ~0% overhead on non-repetitive.

The decode bottleneck is bandwidth, not compute. None of the speculation
variants tested can amortise weight reads on a single 5090.

---

## Research / Future

### CUDA 13.2 / CCCL 3.2 features not yet used
- **Grouped GEMM with CUDA Graphs + device-side shapes** (`cublasLtMatmulGrouped`
  with NVFP4 input, sm_120 since CUDA 13.2 Update 1). Host-sync-free MoE
  expert dispatch — expert routing stays on GPU, no D2H copy. Up to 4×
  vs multi-stream GEMM. Direct unblock for the general MoE D2H bug above.
- **`cub::DeviceTopK`** — O(n) top-k via AIR. 5× faster than radix sort for
  `top_k > 128`.
- **`cub::DeviceSegmentedReduce` (fixed-size)** — uniform segment_size variant,
  up to 66× speedup for small segments. Per-head reductions in MHA.
- **`cudaMemcpyWithAttributesAsync`** — L2 persistence hints on individual
  transfers. Prefix-cache pinning without batched API.
- **`add.f32x2` native PTX** (Blackwell) — softmax reductions, attention
  accumulation. Reduces instruction count.

### PTX ISA 9.2 opportunities
- **`cp.async.bulk` with `.ignore_oob`** — OOB reads return zero. Eliminates
  bounds-checking in TMA descriptors for variable seq lengths. Big simplification
  for paged attention with partial last blocks.
- **`st.async.b128`** — 16-byte async stores for KV cache writeback.
- **`cvt .bf16x2` ↔ narrow** (`.e2m1x2`, `.e4m3x2`) — packed FP4/FP8 pair
  conversion, 2× throughput for KV cache quant pipeline.
- **`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA — finer scale granularity
  (1 per 4 elements vs per block). See PTX MMA survey memo.
- **`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`** — Project B target,
  Stage 4 layouts decoded byte-exact (PR #55), Stage 5 integration open.

### Long-context KV memory reduction
Decode is bandwidth-bound and every sub-byte KV quantisation tested
regresses decode (see `memory/kv_dtype_tradeoffs_2026_04_24.md`). Next wins
come from reducing **token count**, not element precision:
- **K2 MLA (DeepSeek)** — latent vector replaces full K/V, -90% KV-VRAM.
- **K5 Token-eviction (H2O)** — drop 50–70% of tokens by attention score.
- **K8 CPU-offload** — async prefetch for cold tokens, enables 100K+ ctx.

Or kernel-level rewrite of INT4 decode to eliminate dequant overhead
(separate investigation; needs scale-in-register caching + fused dequant+MMA
via `mma.sync.kind::f8f6f4` block-scaled variant).

### BitDecoding / DeltaKV
- **BitDecoding** (arxiv:2503.18773): 8.6× vs FP16 FlashDecoding on Blackwell
  via MXFP4 KV cache. Builds on TurboQuant MXFP4 infrastructure.
- **DeltaKV** (arxiv:2602.08005): residual-based KV compression. 187 tok/s
  at 128K context on Blackwell PRO 6000. Orthogonal to weight quant.

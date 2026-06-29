# Roadmap

Single-author, single-GPU experiment -- "roadmap" means "current focus," not "schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

## Direction: local inference for AI agents

The goal is making imp the fastest local engine for AI agent workloads on consumer Blackwell. Agents generate far more tokens per session (20k-100k+), accumulate context fast, and often run in parallel. This demands long context, concurrent request handling, and high decode throughput.

### Phase 1 -- Long context ✓

**Shipped PR #453.** FMHA kernels got `q_offset` for chunked prefill. S-matrix shrunk 1024→256 MiB. Auto `fmha_prefill_threshold` routes long sequences to FMHA. Context ceiling moved from ~4-6k to 32k+.

### Phase 2 -- Concurrent requests ✓

**Shipped PR #454.** Server handles up to 4 concurrent decode requests. Removed single-request cancellation guard. Added `runtime.max_batch_size` config. SSM/GDN models stay batch=1.

### Phase 3 -- KV streaming for long sessions ✓

**Shipped PR #455.** Auto-enables StreamingLLM when KV cache >90% full. Graceful degradation: sink tokens + sliding window, middle blocks freed. Agent sessions effectively unlimited.

## Performance work — closed

The competitive performance frontier is exhausted: every lever the 06-12 campaign left open has
since shipped or been refuted by measurement. Both items below are now closed; the section is kept
for the record (see also "Investigated and shelved"):

- **NVFP4 prefill vs vLLM — CLOSED.** Making FP16-QK FA2 the primary hd=128 prefill
  (re-measured 2026-06-13, commit `290a163a`) lifted pp4096 +21–24%: **MoE pp4096 is now +4% ahead of
  vLLM**, MoE pp2048 +27%, dense pp2048 ~tie. The **lone residual gap is dense pp4096 at ~1.04×**
  (24.2k vs 25.3k), and that gap is now **structural, not a lever.** Every bounded kernel idea is
  refuted: Cross-Tile pipeline, Grouped-GEMM tile axis, chunk-4096, occupancy/2-CTA, and fp8-QK
  (quality-refuted, #511/#681). The last candidate — scaled fp8-KV storage with f16 compute — was
  **refuted by measurement 2026-06-16** (ncu+nsys spike on Qwen3-14B-NVFP4): at pp4096 FA2 is at ~5%
  DRAM (tensor-pipe/latency-bound, not bandwidth-bound), so halving KV read bytes targets a near-idle
  unit; the `paged_kv_gather_fp8_to_fp16` it would remove is 0.3% of kernel time; and the real
  dominant cost is the NVFP4 GEMMs (~59%), a separately-refuted structural ceiling. The competitive
  case is won — dense pp4096 ~4% is left as a structural ceiling, not active work.

- **kv-fp8 storage default-on** -- SHIPPED for Qwen3 dense + Qwen3 MoE + Llama (Phi-4) + Nemotron-H MoE.
  `kv_cache.dtype` now defaults to `auto`, which honors a model author's `kv_cache_quant_algo=FP8` hint for
  arch families that pass the long-context quality gate (`kv_fp8_hint_default_safe` — Qwen3-14B PPL +1.07%,
  Qwen3-30B-A3B neutral, Phi-4 +0.25%, Nemotron-3-Nano-30B ~0.00%, all coherent; ~768 MiB KV VRAM saved on
  the dense-attention models). The −35% MoE tax was removed in #682. The pending hint-declaring families
  were swept in #749 + 2026-06-29: **Nemotron-H MoE** is now in (its 2026-06-23 single A/B was run-to-run
  noise — the MoE+Mamba2 hybrid is nondeterministic even in deterministic mode, so a 5-run mean was needed,
  and FP16 vs FP8 means coincide). **Remaining (blocked, not actionable):** Qwen3.6-35B and Qwen3.5 don't
  declare the FP8 hint (allowlisting is moot); Gemma-4's baseline PPL on the gate corpus is broken (~243),
  so it can't be gated on this corpus. These stay FP16 (or `--kv-fp8` opt-in).

The two items below are **retained for the record but evidence-refuted**, not active work:

- **Q4_K_M prefill gap** (-38% vs llama.cpp) -- the in-SMEM Q4_K MMQ + FP16 HMMA approach is now **evidence-refuted**: the `feat/q4k-mmq-hmma` forge experiment built exactly this kernel and ncu-proved it's decode-throughput-bound, *tying* (not beating) cuBLAS — and the gap is vs llama.cpp beating cuBLAS, so closing it needs to beat cuBLAS (decode-tax-blocked) or pay 2× weight VRAM via pre-shuffle (rejected). The biggest gaps are also MoE-expert (small-M) which a dense tile kernel doesn't touch. See the "Evidence from the forge experiment" section in [`specs/2026-05-28-q4k-mmq-kernel-design.md`](superpowers/specs/2026-05-28-q4k-mmq-kernel-design.md). Practical resolution: recommend NVFP4 SafeTensors for fast Q4_K-class prefill.

- **Sawtooth Wavefront Reordering** (PR #456) -- alternate KV scan direction per Q tile for L2 locality. **MEASURED 2026-05-29 — no realized benefit, REFUTED.** (1) It lives ONLY in `flash_attention_blackwell` (the WMMA *fallback*), not in the cuBLAS / FP8-FMHA / FA2 paths (the "both FMHA kernels" claim was wrong). (2) That kernel is **unreachable on the NVFP4 prefill hot path**: prefill routes to `attention_cublas_prefill` (≈30% faster than FMHA per the in-tree note), and the per-attention-call seq stays under the auto `fmha_prefill_threshold` (~cap+1) — blackwell only runs if you force `threshold=1` + `fp8_fmha=never` + `fmha_sm120=never`. (3) Even force-routed, A/B is flat-to-slightly-negative (pp8192 13.10k ON vs 13.14k OFF; pp16384 12.45k vs 12.49k; ON ~0.3% slower, within noise). Left in place (harmless, 32k+ untested due to OOM on single-chunk 14B). A/B harness: `tools/analysis/sawtooth_ab.sh`. Memory: `sawtooth_reordering_refuted_2026_05_29`.

## Architecture support

- **MLA (DeepSeek-V2/V3)** -- latent-vector KV for ~93% KV-cache reduction. **Shipped** (#802 Stage A materialized, #803 Phase 3 absorbed latent-KV-cache decode, opt-in). First MLA arch in imp; validated on DeepSeek-V2-Lite (see [`supported-models.md`](supported-models.md)). The design spec + implementation plan were consolidated into [`archive/README.md`](archive/README.md) on completion. Remaining MLA family (V3 / GLM / Kimi / Ling) reuses this path; not yet staged locally.

## Tooling watch -- CUDA Tile for C++ (re-evaluate on 13.3)

NVIDIA CUDA Tile (cuTile / Tile-IR) -- tile-level kernel authoring in C++ where the
compiler orchestrates the tensor-core MMA + smem layout (Triton-style, but native
C++). Long-term this could replace imp's hand-written `mma.sync` + smem-layout kernels
(the FA2 prefill kernel, the NVFP4 grouped GEMM) with far less code.

**CUDA 13.3 flips the two blockers that pinned the prior defer:**
- **C++ surface shipped** -- was "planned, no date" through 13.2 (Python-only DSL).
- **sm_120 + FP4** -- 13.3 adds `f4E2M1FN` (FP4 E2M1) and `i4` Tile types for sm_120,
  i.e. the block-scaled bleeding-edge imp actually uses is now in scope on paper.

**The gate is perf on consumer Blackwell, and the prior evidence is bad:** the only
published sm_120 measurement (Yadav et al. 2026-05, RTX PRO 6000 = same sm_120a arch)
showed cuTile fused attention at **0.53× FA2**, while the *same* kernel hit 2.5× FA2 on
B200 -- a 4.7× cross-arch gap. imp's hand-written FMHA already matches/beats FA2 on
sm_120. So Tile is not a hot-path migration until a 13.3 cuTile-vs-FA2/CUTLASS
benchmark **on sm_120** shows ≥parity.

C++ header confirmed in-toolkit: `/usr/local/cuda-13.3/include/cuda_tile.h` (2026-05-29).

**RESOLVED (2026-05-29) — benchmarked on sm_120, does NOT reach parity → SHELVED.** Built a real Python cuTile FA2 (causal fp16, correct: max_rel_err=0.0) and ran `cuda.tile.tune.exhaustive_search`. **Autotuned ceiling = 26.5 eff-TFLOPS = 3.2% of the 838 roofline** (naive 18; autotune lift only ~1.1–1.5×; tile-size the sole lever). That's ~order-of-magnitude below competitive and far below the native hand-FA2 — confirming Yadav (0.53× FA2, same arch). The "≥parity" gate fails, so **the multi-week Tile FA2 integration (Phase 2-5) is NOT being built** (it would ship a backend slower than native). Tile stays investigated-and-shelved. Harness: `tools/analysis/cutile_fa2.py` (+ `Dockerfile.cutile`). Memory: `cutile_autotune_ceiling_shelve_2026_05_29`.

**Action (not blocking current FMHA/NVFP4 work):** once 13.3 is in, (1) ~~confirm the C++
headers ship in-toolkit~~ ✓ done, (2) prototype one *non-hot-path* kernel (e.g. a rowsum/reduce)
to assess C++ ergonomics + debuggability, (3) micro-benchmark a Tile NVFP4 GEMM and a
Tile FP4 attention vs the current hand-written kernels on the 5090. Migrate only the
kernels that win. Full research note + re-eval triggers: memory `tile_ir_readiness_2026_05_09`.

### CompileIQ auto-tuning (CUDA 13.3) -- low-risk, near-term

NVIDIA's new compiler auto-tuner: evolutionary/genetic search over ptxas/nvcc configs
per kernel, emitting an Advanced Controls File. NVIDIA reports **up to 15% on
already-optimized Triton-attention + CUTLASS-GEMM kernels**. Unlike CUDA Tile this is
*not* a rewrite -- it tunes codegen for existing kernels, so it composes with imp's
hand-written `mma.sync` kernels (mechanism = ptxas/nvcc params; applicability to
hand-written CUDA C++ to be confirmed, headline examples are Triton/CUTLASS). Strong fit
for imp's profile (batch=1, few dominant hotspots). **Plan:** after the FA2 prefill
kernel has a measured baseline, run CompileIQ on it (and on the NVFP4 GEMV/GEMM hotspots)
as a last-mile squeeze; ship the ACF if it survives the perf gate + cooldown methodology.

**RESULT (2026-05-29) — REFUTED, no win.** CompileIQ is operable (v1.0.0; `PtxasSearchSpace(version="13.3")`
downloads its search space). But the ptxas space is *flat* on imp's hotspots. Direct sweep of the
search space's decisive axes on the FA2 kernel (`maxrregcount` 64–200, `--def-load-cache` ca/cg/cs,
`--allow-expensive-optimizations`, `--use_fast_math`) → all within **±0.4%** of baseline (pp4096 ≈
19.6k tok/s, Qwen3-14B-NVFP4). The kernel is **smem-occupancy-bound** (REG:144 but SHARED:40 KiB; cutting
regs 144→64 moved pp by 0%) + barrier-bound — ptxas codegen touches neither. NVFP4 decode is CUTLASS-
generated + HBM-bandwidth-bound (M=1, 64–73% peak) → ptxas cannot add bandwidth. So imp's hand kernels are
already at their structural limits; CompileIQ's Triton/CUTLASS-style codegen slack isn't there. Reusable
harness left in tree: `tools/analysis/Dockerfile.ciq` (→ `imp:ciq`) + `tools/analysis/ptxas_sweep.sh` for
any future codegen-bound kernel. Memory: `compileiq_ptxas_native_refuted_2026_05_29`.

## Known limitations

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Blackwell only.** No Hopper, Ada, Ampere. No AMD, Intel, Apple, CPU.
- **Gemma-4 Q4_K_M code-gen drift** -- the original `gemma-4-26B-A4B-it-Q4_K_M.gguf` degenerated into
  backtick loops on code prompts at temp=0 (Q8_0 stayed clean). **No longer reproduces (verified
  2026-06-13):** the current local `UD-Q4_K_M` produces coherent valid Python on the same prompt.
  Likely closed by intervening work (tokenizer #657, gemma SWA prefill routing #566/#569) and/or the
  better Unsloth-dynamic quant; the original file is gone, so it can't be A/B'd. If you hit
  degeneration on some other Q4_K_M quant of this model, fall back to Q5_K_M or Q8_0.
- **Qwen3.5-27B MXFP4 fails at load** -- blocked on no public MXFP4 GGUF + NaN bug.

## Investigated and shelved

- **Speculative decoding** -- none amortize weight reads on single bandwidth-bound GPU.
- **FFN contextual sparsity** -- warp-cooperative layout masks the skip. +0-1% measured.
- **BitDecoding (TC KV decode)** -- decode is weight-bound, not attention-bound. 0% gain.
- **NVFP4 GEMV tuning** -- 6 approaches refuted. 157 tok/s is 64-73% HBM peak.
- **FMHA rewrites** -- cluster, TMA bulk, long-context heuristic all A/B tested. cuBLAS wins.
- **MoE offload + CUDA Graphs** -- `expert_overhead_pct=10` default keeps most models on-device. Full kernel-driven slot resolution deferred (multi-week, marginal user impact).

# Roadmap

Open work and known limitations. Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

This is a single-author single-target experiment, so "roadmap" is more "current focus" than "schedule." Items here are ordered by impact, not by ETA.

## Known limitations

### ~~Gemma-4 carve-outs~~ — all removed

All three Gemma-4 carve-outs are now gone:

- **FP8 KV cache** — PR #91 (2026-05-01). The "dual head_dim 256/512 needs per-layer-aware kernels" hypothesis was a red herring; the KV write/read kernels handle per-layer head_dim correctly via `Q.shape[3]` template dispatch. Real bugs were (a) FP8 calibration reading the workspace's allocated shape (`max_hd=512`) instead of the live shape (`hd=256` on SWA layers, junk in trailing 256 cols) and (b) warmup-derived absmax poisoning the high-water-mark scale on Gemma-4's `output_norm` outliers (max=588).
- **NVFP4 decode cache for Q*_K source** — PR #186 (2026-05-15). The per-tensor convert→quantize loop in `executor_pre_dequant.cu` already handled mixed (N, K) shapes correctly; the disable was overly defensive. Removing it on Q4_K_M / UD-Q4_K_M: pp512 1713 → 2394 tok/s (**+40%**), tg256 176 → 197 tok/s (**+12%**).
- **FP8 prefill** — 2026-05-15. The 2026-05-09 -5..-19% slowdown was real at the time but mostly closed by intermediate prefill work (PRs #177, #181). Re-measured Q4_K_M: pp128 +1.0%, pp512 -0.9%, pp833 -4.2%, pp2048 **+7.3%** — neutral with long-context advantage. FP8 also halves the activation cache size. Users wanting max prefill at medium pp can opt out via `[attention] fp8_prefill = "never"`.

Default KV dtype is FP16; FP8 is opt-in via `--kv-fp8` (or `kv_cache.dtype = "fp8"` in `imp.conf`). Coherent on Qwen3 dense, Qwen3.5/3.6 GDN, Llama-3.2, and Gemma-4 (post PR #91).

### Chunked prefill scope (full-attention + hybrid GDN/Mamba2 + Gemma-4; FP16/FP8/NVFP4/INT4 KV)

Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral), hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H), and **Gemma-4** with FP16, FP8, NVFP4, or INT4 KV cache. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk before a rectangular `attention_cublas_prefill` with `q_offset`-aware causal masking + `sliding_window`-aware mask (added 2026-05-15 for Gemma-4 SWA layers; the same path now replaces the naive FP32 workaround for Gemma-4 SWA in non-chunked prefill too). INT4 dequant gather added 2026-05-15 (symmetric 4-bit + per-head FP16 scale; INT4 KV's pre-existing long-context quality regression is independent of chunked prefill). PR #114 mitigation (default `prefill_chunk_size = 0`) is replaced by `Engine::resolve_prefill_chunk_size_()` which clamps to 0 for out-of-scope archs.

**Out-of-scope** — stay at `prefill_chunk_size = 0` via per-arch default; explicit `--prefill-chunk-size N` is logged + clamped to 0:

- Gemma-3 (SWA, no test model in repo — kernel work is identical to Gemma-4, just unverified)
- Llama-4 (MoE + SWA)
- TurboQuant / TurboQuant Lite KV dtypes (QJL-sketch storage; would need a sketch-aware gather)

Each excluded class is a separate larger work item.

### ~~`d_pf_block_tables_` undersized for prompts ≥ max_seq_len~~ — FIXED #134

When a single prompt exceeds `max_seq_len`, the engine's pre-allocated device buffer `d_pf_block_tables_` (sized `max_seq_len / block_size`) overflowed during `cudaMemcpyAsync`. Fixed in PR #134: `d_pf_block_tables_` is now sized from `max_blocks` (the total KV cache pool count), so a single request's block_table can grow to the entire cache without overflowing.

### NVFP4 SmoothQuant input_scale (Mistral-3.2 NVFP4)

Mistral-Small-3.2-NVFP4 was calibrated with SmoothQuant 0.9, which records a per-Linear `input_scale` in SafeTensors. imp loads `input_scale` into `nvfp4_scratch_` but does not consume it at inference. PR #78 worked around long-prompt drift by disabling the 600-token default system prompt.

Direct absorption of `input_scale` as a per-tensor scalar GEMM alpha modifier (in either direction) was tested 2026-05-07 on Gemma-4-NVFP4 and refuted: phase4 18/20 → 4/20 (full degeneration). A real fix likely needs the per-channel SmoothQuant scaling vector applied during activation quantization, not a scalar alpha modifier — testable only against Mistral-3.2-NVFP4 (not present in default model set).

### Qwen3.5-27B MXFP4 fails at load

12 GiB of MXFP4 weights plus the 48 GiB FP16 fallback oversubscribes 32 GB of VRAM. PR #60 added a clear diagnostic. A real fix needs host-dequant + a storage planner. Workarounds: 9B Q8_0, 35B-A3B Q4_K_M.

### Gemma-4 Q4_K_M code-gen drift

Q4_K_M decodes coherent for chat but degenerates on complex code-gen prompts (Fibonacci → backtick loop). Cause is accumulated FP16 drift over 30 layers. Practical fix: use Q5_K_M or Q8_0 when output quality matters.

### MoE expert offload disables CUDA Graphs

Decode fast-path (`src/graph/executor_forward_moe.cu:524`) handles all device-resident MoE quants — Q6_K, Q8_0, Q4_0, Q4_K, Q5_K, Q2_K, Q3_K, Q5_1, NVFP4 — fully device-side (no D2H memcpy of routing or expert offsets), so CUDA Graphs capture cleanly. Verified A/B 2026-05-07: Qwen3-Coder Q6_K tg128 117 → 232 tok/s (+97%), Gemma-4 Q4_K_M tg128 65 → 179 tok/s (+177%).

The remaining limitation is **host-offloaded experts**: when the model + KV doesn't fit in VRAM, `experts_on_host_=true` triggers per-layer H2D staging via `expert_cache_` LRU at `executor_forward_moe.cu:1517`, which inserts a host pointer dereference + `cudaMemcpyAsync` per expert per token. `engine.cpp:936` disables CUDA Graphs in that mode. Tip: bumping `IMP_EXPERT_OVERHEAD_PCT` from 30 to 10 trades VRAM headroom for full on-device experts and unlocks +180% decode on Qwen3.6-35B Q4_K_M. Generalising the LRU prefetch to be device-side / async-pipelined would restore Graphs while keeping host-offload available.

### Reasoning models + JSON schema — preamble pass-through

Reasoning models (Qwen3.6, DeepSeek-R1, Gemma-4-thinking) emit `<think>...</think>` before every response. Strict JSON / JSON-Schema enforcement starting at token 0 masks the `<think>` opener, leaving the model with no valid token to sample. Auto-detected via the tokenizer (presence of `<think>` + `</think>` special tokens) and handled by `PreambleGate` (`src/compute/preamble_gate.h`): the gate lets all tokens pass until the close marker, an `{` / `[` is observed, or a budget cap is hit, then strict enforcement kicks in.

Non-reasoning models (Llama-3.2 etc.) get the same gate in budget-only mode (8-token slack) so markdown-fence preambles like ` ```json ` and short verbal openers ("Sure! ") pass through cleanly.

When a request sets both `tools` and `response_format=json_schema`/`json_object`, the engine-side `PreambleGate` enters tool-aware mode. It bypasses the schema mask through the entire tool-call body (delimited by single-token tags for ChatML/Hermes/Mistral/Gemma, or `<function=`/`</function>` char-prefix/suffix for Llama3) and stays unmasked for the rest of the generation, supporting parallel tool calls. If the model emits free-text JSON instead, the schema mask kicks in normally on the first `{`/`[`. Tool argument validation continues to flow through each tool's own `parameters` schema (post-hoc, not in-stream).

## Performance work

### Closing the TurboQuant–FP8 gap

TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode (191 vs 248 tok/s). The gap is algorithm-inherent — QJL sketch computation adds per-token overhead. Closing it would need to drop QJL and switch to MXFP4 K directions with group micro-scales.

### `pp=512` on large dense models

Qwen3-32B Q4_K_M and Mistral-24B Q6_K sit at ~0.5–0.6× llama.cpp at `pp=512` (Qwen3-32B Q4_K_M: 1888 tok/s, RTX 5090). nsys profile (2026-05-15, `tools/analysis/profile_pp512_large_dense.sh`):
- 4.4 % FP16 compute / 3.4 % memory-bandwidth utilization — launch-overhead + dequant-overhead bound.
- 25 % GPU time in `dequant_q4k_kernel` (FP16 cache doesn't fit), 64 % in cuBLAS GEMMs.
- 23 % host time in sync `cudaMalloc`/`cudaFree` (939+930 calls).

imp already ships a Q4_K × Q8_1 kernel (`src/compute/ggml_mmvq.cu::mmvq_kernel`) but it's a *warp-per-output-element batched-GEMV*, not a tiled GEMM. Measured crossover on Qwen3-32B Q4_K_M (`tools/analysis/bench_q4k_mmvq_crossover.sh`): mmvq wins at M ≤ 16 (e.g. M=8: 92 vs 45 tok/s), cuBLAS wins above M=16 (M=512: 1802 vs 251 tok/s). mmvq saturates at ~250 tok/s regardless of M because each output element gets its own warp with no TILE_M × TILE_N weight/activation reuse.

A **direct tiled Q4_K_M GEMM kernel** shipped 2026-05-15 in `src/compute/mmq_q4k.cu` (commits `3b49325` → `8dbfdbd`). Microbench beats mmvq by 2.0–2.4× across M=32..512 (tile `<16,32,1,1>`, 512 thr, SMEM bank-conflict-padded). End-to-end on Gemma-3-12B Q4_K_M: **wins +13–56 % at M=2..16**, **loses to FP16-TC cuBLAS at M ≥ 32** — dp4a peak (~50 TFLOPS) vs FP16-TC peak (~838 TFLOPS) is a 16× ceiling gap that tile tuning cannot close. Dispatched in `[2, 16]` only via `executor_kernels.cu`; high-M Q4_K_M still goes through dequant+cuBLAS. Multi-stream dequant↔GEMM overlap was refuted by measurement (GPU busy ratio ≈ 100 %).

Closing the high-M gap requires porting the inner loop to a Tensor-Core MMA. Two paths exist on sm_120:

1. **FP16 HMMA** (`mma.sync.m16n8k16.f16`) with on-the-fly Q4→FP16 dequant. **Attempted and retired**: shipped across 7 phases as `src/compute/mmq_q4k_v2.cu` on a feature branch, microbench reached **4.87× v1 dp4a** at M=512 (kernel-only). End-to-end on Qwen3.6-35B Q4_K_M was **-4% pp** in production because MoE keeps experts under the `MIN_M=64` v2 threshold, the FP16 weight cache (`wcache_.fp16`) hits skip v2 entirely, and Phase 1 dispatch overhead is paid per call. Retired in PR #193. Re-eval pending a dense Q4_K_M model that bypasses both the MoE-min-M and the fp16_cache hot path. Memo: `mmq_q4k_v2_phase2_shipped_2026_05_16.md`.
2. **INT8 IMMA** (`mma.sync.m16n8k32.s32.s8.s8.s32`, ~838 TOPS) with Q4_K dequant→INT8 reordering. Untried — 16× ceiling vs current dp4a, but requires a Q4→INT8 staging pass and a different MMA register layout than v2's HMMA path.

Memos: `mmq_q4k_phase_a_2026_05_15.md` (v1 dp4a sweep), `mmq_q4k_v2_hmma_design_2026_05_15.md` (v2 blueprint), `q4k_mmvq_crossover_2026_05_15.md` (original mmvq-vs-cuBLAS measurement).

### Speculative decoding — investigated and shelved

EAGLE-3, self-speculative, DFlash, PPM-based TurboDraft, and n-gram speculation were all investigated. None paid off on a single RTX 5090: decode is bandwidth-bound, and the variants tested either failed to amortise weight reads (EAGLE-3, self-spec at 56–50% of baseline) or had unacceptable acceptance rates (PPM 0% on real text). Spec-decode CLI flags were removed in `7380ea8`.

### FMHA cluster-launch — investigated and shelved (M5)

A Blackwell-style cluster-launch variant of the FMHA prefill kernel (DSMEM K-broadcast across a 2-CTA cluster, modelled after `paged_attention_cluster_kernel<HD>`) was implemented across three slices in May 2026: Slice 1 helper (#198), Slice 2 FP16 cluster kernel + dispatch (#200), Slice 2.2 FP8 variant (#202). End-to-end A/B across 4 NVFP4 MoE models (Qwen3.6-35B, Coder-30B, Gemma-4-26B, Qwen3-30B-Modelopt) on 2026-05-17 found the perf signal **noise-dominated**: ±20% same shape, opposite signs across re-runs of the same config; cuBLAS / thermal / scheduler variance drowned any cluster effect. Cluster output is **bit-identical to legacy** (`ClusterMatchesLegacy*` tests: max_abs = 0), so the kernel is sound; the maintainability cost (a parallel kernel + dispatch + dual test files) doesn't pay back. Default flipped OFF in #204 via `attention.no_fmha_cluster=true`; code retained as opt-in for future hardware where the signal might emerge. Memo: `m5_slice2_cluster_refuted_2026_05_17.md`.

### GEMM dispatch unification (R5)

The 21-parameter `gemm_dispatch_impl` god-dispatcher is being replaced by a strategy-keyed `GemmKernel` registry. Slice 1 (PR #197) shipped the interface (`src/graph/gemm_kernel_registry.h`) + an FP16 kernel as proof-of-concept; live dispatch tries the registry first when `gemm.use_kernel_registry=true` (default OFF) and falls back to the legacy switch for unmigrated tiers. Per-tier slices, one PR each off main:

| Slice | Tier | Status |
|---|---|---|
| 1 | FP16 interface + proof | shipped (#197) |
| 2 | FP8 | in flight |
| 3 | NVFP4 GEMV (M==1) | pending |
| 4 | NVFP4 GEMM (M>1) | pending |
| 5 | CUTLASS_NVFP4 | pending |
| 6 | MXFP4 | pending |
| 7 | dp4a / mmvq | pending |
| 8 | Delete legacy switch, flip default ON | pending |

Once all migrated, adding a new qtype becomes a single-file change instead of a 21-parameter dispatcher edit. Cross-axis maintainability win per review `phase5_synthesis.md §5`.

### MoE prefill graph capture (M3 Phase 4)

Decode-side CUDA Graphs default-on (#114-ish) gives +180-234% on NVFP4 MoE decode. The prefill-side equivalent is blocked on **Blocker B**: CUTLASS 3.x grouped-GEMM silently hangs under `cudaStreamCaptureModeGlobal`. PR #196 flipped the default `runtime.graph_capture_mode = "relaxed"` as the cheapest probe (relaxed mode permits more synchronization patterns inside the captured graph). Empirical A/B with `runtime.prefill_graph=true` under relaxed-mode capture is the next step; results pending. If relaxed unblocks, M3 Phase 4 can default-on. If not, options are (a) file a CUTLASS issue upstream, (b) a workaround in our capture code, or (c) carve out the grouped GEMM into a graph-external sub-stream. Memos: `prefill_graph_blockers_2026_05_14.md`, `prefill_graph_cublaslt_blocker_2026_05_15.md`.

## Research interest

These are upstream features that would unlock real wins but haven't been integrated yet.

### CUDA 13.2 / CCCL 3.2 features

- ~~**Grouped GEMM with CUDA Graphs + device-side shapes**~~ — re-tested 2026-05-08 against cuBLAS 13.4.0.1 (`tools/analysis/probe_cublaslt_grouped.cu`): zero algorithms returned for FP16/BF16/FP8/NVFP4 on sm_120. Grouped layout API still marked Experimental in `cublasLt.h` 13.4 and only supported on datacenter Blackwell (SM100/B200), not consumer SM120. Re-run probe on each new cuBLAS release.
- ~~**`cub::DeviceTopK`**~~ — already wired in production (`src/compute/sampling.cu:834`, `cub::DeviceTopK::MaxPairs` for the `top_k > MAX_TOP_K=128` path with a small follow-up `DeviceRadixSort` over just the top-k results for top-p ordering).
- ~~**`cub::DeviceSegmentedReduce`**~~ — re-evaluated 2026-05-08, no applicable use case in imp. The 66× speedup claim applies to host-launched many-small-segments patterns (e.g. CUB benchmarks reducing thousands of fixed-size rows in one call). imp's per-head reductions are all already inside their owning kernel as warp-/block-level shuffle reductions (RMSNorm, attention softmax, MoE gate norm) — fused, optimal, and unrelated to DeviceSegmentedReduce's regime.
- ~~**`cudaMemcpyWithAttributesAsync`** (NUMA hint use case)~~ — shipped #131 at the recurring H2D paths (`src/memory/layer_offload.cu`, `src/runtime/vision_pipeline.cpp`) with `srcAccessOrder=Stream` + `srcLocHint=HostNumaCurrent`. The L2-persistence-hint use case (prefix-cache pinning without batched API) is still open.
- ~~**`add.f32x2` native PTX** (Blackwell)~~ — investigated #131. ptxas on consumer Blackwell (sm_120) accepts the legal PTX op but **decomposes it into 2× scalar FADD at SASS** — the vectorized hardware path is only exposed on datacenter Blackwell (SM100/B200). Helper `imp::add_f32x2` lives in `src/compute/ptx92_utils.cuh` for forward-compat with future toolkits / hardware. No SASS-level instruction-count reduction achievable on RTX 5090.

### PTX ISA 9.2

- **`cp.async.bulk` with `.ignore_oob`** — eliminates bounds-checking in TMA descriptors for variable seq lengths. Simplifies paged attention with partial last blocks.
- **`st.async.b128`** — 16-byte async stores for KV cache writeback.
- **`cvt .bf16x2` ↔ narrow** (`.e2m1x2`, `.e4m3x2`) — packed FP4/FP8 pair conversion, 2× throughput for KV cache quant.
- **`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA — finer scale granularity (1 per 4 elements vs per block).
- **`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`** — QKT path shipped in PR #56 (commit `b51788e`, default-on via `attention.fmha_blockscale = "auto"`); per-16-element UE4M3 SFA/SFB feed real `q_scales_fp8` / `k_scales_fp8`. Measured +1.8% Qwen3-4B MXFP4 at HD=128 (Phase 1 MMA is only ~15% of FMHA wall time, so 2.5× raw MMA → small visible delta). Open lever is **FP4 PV** (Phase 3 P×V): same MMA op for the second GEMM in attention, ~+13% additional upside but quality-risky (FP4-quant of post-softmax probabilities — needs SageAttention3-style two-level accumulator). 200-300 LoC, prereq is a PV-only A/B test harness. HD=256 models (Qwen3.5 GDN, Gemma-4 globals) would also see a larger Phase-1 fraction and better visible speedup, but no clean MXFP4 HD=256 model is currently in the test set.

### Long-context KV memory

The "decode is bandwidth-bound, sub-byte KV quant regresses decode" framing was partially obsolete after Lever 2 NVFP4 KV (PR landed 2026-05-07/08): NVFP4 storage with vectorized-PTX dequant is at parity with FP16 decode at 3.9× compression. The remaining decode-perf headroom comes not from changing the storage format but from **changing what kind of compute does the dequantized math** — see BitDecoding below.

- **K2 MLA (DeepSeek)** — latent vector replaces full K/V, ~–93% KV VRAM, 5.76× max throughput. **DEFERRED**: gates on adding DeepSeek-V2/V3 architecture support to imp; no MLA-arch model in scope today. Bonus paper [arxiv:2502.14837](https://arxiv.org/abs/2502.14837) proposes retrofitting MLA into non-MLA pretrained models — worth tracking but its own research project. Re-eval when imp adds DeepSeek-V2/V3 support or a calibration-only MLA recipe ships.
- **K5 Token eviction (H2O)** ([arxiv:2306.14048](https://arxiv.org/abs/2306.14048)) — Heavy-Hitter Oracle eviction by attention-score power-law; 5–20% retention, ≤20× memory. **POSSIBLE BUT QUALITY-RISKY**: well-documented retrieval-task degradation since 2023 (RULER, NIAH, multi-hop QA). Build only if VRAM-pressure becomes the bottleneck on contexts that NVFP4 KV alone can't fit. Re-eval if a successor with retrieval-quality fix lands (Q-Hitter, SnapKV, PyramidKV are candidates).
- **K8 CPU offload** — async prefetch for cold tokens, enables 100K+ context.

### KV cache compression research

- **BitDecoding** ([arxiv:2503.18773](https://arxiv.org/abs/2503.18773), HPCA 2026) — **HIGHEST-ROI item** in this section. Tensor-Core decode on dequantized NVFP4 KV: 8.6× over FP16 FlashDecoding-v2 on Blackwell (RTX 5090 in benchmarks), 3× E2E latency on Llama-3.1-8B 128K. imp's current NVFP4 KV (Lever 2) gets the VRAM win but uses CUDA cores → parity-only on decode tok/s; BitDecoding is the missing perf piece. Empirically confirmed 2026-05-09: SASS audit of `paged_attention_decode_nvfp4_kernel<128>` shows **346 scalar FP ops, 0 HMMA** (`tools/analysis/sass_nvfp4_paged_decode.sh`). Phase-0 microbench (synthetic isolated Q.K dot): scalar-FFMA vs WMMA-HMMA equivalent within 1e-4 rel, but **0.82× speedup** (TC slower) — anticipated, BitDecoding's 8.6× requires four combined levers, not just TC dispatch alone. Phase-1 production kernel shipped 2026-05-09: `attention_paged_nvfp4_tc.cu` with WMMA Q.K dot, env-var opt-in `IMP_USE_BITDECODING_QK=1`. SASS audit confirms 24 HMMA in TC kernel; default scalar kernel unchanged at 0 HMMA. E2E smoke (Qwen3-8B Q8_0 + `--kv-nvfp4`, pp=129 + tg=64): TC tg=148.22 vs scalar tg=149.08 (within noise). Phase 2 (TC V accumulation) gates on production-shape A/B at long context where KV reads dominate. Stateless, paged-compatible, no retraining, sm_120 explicit. Plan: `docs/superpowers/plans/2026-05-09-bitdecoding-port.md`. Ref impl: [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding).
- **DeltaKV** ([arxiv:2602.08005](https://arxiv.org/abs/2602.08005)) — residual-based sparse KV compression: encode tokens as residuals against retrieved historical references; 29% KV memory, 2× throughput. **DEFERRED**: bundled with [Sparse-vLLM](https://github.com/CURRENTF/Sparse-vLLM) (near-fork, not a library); imp would need to re-architect the dense paged KV layer. Marginal win over the NVFP4 + BitDecoding stack at high engineering cost. Re-eval if imp pushes past 256K context where DeltaKV's long-range similarity advantage grows, or if a paged-vLLM-compatible reference appears.

Detailed per-item evaluation lives in `kv_research_grade_eval_2026_05_09.md` (memory file).

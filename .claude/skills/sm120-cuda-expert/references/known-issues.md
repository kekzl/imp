# sm_120a Known Issues, Dead Ends, Root-Cause Reference

---

## Pre-flight before non-trivial kernel work

Many "obvious" optimizations are proven failures on sm_120. If the installed CUDA version differs from testing time (check `nvcc --version`), "Version-dependent dead ends" entries may be worth retrying. Small edits (parameter tweak, kernel-signature change, fusing two existing kernels) can skip pre-flight.

---

## Version-dependent dead ends (worth retrying when CUDA version changes)

> **CUDA 13.3 re-test (2026-05-29, PTX ISA 9.3): NO new sm_120a capability.** Full
> `ptx_survey_all.sh` at `compute_120a` under 13.2 vs 13.3 = **0 of 247 instructions
> flipped** (none unlocked, none regressed). The "retry on CUDA 13.3+" rows below were
> re-probed and stay ❌. sm_120's ISA surface is silicon-fixed; toolkit bumps don't add
> tcgen05/wgmma/TMA. Baselines: the two `docs/ptx-status-*-sm120a.md` snapshots were consolidated away in #805 - regenerate with `tools/analysis/ptx_survey_all.sh` rather than looking for them.
> 13.3's value is tooling (CUDA Tile C++, CompileIQ) + cuBLAS perf, not instructions.

| Dead end | Blocked by | Retry on |
|----------|-----------|----------|
| cuBLASLt grouped layout sm_120 | Zero algorithms for consumer Blackwell | New cuBLAS release (check algorithm count) |
| CUTLASS TC GEMM at M=1 | Activation quant + TMA overhead | ~~CUTLASS 4.5+~~ pin has moved (read `cmake/imp-deps.cmake`, v4.7.0 as of 2026-08) - never re-probed, and now largely MOOT: the M<=32 gap was closed by the in-tree smallm v2 kernel (#1766), not CUTLASS. M=1 decode GEMV stays at its measured 66-70% HBM ceiling (4-bit-dequant co-limit). |
| `cp.async.bulk` with `.ignore_oob` | Requires TMA descriptor rewrite | ~~CUDA 13.3+~~ still ❌ on 13.3 - TMA not on sm_120; next major |
| `st.async .b128` to global | PTX 9.2 only targets `shared::cluster` | ~~New PTX ISA~~ still ❌ on PTX ISA 9.3 (13.3) |
| CUTLASS NVFP4 sm_120 graph-determinism | Universally non-deterministic for `cudaGraphExecUpdate` re-capture (verified 2026-05-05) | Future CUTLASS NVFP4 deterministic mode |
| ~~Native FP4 GEMM faster than dequant→cuBLAS on sm_120~~ | **RESOLVED 2026-08-25 (#1766):** the retry condition ("future custom kernel") was met in-tree - see "Resolved" below. | - |

---

## Resolved (no longer dead ends)

- **Native FP4 GEMM slower than dequant paths at small M - inverted 2026-08-25 (#1766).**
  `src/quant/nvfp4_gemm_smallm_v2.cu` runs `mma.sync.kind::mxf4nvf4.block_scale` on PLAIN packed layout (same weight bytes as M=1 GEMVs, zero extra VRAM; SF/fragment from `SM120_16x8x64_TN_VS`, M32xN64xK256 tile, 6-deep smem ring, 1 producer + 4 consumer warps). Isolated at M=32 N=5120 K=5120: **10.4 us** vs CUTLASS 41.4, W4A16-dequant+HMMA 23.9, Marlin sidecar 14.4, weight floor 8.2. E2E Qwen3.8-27B-NVFP4: +16.0% at 32 streams, +36.0% at 8. Default ON via `gemm.nvfp4_smallm` / `nvfp4_smallm_impl=2` (`src/core/config/gemm.h`). Marlin W4A16 sidecar REJECTED (#1756/#1757, PR #1764 closed unmerged: needs repacked weight copy, 13% coverage on 27B). In-situ stage/stripe REFUTED (#1768, shipped stages=6/stripes=1 beats overrides by 7-13%). Per-shape residual bounded 68-86% of DRAM floor; perfect-floor bound ~237 us/token vs measured 388.9 (`docs/plans/2026-08-24-qwen38-port.md`).

- **PTX `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}`** and reverse direction work on both `sm_120f` and `sm_120a` under CUDA 13.2.1 (re-verified 2026-05-04). Correct usage routes the FP4 byte through a `.b8` register - see `references/ptx-patterns.md` "FP4 ↔ FP16/FP32/BF16 packed conversion". SASS confirms hardware emission: `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

- **Build target `sm_120a`** (was historically blocked by a `ptxas` C7600 bug on `120f` that needed the `f` workaround). As of CUDA 13.2.1 the `a` arch suffix is the correct target - superset of `120f`, adds `mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM. Switched 2026-05-04 (commit `6568652`).

- **CUDA Graphs + prequant-NVFP4 MoE.** Earlier "non-Gemma-4 MoE blocks graph capture" claim was stale. The MoE decode fast-path (`executor_forward_moe.cu`, the `n=1, device-resident packed experts` branch) is fully device-side, no D2H sync - graph-safe. Verified 2026-05-07 across Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 (all +193%-234% decode vs `--no-cuda-graphs`). GGUF MoE prefill paths still use D2H sync, but prefill isn't graph-captured anyway. Hybrid Mamba2 (Nemotron-H) does NOT benefit yet - SSM layers don't fast-path.

- **Lever 1 SSM dispatch (commit `5b2c5db`).** Registered `ssm_in`/`ssm_out` in `cutlass_nvfp4_cache` so GDN/SSM weights hit the fast NVFP4 GEMM path. +95-376% decode on Qwen3.5/3.6 GDN families (2026-05-04): gain came from CUDA Graph capture enabled by faster GEMM. Always re-bench graphs ON after hot-path kernel change.

---

## Load-bearing root-cause fixes (don't regress these)

| Fix | Symptom if regressed | Where |
|-----|----------------------|-------|
| **FP8 FMHA S_tile pointer advance** | Long-context cliff at prompt > 1024 tokens | `attention_fmha_sm120.cu` - pointer must advance with `sizeof(half)`. Regression test in tree. |
| **Qwen3.5/3.6 GDN `__launch_bounds__(HD,1)` not `(HD,2)`** | HD=128 GDN miscompile, garbage output | GDN kernel - keep `(HD,1)`. |
| **Qwen3.5 partial RoPE pair offset `+ rope_pairs` (not `+ head_dim/2`)** | Sister bug to launch_bounds; partial-RoPE corruption | RoPE kernel. |
| **Qwen3.5 Q8 α/β qtype consistency** | Pre-dequanted Q8→FP16 without updating qtype → dispatcher mis-interprets bytes → state collapse | `upload_weight` path - keep qtype tag in sync with stored bytes. |
| **Qwen 3.6 h_state precision + PyTorch L2 norm** | NaN at L38 in GDN | The old "h_state must be FP32" note was a LAYOUT constraint, not numerics: the scan kernels assumed 4 B/element against a pool allocated at dtype size, and the NaN has the state-region-overflow signature. Since #1776/#1778 `gdn.state_bf16` (default ON) stores h_state as BF16 with FP32 register arithmetic: scan 2.04x isolated, +12.5% aggregate at 32 streams, PPL +0.21%. **FP16 state stays refuted** (~6e-5 subnormal truncation). The FP32 scan kernel itself measured 1527 GB/s isolated = this box's resident ceiling, so bytes were the only lever there. |
| **Gemma-4 per-layer `rope_freqs` for non-SWA layers, `n_rot=hd`** | L13/L14 drift 11-15% (was) → <2% (fixed) | Pass per-layer rope_freqs through. |
| **MoE expert-offload auto-probe at 10% before falling back to 30%** | Qwen3-Coder-30B Q6_K decode 234 → 77 tok/s | MoE offload path (`src/exec/executor_forward_moe*.cu` / `expert_cache.cu`, config `moe.expert_overhead_pct=10`) - keep the 10% probe. |
| **L2 access-policy window `num_bytes` clamp to `cudaDevAttrMaxAccessPolicyWindowSize`** | Silent CUDA error / IMA on 5090 (128 MiB max) | `set_l2_streaming` / `set_l2_persist_kv` in `runtime/`. |
| **NVFP4 dequant graph-safe fallback (PR #121)** | `cudaMallocAsync` inside captured graph crash | `set_nvfp4_dequant_workspace()` + capture-guard in `ensure_dequant_buffer`. |
| **Weight caches built BEFORE the KV pool (#1103 / PR #1106)** | Card ends at ~0 MiB free → WSL2/WDDM spills into host memory → ~7× decode collapse with no error (gpt-oss-20b-mxfp4 55 vs 331-359 tok/s). Nothing fails; bandwidth just drops ~1530 → ~237 GB/s | `src/runtime/engine_kv_cache_init.cpp` - caches (bounded by the model) first, KV pool takes the **measured** residual. Sizing KV from an *estimate* of cache demand is what broke it. |
| **No D2H of MoE host-args under graph capture (PR #859)** | IMA on capture; WSL2 compute-sanitizer can't diagnose it | Hybrid-capture foundation - MoE args must stay device-side in captured regions. Sister bug: da_cache stack-UAF (PR #861). |
| **Deterministic-GEMM cuBLAS algo is warmup-validated (PR #929)** | Intermittent `status 14` mid-run; void-GEMM continues on garbage | Det path must not take `results[0]` blindly; total algo failure THROWS. |
| **Explicit MXFP4→FP16 decode-fallback VRAM reserve on GDN hybrids (PR #935)** | token-0 `!` garbage (silent alloc failure) | VRAM planner reserves the fallback up front + fail-loud - see `quant-formats`. |

---

## Performance-relevant scaling rules

| Rule | Source |
|------|--------|
| Decode at batch=1: launch overhead first, memory second (post Lever 1) | Three Laws #1 in main SKILL.md |
| **Batched decode (M<=32) is its own regime**: grid-shape/launch levers that are refuted at batch=1 PAID there three times in one wave - row-block RMSNorm +6.8% (#1769), shared-activation quantize +4.6% (#1771), producer-side quantize fusion +2.6% (#1773). The GDN-gated fusion half measured NEUTRAL +0.4% (#1774, closed unmerged): the class left after #1773 is under the noise floor. | 32-stream A/Bs 2026-08-25/26, `docs/plans/2026-08-24-qwen38-port.md` |
| `__launch_bounds__` cost on regular paths: -4.5% to -20% | Repeated benchmarks 2026-04 to 2026-05 |
| `mxf4nvf4.block_scale` raw MMA: 2.60× over f8f6f4 | `mxf4nvf4_mma_bench` 2026-04-25 |
| CUDA Graph decode on prequant NVFP4 MoE: +193% to +234% | Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 - verified 2026-05-07 |
| pp512 spread across process starts: **model-dependent**, 0.6-1.2 % on Qwen3-8B Q8_0 vs **37.6 %** on a resident NVFP4 MoE model (cuBLAS algo re-timing itself: 3.50 %; the old "2.6× cuBLAS" figure was retracted 2026-08-03) | Use `tg256` for A/B; ≤5% prefill-kernel deltas need nsys per-kernel sums, not end-to-end pp (PR #648) - see `benchmark-cuda` skill |
| FP4 `mma.sync` measured peak ≈ 2,019 TOPS (~½ datasheet); f32-accumulate = ¼ rate | TC-rate calibration 2026-06-07 (#595/#596) |

---

## Negative results (don't repeat)

- **Generic `compute_120` PTX fallback.** Lacks FP8 MMA + block-scale. Always pin `compute_120a/sm_120a`.
- **FP8×FP8 cuBLAS prefill on sm_120.** Disabled by default since 2026-05-28: cuBLAS FP8 returns `NOT_SUPPORTED` at non-aligned M on consumer Blackwell (`engine_init_resolver.cpp`, config `attention.fp8_prefill`). Prefill levers are the FA2 family instead.
- **NVFP4 on GDN in/out projections.** REGRESSES −9 to −20% on wide shapes; FP16 wins. Shipped answer: byte-aligned `gemm.fp8_ssm_proj` sidecar (+19% #949, +21% #962 GGUF-Q8_0). `gemm.nvfp4_ssm_proj` GGUF opt-in removed 2026-07-11 (bit-rotted to 71 tok/s). Exception: `gemm.nvfp4_attn_proj` remains opt-in.
- **Occupancy raise / KPAR→MR reroute on the NVFP4 decode GEMV path.** Refuted by the 2026-05-30 nsys+ncu roofline sweep - decode plateau is a 4-bit-dequant co-limit (L1TEX 91%), not occupancy.
- **Replacing M=1 decode GEMV with smallm v2 pipeline kernel (2026-08-27).** Isolated A/B (4-copy weight ring, `SmallMV2Pair.DISABLED_M1PipelineVsGemvBench`) has GEMV and v2 inside round spread on all six Qwen3.8 shapes. In-situ switch pays per-projection activation quantize + W4A16->W4A4 numerics change. M=1 GEMV family stands at weight-bandwidth ceiling. (Bench un-defeated: >1792 GB/s, all weight fits 96 MB L2.)
- **Batch-1 MoE decode GEMV beyond 30% roofline.** Structural (#600/PR #642): shallow grids (1.5-2 waves) + tiny K (1.5-4 loads/lane); occupancy is already HIGHER than dense. `moe.mr_nr` is saturated - NR=4 +0.9%, NR≥16 regresses. Don't re-pursue.
- **MoE grouped-GEMM (NVFP4 prefill) beyond 41% roofline.** Structural (#601/PR #644): grid=170, 23% occupancy, M≈32. `moe.nvfp4_smallM` REGRESSES vs device-args default (+25-32%) - keep OFF. (Name collision: `gemm.nvfp4_smallm` is dense batched-decode #1766, default ON, do not confuse.)
- **Q8-IMMA occupancy/fetch tuning beyond #617.** Three refuted attempts documented in PR #618's tuning ladder. Also: per-launch **workspace memos poison IMMA perf**, and f32-accumulate quarters the TC rate.
- **Re-enabling tiled SWA at hd=256 (gemma-3).** Tiled kernels support hd=256 but gemma-3-on-cuBLAS is a deliberate SWA-correctness anchor (#566: tiled hd256+window PPL 42 vs 1.0). Blocked on #566, not wiring (#603/PR #645).
- **FA2 occupancy work without smem surgery (Bq=64 etc.).** Post-#609 FA2 is tensor-pipe-busiest and smem-capped at 16.7% occupancy (#597/PR #643); the shipped levers are `fa2_f16acc` and the Bkv=32 underfill variant (PR #648).
- **Split-D warp-pairing for HD=256 FA2 instance (stage-2 of #930 port).** Two warps per 16-row tile: a_frag/O regs halve (228->138), warps/SM double (4->8). Measured 2026-07-09 (Sq 512/1024/2048/4096, 8Q/2KV): **slower EVERYWHERE, +10-16%**, including grid-underfill (64 CTAs on 170 SMs). Stage-1 4-warp/228-reg instance not latency-limited: split-D pays replicated softmax (2x/CTA), smem exchange, halved MMA chain ILP. Occupancy raises are dead: remaining levers are algorithmic (Bkv tiling, smem layout).
- **FP4-precision attention - the whole family is CLOSED (2026-07-04).** Three independent refutations: NVFP4-attention spike (PR #868), ThriftAttention promotion gate (PRs #870/#871), paged FP4-QK KV-append quant (PR #872). Attention stays FP16/FP8; don't reopen without new hardware. (Related: hd=128 NVFP4 A/Bs need `attention.fa2_fp16qk=never` or you measure the wrong path.)
- **Forcing occupancy on batched spec-verify GEMV `gemv_nvfp4_kpar_mb_fp16_kernel` (2026-08-19).** Cost per row jumps at MR>=3: ptxas 40 regs (MR 1/2, 1444-1508 GB/s) vs 48-53 (MR 3/4, 1045-885 GB/s). Both obvious fixes worse: (a) Drop `__launch_bounds__` = 12.60/12.55 us vs 12.72 before (drift). (b) Pin `__launch_bounds__(kKparThreads, 12)` spills 16 B (MR 3) or 40 B (MR 4), MR=4 goes 14.5->26.8 us (-46% BW). Shipped `__launch_bounds__(kKparThreads)` no min-blocks is right point: register pressure real, occupancy spill cost multiples of return.
- **KPAR-GEMV paired-microblocks tuning (2026-07-07).** PDL (programmatic dependent launch) already overlaps the grid-end/launch latency the pairing tried to hide - measured ±0. Don't re-derive.
- **Launch-elimination levers on graphs+PDL decode loop (2026-07-13) - the whole class.** Roofline lever list from `--no-cuda-graphs`; under conditional-graph+PDL the launch/latency-class instances largely overlap away (Qwen3-30B no-graphs sum ~1.8× real graphs-ON step). Two refutations: (a) fused gate-GEMV+top-k (−2 launches/layer, 6.9% topk_gating at 40 GB/s) **0% e2e**; (b) split-K cap reduce REGRESSED −21...−35% (Q4KM 324→211 at cap=1). Scope: batch=1 verdict - batched decode (M<=32) pays for launch/grid-shape. Block-target sweep ctx 8k (340/512/680 blocks, 85/128/170 splits for 2/3/4 waves/SM) fell 317.25 / 308.15 / 302.32 tok/s (-2.87%, -4.71%, 2026-08-14). Tried after split-K reduce 21.9% faster (#1420). MR=4 spill 40.2 us path. Kernel at 31% peak BW is not bandwidth-bound: FP8 decode critical path softmax + two `__syncthreads` per 16-token. Rule: decode lever must hold real BYTES or critical-path math - validate graphs-ON e2e A/B.
- **C++23 `[[assume]]` in NVFP4 GEMV (2026-07-08).** Byte-identical SASS = provably inert. General rule: **SASS-diff (`cuobjdump -sass`) before any "perf-neutral" or "should help" claim** on compiler-hint changes - it settles the question in seconds.
- **Async `wgmma` / `tcgen05` / TMEM on consumer Blackwell.** Not available - SM100 (B200) exclusives. sm_120 peak path is register `mma.sync`. (Note: the *synchronous* `nvcuda::wmma` API *does* compile on sm_120 but lowers to **HMMA** - it is not async wgmma and not the peak path; it costs extra smem traffic and a smem round-trip vs hand-written `mma.sync` with register-resident fragments.)
- **Materializing the attention score tile (S/P) in shared memory.** Becomes barrier/L1-TEX-bound (compute util in the teens): smem round-trip + `__syncthreads` dominate. True FA2 keeps row max/sum and S/P fragments register-resident and fuses softmax into QK->PV handoff. Don't trust kernel headers claiming register-based softmax - verify against code (in-tree kernels mislabeled).
- **`__noinline__` on device inner-loop helpers.** Spills to Local Memory (DRAM). Use `__forceinline__`.
- **`reinterpret_cast` on Q8_0 blocks.** 34-byte blocks NOT 4-aligned. Use `memcpy()`.
- **Skipping graph re-bench after a hot-path patch.** Compute speedup alone often shows ~0% in tok/s - the win is graph-replay-mediated. Always re-bench graphs ON.
- **Increasing SMEM beyond `cudaDeviceProp::sharedMemPerBlockOptin`** assuming H100's 228 KB. RTX 5090 max is ~99 KB.

## NVFP4 paged decode attention: lever was LOAD WIDTH, not traffic (2026-08-30, #1817)

Previous attempts lost (note in `attention_paged_nvfp4.cu`): GQA tile -9%, smem double-buffer -3% (2026-05-08). At HD=256, inner loop read `k_bytes[i]` one byte at a time (`uint8_t*` unprovable alignment): `cuobjdump -sass` showed **20 `LDG.E.U8` per iteration** vs FP8's `uint32_t`. Fix: read one word per operand, issue K/V before warp reduction, left **2 `LDG.E.U8`** (scale bytes), 56 registers, zero spills: **64.0 -> 74.1 tok/s (+15.7%)** on Qwen3.8-27B-NVFP4 at 77k context, FP8 control 72.3/72.4.

Generalizations: (a) On latency-bound kernels count instructions before bytes; both refuted variants attacked L2-served traffic. (b) Byte-pointer inner loops are a defect class: `const uint8_t*` element-wise is N loads until alignment proven. Quantized KV paths hide them. (c) `cuobjdump -sass` finds the lever, not nsys.

## NVFP4 paged decode attention: GQA-tile sharing REFUTED (2026-08-26)

Scalar NVFP4 kernel re-reads KV per Q head (6x at 24Q/4KV), profiles ~13x above DRAM floor, a classic GQA-sharing lever. FP16/FP8 twins carry the variant. Built for NVFP4 (`perf/nvfp4-gqa-decode`, shared-FP16-tile per seq/kv_head): **-9% e2e** at 32-stream serving (9/9 waves, 3 trials/arm). Cause: one layer's KV ~42 MB fits 96 MB L2, re-reads are L2 hits, tile removes DRAM traffic but costs occupancy (64 KiB smem = 1 block/SM vs scalar's 768 blocks). Rule: check if shared working set fits L2 before building traffic variant - "x-over-DRAM-floor" not headroom when L2-served. `kv_cache.bitdecoding_qk` (NVFP4 TC QK) -5%; both defaults off.

## Ledger additions 2026-08-27 .. 2026-09-02 (verdict + number; records in `docs/roadmap.md` and `docs/plans/`)

| Lever | Verdict | Evidence |
|---|---|---|
| Residual-accumulate into smallm epilog (o/down/gdn-out beta=1) | REFUTED -0.9% median, 3/3 pairs negative (#1793) | Side fix shipped: `ctx.beta` reaches CUTLASS registry args |
| smallm v2 at M=1 | REFUTED (#1789) | All 6 Qwen3.8 shapes within round spread; batch=1 stays on GEMV family |
| smallm v2 at 33..64 rows as two 32-row launches (instead of the CUTLASS 128-row tile) | REFUTED 2026-09-12, reverted | N <= 8704 neutral at 24-stream verify and 40-stream decode; all shapes -11..-13 % (gate\|up: one CUTLASS sweep on 136 CTAs beats two small-M sweeps). Record `docs/plans/2026-09-11-batched-mtp-verify.md` |
| Prefill parallel to decode (second workspace) | NEUTRAL, default off (#1792) | short prompts 1771 -> 1778, heavy ingest 790 -> 791, TTFT unchanged; no SM partitioning on sm_120 (green contexts dead), streams displace each other |
| One-H2D decode-step staging (pinned mirror of the batch pool + sampler args) | NEUTRAL, closed unmerged (#1834) | 32-stream pairs -0.2/-1.1/+0.6%; the 8-14 us H2D gaps overlap host work |
| PDL device half (`griddepcontrol.wait` + `launch_dependents`, ~45 kernels) | SHIPPED (#1833) | M=1 spec-off +1.7% (3/3), 32 streams +0.5..1.3%, idle 13.6 -> 10.8%. Blanket registrations without a wait RACED `GreedyDeterminism`: registered = waits |
| Batched ban + penalty rows | SHIPPED +0.5% median (#1832) | server default `repetition_penalty=1.05` + 19 banned tokens put every row on the inline chain (2 launches/row/step) |
| MTP multi-candidate verify W=2 on GDN hybrid | default off (#1829/#1830) | Ceiling +6..+10 points top-2. Cost: +5-8% per verify, +11-20% verify time. Margin gate -0.8/-5.8% think traffic. Mitigation: `speculative.verify_smallm` adds +11% |
| CUTLASS NVFP4 prefill GEMM stream-K (`gemm.nvfp4_cutlass_streamk`) | SHIPPED (#1841) | pp512 101.1/103.6/102.1 -> 97.1/97.1/97.3 ms (3/3), pp4096 flat, bit-identical. Max-shape workspace = 0 B refused every launch (22k -> 4.2k tok/s; `CutlassWorkspaceContract.MaxShapeSizingCoversEverySmallerCall`). Forced SK DP bit-exactly so `max_rel=0` proves nothing, probe `gemm_nvfp4_cutlass_sm120_streamk_units()`. SK DP mode slower (109 vs 100 us at 640 CTAs). 128x64 vs 128x128 at N=5120: 37 vs 30 us |
| MoE grouped GEMM tile sweep / v2 grouped / multi-tile CTA | CLASS CLOSED at ~60% of the weight floor (#1842/#1846) | 128x64 +18/+32%, pingpong 128x128 +220%; 32-row v2 grouped -12.5% isolated but +3.5/+15% in situ (routing skew re-streams expert rows per small tile); multi-tile mt64 flat, mt128 +2.4..4.1%. Builder rejects M=64 tiles (SF atom 128 rows). Benches `tests/test_cutlass_grouped_tile_bench.cu`; branches `perf/moe-smallm-v2-grouped`, `perf/moe-grouped-multitile` |
| FP8 GDN-projection prefill (`SSM_IN` + `SSM_OUT`) | REFUTED e2e (#1837/#1845), record `docs/plans/2026-08-31-fp8-ssm-prefill.md` | cuBLASLt FP8 2.0-3.6x vs FP16, class 12-13% steady-state (ceiling +5%). SSM_IN `out / row_scale` in FP16 = inf on small-absmax rows (fixed on `perf/fp8-ssm-prefill-v2`, `gemm_fp8_rowscaled`). cuBLASLt `SCALE_OUTER_VEC_32F` applies `scale[n & ~1]` on sm_120/13.3 |
| Chunk-parallel GDN prefill scan | SHIPPED (#1847-#1852) | pp4096 Qwen3.6-35B 12.9k -> 31.0k tok/s; Qwen3.8-27B 10.0k -> 11.7k. Refuted: solve histories global (379 us), scalar smem FMA K2 (628 us), K2 accumulator splits (+22..28%), tf32 state path (3.4e-4 diff) and P@W (cancellation), K2 2 CTAs/SM (flat, +8%), L2 > 2/3 (K1 -21%, K2 +11..13%), swz64 K1 T/P tiles (+10%), fp16 Y_A (+5%), operand-split (-1%) |
| Ragged chunk-parallel scan for serving prefill | REFUTED before building | Ragged fused scans = 2.6% of 32x1000-token burst; `rows_left = min(1024, max_tokens)` makes most forwards single-sequence where chunkpar already runs |
| Dense FA2 (hd=128, Bq=128) two CTAs/SM (`attention.fa2_dense_2cta`) | SHIPPED (#1843) | pp4096 kernel sum 271.7 -> 244.2 ms (-10%, 3/3), bit-identical. `__launch_bounds__(256,2)` on wrapper (137->128 regs, 24 B spill); `(256,1)` on shipped NOT neutral (137->180 regs). #1838 had compared 8 vs 8 warps |
| FA2 softmax: scale inside exp FMA + CTA-uniform interior-tile mask skip | SHIPPED (#1844) | Kernel -7.5% (14B) / -14% (Qwen3.8 hd=256); PPL 10.0277->10.0229 / 4.6283. Refuted: `exp2f` +0.35% PPL, scale*log2e +0.37%, QK MMA interleave = noise |
| HD=256 FA2 at 2 CTAs/SM (`attention.fa2_hd256_bkv=32`) | opt-in only (#1840) | kernel -11.2% (3/3), e2e +0.1..0.4%, PPL 4.6283 -> 4.6529 (+0.53%, doubled f16 O rescales); f32-PV holds PPL but runs at Bkv=64 time. Harness `tools/analysis/fa2_hd256_bkv_ab.sh` |
| Roofline "255 registers" on the dense FA2 instance | CORRECTED | 255 = `device__attribute_max_registers_per_thread`; the kernel allocates 144, reg AND smem limited (70656 of 102400 B) |
| Roofline launch at 0.31 GHz read 99.7% | FIXED in aggregation | `ncu.clock_floor_ghz=1.2` drops it (`n_launches_dropped_clock`); `--clock-control base` does not stop idle downclock between replays |
| Sparse decode attention on NVFP4 KV | SHIPPED (#1818) | 74.3 -> 100.2 tok/s at 77k. NVFP4 branches ignored compacted block table. Budget arithmetic used `kKVBlockSize=16` vs `n_kv_heads <= 4` 32-token blocks; resolved before executor sizing (#1819) |
| Ragged prefill launch floor charged per member | FIXED (#1781) | 256-token floor once per ragged forward: 975.9 -> 1001.3 tok/s at 32 streams |

Kernel-name note: the batched spec-verify GEMV referenced above as `gemv_nvfp4_kpar_mb_fp16` is `gemv_nvfp4_kpar_mb_fp16_kernel` in the tree (`gemv_kpar_mb<3>` in profiles).

# sm_120a Known Issues, Dead Ends, Root-Cause Reference

Heavy reference for the `sm120-cuda-expert` skill. Things that don't work, things that *used* to not work but now do, and historical bugs whose fixes are load-bearing for current hot-path code.

---

## Pre-flight before non-trivial kernel work

1. Check the dead-ends list below — many "obvious" optimizations are proven failures on sm_120.
2. If the installed CUDA version differs from when a dead end was tested (check `nvcc --version`), see "Version-dependent dead ends" — some are worth retrying.

For small edits (parameter tweak, kernel-signature change, fusing two existing kernels), skip pre-flight and go straight to the patterns.

---

## Version-dependent dead ends (worth retrying when CUDA version changes)

> **CUDA 13.3 re-test (2026-05-29, PTX ISA 9.3): NO new sm_120a capability.** Full
> `ptx_survey_all.sh` at `compute_120a` under 13.2 vs 13.3 = **0 of 247 instructions
> flipped** (none unlocked, none regressed). The "retry on CUDA 13.3+" rows below were
> re-probed and stay ❌. sm_120's ISA surface is silicon-fixed; toolkit bumps don't add
> tcgen05/wgmma/TMA. Baselines: the two `docs/ptx-status-*-sm120a.md` snapshots were consolidated away in #805 — regenerate with `tools/analysis/ptx_survey_all.sh` rather than looking for them.
> 13.3's value is tooling (CUDA Tile C++, CompileIQ) + cuBLAS perf, not instructions.

| Dead end | Blocked by | Retry on |
|----------|-----------|----------|
| cuBLASLt grouped layout sm_120 | Zero algorithms for consumer Blackwell | New cuBLAS release (check algorithm count) |
| CUTLASS TC GEMM at M=1 | Activation quant + TMA overhead | ~~CUTLASS 4.5+~~ pin has moved (read `cmake/imp-deps.cmake`, v4.7.0 as of 2026-08) - never re-probed, and now largely MOOT: the M<=32 gap was closed by the in-tree smallm v2 kernel (#1766), not CUTLASS. M=1 decode GEMV stays at its measured 66-70% HBM ceiling (4-bit-dequant co-limit). |
| `cp.async.bulk` with `.ignore_oob` | Requires TMA descriptor rewrite | ~~CUDA 13.3+~~ still ❌ on 13.3 — TMA not on sm_120; next major |
| `st.async .b128` to global | PTX 9.2 only targets `shared::cluster` | ~~New PTX ISA~~ still ❌ on PTX ISA 9.3 (13.3) |
| CUTLASS NVFP4 sm_120 graph-determinism | Universally non-deterministic for `cudaGraphExecUpdate` re-capture (verified 2026-05-05) | Future CUTLASS NVFP4 deterministic mode |
| ~~Native FP4 GEMM faster than dequant→cuBLAS on sm_120~~ | **RESOLVED 2026-08-25 (#1766):** the retry condition ("future custom kernel") was met in-tree - see "Resolved" below. | - |

---

## Resolved (no longer dead ends)

- **Native FP4 GEMM slower than dequant paths at small M - inverted 2026-08-25 (#1766).**
  `src/quant/nvfp4_gemm_smallm_v2.cu` runs `mma.sync.kind::mxf4nvf4.block_scale` on the
  PLAIN packed layout (same weight bytes as the M=1 GEMVs, zero extra VRAM; SF/fragment
  mappings from CUTLASS `SM120_16x8x64_TN_VS` traits; M32xN64xK256 tile, 6-deep smem ring,
  1 producer + 4 consumer warps). Isolated at M=32 N=5120 K=5120: **10.4 us** vs CUTLASS
  41.4 in-situ, v1 W4A16-dequant+HMMA 23.9, Marlin sidecar 14.4, weight floor 8.2.
  E2E on Qwen3.8-27B-NVFP4: +16.0% aggregate at 32 streams, +36.0% at 8. Default ON via
  `gemm.nvfp4_smallm` / `nvfp4_smallm_impl=2` (`src/core/config/gemm.h`). Two verdicts on
  the way there: the **Marlin W4A16 sidecar was built and REJECTED** (#1756/#1757, PR #1764
  closed unmerged: needs a repacked second weight copy, capped at 13% coverage on the 27B)
  and the **in-situ stage/stripe retune is REFUTED** (#1768: shipped stages=6/stripes=1
  beats every override by 7-13% in the real 32-stream step; the v1 "isolated optimum !=
  step optimum" lesson does NOT repeat on v2). Remaining in-class residual is bounded:
  per-shape 68-86% of DRAM floor, perfect-floor bound ~237 us/token vs measured 388.9
  (`docs/plans/2026-08-24-qwen38-port.md`), not reachable through grid geometry.

- **PTX `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}`** and reverse direction work on both `sm_120f` and `sm_120a` under CUDA 13.2.1 (re-verified 2026-05-04). Correct usage routes the FP4 byte through a `.b8` register — see `references/ptx-patterns.md` "FP4 ↔ FP16/FP32/BF16 packed conversion". SASS confirms hardware emission: `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

- **Build target `sm_120a`** (was historically blocked by a `ptxas` C7600 bug on `120f` that needed the `f` workaround). As of CUDA 13.2.1 the `a` arch suffix is the correct target — superset of `120f`, adds `mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM. Switched 2026-05-04 (commit `6568652`).

- **CUDA Graphs + prequant-NVFP4 MoE.** Earlier "non-Gemma-4 MoE blocks graph capture" claim was stale. The MoE decode fast-path (`executor_forward_moe.cu`, the `n=1, device-resident packed experts` branch) is fully device-side, no D2H sync — graph-safe. Verified 2026-05-07 across Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 (all +193%–234% decode vs `--no-cuda-graphs`). GGUF MoE prefill paths still use D2H sync, but prefill isn't graph-captured anyway. Hybrid Mamba2 (Nemotron-H) does NOT benefit yet — SSM layers don't fast-path.

- **Lever 1 SSM dispatch (commit `5b2c5db`).** Registered `ssm_in`/`ssm_out` in the `cutlass_nvfp4_cache` so GDN/SSM weights hit the fast NVFP4 GEMM path. Showed +95–376% decode on Qwen3.5/3.6 GDN families on 2026-05-04 — but the gain came from CUDA Graph capture *enabled by* the faster GEMM, not the GEMM speedup itself. **Always re-bench graphs ON after a hot-path kernel change.**

---

## Load-bearing root-cause fixes (don't regress these)

These bugs were diagnosed at high cost. The current kernels assume the fix is in place.

| Fix | Symptom if regressed | Where |
|-----|----------------------|-------|
| **FP8 FMHA S_tile pointer advance** | Long-context cliff at prompt > 1024 tokens | `attention_fmha_sm120.cu` — pointer must advance with `sizeof(half)`. Regression test in tree. |
| **Qwen3.5/3.6 GDN `__launch_bounds__(HD,1)` not `(HD,2)`** | HD=128 GDN miscompile, garbage output | GDN kernel — keep `(HD,1)`. |
| **Qwen3.5 partial RoPE pair offset `+ rope_pairs` (not `+ head_dim/2`)** | Sister bug to launch_bounds; partial-RoPE corruption | RoPE kernel. |
| **Qwen3.5 Q8 α/β qtype consistency** | Pre-dequanted Q8→FP16 without updating qtype → dispatcher mis-interprets bytes → state collapse | `upload_weight` path — keep qtype tag in sync with stored bytes. |
| **Qwen 3.6 h_state precision + PyTorch L2 norm** | NaN at L38 in GDN | The old "h_state must be FP32" note was a LAYOUT constraint, not numerics: the scan kernels assumed 4 B/element against a pool allocated at dtype size, and the NaN has the state-region-overflow signature. Since #1776/#1778 `gdn.state_bf16` (default ON) stores h_state as BF16 with FP32 register arithmetic: scan 2.04x isolated, +12.5% aggregate at 32 streams, PPL +0.21%. **FP16 state stays refuted** (~6e-5 subnormal truncation). The FP32 scan kernel itself measured 1527 GB/s isolated = this box's resident ceiling, so bytes were the only lever there. |
| **Gemma-4 per-layer `rope_freqs` for non-SWA layers, `n_rot=hd`** | L13/L14 drift 11–15% (was) → <2% (fixed) | Pass per-layer rope_freqs through. |
| **MoE expert-offload auto-probe at 10% before falling back to 30%** | Qwen3-Coder-30B Q6_K decode 234 → 77 tok/s | MoE offload path (`src/exec/executor_forward_moe*.cu` / `expert_cache.cu`, config `moe.expert_overhead_pct=10`) — keep the 10% probe. |
| **L2 access-policy window `num_bytes` clamp to `cudaDevAttrMaxAccessPolicyWindowSize`** | Silent CUDA error / IMA on 5090 (128 MiB max) | `set_l2_streaming` / `set_l2_persist_kv` in `runtime/`. |
| **NVFP4 dequant graph-safe fallback (PR #121)** | `cudaMallocAsync` inside captured graph crash | `set_nvfp4_dequant_workspace()` + capture-guard in `ensure_dequant_buffer`. |
| **Weight caches built BEFORE the KV pool (#1103 / PR #1106)** | Card ends at ~0 MiB free → WSL2/WDDM spills into host memory → ~7× decode collapse with no error (gpt-oss-20b-mxfp4 55 vs 331–359 tok/s). Nothing fails; bandwidth just drops ~1530 → ~237 GB/s | `src/runtime/engine_kv_cache_init.cpp` — caches (bounded by the model) first, KV pool takes the **measured** residual. Sizing KV from an *estimate* of cache demand is what broke it. |
| **No D2H of MoE host-args under graph capture (PR #859)** | IMA on capture; WSL2 compute-sanitizer can't diagnose it | Hybrid-capture foundation — MoE args must stay device-side in captured regions. Sister bug: da_cache stack-UAF (PR #861). |
| **Deterministic-GEMM cuBLAS algo is warmup-validated (PR #929)** | Intermittent `status 14` mid-run; void-GEMM continues on garbage | Det path must not take `results[0]` blindly; total algo failure THROWS. |
| **Explicit MXFP4→FP16 decode-fallback VRAM reserve on GDN hybrids (PR #935)** | token-0 `!` garbage (silent alloc failure) | VRAM planner reserves the fallback up front + fail-loud — see `quant-formats`. |

---

## Performance-relevant scaling rules

| Rule | Source |
|------|--------|
| Decode at batch=1: launch overhead first, memory second (post Lever 1) | Three Laws #1 in main SKILL.md |
| **Batched decode (M<=32) is its own regime**: grid-shape/launch levers that are refuted at batch=1 PAID there three times in one wave - row-block RMSNorm +6.8% (#1769), shared-activation quantize +4.6% (#1771), producer-side quantize fusion +2.6% (#1773). The GDN-gated fusion half measured NEUTRAL +0.4% (#1774, closed unmerged): the class left after #1773 is under the noise floor. | 32-stream A/Bs 2026-08-25/26, `docs/plans/2026-08-24-qwen38-port.md` |
| `__launch_bounds__` cost on regular paths: -4.5% to -20% | Repeated benchmarks 2026-04 to 2026-05 |
| `mxf4nvf4.block_scale` raw MMA: 2.60× over f8f6f4 | `mxf4nvf4_mma_bench` 2026-04-25 |
| CUDA Graph decode on prequant NVFP4 MoE: +193% to +234% | Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 — verified 2026-05-07 |
| pp512 spread across process starts: **model-dependent**, 0.6-1.2 % on Qwen3-8B Q8_0 vs **37.6 %** on a resident NVFP4 MoE model (cuBLAS algo re-timing itself: 3.50 %; the old "2.6× cuBLAS" figure was retracted 2026-08-03) | Use `tg256` for A/B; ≤5% prefill-kernel deltas need nsys per-kernel sums, not end-to-end pp (PR #648) — see `benchmark-cuda` skill |
| FP4 `mma.sync` measured peak ≈ 2,019 TOPS (~½ datasheet); f32-accumulate = ¼ rate | TC-rate calibration 2026-06-07 (#595/#596) |

---

## Negative results (don't repeat)

- **Generic `compute_120` PTX fallback.** Lacks FP8 MMA + block-scale. Always pin `compute_120a/sm_120a`.
- **FP8×FP8 cuBLAS prefill on sm_120.** Disabled by default since 2026-05-28: cuBLAS FP8 returns `NOT_SUPPORTED` at non-aligned M on consumer Blackwell (`engine_init_resolver.cpp`, config `attention.fp8_prefill`). Prefill levers are the FA2 family instead.
- **NVFP4 on GDN in/out projections.** REGRESSES −9 to −20% on wide GDN shapes — FP16 wins there; the byte-aligned `gemm.fp8_ssm_proj` sidecar is the shipped answer (native +19% #949, GGUF-Q8_0 +21% #962). The old `gemm.nvfp4_ssm_proj` GGUF opt-in was removed 2026-07-11 (bit-rotted to 71 tok/s, superseded); `gemm.nvfp4_attn_proj` remains a measured opt-in exception.
- **Occupancy raise / KPAR→MR reroute on the NVFP4 decode GEMV path.** Refuted by the 2026-05-30 nsys+ncu roofline sweep — decode plateau is a 4-bit-dequant co-limit (L1TEX 91%), not occupancy.
- **Batch-1 MoE decode GEMV beyond 30% roofline.** Structural (#600/PR #642): shallow grids (1.5–2 waves) + tiny K (1.5–4 loads/lane); occupancy is already HIGHER than dense. `moe.mr_nr` is saturated — NR=4 +0.9%, NR≥16 regresses. Don't re-pursue.
- **MoE grouped-GEMM (NVFP4 prefill) beyond 41% roofline.** Structural (#601/PR #644): grid=170 (1 wave), 23% occupancy, per-expert M≈32. `moe.nvfp4_smallM` REGRESSES vs the device-args default (which is +25-32%) - keep it OFF. The NVFP4 prefill lever is attention, not grouped GEMM. **Name collision warning:** the differently-scoped `gemm.nvfp4_smallm` (dense batched-decode small-M kernel, #1766) is default ON and a WIN; do not confuse the two keys.
- **Q8-IMMA occupancy/fetch tuning beyond #617.** Three refuted attempts documented in PR #618's tuning ladder. Also: per-launch **workspace memos poison IMMA perf**, and f32-accumulate quarters the TC rate.
- **Re-enabling tiled SWA at hd=256 (gemma-3).** Tiled kernels DO support hd=256, but gemma-3-on-cuBLAS is a deliberate #566 SWA-correctness anchor (tiled hd256+window = PPL 42 vs 1.0). Blocked on #566, not wiring (#603/PR #645).
- **FA2 occupancy work without smem surgery (Bq=64 etc.).** Post-#609 FA2 is tensor-pipe-busiest and smem-capped at 16.7% occupancy (#597/PR #643); the shipped levers are `fa2_f16acc` and the Bkv=32 underfill variant (PR #648).
- **Split-D warp-pairing for the HD=256 FA2 instance (stage-2 of the #930 port).** Two warps per 16-row tile, each owning one D half: a_frag/O regs halve (228→138, zero spills), warps/SM double (4→8), the partial-S exchange even rides the existing TWOSLOT mid-loop barrier (zero extra syncs). Measured 2026-07-09 (Sq sweep 512/1024/2048/4096, 8Q/2KV): **slower EVERYWHERE, +10–16% — including the grid-underfill band** (64 CTAs on 170 SMs). The stage-1 4-warp/228-reg instance is not latency-limited: the long unrolled MMA chains with software-pipelined ldsm supply enough ILP per warp; split-D pays replicated softmax (×2/CTA), smem exchange traffic, and HALVED per-warp MMA chain length (less ILP) with no offsetting win. Occupancy raises on this kernel are dead — the remaining hd=256 levers are algorithmic (Bkv tiling, smem layout), not warp count.
- **FP4-precision attention — the whole family is CLOSED (2026-07-04).** Three independent refutations: NVFP4-attention spike (PR #868), ThriftAttention promotion gate (PRs #870/#871), paged FP4-QK KV-append quant (PR #872). Attention stays FP16/FP8; don't reopen without new hardware. (Related: hd=128 NVFP4 A/Bs need `attention.fa2_fp16qk=never` or you measure the wrong path.)
- **Forcing occupancy on the batched spec-verify GEMV `gemv_nvfp4_kpar_mb_fp16` (2026-08-19).** Its cost per row jumps at MR>=3, and the register counts look like the cause: ptxas gives 40 regs at MR=1/2 (12 blocks/SM) but 48-53 at MR=3/4 (9-10 blocks/SM), and measured weight bandwidth tracks it — 1444-1508 GB/s at MR=1/2 against 1045 and 885. **Both obvious fixes were measured and both are worse.** (a) Dropping `__launch_bounds__` clears MR=3's 4-byte spill (48 regs -> 52, zero spills) and buys **nothing**: 12.60 / 12.55 us against 12.72 before, inside the harness's own drift. (b) Pinning `__launch_bounds__(kKparThreads, 12)` like the sibling KPAR kernels does force 40 regs and 12 blocks — by spilling 16 bytes at MR=3 and **40 bytes at MR=4**, and it is a disaster: MR=4 goes **14.5 -> 26.8 us and 40.2 us in one round** (-46 % to -64 % bandwidth), MR=3 slightly worse too. So the shipped `__launch_bounds__(kKparThreads)` with no min-blocks is the right point on this curve: the register pressure is real, and paying for occupancy with spills costs multiples of what the occupancy returns. Same conclusion as the decode-path entry above, reached independently on the MR path.
- **KPAR-GEMV paired-microblocks tuning (2026-07-07).** PDL (programmatic dependent launch) already overlaps the grid-end/launch latency the pairing tried to hide — measured ±0. Don't re-derive.
- **Launch-elimination levers on the graphs+PDL decode loop (2026-07-13) - the whole class.** The roofline lever list is built from `--no-cuda-graphs` profiles; under the shipped conditional-graph+PDL decode the grid-(1,1,1)/launch-latency classes (moe_routing, rmsnorm, rope, kv_write, elementwise) largely overlap away - on Qwen3-30B-A3B the no-graphs kernel-time sum is ~1.8× the real graphs-ON step. Two direct refutations: (a) a multi-block fused gate-GEMV+top-k kernel (bit-identical outputs, −2 launches/layer, killed the 6.9%-share topk_gating launch) measured **0% e2e**; (b) capping decode split-K to kill the "5.8 µs reduce at 40 GB/s" REGRESSED −21…−35% (Q4KM 324→211 at cap=1) - split parallelism feeds 170 SMs from batch×heads=32; the auto policies are right even at ctx 512. **Scope: this whole entry is a batch=1 verdict - at batched decode (M<=32) the launch/grid-shape class pays; see the scaling-rules table above.** **RAISING it is refuted too (2026-08-14), so the policy is a ceiling as well as a floor:** exposing the block target as a knob and sweeping it at ctx 8k on Qwen3-Coder-30B-A3B gave 340/512/680 blocks (85/128/170 splits) for 2/3/4 waves per SM, and decode fell monotonically: 317.25 / 308.15 / 302.32 tok/s, i.e. **-2.87 % and -4.71 %**, losing in 5 of 5 rounds against arm spreads of 0.1-1.1 %. Tried AFTER the split-K reduce got 21.9 % faster (#1420), so the reduce is not what pays for it. **Reading the stalls matters here:** this kernel sits at 31 % of peak bandwidth with a long-scoreboard stall of only 1.64, while the lm_head GEMV reaches 93 % at 18.4. A LOW memory-stall ratio beside low achieved bandwidth does NOT mean "too few warps to fill the pipeline"; it means the kernel is not bandwidth-bound. At M=1 the FP8 decode, the softmax and two __syncthreads per 16-token chunk are the critical path, so extra block parallelism only adds per-split overhead. Rule: a decode lever must hold real BYTES or critical-path math (dtype shrink, fewer bytes moved, faster big-GEMV) - validate launch/latency-class ideas with a graphs-ON e2e A/B before writing any kernel.
- **C++23 `[[assume]]` in NVFP4 GEMV (2026-07-08).** Byte-identical SASS = provably inert. General rule: **SASS-diff (`cuobjdump -sass`) before any "perf-neutral" or "should help" claim** on compiler-hint changes — it settles the question in seconds.
- **Async `wgmma` / `tcgen05` / TMEM on consumer Blackwell.** Not available — SM100 (B200) exclusives. sm_120 peak path is register `mma.sync`. (Note: the *synchronous* `nvcuda::wmma` API *does* compile on sm_120 but lowers to **HMMA** — it is not async wgmma and not the peak path; it costs extra smem traffic and a smem round-trip vs hand-written `mma.sync` with register-resident fragments.)
- **Materializing the attention score tile (S/P) in shared memory.** A FA-style kernel that writes S to smem, runs softmax over smem, then reads P back for the PV MMA becomes **barrier- / L1-TEX-bound** (tensor cores idle, compute util in the teens) — the smem round-trip + `__syncthreads` dominate, not the MMAs. True FA2 keeps row max/sum and the S/P fragments **register-resident** and fuses softmax into the QK→PV handoff. Don't trust a kernel header that *claims* register-based softmax — verify against the code (some in-tree kernels are mislabeled).
- **`__noinline__` on device inner-loop helpers.** Spills to Local Memory (DRAM). Use `__forceinline__`.
- **`reinterpret_cast` on Q8_0 blocks.** 34-byte blocks NOT 4-aligned. Use `memcpy()`.
- **Skipping graph re-bench after a hot-path patch.** Compute speedup alone often shows ~0% in tok/s — the win is graph-replay-mediated. Always re-bench graphs ON.
- **Increasing SMEM beyond `cudaDeviceProp::sharedMemPerBlockOptin`** assuming H100's 228 KB. RTX 5090 max is ~99 KB.

## NVFP4 paged decode attention: GQA-tile sharing REFUTED (2026-08-26)

The scalar NVFP4 decode kernel re-reads each KV block once per Q head (6x at
24Q/4KV) and profiles ~13x above its per-launch DRAM floor — which reads like
the classic GQA-sharing lever, and the FP16/FP8 twins both carry the variant.
Built for NVFP4 (branch `perf/nvfp4-gqa-decode`, shared-FP16-tile block per
(seq, kv_head), numerically exact) and measured **-9% e2e** at 32-stream
serving (9/9 waves below, 3 alternating trials/arm). Mechanism: one layer's
KV across 32 seqs is ~42 MB against the 96 MB L2 — the re-reads are L2 hits,
the tile removes traffic that never reached DRAM, and it costs occupancy
(64 KiB smem = 1 block/SM; grid batch x n_kv_heads = 128 blocks on 170 SMs
vs the scalar's batch x n_heads = 768). Rule that generalizes: **before
building a traffic-sharing variant, check whether the shared working set
already fits L2** — "x-over-DRAM-floor" is not headroom when the traffic is
L2-served. `kv_cache.bitdecoding_qk` (NVFP4 TC QK) measured -5% on the same
harness; both defaults stay off by measurement.

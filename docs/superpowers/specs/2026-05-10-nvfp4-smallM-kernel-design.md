# NVFP4 Small-M Grouped GEMM Kernel — Design

**Status**: Phase 0 microbench done (commit `a591dac`); spec revised 2026-05-10 to drop the fused-TMA bandwidth assumption and reorient the validation around per-shape A/B testing + auto-heuristic.
**Author**: Raphael Friedmann
**Date**: 2026-05-10
**Branch**: perf/moe-nvfp4-prefill-fast-path
**Effort estimate**: 2-3 weeks (single-engineer)

## Goal

Close some of the remaining ~1.5-1.95× prefill gap between imp and vLLM on
NVFP4 MoE models, on RTX 5090 (sm_120a). Target: pp512 ≥ **19000 tok/s
median** on Qwen3-Coder-30B-A3B-NVFP4 (10 reps, cold-container), up from
current median 16474 tok/s (verified 2026-05-10, commit 33858da).

Original target was 22k tok/s assuming a +10-20% TMA-fusion bandwidth bonus
that Phase 0 microbench (commit `a591dac`) measured at 0% — see
"Phase 0 findings" below. The 19k target is the M-padding-only ceiling
without TMA bonuses, plus auto-heuristic to avoid CUTLASS-path regression
at M_e ≥ 64. **Above-19k results are bonus, not commitment.**

Decode performance must not regress (current 270 tok/s on Qwen3-Coder).

## Why a custom kernel — measurement-based justification

After commit `bc3bc31` routed NVFP4 prefill through CUTLASS Sm120 grouped GEMM
(pp512 1241 → 13046 tok/s, ×10.5), four follow-up CUTLASS-level optimizations
were attempted in iteration-2 and all rejected (smaller M-tile fails to compile,
larger N-tile loses parallelism, pingpong schedule blocked for FP4, fused
gate+up regressed decode -7%). See `docs/archive/bench-2026-05-10/iteration2_findings.md`.

Re-measured baseline 2026-05-10:

| Model | pp512 median | pp512 best | tg256 |
|---|---:|---:|---:|
| Qwen3-Coder-30B-A3B-NVFP4 | 16474 tok/s | 16740 | 270 |
| Qwen3-30B-A3B-NVFP4-Modelopt | 14483 | 16448 | 271 |
| Qwen3.6-35B-A3B-NVFP4 | 8833 | 9097 | 232 |
| Gemma-4-26B-A4B-it-NVFP4 | 8953 | 9610 | 212 |

Nsys profile (`nsys_post_bc3bc31_20260510_104438.nsys-rep`):

| Kernel | % | Calls | Median µs | Min/Max µs | StdDev |
|---|---:|---:|---:|---:|---:|
| **NVFP4 grouped GEMM** | **37.9%** | 1008 | **78** | 55/1117 | **70** |
| per-tensor NVFP4 attention CUTLASS | 6.1% | 1344 | 10 | 9/24 | 2.8 |
| paged_attention splitk | 5.2% | 768 | 15 | 14/473 | 21 |
| causal_softmax | 5.0% | 336 | 36 | 33/440 | 22 |

**The 20× spread (55-1117 µs) on the grouped GEMM is the smoking gun for M-padding
waste**: when M_e per expert is small (~32-48, typical for MoE with 128 experts ×
top-k=8 × 512 prefill tokens), the CUTLASS-pinned M-tile=128 wastes 60-75% of
tile-rows on padding. The current CUTLASS Sm120 NVFP4 grouped path cannot reduce
M-tile below 128 because the SfAtom block-scale layout requires it (see
iteration-2 attempt #1).

**Caveat — this is the *only* root cause we have measurement evidence for.**
Other plausible bottlenecks (instruction-cache pressure, inadequate SMEM swizzle,
schedule-overhead amortization across many small problems) cannot be ruled out
from the nsys data alone. The 20× spread is real but its translation into
predicted speedup depends on M-distribution per call. Conservative estimate
+15-30% pp; optimistic +50-70%; A/B test at the actual deployment shapes is
the only way to commit to a number.

SASS audit of imp:test confirms the build is already on the optimal MMA pipe:
- Symbol: `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` +
  `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120` +
  `SM120_16x8x64_TN_VS<float_e2m1_t, float_ue4m3_t>`
- 1898 HMMA.16816 instructions, 290 F2FP.SATFINITE.E2M1 (HW FP4 saturation),
  36 UTMALDG (Blackwell-Native TMA), 0 BMMA / 0 UMMA (correct — SM120 has no
  BMMA hardware; block-scale runs through `mma.sync.kind::mxf4nvf4` PTX which
  emits HMMA.16816 SASS with hardware-applied scales)
- Tile shape compiled-in: `<_128, _128, _128>`, Cooperative schedule

**MMA microbench validates the hardware peak**:

| Variant | TOPS | Status |
|---|---:|---|
| `mma.sync.kind::mxf4nvf4 vec::4X k64 ue4m3` | **268.05** | matches `sm120_mma_variants_2026_04_25` memory's 268 |
| same variant, ue8m0 scales | 269.41 | +0.5%, marginal |
| vec::2X k64 ue8m0 | 266.14 | within noise |

Theoretical NVFP4 peak on RTX 5090 (3354 TOPS) is unreachable on SM120 — that
peak requires `tcgen05.mma` which is SM100/B200-only hardware. **268 TOPS is the
real ceiling for warp-level mma.sync on Consumer Blackwell**, validated 2026-05-10.

Headroom analysis for Qwen3-Coder-30B-A3B at pp512:
- 3B active × 2 fma × 512 tokens = ~3 TFlops GEMM share per prefill
- At 268 TOPS, GEMM lower bound: 3e12 / 268e12 = 11.2 ms
- Plus ~10 ms non-GEMM = ~21 ms total → ceiling ~24k tok/s
- Current median 16k = ~65% of ceiling
- Custom kernel improving M-tile utilization 65% → 85% reaches ~22-24k → matches vLLM 25513

**The path is not chasing a phantom — the HW supports the goal.**

## Phase 0 findings (2026-05-10, commits `a591dac`)

**Block-scale-aware TMA microbench result (R-CHECK before kernel work):**

| variant | ms (1024 iters, RTX 5090) | speedup |
|---|---:|---:|
| separate (2 `cp.async.bulk.tensor`, mbarrier per-descriptor) | 2.293-2.415 | 1.0× |
| fused (1 `cp.async.bulk.tensor`, contiguous data+scale layout) | 2.292-2.415 | 0.95-1.05× |

**Verdict: fused-TMA-bandwidth claim REJECTED.** Speedup oscillates within
measurement noise (5%). Both variants saturate L2 at ~2.5 GB/s effective.
The spec previously claimed +10-20% from `sm120_real_perf_levers` memory;
that memory was speculation, not measurement. Cross-reference: CUTLASS's
own SM120 NVFP4 mainloop (`sm120_blockscaled_mma_tma.hpp:284-311`) uses two
separate TMA descriptors (TMA_A + TMA_SFA), not a fused descriptor — the
"fused descriptor" was never actually exposed at the hardware level.

**Implication for kernel design:**
- Use the **CUTLASS 2-descriptor TMA pattern** (TMA_A + TMA_SFA, separate `cp.async.bulk.tensor` issues + mbarrier per descriptor).
- The "Bonus-Feature" of native row-major scales (3 GiB VRAM save + skip `convert_scales_sfatom_moe_kernel`) **still applies** — that comes from the layout choice, not from descriptor fusion.
- Expected pp speedup downgraded from "+30-70%" to "+15-30%" under conservative scenario, "+30-50%" under optimistic. **Auto-heuristic + per-shape A/B is the binding validation**, not a single magic number.

## Architecture

Hand-rolled persistent grouped GEMM kernel, written in CUDA C++ with inline
PTX for the inner-loop MMA. FA2-style data path with TMA producer/consumer
warps and FP32 register accumulators.

### Inner-loop MMA

`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`

Identical to the MMA the existing CUTLASS path uses (verified via SASS audit).
Same 268 TOPS hardware ceiling. The custom kernel's win comes from
**eliminating M-tile padding waste**, not from a faster MMA variant.

### Data flow (FA2 + block-scaling)

```
Producer warps (4):  cp.async.bulk.tensor (TMA)  →  SMEM ringbuffer (3-4 stages)
                     - A-tile: M_tile × K_tile FP4 packed
                     - B-tile: N_tile × K_tile FP4 packed
                     - SFA: native row-major UE4M3 (no SfAtom convert)
                     - SFB: native row-major UE4M3
                     mbarrier sync per stage between producer/consumer
Consumer warps (4):  mma.sync.kind::mxf4nvf4 → FP32 register accumulators
                     online accumulate across K-tiles
Online cast:         FP32 → FP16 in registers
Epilogue:            cp.async.bulk.tensor → FP16 D output
```

### M-aware tile selection (host-side)

| M_e per expert | tile_M | N_tile | K_tile | Stages |
|---:|---:|---:|---:|---:|
| ≤ 16 | 16 | 128 | 128 | 4 |
| ≤ 32 | 32 | 128 | 128 | 4 |
| ≤ 64 | 64 | 128 | 128 | 4 |
| ≤ 128 | 128 | 128 | 128 | 3 |
| > 128 | 128 (multi-tile) | 128 | 128 | 3 |

SMEM budget per CTA (RTX 5090: 99 KiB opt-in):

| Config | A buf | B buf | SF buf | C epilogue | Total | Status |
|---|---:|---:|---:|---:|---:|---|
| 16/128/128/4st | 4 KiB | 32 | 4.5 | 4 | **44.5** | ✓ |
| 32/128/128/4st | 8 | 32 | 5 | 8 | **53** | ✓ |
| 64/128/128/4st | 16 | 32 | 6 | 16 | **70** | ✓ |
| 128/128/128/3st | 24 | 24 | 6 | 32 | **86** | ✓ |
| 128/128/128/4st | 32 | 32 | 8 | 32 | 104 | ✗ overflow |

### Persistent scheduler

Host pre-computes work queue, sorted by descending M_tile (big tiles first
→ shorter tail latency):

```cpp
struct WorkItem { int expert_id, m_tile_idx, n_tile_idx; uint8_t m_tile_size; };
std::vector<WorkItem> queue;
for (int e = 0; e < ne; ++e) {
    if (M_per[e] == 0) continue;
    int tile_M = pick_tile(M_per[e]);
    int n_m_tiles = (M_per[e] + tile_M - 1) / tile_M;
    int n_n_tiles = N / 128;
    for (int mi = 0; mi < n_m_tiles; ++mi)
        for (int ni = 0; ni < n_n_tiles; ++ni)
            queue.push_back({e, mi, ni, (uint8_t)tile_M});
}
std::stable_sort(queue.begin(), queue.end(),
    [](const auto& a, const auto& b){ return a.m_tile_size > b.m_tile_size; });
```

Device kernel: 170 CTAs (one per SM), each pulls work items via global atomic
counter. Inside each CTA, runtime dispatch picks the right template
specialization based on `wi.m_tile_size`.

### Native scale layout (preserved post-Phase 0)

The existing CUTLASS path requires SfAtom layout (128-row-padded, swizzled).
`cache_moe_native_nvfp4` already produces native row-major UE4M3 scales in
the `nvfp4_moe_ms_native` buffer at load time — this is what the decode-side
GEMV path consumes today. The CUTLASS prefill path adds a separate per-projection
SfAtom buffer and runs `convert_scales_sfatom_moe_kernel` to populate it
(2.6% of prefill kernel time + ~3 GiB VRAM).

The custom kernel reads native row-major UE4M3 scales **directly**, eliminating:
- The SfAtom buffer (~3 GiB VRAM saved)
- `convert_scales_sfatom_moe_kernel` (-2.6% kernel time)
- The `quantize_fp16_to_nvfp4_cutlass_moe_kernel` variant — replaced by a
  new `quantize_fp16_to_nvfp4_moe_native` that produces native layout (-2.2%
  kernel time)

This is a **necessary** layout change for the kernel, not a bonus optimization
— but it reuses the layout the load-time path already produces and the
decode-time path already consumes. Zero ABI risk.

### Determinism guarantee

Resolves the issue documented in
[`cutlass_nvfp4_sm120_nondeterministic_2026_05_05.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/cutlass_nvfp4_sm120_nondeterministic_2026_05_05.md):

- Each (expert, M-tile, N-tile) work item is owned by exactly one CTA.
  No cross-CTA reduction → no SM-finishing-order non-determinism.
- Inside each tile, K-reduction proceeds in fixed order across FP32 register
  accumulators → deterministic per-tile.
- No `cluster_launch_control`, no persistent tile-scheduler with global K-counter.
- Expected: 4/4 graph_replay byte-identical (vs CUTLASS NVFP4's 1-2/4 today).

## Files

```
src/compute/gemm_grouped_nvfp4_smallM.cu       ~600-800 LoC (kernel + scheduler)
src/compute/gemm_grouped_nvfp4_smallM.h        public C++ API
src/compute/quantize_fp16_nvfp4_moe_native.cu  ~100 LoC (native-layout activation quantize)
tests/test_gemm_grouped_nvfp4_smallM.cu        ~250 LoC (unit + numerical reference)
docs/sm120_smallM_kernel.md                    ~150 LoC (perf doc + audit recipe)
```

CMakeLists.txt: add the .cu sources under the existing sm_120a flag block.

### Public API

Drop-in compatible with `gemm_grouped_cutlass_3x_nvfp4` — same parameter set,
same calling convention.

```cpp
namespace imp {
bool gemm_grouped_nvfp4_smallM(
    int ne_active,
    const int* M_per_expert,        // host array, length=ne_active
    int N, int K,                   // common across experts
    const void* const* A_ptrs,      // host: per-expert packed FP4 (m × k/2 bytes)
    const void* const* SFA_ptrs,    // host: per-expert UE4M3 native row-major
    const void* const* B_ptrs,      // host: per-expert weight packed FP4
    const void* const* SFB_ptrs,    // host: weight scales native row-major
    void* const* D_ptrs,            // host: per-expert FP16 outputs
    const float* alpha,             // host: per-expert global scale
    cudaStream_t stream);
}  // namespace imp
```

### Activation-quantize refactor

The dispatch site needs a layout switch:

```cpp
// in executor_forward_moe.cu around line 1290
//
// Two gates:
//   (1) opt-in:  IMP_NVFP4_SMALLM env var (Phase A-B), or default-on (Phase D)
//   (2) profile: max(M_per) ≤ MAX_M_THRESHOLD
//                Phase A-B: threshold = 128 (validate every shape works)
//                Phase C+:  threshold = 64  (auto-heuristic per R1; CUTLASS keeps M_e>64)
const int max_M = *std::max_element(M_per.begin(), M_per.end());
const int threshold = ::getenv("IMP_NVFP4_SMALLM_FULL") ? 128 : 64;
const bool smallM_optin = ::getenv("IMP_NVFP4_SMALLM") != nullptr;
const bool use_smallM = smallM_optin && max_M <= threshold;

if (use_smallM) {
    quantize_fp16_to_nvfp4_moe_native(gathered_base, ...);   // native layout
    gemm_grouped_nvfp4_smallM(na, active_M.data(), eff, d,
                              hA, hSFA_native, hB, hSFB_native,
                              hD, alpha, stream);
} else {
    // existing CUTLASS 3.x path unchanged
    quantize_once(...);  grouped_gemm(...);
}
```

## Acceptance criteria (hard gates)

| Gate | Target | Measurement |
|---|---|---|
| Numerical | `‖smallM - CUTLASS‖∞ / ‖CUTLASS‖∞ < 1e-3` per expert | unit test |
| Decode | tg256 ≥ 268 tok/s on Qwen3-Coder | `make verify-fast` |
| Prefill (Qwen3-Coder) | pp512 median ≥ 19000 tok/s (10 reps) under best-threshold | `bench/results/smallM_baseline.log` |
| Prefill cross-model | pp512 +15% on ≥3 of 4 models under calibrated heuristic | bench sweep |
| Per-shape sweep | M-threshold table populated for {pp=128,512,1024,2048} × 4 models | `bench/results/smallM_threshold_calibration.csv` |
| Determinism | 4/4 graph_replay byte-identical | `validate_safetensors.py --replays=4` |
| Tests | all 574 GTest pass | `make test-gpu` |
| Build | sm_120a clean, 0 ptxas warnings | `make build` |
| VRAM | ≤ 0 MiB regression vs CUTLASS path | `nvidia-smi` after load |

## Risks

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Kernel slower than CUTLASS for M_e ≥ 64 | medium | high | Auto-heuristic `max_M ≤ 64` enables smallM only when win. Fallback always available. Per-model A/B in Phase B. |
| R2 | mma.sync.mxf4nvf4 inline PTX breaks on CUDA 13.3+ upgrade | medium | medium | Pin CUDA version in Dockerfile until re-validation. Wrap PTX in `#if __CUDA_VERSION__ <= 13030 && __CUDA_ARCH__ >= 1200` with build-error on unknown versions. |
| R3 | TMA bandwidth not realized due to SMEM bank conflicts | low | high | Reuse CUTLASS Sm120 swizzle pattern (XOR-2 for FP4). cute::Swizzle helper. |
| R4 | Variable tile sizes cause kernel divergence in persistent loop | low | medium | tile_M as template param, switch dispatch over runtime tile_M → 4 specialized paths compiled. |
| R5 | Determinism claim wrong — different FP-reduction order than CUTLASS | medium | low | Tile-internal reduction order is fixed → deterministic per call. Cross-call same input = same output. CUTLASS' issue was cross-call; ours not. |
| R6 | Activation native ↔ SfAtom layout creates inconsistency between prefill (smallM) and decode (existing GEMV) | low | high | `cache_moe_native_nvfp4` already lays out SF native row-major; decode-GEMV reads it directly. We only need to ensure `nvfp4_moe_ms_native` pointer is correctly forwarded (no convert). 1-day audit before kernel work. |
| R7 | Maintenance burden: custom kernel re-validated on each CUTLASS upgrade | high | low | `docs/sm120_smallM_kernel.md` with re-runnable bench + numerical gate. Audit in `make verify`. |
| R8 | Best-case not realized (post-Phase 0: target was downgraded from +30-70% to +15-30%, but even that may not materialize at every M-bin) | high | medium | **Auto-heuristic is now the primary mitigation, not a fallback.** Per-(model × pp_size) A/B in Phase B determines per-shape M-threshold. Reject if no threshold delivers ≥ +5% pp without decode regression — abort the entire branch and re-evaluate fusion-first. |
| R9 | Decode regresses subtle due to i-cache pressure from new symbol | low | medium | iteration-2 saw -7% on fused gate+up. Mitigation: kernel symbol in separate compilation unit, marked cold-attribute if needed. |

### Kill switch

`IMP_NVFP4_SMALLM=0` (default) → existing CUTLASS path. Env-flag flip rolls
back without rebuild. After Phase D default-on, the flip becomes
`IMP_NVFP4_SMALLM=0` to disable.

### Abort triggers (escalated after Phase 0)

- Phase A numerical gate fails (PTX inline asm bug) → 2-3 days debug; if more,
  abort and evaluate cute-DSL alternative.
- Phase A bench shows smallM slower than CUTLASS at *all* M-bins → root-cause
  hypothesis wrong, abort. **(Was less likely pre-Phase-0; with the TMA-fusion
  bonus removed, this is the most plausible failure mode.)**
- **New post-Phase-0 trigger:** Phase B per-shape A/B shows no M-threshold
  delivers ≥ +5% pp without decode regression on any of the 4 production
  NVFP4 MoE models → abort, re-evaluate Fusion-First (apsys-blog 5-launch
  pattern) as the alternative path.

## Roll-out (revised post-Phase 0 — auto-heuristic is now first-class)

1. **Phase A** (week 1): kernel + unit tests, opt-in via env, numerical
   validation only. **Adds: per-shape micro-bench harness so Phase B can
   sweep M-thresholds without manual re-runs.**
2. **Phase B** (week 2): **per-shape A/B sweep, not single-pass cross-model
   bench.** For each of the 4 NVFP4 MoE models × {pp=128, 512, 1024, 2048}
   × M-thresholds {16, 32, 48, 64, 80, 96, 128}, measure pp + tg, identify
   the best per-(model, pp_size) threshold. Output: a calibrated heuristic
   table.
3. **Phase C** (week 2-3): bake the per-shape calibration into auto-heuristic
   dispatch, determinism validation, decode regression sweep. Decision gate:
   does the calibrated heuristic deliver net-positive on ≥ 3/4 models? If
   yes, ship; if no, abort.
4. **Phase D** (post-merge): 1-2 weeks prod monitoring, then default-on
   (still env-overridable).

## Dependencies verified pre-Wo-1

- ✓ CUDA 13.2.1 inline-PTX support for mma.sync.mxf4nvf4 — confirmed via
  existing CUTLASS path SASS.
- ✓ TMA cp.async.bulk.tensor — UTMALDG present in audit (36 instances).
- ✗ **block_scale-aware fused TMA descriptor — REJECTED** by Phase 0 microbench
  (commit `a591dac`). Hardware does not expose a fused-descriptor primitive on
  SM120; CUTLASS's own SM120 NVFP4 path uses two separate descriptors. Kernel
  design follows the same 2-descriptor pattern.
- ✓ mbarrier multi-phase — used by CUTLASS templates we already build.
- ✓ atomicAdd on global counter — CUDA standard.

## Out of scope

- FlashInfer / TensorRT-LLM `fp4_gemm` integration (would add a Python dep,
  violates CLAUDE.md "no new third-party deps without strong reason").
- Decode-side GEMV rewrite (separate work; current path is +234% with CUDA
  graphs and not the bottleneck).
- Activation-quant-into-GEMM epilogue fusion (a second optimization, can be
  layered on top after smallM ships).
- compute_120f migration (memory `cuda_arch_120a_2026_05_04` validates 120a
  is the correct target for our CUTLASS 4.4.x + CUDA 13.2.1 stack).

## References

- `bench/iteration2_findings.md` — root-cause analysis of the M-tile constraint
- `docs/archive/bench-2026-05-10/profile_findings.md` — pre-bc3bc31 profile (slow-fallback path)
- `docs/archive/bench-2026-05-10/pp_optimization_report.md` — bc3bc31 win documentation
- memory `sm120_real_perf_levers_2026_05_04.md` — authoritative SM120 hardware reference
- memory `sass_audit_120a_no_tcgen05_2026_05_04.md` — SASS-level proof that we're on the peak MMA pipe
- memory `cutlass_nvfp4_sm120_nondeterministic_2026_05_05.md` — what we want to avoid
- memory `sm120_mma_variants_2026_04_25.md` — MMA TOPS table, validated again 2026-05-10
- FlashInfer issue #2723 (RTX PRO 6000 SM120 patches) — context only, we don't depend on FlashInfer
- HuggingFace blog `apsys/blackwell-nvfp4-comparison` — engineering insights (B200 not SM120, but transferable kernel-fusion patterns)

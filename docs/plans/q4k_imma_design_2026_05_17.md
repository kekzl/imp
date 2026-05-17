# Q4_K_M direct GEMM via INT8 IMMA — design scoping memo

**Date**: 2026-05-17
**Status**: design only — no source changes
**Subject**: Close the FP16-TC ceiling gap on Q4_K_M at M ≥ 32 via
`mma.sync.aligned.m16n8k32.s32.s8.s8.s32` (~838 TOPS on sm_120a).
**Branch**: `main` @ `9520223` (HEAD)

## Table of contents

1. [Status check](#1-status-check)
2. [sm_120 INT8 IMMA capability](#2-sm_120-int8-imma-capability)
3. [Q4_K → INT8 reordering](#3-q4_k--int8-reordering)
4. [Kernel skeleton](#4-kernel-skeleton)
5. [Risks & blockers](#5-risks--blockers)
6. [Implementation plan](#6-implementation-plan)
7. [Decision recommendation](#7-decision-recommendation)

---

## 1. Status check

### 1.1 v2 (HMMA) is retired

```
$ git log --all --oneline -- 'src/compute/mmq_q4k_v2*'
340fb79 chore(recovery): bring Day-30-60 stack content to main (#192-#198 fix) (#199)
f58eb9e feat(compute): Q4_K direct-mmq (v1 dp4a + v2 HMMA) + MoE prefill refactor (#189)

$ git show 340fb79 --stat -- 'src/compute/mmq_q4k_v2*' 'tests/test_mmq_q4k_v2*'
 src/compute/mmq_q4k_v2.cu | 1667 ----------------------------------
 src/compute/mmq_q4k_v2.h  |  203 ---
 tests/test_mmq_q4k_v2.cu  |  695 ----------
```

PR #199 (recovery commit, 2026-05-16) deleted the v2 HMMA kernel + its
tests. Roadmap §`pp=512 on large dense models` references PR #193 for
v2 retirement; the actual deletion landed in #199. Either way, **v2 is
gone from main**. Memo `mmq_q4k_v2_phase2_shipped_2026_05_16.md`
records the -4 % end-to-end regression on Qwen3.6-35B Q4_K_M that
motivated the retirement.

### 1.2 v1 (dp4a tiled) — **not present in main** (discrepancy with roadmap)

The roadmap (`docs/roadmap.md:78`) claims:

> A direct tiled Q4_K_M GEMM kernel shipped 2026-05-15 in
> `src/compute/mmq_q4k.cu` (commits `3b49325` → `8dbfdbd`). […]
> Dispatched in `[2, 16]` only via `executor_kernels.cu`.

Verification:

```
$ git ls-files | grep -iE 'q4k|mmq'
tools/analysis/bench_q4k_mmvq_crossover.sh

$ git ls-tree HEAD src/compute/ | grep -iE 'q4k|mmq'
# empty

$ find . -path ./build -prune -o -name 'mmq_q4k*' -print
# empty

$ grep -rn 'mmq_q4k\|MMQ_Q4K' src/ include/
# empty
```

Neither `src/compute/mmq_q4k.cu`, `src/compute/mmq_q4k.h`,
`tests/test_mmq_q4k.cu` nor `tools/imp-bench/bench_mmq_q4k.cu` exist
in the working tree, and no commit reachable from `main` ever added
them. PR #189 (`f58eb9e`) introduced **only** `mmq_q4k_v2.{cu,h}`
(scaffold), not v1; v1 lived on the `feat/q4k-mmq-gemm` work branch
that was squashed into PR #189 with only the v2 files surviving. The
recovery commit #199 then removed the v2 files too.

**Net state**: Q4_K_M is **not** owned by a direct-mmq kernel in main.
The relevant code paths in `src/graph/executor_kernels.cu:2200-2300` are:

- `use_mmvq` (Gemma-4 force flag + Q4_K/Q5_K/Q5_1/Q8_0) → `ggml_mmvq_q4k`
  (warp-per-output-element batched-GEMV; `src/compute/ggml_mmvq.cu:513`).
- `use_dp4a` (M == 1, Q4_K eligible) → `dispatch_dp4a_gemv` (scalar
  dp4a GEMV; `src/compute/gemm_dp4a.cu`).
- All other M ≥ 2 Q4_K dispatches → `dequant_gpu` → FP16-TC `cuBLAS`
  via `dequant_scratch`.

This matters for the IMMA design **because there is no v1 to compare
against in production today**. The microbench numbers cited in the
v1 memo (`mmq_q4k_phase_a_2026_05_15.md`) — 2.0-2.4× over mmvq across
M = 32..512, end-to-end +13–56 % at M = 2..16 on Gemma-3-12B Q4_K_M —
were measured against `ggml_mmvq` on the now-removed v1 code. Any
INT8-IMMA Phase 1 microbench will compete against **`ggml_mmvq` + FP16
dequant+cuBLAS**, not against the absent v1.

> **Roadmap correction owed**: `docs/roadmap.md:78` should be amended
> to note that v1 was retired alongside v2 in #199. Not part of this
> design memo's scope.

### 1.3 FP16-TC cuBLAS owns M ≥ 32 today

`src/graph/executor_kernels.cu:2287` shows the fallback:

```cpp
} else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
    int rows = static_cast<int>(weight.shape[0]);
    int cols = static_cast<int>(weight.shape[1]);
    // dequant Q4_K → FP16 into dequant_scratch, then cuBLAS FP16-TC
    ...
}
```

The `pp=512` nsys profile in roadmap §`pp=512 on large dense models`:

- 25 % GPU time in `dequant_q4k_kernel`
- 23 % host time in synchronous `cudaMalloc`/`cudaFree` (939 + 930 calls)
- 64 % in the cuBLAS GEMM itself

The IMMA kernel's payoff zone is the 25 % currently spent on the
dequant pre-pass — INT8 IMMA replaces both `dequant_q4k_kernel` and
the cuBLAS call with a single fused kernel.

---

## 2. sm_120 INT8 IMMA capability

### 2.1 PTX instruction support

`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` is the standard
integer Tensor Core MMA introduced for sm_75 (Turing) and carried
forward through every NVIDIA arch since. The PTX ISA spec (CUDA 13.2,
§9.7.13.4.13 "Matrix multiply-accumulate instruction: mma") lists s8
operands at `m16n8k32` as supported for `target-architecture ≥
sm_75`. Consumer Blackwell (`sm_120a`) inherits this — the PTX MMA
acceptance survey (`tools/analysis/ptx_mma_survey.sh`, memo
`ptx_mma_survey_2026_04_26.md`) does not test it explicitly because
it focuses on the newer FP4/FP6/FP8 `kind::{f8f6f4,mxf4nvf4,mxf8f6f4}`
variants where compute-cap support was uncertain. INT8 IMMA at
`m16n8k32` has been universally compilable on every arch since Turing.

**ptxas-acceptance pre-flight**: trivial — extend
`tools/analysis/ptx_mma_survey.sh` with one `template_dense_noscale`
case `"s32.s8.s8.s32" "m16n8k32"`. Confidence: 99 % accepts. Should
still be done as a 5-minute sanity step before Phase 1 implementation
because PTX-ISA-table != ptxas-acceptance.

### 2.2 Throughput: dp4a peak vs INT8 IMMA peak

The 16× ratio cited in `mmq_q4k_v2_hmma_design_2026_05_15.md` and
`docs/roadmap.md:83` is:

| Path | Op | Peak (RTX 5090 / sm_120a) | Source |
|---|---|---|---|
| Scalar dp4a | `IDP.4A.S8.S8` (4 × s8 → s32 per warp-lane per cycle) | ~50 TOPS | NVIDIA whitepaper + measured via `IDP.4A.S8.S8` × 4638 in HW capability audit |
| INT8 IMMA | `mma.sync.m16n8k32.s32.s8.s8.s32` (16 × 8 × 32 = 4096 multiplies per warp per issue) | ~838 TOPS | Same TC peak as FP16 HMMA on GB202 — 2× higher than data-center FP16 because TC throughput scales with element width |

GB202 spec sheet from `cuda_arch_120a_2026_05_04`: 838 TFLOPS FP16
(via HMMA), 3354 TOPS FP4 (via OMMA at `m16n8k64`). INT8 sits at
838 TOPS — same per-tile throughput as FP16 because both consume
the same `m16n8k16` / `m16n8k32` slot capacity on the TC pipe; the
4× ratio between INT8 and FP4 comes from K = 32 vs K = 64.

Net ratio: **16.7× ceiling lift** vs scalar dp4a, assuming both reach
their nominal peaks. The realised speedup will be less — the dequant
+ pack pipeline adds overhead. v2 HMMA hit ~15 % of FP16-TC peak in
the Phase 7 roofline measurement (per `mmq_q4k_v2_phase2_shipped`,
M=512 N=K=5120 at 0.215 ms ≈ 124 TFLOPS effective vs 838 peak). A
similar 15 % realisation on the INT8 path → 838 × 0.15 = 126 TOPS,
which is 2.5× over dp4a's 50 TOPS peak (best case, realised <50 %).

### 2.3 Register fragment layout for `m16n8k32.s32.s8.s8.s32`

Per PTX ISA §9.7.13.4 (matrix shape `.m16n8k32`, type `.s8.s8`):

- **A fragment** (16 × 32 s8 = 512 bytes): `4 × .b32` per thread, 32
  threads/warp. Each thread holds 16 s8 values, packed 4 per `b32`.
  Layout: rows split across thread groups of 4 (one quadrant per
  group), each group covers 8 rows × 32 columns.
- **B fragment** (32 × 8 s8 = 256 bytes): `2 × .b32` per thread.
  Layout: columns split, each thread holds 8 s8 values packed 2 ×
  `b32`.
- **C / D fragment** (16 × 8 s32 = 512 bytes): `4 × .b32` per thread,
  s32 accumulator. Same layout as the `m16n8k16.f32` HMMA accumulator
  used by v2.

The PTX call form (same operand-count template as v2's HMMA):

```ptx
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  {d0, d1, d2, d3},          // s32 accumulator (4× b32)
  {a0, a1, a2, a3},           // s8×s8 A operand (4× b32)
  {b0, b1},                   // s8×s8 B operand (2× b32)
  {c0, c1, c2, c3};           // s32 accumulator in (4× b32)
```

A operands are loaded via `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
(same as v2 — `ldmatrix` is dtype-agnostic at the byte level). B
operands via `ldmatrix.sync.aligned.m8n8.x2.shared.b16`. Critical
detail: `ldmatrix` operates on **8 × 8 matrices of 16-bit elements**;
for s8 operands we treat the SMEM as a 16-bit tile of packed s8 pairs.
This means the SMEM staging layout for the Q4 → s8 buffer must be
exactly the form `ldmatrix` expects — 8 rows × 8 cols of 16-bit
"elements", where each "element" is 2 packed s8s. Effective tile is
8 × 16 s8.

For `m16n8k32` we need K = 32 s8 columns per A row = 4 lanes of 8
columns each = `ldmatrix.x4` (4 matrix loads issued by one PTX call,
each 8 × 8 of `b16` = 8 × 16 of s8). One `ldmatrix.x4` fills exactly
the A operand for one MMA tile. Symmetric: B needs `ldmatrix.x2`
(2 × 8 × 8 b16 = 2 × 8 × 16 s8 → 32 × 8 s8 = right shape).

This layout is **identical to v2's plan** at the `ldmatrix.x4` /
`ldmatrix.x2` level; only the dtype of what each "b16" holds differs
(2 × s8 instead of 1 × f16).

References:
- PTX ISA 8.5, §9.7.13.4 "Matrix Multiply-Accumulate Instructions"
- PTX ISA 8.5, §9.7.13.5 "ldmatrix" (b16 mode)
- v2 PTX usage: `mmq_q4k_v2_hmma_design_2026_05_15.md` §"Inner-loop sketch"

---

## 3. Q4_K → INT8 reordering

### 3.1 Q4_K block layout (verified against current code)

Source: `src/compute/ggml_mmvq.cu:18-24`.

```cpp
struct ggml_block_q4_K {
    half d;              // super-block scale       (2 B)
    half dmin;           // super-block min         (2 B)
    uint8_t scales[12];  // 8 sub-block scales + 8 sub-block mins,
                         //   6-bit each, packed into 12 bytes
    uint8_t qs[128];     // 256 × 4-bit quants
};
static_assert(sizeof(ggml_block_q4_K) == 144);
```

Per super-block (256 elements):
- 1 × FP16 `d` (super-block scale)
- 1 × FP16 `dmin` (super-block min)
- 8 sub-blocks × 32 elements each
- Per sub-block i ∈ [0, 8): one 6-bit `sc[i]` (sub-block scale factor)
  + one 6-bit `m[i]` (sub-block min factor), packed into `scales[12]`
  via the cross-byte layout in `vec_dot_q4_K_q8_1` (lines 194–203).
- Dequant rule: for value `q` in sub-block `i`,
  `fp16_value = d * sc[i] * q  -  dmin * m[i]`.

### 3.2 INT8 conversion strategies — pick (a)

**Strategy (a): symmetric s8, affine offset post-MMA.**

The Q4 nibble `q ∈ [0, 15]` is unsigned. Subtract a fixed bias of 8:
`q_sym = q - 8 ∈ [-8, 7]`, fits cleanly in `int8_t`. The dequant rule
becomes:

```
fp16_value = d * sc[i] * (q_sym + 8) - dmin * m[i]
           = (d * sc[i]) * q_sym + (8 * d * sc[i] - dmin * m[i])
```

Per sub-block (per output row, per K = 32 column slab):

- **Multiplicative factor** `α[i] = d * sc[i]` — applied as a single
  per-(row, sub) FP32 multiply on the s32 IMMA accumulator after the
  inner K-loop completes one sub-block.
- **Additive constant** `β[i] = 8 * d * sc[i] - dmin * m[i]` —
  contributes a per-(row, sub) **bias** to the output. It does NOT
  multiply the activation values, but it DOES need to sum across the
  activation per-row-sum, because `(q_sym + 8) · a = q_sym · a + 8 · a`
  and the `8 · a` term collapses into `(8 · sum(a))`. With activation
  also int8-quantized, the per-tile-K activation sum becomes an extra
  pre-computed scalar.

Reformulated:

```
out[m, n] = Σ_k   x[m, k] · W[n, k]
         ≈ Σ_subs ( α[n, sub] · Σ_{k in sub}  x_s8[m, k] · w_sym_s8[n, k] · x_scale[m, sub_x]
                  + β[n, sub] · Σ_{k in sub}  x_s8[m, k] · x_scale[m, sub_x] )
```

…which factors as: the IMMA result for each sub-block, multiplied by
`α[n, sub] · x_scale[m, sub_x]` and added to a row-sum term.

Memory format on device (one-shot at model load, into a new
`WeightCaches::q4k_imma` map):

- **`w_sym_s8[N, K]`** — symmetric s8 weight tensor, K-major.
  Reordered to the exact `ldmatrix` fragment layout (rows split
  across thread quadrants, 8 × 16 s8 tiles). Memory cost:
  Qwen3-32B Q4_K_M (≈10 GB Q4_K weight) → 2× growth to **~20 GB s8**.
  This is the same 2× growth v2 had with its FP16 expansion — but s8
  doesn't gain us memory back vs Q4. **Significant blocker** (see
  §5.3).
- **`eff_alpha[N, K/32]`** — FP16, `d * sc[i]` per (row, sub-block).
  Same size as v2's `eff_scale` — ~10 MB for Qwen3-32B.
- **`eff_beta[N, K/32]`** — FP16, `8 * d * sc[i] - dmin * m[i]` per
  (row, sub-block). Same size — ~10 MB for Qwen3-32B.

The 2× weight-storage blow-up is the central drawback of strategy (a)
and is not amortisable: an s8 byte per Q4 nibble is a hard floor.
Mitigations in §5.3.

**Strategy (b): unsigned u8, no symmetric shift.**

PTX exposes `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` — fully
supported on sm_75+. Q4 `q ∈ [0, 15]` fits in u8 directly. But:

- The B (weight) tensor is u8 and the A (activation) tensor must also
  be u8 — and activations are signed. Quantising FP16 → u8 needs an
  additive shift that adds the same `β`-style bias term, just on the
  activation side. No simplification vs (a).
- The mixed `s32.u8.s8.s32` variant exists per PTX ISA — A as u8, B
  as s8, accumulator s32. This **does** let us keep weights as u8 (no
  symmetric shift, no `β` term on the weight side) and activations
  s8 (the natural FP16-quantize-to-s8 path). The `β` term degenerates
  to `β[i] = -dmin * m[i]` — strictly cheaper.
- **Risk**: `s32.u8.s8.s32` is in PTX ISA 7.5+ (CUDA 11.5+) but I have
  not verified ptxas-acceptance on sm_120a explicitly. The 5-minute
  ptx survey extension (§2.1) should cover both variants.

**Decision**: go with **strategy (a) symmetric s8**. Reasons:

1. The standard library reference (CUTLASS `tools/library/include/
   cutlass/library/operation_table.h`) uses `s8.s8.s32` overwhelmingly
   for INT8 GEMM — better tooling familiarity.
2. The `β` term in (a) is one extra FP32 FMA per (row, sub-block)
   after the IMMA — negligible vs IMMA throughput.
3. Mixed-sign `u8.s8` saves nothing in instruction count, and adds an
   unverified PTX path. Defer to a follow-up if Phase 1 hits an
   accuracy issue with the symmetric quantisation.

### 3.3 Activation quantisation

Input `x[M, K]` is FP16. INT8 IMMA needs s8 with a scale that
faithfully represents the FP16 dynamic range. Strategies, ordered
by complexity:

- **Per-row** (`x_scale[M]`, single FP16 per token): coarsest, max
  absolute on the K dimension per row. Risk: heavy-tailed activations
  (one outlier blows up the scale) degrade precision.
- **Per-row-per-sub-block** (`x_scale[M, K/32]`, K/32 scales per
  token): same granularity as the weight sub-blocks. This is what
  llama.cpp's Q8_K_R8 / Q8_1 (used by the current `mmvq` and `dp4a`
  paths) does for activation quantisation — see
  `src/compute/ggml_mmvq.cu:226 quantize_fp16_to_q8_1_ggml_kernel` for
  the existing implementation. **Reuse this**: it's tested, fast, and
  the K/32 sub-block alignment matches the Q4_K weight's sub-block
  layout exactly.
- **Per-token-per-channel**: finest, never used in practice for
  decode shapes.

**Choice**: per-row-per-sub-block (`x_scale[M, K/32]`), same as Q8_1.
Activation conversion runs once at the start of the GEMM (the same
`quantize_fp16_to_q8_1` kernel currently called from
`executor_kernels.cu:2253`) and writes into a graph-capture-safe
preallocated scratch in `WeightCaches`.

This means our IMMA kernel sees:

- `x_s8[M, K]` — int8 activation, packed for `ldmatrix.x4`.
- `x_scale[M, K/32]` — FP16 per-row-per-sub-block activation scale.
- `w_sym_s8[N, K]` — int8 weight (symmetric, pre-shifted by 8).
- `eff_alpha[N, K/32]` — FP16 per-row-per-sub-block weight α.
- `eff_beta[N, K/32]`  — FP16 per-row-per-sub-block weight β.
- `x_rowsum[M, K/32]` — FP32 per-row-per-sub-block activation sum
  (needed for the β-term coupling). One reduction over K=32 elements
  per (M, sub) — produced by a small follow-on kernel after activation
  quantisation, or fused into `quantize_fp16_to_q8_1`.

llama.cpp reference: the Q8_K_R8 activation format adds a per-block
sum field exactly for this β-coupling. Confirmed pattern; we can
follow it.

### 3.4 Per-block scale overhead: K_tile = 32 hard constraint

The IMMA `m16n8k32` consumes exactly K = 32 s8 columns per issue,
which lines up **perfectly** with one Q4_K sub-block. **One IMMA call =
one sub-block of one super-block of one column slab**. After the IMMA
call returns, we apply `(α[n, sub] · x_scale[m, sub_x])` and add the
β-term — both per-(m, n, sub) FP32 operations on the s32 accumulator.

This forces `BLOCK_K = 32` if we want a single set of per-sub-block
scales per inner-loop iteration. Going to `BLOCK_K = 64` would force
two sub-blocks per iteration with different scales — feasible by
either:

- Issuing two IMMA calls per inner-loop iteration (one per sub-block),
  applying scales between them. Same effective throughput, more
  instruction-level overhead.
- Accumulating both sub-blocks in s32, then de-mixing via two passes
  of scale application. Cheaper but doubles the s32 accumulator
  register pressure (8 × b32 instead of 4 × b32 per warp).

**Decision**: stick with `BLOCK_K = 32`. The 16 sub-blocks per
512-element K-dimension means 16 IMMA + 16 scale-apply iterations per
output tile — manageable. CUTLASS's INT8 GEMM tile kernels routinely
do similar per-iteration scale applications via the epilogue.

The tilus blueprint (v2 design memo §"Tile-level layout") used
exactly `block_k = 32 = 16 × 2` for the same reason. The constraint
is intrinsic to Q4_K, not specific to INT8 vs FP16.

---

## 4. Kernel skeleton

Reference style: hand-rolled C++/PTX, same as
`src/compute/attention_fmha_sm120.cu`,
`src/compute/gemm_cutlass_sm120.cu`, and the dispatched-but-now-deleted
v2. NUM_STAGES = 3 cp.async pipeline; 4 warps per block in 2 × 2 spatial
arrangement; one output tile of 64 × 64 per block.

```cuda
template <int BLOCK_M, int BLOCK_N, int BLOCK_K = 32, int NUM_STAGES = 3>
__global__ void mmq_q4k_imma_kernel(
    const int8_t*  __restrict__ w_sym_s8,    // [N, K] reordered
    const half*    __restrict__ eff_alpha,    // [N, K/32]
    const half*    __restrict__ eff_beta,     // [N, K/32]
    const int8_t*  __restrict__ x_s8,         // [M, K] reordered
    const half*    __restrict__ x_scale,      // [M, K/32]
    const float*   __restrict__ x_rowsum,     // [M, K/32]
    half*          __restrict__ out,          // [M, N]
    int M, int N, int K)
{
    // -------- SMEM declarations (sm_120a: 99 KB/block budget) --------
    __shared__ int8_t  sA[NUM_STAGES][BLOCK_M][BLOCK_K + PAD];   // pad → no bank conflicts
    __shared__ int8_t  sW[NUM_STAGES][BLOCK_N][BLOCK_K + PAD];
    __shared__ half    sXscale[NUM_STAGES][BLOCK_M];             // K/32 → 1 per stage per row
    __shared__ float   sXrowsum[NUM_STAGES][BLOCK_M];
    __shared__ half    sAlpha[NUM_STAGES][BLOCK_N];
    __shared__ half    sBeta [NUM_STAGES][BLOCK_N];

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int warp_m  = warp_id / 2;        // 0..1
    const int warp_n  = warp_id % 2;        // 0..1

    // Each warp owns a (16 × 8) output sub-tile per WRM × WRN repeat.
    // WRM=2, WRN=4 (mirror v2): warp covers 32 × 32 of the 64 × 64 tile.
    int32_t c_acc[WRM][WRN][4] = {{{0,0,0,0}}};  // s32 accumulators (4× b32 per MMA)

    const int K_blocks = K / BLOCK_K;          // # of inner-loop iterations
    const int n_block  = blockIdx.x;
    const int m_block  = blockIdx.y;

    // -------- Prologue: kick off NUM_STAGES-1 cp.async prefetches --------
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        cp_async_load_stage(s, /* k_block = */ s, ...);    // sA[s], sW[s], scales
        cp_async_commit_group();
    }

    // -------- Main pipelined loop --------
    for (int kb = 0; kb < K_blocks; ++kb) {
        const int stage = kb % NUM_STAGES;
        const int next  = (kb + NUM_STAGES - 1) % NUM_STAGES;

        // Wait for stage `stage` to land
        cp_async_wait_group<NUM_STAGES - 2>();
        __syncthreads();

        // Read α, β, x_scale, x_rowsum for this sub-block
        const float alpha_xscale = __half2float(sAlpha[stage][warp_n_lane])
                                 * __half2float(sXscale[stage][warp_m_lane]);
        const float beta_x       = __half2float(sBeta [stage][warp_n_lane])
                                 * sXrowsum[stage][warp_m_lane];

        // -------- Inner MMA over WRM × WRN warp-repeats --------
        #pragma unroll
        for (int wm = 0; wm < WRM; ++wm) {
            int32_t a_frag[4];
            ldmatrix_x4_b16(a_frag, &sA[stage][warp_m * 32 + wm * 16][0]);
            // ↑ 4× b32 (= 16× s8) per thread; covers 16 × 32 of A
            #pragma unroll
            for (int wn = 0; wn < WRN; ++wn) {
                int32_t b_frag[2];
                ldmatrix_x2_b16(b_frag, &sW[stage][warp_n * 32 + wn * 8][0]);
                // ↑ 2× b32 (= 8× s8) per thread; covers 32 × 8 of B

                int32_t* c = c_acc[wm][wn];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));
            }
        }

        // -------- Per-sub-block scale application (s32 → f32 → f32) --------
        // After the MMA, c_acc holds Σ_k (x_s8 · w_sym_s8) for this sub-block.
        // Multiply by α · x_scale, add β · x_rowsum, accumulate into an f32
        // output buffer (kept in registers across the K loop).
        #pragma unroll
        for (int wm = 0; wm < WRM; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WRN; ++wn) {
            const float a = alpha_xscale_for_lane(wm, wn);
            const float b = beta_x_for_lane(wm, wn);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                out_acc[wm][wn][i] = __fmaf_rn(__int2float_rn(c_acc[wm][wn][i]), a,
                                               out_acc[wm][wn][i] + b);
                c_acc[wm][wn][i] = 0;       // reset for next sub-block
            }
        }

        // Issue the next cp.async prefetch (for kb + NUM_STAGES - 1)
        if (kb + NUM_STAGES - 1 < K_blocks)
            cp_async_load_stage(next, kb + NUM_STAGES - 1, ...);
        cp_async_commit_group();
    }

    // -------- Epilogue: f32 → f16 store to gmem --------
    #pragma unroll
    for (int wm = 0; wm < WRM; ++wm)
    #pragma unroll
    for (int wn = 0; wn < WRN; ++wn) {
        // stmatrix.sync.aligned.m8n8.x4.shared.b16 via SMEM staging, then
        // cooperative store to out[m_block * BLOCK_M + ..., n_block * BLOCK_N + ...]
        store_tile_fp16(out, out_acc[wm][wn], m_block, n_block, wm, wn);
    }
}
```

Key differences vs v2 HMMA skeleton:

| Aspect | v2 HMMA | IMMA (this design) |
|---|---|---|
| MMA primitive | `m16n8k16.f32.f16.f16.f32` | `m16n8k32.s32.s8.s8.s32` |
| K per MMA | 16 | 32 |
| K per sub-block | 32 (= 2 MMAs / sub) | 32 (= 1 MMA / sub) ✓ |
| B fragment dtype | half (4 × b32 / tile) | s8 packed (2 × b32 / tile) |
| Accumulator | f32 | s32 → f32 cast post-MMA |
| Scale application | inline FMA on f32 | s32→f32→FMA (one extra cast) |
| Bias term | none (eff_min absorbed in dequant) | yes (β · rowsum per sub-block) |
| SMEM per stage (BLOCK_M=BLOCK_N=64, BLOCK_K=32) | sA: 4 KB, sW: 4 KB → 8 KB/stage | sA: 2 KB, sW: 2 KB → 4 KB/stage |

The smaller SMEM footprint (s8 = 1 B vs f16 = 2 B) means we can run
NUM_STAGES = 4 or BLOCK_M = BLOCK_N = 128 if useful — tile autotuner
decides in Phase 2.

---

## 5. Risks & blockers

### 5.1 Per-block scale overhead — manageable

The kernel applies α and β per sub-block (every 32 K elements). At
WRM × WRN = 2 × 4 = 8 output sub-tiles per warp, each with 4 s32
accumulators, the scale-apply phase is 8 × 4 = 32 FMAs per warp per
sub-block. Compared to the IMMA throughput (one IMMA does 16 × 8 × 32
= 4096 multiplies per warp), the scale phase is ~1 % overhead. Not a
blocker.

The α and β tensors are loaded via cp.async from `[N, K/32]` and
`[M, K/32]` — 64 × 2 B × 2 fields = 256 B per stage. Negligible
bandwidth. Not a blocker.

The `x_rowsum` term needed for the β-coupling adds one float32
reduction per (m, sub) at activation-quantisation time. Reuse the
existing `quantize_fp16_to_q8_1_ggml_kernel` (already computes
`d * sum(qs)` for Q8_1's `s` field — line 233-244 in
`src/compute/ggml_mmvq.cu`). The fp32 row-sum extraction is one extra
line in that kernel. Not a blocker.

### 5.2 Activation INT8 quantisation accuracy

Per-row-per-sub-block s8 quantisation is the same precision Q8_1
gives today for dp4a, and dp4a is the production GEMV path for M = 1
Q4_K. The numeric error budget is **bounded by Q8_1**, which is
already validated to ~FP16 ulp parity by `src/compute/ggml_mmvq.cu`'s
tests. Not a blocker.

### 5.3 Memory footprint: 2× weight expansion

This is the largest concern. Strategy (a) requires `w_sym_s8[N, K]` —
one s8 per Q4 nibble, doubling Q4_K's storage. For Qwen3-32B Q4_K_M:

- Q4_K weight bytes: ~10 GB.
- s8 reordered weight: ~20 GB.
- Plus `eff_alpha[N, K/32]` ~10 MB and `eff_beta[N, K/32]` ~10 MB
  (negligible).
- VRAM budget on RTX 5090: 32 GB total.

A naive "load all Q4_K weights, materialise s8 copies for all" wastes
~30 % of VRAM and crowds out KV cache. Mitigations, in order of
preference:

1. **In-place replacement**: drop the original Q4_K once `w_sym_s8` is
   built. Net storage: only s8 + α + β. But this breaks fallback to
   `ggml_mmvq` / dequant+cuBLAS for any case the IMMA dispatch
   doesn't cover (M = 1 GEMV; weights below MIN_M heuristic). Either
   (i) commit to IMMA + s8-GEMV everywhere on Q4_K — multi-week port
   that subsumes the GEMV path too, or (ii) reconstruct Q4 on demand
   from s8 — adds 1× FP16 dequant + (Q4-recovery) pass.
2. **Selective materialisation**: only materialise s8 copies for
   weights that the dispatch heuristic expects to fire IMMA. For
   dense Q4_K_M models that's "all of them" anyway, but for MoE the
   expert weights stay below MIN_M and can be skipped.
3. **Fused dequant-and-IMMA**: skip the precomputed `w_sym_s8`
   entirely; do Q4 → s8 in the inner SMEM-staged loop, like v2 did
   for Q4 → FP16. Saves all VRAM growth but adds 32 nibble-decode
   operations per K-block per output column — non-trivial overhead.
   Per the v2 experience (Phase 7a hybrid did this for Q4 → FP16),
   it cost ~30 % in throughput but unblocked the memory budget.

**Recommendation**: defer the storage decision to Phase 1. The Phase 1
microbench can run with pre-materialised s8 to isolate the IMMA
question. If Phase 1 wins, evaluate (1) vs (3) for Phase 2
integration — likely (3) for compatibility with the existing GEMV
fallback path.

### 5.4 v2 lessons — what carries over and what doesn't

v2 regressed -4 % end-to-end on Qwen3.6-35B Q4_K_M for three
reasons:

| v2 regression cause | IMMA mitigation |
|---|---|
| MoE keeps experts under `MIN_M = 64` v2 threshold | Same on IMMA — MoE experts see M ≈ prompt_tokens / num_experts ≈ 3-8. No mitigation. |
| `fp16_cache` hits skip v2 entirely (the `!fp16_cache_hit` guard) | Same on IMMA — if a weight is already in the FP16 cache (R5-Slice 4 cache), cuBLAS FP16-TC + the cached tensor wins. |
| Phase 1 dispatch overhead (~10 µs per call) | Same on IMMA — any new dispatch path adds a per-call lookup. R5 Slice 5b's eager `WeightCaches::q4k_v2` pattern can be reused for `WeightCaches::q4k_imma` (lookup is a single `map::find`). |

**What IMMA changes**: only the kernel ceiling. v2 ceiling was 4.87×
v1 dp4a in microbench but realised <1× e2e because of the above three
reasons. IMMA's ceiling is **16×** dp4a (3.3× v2 HMMA's ~4.87×
realised microbench). The interesting question is:

> Does the 3.3× microbench headroom over v2 translate to enough
> realised e2e win to overcome the same dispatch / fp16_cache / MoE
> overheads?

If v2 hit 50 % of FP16-TC peak in microbench and lost 4 % e2e, IMMA
would need to hit a much higher fraction of INT8 peak to win e2e by
a measurable margin. The pessimistic answer:

- v2 microbench was 4.87× v1 dp4a, e2e was 0.96× → **microbench
  inflated by 5×**.
- IMMA target microbench: 2-3× v1 dp4a (Phase 1 gate, §6) → expected
  e2e: 0.4-0.6× v1. **Likely worse than the FP16-cache fast path.**

This is the central risk. **IMMA's payoff zone is the subset of Q4_K
weights that are (a) not in fp16_cache, (b) dispatched at M ≥ 32, and
(c) ideally on dense (non-MoE) models.** That zone is small for
imp's current model set (see §5.5).

### 5.5 Which models actually qualify?

From the test set in `models/` and `/home/kekz/models/`:

| Model | Type | Q4_K weights? | IMMA payoff? |
|---|---|---|---|
| Qwen3-32B Q4_K_M | dense | yes | **prototype target** (this is the roadmap motivation) |
| Mistral-24B Q6_K | dense | Q6_K, not Q4_K | no — IMMA design is Q4_K-specific |
| Gemma-3-12B Q4_K_M | dense | yes | yes (the original v1 microbench shape) |
| Gemma-4-26B-A4B Q4_K_M | MoE | yes | no — experts stay below M=32 |
| Qwen3.6-35B-A3B Q4_K_M | MoE | yes | no — same reason |
| Qwen3-Coder-30B-A3B Q6_K | MoE | Q6_K | no — Q4_K-specific |
| Qwen3-4B / 3-8B / 3.5-* | dense | Q8_0 baselines | no — Q8_0 already TC-friendly via cuBLAS |
| Llama-3.2-3B Q8_0 | dense | Q8_0 | no |

**Only two models in the local set qualify: Qwen3-32B Q4_K_M and
Gemma-3-12B Q4_K_M.** Both are dense, both bypass `fp16_cache` on the
heavier weight-rows (KV-cache-budget trade-off), and both hit pp=512
prefill at M ≥ 32.

The 35B-MoE case is **explicitly excluded** by the M < 32 expert
shape. Lowering `MIN_M` would push IMMA into a regime where it loses
to dp4a anyway (per the v1 microbench at M = 16: v1 was 1.94×; that
ratio inverts well below MIN_M = 32).

### 5.6 Q4_K reordering complexity at load time

The one-shot `mmq_q4k_imma_layout` kernel must:

1. Walk each 144-byte Q4_K super-block.
2. Unpack the 6-bit sub-block scales / mins from `scales[12]` (the
   cross-byte layout in `vec_dot_q4_K_q8_1:194-203`).
3. Compute `eff_alpha[n, sub] = d * sc[sub]` and
   `eff_beta[n, sub] = 8 * d * sc[sub] - dmin * m[sub]`. Store both
   as FP16 in `[N, K/32]`.
4. For each `q4 ∈ qs[128]`: decode the high and low nibbles, subtract
   8, write into `w_sym_s8[N, K]` at the position dictated by the
   `ldmatrix.x2` fragment layout.

The reordering is the cognitively heavy part — same complexity as
v2's Phase 1a/1b (`mmq_q4k_v2_phase2_shipped` cites commits `cfdb85a`
+ `e18e5af` for the v2 versions). Estimated ~300 LoC for the kernel +
test, ~200 LoC for the integration into `pre_dequant_weights()`. The
v2 versions can be lifted as a reference for the permute layout, with
the f16 → s8 substitution.

**Not a blocker, but a 2-3 day Phase 2 item.**

### 5.7 Build / nvcc considerations

- INT8 IMMA inline PTX is well-trodden territory — no nvcc bugs
  expected. CUTLASS templates use it heavily; CUDA samples
  (`cudaTensorCoreGemm`) show `m16n8k32.s32.s8.s8.s32` patterns.
- Register pressure: 4× s32 accumulators × 8 warp-repeats = 32 × b32
  registers per warp for the accumulator, plus 4 × b32 (A) + 2 × b32
  (B) per MMA call. Tight but fits — v2 HMMA at similar tile sizes
  ran cleanly.
- No new `launch_bounds` issues expected (v2 hit a `__noinline__`
  spill issue on Q8_0; the symmetric s8 path has fewer intermediate
  values than v2's FP16 dequant).

---

## 6. Implementation plan

| Phase | Scope | LoC est. | Risk | Days |
|---|---|---|---|---|
| 1 | **Microbench** — isolated INT8 IMMA on synthetic Q4_K data. Single tile config `<64, 64, 32, NUM_STAGES=3>`. Pre-materialise `w_sym_s8`, α, β, x_s8, x_scale, x_rowsum **in the bench harness**, not in production paths. Target shapes: M ∈ {32, 64, 128, 256, 512}, N = K = 5120 (Qwen3-32B FFN). **Gate**: ≥ 2× v1 dp4a kernel-time at M = 512 (i.e., ≤ 0.55 ms on the v1's 1.064 ms baseline). Compare to FP16-TC cuBLAS at the same shape. | ~250 (kernel) + ~200 (bench) | low — ptx survey first (~5 min), then 1 kernel + 1 bench harness | 3-4 |
| 2 | **Production kernel + dispatch** — wire into `gemm_dispatch_impl` and the new `GemmKernel` registry. Add load-time `mmq_q4k_imma_layout` kernel to populate `WeightCaches::q4k_imma`. Dispatch in M ∈ [32, 256] initially (avoid the MoE-expert range below 32 and the very-large-M range where cuBLAS FP16-TC bandwidth-saturates). Skip when `fp16_cache` hits the weight. | ~400 (kernel) + ~300 (layout) + ~200 (dispatch) | medium — three integration points, all tested by existing executor tests | 4-5 |
| 3 | **E2E A/B** — run `profile_pp512_large_dense.sh` on Qwen3-32B Q4_K_M and Gemma-3-12B Q4_K_M. Expect: pp512 +15-30 % on Qwen3-32B, +10-20 % on Gemma-3-12B. **Gate**: ≥ +10 % pp512 on at least one of the two; no regression on any other Q4_K_M model in the verify-fast suite. | ~100 (test additions) | low — measurement only | 1-2 |
| 4 (optional) | **MoE expansion** — only if Phase 3 shows headroom in dense models. Lower the MIN_M threshold for IMMA selectively when the expert weight is large enough that the IMMA fixed cost amortises. Requires per-(expert, batch) cost model. | ~300 | medium-high — touches MoE dispatch hot path | 3-5 |

**Total Phases 1-3**: ~1.5 weeks. Comparable to v2's per-phase cost
but with a more definitive go/no-go at Phase 1.

### Tile autotuner

`<BLOCK_M, BLOCK_N, NUM_STAGES>` candidates worth sweeping in Phase 2:

| BLOCK_M | BLOCK_N | NUM_STAGES | SMEM | Why |
|---|---|---|---|---|
| 64 | 64 | 3 | ~12 KB/block | baseline, mirrors v2 Phase 7 default |
| 64 | 128 | 3 | ~18 KB/block | wider N for FFN shapes |
| 128 | 64 | 3 | ~18 KB/block | wider M for prefill |
| 64 | 64 | 4 | ~16 KB/block | extra stage for DRAM latency |
| 32 | 32 | 3 | ~6 KB/block | small-tile fallback near M=32 boundary |

v1's auto-tuner default (`<16, 32, 1, 1>`) doesn't apply — that was a
dp4a tile, not a TC tile.

---

## 7. Decision recommendation

**Ship Phase 1 microbench.**

Justification: a ~3-4 day standalone microbench definitively answers
whether INT8 IMMA on sm_120a can clear the 2× v1-dp4a bar that v2
HMMA cleared at 4.87×; if it does, the e2e payoff zone is real on
the two qualifying dense Q4_K_M models (Qwen3-32B, Gemma-3-12B), and
if it doesn't, we stop without paying the multi-week v2 cost again.

---

## Cross-references

### Memos (auto-loaded from project memory)

- `mmq_q4k_phase_a_2026_05_15.md` — v1 dp4a tiled kernel that won
  +13-56 % at M = 2..16 on Gemma-3-12B Q4_K_M. Retired alongside v2
  in PR #199 (per §1.2, the v1 files were never on main; only the
  squashed v2 made it through PR #189).
- `mmq_q4k_v2_hmma_design_2026_05_15.md` — v2 HMMA design memo;
  blueprint for tile layout, cp.async pipelining, weight permutation.
- `mmq_q4k_v2_phase2_shipped_2026_05_16.md` — v2 microbench +
  end-to-end results, including the -4 % e2e regression on Qwen3.6-
  35B-A3B that motivated retirement.
- `q4k_mmvq_crossover_2026_05_15.md` — original mmvq-vs-cuBLAS
  crossover measurement at M ≈ 16.
- `sm120_ptx_capability_map_2026_04_26.md` — master PTX capability
  survey; does not cover INT8 IMMA explicitly because it predates the
  v2 / IMMA design work.
- `ptx_mma_survey_2026_04_26.md` — focuses on FP4/FP6/FP8 `kind::*`
  MMA variants. INT8 `m16n8k32` is universal-since-Turing and not
  re-probed here.
- `hw_capability_audit_complete_2026_05_10.md` — confirms 4638 ×
  `IDP.4A.S8.S8` SASS instructions (scalar dp4a path) and 1898
  × `HMMA.16816.F32` (FP16 TC path). Notably, **zero** `IMMA.*S8`
  TC SASS in the production cubin today — INT8 TC is unused.

### Code

- `src/compute/ggml_mmvq.cu` — current Q4_K × Q8_1 path
  (`mmvq_kernel`, `vec_dot_q4_K_q8_1`, `quantize_fp16_to_q8_1_ggml_kernel`).
  Reference for the Q4_K layout (lines 18-24), Q8_1 layout (37-42),
  and the existing FP16 → Q8_1 activation quantiser (226+).
- `src/compute/gemm_dp4a.cu` — scalar dp4a GEMV (`dispatch_dp4a_gemv`,
  M = 1 path). Confirms the dp4a peak baseline.
- `src/graph/executor_kernels.cu:2200-2300` — current Q4_K dispatch
  site. The IMMA hook would go between `use_dp4a` (M=1) and the
  `dequant_gpu → cuBLAS` fallback.
- `src/graph/gemm_kernel_registry.h` + `gemm_kernel_registry.cu` —
  R5 Slice 1+ registry; the IMMA kernel registers a new
  `StorageTier::Q4K_IMMA` (or extends `StorageTier::Q4K`) strategy.
- `src/graph/executor_pre_dequant.cu` — model-load-time cache
  population. The `mmq_q4k_imma_layout` kernel would be invoked here,
  alongside the existing `convert_*` paths.
- `tools/analysis/ptx_mma_survey.sh` — extend with one s8 IMMA case
  as the Phase 1 pre-flight.
- `tools/analysis/profile_pp512_large_dense.sh` — the Phase 3 e2e
  gate.

### External

- PTX ISA 8.5, §9.7.13.4 "Matrix Multiply-Accumulate Instructions"
  (NVIDIA documentation; `m16n8k32.s32.s8.s8.s32` operand layout +
  fragment-to-register mapping).
- PTX ISA 8.5, §9.7.13.5 "ldmatrix" (b16 mode, applies to our s8 tile
  loading via 16-bit element packing).
- llama.cpp's Q8_K_R8 activation format (per-block row-sum field for
  affine-bias coupling) — reference for the `x_rowsum` design in §3.3.
- CUTLASS `include/cutlass/arch/mma_sm75.h` — reference INT8 IMMA
  fragment definitions and PTX inline-asm patterns.
- NVIDIA/tilus `examples/quantization/matmul_a16wx.py` — the v2
  blueprint; same structure carries to IMMA with the f16 → s8 swap.

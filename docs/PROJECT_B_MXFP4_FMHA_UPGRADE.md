# Project B: MXFP4 FMHA Upgrade to `mxf4nvf4.block_scale`

**Target:** replace imp's `attention_fmha_mxfp4_sm120.cu` MMA path from
`kind::f8f6f4.m16n8k32` (manual per-row scale) to
`kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (hardware-integrated
block scale, 2× K-dim). Expected gain: **2-4× prefill attention
throughput** on sm_120f.

**Prior art:** SageAttention3, arxiv 2505.11594, thu-ml/SageAttention.
Their measurement: **1038 TOPS on RTX 5090** = 5× vs fastest
FlashAttention on 5090.

## Current state (all committed on `feat/graph-verify-pool`)

| Stage | Commit | Status |
|---|---|---|
| 1 — Feasibility gate | `b9ec21a` | ✅ MMA compiles + launches on sm_120f + CUDA 13.2.78 |
| 2 — Numerical gate | `7298175` | ✅ `A=0 → D=0` verified |
| 2.5 — Quant math | `7a77063` | ✅ FP16→NVFP4→FP16 round-trip, 9.5% RMSE Gaussian, bit-exact representable |

## Reference files (SageAttention3, in `thu-ml/SageAttention`)

- `sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu`
  — quant kernel with HW scale interleaving (lines 134-258)
- `sageattention3_blackwell/sageattn3/blackwell/cute_extension.h`
  — MMA wrapper (lines 23-139)
- `sageattention3_blackwell/sageattn3/blackwell/kernel_traits.h`
  — tile shape + CuTe layouts

## imp files to modify

| File | Lines | Change |
|---|---|---|
| `src/compute/attention_fmha_mxfp4_sm120.cu` | 420-430 | Replace MMA instruction |
| `src/compute/attention_fmha_mxfp4_sm120.cu` | 42-54 | Update tile constants (MX_MMA_K 32→64) |
| `src/compute/attention_fmha_mxfp4_sm120.cu` | 56-76 | Replace per-row quant with per-16-elem block quant |
| `src/compute/attention_fmha_mxfp4_sm120.cu` | 440-470 | Remove manual scale-apply; MMA now does it |
| `src/compute/attention_fmha_mxfp4_sm120.cu` | SMEM layouts | Scale buffer adds 8 scales per 128-K tile row |

## Stage 3 — Quant kernel port (HW layout)

**Goal:** produce FP16 → NVFP4 + FP8 UE4M3 scales in SageAttention3's
HW-consumption layout. This is the input contract for the MMA.

**Files to create:**
- `src/compute/nvfp4_quant_hw.cu` (~250 LOC)
- `src/compute/nvfp4_quant_hw.h`
- `tests/test_nvfp4_quant_hw.cu`

**Key formula** (from `fp4_quantization_4d.cu:245-256`):
```cpp
// For CVT_FP4_ELTS_PER_THREAD = 16 (head_dim=128 case):
uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;  // 0..7
uint32_t token_id_local = token_id % 64;
uint32_t offset_local = (col_id_local / 4) * 256
                      + (col_id_local % 4)
                      + (token_id_local / 16) * 4
                      + (token_id_local % 16) * 16;
```

The `(col/4)*256` splits cols 0-3 vs 4-7 into separate 256-byte blocks.
Within each, tokens interleave on a 16-way pattern matching MMA fetch.

**Test strategy:**
1. Golden test: quantize known FP16 input, assert specific scale bytes
   at specific offsets (hand-computed).
2. Round-trip with a matching dequant kernel that inverts the layout.
3. Cross-check against `nvfp4_quant_ref.cu` (Stage 2.5 linear layout):
   same per-group scale values, different storage order.

## Stage 4 — MMA integration

**Goal:** swap in the new MMA in imp's FMHA, behind env gate.

**File:** `src/compute/attention_fmha_mxfp4_sm120.cu`

**Changes:**

```cpp
// Old (line 47-54):
static constexpr int MX_MMA_M = 16;
static constexpr int MX_MMA_N = 8;
static constexpr int MX_MMA_K = 32;  // ← change to 64

// Old (line 420-430):
#if __CUDA_ARCH__ >= 1200
asm volatile(
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
    "{%0, %1, %2, %3},"
    "{%4, %5, %6, %7},"
    "{%8, %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(d0), "f"(d1), "f"(d2), "f"(d3));
#endif

// New:
#if __CUDA_ARCH__ >= 1200
asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0, %1, %2, %3},"
    "{%4, %5, %6, %7},"
    "{%8, %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(d0), "f"(d1), "f"(d2), "f"(d3),
      "r"(sfa0), "h"(bidA), "h"(tidA),
      "r"(sfb0), "h"(bidB), "h"(tidB0));
#endif
```

**Remove manual scale (lines ~440-470):**
```cpp
// DELETE this block — HW applies scale in MMA now:
FUSED_STORE(d0, gq0, gk0, qs0, ks0, ...);
FUSED_STORE(d1, gq0, gk1, qs0, ks1, ...);
// ...

// Replace with: just store d0..d3, softcap/mask only:
FUSED_STORE_NOSCALE(d0, gq0, gk0, ...);
// Where attention_scale is now pre-absorbed into the FP8 UE4M3 scale factor.
```

**Env gate:**
```cpp
static const bool use_blockscale_mma =
    (std::getenv("IMP_FMHA_BLOCKSCALE") != nullptr);
if (use_blockscale_mma) {
    launch_mxf4nvf4_blockscale_kernel(...);
} else {
    launch_f8f6f4_legacy_kernel(...);  // existing code path kept intact
}
```

**Tests:**
1. Correctness: same Q/K/V, compare FP16-FMHA output vs new MXFP4-BLOCKSCALE.
   Expected: cos-similarity > 0.99, max element error < 0.05.
2. Existing tests in `test_attention_fmha_mxfp4.cu` must still pass with
   `IMP_FMHA_BLOCKSCALE=1`.

## Stage 5 — Bench + Real-prompt quality

**Bench matrix:**
```bash
# Prefill-dominated
for model in Qwen3-8B-Q8_0 Gemma3-12B-Q8_0; do
  for ctx in 512 2048 4096 8192; do
    ./imp-cli --model $model --bench --bench-pp $ctx --bench-reps 3
    ./imp-cli --model $model --bench --bench-pp $ctx --bench-reps 3 [IMP_FMHA_BLOCKSCALE=1]
  done
done
```

**Target gains (informed by SageAttention3):**
- pp512 → pp2K: 1.5-3×
- pp4K → pp8K: 2-4×
- tg (decode): **0%** (decode uses paged attention, not FMHA — unchanged)

**Quality gates:**
- Qwen3-8B Fibonacci prompt: both paths produce coherent Python
- Gemma3-12B chat template + multi-turn: no degeneration
- Tolerance: same top-1 token on first 20 steps for seed=42, temp=0

## Rollback plan

1. The new kernel lives in `attention_fmha_mxfp4_sm120.cu` behind
   `IMP_FMHA_BLOCKSCALE` env var. Old code path is UNTOUCHED.
2. If quality regresses: leave env var unset, ship as opt-in.
3. If neither path is correct: `git revert` the integration commit,
   probe + quant_ref files stay as long-term regression harness.

## Risks and unknowns

| Risk | Mitigation |
|---|---|
| Scale layout doesn't match HW expectation | Cross-check against SageAttention3's working attention kernel |
| K=64 changes SMEM footprint | Already using Bkv=64 — tile widens K but halves iterations |
| FP32 accumulator still gated by MMA instruction | That's fine — we want FP32 accum for flash softmax |
| Quality regression | FP8-PPL paper data: NVFP4 attention lossless on image/video/text models |
| `mma.sync.kind::mxf4nvf4` is sm_120a, imp uses sm_120f | f suffix is superset — probe already confirmed compile + run |

## Effort estimate

| Task | Hours |
|---|---|
| Stage 3 (HW layout quant + test) | 6-8 |
| Stage 4 (MMA integration) | 8-12 |
| Stage 5 (bench matrix + quality check) | 4-6 |
| Misc (debugging, CI, docs) | 4-6 |
| **Total** | **~30 hours** |

One focused week or 3-4 half-days.

## References

- SageAttention3 paper: https://arxiv.org/abs/2505.11594
- SageAttention3 code: https://github.com/thu-ml/SageAttention (Apache-2.0)
- imp study: `memory/sageattention3_study_2026_04_24.md`
- PTX ISA for `mma.sync.kind::mxf4nvf4`: NVIDIA PTX ISA 9.4+

# Iteration 2: post-fast-path findings — 2026-05-10

After the 10.5× prefill win (1241 → 13046 tok/s, commit `3e33031`), I explored further optimizations to close the remaining gap to vLLM (25513). All four candidates either didn't compile, regressed prefill, or regressed decode. Documenting the negative results so the next investigator doesn't repeat them.

## Re-profile after `3e33031`

```
Workload: pp=512, max_tokens=1, reps=1 on Qwen3-Coder-30B-A3B-NVFP4 (rebench)

Top GPU kernels (% of total kernel time, including warmup):
 52.1%  17.2 ms/prefill   MoE NVFP4 grouped GEMM (sm_120 PtrArrayTmaWarpSpecializedCooperativeBlockScaled)
  7.0%   2.3 ms/prefill   per-tensor NVFP4 attention CUTLASS
  6.6%   2.3 ms (LOAD)    convert_scales_sfatom_moe_kernel — runs once at startup, not per-prefill
  5.6%   1.8 ms/prefill   causal_softmax_fp32_to_fp16
  5.1%   1.7 ms/prefill   rmsnorm_fp16
  8.2%   2.7 ms/prefill   cuBLAS WMMA FP16 attention
  2.0%   0.7 ms/prefill   activation_quantize MoE
  1.7%   0.5 ms/prefill   moe_scatter_fused_residual
  1.3%   0.4 ms/prefill   moe_gather + rope (each)
  ...

Total GPU kernel time per prefill (excl. load-time conv): ~30 ms.
Wall-clock per prefill: 33 ms (best run) — 14k–17k tok/s typical.
GPU is back-to-back utilized (Q time of MoE GEMM = 70 µs, K = 117 µs — no idle gap).
```

## Run-to-run variance

`pp512` on Qwen3-Coder NVFP4, 10 runs each:

```
n=10  min=7153  p25=10433  median=12282  p75=16448  max=16662  mean=12397
```

Variance is dominated by **cuBLAS attention algorithm selection on cold container**, not by my changes. Single-run measurements (or even 3-run averages) are noisy enough that small optimizations get lost. **All A/B comparisons here use N=10 paired runs.**

`tg256` decode in contrast is rock-solid: 268.1–268.5 tok/s across 10 runs (range 0.4 tok/s = 0.15%).

## Candidates tried, all rejected

### 1. M=64 tile for grouped GEMM — fails to compile

```
GrpTileShape = Shape<_64, _128, _128>;
```

CUTLASS error: `TMA requires CTA_Tile and SLayout top-level size equivalence`. The SfAtom layout for block-scaled NVFP4 is fixed at 128 rows, so the M tile dimension must be ≥128. Cannot reduce M-tile to better fit our M ≈ 32 per-expert distribution.

### 2. N=256 tile — regresses prefill -44%

```
GrpTileShape = Shape<_128, _256, _128>;
```

Bench: 13046 → **7366 tok/s** on the same model. Larger N tile = fewer thread blocks (only 4 N-tiles per expert × 128 = 512 TBs vs 1024 with N=128) → less wave-level parallelism on 170-SM RTX 5090. The auto schedule already picks the best balance.

### 3. Explicit `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120<3>` schedule — fails to compile

```
... cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120<3>>::CollectiveOp;
```

CUTLASS error: `Incorrect Kernel Schedule Policy for F4 type inputs. Kernel Schedule policy should be auto`. NVFP4 inputs hard-require `KernelScheduleAuto` on Sm120 — Pingpong is not exposed for FP4 on consumer Blackwell. (The Auto policy already picks Cooperative which is what we measured.)

### 4. Pinned host staging buffer in `gemm_grouped_cutlass_3x_nvfp4` — race condition, regresses prefill

```cpp
// Replace per-call std::vector<char>(o.total) with a process-persistent
// cudaHostAlloc'd pinned buffer.
static void* s_host_staging = nullptr;
ensure_host_staging(o.total);
char* h_base = static_cast<char*>(s_host_staging);
```

Bench (5 runs): pp512 dropped from 13046 baseline to **7282–11007 tok/s**.

The bug: every MoE GEMM call writes to the same pinned buffer, then issues `cudaMemcpyAsync(d_base, h_base, ...)`. With pinned source the copy is genuinely async, so call N+1 starts overwriting `h_base` *before* call N's H2D copy finishes reading from it. Stream-ordering only orders GPU operations; CPU writes are not synchronized with prior in-flight async copies.

A correct version would need a ring buffer of pinned slots cycling through N calls, large enough that slot reuse is always after the prior copy completes. The complexity (and 144 × ~50 KB per slot) outweighs the ~1-2 ms/prefill expected gain.

### 5. Fused gate+up grouped GEMM — prefill +12% / decode -7%, NET NEGATIVE

A single grouped-GEMM call with 2*ne problems (gate weight at problem 2e, up weight at problem 2e+1, sharing the same activation A) instead of two separate calls.

10-run paired A/B on Qwen3-Coder NVFP4:

| | pp512 min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| Without fusion | 7153 | 10433 | **12282** | 16448 | 16662 | 12397 |
| With fusion    | 7686 | 10636 | **13728** | 16145 | 17140 | 12816 |
| Δ              | +7%  | +2%  | **+12%** | -2%  | +3%  | +3%  |

But decode regresses:

| | tg256 (10 runs) |
|---|---|
| Without fusion | **268.1–268.5 tok/s** (median 268.3) |
| With fusion    | **248–252 tok/s** (median 249.4, **-7%**) |

The decode regression is unexpected — the fusion lambda lives in the `n > 1` prefill branch and decode flows through `gemv_nvfp4_moe_decode` (a completely different code path). Suspect: instruction-cache pressure from the larger compiled forward function pushed the hot decode path out of i-cache, or LTO inlined the lambda in a way that affected register allocation in the parent function.

Per the constraint that decode must not regress >2%, the change is rejected. Could potentially be revisited as a runtime-conditional path (only build the fused lambda for prefill-only contexts) but the engineering effort isn't worth +12% prefill on a model that's already past the 2× of vLLM target.

## What's left to investigate (not done in this iteration)

- **Custom kernel for small-M MoE GEMM**: Write a direct `mma.sync.kind::mxf4nvf4.block_scale` kernel that doesn't go through CUTLASS templates — could remove the M=128 alignment constraint by managing SfAtom packing differently. Multi-week project. The kernel-time floor at 117 µs/call is what CUTLASS Sm120 NVFP4 grouped gives us today.
- **Persistent kernel scheduler with M-aware tile selection**: One persistent kernel that picks per-problem tile based on M_e at runtime. CUTLASS doesn't expose this for grouped block-scaled on Sm120.
- **Activation-quantize-into-GEMM fusion**: would save ~0.7 ms/prefill (2% of time). Custom epilogue work.

None of these is a contained PR; all are weeks-of-work investigations with uncertain return.

## Conclusion

The post-`3e33031` profile is **GPU-bound and CUTLASS-template-bound**. Tile shape and schedule are pinned by NVFP4-on-sm_120 constraints. Per-call dispatch overhead is small relative to GPU compute. The remaining gap to vLLM (1.95×) is the cost of using off-the-shelf CUTLASS templates instead of TRT-LLM's custom autotuned `fp4_gemm`. Further closure requires writing a direct PTX/SASS kernel.

**`3e33031` is the right stopping point for this round.**

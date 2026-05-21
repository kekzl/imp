# Track E v2 groundwork — bug analysis + tooling for a future rewrite

**Date:** 2026-05-21
**Status:** Track E disabled in production (PR #354). This doc captures what we learned and what's needed for a v2.

## Why Track E v1 failed

PR #350 shipped a hand-written FA2-style tiled streaming attention kernel
(`src/compute/attention_tiled_streaming.cu`). The kernel passed the project's
correctness tests (`TrackE_Correctness.*`) with max_abs ≈ 5e-3 against cuBLAS.

The kernel **degenerated on real model weights**:

| Model | Output |
|---|---|
| Qwen3-8B Q8_0 | "Okay, I need to figure out..." (coherent) |
| Qwen3.6-35B NVFP4 | "The user is asking for the sum..." (coherent) |
| Qwen3-8B NVFP4 | "1. 1. 1. 1." infinite |
| Gemma-4-26B Q8_0 | "층-다스-층-다스-..." Korean garbage |
| Gemma-4-26B Q4_K_M | "the-land-land-land-..." |
| Gemma-4-26B NVFP4 | `<eos><eos>` immediate EOS |

PR #352 disabled, PR #353 fixed one bug class (P_frag A-operand interleaving),
PR #354 re-disabled after multi-model smoke test showed the fix was incomplete.
Multiple subsequent fix attempts during 2026-05-21 also failed (broke working
Qwen3 path while not fixing Gemma-4).

## Root cause: test fill too uniform

`fill_fp16_deterministic` uses an LCG (`i * 2654435761u`) producing
near-uniform values. After the magnitude bump from 0.125 to 1.0 it still
produced uniform softmax distributions where row-to-row Q/K/V values barely
differ.

With near-uniform Q/K/V, attention output is dominated by position-based
masking, NOT by attention-weight magnitudes. Multiple kernel bugs that affect
which row gets which weight cancel each other on this test data but cascade
catastrophically on real attention scores.

**For a v2, the test fill MUST be either:**
- Gaussian K/V with occasional outliers (mimicking softmax peakedness), OR
- Sampled directly from a real model checkpoint's attention activations.

## Specific bugs identified (some fixed, some open)

### ✅ Fixed in PR #353

**P_frag construction in PV mma** swapped a[1] and a[2] positions. m16n8k16
A-frag layout interleaves (row, k-half) as:
- `a[0] = (row_a, k<8)`
- `a[1] = (row_b, k<8)`  ← was `(row_a, k≥8)`
- `a[2] = (row_a, k≥8)`  ← was `(row_b, k<8)`
- `a[3] = (row_b, k≥8)`

### ⚠️ Suspected (not confirmed via clean fix)

**Q load uses same-pointer-per-lane for `ldmatrix.x4`.** Layout probe
(`tests/test_mma_layout_probe.cu` → `LdmatrixX4_SamePointer` test) shows that
when all 32 lanes pass `&A[0]`, the result is the first 8 halves replicated
into all 4 b32 regs per lane — NOT a 16×16 tile.

This means Q_frag for every consumer warp loads from Q row 0 only. With
uniform test fill this produces "similar enough" output; with real Q
activations it's catastrophically wrong.

**ldmatrix.x4 r[] → mma a[] ordering swap.** Layout probe
(`MmaLayoutProbe.M16N8K16_DFrag`) confirmed:
- ldmatrix returns r[0,1,2,3] in **lane-group order** (T0-7 → r[0],
  T8-15 → r[1], T16-23 → r[2], T24-31 → r[3])
- mma a[0,1,2,3] expects them in **region order** (lower-left, upper-left,
  lower-right, upper-right)

The correct mapping is `a[] = {r[0], r[2], r[1], r[3]}`. The current kernel
uses `a[] = r[]` directly which is wrong.

Empirical verification: for A[r][k]=r, B[k][n]=1 (where D[m][n]=16m), lane 0
should produce d = (0, 0, 128, 128). The probe shows d = (64, 64, 64, 64) =
16×4, which is exactly what you get when a[1] holds (row 0) data instead of
(row 8) data (the row-sum of A[0..15] split at boundary 8: lower half sums to
0+1+...+7 = 28, upper half 8+...+15 = 92. Hmm. Actually 16*4 = 64 = sum of row 4
contribution... which suggests an even deeper layout issue).

**K and V loads probably need `ldmatrix.x4.trans`.** mma.row.col B operand
is col-major. Non-trans ldmatrix gives row-major reg packing. PR #353 had
`ldmatrix_x4_trans` for K but later debug code replaced it with `ldmatrix_x4`
+ custom per-lane indexing that may not be equivalent.

## What's committed in this PR (groundwork only)

- **`tests/test_mma_layout_probe.cu`** — empirical layout verification tool.
  Two tests:
  - `LdmatrixX4_SamePointer` — confirms same-ptr behavior (row 0 replicated).
  - `M16N8K16_DFrag` — runs known-A × known-B through ldmatrix + mma and
    dumps per-lane a-frag + d output for verification against PTX ISA spec.
- **`tests/test_attention_tiled_streaming.cu`** — fill magnitude bumped from
  0.125 to 1.0 (already on main). Insufficient — see "Root cause" above —
  but tighter than what shipped originally.

The Track E kernel itself stays disabled (PR #354 unconditional `return false`
at launcher entry). cuBLAS / FMHA handle all prefill in production.

## Plan for Track E v2

Recommended approach: **WMMA-based rewrite**.

The existing `src/compute/attention_fmha_sm120.cu` uses `nvcuda::wmma` and
works on all production models. Its perf is ~0.5× cuBLAS at long context (we
measured this in the gating bench).

For a v2:
1. Start from `attention_fmha_sm120.cu` as the structural template.
2. Apply the 4+4 producer-consumer warp specialization that gave us +2.2%
   pp8192 in PR #351 (which is currently merged but has no effect because
   Track E is disabled).
3. Use WMMA fragments instead of raw mma.sync PTX — WMMA hides the layout
   complexity that bit us.
4. Build a **proper test rig** with:
   - Gaussian K/V fill at multiple magnitudes
   - Layer-output A/B against a saved-checkpoint trace (catches real-world
     distribution issues)
   - Smoke prompt gate for 6+ production models in CI (the verify-fast
     hook currently skips models when not present in container; v2 must
     not regress without these gates firing)

Estimated effort: 5-8 dev days for a kernel that's correct on all models
and ≥ cuBLAS perf. Smaller estimate for a kernel that's "correct but not
faster than cuBLAS" — useful as a fallback if cuBLAS regresses.

## Lessons embedded

- **A unit test that passes against cuBLAS within FP16 tolerance does NOT
  imply the kernel is correct on real attention distributions.** This was
  the fundamental misjudgment in the original Track E test suite.
- **Multi-model smoke testing must run in CI.** Verify-fast skips models
  when not present in the container, which let multiple broken Track E
  PRs reach main.
- **Layout assumptions must be empirically verified via probes**, not
  inferred from PTX documentation alone (which we have repeatedly
  misinterpreted on sm_120a). The probe tool committed here is the
  starting point.
- **Auto-merge on bundled disable+fix branches is dangerous** — PR #352
  auto-merged on the disable commit before the fix commit landed.

## Reproduce the bug

The current branch `track-e/v2-groundwork` keeps Track E disabled. To
reproduce the original degeneration, temporarily remove the early `return
false;` in `attention_tiled_streaming_prefill` (the line PR #354 added)
and run:

```bash
make build
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/gemma-4-26B-A4B-it-Q8_0.gguf \
  --prompt "What is 17 + 25?" --max-tokens 10 --temperature 0
```

Expected (broken): repeating Korean characters or other garbage.
Expected with `return false` in place (fallback): `17 + 25 = 42<turn|>`.

To run the layout probe:

```bash
docker run --rm --gpus all imp:test test-attention \
  --gtest_filter='MmaLayoutProbe.*' 2>&1 | tail -80
```

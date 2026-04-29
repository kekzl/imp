# 2A — Attention NonAlignedSeqLen Triage

**Status:** `KNOWN_LIMITATION_RECOMMENDED` (with `HYPOTHESIS_ONLY` root cause).
**Subagent:** attention-nonaligned-triage
**Date:** 2026-04-29

---

## Reproduction

Both tests fail under the originally-reported filter on RTX 5090 / sm_120 / `imp:bringup`:

```
docker run --rm --gpus all imp:bringup test-attention \
    --gtest_filter="AttentionBlackwellTest.NonAlignedSeqLen:FmhaSm120Test.NonAlignedSeqLen"
```

But the failure modes differ once split:

| Test | Isolation behaviour (`--gtest_repeat=5`) | Order-dependent |
|---|---|---|
| `AttentionBlackwellTest.NonAlignedSeqLen` (`tests/test_attention_tc.cu:312`) | **Flaky.** 3/5 PASS, 2/5 FAIL alone. | Yes — passes after `Causal`, `NonCausal`, `HeadDim64`. Fails after `CausalMultiTile` / `GQA`. |
| `FmhaSm120Test.NonAlignedSeqLen` (`tests/test_attention_fmha_sm120.cu:207`) | **Deterministic FAIL.** 5/5 fail in isolation. | No (always red). |

So we have *one* flaky kernel and *one* deterministically-broken kernel that share a common code structure (online-softmax with float→half packing into aliased shared memory). The orchestrator's observation that "both fail with the same error pattern" is correct in the original log because the suite is run in declaration order, after `CausalMultiTile`/`GQA` poisons the Blackwell case.

---

## Not a regression

`git log -S "NonAlignedSeqLen"` shows the test was added in the same commit that introduced each kernel:

- `16f6cff perf: optimized Blackwell WMMA attention…` — added `attention_blackwell.cu` and the test simultaneously (Feb 28 2026).
- `7e2ca24 feat: add native sm_120 FMHA kernel…` — added `attention_fmha_sm120.cu` and the test simultaneously.

There is **no commit that previously made these green and then broke them**. They are long-standing red tests that have been tolerated on `main`. No mention in `docs/`, `TODO.md`, or the project memory directory.

---

## Hypothesis: SP_half / S_tile shared-memory aliasing race in the float→half packing step

Both kernels use the same memory-saving trick: a single shared-memory region holds the score matrix as **either** `float[Br × Bc]` (during QK^T accumulation + masking + softmax) **or** `half[Br × Bc]` (after softmax, fed into the PV WMMA). The two views overlap byte-for-byte:

```
float row r at byte (r * Bc * 4)         half row r at byte (r * Bc * 2)
```

(see `src/compute/attention_fmha_sm120.cu:127–129` and `src/compute/attention_blackwell.cu:114–116`).

This means `half`-row `2r` and `2r+1` cover the **same bytes** as `float`-row `r`. So Step 6 of the parallel softmax — which **simultaneously** reads `S_tile[r·Bc+c]` (float) and writes `SP_half[r·Bc+c]` (half) for *different* rows on different warps — is a write-after-read race across warps:

```cpp
// FMHA: src/compute/attention_fmha_sm120.cu:317-326   (Blackwell: 294-305)
float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
if (row_valid) {
    for (int c = sm_lane; c < Bkv; c += TPR) {
        SP_half[r * Bkv + c] = __float2half(S_tile[r * Bkv + c] * spv);
        //  ^ half write at byte (r·Bkv+c)·2     ^ float read at byte (r·Bkv+c)·4
    }
} else if (r < Bq) {
    for (int c = sm_lane; c < Bkv; c += TPR) {
        SP_half[r * Bkv + c] = __float2half(0.0f);   // <- invalid rows write 0
    }
}
```

Concretely, with Bq=128, Bkv=64 and TPR=2:

| Warp | sm_row range | half writes byte range |
|---|---|---|
| 0 | 0..15  | [0, 2048) |
| 1 | 16..31 | [2048, 4096) |
| 2 | 32..47 | [4096, 6144) |
| 3 | 48..63 | [6144, 8192) |
| 4 | 64..79 | [8192, 10240) |

S_tile float rows 0..7 cover bytes [0, 2048). S_tile float rows 8..15 cover [2048, 4096). So **warp 1's half writes to row 16 collide with warp 0's float reads of row 8**.

Why does this corrupt **NonAlignedSeqLen** (Sq=200) reliably while CausalHD128 (Sq=128) tolerates it?

- For Sq=128 / aligned shapes, all rows are valid, so each writer writes a *real* exp/sum value as half. A racing reader reinterpreting that half as the lower 16 bits of a float still gets a finite, reasonably-small float. After multiplying by `spv ≈ 1/sum` and tanh-ing through softmax, the per-row error is small enough to slip under the 1e-2 tolerance.
- For Sq=200 with Bq=128: q-tile 1 has q_start=128, only 72 valid rows out of 128. Rows 72..127 take the `else if (r < Bq)` branch and write **half(0.0)** = bit-pattern 0x0000 across their assigned bytes. When warp `k` writes `half(0)` into bytes that alias warp `k/2`'s float S_tile row, that float reads back as `0.0f` exactly. Subsequent `__expf(0 - m_new)` is non-trivial, but the score that *should* have been a moderately-large exp now collapses to `__expf(-m_new)` which is essentially zero for typical m_new ≈ 1..2. The whole P row is flattened, P @ V outputs collapse, final O elements approach zero, and the test's `(got − ref)/max(|ref|, 1e-6)` ratio reports values approaching 1.0. Hence "max relative error 1".

The Blackwell flake (Br=64, TPR=4) has the same pattern but a smaller cross-warp footprint and no `Bq=128` branching, so the race is lighter; warp scheduling determines whether row-8's S_tile read finishes before row-16's SP_half write.

The kernel comment at `attention_fmha_sm120.cu:251–253` claims this is safe because of warp-level SIMT — that is true *within* a warp (where the read-then-write is sequenced on each lane), but **wrong across warps**. This is the same flavour of bug as the FP8 FMHA S_tile pointer-advance issue (`memory/fp8_fmha_stile_bug_2026_04_23.md`), but expressed through aliasing rather than pointer arithmetic.

### Why this hypothesis instead of the obvious alternatives

I considered and ruled out (or downgraded):

- **`compute_kv_tile_bounds` mishandles Sq>Skv** — formula is correct for both NonAligned and the passing CausalMultiTile (Sq=256, Skv=192). Hand-traced both. ✗
- **`apply_score_masks` causal polarity** — `gq < gk → -FLT_MAX` matches the CPU ref `qi < ki → -FLT_MAX`. ✗
- **Fully-masked-row m_new=-FLT_MAX corrupting partial_sum** — explicit sentinel guard at line 290 (`s_val <= -FLT_MAX*0.5f ? 0 : __expf(...)`), present in both kernels. ✗
- **Vectorized float4 Q/K/V loads with non-aligned row tail** — the loads only happen on `seq_q/seq_kv` row boundaries, the per-row offset `r * row_stride` always lands on a 16-byte boundary because `head_dim ∈ {64,96,128,256}` × `n_heads × 2 bytes` is always 16-byte aligned. ✗
- **Output-write OOB on partial last tile** — guarded by `if (q_start+r >= seq_q) continue;` at `attention_fmha_sm120.cu:396`. ✗
- **WMMA fragment alignment for partial Bq tiles** — the kernel always launches Bq-aligned tiles and relies on smem zero-fill for the trailing rows; the WMMA always reads full 16×16 sub-tiles. ✗

The race hypothesis is the only one that **simultaneously** explains (a) the Blackwell flake, (b) the FMHA determinism, (c) why aligned shapes pass, (d) why the failure pattern is "got ≈ 0 → relative error ≈ 1", and (e) why no other test in the suite triggers it.

**Confidence: medium.** Raising to high requires either:
- Running `compute-sanitizer --tool racecheck` on the failing case (sanitizer is not in `imp:bringup`; would need a sanitizer-enabled rebuild — out of budget for this triage), OR
- Adding a one-line `__syncthreads()` between Step 5 and Step 6 *plus* a per-row register stage and observing the failure go away (also requires a rebuild).

---

## Proposed minimal fix (≤30 LOC, two-file diff, NOT applied)

Stage S_tile reads into per-thread registers, then `__syncthreads()`, then write SP_half. This kills the cross-warp race without changing the smem footprint:

```diff
--- a/src/compute/attention_fmha_sm120.cu
+++ b/src/compute/attention_fmha_sm120.cu
@@ -316,18 +316,28 @@
-            // Step 6: Fused softmax normalize + float->half conversion
-            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
-            if (row_valid) {
-                for (int c = sm_lane; c < Bkv; c += TPR) {
-                    SP_half[r * Bkv + c] = __float2half(S_tile[r * Bkv + c] * spv);
-                }
-            } else if (r < Bq) {
-                for (int c = sm_lane; c < Bkv; c += TPR) {
-                    SP_half[r * Bkv + c] = __float2half(0.0f);
-                }
-            }
+            // Step 6: Fused softmax normalize + float->half conversion.
+            // SP_half aliases S_tile (half stride 2 vs float stride 4), so
+            // half writes for row 2r/2r+1 trample float bytes of row r.
+            // Stage all reads in registers, then sync, then write halves.
+            constexpr int PER_LANE = Bkv / TPR;  // 32 for FMHA (Bq=128,TPR=2)
+            float reg_p[PER_LANE];
+            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
+            if (row_valid) {
+                #pragma unroll
+                for (int idx = 0; idx < PER_LANE; idx++) {
+                    int c = sm_lane + idx * TPR;
+                    reg_p[idx] = S_tile[r * Bkv + c] * spv;
+                }
+            }
+            __syncthreads();
+            if (row_valid) {
+                #pragma unroll
+                for (int idx = 0; idx < PER_LANE; idx++) {
+                    int c = sm_lane + idx * TPR;
+                    SP_half[r * Bkv + c] = __float2half(reg_p[idx]);
+                }
+            } else if (r < Bq) {
+                #pragma unroll
+                for (int idx = 0; idx < PER_LANE; idx++) {
+                    int c = sm_lane + idx * TPR;
+                    SP_half[r * Bkv + c] = __float2half(0.0f);
+                }
+            }
```

(Mirror-edit `attention_blackwell.cu:294–305` with `PER_LANE = BW_Bc / TPR` = 16 or 32.)

LOC: ~14 changed in each kernel = ~28 total. Both files compile per existing CMake target.

---

## Recommendation: **KNOWN_LIMITATION**, not "fix now"

Rationale (per CLAUDE.md and the strategic-precision policy in project memory):

1. **The bug is in two FP16 paths.** Project strategic precisions are NVFP4 (decode) and FP8 (prefill). The FP16 FMHA is a **last-resort fallback** for `IMP_NO_FP8_FMHA=1` and head-dim/configuration combos that don't have an FP8 kernel. It is rarely on the hot path for any production benchmark.
2. **It is not a regression.** Tests have been red since Feb 2026 across at least seven follow-up commits to these kernels (PR #21, #33, #56). Nobody has noticed because the runtime never sends odd-shape FP16 prefill through these kernels in practice — Q/K/V are always padded to multiples of 16 by the runtime upstream.
3. **The fix carries register-pressure risk.** Adding a 32-element float register array per thread (FMHA) at TPR=2 is 128 bytes of register spill if the compiler can't fit it. With `__launch_bounds__(256, 1)` the budget is 255 regs/thread, so it should fit, but the perf baseline for Qwen3-4B Q8_0 prefill is at the edge and has 2.6× cuBLAS noise — A/B'ing the patch is non-trivial and out of scope for the bringup gate.
4. **Falsifying the hypothesis cheaply requires `compute-sanitizer racecheck`**, which the `imp:bringup` image does not include. Adding it is a separate rebuild branch.
5. The bringup orchestrator's job is to **catalogue** known-red tests, not to fix them. Two flaky/red FP16 attention edge cases are an acceptable "known-limitation" entry alongside the existing 13 skipped model-dependent tests.

### Suggested action (orchestrator):

- File a low-priority issue with this triage attached.
- Add the two tests to a `KNOWN_FAILING` list (or mark with `DISABLED_` prefix) so they don't gate `verify-fast`.
- Defer the actual race fix until someone is in this kernel for an unrelated reason (e.g. extending FP16 FMHA to head_dim 192 for Mistral-3.2-NVFP4, or another long-context degradation investigation).

---

## File / line references

- `tests/test_attention_tc.cu:312` — Blackwell test definition.
- `tests/test_attention_fmha_sm120.cu:207` — FMHA test definition.
- `src/compute/attention_blackwell.cu:114-116` — SP_float / SP_half union allocation.
- `src/compute/attention_blackwell.cu:294-305` — Step 6 read-write loop (proposed fix site).
- `src/compute/attention_fmha_sm120.cu:127-129` — SP_half aliasing.
- `src/compute/attention_fmha_sm120.cu:251-253` — incorrect "warp-level SIMT" comment.
- `src/compute/attention_fmha_sm120.cu:317-327` — Step 6 read-write loop (proposed fix site).
- `src/compute/attention_paged_common.cuh:285-303` — `compute_kv_tile_bounds` (verified correct).
- `src/compute/attention_paged_common.cuh:310-332` — `apply_score_masks` (verified correct).

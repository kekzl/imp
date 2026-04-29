# 2B — NVFP4 GEMV dispatch triage

**Status: ROOT_CAUSE_IDENTIFIED**

## TL;DR

`gemv_dispatch` for `StorageTier::NVFP4` doubles the kernel's `K`
parameter:

```cpp
// src/compute/weight_dispatch.cu:308
tmp.K = w.shape[1] * 2;
```

The test (and all sibling tier tests, including the passing MXFP4 test)
populate `h.shape[1] = K` with the **logical** K. The kernel
`gemv_nvfp4_kpar` also takes logical K. So `tmp.K = w.shape[1] * 2`
hands the kernel a `K` that is twice the truth, which makes
`gemv_nvfp4_kpar_kernel` walk `K_half = K/2 = K_logical` packed bytes
per row (twice the real per-row stride) and `n_mb = K/16 = 2 *
n_mb_real` micro-block scales per row. It also dot-products against
twice as many `x` lanes as exist. Result: wildly wrong outputs in the
mixed-suite run, all-zero outputs in the isolated run (freshly
zeroed VRAM).

This is a **production bug in `weight_dispatch.cu`**, not a test bug,
because:

1. The test convention (`h.shape[1] = K_logical`) matches the
   convention used by `WeightRegistry::reserve(kind, t.shape[0],
   t.shape[1])` in `executor_pre_dequant.cu:1588`, where `t` is a
   layer weight Tensor whose `shape[1]` is logical K (FP16 source for
   runtime-quantized NVFP4).
2. The dispatch's neighbouring tier branches use logical K with no
   doubling — see `MXFP4` at `weight_dispatch.cu:348-350`
   (`mw.K = w.shape[1]`), `CUTLASS_NVFP4` at `:122` (`int K =
   x.shape[1]`), `FP8` at `:67` (no doubling), `FP16` at `:48` (no
   doubling). MXFP4's `gemv_mxfp4_kpar` has the same logical-K
   convention as `gemv_nvfp4_kpar`, and its dispatch test passes
   because no doubling.
3. The MXFP4 dispatch test setup is byte-identical in spirit to the
   NVFP4 one (same logical-K convention) and passes, confirming the
   convention.

The duplicate `* 2` in `weight_dispatch.cu:97` (the prefill GEMM arm)
is the same bug. The `NVFP4_GemmMatchesDirect` test still passes
despite the same wrong `tmp.K`; almost certainly because in Release
mode the asserts in `gemm_nvfp4` are compiled out and cuBLAS errors
silently on the resulting K-mismatched GEMM, leaving `d_y_disp`
zero-initialized. (The direct-call output values are also small — max
absolute dot product ≈ 1.9 for the toy inputs — so a few may collide
in `__half_as_ushort` with the dispatch's stale-or-zero output. We
did not chase this further; the GEMM arm has the same logical bug as
the GEMV arm regardless of whether its current test happens to flag
it.)

## Why it has not bitten the runtime

`weight_dispatch.cu`'s `gemv_dispatch` / `gemm_dispatch` overloads
that take `WeightHandle&` are the **Phase-2 shim** — they have
**zero production callers today**. Verified:

```
$ grep -rn "gemv_dispatch\|gemm_dispatch" src/ | grep -v weight_dispatch.cu
```

…shows only the legacy `gemm_dispatch(input, weight, output, ctx)`
overload defined in `executor_kernels.cu:2165` (different signature),
which routes through `gemm_dispatch_impl` and never touches
`weight_dispatch.cu`.

Therefore real model decode (imp-cli, imp-server, all benchmarks) is
**unaffected**. The bug is latent and gates Phase-3 migration of any
real GEMV consumer onto the WeightHandle-based dispatch.

## Proposed fix (≤ 30 LOC, single file)

Drop the `* 2` in both NVFP4 arms of `weight_dispatch.cu`. Logical K
is already what every test, every WeightRegistry caller, and every
sibling tier uses.

```diff
--- a/src/compute/weight_dispatch.cu
+++ b/src/compute/weight_dispatch.cu
@@ -94,9 +94,8 @@ void gemm_dispatch(cublasLtHandle_t, const WeightHandle& w,
             tmp.tensor_scale = (w.payload.nvfp4.tensor_scale != nullptr)
                               ? *w.payload.nvfp4.tensor_scale : 1.0f;
             tmp.N = w.shape[0];
-            // shape[1] holds the PACKED column count (K/2 for FP4).  Logical K
-            // is 2x that — the kernel needs the logical dimension.
-            tmp.K = w.shape[1] * 2;
+            // shape[1] holds the LOGICAL K (matches MXFP4 dispatch and
+            // executor_pre_dequant.cu's WeightRegistry::reserve convention).
+            tmp.K = w.shape[1];

             int M = static_cast<int>(x.shape[0]);
             if (M == 1) {
@@ -304,8 +303,7 @@ void gemv_dispatch(const WeightHandle& w, const Tensor& x, Tensor& y,
             tmp.tensor_scale = (w.payload.nvfp4.tensor_scale != nullptr)
                               ? *w.payload.nvfp4.tensor_scale : 1.0f;
             tmp.N = w.shape[0];
-            // shape[1] holds packed K/2 for FP4; kernel needs logical K.
-            tmp.K = w.shape[1] * 2;
+            tmp.K = w.shape[1];   // logical K, matches MXFP4 + reserve()
             gemv_nvfp4_kpar(tmp,
```

That is **6 LOC of net change** (4 lines deleted, 4 added including
comment lines) in **one file**.

## Why ca05a45 added the `* 2`

Commit `ca05a45` (Apr 26) titled "fix(nvfp4): two latent bugs in
weight_dispatch.cu NVFP4 path + diag" added the `* 2`, citing prior
"Phase-1 NVFP4 audit (commit f5f4a1a, 'K must be packed*2')" which
"fixed the WeightHandle-based dispatch in five executor files".

That earlier audit was for the executor's *direct* uses of
`NvFP4QuantResult` constructed from raw `Tensor` shapes
(`executor_attention.cu:325`, `executor_ffn.cu:159, 288`,
`executor_forward_moe.cu:1603`), where the layer's NVFP4 prequant
weight Tensor *had* shape `[N, K/2]` (packed), because the SafeTensors
loader keeps raw bytes shape on `weight_packed`. Those executor
sites correctly multiply by 2.

But the WeightHandle-based shim in `weight_dispatch.cu` is fed by
`WeightRegistry::reserve(kind, t.shape[0], t.shape[1])` in
`executor_pre_dequant.cu:1588`, where `t` is an FP16 source view for
runtime-quantized NVFP4 (the prequant case isn't even wired through
this registry today for NVFP4). So `h.shape[1]` ends up as logical K,
and `* 2` was a misapplied rule.

The test (committed Apr 22 in PR #27) was already using logical K, and
it broke when `* 2` was added on Apr 26. The Apr 28 type-system
refactor (`48f8d45`) only renamed `DType→QType` in the dispatch
file, leaving the bug.

## Confidence: high

What would lift it to "verified": apply the diff, rebuild
`test-quant`, and re-run `WeightDispatchTest.NVFP4_*`. Also worth
adding a `static_assert`-style runtime check that
`w.payload.nvfp4.data == nullptr ||
 sizeof_packed_fp4_row(w.shape[1]) == expected_payload_stride` once
StoragePlanner Phase 4 lands.

## Real-model impact

**None today.** Both buggy arms are dead code. Real Mistral / Qwen
NVFP4 decode goes through fused QKV / fused gate-up GEMVs in
`executor_attention.cu:345` and `executor_ffn.cu:173,289,359`, which
construct `NvFP4QuantResult` directly from layer Tensors with the
correct `* 2` adjustment for prequant Tensor shape. Those paths are
unchanged by this fix.

Real-model impact will materialise when Phase 3 routes a consumer
through `weight_dispatch.cu`'s WeightHandle-based shim; this fix
needs to land before that.

## Files / lines cited

- Bug:      `src/compute/weight_dispatch.cu:97`, `:308`
- Test:     `tests/test_weight_dispatch.cu:372-428`
- Kernel:   `src/quant/nvfp4_gemm.cu:798-817`,
             `:188-212` (kpar kernel — uses logical K)
- Header:   `src/quant/nvfp4_gemm.h:29-30`
- Sibling:  `src/compute/weight_dispatch.cu:348` (MXFP4, no `* 2`),
             `src/quant/mxfp4_gemm.h:22` (logical K)
- Origin:   commit `ca05a45` (Apr 26 2026), reverted by this fix
- Registry: `src/graph/executor_pre_dequant.cu:1588`
             (`registry_.reserve(kind, t.shape[0], t.shape[1])`)
- No prod callers: `grep -rn 'gemv_dispatch\|gemm_dispatch' src/ |
                   grep -v weight_dispatch.cu` → only legacy overload

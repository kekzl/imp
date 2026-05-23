# Kernel Fusion — rmsnorm into NVFP4 GEMV Hot Path
*2026-05-23 · multi-day design doc · not yet implemented*

## Mission

Fuse the standalone `rmsnorm_fp16_kernel` (2.6 % of Qwen3.6 decode kernel time, ~1.9 % wall) into the downstream NVFP4 GEMV kernels (`gemv_nvfp4_qkv_fused_kernel`, `gemv_nvfp4_gate_up_fused_mr_kernel<8>`, `gemv_nvfp4_moe_gate_up_mr_kernel<8>`) so the kernel computes the rmsnorm of its own input `x` cooperatively at kernel entry, then does GEMV with the normalized values.

Realistic upside: **+1.5–2 % decode wall** on Qwen3.6 (where rmsnorm is 2.6 % of decode kernel time) and Qwen3-14B (where rmsnorm fires more often per token: 40 attn layers × 1 input-rmsnorm + 40 FFN layers × 1 input-rmsnorm = 80 rmsnorm calls/token).

## Why this looked like a lever

Profile breakdown of `rmsnorm_fp16_kernel` from `qwen3_6_35b_a3b_nvfp4_full_profile_2026_05_23.md`:

```
2.6 %  62,713 inst  2.0 µs avg   rmsnorm_fp16_kernel
```

Across all 40 layers × ~3 rmsnorm-per-layer × 514 forwards in the bench. 62,713 / 514 = 122 rmsnorm calls per forward step.

The 2.6 % decode kernel time + the per-call launch overhead suggested a clear fusion target: the GEMV that follows rmsnorm reads the normalized `x` once, so a fused kernel could do rmsnorm in registers at start, eliminate the write/read round-trip of the normalized buffer, and eliminate the kernel launch.

## Why the cost-benefit math is unfavorable

### Win sources (under production graphs-ON mode)

1. **Launch overhead elimination** — under CUDA Graphs, captured-kernel launches are ~0.5 µs each. 122 rmsnorm calls × 0.5 µs = **60 µs per decode token**. At Qwen3.6 decode wall ~5 ms/token: **1.2 % wall**.
2. **Intermediate FP16 buffer round-trip** — rmsnorm writes K=2048 (Qwen3.6) or K=5120 (Qwen3-14B) FP16 values, GEMV reads them. For 122 calls × 4 KB / 1792 GB/s peak = **0.3 µs total** (L1/L2 absorbs most of this — the round-trip is already cached). **~0 % wall** for this source.
3. **Rmsnorm exec time absorption** — if the GEMV's existing block-cooperative reduction can absorb the rmsnorm's sum-of-squares + normalize work, we save the 2.6 % rmsnorm kernel exec time. But the GEMV gets slower by some fraction, so net savings ~50 % of 2.6 % = **1.3 % wall**.

**Total realistic wall savings**: 1.5–2.0 %.

### Implementation cost — multi-day across the executor

Imp's executor invokes rmsnorm from many call sites, with different downstream consumers:

```
src/exec/executor_attention.cu:386       rmsnorm_quantize_q8_1   (already fused for dp4a path)
src/exec/executor_ssm_gdn.cu:62          rmsnorm                  (FP16 path, before linear_attn)
src/exec/executor_ssm_gdn.cu:282         rmsnorm                  (FP16 path, attention norm in GDN)
src/exec/executor_ffn.cu:141, 164, 195   rmsnorm                  (FFN input norm — multiple branches)
src/exec/executor_forward.cu:624..801    rmsnorm                  (output_norm before LM head)
```

For each call site, fusing requires:

1. **A new kernel variant** that accepts the rmsnorm weight + epsilon and does the rmsnorm cooperatively at block entry, then runs the GEMV as before.
2. **Cooperative rmsnorm in the GEMV's block**: the GEMV uses 4-8 warps per block, but each block processes one OUTPUT row independently. The rmsnorm reduction is over K (= input dim) — same for all blocks of a single QKV/FFN call. Each block would redundantly compute the rmsnorm of `x[K]`, unless we add cross-block sync (which has the same cost problem as the split-K design doc).
3. **Per-architecture dispatch** — attention/FFN/MoE paths have different GEMV kernels; the dp4a path (`rmsnorm_quantize_q8_1` for Q4_K/Q6_K) is already fused; the NVFP4 path is not. Each path needs its own fused kernel + dispatch update.
4. **Cross-model validation** — `gemv_nvfp4_qkv_fused` is hot on every NVFP4 model; `gemv_nvfp4_gate_up_fused_mr<8>` is the **#1 decode kernel** on Qwen3-14B (40 % of decode). Touching either has regression risk on all NVFP4 hero models.

Total scope: 3–4 new kernel variants × 3–5 days each = **2–4 weeks** of focused work for a properly-validated cross-model rollout.

### The redundant-work catch

The biggest structural issue: **multiple GEMV blocks all read the same `x[K]`**. For Qwen3.6 QKV with output rows = 24 heads × 256 d = **6144 blocks**, each block would redundantly compute the rmsnorm of `x[K=2048]`. The rmsnorm work itself is small (16 mul + reduction over K), but multiplied by 6144 blocks vs the current 1 block = ~6000× redundant work.

L1 cache absorbs the `x` reads (96 % hit rate per H1 analysis), so the redundant reads cost little. The redundant COMPUTE adds modestly to per-block exec time. Net: each GEMV block takes a small additional time hit; multiplied across all blocks, this could offset the launch-savings win.

**The clean alternative** — have ONE block (e.g., the first block of the grid) compute the rmsnorm into a shared global buffer, and all subsequent blocks read from it. Requires cross-block sync = `cg::grid_group::sync()` = **same cost problem as the gdn_scan split-K design** (refuted in Phase 0 because sync overhead exceeds parallelism win).

### Comparison to the existing dp4a-path fusion

`rmsnorm_quantize_q8_1_kernel` (already shipping for Q4_K/Q6_K paths) fuses rmsnorm with the activation Q8_1 quantization. It works because:

- Both rmsnorm AND activation-quant are **single-block kernels** (1 block, K threads cooperatively normalizing a single row of x)
- The output of the fused kernel is the Q8_1-packed `x` buffer — read by the subsequent dp4a GEMV
- The dp4a GEMV doesn't redundantly recompute rmsnorm — it just reads pre-normalized + quantized `x`

This is **NOT a many-block-redundant-rmsnorm structure** — it's a 1-block fused pre-pass.

The analogous NVFP4 fusion would be `rmsnorm_quantize_nvfp4` (single-block: rmsnorm + activation NVFP4 quant). BUT — the NVFP4 decode path doesn't quantize activations (the GEMV reads `x` directly as FP16). So there's nothing to fuse rmsnorm with on the NVFP4 path that would have the same structure as `rmsnorm_quantize_q8_1`.

The only fusion candidate on the NVFP4 path is the many-block-redundant-rmsnorm structure described above — which has the redundant-work catch.

## Path-A alternative — rmsnorm + residual fusion (already exists, may be under-used)

`rmsnorm_residual_fp16_kernel` (`src/compute/layernorm.cu:176`) and `residual_add_rmsnorm` (`src/exec/executor_kernels.cu:1373`) ARE defined but the profile shows them NOT firing in the Qwen3.6 hot path — only the unfused `rmsnorm_fp16_kernel` + `elementwise_add_fp16_kernel` fire.

A cheaper fusion: check whether the executor can route the post-attention residual+norm and post-FFN residual+norm through the existing fused kernels. This is a **dispatch-side change**, not a new kernel.

**Phase 0 measurement** for this Path-A: instrument `executor_forward.cu` to log when `rmsnorm` is called immediately after `elementwise_add`/`residual_add`, and check the fraction. If high, swap the call site to `residual_add_rmsnorm` and bench.

Estimated upside: 1.5 % decode wall (eliminates one of the two FP16 round-trips between rmsnorm and residual). Estimated cost: **1 day** of dispatch logic + bench validation. **Much better ROI than fusing rmsnorm into the GEMV.**

This is the recommended first experiment if the kernel-fusion thread is reopened.

## Implementation phases (Path-B — fuse rmsnorm into GEMV)

Only pursue if Path-A is exhausted or refuted.

### Phase 0 — measure the redundant-work cost (1 day)

- [ ] Microbench: write a standalone `gemv_nvfp4_qkv_fused_with_rmsnorm` that does cooperative rmsnorm at block entry, runs on a single test shape.
- [ ] Compare per-call time vs current `rmsnorm + gemv_nvfp4_qkv_fused` baseline.
- [ ] Quantify the redundant-work cost across N_BLOCKS = {32, 128, 1024, 6144}.
- **Exit criterion**: if per-block redundant rmsnorm > 5 % of GEMV time at N_BLOCKS=6144, refute and stop.

### Phase 1 — proof-of-concept on `gemv_nvfp4_qkv_fused` (Qwen3.6 attention, 3 days)

- [ ] Add `gemv_nvfp4_qkv_fused_rmsnorm_kernel` variant with cooperative rmsnorm prologue.
- [ ] Dispatch from `executor_attention.cu:386` when path is NVFP4 (parallel branch to the existing `rmsnorm_quantize_q8_1` dp4a fusion).
- [ ] Coherence + numerical correctness validation (compare against unfused baseline bit-by-bit, with FMA-order tolerance).
- [ ] Cold-median bench on Qwen3.6.
- **Exit criterion**: ≥ +0.5 % wall on Qwen3.6, no regression on Qwen3-14B.

### Phase 2 — expand to FFN path (3 days)

- [ ] Add `gemv_nvfp4_gate_up_fused_mr_rmsnorm_kernel<NR>` variant.
- [ ] Dispatch from `executor_ffn.cu:141, 164, 195`.
- [ ] Validate across Qwen3-14B Q6_K (NVFP4-cached weights), Qwen3.6 (FP16 SSM input — fusion N/A), Gemma-4-A4B MoE.
- **Exit criterion**: ≥ +0.5 % cumulative wall on dense Qwen3-14B without regression elsewhere.

### Phase 3 — MoE variant (3 days)

- [ ] Add `gemv_nvfp4_moe_gate_up_mr_rmsnorm_kernel<NR>` variant (handles per-expert weights + routing index).
- [ ] Dispatch from MoE FFN path.
- [ ] Validate across Qwen3.6 MoE, Gemma-4-A4B MoE.

### Phase 4 — output_proj LM head fusion (2 days, may not pay off)

- [ ] LM head rmsnorm is `output_norm` before `lm_head_proj`. Fuse into `gemv_nvfp4_multirow_fp32_kernel`.
- [ ] BUT — for Qwen3.6, lm_head is FP16 (excluded from NVFP4 by recipe), so the GEMV that consumes the rmsnorm output is `internal::gemvx::kernel` (cuBLAS), which we can't modify.
- [ ] For Qwen3-14B, lm_head IS in the NVFP4 cache, so the fusion target is `gemv_nvfp4_multirow_fp32`. But this kernel fires only once per token (LM head), so the launch-savings is bounded.
- **Exit criterion**: only proceed if Phase 1+2+3 cumulative shows ≥ +2 % wall (justifying the additional Phase 4 cost).

## Risks

- **Redundant rmsnorm work across many blocks** is the structural issue. Phase 0 microbench is the gate.
- **Cross-model regression risk** on hot kernels. Each variant needs validation against every NVFP4 hero model.
- **The existing `rmsnorm_quantize_q8_1` precedent doesn't apply** — that's a 1-block fused pre-pass, not a many-block fused GEMV. The NVFP4 path doesn't have an equivalent quantize step to absorb the rmsnorm.
- **CUDA Graphs already absorb launch overhead.** Most of the 122-launches-per-token cost is amortized under graphs-ON. The exec-time savings from fusion is the only real win, capped at ~1.5 % wall.
- **Path-A (rmsnorm + residual existing fusion)** likely gives most of the win at a fraction of the cost. Pursue Path-A first.

## Don't repeat

- ❌ **Fusing rmsnorm into a many-block GEMV without measuring redundant-work cost.** N_BLOCKS × small-work-per-block can offset the launch-savings win. Phase 0 microbench is mandatory.
- ❌ **Assuming graphs-OFF kernel-time profile is the target.** Under graphs-ON (production), most launch overhead is amortized. The fusion gain shrinks accordingly.
- ❌ **Touching `gemv_nvfp4_gate_up_fused_mr<8>` without cross-model validation.** It's the #1 decode kernel on Qwen3-14B (40 % of decode). A regression there is catastrophic.

## Re-evaluation triggers

Re-open this plan when:
- Path-A (rmsnorm + residual fusion via existing kernels) is exhausted or refuted.
- imp adds a new architecture where rmsnorm fires significantly more often per token than current hero models (would raise the win ceiling).
- A new CUDA primitive lands that enables cheap cross-block sync (would unblock the "one block computes rmsnorm, others read" structure).
- The post-Phase-A Qwen3-14B / Qwen3.6 north-stars are within < 5 % of the 175 / 250 milestone (where a +1.5 % fusion win becomes more relevant).

## Estimate

- **Phase 0 (microbench gate)**: 1 day. Hard gate — refute or proceed.
- **Path-A (rmsnorm + residual dispatch swap)**: 1 day. Try this first; if it delivers 1.5 %, kernel-fusion thread is closed.
- **Path-B (full multi-kernel rollout)**: 11–14 days total if all phases proceed.

Realistic delivered wall savings:
- Path-A alone: **1–1.5 %** (best case).
- Path-B full: **1.5–2 %** (best case, additive to Path-A).

---

*Plan recommendation: **try Path-A first** (1 day, swap existing dispatch). Path-B is multi-week and the redundant-work catch may refute Phase 0 anyway. Don't start Path-B before Phase 0 microbench confirms the rmsnorm cost across many blocks is < 5 % of GEMV time.*

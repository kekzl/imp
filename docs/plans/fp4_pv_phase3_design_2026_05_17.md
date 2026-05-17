# FP4 PV Phase 3 — Attention design memo (2026-05-17)

Scoping memo for **FP4 P×V (Phase 3)** of the MXFP4 FMHA kernel on
RTX 5090 / sm_120a. **No source changes** — this is a decision-support
document.

Author: Claude (Opus 4.7), prompted by the open-lever paragraph in
`docs/roadmap.md` (PTX ISA 9.2 section).

---

## Table of contents

1. [Status check](#1-status-check)
2. [SageAttention3 two-level accumulator](#2-sageattention3-two-level-accumulator)
3. [PV-only A/B test harness (prereq)](#3-pv-only-ab-test-harness-prereq)
4. [Implementation phases](#4-implementation-phases)
5. [Quality risks](#5-quality-risks)
6. [Models in scope](#6-models-in-scope)
7. [Decision recommendation](#7-decision-recommendation)
8. [Cross-references](#8-cross-references)

---

## 1. Status check

### Phase 1 (QKT FP4) — SHIPPED

`src/compute/attention_fmha_mxfp4_sm120.cu` carries the block-scale MMA
path on the QKT GEMM:

- Kernel template flag `UseBlockScaleMma` (`attention_fmha_mxfp4_sm120.cu:107`)
- Dispatch comment (`:102-103`):
  > UseBlockScaleMma=true: `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
  > with per-16-element UE4M3 scales applied in the MMA instruction.
- Inline PTX issue site (`:589-599`):
  ```
  mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
  .f32.e2m1.e2m1.f32.ue4m3 …
  ```
- Runtime knob: `RuntimeConfig::attention.fmha_blockscale = "auto"`
  (`src/runtime/config.h:85`).
- Roadmap pointer: PR #56 (squash-merge `208f25b`, "compute: SM120 FMHA
  optimization pass — +12.6% Q8_0 prefill, Project B complete"). Note:
  `docs/roadmap.md` cites `b51788e` which does not resolve in the
  current main tree — the actual landed SHA is `208f25b`. Worth a one-line
  roadmap fix in a separate docs commit.
- Measured impact: **+1.8% Qwen3-4B MXFP4 at HD=128** (Phase-1 MMA is
  only ~15% of FMHA wall time, so the 2.5× raw MMA gain only moves the
  needle a little).

### Phase 3 (P×V FP4) — NOT shipped

`grep -rn "mxf4nvf4.*pv\|FP4 PV\|fmha_pv_fp4\|pv_fp4"` across `src/`
returns **zero hits**. The PV path is FP16 WMMA:

- Header comment (`attention_fmha_mxfp4_sm120.cu:5`):
  > Tiled flash attention with FP4 E2M1 Q·K^T score compute and **FP16 P·V**.
- Pipeline comment (`:18`):
  > 5. FP16 WMMA: O += P · V
- Section banner (`:52`):
  > FP16 WMMA for P·V (unchanged from FP16/FP8 FMHA)
- Actual issue (`:847-867`):
  ```
  Phase 3: O_acc += P · V using FP16 WMMA (m16n16k16)
  …
  wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
  ```

So the current shape of the FMHA inner loop is:

| GEMM    | A (in regs)         | B (in regs)         | MMA op                                                 | Acc  |
|---------|---------------------|---------------------|--------------------------------------------------------|------|
| QK^T    | Q FP4 (per-16 UE4M3)| K FP4 (per-16 UE4M3)| `mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`          | FP32 |
| Softmax | row-online (FP32)   | —                   | —                                                      | —    |
| P × V   | P FP16              | V FP16              | `wmma::mma_sync` ≡ `mma.sync …m16n16k16` (HMMA pipe)   | FP32 |

The "open lever" Phase 3 wants to convert the second row to the same
`mxf4nvf4` op with P quantised to FP4 + UE4M3 scales and V transposed +
quantised to FP4 ahead of the loop.

---

## 2. SageAttention3 two-level accumulator

The mathematical wall here is independent of hardware: **post-softmax
probabilities span 6+ orders of magnitude per row**. After
`P = softmax(QK^T / sqrt(d))` a typical row holds 1-3 values near
`{0.1 … 0.9}` and the remaining `Bkv - {1..3}` values in
`{1e-3 … 1e-9}`. FP4 E2M1 has 7 representable positive magnitudes
(`{0.5, 1, 1.5, 2, 3, 4, 6}` after UE4M3 scaling), so a per-16-element
absmax-scaled quant maps the tail to `0` — i.e. attention loses the
long-tail mass entirely, which is exactly where coherence lives at long
context.

A naïve P→FP4 cast therefore degenerates (memo
`fp4_pv_potential_2026_04_25.md` calls this out;
`sageattention3_study_2026_04_24.md` analyses the mitigations).

The SageAttention3-style **two-level accumulator** is the mitigation
referenced in the roadmap:

1. Split each row of `P` into two components:
   - **Coarse**: `P_c = quantise_fp4(P, scale=per_16_absmax)`
   - **Residual**: `P_r = P - dequantise(P_c)` (kept in FP16/BF16)
2. Run **two MMAs** per K-stripe:
   - `O += mxf4nvf4(P_c, V_fp4)` (block-scaled FP4)
   - `O += hmma     (P_r, V_fp16)` (or BF16 PV pass over the residual)
3. Both contributions land in the same FP32 `o_frag`.

The +13% perf claim **only holds if the coarse path dominates wall
time** — i.e. the residual pass uses a sparser/cheaper code path
(SageAttention3 does this by exploiting the structural sparsity of the
residual: after FP4 quant the residual is mostly near-zero except for
2-3 entries per row, allowing a packed-sparse HMMA or a small
side-vector dot product instead of a full m16n16k16 MMA).

If we naively run the residual as a full HMMA m16n16k16 (same as
today), we *add* the FP4 work on top of the existing HMMA and net
negative perf despite higher peak TOPS.

**Open design choice:**

- **2L-A (faithful SageAttention3)**: implement structural-sparse
  residual path. Big complexity bump but matches the paper's gain.
- **2L-B (full residual)**: keep residual as plain HMMA m16n16k16. Net
  perf is roughly neutral (FP4 path is gravy); the win has to come from
  HD=256 / large-Bkv regimes where FP4's 4× peak compounds harder.

For an MMA-isolated microbench (Phase 3a below), 2L-A and 2L-B differ
only in residual cost — both can be measured.

---

## 3. PV-only A/B test harness (prereq)

The QKT change in PR #56 was A/B-able because `UseBlockScaleMma` is a
template flag and the legacy `f8f6f4.m16n8k32` path is still compiled.
The PV path is **monolithic** today: one HMMA loop, no template
parameter. To run a PV-isolated A/B we need infrastructure.

Three options ranked by cost:

### Option A — kernel template variant
`attention_fmha_mxfp4_sm120<PV_MODE = FP4 | FP16>` plus parallel inline
PTX issue blocks. Pros: end-to-end measurement, drops cleanly into the
existing dispatch. Cons: large diff (~200-300 LoC plus SMEM bookkeeping
for V_T_fp4 + P_fp4 + scale tiles), and any quality bug requires a
full rebuild + e2e bench loop to find. Estimated 3-5 days incl.
debugging.

### Option B — `#if FP4_PV` preprocessor flag
Mark each MMA-call site with `#if FP4_PV` / `#else` blocks, gate
compile-time. Pros: smallest diff. Cons: doubles maintenance cost of
the file; can't A/B at runtime; doesn't address SMEM layout difference
cleanly. **Not recommended** — same effort as Option A without the
runtime A/B win.

### Option C — standalone microbench
`tools/imp-bench/bench_fp4_pv.cu` (~100 LoC) operating on synthetic
post-softmax distributions (e.g. row drawn from `softmax(N(0, 1) +
spike)` then optionally quantised). Issues the bare `mxf4nvf4
m16n8k64` PV MMA in a loop vs. the HMMA `m16n16k16` PV reference,
reports raw throughput and a numerical-error histogram against an
FP32 reference. Pros:
- Definitive answer on whether the MMA-level speedup is real **before**
  any kernel surgery.
- Lets us prototype the two-level accumulator on synthetic data.
- Cheap to throw away if Phase 3a says no.

Cons: doesn't measure end-to-end FMHA wins; doesn't prove SMEM/quant
overhead. Those come in 3b/3c.

**Recommendation: Option C as Phase 3a** (the meta-irony of starting
Phase 3 with a Phase 3 phase noted). Memo `fp4_pv_potential_2026_04_25.md`
proposed the same staged path 22 days ago — the prerequisite then was
"PV-only A/B test infrastructure" and it still is.

---

## 4. Implementation phases

| Phase | Scope | Effort | Gate to next |
|-------|-------|--------|--------------|
| **3a** | Standalone `bench_fp4_pv.cu` microbench on synthetic post-softmax data. Measure raw `mxf4nvf4 m16n8k64` PV throughput vs HMMA `m16n16k16` PV. No two-level accumulator yet — single-level FP4 baseline only. | ~100 LoC, 1-2 days | **≥2.5× raw MMA speedup** in microbench |
| **3b** | Add two-level accumulator (2L-A and 2L-B variants) to the microbench. Measure MMA + accumulator + quant overhead on synthetic data. | ~150 LoC additional, 2-3 days | **≥1.5× combined speedup** including overhead |
| **3c** | Integrate winning variant into `attention_fmha_mxfp4_sm120.cu`. Opt-in via new field `RuntimeConfig::attention.fmha_pv_fp4 = false` (default off). Run on Qwen3-4B MXFP4. Add `Pv_Fp4_MatchesLegacy` GTest (looser tolerance than current 0.25 max_err — likely 0.5-1.0 acceptable per memo `fp4_pv_potential_2026_04_25.md` §4). | ~300 LoC + tests, 3-5 days | **≥+5% e2e tok/s** on Qwen3-4B AND quality smoke test (§5) passes |
| **3d** | HD=256 evaluation. Requires an HD=256 MXFP4 model — none in current test set (see §6). Either quantise an existing FP16 HD=256 model in-house or wait for upstream. | Variable: 1-2 days if model exists; 1+ week to build one | **Consistent gain on ≥2 HD=256 models** |
| **3e** | Default flip if 3c+3d show consistent gains across HD=128 and HD=256. Otherwise stay opt-in. | ~10 LoC config change + verify-fast | — |

**Hard rule**: each phase fails-closed. If 3a shows <2× MMA speedup, do
not proceed to 3b — the 268 TOPS figure in
`sm120_mma_variants_2026_04_25.md` was measured in isolation, not with
realistic data distributions.

---

## 5. Quality risks

- **Central risk**: FP4-quant of post-softmax probabilities. The
  two-level accumulator IS the mitigation; **without it, expect
  degeneration** (greedy-decode generates `aaaaa` after ~30 tokens, the
  classic FP4-attention failure mode). The `check-degeneration` skill is
  the right harness for this — run after any 3c integration.
- **Tolerance widening**: existing `Blockscale_MatchesLegacy` GTest
  uses `max_err = 0.25`. FP4 PV adds a second quant step. Expect
  `Pv_Fp4_MatchesLegacy` to need 0.5-1.0. If it needs >2.0 the
  two-level accumulator isn't doing its job.
- **Long-context behaviour**: FP4 PV degenerates fastest at long
  context where attention is most distributed. Quality smoke test must
  include ≥4k-token prompts with coherent expected continuations.
- **HD=128 vs HD=256 quality profile**: different head-dim changes the
  per-row probability distribution (HD=256 typically sharper softmax →
  more tail-truncation tolerance). Test runs must be **independent
  per head-dim**; HD=128 OK does not imply HD=256 OK.
- **Two-level accumulator residual sparsity**: if the residual carries
  >3 non-zero entries per row, the 2L-A "structural sparse residual"
  optimisation breaks and we fall back to 2L-B (full HMMA residual)
  which is unlikely to net positive.

---

## 6. Models in scope

### HD=128 MXFP4 (the existing regime)
- `qwen3-4b-instruct-2507-mxfp4.gguf` (in test set, tg≈124 tok/s)
- Phase-3 ceiling: **~+13% on top of current +1.8%** = ~+15% cumulative
  vs pre-Phase-1, per the roadmap quote and `fp4_pv_potential_2026_04_25.md`
  measurement (PV is ~15% of FMHA wall time at HD=128).

### HD=256 MXFP4 (the bigger payoff regime)
- **Not in the test set.** The natural candidates are:
  - Qwen3.5-GDN (HD=256) — currently exists as Q8_0 only in test set
  - Gemma-4 globals (HD=256) — currently exists as Q4_K_M / Q8_0
- Sourcing options:
  - (a) Quantise an existing FP16 HD=256 model to MXFP4 ourselves —
    requires a working MXFP4 GGUF quantiser path; loader was updated in
    PR #185 (`cb83c3b`, "support modern llama.cpp MXFP4 GGUF type 39")
    so the load side is fine; production side is open.
  - (b) Wait for upstream HuggingFace / llama.cpp MXFP4 publications of
    an HD=256 model.
- HD=256 means Phase-1 MMA is a larger fraction of FMHA wall time
  (longer per-row dot product) and Phase-3 PV correspondingly larger →
  visible speedup should compound. The roadmap quote calls this out
  explicitly.

### NVFP4 — out of scope
NVFP4 uses the same PTX opcode family but **different scale layout**
(per-16 E4M3 in MXFP4 vs per-16 E4M3 with different group origin in
NVFP4) and a different end-to-end pipeline. The infrastructure here is
MXFP4-specific (`mxf4nvf4` is one PTX kind that supports both, but
imp's NVFP4 attention path is separate and decode-focused). Treat NVFP4
PV as a follow-on memo if Phase 3 MXFP4 ships and proves out.

---

## 7. Decision recommendation

Three viable paths:

### Option I — Ship Phase 3a microbench
~100 LoC, 1-2 days. Definitively answers whether the MMA-level gain
exists on realistic post-softmax distributions. If it doesn't (i.e. the
268 TOPS figure was a synthetic-data peak that real data doesn't hit),
we save the 3b/3c work. If it does, we proceed to 3b with confidence.
**Lowest-risk, highest-information-density next step.**

### Option II — Defer entirely
Phase 1 shipped at +1.8% e2e on Qwen3-4B. If HD=128 is the dominant
model class for the foreseeable future, Phase 3 ceiling is ~+13% on the
15% of wall time that is PV — i.e. realistic e2e ≤ +5% even with
perfect implementation. Phase 1 already paid back the FP4 MMA
infrastructure investment. Wait for HD=256 MXFP4 models to appear in
production before investing further.

### Option III — Build the model first
If HD=256 MXFP4 is the real payoff regime, the bottleneck isn't the
kernel work but the model availability. Spending a week to quantise
and validate an HD=256 MXFP4 model (e.g. Qwen3.5-GDN HD=256) may be
the highest-leverage step, since it (a) unlocks Phase 3 e2e
measurement, (b) is useful independently for current MXFP4 coverage,
(c) the kernel work is well-understood once we have a target.

### Recommendation: **Option I (ship Phase 3a microbench)**.

**One-sentence justification:** A 100-LoC microbench on synthetic
post-softmax data is the cheapest way to discriminate between "the
gain is real, build the rest" and "the gain evaporates on realistic
distributions, defer" — and the answer informs Options II and III, so
running Phase 3a is strictly dominant over deciding between them now.

Phase 3a is also independent of model availability (synthetic data) so
it can run in parallel with Option III's HD=256 model-building work if
the team wants to hedge.

---

## 8. Cross-references

### Memos (`~/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`)
- `fp4_pv_potential_2026_04_25.md` — the original measurement (PV is
  ~15% of FMHA wall time at HD=128, +13% theoretical upside, V
  transpose + P quant + two-level accumulator costed at 200-300 LoC).
- `sageattention3_study_2026_04_24.md` — two-level accumulator origin;
  `mxf4nvf4.block_scale` upgrade path (5× FA potential).
- `mxfp4_fmha_optimization.md` — bank conflict fix, cp.async V overlap
  (relevant for SMEM budget when adding V_T_fp4 + P_fp4 + scale tiles).
- `fmha_vectorization_2026_04_25.md` — PR #56 +11% via float4 + HW FP4
  conv; precedes Phase 1 block-scale MMA shipping in same PR family.
- `stage4_qkt_layout_verified_2026_04_24.md` — CuTe operand layouts
  for the QKT path; PV path will need an analogous derivation.
- `sm120_mma_variants_2026_04_25.md` — variant bench: `mxf4nvf4 vec::4X
  k64 ue4m3` at 268 TOPS (the peak the +13% projection assumes).

### Code
- `src/compute/attention_fmha_mxfp4_sm120.cu` — sole FMHA MXFP4 kernel;
  Phase 1 lives in `:107-…` (template), `:571-…` (block-scale issue
  branch), `:589-599` (PTX). Phase 3 PV lives in `:844-873` (FP16 WMMA
  loop) — this is the section that would be templated/duplicated.
- `src/runtime/config.h:85` — `fmha_blockscale = "auto"`; Phase 3
  would add `fmha_pv_fp4 = false` (default off) here.

### PRs / commits
- PR #56 (squash `208f25b`) — "compute: SM120 FMHA optimization pass —
  +12.6% Q8_0 prefill, Project B complete" — Phase 1 reference.
  Roadmap currently cites `b51788e` which doesn't resolve in main;
  worth a one-line roadmap fix.
- PR #185 (`cb83c3b`) — modern llama.cpp MXFP4 GGUF (type 39) loader,
  unblocks any MXFP4 model sourced upstream.

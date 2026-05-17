# NVFP4 SmoothQuant `input_scale` design memo — 2026-05-17

**TL;DR — defer.** The only model in our corpus that ships SmoothQuant-calibrated
NVFP4 weights is `Mistral-Small-3.2-24B-Instruct-2506-NVFP4`, and that
model is **not present locally** (see `ls /home/kekz/models | grep -i mistral`
returns no NVFP4 Mistral). PR #78
(`use_default_system_prompt=false`) already keeps the long-context drift
out of the typical chat flow. The remaining engineering — a per-channel
SmoothQuant scaling vector applied during activation quantization —
costs ~1-2 weeks once a test model is available, and only pays back on
models actually quantized with SmoothQuant. Today imp has zero such
models in regular use. Recommendation: **Phase 1 only**
(model availability) when scheduling allows; defer Phases 2-4 until a
real SmoothQuant-calibrated workload appears.

## Table of contents

1. [SmoothQuant primer](#1-smoothquant-primer)
2. [What imp currently does](#2-what-imp-currently-does)
3. [The 2026-05-07 refutation](#3-the-2026-05-07-refutation)
4. [The real fix — per-channel scaling at activation quantization](#4-the-real-fix--per-channel-scaling-at-activation-quantization)
5. [Implementation phases](#5-implementation-phases)
6. [Risks](#6-risks)
7. [Decision recommendation](#7-decision-recommendation)

---

## 1. SmoothQuant primer

SmoothQuant (Xiao et al., ICML 2023) is a pre-quantization smoothing
transform applied at calibration time. It migrates activation outliers
into the weights so the resulting activation distribution becomes
quantization-friendly. The mechanism:

For a Linear layer `Y = X @ W^T + b` with input `X ∈ R^{T,K}` and
weight `W ∈ R^{N,K}`, define a per-input-channel scale vector
`s ∈ R^{K}` derived from activation statistics over a calibration set
(typically `s_k = max_t(|X[t,k]|)^α / max_n(|W[n,k]|)^(1-α)` where
`α` is the smoothing strength, 0.9 in Mistral-Small-3.2-NVFP4's
recipe — see `nvfp4_long_context_regression_2026_04_28.md`).

The transform replaces the layer with:

```
Y = (X · diag(1/s)) @ (diag(s) · W^T) + b
  = X' @ W'^T + b
```

Mathematically `(1/s)` and `(s)` cancel — in IEEE FP this is exact up
to round-off. The point is that the **quantized** versions are
different:

- `X' = X · diag(1/s)` has smaller per-channel max → activation
  quantization scales are tighter → less FP4 quant noise on the
  hot path.
- `W' = diag(s) · W^T` has wider per-output-channel max — but
  weights are quantized offline once, and per-row absmax for FP4
  weight quant tolerates the increase much better than activation
  quant tolerates outlier-driven scales.

The SafeTensors file ships `W'` (already pre-multiplied by `s`) plus
a sidecar tensor describing `1/s` so the inference engine can
reproduce `X' = X · diag(1/s)` at runtime before activation
quantization. In llm-compressor's NVFP4 export this sidecar is
named `input_global_scale` per Linear (renamed to `input_scale` by
imp's loader at `src/model/llm_compressor_loader.cpp:149`).

**Important wrinkle for this design memo.** The llm-compressor
`input_global_scale` tensor as currently exported is a **per-tensor
FP32 scalar**, not a per-channel `[K]` vector — verified by
`weight_upload.cu:2002-2020` reading `sc.input_scale.numel()` as a
flat scalar count. This is one piece of `s` rolled up into a single
calibration scalar; whether the true per-channel `s_k` is recoverable
from the SafeTensors checkpoint depends on the format generation and
is open work (see §4 and §6).

## 2. What imp currently does

### Load path

`src/model/safetensors_loader.cpp` discovers `*.input_scale`
tensors and routes them through `src/model/weight_map.cpp` (`kind ==
"input_scale"`, lines 59-70, 591, 610, 700, 738, 784, 1049) into
`Model::nvfp4_scratch_[key].input_scale` — declared in
`src/model/model_config.h:138-143`:

```c++
struct NvFP4PreQuantWeight {
    Tensor weight_scale;
    Tensor weight_scale_2;
    Tensor input_scale;   // FP32 scalar per Linear (optional)
    bool valid() const { return weight_scale.data != nullptr; }
};
```

### Upload path

`src/model/weight_upload.cu:1950-2057` walks `nvfp4_scratch_` and
uploads `weight_scale` + `weight_scale_2` to GPU
unconditionally. `input_scale` upload is **gated to
diagnostics.audit_nvfp4_scales** (`upload_scale(sc.input_scale)` at
line 2028) — under default config the tensor stays host-side, then
the entry is dropped at `executor_pre_dequant.cu:408`
(`mut_model->nvfp4_scratch_.clear()`). No VRAM cost for production
runs.

When `diagnostics.audit_nvfp4_scales=true` we emit per-Linear
min/max/mean stats (`weight_upload.cu:2040-2053`) — used in the
2026-04-28 bisection to surface Mistral-3.2-NVFP4's 537× weight_scale_2
dynamic range.

### Inference path

`executor_pre_dequant.cu:316-318` counts
`n_with_input_scale++` per Linear during Phase 0 promotion but does
**not** read the value:

```c++
if (sc.input_scale.data) {
    n_with_input_scale++;
}
w.qtype = QType::NVFP4;
w.scales = sc.weight_scale.data;
w.tensor_scale = promoted_scale;
```

A final log line (`executor_pre_dequant.cu:422-435`) records the
count and explicitly states it is intentionally not applied:

> "NVFP4 prequant: %d Linears carry input_scale (intentionally NOT
> applied; set IMP_AUDIT_NVFP4_SCALES=1 for stats)."

### Activation quantization today

The current dynamic-input activation quantization is **per-tensor
symmetric scale + per-micro-block FP8 scale**, fully driven by the
activation's own absmax — no per-channel awareness:

- `src/quant/nvfp4_quant.cu:421-462` `quantize_fp16_to_nvfp4(...)`:
  takes 2D FP16 input `[N, K]`, kernel `quantize_nvfp4_kernel`
  computes `tensor_scale = max(|X|) / kAbsmaxScale` once per tensor,
  then per-16-element micro-block computes FP8 E4M3 micro-scales.
- `src/quant/nvfp4_quant.cu:465-510` `quantize_fp16_to_nvfp4_async`:
  decode hot path; absmax computed in a separate kernel feeding into
  `quantize_nvfp4_from_absmax_kernel`.

Neither path reads any per-channel scaling vector — that's where the
SmoothQuant `1/s` factor would have to live.

## 3. The 2026-05-07 refutation

Two patches to `executor_pre_dequant.cu` were tested on
Gemma-4-26B-A4B-it-NVFP4 (see
`memory/llm_compressor_input_scale_dead_end_2026_05_07.md`):

1. **Divide**: `promoted_scale = promoted_scale / h_input_scale`
2. **Multiply**: `promoted_scale = promoted_scale * h_input_scale`

Both gated to `cfg.is_llm_compressor_nvfp4 && sc.input_scale.data
&& numel==1`.

| Build | phase4 | typical failure |
|---|---|---|
| baseline (skip-guard ON) | **18/20** | (control) |
| input_scale DIVIDE | 4/20 | `own own own own…` (stuck token) |
| input_scale MULTIPLY | 4/20 | `own- own own own…` (stuck token) |

Gemma-4's typical `input_scale ≈ 268` — either direction shifts the
GEMM alpha by ~268× per layer. Divide → outputs ~0 → softmax loses
signal. Multiply → FP16 saturation → Inf propagation. Both routes
collapse the model.

The math (re-derived in the memo): imp's dynamic per-block input
quantization already produces the correct GEMM output because the
input quantization recomputes `micro_scale_x = block_amax_x / 6`
dynamically. A scalar alpha rescale **double-applies** correction
that the dynamic quantizer is already supplying — there is nothing
useful for a scalar `input_scale` to do here.

**Why a scalar can't carry SmoothQuant.** SmoothQuant's `1/s` is a
per-input-channel vector; the whole point of the transform is to
flatten the activation's channel-wise outlier profile. Averaging it
to a scalar destroys exactly the per-channel correction the
transform encodes. The 2026-05-07 result is the empirical
demonstration of that.

## 4. The real fix — per-channel scaling at activation quantization

The correct SmoothQuant absorption point sits **before**
activation quantization, not in the GEMM alpha:

1. **Pre-quantization smoothing**: compute
   `X'[t,k] = X[t,k] / s[k]` element-wise. This shrinks per-channel
   outliers along the K axis.
2. **NVFP4 activation quantization on `X'`**: standard
   `quantize_fp16_to_nvfp4` — micro-scale computed on the smoothed
   distribution, narrower per-block dynamic range, less FP4 noise.
3. **NVFP4 GEMM**: standard kernel, **no alpha modifier**. Weights
   were pre-multiplied by `s` at SafeTensors-build time — the
   product `X' @ W'^T = X @ W^T` mathematically.
4. **Post-GEMM**: standard FP16 output; nothing extra to do.

Kernel-surface change, two options:

- **Option A — extend the quantizer.** Add an optional
  `const float* per_channel_scale` argument to
  `quantize_fp16_to_nvfp4` (and the `_async` variant). When non-null,
  divide each element by `s[col]` before the micro-block absmax
  step. Costs one extra global-memory load + multiply per element.
  Touches `src/quant/nvfp4_quant.h`, `src/quant/nvfp4_quant.cu`, and
  every call site that builds the act-quant input (`executor_*.cu`
  GEMV/GEMM dispatch, `gemm_dispatch_impl` in `executor_kernels.cu`).
- **Option B — separate pre-pass.** Add a small CUDA kernel
  `smooth_activations(act_in, per_channel_scale, act_out)` and call
  it before the existing quantizer. Simpler API surface (no change
  to the quantizer signature), but adds a kernel launch + writes a
  scratch tensor (~bandwidth overhead, similar to the current
  PreAttnNorm scratch). For decode (M=1) this is a 1×K kernel which
  is cheap; for prefill it's a single fused-elementwise pass over
  `[T, K]`.

Option B is more conservative for shipping incrementally; Option A
is more performant once stable.

**Where does `s[k]` come from in the SafeTensors file?** This is the
open question. Three sub-cases to verify against an actual Mistral-3.2-
NVFP4 checkpoint:

(a) **`input_scale` is already per-channel `[K]`** — `weight_map.cpp`
    accepts any shape; `weight_upload.cu` would just see a larger
    `numel()`. Today imp treats it as scalar by reading
    `numel()` blindly. If the SafeTensors actually ships
    `[K]` we already have it on disk and just need to plumb it
    through.

(b) **`input_scale` is scalar + a separate `s_channel` tensor**
    elsewhere in the checkpoint (e.g. fused into the upstream
    `input_layernorm.weight`). If SmoothQuant was applied at the
    layernorm boundary — which Mistral's recipe.yaml specifies
    (`mappings: [['q_proj','k_proj','v_proj'], 'input_layernorm']`)
    — then `1/s` is already baked into the layernorm weight (the
    `diag(1/s)` premultiplication fuses cleanly into a preceding
    RMSNorm gain), and the per-Linear `input_scale` scalar is just
    the resulting global activation scale for NVFP4 calibration.
    **This is the most likely case** given how llm-compressor exports
    SmoothQuant-fused checkpoints. In that case there is nothing
    extra for the engine to do — the layernorm weight already does
    the smoothing — and the long-context drift must have a different
    root cause (most plausibly the FP16-act-vs-NVFP4-act mismatch
    discussed in `nvfp4_long_context_regression_2026_04_28.md` §
    "Suggestive evidence").

(c) **`input_scale` is per-channel but stored at a non-obvious key**
    (e.g. `*.input_scale` for a `[K]` rank-1 tensor that imp's weight
    mapper drops because it expects scalar). This is testable by
    just running the loader against the model with `numel()` logging.

Phase 1 below resolves which of (a)/(b)/(c) is actually true.

## 5. Implementation phases

### Phase 1 — Model availability (BLOCKING, 1-2 days)

- Pull `Mistral-Small-3.2-24B-Instruct-2506-NVFP4` from HF
  (RedHatAI mirror; ~16 GB). Place under `/home/kekz/models/`.
- Run with `--set diagnostics.audit_nvfp4_scales=true` and the
  `LONG_SYS` Lorem-ipsum repro from
  `nvfp4_long_context_regression_2026_04_28.md` to confirm the
  drift still reproduces on current main.
- **Diagnostic add** (one-line, no semantic change): in
  `weight_upload.cu:2002-2020` log the per-Linear `input_scale.numel()`
  + ndim + shape, not just `numel()` flat-summed. Resolves
  sub-case (a) vs (b)/(c) above.
- Identify the actual shape of `input_scale` (scalar vs `[K]`) and
  whether the upstream `input_layernorm.weight` already shows
  per-channel SmoothQuant fingerprints (compare its variance to a
  non-SmoothQuant model like Mistral-3.1-Q6_K).

If Phase 1 result is sub-case (b) — layernorm-fused — then **stop
here**. The "real fix" is not an engine change; the long-context
drift is the NVFP4-noise-grows-with-`||X||` problem, and the
correct lever is dynamic NVFP4 activation quantization end-to-end
(see `nvfp4_long_context_regression_2026_04_28.md` § "Next-step
methodology" item 2) — a separate, much larger work item.

If Phase 1 result is sub-case (a) or (c), continue to Phase 2.

### Phase 2 — Per-channel pre-quant smoothing (3-5 days, gated)

- Implement Option B (`smooth_activations` pre-pass kernel) in
  `src/quant/nvfp4_quant.cu`. Signature:
  `void smooth_activations(const Tensor& in, const Tensor& s_inv,
  Tensor& out, cudaStream_t stream)`. Skip kernel launch if `s_inv`
  is null (no-op for non-SmoothQuant models).
- Plumb a pointer to `s_inv` onto the main weight tensor sidecar
  (extend `Tensor` or add a parallel field on `Model::nvfp4_scratch_`
  → promoted to a stable per-Linear lookup, analogous to how
  `weight_scale` already promotes to `w.scales`).
- Insert the pre-pass call ahead of `quantize_fp16_to_nvfp4` in:
  - `src/graph/executor_kernels.cu` `gemm_dispatch_impl` NVFP4 path
    (dense prefill + decode; both call act-quant).
  - Any NVFP4 MoE prefill / decode hot path that builds an
    `NvFP4QuantResult` on the fly. Verify call sites via
    `grep -rn quantize_fp16_to_nvfp4 src/graph src/compute`.
- Gate at the call site on `s_inv != nullptr` so non-SmoothQuant
  models (Gemma-4-NVFP4, Qwen3.6-NVFP4, Qwen3-30B-Modelopt) bypass
  the kernel entirely — preserves baseline perf and bit-exactness
  for those models.

### Phase 3 — Mistral-3.2-NVFP4 long-context verification (1-2 days)

- Run the `LONG_SYS` Lorem-ipsum prompt repro that historically
  triggered `"elit dolor elit dolor…"` at N≥11 (~95 toks). With
  Phase 2 applied, verify coherent "The capital of France is
  **Paris**…" output.
- Run `scripts/validate_safetensors.py --model
  Mistral-Small-3.2-24B-Instruct-2506-NVFP4` (full 20-prompt
  battery). Expected: phase4 ≥ 18/20 (matching Gemma-4-NVFP4
  parity). Anything below 15/20 means we have a different bug
  surfacing (e.g. the FP16-act-vs-NVFP4-act mismatch is dominant
  even after SmoothQuant absorption).
- Re-enable the 600-token default system prompt under a flag and
  verify it no longer triggers drift — that retires PR #78's
  workaround for this model.

### Phase 4 — Regression test on non-SmoothQuant NVFP4 models (1 day)

- Run `validate_safetensors.py` battery on:
  - `Gemma-4-26B-A4B-it-NVFP4` (no SmoothQuant; should be bit-exact
    via the null-`s_inv` bypass)
  - `Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4`
  - `Qwen3.6-35B-A3B-NVFP4`
  - `Qwen3-30B-A3B-NVFP4-Modelopt`
  - `Nemotron-3-Nano-30B-A3B-NVFP4`
- Expectation: bit-identical or within decode 3% / prefill 5% of
  current main (i.e. `tests/perf_baseline.json` thresholds hold).
  Any decode regression on these models means the `s_inv != nullptr`
  bypass leaked.

## 6. Risks

- **Phase 1 blocker — model availability**: the design assumes
  Mistral-Small-3.2-24B-Instruct-2506-NVFP4 (or another
  SmoothQuant-calibrated NVFP4 checkpoint) can be downloaded. RedHatAI
  is the most likely mirror but is not in our local model set today;
  if upstream removes it we would have to re-quantize from FP16
  Mistral-3.2 + a SmoothQuant calibration step using llm-compressor
  (multi-day, requires a calibration dataset and a tuned `α`).
- **Sub-case (b) outcome**: ~50% probability Phase 1 reveals that
  SmoothQuant is already fused into `input_layernorm.weight`, in
  which case Phases 2-4 don't apply and the long-context drift root
  cause is the NVFP4-noise-vs-FP16-act issue from
  `nvfp4_long_context_regression_2026_04_28.md`. That fix is a
  much larger workitem (dynamic NVFP4 activation quantization
  end-to-end + a true NVFP4×NVFP4→FP32 GEMM in the dense path).
- **Performance**: per-channel scale read adds one global memory
  load + multiply per element of `X`. Bandwidth-bound — small but
  measurable: ~+2-5% on activation quant alone (one extra
  `[K]`-byte read per `[T,K]` activation), tiny on overall e2e
  (act-quant is single-digit % of decode time).
- **Generalization**: if Phase 1 shows that other NVFP4 models also
  have a non-trivial `input_scale` tensor we hadn't noticed, the
  bypass-on-null logic must be correct. Audit log at upload time
  already counts per-Linear input_scale presence — add a single
  line that also asserts `numel() == K` (per-channel) vs `numel() ==
  1` (scalar legacy) so we don't silently mis-route a scalar through
  a per-channel kernel.
- **Tensor shape ambiguity**: if `input_scale.shape` is `[1,K]` vs
  `[K]` vs `[K,1]` we need a normalizer. weight_map.cpp does no
  shape validation today; weight_upload.cu reads flat numel. Both
  need a small contract update.

## 7. Decision recommendation

**Defer (recommended given model scarcity).** Three reasons:

1. **No production workload depends on this.** PR #78
   (`use_default_system_prompt=false`) keeps the Mistral-3.2-NVFP4
   drift below the threshold for typical chat. Long-context users
   on this specific model are zero in our user base today.
2. **The model isn't even available locally.** Phase 1 (~1-2 days
   to download + diagnose) is the smallest prerequisite for any
   further work, and gating engineering investment on Phase 1
   completing first is appropriate.
3. **Phase 1 result is non-trivially likely to retire Phases 2-4
   entirely.** Sub-case (b) (SmoothQuant fused into layernorm,
   nothing left for the engine to do) is the most likely outcome
   given how llm-compressor exports SmoothQuant-calibrated NVFP4 —
   and that result reframes the open issue as the much larger
   "dynamic NVFP4 activation quantization" workitem, which has its
   own design memo home (`nvfp4_long_context_regression_2026_04_28.md`
   § "Next-step methodology" item 2).

**Conditional upgrade — Ship Phase 1 only** (~1-2 days). If
scheduling permits a 1-2 day window before the next major perf
push, do Phase 1 to (a) confirm or refute the per-channel
hypothesis, (b) decisively close out the roadmap entry one way or
the other, and (c) give future bench / debug sessions concrete
data on Mistral-3.2-NVFP4 behavior. Phases 2-4 stay deferred until
a SmoothQuant-calibrated workload becomes important to a real user
of imp.

**Full Phase 1-4 — not recommended today.** The Mistral-Small-3.2-Instruct
family is a popular open-source model, so if a user surfaces, the
full work is justified — but speculatively shipping it without
that signal is gold-plating.

### Justification (one sentence)

The work is bottlenecked on a model imp doesn't have, the most
likely Phase 1 outcome would retire Phases 2-4 anyway, and PR #78
already covers the only production-visible symptom — making any
engineering effort before Phase 1 evidence appears speculative.

## Cross-references

- Memos:
  `memory/mistral_3_2_nvfp4_use_default_system_2026_04_28.md` —
  PR #78 chat-template fix + initial bisection of the drift.
  `memory/nvfp4_long_context_regression_2026_04_28.md` —
  detailed bisection + sub-case (b) "noise grows with ||X||"
  hypothesis.
  `memory/llm_compressor_input_scale_dead_end_2026_05_07.md` —
  the scalar-alpha-multiplier A/B that refuted the easy fix.
  `memory/llm_compressor_phase2_item1_2026_04_26.md` —
  Mistral3 prefix + config_groups parser shipped; loader is ready.
- Code:
  `src/quant/nvfp4_quant.h`, `src/quant/nvfp4_quant.cu` —
  activation quantization (where Phase 2 lives).
  `src/model/weight_upload.cu:1950-2057` — current `input_scale`
  load + audit.
  `src/model/weight_map.cpp:59-70, 591, 610, 700, 738, 784, 1049`
  — `input_scale` routing into `nvfp4_scratch_`.
  `src/model/llm_compressor_loader.cpp:149` —
  `.input_global_scale` → `.input_scale` rename.
  `src/graph/executor_pre_dequant.cu:316-435` — Phase 0 promotion
  + the "intentionally NOT applied" log line.
  `src/model/model_config.h:138-143` — `NvFP4PreQuantWeight`
  struct.
- Roadmap: `docs/roadmap.md:35-39` ("NVFP4 SmoothQuant
  input_scale (Mistral-3.2 NVFP4)").

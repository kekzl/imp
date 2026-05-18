# NVFP4 SmoothQuant `input_scale` Phase 1 findings

**Date:** 2026-05-18
**Branch:** main (`weight_upload.cu` diagnostic edit only, no semantic change)
**Scope:** Resolve the design memo §4 sub-case question (`input_scale` per-tensor scalar vs per-channel `[K]`) for every NVFP4 checkpoint imp can reach today, **including** `Mistral-Small-3.2-24B-Instruct-2506-NVFP4` (resolved via HF range-fetch of the safetensors headers + recipe.yaml, no full download required for the answer; full download running in parallel as a follow-up artifact).
**Design memo:** `docs/plans/nvfp4_smoothquant_input_scale_design_2026_05_17.md` §5 Phase 1
**Diagnostic patch:** `src/model/weight_upload.cu:1965-2079` — adds `is_scalar_count` / `is_per_channel_count` split + per-sample `ndim/shape/numel` logging to the `diagnostics.audit_nvfp4_scales=true` audit. Default-off; no perf or semantic change.

## Measurements

Two independent paths cross-check the same question.

### Path 1 — SafeTensors header scan (Python, no engine load)

`input_scale` / `input_global_scale` tensor shapes pulled directly from the SafeTensors shard headers across every local NVFP4 checkpoint. No engine, no GPU.

| Model | input_scale tensor count | shape distribution |
|---|---:|---|
| `Qwen3-8B-NVFP4-cortecs`              |    252 | `(1,)` × 252 |
| `Gemma-4-26B-A4B-it-NVFP4`            | 11 725 | `(1,)` × 11 725 |
| `Qwen3.6-35B-A3B-NVFP4`               | 30 880 | `(1,)` × 30 880 |
| `Qwen3-30B-A3B-NVFP4-Modelopt`        | 18 624 | `()`   × 18 624 (rank-0) |
| `Nemotron-3-Nano-30B-A3B-NVFP4`       |  5 968 | `()`   × 5 968  (rank-0) |
| `Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4` | 31 030 | `(1,)` × 31 030 |

**All 6 models, 100 % scalar.** rank-0 vs rank-1 is a llm-compressor-version cosmetic; `numel() == 1` either way.

### Path 2 — Engine audit log (C++, full weight upload)

`imp-cli --model Qwen3-8B-NVFP4-cortecs --set diagnostics.audit_nvfp4_scales=true --prompt Hi --max-tokens 1`:

```
NVFP4 audit: input_scale present in 252/252 Linears (scalar=252 per_channel=0),
  stats — count=252 zeros=0 min=0.25 max=7776 mean=457.443
  sample: L17.w_up.input_scale  ndim=1 shape=[1] numel=1 first=231
  sample: L24.wq.input_scale    ndim=1 shape=[1] numel=1 first=44.25
  sample: L23.wk.input_scale    ndim=1 shape=[1] numel=1 first=58.75
  sample: L21.wo.input_scale    ndim=1 shape=[1] numel=1 first=516
  sample: L2.wv.input_scale     ndim=1 shape=[1] numel=1 first=2320
  sample: L20.w_down.input_scale ndim=1 shape=[1] numel=1 first=154
  sample: L11.wo.input_scale    ndim=1 shape=[1] numel=1 first=1312
  sample: L11.wk.input_scale    ndim=1 shape=[1] numel=1 first=312
NVFP4 prequant: 252 Linears carry input_scale (intentionally NOT applied; set IMP_AUDIT_NVFP4_SCALES=1 for stats).
```

C++ audit matches the Python header scan exactly: 252 Linears, all scalar, all `ndim=1 shape=[1]`. Per-Linear values span 0.25 – 7 776 (≈30 000× dynamic range) — characteristic of llm-compressor's per-Linear activation absmax calibration (`input_global_scale`), not a per-channel SmoothQuant `1/s` vector.

### Recipe corroboration

Every local NVFP4 model's `recipe.yaml` ships **only** a `QuantizationModifier` — none ship a `SmoothQuantModifier`. SmoothQuant calibration was never applied to any of these checkpoints.

```yaml
# representative recipe.yaml (Qwen3-8B-NVFP4-cortecs)
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: NVFP4
```

## Acceptance criteria evaluation

Design memo §4 enumerates three sub-cases for `input_scale`:

- **(a)** `input_scale` is already per-channel `[K]` on disk.
- **(b)** `input_scale` is scalar; the per-channel `1/s` was fused into the upstream `input_layernorm.weight` at calibration time.
- **(c)** `input_scale` is per-channel but stored at a non-obvious key the loader currently drops.

For the local NVFP4 corpus:

| Question | Answer |
|---|---|
| Per-channel `input_scale` on disk? | **No** — all 6 models, every Linear is `numel=1`. Sub-case (a) **refuted**. |
| Hidden per-channel key dropped by loader? | **No** — header scan covers `*input_global_scale` *and* `*input_scale`; nothing else matches. Sub-case (c) **refuted**. |
| SmoothQuant fused into `input_layernorm.weight`? | **N/A** — no SmoothQuantModifier ran during calibration. Sub-case (b) **vacuously true** (scalar shipped, no per-channel `s` exists in the first place). |

**Verdict for local corpus: there is nothing for the engine to do.** The per-Linear scalar is llm-compressor's `input_global_scale` activation calibration anchor, and imp's dynamic input quantizer (`src/quant/nvfp4_quant.cu:421-462` `quantize_fp16_to_nvfp4`) recomputes per-micro-block absmax at inference — correctly ignoring the calibration-time scalar that would otherwise double-apply correction. The 2026-05-07 DIVIDE/MULTIPLY refutation on Gemma-4-NVFP4 (`memory/llm_compressor_input_scale_dead_end_2026_05_07.md`) is now structurally explained: the scalar carried no SmoothQuant information *because no SmoothQuant was ever calibrated into it*.

## Decision

**Local corpus: CLOSE.** No engine work required for any of the 6 NVFP4 models in scope today. The "intentionally NOT applied" stance at `executor_pre_dequant.cu:431` is the correct behavior. Diagnostic patch ships as the lasting deliverable (lets future debug sessions distinguish scalar from per-channel without a re-scan).

**Mistral-3.2-NVFP4 — RESOLVED (sub-case b confirmed) 2026-05-18 via HF range-fetch.** Without downloading the full 15 GB SafeTensors, the recipe.yaml + the first 16 MB of each shard (just enough for the safetensors header JSON) gave a definitive answer.

### recipe.yaml — `SmoothQuantModifier` mappings

```yaml
SmoothQuantModifier:
  smoothing_strength: 0.9
  mappings:
  - !!python/tuple
    - ['re:.*q_proj', 're:.*k_proj', 're:.*v_proj']
    - re:.*input_layernorm
  - !!python/tuple
    - ['re:.*gate_proj', 're:.*up_proj']
    - re:.*post_attention_layernorm
  ignore: []
```

This is **exactly** the structure the design memo §4 sub-case (b) predicted: SmoothQuant migrates the per-channel `diag(1/s)` factor into the **upstream RMSNorm weights** (`input_layernorm.weight` for attention Linears, `post_attention_layernorm.weight` for FFN Linears) at calibration time.

### Header scan — `input_scale` is 100 % scalar on Mistral-3.2-NVFP4 too

```
Mistral-3.2-NVFP4 input_scale: 280 tensors total
shape distribution: {(1,): 280}
  example shape=(1,): language_model.model.layers.0.mlp.down_proj.input_global_scale

input_layernorm.weight + post_attention_layernorm.weight:
  shape distribution: {(5120,): 80}
```

280 scalar `input_scale` tensors confirm sub-case (a) and (c) are refuted on the SmoothQuant model too. The 80 `(5120,)` layernorm weights (`hidden_size=5120 × 40 layers × 2 norm types`) are exactly the migration targets the recipe specifies — `diag(1/s)` lives there.

### Post-download engine audit (full cross-check)

After the 15 GB download completed, `imp-cli --model /models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 --set diagnostics.audit_nvfp4_scales=true --prompt Hi --max-tokens 1` confirms the header-scan result through the actual C++ loader path:

```
NVFP4 audit: input_scale present in 280/280 Linears (scalar=280 per_channel=0),
  stats — count=280 zeros=0 min=12 max=8832 mean=2167.09
  sample: L29.wv.input_scale     ndim=1 shape=[1] numel=1 first=3072
  sample: L30.wo.input_scale     ndim=1 shape=[1] numel=1 first=692
  sample: L29.wq.input_scale     ndim=1 shape=[1] numel=1 first=3072
  sample: L30.w_up.input_scale   ndim=1 shape=[1] numel=1 first=3200
  sample: L29.w_gate.input_scale ndim=1 shape=[1] numel=1 first=3136
  ...
NVFP4 prequant: 280 Linears carry input_scale (intentionally NOT applied; set IMP_AUDIT_NVFP4_SCALES=1 for stats).
```

Stats vs Qwen3-8B-NVFP4-cortecs (non-SmoothQuant): Mistral's per-Linear absmax range is `12 – 8832 (mean 2167)` vs Qwen3-8B's `0.25 – 7776 (mean 457)` — Mistral's higher mean and tighter floor are consistent with SmoothQuant deliberately widening per-Linear (output-channel) absmax in exchange for tighter per-input-channel ranges. The distributions are different but both are pure scalars; neither carries any information the engine could use beyond what the dynamic input quantizer already recomputes per micro-block at inference.

### What this means for the engine

Sub-case (b) — fully confirmed. There is nothing for the engine to add: SmoothQuant's per-channel `1/s` is already baked into the upstream RMSNorm weights, which imp already loads and applies via the standard layernorm path (`src/compute/norm_*.cu`). The per-Linear scalar `input_global_scale` on disk is just llm-compressor's post-smoothing NVFP4 activation calibration anchor, which imp's dynamic input quantizer correctly ignores by recomputing per-micro-block absmax at inference (same mechanism as for the 6 non-SmoothQuant local NVFP4 models).

**The roadmap item "NVFP4 SmoothQuant input_scale" retires.**

PR #78's long-context drift workaround (`use_default_system_prompt=false`) is still load-bearing for Mistral-3.2-NVFP4, but its root cause is **NOT** missing SmoothQuant absorption — it's the **NVFP4-activation-noise-grows-with-`||X||`** hypothesis tracked separately in `memory/nvfp4_long_context_regression_2026_04_28.md` §"Suggestive evidence". Fixing that is a much larger workitem (dynamic NVFP4 activation quantization end-to-end + a true NVFP4×NVFP4→FP32 GEMM in the dense path) with its own design memo home, not part of this roadmap entry.

**Phase 2-4 (Option-B `smooth_activations` pre-pass kernel + regression tests) — RETIRED, never needed.** No checkpoint in the foreseeable workload triggers a non-null `s_inv` path; SmoothQuant-calibrated NVFP4 checkpoints all bake `1/s` into upstream RMSNorm at calibration time.

## Diagnostic patch lasting value

The `scalar=N per_channel=M` split + `ndim/shape/numel` per-sample log lives in `src/model/weight_upload.cu:1965-2079`, gated to `diagnostics.audit_nvfp4_scales=true` (default-off, env-compat `IMP_AUDIT_NVFP4_SCALES=1`). Re-runs on any future NVFP4 checkpoint immediately answer the per-channel question without re-reading SafeTensors headers. Cost: 8 extra audit-mode-only `INFO` log lines per model load; zero VRAM, zero perf delta on production runs.

## Cross-references

- Design memo: `docs/plans/nvfp4_smoothquant_input_scale_design_2026_05_17.md`
- 2026-05-07 refutation memo: `memory/llm_compressor_input_scale_dead_end_2026_05_07.md`
- Long-context bisection: `memory/nvfp4_long_context_regression_2026_04_28.md`
- PR #78 (production workaround for Mistral-3.2-NVFP4): `memory/mistral_3_2_nvfp4_use_default_system_2026_04_28.md`
- Roadmap entry: `docs/roadmap.md` "NVFP4 SmoothQuant input_scale (Mistral-3.2 NVFP4)"

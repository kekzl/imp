# AWQ groups for MoE experts

Roadmap row 99 (#2476). Question: do per-expert AWQ groups bring a calibrated Gemma-4-26B-A4B export below round-to-nearest and the reference NVFP4 export?

## Groups

- `imp-cli --calibrate` records `EXPERT_UP` (expert input, all routed rows) and `EXPERT_DOWN.<e>` (down input per expert, unrouted experts absent).
- Y: expert e's `down_proj` columns scaled, its `up_proj` rows divided (exact: GLU is elementwise); every MoE layout, 2-D per-expert or 3-D stacked.
- X: all experts' gate/up share one scale folded into `pre_feedforward_layernorm_2`; only Gemma-4 has an expert-only norm (Qwen-MoE experts share `post_attention_layernorm` with the router).
- Gemma-4 dense groups: B folds into `pre_feedforward_layernorm` (sandwich norms), C is off (`v_norm` divides a v_proj row scale back out per head).

## Result

| arm | ref export | RTN | X | Y | XY | ABD | ABDXY |
|---|---:|---:|---:|---:|---:|---:|---:|
| 704 tokens | 16.9208 | **16.6629** | | | 17.1923 | 19.8608 | 18.3303 |
| 14676 tokens | 18.0483 | **16.9442** | 17.2090 | 17.3654 | 17.1043 | 17.1807 | 17.5810 |

[PROV: commit=65e30faa date=2026-10-04 hw=RTX5090 model=gemma-4-26B-A4B-it quant=NVFP4 cuda=13.4.2
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`tools/analysis/awq_moe_gemma4_ab.sh` harness_md5=2a8243ee
       note=tree = 65e30faa + the #2476 diff; judges ppl_corpus_gemma4_turn.txt and ppl_corpus_45k.txt in turn framing; calibration from the RTN checkpoint]

- Search-side variants, 14676 tokens, none below RTN: scales held in [1, 64] (activations never amplified) Y 17.0171, XY 17.1462; an added W4A4 activation-noise term Y 17.5069, XY 17.7023.
- The search objective drops while PPL rises (Y -41 %, X -5 %, A -38 %): the diagonal weight-error proxy does not predict this model's output error.
- RTN re-quantized with this tree is byte-identical to the 2026-09-16 export: the uncalibrated path is unchanged.
- The issue's baseline moved: 26.99 (imp-quantize) vs 26.06 (reference) on 2026-09-16, today 16.6629 vs 16.9208 on the same 704-token corpus.

Decision: X/Y opt-in (`kAwqDefaultGroups` = `ABCDEG`, `tools/imp-quantize/awq_sites.h`); `gemma4` refuses `--calib` without `--calib-groups`. Open: an objective that predicts W4A4 output error.

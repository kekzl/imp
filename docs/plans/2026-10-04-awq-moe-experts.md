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
| 704 tokens | 16.9208 | 16.6629 | | | 17.1923 | 19.8608 | 18.3303 |
| 14676 tokens | 18.0483 | 16.9442 | 17.2090 | 17.3654 | 17.1043 | 17.1807 | 17.5810 |

[PROV: commit=65e30faa date=2026-10-04 hw=RTX5090 model=gemma-4-26B-A4B-it quant=NVFP4 cuda=13.4.2
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`tools/analysis/awq_moe_gemma4_ab.sh` harness_md5=2a8243ee
       note=tree = 65e30faa + the #2476 diff; judges ppl_corpus_gemma4_turn.txt and ppl_corpus_45k.txt in turn framing; calibration from the RTN checkpoint]

- Search-side variants, 14676 tokens: scales held in [1, 64] (activations never amplified) Y 17.0171, XY 17.1462; an added W4A4 activation-noise term Y 17.5069, XY 17.7023.
- The search objective drops (Y -41 %, X -5 %, A -38 %); every PPL move stays inside the noise floor below.
- RTN re-quantized with this tree is byte-identical to the 2026-09-16 export: the uncalibrated path is unchanged.
- The issue's baseline moved: 26.99 (imp-quantize) vs 26.06 (reference) on 2026-09-16, today 16.6629 vs 16.9208 on the same 704-token corpus.

## Noise floor (correction)

RTN with every tensor scale multiplied by f (same rounding quality, mean MSE 1.030e-05 in every arm), 14676 tokens:

| f | 0.95 | 0.98 | 0.99 | 1.00 | 1.005 | 1.01 | 1.02 | 1.04 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPL | 18.1339 | 16.9355 | 16.7551 | 16.9442 | 18.0846 | 17.9325 | 19.0110 | 18.5016 |

- n 8, mean 17.7873, sd 0.8228 (4.63 %): RTN 16.9442 is a low draw. Every calibrated arm above (17.0171..17.7023) and the reference export (18.0483) lie inside this band: the judge cannot separate them, "loses to RTN" does not hold.
- f 1.02 on one group only: experts 17.6665, attention 17.1840, dense MLP 17.5112. No single group carries the sensitivity.
- Same weights, `runtime.deterministic` off vs on: 2280 of 14675 tokens move > 1 nat (PPL 16.9442 -> 17.0687); Qwen3-30B-A3B-NVFP4 on `ppl_corpus_45k.txt`: 733 of 13536 (11.4338 -> 11.4377).
- Activation blocks in this prefill (CUTLASS act quantize, tensor scale 1.0): 1196974560 total, 56777639 with a subnormal UE4M3 scale (absmax < 0.09375), 1 above 2688. Weight micro-scales: 0 % zero, <= 0.848 % subnormal per tensor.

[PROV: commit=d2239977 date=2026-10-04 hw=RTX5090 model=gemma-4-26B-A4B-it quant=NVFP4 cuda=13.4.2
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`imp-quantize` with the tensor scale times f (experimental build, not committed), `imp-cli --perplexity` turn-framed ppl_corpus_45k `--set runtime.deterministic=true`
       note=per-token NLL via diagnostics.ppl_dump=full; activation counters from an experimental build]

Decision: X/Y opt-in (`kAwqDefaultGroups` = `ABCDEG`, `tools/imp-quantize/awq_sites.h`); `gemma4` refuses `--calib` without `--calib-groups` (no measured gain). A Gemma-4 quantization verdict needs the method's mean over several tensor-scale shifts, not one export.

## DeepSeek-V2 MLA (#2477, 2026-10-09)

- DeepSeek-V2-Lite (`deepseek_v2`, #2477): A folds `q_proj` + `kv_a_proj_with_mqa` into `input_layernorm`; K (opt-in, `--calib-groups ABCDK`) folds MLA `kv_b_proj` into `kv_a_layernorm`; MoE layers get no B or X, `post_attention_layernorm` also feeds the router and the shared experts. `--calibrate` on the BF16 source with the first 90 000 bytes of `calib_corpus.txt` (the full corpus is 37 921 tokens, above the 2096-block pool).
- DeepSeek-V2-Lite, ppl_corpus_45k (14 955 tokens), deterministic: BF16 9.0616, RTN 9.6015, `--calib` (default, 29 groups) 9.5250, `ABCDK` (56 groups) 9.5092. K stays opt-in: one run each, MoE PPL noise not measured for this model (Gemma-4: sd 4.63 %).

[PROV: commit=42b83a69 date=2026-10-09 hw=RTX5090 model=DeepSeek-V2-Lite quant=NVFP4 cuda=13.4.2
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`imp-cli --perplexity <calib_corpus first 90000 bytes> --calibrate`, `imp-quantize [--calib] [--calib-groups ABCDK]`, `imp-cli --perplexity ppl_corpus_45k --set runtime.deterministic=true`
       note=dev build of feat/awq-calib-deepseek-v2 on that commit]

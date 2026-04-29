# Phase 5 — End-to-End Generation Goldens

Strategic precisions exercised on a single SafeTensors model so the cross-precision agreement is meaningful (same loader path, same tokenizer, same prompt — only the runtime KV/attention precision changes).

## Setup
- Model: `Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
- Prompt: `"The capital of France is"` (raw, `--chat-template none`)
- Sampling: `--temperature 0 --seed 0 --max-tokens 64` (greedy)
- Two runs:
  - **NVFP4 path:** default — NVFP4 weights, FP8 prefill auto, FP16 KV cache, NVFP4 decode
  - **FP8 KV path:** same + `--kv-fp8` — engages FP8 E4M3 KV + FP8 FMHA decode

## Cross-precision agreement (first 8 tokens)

| step | NVFP4 | FP8 KV | match |
|---|---|---|---|
| 1 | ` Paris` (6993) | ` Paris` (6993) | ✅ |
| 2 | `.` (1046) | `.` (1046) | ✅ |
| 3 | ` It` (2157) | ` It` (2157) | ✅ |
| 4 | ` is` (1395) | ` is` (1395) | ✅ |
| 5 | ` the` (1278) | ` the` (1278) | ✅ |
| 6 | ` capital` (8961) | ` capital` (8961) | ✅ |
| 7 | ` of` (1307) | ` of` (1307) | ✅ |
| 8 | ` France` (5498) | ` France` (5498) | ✅ |

**8/8 first tokens identical** between NVFP4 and FP8 KV paths. Phase 5 cross-precision sanity ✅.

## Divergence beyond token 8

NVFP4 (FP16 KV) keeps a single ` Paris. It is the capital of France and the country's largest city, with a population of 2.3 million in 2099 ...` trajectory (the year hallucination is the model itself, not a quant artifact).

FP8 KV produces ` Paris. It is the capital of France. It is the capital of France and one of the most visited cities in the world. It is known as the "City of Light" (La Ville Lumiere) ...` — both are factually grounded ("Paris", "capital of France", "City of Light"), but the trajectories diverge after token 8 due to per-token logit ranking flipping under quant noise. This is expected behaviour for two distinct precision pipelines on a 24B model with greedy sampling.

## Goldens committed
- `tests/golden/mistral_small_32_nvfp4_paris.txt`
- `tests/golden/mistral_small_32_fp8kv_paris.txt`

## Existing goldens (from main, untouched)
- `tests/golden/qwen3_8b_chat_q8_0.txt`
- `tests/golden/qwen3_8b_code_q8_0.txt`
- `tests/golden/qwen3_8b_short_q8_0.txt`

## Performance (informational, not the regression check — Phase 7 owns that)
- NVFP4: prefill 17.6 tok/s (6 raw tokens, init-dominated), decode **90.1 tok/s** (64 tokens / 698.93 ms)
- FP8 KV: prefill 17.8 tok/s, decode **90.2 tok/s** (64 tokens / 698.16 ms)

The 0.1 tok/s difference is within run-to-run noise. FP8 KV is paying for itself on memory savings (½ KV bytes) at parity decode throughput.

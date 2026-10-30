# Benchmarks on GGUF files (archived 2026-10-30)

Moved verbatim from `docs/BENCHMARKS.md` when GGUF went legacy and vLLM became the only
reference engine (`docs/DESIGN_DECISIONS.md`).

## Decode vs llama.cpp (dense GGUF)

Same card, same GGUF, same flags, decode tok/s; imp defaults (n-gram speculation on; dense Q8_0 / Q6_K
decode through an NVFP4 copy, level with llama.cpp at source precision: [cost](../QUANTIZATION.md#choosing-a-quant))
against llama.cpp defaults, full offload, flash attention on.

| Model (shared quant) | imp default | imp spec-off | llama.cpp | imp lead |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | **397.51** | 278.42 | 152.74 | +160% (+82% spec-off) |
| Qwen3-14B Q6_K | **167.43** | 168.65 | 112.84 | +48% (+49% spec-off) |
| Qwen3.6-35B-A3B UD-Q4_K_M | **317.41** | 319.01 | 220.09 | +44% (+45% spec-off) |
| Gemma-4-26B-A4B UD-Q4_K_M | **262.71** | 262.32 | 212.49 | +24% (+23% spec-off) |
| gpt-oss-20b MXFP4 | **391.52** | 397.67 | 317.11 | +23% (+25% spec-off) |
| Qwen3-30B-A3B Q4_K_M (non-hero) | 357.23 | 359.23 | 303.53 | +18% (+18% spec-off) |

[PROV: commit=54921da6 date=2026-10-11 hw=RTX5090 model=six-model-sweep quant=per-row cuda=13.4.2
       path=gguf cmd=`make bench-competitive` (v0.51.0 release gate, 11 idle service containers up)
       note=imp defaults vs llama.cpp defaults, full offload, flash attention on; spec-off =
       `--set speculative.ngram=false`. Previous sweep 83cb5178 (2026-08-30): Qwen3-8B 385.4 vs 160.1]

The spec-off column exists because `imp-cli --bench` builds a strictly-increasing synthetic
prompt (`tools/imp-cli/mode_bench.cpp:34 tokens[i] = i % vocab_size`) that the n-gram drafter's own generation can loop
into: a decode A/B that leaves speculation on can measure whether a checkpoint happens to loop,
not only the verify path. Reproduce with `make bench-competitive` (competitor image pinned by
digest in [`scripts/bench_competitive.sh`](../../scripts/bench_competitive.sh), not by tag); two
earlier sweeps (2026-07-12, 2026-08-21) with the full acceptance-mechanism analysis and a
bimodal-drafter case study: [archive](BENCHMARKS_PRE_V0.44.md).

## GGUF prefill (pp512, INT8-IMMA family, default on since #617)

2026-10-09, `7e720263`, CUDA 13.4, `imp-cli --model <gguf> --bench --bench-pp 512 --bench-reps 10`, fresh
container, 3 alternating runs per engine, median (range); llama.cpp `b10524` (digest in
`scripts/bench_competitive.sh`), same GGUF, same day, `-fa 1 -ngl 999 -r 5 -n 0`.

| Model (quant) | imp tok/s | llama.cpp | verdict | 2026-06-07 imp / llama.cpp |
|---|---:|---:|---|---|
| Qwen3-30B-A3B MoE (Q4_K_M) | **15206** (15142-15256) | 9787 (9681-10209) | imp 1.55x | 9970 / 9288 |
| Qwen3-14B (Q6_K) | **6820** (6810-6850) | 6732 (6718-6745) | imp +1.3% | 6617 / 6522 |
| Gemma-4-26B-A4B MoE (UD-Q4_K_M) | **13765** (13761-13778) | 10912 (10678-11457) | imp 1.26x | 8946 / 10749 |
| Qwen3-8B (Q8_0) | **14195** (13953-14583) | 14258 (13644-14286) | parity (-0.4%) | 12131 / 13724 |
| Qwen3.6-35B-A3B hybrid (UD-Q4_K_M) | **13603** (13259-13618) | 8468 (8403-8529) | imp 1.61x | 5165 at `62d96a0e` / 8027 |

**Gemma-4 pp depends on context**: the command decodes 8192 tokens (context 9792), the KV guarantee
caps the IMMA Q8_0 prefill planes 1765 -> 0 MiB (#2652); `--max-tokens 128` keeps them: 15050.
**2026-06-07 column reads high** (prefix-cache hits, #758 -> #1061): [archive](BENCHMARKS_PRE_V0.44.md).

## Agentic reliability vs llama.cpp (2026-07-26)

Qwen3-8B-Q8_0 GGUF on both engines, same prompts, `temperature=0`, 5 reps per case,
`max_tokens=200`; llama.cpp `ff067f76d` (build 10133) with `--jinja -fa 1 -ngl 99`. Harness:
`tools/analysis/agentic_compare.py`.

| Case | imp (default) | llama.cpp (default) | llama.cpp (`enable_thinking:false`) |
|---|:--:|:--:|:--:|
| `json_schema` schema-valid | **5/5** (10 tok) | 0/5 | 5/5 (23 tok) |
| `json_object` parses | **5/5** (27 tok) | 0/5 | 5/5 (18 tok) |
| `tool_choice=required` emits a call | 5/5 | 5/5 (197 tok) | 5/5 |
| tool args parse + required field | 5/5 | 5/5 | 5/5 |
| `tool_choice=auto` does not force a call | 5/5 | 5/5 | 5/5 |

- Defaults difference, not capability difference: llama.cpp's think-capable model spends its whole budget on reasoning and returns empty `content` unless told not to think; imp suppresses thinking for json/tool requests automatically.
- Budget sweep (Qwen3-8B, 3 reps): imp passes all 6 categories from budget 100; llama.cpp needs budget 800 (`json_schema` then costs 447 tok) to match.
- Three families, 8-turn sessions (budget 200, 3 reps): imp 6/6, 6/6, 5/6 (Gemma-3 has no native function calling, a model limit) against llama.cpp 3/6, 4/6, 4/6 (reaches 6/6 at budget 800).
- `json_object` holds on imp across all three models, fails on llama.cpp for two.

**Scope**: three model families, four budgets, 8-turn sessions, 3-5 repetitions, one llama.cpp
build, GGUF only; a non-thinking-model control (Llama-3.2-3B) found and fixed an imp tool-call
regression along the way; vLLM/SGLang not covered. Full per-budget tables and the control-run
detail: [archive](BENCHMARKS_PRE_V0.44.md).

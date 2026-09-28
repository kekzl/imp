---
name: numerics-bisect
description: Use when imp outputs differ between two paths that should agree - chunk sizes, prefill vs decode, prefix-cache resend, two engines (imp vs llama.cpp vs HF), two builds or kernels - "output changes with chunk size", "first token differs", "logprobs moved", "which layer diverges", "who is right, imp or llama.cpp", "near-tie". Do NOT use for repetition loops or garbage output (check-degeneration), a new arch that loads wrong (add-model-arch), or perf (benchmark-cuda).
---

# Numerics bisect - imp

Order is fixed: ids, logprobs, hidden states, reference. Each rung is cheaper than the next and kills a class of false leads.

## Rungs

| # | Step | Tool | Decision |
|---|---|---|---|
| 1 | Compare token ids of both prompts first | `imp-cli` / server token ids vs the other path | ids differ = tokenizer or template bug (add-model-arch step 5), stop here |
| 2 | Logprob grid over the axis (chunk size, engine, build) | first-token logprobs per arm; chunk axis via `--prefill-chunk-size <n>` (`runtime.prefill_chunk_size`); raw logits via `diagnostics.dump_logits_dir` | a spread across the axis is the signal; greedy-text equality alone amplifies near-ties |
| 3 | Hidden-state dumps per layer | `diagnostics.dump_hidden_dir` + `tools/analysis/layer_ab_diff.py` (two imp runs) or `tools/analysis/layer_diff.py` (vs llama.cpp) | first block with non-zero added divergence (rel@out - rel@in) owns the bug |
| 4 | Reference arbitration | HF transformers fp32 (GGUF: `from_pretrained(..., gguf_file=...)`, dequantized) plus llama.cpp with `-fa off` | the arm that matches both references is right |
| 5 | Judge near-tie tokens by multi-token NLL | `imp-cli --perplexity tools/analysis/ppl_corpus_45k.txt` (rebuild with `tools/analysis/make_ppl_corpus.sh`), `--set runtime.deterministic=true` both arms | never by one token: llama.cpp FA on vs off alone moved a near-tie token by 0.082 |

## Dump traps

| Trap | Fix |
|---|---|
| `[DUMP_NPY] open failed` | docker `--user $(id -u):$(id -g)` so the container can write the mounted dump dir |
| Dumps per stage, overwritten per forward | names carry layer, step and n, not the prefill offset: `imp_L<layer>_step<step>_n<n>.bin` (`src/exec/executor_forward.cu`), `imp_step<step>_L<layer>_<tag>.npy` per stage (`src/exec/executor_debug.h`); prefill is step 0, so chunks of equal n overwrite each other: copy the dir between forwards and key it by prefill offset for chunk studies |
| Default layer set | `diagnostics.dump_hidden_dir=<path>` dumps layers 0/5/15/29; `=all` dumps every layer into `/tmp` |

## Known causes of chunk-dependent rows

- Kernel variant chosen by row count (sm120-cuda-expert numerics rules, #2152).
- MoE NVFP4 activation tensor scale = per-expert batch absmax (quant-formats, #2167).
- Resend check: `tools/analysis/prefix_resend/prefix_resend_probe.py` (check-degeneration, from #2171).

---
name: check-degeneration
description: Use when verifying that a model in the imp inference engine produces coherent output without repetition loops, token-stuck states, or state corruption across turns. Triggers on "degenerates", "check degeneration", "repetition loop", "own own own", "stuck token", "multi-turn regression", "does it still work", and after enabling CUDA graphs / changing forward pass / MoE routing / KV cache / GDN state.
---

# Degeneration Check — imp Inference Engine

Models in imp regress in specific, recurring ways. Run the battery whenever you touch the forward pass, graph capture, MoE routing, KV cache, or GDN state.

## Historical failure modes (what we are guarding against)

| Pattern | Looks like | Root-cause class |
|---|---|---|
| Repetition loop | ` own own own…`, ` the the the…` | MoE router precision drift / graph stale pointers / sampling NaN |
| Short OK, long fails | 15 tokens sensible then loop | KV-block boundary in graph replay; D2H memcpy in captured region |
| ~3-token abort | `<eos>` or stop_id at step 0–3 | `forward_decode_async` ≠ `forward_logits`; sampler divergence |
| Multi-turn garble | Turn 1 OK, turn 2 `The I…` | KV not reset; GDN recurrent state leaked; warmup CUDA error |
| Stuck single token | ` a a a a a` | Logits NaN/Inf; banned mask wrong value; argmax on zeroed buffer |
| Structurally valid garbage | Grammatical but wrong language mid-stream | Weight-upload bug; quant dequant layout |

## Pass criteria (quantitative)

A run **passes** only when ALL of these hold:

1. **No verbatim repetition**: no token repeats >4 times in a row, no 3-gram repeats >3 times.
2. **No early abort**: ≥10 generated tokens before any stop condition (unless prompt is single-word factual like "Paris").
3. **stderr clean**: no match for the grep pattern below (silent fallback masks the bug).
4. **Decode within 30% of baseline** in `tests/perf_baseline.json` for the same model. >30% drop almost always means graphs fell back silently.
5. **Multi-turn**: turn 2+ output is grammatical and topical for the new question; KV/state from turn 1 didn't leak.

## The battery

Use existing tooling — do **not** write parallel inline checks.

### 1. GTest battery (covers Short / Second-request / Long / NoLeakedSpecialTokens)

```bash
make test-gpu GTEST_FILTER="DegenerationTest.*"
# or with an explicit model:
IMP_TEST_MODEL=/models/<MODEL>.gguf \
  ./build/imp-tests --gtest_filter="DegenerationTest.*"
```

Source: `tests/test_degeneration.cpp` — uses the same imp C API the production engine uses, default model `Qwen3-8B-Q8_0.gguf`.

### 2. Smoke + perf gate (covers real-prompt detector + baseline regression)

```bash
make verify-fast    # build + filtered tests + perf baseline + 1 smoke prompt (~90s)
```

Source: `scripts/verify.sh` — runs the canonical degeneration detector against a real model and fails on >3% decode regression vs. `tests/perf_baseline.json`.

### 3. Cross-stack docker smoke (covers tokenizer / chat-template / vision)

```bash
docker run --rm --gpus all -v $PWD/models:/models imp:test \
  bash /scripts/smoke_test.sh
```

Source: `scripts/smoke_test.sh` — runs unit + GPU + E2E + tokenizer + chat-template checks in 6 stages.

### 4. Graphs vs no-graphs parity (only when graph capture / MoE fast-path / async loop changed)

```bash
# graphs off
./build/imp-cli --set runtime.cuda_graphs=never \
  --model models/<MODEL>.gguf --prompt "..." --max-tokens 64 \
  --temperature 0 --seed 42 > /tmp/no_graph.txt

# graphs on (default)
./build/imp-cli \
  --model models/<MODEL>.gguf --prompt "..." --max-tokens 64 \
  --temperature 0 --seed 42 > /tmp/graph.txt

diff /tmp/no_graph.txt /tmp/graph.txt
```

**Pass**: identical output for greedy (`temperature=0`). First ~16 tokens identical is a strong signal; later drift is allowed only if non-degenerate.

## Reading stderr (mandatory even if output looks fine)

A silent fallback to per-step decode masks the real bug. Use word boundaries — plain `Inf` matches the normal `Inferred vocab_size=` log line:

```bash
grep -E "CUDA error|capture failed|falling back|warmup.*invalid|\bNaN\b|is NaN|is Inf" <log>
```

Any match = **fail**, even if output passed the heuristic. Cross-check measured `tg/s` against the model's row in `tests/perf_baseline.json`; if measured tg is >30% below baseline, graphs almost certainly fell back silently.

## Model-specific known-good probes

| Model | Prompt | Expected contains |
|---|---|---|
| Qwen3-4B Q8_0 | "What is the capital of France?" | `Paris` |
| Qwen3.5-GDN Q8_0 | "Say hello." | non-empty, len ≥ 5 |
| Gemma-4-26B-A4B Q4_K_M | "What is the capital of France?" | `Paris` (after `<|channel>thought` block) |
| Llama-3.2-3B | "The capital of France is" | `Paris` |

Pick the first model from this table whose family matches your change. Stable prompts give stable regressions — do not invent new probes.

## Red flags — STOP and re-run

- Judging by short prompt alone → long-decode bugs pass `--max-tokens 30`. Use ≥64.
- Skipping the stderr grep → silent graph-capture fallback produces wrong output at 20% normal speed.
- Testing only one model after a shared-code change → MoE / GDN / dense paths diverge; pick 2+ probes.
- Declaring success on a single seed → for borderline cases re-run with seeds 42, 1, 7.
- Trusting "looks fine" on multi-turn without a turn-2 probe → KV/GDN state bugs only show up turn 2+.

## When the battery fails

Run `make test-e2e` first — it exercises `Gemma4GraphsTest.LongDecodeStaysCoherent`, `PrimaryModelTest.MultiTurnConversation`, `GDNModelTest.MultiTurnGDNState` which together cover all three state-management classes. Narrow to the failing model + class **before** editing.

---
name: check-degeneration
description: Use when verifying that a model in the imp inference engine produces coherent output without repetition loops, token-stuck states, or state corruption across turns. Triggers on "degenerates", "check degeneration", "repetition loop", "own own own", "stuck token", "multi-turn regression", "does it still work", and after enabling CUDA graphs / changing forward pass / MoE routing / KV cache / GDN state / ragged prefill batching / batched-decode kernels (smallm, producer quantize).
---

# Degeneration Check — imp Inference Engine

Models in imp regress in specific, recurring ways. Run the battery whenever you touch the forward pass, graph capture, MoE routing, KV cache, or GDN state - and since 2026-08 also the default-ON batched paths: ragged prefill (`runtime.prefill_batch`, #1780), batched GDN decode (#1750), the smallm GEMM (`gemm.nvfp4_smallm`, #1766), producer-fused quantize (#1771/#1773) and BF16 GDN state (`gdn.state_bf16`, #1778).

## Historical failure modes (what we are guarding against)

| Pattern | Looks like | Root-cause class |
|---|---|---|
| Repetition loop | ` own own own…`, ` the the the…` | MoE router precision drift / graph stale pointers / sampling NaN |
| Short OK, long fails | 15 tokens sensible then loop | KV-block boundary in graph replay; D2H memcpy in captured region |
| ~3-token abort | `<eos>` or stop_id at step 0–3 | `forward_decode_async` ≠ `forward_logits`; sampler divergence |
| Multi-turn garble | Turn 1 OK, turn 2 `The I…` | KV not reset; GDN recurrent state leaked; warmup CUDA error |
| Stuck single token | ` a a a a a` | Logits NaN/Inf; banned mask wrong value; argmax on zeroed buffer |
| Structurally valid garbage | Grammatical but wrong language mid-stream | Weight-upload bug; quant dequant layout |
| Token-0 garbage (`!`, `!!!` from the first token) | `!!!!…` immediately, sometimes recovers | Silent VRAM-alloc failure in a decode fallback (#934/#935 — MXFP4→FP16 on GDN hybrids); check init logs for failed reserves |
| Answer in `reasoning_content`, `content` empty (server only) | API "empty response", CLI fine | Thinking-state ≠ rendered prompt tail (closed `<think></think>` template block) - see `server-api` reconcile bullet, PR #937. Note: since #1560/#1743 extended thinking on `/v1/messages` is opt-in; the suite's default arm asserts NO thinking block |
| Cross-sequence contamination | Single-stream clean, turn-1 clean; garbled ONLY under a concurrent burst | Ragged prefill row/offset bug, batched GDN slot mixing, shared act-quant scratch collision - the failure class #1780/#1750 introduce. Needs a concurrency probe (below); no single-stream step can see it |

## Pass criteria (quantitative)

A run **passes** only when ALL of these hold:

1. **No verbatim repetition**: no token repeats >4 times in a row, no 3-gram repeats >3 times.
2. **No early abort**: ≥10 generated tokens before any stop condition (unless prompt is single-word factual like "Paris").
3. **stderr clean**: no match for the grep pattern below (silent fallback masks the bug).
4. **Decode within 30% of baseline** in `tests/perf_baseline.json` for the same model. >30% drop almost always means graphs fell back silently.
5. **Multi-turn**: turn 2+ output is grammatical and topical for the new question; KV/state from turn 1 didn't leak.

## The battery

Use existing tooling — do **not** write parallel inline checks.

### 0. Extended server-level suite (categories: repetition, think-leak, special-tokens, adherence, long-context, kv-growth, multi-turn, stream, constrained, anthropic-thinking; `--corpus` adds a ~250-prompt adversarial battery)

The deepest battery — probes a **running imp-server** through the OpenAI API,
where the recurring production failures live (reasoning/think separation,
channel stripping, stop handling, truncated-think spill, prompt-blindness):

```bash
# server must be running, e.g.:
#   docker run --rm --gpus all -p 8080:8080 -v $HOME/models:/models imp:test \
#     imp-server --host 0.0.0.0 --model /models/<MODEL>
# the suite itself is stdlib-only and runs on the host:
python3 tools/analysis/degen_suite.py --url http://localhost:8080
# Qwen3.6 is non-deterministic at temp=0 — skip the equality check:
python3 tools/analysis/degen_suite.py --skip-deterministic
# focused / fast / machine-readable:
python3 tools/analysis/degen_suite.py --only think-leak,adherence --quick --json /tmp/degen.json
```

**Long sessions need the deeper probe.** `degen_suite`'s multi-turn category is
three turns; state that decays and thinking that eats the token budget only show
up later:

```bash
python3 tools/analysis/multiturn_deep.py --url http://localhost:8080 \
    --model <id> --filler 60 --max-tokens 600   # ~74 turns, recalls across all of them
```

It reports `finish_reason` and reasoning length on every failure, which is what
separates a defect from a budget the thinking consumed. **Qwen3.8-27B at
`--max-tokens 260` fails that way and is clean at 600**, in imp and in vLLM
alike, so check the budget before filing an engine bug.

Source: `tools/analysis/degen_suite.py`. Exit 0 = clean, 1 = failures
(printed with evidence), 2 = server unreachable. Run this whenever output
quality is in question — it catches what the C-API GTest battery cannot see.
It is also a release-gate step (`scripts/test_server.sh` invokes it).

The `constrained` category covers json_object/json_schema under three sampler
states and forced tool_choice - constrained-decoding changes ARE covered here;
for deeper changes also run `tests/api/` schema cases (`server-api` skill).

**Coverage gap that remains: the whole battery is single-stream.** Add a burst
arm for any batched-path change: `python3 tools/analysis/conc_client.py <port>
32 4` (or `tools/analysis/load_test.py --levels 1,8,32`) and read the outputs
for coherence, then a byte A/B 32-concurrent vs one-at-a-time on the same
server (deterministic, prefix cache off) - the #1780 ship gate did exactly
that (27/32 identical vs 24/32 control; the divergences are known batch-shape
decode noise, same class both arms).

### 1. GTest battery (covers Short / Second-request / Long / NoLeakedSpecialTokens)

```bash
# GTEST_FILTER is NOT threaded through make test-gpu (it runs the full suite);
# filter via a direct container run:
docker run --rm --gpus all -v $HOME/models:/models \
  -e IMP_TEST_MODEL=/models/<MODEL>.gguf \
  imp:test imp-tests --gtest_filter="DegenerationTest.*"
# after make dev, the incremental binary is build-dev/imp-tests (not build/)
```

Source: `tests/test_degeneration.cpp` - uses the same imp C API the production engine uses, default model `Qwen3-8B-Q8_0.gguf`. Equivalence gates for the new batched defaults live beside it: `tests/test_prefill_ragged.cu` (`RaggedPrefillTest.*` - ragged vs serial byte-EQUAL under `runtime.deterministic`), `tests/test_gdn_batched.cu` (`GdnBatchedScanTest.*` - bit-identical batched decode), `tests/test_nvfp4_batched_smallm_equiv.cu` (`BatchedSmallM.*`).

### 2. Smoke + perf gate (covers real-prompt detector + baseline regression)

```bash
make verify-fast    # build + filtered tests + perf baseline + peak VRAM + 1 smoke prompt
```

Source: `scripts/verify.sh` — runs the canonical degeneration detector against a real model and fails on >3% decode regression vs. `tests/perf_baseline.json`, on >10% growth in own-peak VRAM vs. its `metrics.memory_mb.own_peak_mb` pin, and on a graphs-ON/OFF decode speedup below 1.3×. The last two matter here: a path that fell out of graph capture and a change that quietly claims more memory are both things this battery is meant to catch.

### 3. Cross-stack docker smoke (covers tokenizer / chat-template / vision)

```bash
docker run --rm --gpus all -v $PWD/models:/models imp:test \
  bash /scripts/smoke_test.sh
```

Source: `scripts/smoke_test.sh` - 5 printed stages (unit lane, GPU subset, E2E subset, server smoke, CLI smoke); tokenizer/chat-template coverage is only the ChatML structure test inside stage 1, no vision stage.

### 4. Graphs vs no-graphs parity (only when graph capture / MoE fast-path / async loop changed)

```bash
# graphs off (binary from make dev lives in build-dev/, or run inside imp:test)
build-dev/imp-cli --set runtime.cuda_graphs=never \
  --model models/<MODEL>.gguf --prompt "..." --max-tokens 64 \
  --temperature 0 --seed 42 > /tmp/no_graph.txt

# graphs on (default)
build-dev/imp-cli \
  --model models/<MODEL>.gguf --prompt "..." --max-tokens 64 \
  --temperature 0 --seed 42 > /tmp/graph.txt

diff /tmp/no_graph.txt /tmp/graph.txt
```

For ragged-prefill changes the analogous parity arm is `runtime.prefill_batch`
ON vs OFF on the same prompts (serial fallback exists per request for vision,
constraints, logprobs, embeddings, rerank - those bypass the ragged path
anyway).

**Pass**: identical output for greedy (`temperature=0`). First ~16 tokens identical is a strong signal; later drift is allowed only if non-degenerate.

**Byte-level A/Bs: do NOT diff CLI stdout** — `imp-cli` interleaves log lines with generated text and stripping is error-prone (burned a 2026-07-09 A/B). Run imp-server and diff the JSON `content` field instead. Qwen3.6-**27B** is byte-deterministic at temp=0 (proven on/off-identical, PR #933) — a good graph-parity model; 35B is NOT (see below).

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
| Qwen3.5-4B mxfp4 (the GDN e2e model) | "Say hello." | non-empty, len ≥ 5 |
| Qwen3.8-27B-NVFP4 (GDN hybrid hero; spec-fidelity default) | "What is the capital of France?" | `Paris`; at `--max-tokens <400` an empty `content` is the shared think budget, not a bug |
| Gemma-4-26B-A4B Q4_K_M | "What is the capital of France?" | `Paris` (after `<|channel>thought` block) |
| Llama-3.2-3B | "The capital of France is" | `Paris` |

Pick the first model from this table whose family matches your change. Stable prompts give stable regressions — do not invent new probes.

**Perplexity corpus (2026-07-26):** when you fall back to PPL, use
`tools/analysis/ppl_corpus_45k.txt` (13 537 tokens), **not**
`tools/analysis/ppl_corpus.txt` (199 tokens). The short one does not merely add
noise, it inverts conclusions: the same quantization pair reads +42%/+57% on it
vs +25%/+19% on the real corpus, and appears to get *worse* with model size when
it actually gets better. Pass `--set runtime.deterministic_gemm=true` on both
arms. And PPL is never sufficient on its own — it cannot see a
degenerate-but-low-perplexity model, which is what this battery is for. When
validating quantized weights, run the suite in §0 against a server on them
(~50 checks incl. constrained decoding and tool calls; count grows) *and*
report PPL. **GDN-model PPL A/Bs: `gdn.state_bf16` (default ON) carries
+0.21% PPL by design** - pin it equal in both arms (`--set
gdn.state_bf16=false` to compare against pre-#1778 numbers), and grep stderr
for the resolver silently keeping FP32 on unsupported shapes.

**Quant-file caveat (2026-06-06):** the local Qwen3-4B (unsloth) and Llama-3.2-3B (bartowski) GGUFs are re-downloads — *different quant files* than the originals. Greedy behavior on logit-tie prompts differs (the unsloth 4B degenerates on synthetic list prompts even unchunked — that's model-intrinsic, NOT an engine bug). For byte-level A/B prefer NLL/perplexity comparison over exact-output equality (the ChunkedPrefill tests switched to NLL for this reason, PR #553). Qwen3.6-35B is non-deterministic even at temp=0 — greedy-token A/B is INVALID there; use perplexity or `degen_suite.py --skip-deterministic`.

## Red flags — STOP and re-run

- Judging by short prompt alone → long-decode bugs pass `--max-tokens 30`. Use ≥64.
- Skipping the stderr grep → silent graph-capture fallback produces wrong output at 20% normal speed.
- Testing only one model after a shared-code change → MoE / GDN / dense paths diverge; pick 2+ probes.
- Declaring success on a single seed → for borderline cases re-run with seeds 42, 1, 7.
- Trusting "looks fine" on multi-turn without a turn-2 probe → KV/GDN state bugs only show up turn 2+.

## When the battery fails

Run `make test-e2e` first — it exercises `Gemma4GraphsTest.LongDecodeStaysCoherent`, `PrimaryModelTest.MultiTurnConversation`, `GDNModelTest.MultiTurnGDNState` which together cover all three state-management classes. Narrow to the failing model + class **before** editing.

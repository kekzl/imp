---
name: check-degeneration
description: Use when verifying that a model in the imp inference engine produces coherent output without repetition loops, token-stuck states, or state corruption across turns or streams. Triggers on "degenerates", "check degeneration", "repetition loop", "own own own", "stuck token", "empty content", "multi-turn regression", "does it still work", "NIAH", and after enabling CUDA graphs / changing forward pass / MoE routing / KV cache or KV dtype / GDN state or scan / sparse attention / PDL / speculation (MTP, n-gram) / ragged prefill / batched-decode kernels (smallm, producer quantize) / FA2 softmax.
---

# Degeneration Check - imp

Run the battery after touching the forward pass, graph capture, MoE routing, KV cache, GDN state, and every default-ON batched or fused path: ragged prefill (`runtime.prefill_batch`, #1780), batched GDN decode (#1750), chunk-parallel GDN prefill (`gdn.chunkpar_scan`, #1847-#1852), smallm GEMM (`gemm.nvfp4_smallm`, `_pair`, #1766/#1788), producer-fused quantize (#1771/#1773), BF16 GDN state (`gdn.state_bf16`, #1778), PDL device half (#1833), FA2 softmax/2-CTA (#1843/#1844), sparse decode attention (`attention.sparse_topk_tokens`), MTP auto/adaptive (#1809/#1801).

## Failure modes

| Pattern | Looks like | Root-cause class |
|---|---|---|
| Repetition loop | ` own own own`, ` the the the` | router precision drift, graph stale pointers, sampling NaN |
| Short OK, long fails | 15 tokens fine, then loop | KV-block boundary in graph replay; D2H memcpy in captured region |
| ~3-token abort | eos/stop at step 0-3 | `forward_decode_async` != `forward_logits`; sampler divergence |
| Multi-turn garble | turn 1 OK, turn 2 `The I...` | KV not reset; GDN state leaked; warmup CUDA error |
| Stuck single token | ` a a a a` | NaN/Inf logits; banned mask value; argmax on zeroed buffer |
| Structurally valid garbage | wrong language mid-stream | weight upload / dequant layout |
| Token-0 garbage `!!!` | from the first token | silent VRAM-alloc failure in a decode fallback (#934/#935) |
| Empty `content`, exit 0 | API "empty response", CLI fine | (a) thinking-state != rendered prompt tail (#937, server-api); (b) spec verify argmax emitted a chat delimiter mid-think (banned mask missing, fixed #1796); (c) shared think budget at `max_tokens < ~400` on Qwen3.8 (not a bug) |
| Cross-sequence contamination | single-stream clean, garbled only under a burst | ragged prefill row/offset, batched GDN slot mixing, shared act-quant scratch; needs the concurrency probe |
| Greedy output differs between graph replays | `DegenerationTest.GreedyDeterminism` red | PDL registration without `griddepcontrol.wait` (registered = waits, #1833) |
| Digits inside the retrieved needle corrupted | NIAH `ZEBRA-1550-25` for `ZEBRA-155000-25` | sparse budget too small (floor 8192 on Qwen3.8: 8/10; 4096 = 5/10), an approximation not a defect |

## Pass criteria

1. No token repeats >4x in a row, no 3-gram repeats >3x.
2. >=10 generated tokens before any stop (unless single-word factual).
3. stderr clean (grep below).
4. Decode within 30% of the model's row in `tests/perf_baseline.json` (>30% drop = graphs fell back silently).
5. Turn 2+ grammatical and on-topic.

## The battery

### 0. Server-level suite (deepest)

```bash
# server, e.g.:
docker run --rm --gpus all -p 8080:8080 -v $HOME/models:/models imp:test \
  imp-server --host 0.0.0.0 --model /models/<MODEL>
python3 tools/analysis/degen_suite.py --url http://localhost:8080            # exit 0 clean / 1 fail / 2 unreachable
python3 tools/analysis/degen_suite.py --skip-deterministic                   # Qwen3.6-35B (non-deterministic at temp=0)
python3 tools/analysis/degen_suite.py --only think-leak,adherence --quick --json /tmp/degen.json
python3 tools/analysis/degen_suite.py --corpus                               # ~250-prompt adversarial battery
```

Categories: repetition, think-leak, special-tokens, adherence, long-context, kv-growth, multi-turn, stream, constrained (json_object/json_schema under greedy, temp=1.3, min_p; forced tool_choice), anthropic-thinking (default arm asserts NO thinking block since #1560/#1743). ~50 checks and growing. `make test-server` (`scripts/test_server.sh`) runs it (#1573).

- A single `think-leak: truncated think spills` FAIL on Qwen3.8 flaked once (spec-fidelity non-determinism); reproduce 3x before debugging.
- Long sessions: `python3 tools/analysis/multiturn_deep.py --url http://localhost:8080 --model <id> --filler 60 --max-tokens 600` (~74 turns; reports `finish_reason` and reasoning length). Qwen3.8-27B fails at `--max-tokens 260` and is clean at 600, in vLLM too.
- Concurrency probe (the suite is single-stream): `python3 tools/analysis/conc_client.py <port> 32 4` or `tools/analysis/load_test.py --levels 1,8,32`, read outputs, then byte A/B 32-concurrent vs one-at-a-time on the same server (deterministic on, prefix cache off). #1780 gate: 27/32 identical vs 24/32 control.
- Speculation alive? `/metrics` `imp_spec_drafted_total` after an essay prompt: 1 = dead. `[mtp-econ]` log lines show the economics. Chunk-greedy != eager-greedy: "List 1 to 10" derails deterministically with ngram+MTP; always pair `speculative.mtp_k` with `speculative.ngram=false`.
- Retrieval: `make test-niah` / `tools/analysis/niah_check.py`: `--max-gen-tokens 768` (the 384 cap reads think-budget exhaustion as a miss), `speculative.ngram=false` in both arms (n-gram drafts the answer from the needle and masks broken selection), 32k prompts through `imp-cli --prompt-file`.

### 1. GTest battery

```bash
docker run --rm --gpus all -v $HOME/models:/models \
  -e IMP_TEST_MODEL=/models/<MODEL>.gguf \
  imp:test imp-tests --gtest_filter="DegenerationTest.*"
# after make dev: build-dev/imp-tests inside imp:toolchain
```

`tests/test_degeneration.cpp` (`ShortPromptNoRepetition`, `SecondRequestNotCorrupt`, `LongGenerationStability`, `NoLeakedSpecialTokens`, `GreedyDeterminism`; default model `Qwen3-8B-Q8_0.gguf`). Equivalence gates: `RaggedPrefillTest.*` (`tests/test_prefill_ragged.cu`, byte-equal vs serial), `GdnBatchedScanTest.*` (`tests/test_gdn_batched.cu`, bit-identical 8/32 sequences), `BatchedSmallM.*` (`tests/test_nvfp4_batched_smallm_equiv.cu`), `PagedOracle.HD128_Sweep` / `HD256_Sweep` (`tests/test_attention_paged_oracle.cu`, all 7 KV dtypes; HD256 is the shipped Qwen3.5/3.8 shape and catches a byte-order mutant HD128 passes), `SpecCaptureFidelityTest` (`make test-spec-fidelity`).

### 2. Smoke + gates

`make verify-fast` (`scripts/verify.sh`): degeneration detector on a real prompt, decode/prefill 8%, own-peak VRAM 10%, graphs-ON/OFF speedup >= 1.3x (2.64x at v0.34.0).

### 3. Cross-stack smoke

`docker run --rm --gpus all -v $HOME/models:/models imp:test bash /scripts/smoke_test.sh` (`scripts/smoke_test.sh`: unit lane, GPU subset, E2E subset, server smoke, CLI smoke; no vision stage).

### 4. Parity arms

- Graphs: `--set runtime.cuda_graphs=never` vs default, greedy, `--seed 42`, `--max-tokens 64`. Pass = identical; first ~16 tokens identical is the strong signal.
- Ragged prefill: `runtime.prefill_batch` ON vs OFF (serial fallback anyway for vision, constraints, logprobs, embeddings, rerank).
- Chunk-parallel scan: `gdn.chunkpar_scan` ON vs OFF; PPL on Qwen3.8-27B-NVFP4-vllm with `runtime.deterministic=true` (fused reference 4.6283); unit-test state diff vs fused ~1e-6; `tools/analysis/layer_ab_diff.py` on `diagnostics.dump_hidden_dir` dumps: the GDN blocks' ADDED divergence (rel@out - rel@in) must be ~0 (fused -> chunkpar median -0.0003).
- Sparse attention: the `sparse decode attention ACTIVE` line in the sparse arm only, budget in tokens equal to the configured value (double = old image, #1819).
- Byte A/Bs never diff CLI stdout (log lines interleave): diff the server JSON `content`. Qwen3.6-27B (proven on/off identical, #933) and Qwen3.8-27B are byte-deterministic at temp=0 with `runtime.deterministic=true` (implies `runtime.deterministic_gemm`); Qwen3.6-35B is not.

## stderr (mandatory)

```bash
grep -E "CUDA error|capture failed|falling back|warmup.*invalid|\bNaN\b|is NaN|is Inf" <log>
```

Any match = fail. Plain `Inf` matches `Inferred vocab_size=`.

## Known-good probes

| Model | Prompt | Expect |
|---|---|---|
| Qwen3-4B Q8_0 | "What is the capital of France?" | `Paris` |
| Qwen3.5-4B mxfp4 (GDN e2e model) | "Say hello." | non-empty, len >= 5 |
| Qwen3.8-27B-NVFP4-vllm (`kekzle/Qwen3.8-27B-NVFP4-vllm`; the Modelopt repo is gone) | "What is the capital of France?" | `Paris`; empty content at `--max-tokens <400` is the think budget |
| Gemma-4-26B-A4B Q4_K_M | "What is the capital of France?" | `Paris` after the `<|channel>thought` block |
| Llama-3.2-3B | "The capital of France is" | `Paris` |

Pick the first probe whose family matches the change; two probes after shared-code changes; seeds 42, 1, 7 for borderline cases; `--max-tokens >= 64`.

## Perplexity

- Corpus `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens). The 199-token `tools/analysis/ppl_corpus.txt` inverts verdicts (+42%/+57% vs +25%/+19% real).
- `--set runtime.deterministic=true` both arms (0.35% run-to-run otherwise); `--set speculative.mtp_k=0` (auto loads the head, +0.79 GiB); `gdn.state_bf16` pinned equal (+0.21% by design).
- Qwen3.6-35B PPL moves +-0.2..0.5% between fp32-equivalent kernels (routing flips): >1% = broken, below that no verdict. Judge on Qwen3.8-27B-NVFP4-vllm.
- PPL runs prefill; it cannot see decode-only paths or a degenerate low-PPL model. Pair it with the suite in section 0.
- Quant-file caveat: the local Qwen3-4B (unsloth) and Llama-3.2-3B (bartowski) GGUFs are re-downloads; greedy tie prompts differ; use NLL not byte equality (ChunkedPrefill tests do, #553).

## When the battery fails

`make test-e2e` first: `Gemma4GraphsTest.LongDecodeStaysCoherent`, `PrimaryModelTest.MultiTurnConversation`, `GDNModelTest.MultiTurnGDNState` cover the three state classes. Narrow to model + class before editing.

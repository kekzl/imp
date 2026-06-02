# Known issue: sporadic context carryover — model answers a *previous* question

**Status:** open, root cause not yet found. Reported behavior; not reproduced in a
short harness. Documented so the next session can target it instead of re-discovering.

## Symptom (user-reported)
Sporadically the input is ignored and the model answers a **previous** question/turn.
- **Scope: global** — observed across models (not one architecture).
- **Trigger: long multi-turn chats** — long conversation history.
- **Age: long-standing** — not introduced by a recent change.

## What was tested (could NOT reproduce — all correct, ~170 requests)
Clean `main` **and** the agent-readiness build (PR #492), Qwen3-8B-NVFP4:
- 48× sequential distinct factual Q&A → 0 carryover.
- 4-way concurrent distinct → 0.
- Streaming concurrent distinct → 0.
- Long-context request followed by short distinct → 0.
- 2-turn conversation, turn 2 distinct from turn 1 → 0 (always answered turn 2).

So it is NOT triggered by simple sequential/concurrent/streaming/2-turn flows. The
"long multi-turn" + "global" + "long-standing" signal points at **KV-cache pressure**.

## Leading hypothesis (ranked)
1. **KV-cache eviction in long conversations (most likely).** When history approaches
   KV capacity, blocks are evicted (StreamingLLM middle-eviction `evict_middle_blocks`
   + LRU, `src/memory/kv_cache_manager.h:124-141`, `:57-97`). If eviction drops the
   wrong blocks (e.g. the *recent* turn) or leaves the block-table / position mapping
   inconsistent, the model loses the latest input and attends to a retained earlier
   turn → "answers the previous question". Fits global + long-multi-turn + long-standing.
   → **Repro target:** drive a conversation past `min_kv_tokens`/capacity and watch
   `evict_*` + the post-eviction block table + RoPE positions.
2. **CUDA decode-graph block-table reuse.** The decode graph captures the per-`req->id`
   block-table device pointer (`src/runtime/engine_graph_decode.cpp:102,167,180`). If a
   captured graph is replayed for a later request/turn without recapture, it reads a
   stale block table. Less likely to be "long-multi-turn"-specific, but check the
   recapture trigger.
3. Ruled out so far: `req->id` collision (`next_request_id_++` is unique,
   `src/runtime/engine.cpp:539`); per-sequence free on completion happens engine-side
   (`engine.cpp:338,509`, `engine_scheduler.cpp:349,406,753`).

## Recommended next steps
1. **Reproduce under KV pressure:** a long multi-turn conversation (or low
   `--min-kv-tokens`) that forces eviction; alternate distinct topics per turn; detect
   when a turn answers an earlier turn.
2. **Add opt-in diagnostics** (gated behind a `diagnostics.*` flag): per forward log
   `req->id`, `kv_seq_id`, ctx_len, #blocks, evicted-block ids, and the first/last few
   prompt token ids — so the next real-world occurrence is diagnosable from logs.
3. Once reproduced, bisect: `--no-cuda-graphs` (tests hypothesis 2) and disabling
   middle-eviction (tests hypothesis 1).

## Possible relation to the Phi-4 long-context finding — RESOLVED (PR #503)
PR #494 surfaced a Phi-4-specific single-prefill degradation >~256 tokens ("ignores
recent, echoes early context"). That is single-request (no eviction) and Phi-specific,
whereas this issue is global + multi-turn + eviction-suspected.

**Root cause found and fixed (PR #503, commit `e75d51b9`).** The Phi-4 finding was the
**interleaved-vs-NeoX RoPE** bug: SafeTensors LLAMA/MISTRAL/MIXTRAL/LLAMA4 models
(Phi-3/Phi-4 map to `ModelArch::LLAMA`) were using interleaved RoPE while HF-native Q/K
require NeoX/rotate-half, scrambling per-position encoding → prompt-blind / position-
agnostic output. `load_safetensors` now forces `cfg.rope_neox=true` for those families.

**Re-verified on RTX 5090 (sm_120a) 2026-06-03**, with an `imp:test` image rebuilt after
the #503 commit. Single-prefill prompts well past the 256-token boundary (337 / 345 / 417
tokens, greedy temp=0, no eviction) now answer from the **recent** end of the prompt, not
early context:
- 417-tok Q/A, needle at end ("charter placed by the miller's son") → "the miller's son". ✓
- 337-tok Q/A, needle at end ("hidden in the attic of the Hartwell mill") → "In the attic of
  the miller's house". ✓
- 345-tok passage, recent mayor topic → answers about the mayor/governance (recent), not the
  1631 founders / Josiah-Hartwell early context.
- Discriminator: with a raw (non-chat-templated) format the extraction is weak, but that weakness
  is **position-invariant and reproduces on the healthy Qwen3-8B-NVFP4 path too** (needle-at-start
  and needle-at-end give the *same* wrong answer) — i.e. a prompt-format/reasoning artifact, NOT a
  Phi-specific positional-RoPE bug. The "echoes early context" signature is gone.

So this carryover issue and the Phi-4 finding turned out **distinct**; the Phi-4 half is **closed**.

## Update — CUDA-graph-reset hypothesis investigated (user's lead)
User strongly suspects a **missing graph reset**. Traced every decode-graph reset path:
- `async_graph_runner_` (conditional full-loop graph, `engine.h:269`): `setup()` calls
  `cleanup()` + fresh `cudaMalloc` per request (`cuda_graph.cu`), and `step_async_graph_resume`
  (`engine_scheduler.cpp:84-137`) binds pending tokens to `async_graph_req_` and clears on
  completion. Reset logic present.
- `decode_graph_pool_`: **deliberately preserved across requests** (`engine.cpp invalidate_graphs`
  comment) — recapture only when bucketed `max_blocks_per_seq` changes (`engine_scheduler.cpp:1066-1073`).
  Correctness relies entirely on the per-step batch upload (token/position/**block-table**) landing
  in the fixed pool buffers before each replay. **This is the architecturally most suspicious spot**:
  two same-bucket requests reuse the captured graph, and a stale/late block-table upload (e.g. under
  a batch-size transition or stream-ordering race) would make the replayed graph read the previous
  request's KV → "answers previous question".
- `prefill_graph_runner_`: recapture on `chunk_len`/`block_count` change (`engine_scheduler.cpp:570-576`).
- `invalidate_graphs()` is called only on the worker **exception** path, NOT on normal completion
  (`batching_engine.cpp:123,140`) — but it preserves `decode_graph_pool_` anyway.

**Reproduction status:** NOT reproduced across sequential, 4-/12-way concurrent, streaming,
long→short, multi-turn, and forced-StreamingLLM-eviction (~250 requests, all answered their own
question). Note: 12-way concurrency surfaced sporadic **empty** responses for some requests — a
*separate* phenomenon (not wrong-answer carryover), worth a follow-up.

**Decisive next experiment (run in the actual failing long-multi-turn scenario):**
`--no-cuda-graphs`. If carryover disappears → confirmed graph issue → fix target is the
`decode_graph_pool_` cross-request reuse / per-step block-table upload ordering. If it persists →
graphs are exonerated. A defensive fix (force decode-graph recapture or unconditional block-table
re-upload per new request) is available but should be gated on this confirmation + a perf check
(recapture ≈ 5–100 ms/request).

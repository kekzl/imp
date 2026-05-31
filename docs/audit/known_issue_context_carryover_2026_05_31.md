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

## Possible relation to the Phi-4 long-context finding
PR #494 surfaced a Phi-4-specific single-prefill degradation >~256 tokens ("ignores
recent, echoes early context"). That is single-request (no eviction) and Phi-specific,
whereas this issue is global + multi-turn + eviction-suspected — likely **distinct**,
but both are "recent input ignored in favor of earlier context" and worth checking for
a shared position/attention root cause.

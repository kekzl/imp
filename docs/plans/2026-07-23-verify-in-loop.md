# Verify-in-loop: conditional-graph token-recycling verify (#1055)

> **REMOVED 2026-07-25 — the feature is gone from the tree.** After the sweep
> below found no winning prompt class, `speculative.recycle_loop`,
> `recycle_loop_min_emit`, `TrVerifyLoopRunner`, the device adjacency table and
> the engine wiring were deleted (~1.5k LOC; see CHANGELOG "Removed"). The eager
> `speculative.token_recycling` drafter stays. This document is kept as the
> design record and the measurement history — the reuse map and the build
> lessons below (ctx-tier baking, miss-exit backoff, two-stage top-M harvest)
> are the parts worth re-reading if anyone revisits conditional-graph verify.
>
> **UPDATE 2026-07-25 — the win below is NOT reproducible; the flag is a
> measured loss on everything since tried.** A nine-class sweep (planning,
> math, repetitive, reasoning self-talk, enumeration, templated code, free
> prose, long explanation, chain-over-list; Qwen3-14B-NVFP4, 1024-tok greedy,
> interleaved, healthy host at 2835-2880 MHz SM / 13801 MHz mem / ~510 W)
> found **no** class where the loop beats the same configuration with the loop
> off. Isolated against eager `token_recycling` (same flags, loop off) it costs
> a consistent **5.6-8.3%**; the loop demonstrably runs in those arms
> ("verify loop built" appears, three graph instantiations per 1024 tokens).
> An era-image bisect on 2026-07-25 rules out a regression from #1059-#1066,
> so the numbers below were real — but the prompt class that produced them was
> never recorded and has not been re-found. `speculative.recycle_loop_min_emit`
> (#1060) bounds the damage to −0.3..−3.0% rather than creating a win.
> **Do not extend the flag's reach (MoE, constrained decoding) before a win is
> reproduced first.** Note also that the "byte-identical" claim below holds for
> those three prompts, not in general: a 2026-07-25 server-route check found the
> ungated loop diverging from loop-off on one prompt (greedy near-ties flipping,
> the #957 FP-summation-order class; output stays coherent).

**Status: PHASE 2 BUILT AND MEASURED 2026-07-23** — `TrVerifyLoopRunner`
(`src/runtime/tr_verify_loop.{h,cu}`) + engine wiring
(`src/runtime/engine_tr_loop.cpp`) behind `speculative.recycle_loop`
(default OFF). The riskiest unknown resolved positive: the capture-mode
verify forward captures cleanly inside a conditional WHILE body under
Relaxed mode (161 body edges PDL-converted). Measured (Qwen3-14B-NVFP4,
byte-identical greedy output on all three reasoning prompts):
CLI cold-start 162→281 / 163→321 / 162→224 tok/s (**+38–97%**); warm
server agent-loop 163.9→194.7 (**+19%**, single turns +40%). Build
lessons: (a) bake the ctx tier for the FULL remaining generation or every
burst re-instantiates (first probe: 91 rebuilds/256 tokens); (b) on a
miss-exit, launch the async decode burst DIRECTLY and suppress the eager
host-table TR fallback — leaving the backoff window to the eager
fallthrough cost 51 stray 10-ms verifies per 1024 tokens; (c) the
one-block-per-row top-M harvest kernel was 1.4 ms/iteration at 151k
vocab — now two-stage split (32 splits + merge, `rowwise_topm_reserve`
before capture). Phase-1 notes follow.

**Phase 1 (superseded):** device adjacency table
(`src/compute/token_recycle_device.{h,cu}`, semantics pinned equal to the
host table) and the complete loop-body tail kernel `tr_verify_step`
(accept + ring emission + EOS/stop/budget/ceiling exits + adjacency feed +
next-draft + chunk staging; `tr_verify_step_conditional` variant sets the
WHILE handle), 7 GPU tests green. **Remaining (phase 2, next session):**
(a) `TrVerifyLoopRunner` — own class, NOT surgery on the decode
ConditionalRunner: conditional node + body capture of
[capture-mode chunk forward → `greedy_argmax_all`(+top-M) →
`tr_verify_step_conditional`] + mapped ring/done plumbing (reuse map with
file:line refs lives in the 2026-07-23 session notes / #1055 comments);
(b) engine wiring: `speculative.recycle_loop` flag, KV burst
pre-allocation (prepare_graph_loop pattern), first-chunk host staging,
drain-resume with think tracking (step_async_graph_resume pattern), KV
reconcile at burst end; device table becomes the single drafter source
when the flag is on (probes read it via tiny D2H). Riskiest unknown:
the capture-mode verify forward inside a conditional WHILE body under
Relaxed capture mode — probe FIRST.

Goal: remove the ~1.3 ms/step host
scheduler/API tax between spec-verify steps (measured; effective verify
ratio ~1.5× vs the 1.33× intra-step wall) by running the whole
draft→verify→accept cycle as a conditional CUDA-graph WHILE loop, host
only draining tokens — the spec-verify analog of the async decode loop.

## Why token-recycling is the loop-able drafter

The suffix/n-gram drafters are host data structures. The TR adjacency
table is `vocab × slots` int32 + a streak byte — device-resident
(`src/compute/token_recycle_device.{h,cu}`, semantics pinned equal to the
host table by tests), and its inputs (accepted tokens, verify top-M
logits) are already on device. Drafting becomes a table walk in the loop
body; no host round-trip anywhere in the cycle.

## Loop body (fixed bucket-4 chunk, one iteration)

1. **`tr_verify_step_kernel`** (serial `<<<1,32>>>`, the post_decode_step
   analog): given the previous iteration's argmax buffer + staged draft —
   compare, emit `1+matched` accepted tokens into the mapped ring
   (token-count counter, not step counter), check EOS/stop per emitted
   token, advance device position/context by the emission, feed the
   adjacency (pairs along accepted path + top-M harvest rows), walk the
   table for the NEXT draft (depth 3, min_streak) and stage the next
   chunk (tokens/positions/row-ctx-lens/past_len/chunk_len — all device
   buffers the capture-mode forward already reads), and set the
   conditional handle to 0 on: draft miss, EOS/stop, step/token limit,
   or context reaching the baked tier ceiling.
2. **Verify forward**: the existing capture-mode chunk forward
   (`spec_verify_chunk` + `chunk_decode_attn` route, n_tokens = 4,
   device `d_past_len`/`d_chunk_len`/per-row ctx lens) captured INLINE
   into the body (the async-loop pattern), not nested as a child graph.
3. **`greedy_argmax_all`** over the 4 rows (+ top-M harvest) — already
   device-side and capture-compatible.

Iteration order note: the kernel runs accept-for-the-PREVIOUS-forward
first, then stages the next draft — so the body is
[accept+draft+stage → forward → argmax] with the first accept seeded as
a no-op (launch stages draft 0 host-side).

## KV correctness without device rollback

Blocks are pre-allocated for the whole burst (prepare_graph_loop
pattern). Rejected draft rows leave stale KV at positions ≥ the new p0 —
but the NEXT chunk re-writes exactly those positions (its rows cover
p0'..p0'+3 and KV-write precedes attention per layer), so no read ever
sees stale entries. The host reconciles the KV manager once at burst end
(rollback to the final context length).

## Reuse (mapped, from the async decode loop)

conditional node + WHILE handle, mapped ring + `__threadfence_system`
publish protocol, `try_finish_burst` done-flag (NOT cudaStreamQuery —
known WSL2 lie), rearm vs fresh capture, KV burst pre-allocation, host
drain with think tracking (`step_async_graph_resume` does
`track_think_state` per drained token — think stays host-side), split-K
non-pipeline fallback under capture (automatic), PDL edge conversion.

## v1 gates (all fall back to the eager verify path)

dense non-MLA/SWA/MoE/hybrid only (= decode-attn route), greedy, no
penalties/DRY/bias/logprobs/constraints, no think-budget-in-think, flag
`speculative.recycle_loop` (default off). New runner lives in its own
class (`TrVerifyLoopRunner`) — the decode ConditionalRunner's stability
is hard-won; no surgery there.

## Kill criteria

- Loop-body capture fails on the chunk forward under the body's capture
  mode → fall back eager, keep the flag off.
- Measured effective verify cost in-loop not ≤ ~1.25× a decode step, or
  TR reasoning A/B still ≤ spec-off warm → document, revert default.

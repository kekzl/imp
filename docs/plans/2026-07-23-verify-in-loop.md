# Verify-in-loop: conditional-graph token-recycling verify (#1055)

**Status:** phase 1 built 2026-07-23 — device adjacency table
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

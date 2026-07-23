# Token-Recycling multi-candidate speculative decode for reasoning/agentic text

**Status:** milestones 1–3 implemented 2026-07-23 (flag-gated
`speculative.token_recycling`, default OFF; route (a) multi-candidate via
per-candidate private KV blocks — no mask needed). Verify cost cut the same
day (#1055 first tranche: single-sweep batched-GEMV route for chunks ≤4,
2.05× → 1.53× a decode step; bench k=2 +34%). Measured end state: accept
1.44 cold / **1.77 warm** (server agent-loop, plateaus immediately) vs the
~1.5× break-even → **neutral in both cold and warm cases — the warm-table
hypothesis is refuted for this class**. Route (b) (true tree mask) is NOT
warranted by these accept numbers (kill criterion §Risks: <1.3 applies to
the deeper-hop accept, which collapses to ~0.05). TR stays default-off;
re-evaluate only if the remaining #1055 work pushes the verify ratio below
~1.3×. Full numbers: docs/audit/PERF_AUDIT_2026_07_23.md.

## Motivation (measured, not assumed)

Single-stream decode is comprehensively at the HBM/compute ceiling on every hot
path (dense NVFP4 GEMV 66–70% HBM; MoE 38%; #558 MoE-prefill closed — imp leads
vLLM). The only remaining way past the bandwidth wall at batch=1 is **more
accepted tokens per weight-sweep** — i.e. better speculation than the shipped
n-gram/prompt-lookup.

**imp's n-gram is dead on reasoning/agentic prose.** Measured 2026-07-22,
Qwen3-14B-NVFP4, real reasoning prompt (200 tok):

```
[spec-ngram] verify_steps=0 miss_steps=18 drafted=0 accepted=0 (0.0%)
```

Prompt-lookup only fires when the output repeats a suffix already in context
(code, structured text — Coder-30B hits 15.9 tok/verify). On fresh reasoning it
finds no match → 0 drafts → plain decode. This is exactly the GOAL's core
agentic use-case, and imp leaves the single durable lever unused there.

**Linear low-order drafting does NOT fix it** (measured, don't repeat): a
`speculative.min_match` sweep 1/2/3/6 on the same reasoning prompt left total
time flat (1739–1764 ms). Mechanism: imp swaps the fast async-graph decode loop
for the eager verify loop while spec is active; a linear top-1 draft's accept
(9–12%, 2.5–2.9 tok/verify at low min_match) can't amortize the eager overhead.
min_match=6 gives 7 tok/verify but fires ~never on fresh text. Precision and
recall both wash out for a single linear path.

**The win needs a TREE / multi-candidate verify** (Token Recycling, ACL 2025,
arXiv 2408.08696, ~2× lossless on general text; LogitSpec, arXiv 2507.01449,
2.61× training-free). Key economic enabler already in imp: the verify chunk
cost is **~flat in k** (config comment, `speculative.k=16`) up to the capture
bucket — so a wide candidate set is nearly free, and each extra accepted token
is ~pure gain.

## Design

### 1. Adjacency table (the drafter)
Global (engine-scoped, cross-request) `token -> top-M likely next tokens`,
built from the model's own top-K logits per decode step. `<2 MB` at M=8 over a
150k vocab. This is the Token-Recycling adjacency: for each emitted token `v`,
record the step's top-K logit ids as `v`'s successors (recency-ordered). Unlike
n-gram it fires on unigram context (the last token has almost always been seen)
→ it drafts on fresh reasoning text where suffix matching finds nothing.

Requires top-K logit extraction per step. imp already computes greedy argmax
(top-1); extend to a device-side top-K after the lm_head. In the verify loop
`greedy_argmax_all` already runs the lm_head per chunk position — extend it to
emit top-K there for free. In the plain-decode/miss path, the top-K costs a
small per-step device top-K; gate it behind the new flag.

### 2. Tree / multi-candidate draft
BFS the adjacency from the last token to build a small draft tree (depth 3–4,
branching 2–4) within the k=16 budget. Verify the whole tree in one chunk
forward and accept the best root-to-leaf path.

### 3. Verify — the hard part
imp's verify is **linear** today (one draft chunk, longest-prefix accept via
`greedy_argmax_all`; `engine_spec_ngram.cpp` `step_spec_verify_`). Two routes to
a tree/multi-candidate verify, in increasing risk:

- **(a) Multi-candidate linear, disjoint KV blocks** (lower risk). Emit N
  independent linear draft candidates. Write each candidate's tokens into its
  OWN KV block(s) so the existing `decode_attn_route` per-row `context_lens` +
  row block-tables (already present, `engine_spec_ngram.cpp:557`) isolate each
  candidate. Accept the candidate with the longest matching prefix. Cost: wastes
  KV blocks (1+/candidate, rolled back after) and re-computes shared prefixes,
  but needs NO new attention mask. Prove the win here first.
- **(b) True tree, token-level mask** (higher risk, higher win). Shared prefixes,
  block-diagonal causal mask over the <32-token chunk so each node attends only
  its ancestors. Needs a token-level mask in the FA2-tile / paged-decode verify
  kernel — the multi-week core. Only pursue if (a) proves the accept-rate lifts.

### 4. Integration point
`engine_spec_ngram.cpp` `step_spec_verify_`, ~line 354:
`if (draft.empty() && mtp_spec_decode_enabled()) draft = mtp_take_draft_(*req)`.
Add the adjacency/tree draft as a fallback **exactly parallel to the MTP
fallback** — the entire verify/accept/rollback/stats machinery consumes a plain
`std::vector<int32_t>` (linear) unchanged; the tree route needs the per-row
staging in the same function. New flag `speculative.token_recycling` (default
off) for clean A/B. The lossless argmax-accept is the correctness safety net:
even a wrong draft/mask can only cost speed, never token identity — as long as
the verify attention is correct per candidate.

## Incremental milestones (measure at each; STOP if a step doesn't lift accept)

1. Adjacency table (host-side, from emitted tokens) + linear fallback draft.
   Expect ≈ the measured net-neutral baseline — this validates the plumbing, not
   a win.
2. Top-K logit extraction → richer adjacency; re-measure linear accept.
3. **Multi-candidate linear verify via disjoint-KV `decode_attn_route`** (route
   a). First real win attempt — target >1.5 accepted tok/verify on reasoning.
4. True tree + token-level mask (route b) — only if 3 shows the accept lift.

## Measurement contract (critical — the default bench hides this)
- **NEVER measure on `--bench`**: its prompt is self-repetitive → n-gram already
  ~99.9% accept, so it measures the verify path, not the reasoning gap this
  targets. Use real reasoning/creative prompts via `--prompt` or the server.
- Metric: `[spec-ngram]` accept% + tok/verify, and spec-ON vs spec-OFF wall-clock
  decode tok/s on the SAME reasoning prompt (3+ trials, warm clocks).
- Guard the async-loop-vs-eager trade: the win must beat the eager-verify
  overhead the linear draft could not.

## Risks / kill criteria
- Route (a) accept stays <1.3 tok/verify on reasoning → tree unlikely to pay;
  reconsider (LogitSpec-style next-next retrieval is a cheaper alternative).
- Token-level mask breaks the lossless guarantee if attention leaks across
  candidates → gate every step on a greedy-identity check vs spec-OFF.
- Hybrid/GDN recurrent state: the existing hybrid verify snapshot/restore
  (`spec_state_scratch_`) must extend to the multi-candidate case, or gate
  token_recycling to non-recurrent requests first.

## Expected payoff
Token Recycling reports ~2× lossless on general text; realistically **+20–50%
decode on reasoning-heavy agentic workloads** where n-gram is currently 0%.
Code/structured agentic text already benefits from n-gram, so this is additive
on the reasoning half of the agentic mission, not a uniform speedup.

See memory `perf_ceiling_reality_and_spec_headroom_2026_07_22` for the full
campaign evidence.

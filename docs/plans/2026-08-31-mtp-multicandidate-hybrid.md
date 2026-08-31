# MTP multi-candidate speculation on the GDN hybrid (roadmap gap 5)

Status: DESIGN + Stage 1-2 in progress (2026-08-31). Stage 0 measurement
blocked on the GPU (nina-imp-server-1 holds the card).

## Why this is the open ceiling

The M=1 way past the bandwidth roofline is tokens per weight-sweep. The MTP
chain saturates near 2.5 accepted/verify (`docs/roadmap.md` item 3): one wrong
token kills the whole tail, and k=3 is uneconomic as a chain. External engines
target acceptance length >5 with multi-candidate trees. imp has every piece of
a tree EXCEPT their composition:

| piece | state | where |
|---|---|---|
| multi-candidate verify, private KV blocks (route a) | SHIPPED for token_recycling | `engine_spec_ngram.cpp` `mc` staging (~445-730) |
| trained drafter with real accept (74-78% @d1) | SHIPPED | MTP head, `engine_spec_mtp.cpp` |
| top-W extraction from the head | SHIPPED (probe-grade) | `mtp_forward.cu` top-w kernel, `kMtpMaxTopW=8` |
| tree-ceiling measurement | SHIPPED, never run to a recorded table | `diagnostics.mtp_tree_probe`, imp-cli table |
| multi-sequence scan with private state slots | SHIPPED for ragged prefill | `gdn_scan_fused_kernel` `seq_slots`/`seq_row_offsets` (#1780) |
| hybrid verify snapshot/restore/graph-replay | SHIPPED for the linear chain | `spec_state_scratch_`, `spec_snap_slab`, captured replay |

The blocker that kept them apart: `mc_route_ok` requires
`ssm_state_ == nullptr && !is_moe && !mtp_spec_decode_enabled()` - and every
checkpoint with an MTP head is a GDN hybrid. So the mc route runs on no model
that could feed it a good draft. Composing them is this plan.

## Expected value (to be replaced by Stage 0 numbers)

Chain accept per position p~0.75. Fixed depth-2 chain: E = p + p^2 + bonus.
A width-2 branch at position 1 replaces p with top-2 coverage p2 at the
weakest link. The whole bet is the size of (p2 - p) at depth 1; nobody has
measured it on this head, and the probe exists to do exactly that.

Cost side, measured pieces: verify chunk cost is ~flat in rows on the
decode-attn route; extra drafting is (W-1)x(D-1) head forwards (~0.4 ms at
W=2/D=2 on a ~12-15 ms verify cycle) plus a top-w kernel (probe-grade one is
713 us single-CTA - needs the production multi-CTA rewrite, target <30 us).

## Stage 0 - measure the ceiling FIRST (GPU, ~minutes, no build)

`imp-cli --set diagnostics.mtp_tree_probe=true` on Qwen3.8-27B-NVFP4, >=3 real
thinking prompts (NOT --bench). The table prints per-lookahead top-1..4
cumulative accept, teacher-forced at lookahead 0, self-chained >=1.

**Kill criterion: top-2@lookahead-1 minus top-1@lookahead-1 < 4 points**
=> the branch cannot buy enough accepted length to cover its ~3-5% drafting
overhead; stop at Stage 1 (which still ships the production top-w kernel and
the probe results) and close gap 5 as REFUTED-for-this-head.

Also record the top-1/top-2 logit margin distribution if cheap: an
uncertainty-gated width (branch only when the margin is small) is the
follow-up lever if flat branching is marginal.

## Stage 1 - MTP multi-chain drafting (CPU-buildable, flag-gated)

`speculative.mtp_tree_width` (int, default 1 = exactly today's behavior).
W>1: after the last feed pair drafts token a1 (top-1) and the top-w kernel
yields b1..b_{W-1} (ranks 2..W), draft W chains:

- chain 0 (primary): a1 -> chain on h_final as today.
- chain i: b_i -> continue top-1 on h_final from b_i's head forward.

MTP KV appends per chain are speculative; roll `ws->mtp_pos` back between
chains (same rollback the linear chain already does). Chains land in
`mtp_pending_chains_` (vector<vector>); `mtp_take_draft_` keeps returning
chain 0 so W>1 with the mc route unavailable degrades to today's linear path.

Production top-w kernel: multi-CTA partial top-w + tiny final merge, replacing
the 713 us single-CTA probe scan. The probe keeps its kernel (it is
measurement mode); serving uses the new one.

## Stage 2 - mc accepts MTP as a draft source

- `mc_route_ok` loses the `!mtp_spec_decode_enabled()` term; a new arm fills
  `mc` from `mtp_pending_chains_` when W>1 and the route is up.
- Winner hidden harvest: `mtp_post_verify_update_` consumes rows at the
  winner's offset (`mc_row0`), not rows [0, emitted). `view_hidden` grows an
  offset (or the call sites pass `mc_row0`).
- Economics guard prices the mc chunk rows (K stays the winner-depth proxy,
  already the mc convention).

## Stage 3 - hybrid mc verify (the hard core)

Present the W candidates to the RECURRENT path as W uniform sequences, and to
the ATTENTION path as chunk_pad row-sequences (the existing mc trick):

- **State slots:** W-1 scratch slabs seeded from the committed state by D2D
  copy (~50-100 us each at 27B's 79.5 MiB); the PRIMARY chain binds the real
  slot in place, exactly like today's linear chunk. `seq_slots` +
  `h_state_seq_stride` already rebase per-sequence state in the scan kernel.
- **Scan geometry:** uniform `seq * n_tokens` rebase (n_tokens =
  1 + mc_depth), gridDim.y = W. `d_real_n` commits each sequence's state at
  its real row - the per-block computation is already per-sequence under the
  uniform rebase; the "nullptr when ragged" contract stays (mc is uniform,
  not ragged).
- **Conv state:** same W-sequence treatment on the conv window update
  (`executor_ssm_gdn.cu` conv sites; the batched path from #1780 is the
  template).
- **Snapshot:** every candidate's row 0 is t0 from the same committed state,
  so the state-after-row-0 is identical across candidates - any single writer
  serves the zero-accept adoption path unchanged.
- **Attention:** `chunk_decode_attn` per-row tables become legal on hybrid
  when the scan runs the mc geometry. New InferenceState fields carry the mc
  geometry (groups, rows/group, slot table) so run_ssm and run_attention stop
  sharing one n_sequences meaning.
- **Accept:** full accept of the PRIMARY chain keeps its in-place state (the
  dominant case, no extra work - this is why the primary binds the real
  slot). Any other winner or a partial accept: restore committed + captured
  graph replay of the winner's accepted rows (existing machinery; the replay
  restage uses the winner's tokens instead of the chunk head).
- **VRAM:** (W-1) x per_seq_bytes scratch, priced in the planner next to the
  existing spec slabs (27B: +79.5 MiB per extra candidate).

## Stage 4 - capture buckets for the mc geometry

Eager-only mc measured 25.1 vs 17.8 ms on the linear path - the launch-pacing
penalty alone could eat the accept gain, so capture support is REQUIRED for
the win, not polish. W is config-fixed per process; bucket by depth as today.
gridDim.y = W bakes fine; per-row ctx lens and d_real_n are already device
data.

## Measurement gates (each stage)

- accept/verify and emitted/verify vs linear adaptive-k, same prompts,
  alternating arms, `tools/analysis/mtp_adaptive_ab.sh` extended with a width
  axis.
- tok/s on think traffic; degen_suite 50/0; greedy identity vs spec-off
  (lossless by construction - any deviation is a candidate-isolation bug).
- `IMP_SPEC_TRACE` already prints mc groups (`spec_trace_emit_verify` takes
  the mc pointer).

## Kill criteria

- Stage 0 fails the 4-point bar => stop, record, close gap 5 as refuted for
  this head.
- Stage 3 greedy-identity failure that survives a day of isolation => the
  conv/scan group semantics are wrong in a way the tests cannot see; stop and
  record rather than shipping a lossy speculator.
- E2E: W=2 must beat linear adaptive-k by >=3% tok/s on think traffic to earn
  default-on consideration; below that it ships default-off with the numbers.

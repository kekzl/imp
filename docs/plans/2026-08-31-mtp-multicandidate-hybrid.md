# MTP multi-candidate speculation on the GDN hybrid (roadmap gap 5)

Status (2026-08-31): Stage 1-3 implemented, flag-gated default-off
(`speculative.mtp_tree_width`, default 1). Stage 0 passed its kill bar
(+6..+10 points top-2 over top-1 at depth 1), the E2E gate did not
(+5-8% emitted/verify for +11-20% verify time): see the measurement
section at the end, including the next levers.

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

Implemented as designed below, with these concrete choices:

| piece | implementation |
|---|---|
| state slots | `SSMState::init(n_reserved = W-1)`: slots past `max_batch_size`, `reserved_slot(i)`; priced in `compute_vram_budget` and the auto-batch resolver |
| geometry | `InferenceState::ssm_seq_tokens` (+ `ssm_grouped_chunk()`), `ssm_seq_slots` = spec-stage slot table, `d_chunk_len` = per-GROUP real rows |
| conv | per-candidate loop in `run_gdn` (the #1780 ragged loop as template), snapshot from group 0 only |
| scan | `gdn_scan_fused_{f32,bf16}_batched` with `n_tokens = 1 + depth`, `d_real_n`, and new `h_snap/d_snap_n` (kernel writes it from `blockIdx.y == 0`) |
| pads | rows past the last group: conv/scan rows zeroed in `run_gdn` |
| accept | winner 0 full: in place; winner c full: one slab copy from slot c; zero accept: row-0 snapshot; partial: restore + swap winner's slot onto the live one + replay through the SAME captured graph (slot table and real-row count are device data), BEFORE the KV rollback (the replay's KV writes alias the private blocks) |
| capture key | `spec_graphs_` gains the grouped row count (0 linear) |
| gate | `spec_mc_hybrid_ok_`: reserved slots present, no `gdn.fp32_scan`/`ref_kernel`, GDN layers only (no Mamba2 grouped path) |

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

## Measurements (2026-08-31, Qwen3.8-27B-NVFP4, greedy, dev build)

### Stage 0 - tree ceiling (`diagnostics.mtp_tree_probe`, `miss_burst=0`, 900 tokens/prompt)

The probe is measurement-only since this branch: with the verify consuming
the chain there was no eager step to score it on (262 verifies, 0 eager
steps, empty table on the first attempt), and with miss bursts on, 8 of every
9 tokens bypassed the scorer (n=63 of 700).

| prompt | depth-1 top-1 | top-2 | top-3 | depth-2 top-1 | top-2 | top-3 | E[accept] top-1 | top-2 | top-3 |
|---|---|---|---|---|---|---|---|---|---|
| train meet (reasoning) | 90.5 | 98.4 | 98.4 | 80.6 | 88.7 | 90.3 | 1.634 | 1.857 | 1.873 |
| merge intervals (code) | 85.7 | 95.2 | 98.4 | 64.5 | 82.3 | 87.1 | 1.410 | 1.736 | 1.841 |
| spec-vs-batch (prose) | 92.1 | 98.4 | 98.4 | 87.1 | 90.3 | 91.9 | 1.722 | 1.873 | 1.889 |

n = 63 per depth per prompt (n_predict 900, the probe scores one chain per
eager step). top-2 minus top-1 at depth 1 = +7.9 / +9.5 / +6.3 points: above
the 4-point kill bar on every prompt. Expected accepted length top-2 vs
linear: +0.22 / +0.33 / +0.15 draft tokens per verify.

### Stage 3 - W=2 verify on the hybrid, same state (`diagnostics.spec_trace`, 44 verifies)

Both candidates' row 0 (t0) produce the same argmax in all 44 verifies
(candidate isolation holds). Scored on the traced argmax rows: the winner
rule emits 2.705 tokens/verify against 2.409 for chain 0 alone (+12.3%);
candidate 1 wins 7 of 44. Output coherent (correct answer to the train
problem).

### Cost, 400 tokens (dev build, verify_smallm default off)

| arm | tok/s | accept | emitted/verify | ms/verify |
|---|---|---|---|---|
| spec off | 88.4 | - | - | - |
| W=1, k=2 fixed | 135.4 | 85.7% | 2.71 | 19.87 |
| W=2, k=2 fixed | 91.2 | 84.8% | 2.70 | 29.46 |

The mc chunk is 6 real rows padded to the 9-row bucket; on native ST-NVFP4 a
verify chunk with M <= 4 takes `gemm_nvfp4_batched` (one weight sweep) and
anything larger the CUTLASS prefill tile (~51% of sweep bandwidth at tiny M,
per executor_gemm_dispatch.cu), so the width buys +12% accepted length for
+48% verify time. `speculative.verify_smallm=true` routes both through the
mxf4nvf4 small-M v2 kernel (M <= 32); see the next table.

### Cost with `speculative.verify_smallm=true`, 600 tokens, same prompt (dev build)

| arm | rows (bucket) | tok/s | accept | emitted/verify | ms/verify |
|---|---|---|---|---|---|
| W=1, k=2 fixed | 3 (3) | 130.3 / 135.0 | 77.4% / 84.9% | 2.55 / 2.70 | 19.54 / 19.95 |
| W=2, k=2 fixed | 6 (9) | 127.9 | 91.3% | 2.83 | 22.07 |
| W=2, k=2 fixed | 6 (6, new bucket) | 126.4 | 87.4% | 2.75 | 21.72 |
| W=1, k=3 fixed | 4 (4) | 139.0 / 138.4 | 74.0% / 73.4% | 3.22 / 3.20 | 23.14 / 23.11 |
| W=2, k=3 fixed | 8 (8, new bucket) | 121.1 | 78.8% | 3.37 | 27.76 |
| W=2, k=1 fixed | 4 (4) | 122.9 | 97.7% | 1.98 | 16.06 |

Two entries per linear arm = two runs (greedy trajectories diverge, the
forward is not bit-deterministic); the spread is the noise floor for the
single-prompt numbers.

**Verdict (E2E gate, "W=2 must beat linear by >= 3% tok/s"): not met.** The
width buys +5-8% emitted/verify at k=2 and +4.7% at k=3, for +11% / +20%
verify time. Ships default-off (`mtp_tree_width=1`).

### Think traffic, server, `tools/analysis/mtp_adaptive_ab.sh` (imp:test image)

`THINK=1 ROUNDS=2 CLASSES=poor ARMS="k2ad w2ad" EXTRA="--set speculative.verify_smallm=true"`:
3 reasoning prompts, max_tokens 1024, greedy, fresh process per arm, arms
alternated across rounds, ngram off, prefix cache off.

| arm | round | tokens | ms | tok/s | drafted | accepted | verifies |
|---|---|---|---|---|---|---|---|
| k2ad (linear, adaptive k=2) | 1 | 3072 | 27325 | 112.42 | 2367 | 1623 | 1449 |
| w2ad (W=2, adaptive k=2) | 1 | 2518 | 23898 | 105.36 | 1948 | 1306 | 1212 |
| w2ad | 2 | 2862 | 26831 | 106.67 | 2207 | 1500 | 1360 |
| k2ad | 2 | 3072 | 26820 | 114.54 | 2323 | 1667 | 1403 |

W=2: -6.4% / -6.9% tok/s against linear adaptive-k on think traffic. Gate
not met; default stays 1.

Where the verify time goes (rows are the currency, not candidates):
- LM head: `gemv_nvfp4_kpar_batched_fp32` reads the 0.64 GB head once per
  MR=4 rows - 6 rows = 2 passes where 3 rows = 1 (+~0.45 ms), 8 vs 4 the same.
- Drafting: chains 1..W-1 cost (K-1) extra head forwards each, serial at
  M=1 (+~0.5 ms per forward: lm_head GEMV over 248k vocab).
- Layer GEMMs: 6/8 rows on the small-M v2 kernel vs 3/4 - not separated
  from the above without an nsys node trace.

Next levers, in order of expected yield (none implemented):
1. Batch the chain drafting: chains 0..W-1 as one M=W head forward per
   depth (same weight sweep) instead of W serial M=1 forwards - removes the
   (W-1)(K-1) x 0.5 ms drafting term.
2. Tree-shaped chunk sharing row 0 (t0 forwarded once): W(1+k) -> 1+Wk
   rows; needs candidate 1's row to read t0's KV from candidate 0's private
   block and the scan to start group 1 from the row-0 state (an in-kernel
   dependency the batched scan does not have today).
3. Uncertainty-gated width (branch only when the head's top-1/top-2 logit
   margin is small): the probe's `gap=` column in `diagnostics.spec_trace`
   already prints the margin per row; most verifies would then run the
   3-row linear chunk.

## Lever 3 - margin-gated width (`speculative.mtp_tree_margin`, 2026-08-31)

Branch only when the head's top-1/top-2 logit margin (serving top-W kernel
now returns the values) is below the threshold; otherwise the step verifies
the linear chunk. The alternate chains are still drafted every step.

CLI, 600 tokens, greedy, k=2 fixed, `verify_smallm=true`; r1 = train prompt,
r2 = merge-intervals prompt:

| arm | r1 tok/s | r1 emitted | r1 ms/verify | r1 branched | r2 tok/s | r2 emitted | r2 ms/verify | r2 branched |
|---|---|---|---|---|---|---|---|---|
| linear | 135.1 | 2.60 | 19.25 | - | 133.5 | 2.62 | 19.57 | - |
| W=2, margin 0 | 125.1 | 2.77 | 22.15 | 216/216 | 109.6 | 2.53 | 23.03 | 237/237 |
| W=2, margin 1 | 135.3 | 2.81 | 20.76 | 19/213 | 113.3 | 2.47 | 21.74 | 72/243 |
| W=2, margin 2 | 140.7 | 2.85 | 20.25 | 37/210 | 103.6 | 2.41 | 23.20 | 115/128 |
| W=2, margin 4 | 132.3 | 2.80 | 21.12 | 84/214 | 106.6 | 2.63 | 24.49 | 136/228 |

Think traffic, server harness, `ARMS="k2ad w2ad"` (w2ad = W=2 at the default
margin 2.0), 2 rounds alternated:

| arm | round | tokens | ms | tok/s | drafted | accepted | verifies |
|---|---|---|---|---|---|---|---|
| k2ad | 1 | 3072 | 27505 | 111.69 | 2343 | 1641 | 1428 |
| w2ad | 1 | 3072 | 27714 | 110.85 | 2315 | 1684 | 1385 |
| w2ad | 2 | 3072 | 28953 | 106.10 | 2362 | 1621 | 1448 |
| k2ad | 2 | 2986 | 26515 | 112.62 | 2279 | 1599 | 1387 |

W=2 with the gate: -0.8% / -5.8% against linear adaptive-k (was -6.4% /
-6.9% ungated). The gate removes most of the row cost but not the drafting
cost (the W-1 alternate chains are (K-1) serial head forwards per step,
paid whether or not the step branches), and on the code prompt the head
branches far more often (115 of 128 drafts at margin 2 vs 37 of 210 on the
reasoning prompt) without an accept gain. Gate not met; W stays 1.


# Batched MTP verify on the GDN hybrid (the concurrent-serving lever)

Status (2026-09-13): stages 1-4 DONE, shipped opt-in as `speculative.batch_verify`
(#1993, default off). The spare-slot price continues in
[2026-09-12-factored-verify-spare](2026-09-12-factored-verify-spare.md). Stage
table at the end carries the state of every stage.

Status (2026-11-04): the same step runs by default for the BOUND requests while <= 4 decode
(`speculative.mtp_batch_verify`, section "Bound verify at 2-4 rows"), with 2 drafts per request
(`speculative.mtp_batch_verify_k`, section "Two drafts per bound request"), both draft passes on the
NVFP4 draft head (section "NVFP4 draft head in the batched passes"), their tables in one H2D (section "One
table upload per head pass"), the pass replayed as a CUDA graph (section "The head pass as a CUDA graph");
the full `batch_verify` stays opt-in.

Status (2026-11-11): at 2 drafts the second spare slot per row is replaced by a factor row of draft 1
(`speculative.mtp_batch_verify_factor_row`, section "Draft 1 as a factor row" at the end); a concurrent
GDN-hybrid server keeps 2 drafts by default.

Status (2026-11-11, later): the factor rows are indexed by verify group and carry up to 2 drafts, a bound
request verifies 3 drafts by default (`speculative.mtp_batch_verify_k` 3, section "Up to 3 drafts per bound
request" at the end).

Status (2026-11-11, depth 4): the depth is per step, 4 drafts below 4 requests, 3 from 4 up
(`speculative.mtp_batch_verify_k` 4, `speculative.mtp_batch_verify_k_wide` 3, section "4 drafts per bound
request, 3 from 4 requests up").

Status (2026-11-11, think budget): a verify chunk ends at the think budget, the forced closer lands at the
same reasoning token for every `mtp_k` (last section).

Status (2026-11-11, coupled drafts): a sampled bound request's drafts are draws with the seeds of the verify
rows that check them (`speculative.mtp_draft_coupled`, default on, section "Coupled MTP drafts" at the end).

Status (2026-11-11, ladder): below 4 requests a step that accepted all of its drafts is followed by a deeper one,
4 -> 6 -> 7 drafts inside the 8-row GEMM tile (`speculative.mtp_batch_verify_k_streak`, default 7, section "Verify
depth ladder after a full accept" at the end).

## Why

Qwen3.8-27B-NVFP4-vllm at 32 streams: step 14.75-14.98 ms = 1977-2005 tok/s,
of which 12.06 ms is the DRAM floor (weights 13.7 GB + LM head 0.72 + GDN
state 4.83 r+w + KV 0.4 = 19.6 GB at 1628 GB/s, `docs/ROADMAP.md` 2026-09-08).
Every remaining lever inside that step is worth 1-3 %. The only way past the
floor is tokens per weight sweep: the MTP head accepts 83.5-84.5 % at depth 1
(`docs/ROADMAP.md`, "MTP acceptance gap"), so a step that forwards 2 rows per
stream emits ~1.84 tokens per stream instead of 1.

Today the MTP verify is batch-1: one MTP workspace (`engine_spec_mtp.cpp`,
`mtp_bound_req_`), the verify chunk a single request
(`step_spec_verify_`), and `speculative.batch_rr` (one request verifies per
step while the rest decode) is switched off on recurrent models
(`spec_gates.h`, `!recurrent`).

## Design

One forward per decode step with N groups of 1+K rows, K = 1 in v1:

| piece | mechanism | exists today |
|---|---|---|
| rows | group g = request g: row 0 = its last emitted token, row 1 = its MTP draft; positions p0_g, p0_g+1 | the mc hybrid chunk (`ssm_grouped_chunk`) has N groups x T rows on N slots |
| attention | decode-attn route: every row is a "sequence" with its own block table (the request's) and ctx len p0_g+1+i | `chunk_decode_attn` with row block tables (`d_spec_row_block_tables_`) |
| GDN state | group g reads slot L_g, writes the 2-row state to spare slot P_g and the 1-row state in place into L_g; accept -> the request's live slot becomes P_g, reject -> stays L_g. No state copies. | scan/conv take `seq_slots` + one snapshot slab for group 0 only. Stage 1 adds per-group destination and snapshot slots |
| conv window | the commit kernel runs twice after the prefill grid: window at 2 rows into P_g, then window at 1 row in place into L_g (in-place shift is per-thread ordered, stream order covers the grid's reads) | `ssm_conv1d_commit_kernel` reads and writes one slot; Stage 1 splits src/dst |
| accept | LM head + argmax over all 2N rows, one D2H, per-request accept/emit/rollback exactly as the linear verify | `greedy_argmax_all` (one ban list for all rows; Stage 2 adds per-row think-mask) |
| MTP draft | N MTP KV slots in one workspace; after the verify ONE ragged head pass over every emitted (token, hidden) pair (row r: slot_r, pos_r; append before scan gives causality), LM head on each request's last pair via the small-M NVFP4 LM head, argmax = next draft | single-slot workspace, `mtp_feed_batch` is one sequence; Stage 3 |
| VRAM | N spare slots = N x 79.5 MiB (27B, BF16 state). Priced by the planner as reserved slots (`plan_fitting_batch`); the verify runs only when every decoding request has a spare | `SSMState::n_reserved` + `reserved_slot(i)` |

Gates for the batched step (v1): hybrid with a loaded head, every decoding
request greedy, no penalties, no constraints, no logprobs, batch <= spare
slots. Anything else takes the plain batched decode step, unchanged.

Expected at 32 streams: 19.6 GB -> 19.6 + 2.5 (second state write) GB per
step for 1.84x tokens; M=64 rows leave the small-M GEMM path (M <= 32,
CUTLASS tile above: batch-41 step measured 14.8-20.8 ms vs 14.9 at 32).
Ceiling before measurement: +35..+45 % aggregate. At 8-16 streams every
row stays on the small-M kernel: +60..+80 %.

## Stages

| stage | content | gate | state |
|---|---|---|---|
| 1 | scan + conv kernels: `out_slots` (per-group commit destination) and `snap_slots` (per-group snapshot slot), conv commit with separate src/dst slots | `test_gdn_batched.cu`: N groups x 2 rows == N single runs; L_g holds the 1-row state, P_g the 2-row state; unowned slots untouched | DONE 2026-09-11: 2 tests, both red on the mutants (scan ignores out_slots; conv drops the snapshot commit) |
| 2 | engine: `step_spec_verify_batched_` (chunk build, forward, per-row masked argmax, accept/emit/rollback, slot swap), spare-slot pool, planner pricing, config `speculative.batch_verify`, graph replay keyed (rows, ctx tier) | e2e greedy 8 streams: coherent, acceptance stats logged | DONE 2026-09-11, n-gram drafts as the source (table below) |

Stage 2 measurement (dev build, Qwen3.8-27B-NVFP4-vllm, 8 streams x 300
greedy tokens, `--think-budget 0`, prompt "repeat this sentence 60 times",
`speculative.verify_smallm=true` on both arms, fresh server per arm, 3 rounds):

| arm | tok/s | verify |
|---|---|---|
| `batch_verify=false` | 618.1 / 628.7 / 627.3 | none (batch_rr is off on hybrids) |
| `batch_verify=true` | 876.8 / 920.2 / 922.8 (+42..+47 %) | 461 steps, 99.3 % accept, 15.38 tok/verify, 15.84 ms/verify |

Two prices found on the way: the eager chunk forward costs 28.6 ms/verify
against 15.8 replayed, and `verify_smallm=false` (the default) sends the
16-row GEMMs to the CUTLASS tile: 21.3 ms/verify. Both stay on the plain
decode arm's numbers untouched. With the server's think budget on (0.5 x
max_tokens = 150 reasoning tokens), the same run reads 560-578 vs 582-590:
the n-gram source drafts nothing inside the think block, and every verify
step then forwards 16 rows for 8-9 tokens. The MTP source (stage 3) is what
carries that half.
| 3 | MTP: N-slot KV (`mtp_select_slot`, per-request bindings parked in `mtp_binds_`), one ragged head pass over every emitted pair (`mtp_feed_rows_multislot`), CUTLASS LM head on the last row per request (`lm_head_rows_argmax`), K = 1 | `MtpFeedBatchTest.MultiSlotFeedMatchesPerSlotBatches` (KV rows and final_norm vs per-slot batches) | DONE 2026-09-11 (table below) |

Stage 3 measurement (dev build, Qwen3.8-27B-NVFP4-vllm, 8 streams x 300
greedy tokens, unique prompts (`conc_client.py` with `repetition_penalty`
1.0), `--think-budget 0`, `mtp_k=2 ngram=false verify_smallm=true` on both
arms, fresh server per arm, 3 waves):

| arm | tok/s | verify |
|---|---|---|
| `batch_verify=false` (head loaded, nothing drafts at batch > 1) | 586.7 / 605.7 / 610.7 | none |
| `batch_verify=true` | 703.8 / 758.3 / 766.8 (+20..+26 %) | 516 steps, 78.0 % accept, 13.9 tok/verify, 16.9 ms/verify: 15.0 forward + argmax, 1.84 head feed + LM head + sync (14.1 rows/call) |

At 32 streams the spares do not fit: the planner clamps `max_batch_size`
32 -> 18 (5088 MiB of state for 32 + 32 slots), 32 clients then queue behind
18 slots and the wall reads 1149 / 1159 / 1160 tok/s against 1928 / 1966 /
1980 with 32 plain slots. The lever at full load is VRAM headroom (the FP16
LM head, 2425 MiB, is decode-redundant), not this step.

Price found on the way: `diagnostics.mtp_prenorm_h` (default on) sent the
first build down the per-request fallback (per-pair M=1 feeds and chain
drafts with a full-vocab GEMV each): 37 ms/verify, 356 tok/s. The batched
feed now applies the target's final norm to the gathered rows itself.
| 4 | per-row rep / freq / presence penalties (the sampler's device history slots, the draft written behind the history, emitted tokens appended from the pinned argmax buffer), MTP KV slots on a 1 GiB budget, gates (engine.h at 700, alloc sites, lanes pin) | A/B with the server's default `repetition_penalty` 1.05 and at 15 / 18 streams | DONE 2026-09-11 (table below); `check-degeneration` and the perf baseline for spec-off still to run |

Stage 4 measurement (same setup as stage 3, `--think-budget 0`, 3 waves):

| streams (server batch) | client | `batch_verify=false` | `batch_verify=true` |
|---|---|---|---|
| 8 (8) | default `repetition_penalty` 1.05 | 592.1 / 625.7 / 619.0 | 670.5 / 713.8 / 704.7 (+13..+14 %); 73.2 % accept, 13.2 tok/verify, 17.2 ms |
| 15 (16, clamped to 15 on) | rep 1.0 | 990.0 / 1038.9 / 1057.0 | 1098.7 / 1209.2 / 1213.5 (+11..+16 %); 80.4 % accept, 25.7 tok/verify, 18.9 ms |
| 18 (18, clamped to 15 on) | rep 1.0 | 1146.0 / 1182.6 / 1192.6 | 842.3 / 859.4 (-27 %): 18 clients queue behind 15 slots |

The lever pays while the clients fit the clamped batch; past it the queue
costs more than the extra tokens per step return. The head at 15 streams:
25.7 tokens per 18.9 ms step.

## What the spare slots actually cost, and why the clamp is the wrong half to break (2026-09-12)

Qwen3.8-27B-NVFP4-vllm, `runtime.max_batch_size=32`, one image, startup logs:

| `speculative.batch_verify` | SSM/GDN state | KV pool | batch |
|---|---:|---:|---:|
| off | 2544 MiB (32 slots) | 2339 blocks, `max_model_len` 131 072 | 32 |
| on | 5088 MiB (32 slots) | 1429 blocks, `max_model_len` 33 680 | clamped to 18 |

The spare is an exact duplicate: one full recurrent slot per batch slot, 79.5 MiB each on this
model, held for the request's lifetime because accept swaps slots instead of copying. So enabling
the batched verify at 32 slots costs the batch (32 -> 18) AND three quarters of the usable context,
in the regime where the verify is already a measured loss (1966 -> 1159 tok/s above).

**Attempted and reverted**: making `clamp_max_batch_to_plan_` drop `speculative.batch_verify`
instead of the batch slots. It keeps `max_batch_size=32` and logs both options, but only half
works: `compute_vram_budget` runs before the shadow plan (`engine_kv_cache_init.cpp`), so the KV
pool stays sized against the spares (1429 blocks, not the 2339 the operator would get by setting
`batch_verify=false` themselves) and the log then overstates what it recovered. Doing it correctly
means deciding before the budget call, which means one captured `effective_free_vram()` snapshot
reused across two budget computations and the probe/plan block (lines 240-325) restructured to run
twice - in the #1103 order, in a function already at 682 of its 700-LOC ceiling. Its own change.

**The structural answer** is to stop duplicating the state at all. The spare holds the state after
two more tokens, and the gated delta rule's per-token update is a rank-1 term plus a diagonal
decay, so the difference is two (k, v, beta) triples per layer rather than a full state matrix:
O(d_k + d_v) against O(d_k * d_v) per head. That is the TreeWY / Bole direction
(arXiv 2608.20961, 2608.01651, which report 82-99x lower transient state memory) and it would put
the batched verify back on 32 streams with the full KV pool. The cost to price first is the accept
path: applying the factors writes the live state instead of swapping a pointer.

## "M=64 rows leave the small-M GEMM path" is not the price (2026-09-12)

Built and measured, then reverted: 33..64 rows as two 32-row `gemm_nvfp4_smallm_v2` launches
instead of the CUTLASS 128-row tile, gated to N <= 8704 (no flag shipped).

| measurement | result |
|---|---|
| isolated, L2-defeated ring, two halves at M=64 vs one tile at M=32 | q/o 13.94 -> 24.98 us, k/v 15.33 -> 29.52, down 35.39 -> 61.13, gate\|up pair 70.63 -> 121.69 |
| 24 streams, batched verify (48 rows), row split on N <= 8704 | 1076.8 / 1058.9 / 1041.9 vs 1049.0 / 1071.0 / 1060.1 tok/s: neutral, step +0.2 ms |
| same, row split on every shape (gate\|up N=17408, 136 CUTLASS CTAs) | 925.1 / 959.8 / 937.0: -11..-13 %, step +3.5..+5.2 ms |
| plain decode, 40 streams, mbs 64 (36 decoding rows) | 1156.1 / 1154.2 / 1157.1 vs 1161.1 / 1147.1 / 1151.7: neutral |

At 33..64 rows the CUTLASS tile reads each weight once at the rate two small-M sweeps reach
together; its row padding costs math, and these GEMMs are weight-bound. The verify's price at
24 streams is draft coverage and acceptance: `2026-09-12-factored-verify-spare.md`, correction.

## Bound verify at 2-4 rows (2026-11-04, `speculative.mtp_batch_verify`, default 4)

Qwen3.8-27B-NVFP4-vllm, default `imp-server`. Since #3029-#3031 up to 4 requests bind the head and
carry their hidden rows through batched decode, but they decoded one row per step while they
overlapped. At 2-4 rows a step is weight-bound, so doubling the rows is cheap:

| rows per step | plain decode (main, no head) | bound verify (2 rows per request) |
|---:|---:|---:|
| 2 / 4 | 11.34 ms, 2 tokens | 12.57-13.04 ms forward + argmax, 2.00-2.64 ms head feed + draft, 3.66 tokens |
| 4 / 8 | 11.72 ms, 4 tokens | 13.12-13.65 ms + 2.02-2.65 ms, 7.15-7.20 tokens |

| piece | rule |
|---|---|
| when | 2..`mtp_bound_verify_rows` (min of the key, `mtp_history_slots`, the batch) decoding requests, at least half of them bound and drafted (`mtp_bound_verify_fits`); an undrafted row forwards a pad |
| state | slot-swapping spares, one per row (4 x 79.5 MiB charged in the KV plan, 4812 -> 4531 blocks); the factored spare is not used: its row must be applied by whatever step runs next, and a plain step does not |
| drafts | the request's pending chain (first token), else its carried rows flushed with a 1-token draft; after the step one ragged head pass feeds every bound row and drafts the next token over the FP8 LM head |
| plain chain | a pipelined plain chain yields to the verify (`pipeline_yields_`, backoff 1 -> 64 steps per decline) |
| bursts | a burst of <= 4 prompts binds (each prefills serially), a wider one stays ragged and unbound |

Measured (BENCHMARKS.md "MTP verify while chats decode together", 3 pairs): 2 overlapping chats
94.21 -> 117.06 tok/s, 4 chats 83.72 -> 117.45, turns drafting 13/72 -> 72/72.

Two defects found on the way, both in the opt-in `batch_verify` too: the step capped its rows at
`runtime.max_batch_size`, 0 = auto on a default server, so it declined every batch; and
`lm_head_rows_argmax` needed the W4A4 batched-decode head, which an FP8-served head never builds.

Next: deeper drafts (3 rows per request, a second head pass per step), section below.

## Two drafts per bound request (2026-11-04, `speculative.mtp_batch_verify_k`, default 2)

Each bound request forwards its last token plus a 2-token MTP chain (3 rows). Per-step price on the
same image, `spec-batch price` log lines, cumulative per arm (`overlap_and_step_price.txt` in
docs/audit/qwen38_mtp_batch_verify_k2_2026_11_04/):

| requests | k 1: rows, forward + argmax, head, tokens/step | k 2: rows, forward + argmax, head, tokens/step |
|---:|---|---|
| 2 | 4, 12.14-12.15 ms, 1.96-1.99 ms, 3.39-3.48 | 6, 12.75-12.76 ms, 3.73-3.78 ms, 4.59-4.74 |
| 3 | 6, 12.41-12.43 ms, 1.99-2.02 ms, 5.30-5.42 | 9, 13.12-13.15 ms, 3.71-3.74 ms, 6.66-6.98 |
| 4 | 8, 12.86-12.89 ms, 2.02-2.03 ms, 6.87-7.16 | 12, 13.87-13.93 ms, 3.79-3.80 ms, 9.24-9.42 |

| piece | rule |
|---|---|
| state | three recurrent states per request: row 1 in place in the live slot, rows 1-2 in a second spare (a commit-only scan pass before the main one, a third conv commit, `InferenceState::ssm_snap_rows`), all 3 rows in the spare; k accepted drafts swap the slot holding 1 + k rows in (`bound_verify_state_slot`). 2 spares per row: 8 x 79.5 MiB charged, KV plan 12305 -> 11380 blocks. Since 2026-11-11 the default replaces the second spare by a factor row (section "Draft 1 as a factor row"); this row describes `speculative.mtp_batch_verify_factor_row=false` |
| drafts | the batched head pass drafts token 1 per request, a second ragged pass (draft 1 on device, the head's own final-norm row) drafts token 2 at the next head position, overwritten by the next feed; a request whose own `mtp_k` is 1 keeps one draft |
| kernels | no scan kernel change: the 2-row state is a second launch of `gdn_scan_fused_*_batched` with `out_slots` = the second spares, its y rows equal to the main pass's (`GdnVerifyTwoDraftsTest`) |

Measured (3 alternating pairs, main daa62a92 vs k 2 vs k 1 on this branch): 2 overlapping chats
119.44 -> 132.79 tok/s per turn, 4 chats 121.59 -> 134.17; k 1 on this branch 119.44 / 120.81 (main
again); c2 241.3-255.0 -> 279.2-291.2, c4 457.6-469.6 -> 485.8-507.8, c1 and c32 unchanged; lone chat
6/6 byte-identical to main on fresh servers.

Next: the head pass, 3.73-3.80 ms of a 16.5-17.7 ms step: both passes read the 1213 MiB FP8 LM head; a
multi-row NVFP4 draft head (682 MiB, built for MTP) would read it once per pass. The economics guard
counts only lone verifies (1-2 unbinds per arm, main too).

## NVFP4 draft head in the batched passes (2026-11-04)

Both batched draft passes read the lone chain's NVFP4 draft head (`speculative.mtp_nvfp4_head`)
through `gemv_nvfp4_kpar_batched_fp32` (one weight pass per 4 rows) instead of the FP8 head;
verification stays on FP8. Evidence: docs/audit/qwen38_mtp_batched_draft_head_2026_11_04/.

| per pass, [248320 x 5120] (`microbench.txt`) | 1 row | 2 rows | 3 rows | 4 rows |
|---|---:|---:|---:|---:|
| FP8 rowscale, 1213 MiB | 926.3 us | 927.4 us | 928.8 us | 930.3 us |
| NVFP4 kpar_mb, 682 MiB | 425.3 us | 428.1 us | 504.1 us | 558.0 us |

| requests (`step_price.txt`, 3 pairs) | main head | new head | tokens/step main -> new |
|---:|---|---|---|
| 2 | 3.74-3.93 ms | 2.73-3.07 ms | 4.41-4.68 -> 4.38-4.69 |
| 3 | 3.77-3.97 ms | 2.91-3.19 ms | 6.92-7.01 -> 7.01-7.09 |
| 4 | 3.78-4.02 ms | 3.14-3.31 ms | 9.28-9.36 -> 9.10-9.39 |

Measured (3 alternating pairs vs 15d35035, `means.txt`): 2 overlapping chats 129.67 -> 135.18 tok/s,
4 chats 133.07 -> 138.15, 3/3 pairs higher each; c2 269.90 -> 293.50, c4 501.43 -> 534.43, c32
2023.50 -> 2033.67, c1 131.36 -> 131.31; accepted / drafted 68.34 -> 67.96 % (2 chats), 67.14 -> 67.20 %
(4 chats); lone chat 6/6 byte-identical on fresh servers, follow-ups 122.45 / 122.48 tok/s (the
side mode after the overlap modes, 122.94 -> 121.63, pairs mixed sign: 122.34 / 123.41 / 123.06 -> 121.26 / 123.51 / 120.13).

Next: the NVFP4 kernel loses bandwidth at 3-4 rows (1419 / 1282 GB/s vs 1681 at 1 row; closed in section
"Four weight rows per draft head block"); the rest of the head pass is the two BF16 MTP layer feeds (832 MiB
weights each).

## One table upload per head pass (2026-11-04)

The batched head pass copied 8 pageable host tables per step (feed tokens / slots / positions /
source rows, the draft gather rows, the chain's source rows / slots / positions); `mtp_head_pass_upload`
packs all 7 (4n + 3m ints) into a pinned buffer, one H2D, and both feeds read them on the device
(`mtp_feed_rows_multislot_dev`). Evidence: docs/audit/qwen38_mtp_head_pass_one_h2d_2026_11_04/.

| per head pass, nsys c2 (`nsys_head_pass.txt`, ~375 passes) | main 11088fbb | this branch |
|---|---:|---:|
| host wall, first table H2D to draft sync | 2813.0 us | 2498.0 us |
| GPU busy in that window | 2102.2 us | 2098.7 us |
| GPU idle in that window | 715.2 us | 393.1 us |
| `cudaMemcpyAsync` calls | 10 | 2 |

| requests (`step_price.txt`, 3 pairs, overlap A/B) | main head | new head |
|---:|---|---|
| 2 | 2.74-2.77 ms | 2.69-2.73 ms |
| 3 | 2.95-2.96 ms | 2.82-2.89 ms |
| 4 | 3.09-3.15 ms | 3.00-3.05 ms |

End to end (3 alternating pairs, `means.txt`): 2 overlapping chats 134.31 -> 137.35 tok/s (3/3 pairs
higher, but accepted / drafted also moved 65.7-67.8 -> 68.0-69.4 %, `acceptance.txt`), 4 chats 140.32 ->
140.37, side follow-ups 121.83 -> 122.21; c1 / c2 / c4 / c32 131.02 / 297.70 / 524.27 / 2024.20 ->
131.15 / 298.13 / 525.33 / 2035.40, within the pair spread. Lone chat 6/6 + 6/6 byte-identical to main
on fresh servers (`identity_summary.txt`).

Next: the 393 us of GPU idle left per pass are launch gaps between 56 kernel launches (rmsnorm, scan,
mrope, gather, split-K reduce); a captured graph per (rows, requests, scan tier) would replay them.

## The head pass as a CUDA graph (2026-11-04)

`speculative.mtp_head_graph` (default on): the pass after the table upload (feed, draft LM head, chain feed,
chain LM head) replays one graph per (fed rows, requests, chain), `src/runtime/mtp_head_graph.h`. First use of
a shape runs eager (cuBLASLt picks and caches the algos), the second captures it with Lt allowed in the
capture, later uses replay. No scan tier: before each replay `mtp_scan_nodes_set_rows` sets the two scan
nodes' dynamic smem to the pass's largest position + 1, rounded up to 256 rows
(`cudaGraphExecKernelNodeSetParams`), and the scan kernel traps when a row's scores exceed it. Bound: 48
graphs (4 requests x 1-3 rows x chain); the A/B servers held 19-24. The pinned draft D2H was priced and
dropped (nsys c2 host window 2561.8 -> 2687.2 us, n=1). Evidence:
docs/audit/qwen38_mtp_head_pass_graph_2026_11_04/.

| per head pass, nsys c2 (`nsys_head_pass.txt`, ~360 passes) | main 0989c2bb | this branch |
|---|---:|---:|
| host window, table H2D to draft sync | 2561.8 us | 2244.0 us |
| GPU busy in that window | 2101.3 us | 2087.6 us |
| GPU idle in that window | 433.4 us | 145.2 us |
| launch calls (`cudaLaunchKernel` + `cudaLaunchKernelExC`) | 25 + 31 | 1 `cudaGraphLaunch` |

| requests (`step_price.txt`, 3 pairs, overlap A/B) | main head | new head |
|---:|---|---|
| 2 | 2.66-2.77 ms | 2.33-2.46 ms |
| 3 | 2.87-2.96 ms | 2.48-2.64 ms |
| 4 | 2.99-3.13 ms | 2.67-2.77 ms |

End to end (3 alternating pairs, `means.txt`): 2 overlapping chats 134.75 -> 140.92 tok/s (pairs 136.62 /
132.63 / 135.00 vs 134.82 / 141.97 / 145.96), 4 chats 140.40 -> 143.47 (3/3 pairs higher), side follow-ups
122.43 -> 122.99; c1 / c2 / c4 / c32 129.30 / 293.50 / 511.83 / 1943.33 -> 127.80 / 295.40 / 527.70 /
1951.07 (c1 main pairs 132.95 / 127.68 / 127.27 vs 128.10 / 127.78 / 127.51), drafted / accepted 1822 / 1275
in every arm, own peak 25176 -> 25196 MiB. Lone chat and side mode 6/6 + 6/6 byte-identical to main; two
overlapping chats match main 1/12 turns, the same 1/12 as main against a second main server
(`identity_summary.txt`).

Next: the 145 us left per pass are host time before the graph's first node (86 us ends at the first
gather: table H2D issue plus `cudaGraphLaunch`, 45.5 us host); moving the table upload into the graph
(a memcpy node from fixed pinned staging) would remove the H2D call.

## Four weight rows per draft head block (2026-11-04)

`gemv_nvfp4_kpar_mb_fp32_kernel` (`src/quant/nvfp4_gemv_batched.cu`) gave each 128-thread block one weight
row: per weight byte it re-read MR x 4 bytes of x from L1 and converted every f16 operand (`HADD2.F32`)
before the `FFMA`. Now a block takes `kKparNR` = 4 weight rows against x fragments held in registers,
multiplies with `fma.rn.f32.f16` (SASS `FHFMA`, one rounding, equal to `__fmaf_rn` on the converted halves)
and reduces all rows behind one barrier. Per (row, x row) the float op order is unchanged: bit-identical,
pinned by `LmHeadDraftRows` against a host replica of that order (red under a swapped-FMA mutant).
Evidence: docs/audit/qwen38_nvfp4_head_kpar_nr4_2026_11_04/.

| [248320 x 5120], per pass (`microbench.txt`) | 1 row | 2 rows | 3 rows | 4 rows |
|---|---:|---:|---:|---:|
| main, median of 8 alternating rounds | 445.9 us | 461.8 us | 583.1 us | 659.8 us |
| this branch | 436.3 us | 447.9 us | 463.0 us | 456.6 us |
| ncu, locked clocks: main / branch | 456.8 / 456.9 us | 469.1 / 455.8 us | 568.0 / 457.1 us | 629.1 / 470.9 us |
| ncu DRAM throughput: main / branch | 89.40 / 89.25 % | 86.76 / 89.27 % | 74.66 / 88.97 % | 64.82 / 86.47 % |

Registers MR 1-4: 40 / 40 / 48 (4 B spill) / 52 -> 80 / 92 / 122 / 128; SASS instructions at MR 4 642M ->
343M. Refuted in the same harness (rows 3-4 slower than 4 rows per block): front-loading one row's weights,
a persistent grid prefetching the next row, `FHFMA` alone, 8 rows per block (8 B spill at MR 4).

| requests, head pass (`step_price.txt`, 3 pairs) | main 173d66be | this branch |
|---:|---|---|
| 2 (overlap A/B) | 2.44-2.45 ms | 2.32-2.46 ms |
| 3 (overlap A/B) | 2.55-2.64 ms | 2.35-2.50 ms |
| 4 (overlap A/B) | 2.72-2.80 ms | 2.41-2.51 ms |
| 4 (c4 bursts) | 2.54-2.55 ms | 2.24 ms |

End to end (3 alternating pairs, `means.txt`): 4 overlapping chats 141.26 -> 142.76 tok/s (3/3 pairs
higher), c4 534.17 -> 564.33 (3/3), c1 132.86 -> 132.97. At 2 requests the head pass does not move, and
neither does tok/s beyond spread: 2 chats 137.57 -> 134.47 (pairs 142.13 / 134.69 / 135.90 vs 137.88 /
129.60 / 135.93), c2 302.80 -> 303.17 over 6 pairs (second run arms reversed). The burst prompts carry the
arm label, so acceptance differs by prompt, not by kernel. Lone chat and side mode 6/6 + 6/6 byte-identical
to main; two overlapping chats match main 1-2/12 turns, main against main 2/12 (`identity_summary.txt`).

Next: one pass for 5-8 rows (MR 8, 4 rows per block: rows 8 667.9 us vs 947.9 for 4 + 4) serves the
batched decode LM head and `gemm_nvfp4` at M <= 16, not the draft head (<= 4 rows). The FP16 twin
`gemv_nvfp4_kpar_mb_fp16_kernel` is refuted as a lever: 0 launches on Qwen3.8-27B-NVFP4 at c1 / c4 / c32
and a 33612-token prompt (captured verify chunks take the small-M kernel, MTP stops past its 16384-row
head KV); `docs/audit/qwen38_kpar_fp16_verify_share_2026_11_04/`, SETTLED.md section H.

## Draft 1 as a factor row (2026-11-11, `speculative.mtp_batch_verify_factor_row`, default true)

At 2 drafts the second spare slot per row (the 1 + 1 row state) is gone: draft 1 leaves the main scan pass
as a GDN factor row plus a conv tap, folded into the live slot when exactly 1 of 2 drafts is accepted. A
concurrent GDN-hybrid server keeps 2 drafts by default again (`tools/common/mtp_auto.cpp`; the NVFP4 draft
head stays off there). Evidence: docs/audit/qwen38_mtp_verify_factor_row_2026_11_11/.

| piece | rule |
|---|---|
| scan | one pass, 3 rows: the spare (`ssm_out_slots`) commits all 3, the live slot takes row 1 in place (`ssm_snap_slots`, `ssm_snap_rows` 1), row 2 goes to `ssm_fac_out` at the live slot (`gdn_factor_mid_row`: `fac_out` WITH `out_slots` stores the row after the snapshot, the commit stays); the commit-only pass and the third conv commit are gone |
| conv | `ssm_conv_tap_stash` keeps draft 1's input column per live slot (`ssm_tap_out`), `ssm_conv_tap_fold` shifts it into the live window |
| accept | `bound_verify_kept_slot`: 0 matched = live slot, 2 matched = spare (the same bytes as with a second spare), 1 matched = `gdn_factor_fold` + `ssm_conv_tap_fold` on the live slot (`bound_verify_factor_row_folds`, `src/runtime/spec_verify_factor_rows.cpp`) |
| numerics | the fold is the scan's own update H = g * H + k * delta on the stored 1-row state: bit-equal to the 2-row scan with an f32 state, the rounding of a token-by-token decode with BF16 |
| VRAM | Qwen3.8-27B, `--max-batch 32`: 48 layers x 36 slots, 82.3 MiB of factor rows + 33.8 MiB of conv taps (3.22 MiB per recurrent slot) instead of 4 x 80.0 MiB; spare recurrent slots 8 -> 4 |

Measured (`ab_table.txt`: `tools/analysis/mtp_longctx_ab.sh`, Qwen3.8-27B-NVFP4-vllm server defaults, 384
greedy tokens per chat, 2 alternating pairs, tok/s over all streams, min-max; k 1 = main's concurrent default):

| chats x prompt | k 1 | factor row | second spare |
|---|---|---|---|
| 1 x 20k | 137.08-137.79 | 165.83-166.20 | 149.76-151.26 |
| 1 x 40k | 133.81-133.86 | 156.86-156.98 | 155.09-155.70 |
| 1 x 80k | 128.41-128.80 | 154.30-154.34 | 149.54-149.57 |
| 2 x 20k | 248.94-249.24 | 287.77-288.06 | 289.06-297.90 |
| 2 x 40k | 243.31-244.16 | 288.09-292.06 | 266.97-272.43 |
| 2 x 80k | 240.15-240.44 | 284.98-286.72 | 270.32-271.41 |
| 4 x 8k | 476.19-482.75 | 575.17-583.27 | 531.67-569.19 |
| 4 x 20k | 450.19-451.67 | 509.62-509.81 | 501.18-508.91 |
| 8 x 8k (no bound verify) | 569.98-571.44 | 570.43-573.75 | 570.40-572.97 |

| | k 1 | factor row | second spare |
|---|---|---|---|
| KV plan, blocks | 11239 | 10901 | 10313 |
| step at 2 requests: forward + argmax, head, tokens/step | 13.13 ms, 1.27 ms, 3.66 | 13.52 ms, 2.43 ms, 4.90 | 13.91 ms, 2.43 ms, 4.83 |
| step at 4 requests | 13.56 ms, 1.38 ms, 7.45 | 14.28 ms, 2.73 ms, 10.42 | 15.02 ms, 2.70 ms, 10.12 |

Gates: `GdnVerifyTwoDraftsTest.ScanFactorRowFoldsTheLiveSlotToTheTwoRowState` (+ `Bf16`) and
`ConvFactorRowFoldsTheLiveWindowToTwoRows`; mutant sets A and B (2 kernel mutants each) fail 3 and 2 tests
(`mutants.txt`). Engine level: `tools/analysis/mtp_factor_row_equiv.sh` (GPU, manual): with
`gdn.state_bf16=false`, 2 chats x 20k come out md5-equal with the factor row and with the second spare, 2
runs, 606 = 606 drafts (`equiv.txt`); an engine that never folds (exit 1 in that script) or stashes the wrong tap row changes both
texts. Tok/s and acceptance do not see those two mutants (dev build, 303.14 and 318.42 tok/s against 287.41 clean, the
text corrupted), so the md5 check is the gate. The streams start 300 ms apart (`STAGGER_MS`): started
together, the texts follow which stream arrives first (8 of 8 servers), not the arm.

Refuted first variant (`spare_fold_variant_ab_table.txt`): the live slot keeps row 1, the spare takes rows
1-2 plus a factor row of draft 2, folded on every full accept. 4 chats x
20k 487.96-491.16 tok/s against 503.68-510.05 with the second spare; the variant above folds only the
1-of-2 groups and reads 509.62-509.81 there.

Next: factor rows indexed by verify group instead of by recurrent slot (4 groups instead of 36 slots of
3.22 MiB), done in the next section; adopting the spare when the request's own chain is 1 draft long.

## Up to 3 drafts per bound request (2026-11-11, `speculative.mtp_batch_verify_k`, default 3)

A bound request stages D drafts + its last token = T = D + 1 rows; one spare slot per row keeps the state
after all T rows, every shorter accept is rebuilt from factor rows. Evidence:
`docs/audit/qwen38_mtp_verify_depth3_2026_11_11/`.

| Piece | Rule |
|---|---|
| Factor rows, conv taps | indexed `[verify group g][level lv]`, row `g * L + lv`, L = D - 1 levels (`ssm_tap_levels`); drafts 1..D-1 leave the scan mid-pass next to the commit |
| Memory | 48 layers x 4 groups x 2 drafts = 18.3 MiB of factor rows + 7.5 MiB of conv taps at D 3, 9.1 + 3.8 MiB at D 2; by recurrent slot (36 slots, #3088) it was 82.3 + 33.8 MiB at D 2 |
| Accept | 0 matched = the live slot; D matched = the spare; 1..D-1 matched = fold that many rows + taps into the live slot (`bound_verify_factor_row_folds`, fold list `[live slot][row base g * L][count]`, `kGdnFactorFoldMaxRows` 2) |
| Depth | `mtp_bound_verify_depth` = clamp(key, 1, 3 with factor rows, else 2); `mtp_k=auto` on a concurrent GDN hybrid follows it up; an unset key follows a smaller `mtp_k` down (`mtp_verify_depth_follows_k`, in `tools/common/mtp_auto.cpp` and `Engine::init`) |
| KV plan | 10901 blocks on main (#3088) -> 11201 at D 2 -> 11164 at D 3 (Qwen3.8-27B-NVFP4-vllm, 32 GB) |

Qwen3.8-27B-NVFP4-vllm server defaults, RTX 5090, `tools/analysis/mtp_longctx_ab.sh`, one image, arms default
(D 3) against `speculative.mtp_batch_verify_k=2`, 3 sessions x 768 greedy tokens per load, streams started
together (`ab_768x3_table.txt`, `ab_768x3_digest.txt`):

| load | D 3, tok/s per stream | D 2 | D 3, all streams (min-max) | D 2, all streams (min-max) | accepted / drafted D 3 | D 2 |
|---|---|---|---|---|---|---|
| 1 x 20k | 181.83 | 174.52 | | | 1505 / 2370 | 1409 / 1772 |
| 1 x 80k | 181.18 | 164.89 | | | 1536 / 2277 | 1390 / 1812 |
| 2 x 20k | 164.40 | 156.36 | 318.54 (306.59-334.47) | 295.45 (292.43-299.53) | 2954 / 4860 | 2697 / 3738 |
| 2 x 80k | 160.96 | 150.90 | 310.53 (286.91-322.44) | 297.62 (283.43-307.35) | 3010 / 4650 | 2739 / 3654 |
| 4 x 8k | 163.02 | 151.03 | 602.79 (580.98-622.74) | 581.40 (579.48-584.12) | 6040 / 9096 | 5450 / 7244 |
| 4 x 20k | 144.64 | 136.78 | 544.05 (536.57-554.28) | 522.60 (508.96-536.42) | 5890 / 9510 | 5296 / 7528 |

| | D 2 (3 rows) | D 3 (4 rows) |
|---|---|---|
| step at 2 requests, 80k: forward + argmax, head, tokens/step | 13.90 ms, 2.48 ms, 5.01 | 14.48 ms, 3.63 ms, 5.95 |
| step at 4 requests, 20k | 14.44 ms, 2.91 ms, 9.56 | 15.23 ms, 4.31 ms, 11.27 |
| own peak, 1 x 20k / 4 x 20k / 2 x 80k | 22166 / 24473 / 24950 MiB | 22208 / 24531 / 24935 MiB |

- The third draft is accepted less often (1 x 20k 63.5 % of the drafts against 79.5 %) and still pays: the
  step at 4 requests carries 11.27 tokens in 19.54 ms against 9.56 in 17.35 ms.
- The gain is text dependent: worst session 1 x 20k, 158.17 tok/s at D 3 against 165.64 at D 2 (464 of 903
  against 453 of 624 drafts accepted).
- D 2 on this tree against main d1f91bd6 (`ab_main_control_digest.txt`): 16 of 16 streams equal in md5,
  drafted and accepted counts.
- A staged row that carries no draft costs: main, 1 chat x 20k, 1 draft, 124.11 tok/s at 3 rows against
  137.69 at 2 rows. Hence the verify depth follows a smaller `mtp_k`.

Gates. Kernel: `GdnVerifyTwoDraftsTest.ScanFactorRowsFoldOneOrTwoDraftsOfThree` (+ `Bf16`) and
`ConvFactorRowsFoldOneOrTwoDraftsOfThree`; 4 kernel mutants fail 2-6 of the 9 tests. Depth resolution:
`MtpAuto.*` and `BatchVerifyPredicates.*` (42 tests), 5 mutants fail 2-4 each; `Engine::init` without its
follow-down call serves 3 drafts at `speculative.mtp_k=2` (all in `mutants.txt`). Engine level:
`tools/analysis/mtp_factor_row_equiv.sh` (GPU, manual, `equiv.txt`: 5 of 5 arms md5-equal on a quiet host,
its 2-chat pair check differed once beside `make preflight`). Final image: `degen_suite.py` 52 checks 0 FAIL,
`make test-spec-fidelity` passed, `make verify-fast` OK (`gates.txt`).

The engine detector is 1 chat at a fixed depth with `gdn.state_bf16=false` and `speculative.ngram=false`,
`speculative.mtp_k` varied: at D 3 the texts at `mtp_k` 3, 2, 1 are md5-equal (531fa8137d6b); an engine
that never folds or folds 1 row at most changes the `mtp_k=3` text (a0e675b52ef9, afc4f7b6b391). The
greedy text is NOT equal across depths: 5e231af2f6e6 at D 2 on this tree and on main, 62b9754c388b at D 1
on main (rows per step 4, 3, 2). At 2 chats the text also follows which stream joins first, so a 2-chat md5
is no detector.

Next: verify depth per request by its acceptance (the worst session above); adopting the spare when the
request's own chain is shorter than D.

## 4 drafts per bound request, 3 from 4 requests up (2026-11-11, `speculative.mtp_batch_verify_k` 4, `_k_wide` 3)

The verify depth is a property of the step: 1 to 3 requests stage 4 drafts each (T = 5 rows), 4 requests stage
3 (T = 4). Evidence: `docs/audit/qwen38_mtp_verify_depth4_2026_11_11/`.

| Piece | Rule |
|---|---|
| Step depth | `mtp_bound_verify_step_depth(depth, wide_depth, n)`: `speculative.mtp_batch_verify_k` (default 4, `kMtpBoundVerifyMaxDepth` 4) below `kMtpBoundVerifyWideRows` (4) requests, `speculative.mtp_batch_verify_k_wide` (default 3, clamped to 2..depth) from there up; staged rows, head chain levels and the factor row stride follow the step |
| Factor rows, conv taps | allocated for the deeper setting, 48 layers x 4 groups x 3 drafts = 27.4 MiB + 11.2 MiB (18.3 + 7.5 at D 3); a step at depth D writes row `g * (D - 1) + lv` (`bound_verify_fold_lists(levels, ...)`) |
| Accept | 1..D-1 matched = fold that many rows + taps, `kGdnFactorFoldMaxRows` 3 |
| Head | a fourth chain level reads `chain_pos3` (`src/compute/mtp_pass_tables.cpp`); the chain depth of a pass is the step's |
| KV plan | 11164 blocks at D 3 -> 11126 at D 4 (Qwen3.8-27B-NVFP4-vllm, 32 GB); own peak 1 x 20k 22208 -> 22202 MiB, 2 x 80k 24935 -> 24937 |
| Trace | `diagnostics.spec_trace` prints `spec-batch-trace n= T= fwd= head= \| id:nd:m:e` per step (`src/runtime/spec_trace.cpp`), `tools/analysis/mtp_trace_hist.py` groups it by (n, T) |

Qwen3.8-27B-NVFP4-vllm server defaults, RTX 5090, `tools/analysis/mtp_longctx_ab.sh`, one image, arms default
(new) against `speculative.mtp_batch_verify_k=3` (d3), 2 pairs x 2 sessions x 768 greedy tokens per load, streams
started together (`ab_table.txt`, `ab_digest.txt`):

| load | new, tok/s per stream (min-max) | d3 | new, all streams (min-max) | d3, all streams (min-max) | accepted / drafted new | d3 |
|---|---|---|---|---|---|---|
| 1 x 20k | 211.32 (209.64-212.82) | 194.25 (193.21-195.18) | | | 2218 / 3368 | 2082 / 2934 |
| 1 x 80k | 188.66 (170.44-199.89) | 176.56 (176.01-177.08) | | | 2126 / 3736 | 2018 / 3126 |
| 2 x 20k | 172.78 (156.53-190.00) | 164.74 (155.65-178.20) | 324.78 (313.48-334.59) | 315.45 (307.78-333.88) | 4191 / 7588 | 3939 / 6459 |
| 2 x 80k | 165.97 (154.29-177.87) | 162.12 (143.50-173.87) | 320.01 (304.66-336.94) | 311.60 (287.36-326.30) | 4242 / 7384 | 4031 / 6153 |
| 3 x 8k | 178.99 (166.49-201.26) | 173.98 (155.36-191.91) | 500.39 (491.65-517.76) | 493.42 (466.70-507.55) | 6408 / 10752 | 6069 / 9108 |
| 3 x 20k | 161.58 (147.29-185.76) | 154.48 (141.30-170.00) | 444.28 (433.91-459.37) | 427.93 (416.31-443.20) | 6270 / 11340 | 5867 / 9687 |
| 4 x 8k | 159.64 (148.83-173.79) | 161.89 (147.98-177.47) | 589.75 (582.87-596.41) | 594.82 (578.27-605.79) | 8012 / 12520 | 8046 / 12138 |
| 4 x 20k | 143.83 (132.51-160.58) | 144.32 (130.07-159.79) | 530.61 (518.46-550.50) | 530.84 (509.48-538.64) | 7866 / 13064 | 7819 / 12855 |

The wide key confirmed by its opposite (`speculative.mtp_batch_verify_k_wide=4`, same runs): 4 x 20k 517.18 tok/s
all streams (504.29-535.30) against 530.61, 4 x 8k 588.57 (558.74-618.82) against 589.75.

Step medians (dev build, `step_trace.txt`), forward + argmax, head, tokens per step:

| requests in the step | 3 drafts | 4 drafts |
|---|---|---|
| 1 (20k + 80k) | 12.871 ms, 3.204 ms, 2.83 | 13.078 ms, 4.227 ms, 3.09 |
| 3 (8k + 20k) | 13.772 ms, 3.670 ms, 8.54 | 14.192 ms, 4.887 ms, 9.59 |
| 4 (8k + 20k) | 14.709 ms, 4.010 ms, 11.52 | 15.506 ms, 5.333 ms, 12.77 |

- The fourth draft is one more head pass over every request: +1.02 ms at 1 request, +1.32 ms at 4. At 4
  requests the step carries 12.77 tokens in 20.84 ms against 11.52 in 18.72 ms: no gain, hence the wide key.
- 1 x 80k, session 0 of the default in pair 1 and pair 2 (same text, 515 of 1000 drafts accepted both times):
  170.44 and 186.15 tok/s; at 3 drafts 176.88 and 176.01. One session of a load is no verdict; the table
  carries min-max.
- The worst session of the 3-draft section (1 x 20k, third session, `gates.txt`): 158.72 tok/s at 3 drafts
  (464 of 903 drafts accepted) -> 169.60 on the default (501 of 1056).
- d3 on this tree against main 2b6a1d37 (`main_control.txt`): 14 of 14 streams equal in md5, drafted and
  accepted counts; 194.67 against 193.97, 166.47 against 166.74, 150.53 against 150.51 tok/s per stream.
- Acceptance along the chain does not decay (`depth_replay.txt`, 6 drafts single-stream, 530 steps):
  P(level k matches, given k - 1 matched) = 0.800, 0.705, 0.732, 0.799, 0.743, 0.838.

Depth per request by its own acceptance is not built (ledger B-67.1). Replay of traced depth-2 and depth-3 steps
through 50 policies (`tools/analysis/mtp_depth_replay.py`, `depth_replay.txt`, a calculation, no A/B): the best
policy without the outcome reaches +0.01 %, -0.02 %, -1.62 % against fixed depth 3 at 1, 2, 4 chats; knowing each
step's outcome is worth +3.95 %, +2.50 %, +1.34 %. A full accept repeats with P = 0.645, 0.648, 0.665: the last
steps do not predict the next one.

Until B-69.1 (next section) the think budget moved the greedy text with `mtp_k`: an MTP chunk landed up to
`mtp_k` tokens past the budget before the closer was forced (`src/runtime/engine_spec_ngram.cpp`). 1 chat x 20k,
384 tokens, default budget, main imp:m3090 at 3 drafts: `mtp_k` 1, 2, 3 give 3 texts (18844088bac2, 81466897ba0b,
bb3c0b591cc5); with `think_budget: 1` 3 of 3 equal (5ac4d15b4170). The depth check of this section ran with
`THINK_BUDGET=1` (`equiv.txt`); the check of the section above ran with the default budget, main at 3 drafts
fails it on the corpus used here.

Gates. Kernel: `GdnVerifyTwoDraftsTest.ScanFactorRowsFoldOneToThreeDraftsOfFour` (+ `Bf16`) and
`ConvFactorRowsFoldOneToThreeDraftsOfFour`; a fold of 2 rows at most fails 2 of 11, of 2 taps at most 1 of 11,
and both change the `mtp_k` 3 and 4 texts of the depth check. Step depth:
`BatchVerifyPredicates.StepDepthFollowsTheRequestsInTheStep` and `FoldListsUseTheStepsRowStride`, each red under
its mutant (`mutants.txt`). Engine level: `mtp_factor_row_equiv.sh` on the image, pair 2 of 2 and depth 4 of 4
arms md5-equal (`equiv.txt`). Final image: `degen_suite.py` 52 checks 0 FAIL at defaults and at
`--max-batch 1`, `make test-spec-fidelity` passed, `make verify-fast` OK (`gates.txt`).

Next: the head chain level at 4 requests (1.32 ms); adopting the spare when the request's own chain is shorter
than D.

## A verify chunk ends at the think budget (2026-11-11, B-69.1)

Rule: inside an open, budgeted think block a verify chunk emits at most the reasoning tokens the budget has
left (`think_logic::think_tokens_left`, `src/runtime/think_stop_logic.h`; `Engine::think_tokens_left_`); the
accept loop breaks at `emitted >= think_left` (`spec_accept_rows`, `src/runtime/spec_accept.h`), the last token
is the chunk's bonus and the next step is the eager step that forces the closer. Both callers pass it: the bound
batched verify (`engine_spec_batch_verify.cpp`) and the single-stream verify (`engine_spec_ngram.cpp`).

Qwen3.8-27B-NVFP4-vllm server, 1 greedy chat x 20k, 384 tokens, default think budget, reasoning tokens before
the closer (`reasoning_content` through `/tokenize`, the `think` field of `mtp_longctx_client.py`), main
e2906af1 -> fix, evidence `docs/audit/qwen38_think_budget_exact_2026_11_11/think.txt`:

| Path | `mtp_k` | main | fix |
|---|---|---|---|
| bound batched verify (server defaults) | 0 / 1 / 2 / 3 / 4 | 193 / 193 / 194 / 195 / 196 | 193 / 193 / 193 / 193 / 193 |
| single-stream verify (`runtime.max_batch_size=1`) | 0 / 1 / 2 / 3 | 193 / 193 / 193 / 194 | 193 / 193 / 193 / 193 |

Texts. Bound verify: `mtp_k` 1 to 4 give 4 texts on main and 1 on the fix (91e7ba2d359b);
`mtp_factor_row_equiv.sh` runs its depth check with the default budget again, 4 of 4 arms md5-equal on the
fix, 1 of 4 on main (`equiv.txt`). Single-stream verify: 3 texts for `mtp_k` 1 to 3 on both images, `mtp_k` 1
and 2 md5-equal between the images, 3 differs (main overshoots there); why the single-stream text follows
`mtp_k` is not measured. MTP off against MTP on differs with `think_budget: 1` too (350e66ac9bb9 against
6c3481086686 on both images): the eager md5 is no detector for the budget, the reasoning token count is.

Gates. CPU: `SpecAcceptRows.ChunkEndsAtTheThinkBudget` (`tests/test_spec_gates.cpp`), red with the budget break
removed (`mutants.txt`). Image: `degen_suite.py` 52 checks 0 FAIL at defaults and at `--max-batch 1`,
`make test-spec-fidelity` passed, `make verify-fast` OK (`gates.txt`). Speed at server defaults is not compared:
the texts of the two images differ from the budget cut on (`gates.txt`, last block).

## Suffix drafts for bound requests: shadow trace, not built (2026-11-11, B-70)

Question: `Engine::batch_verify_draft_` takes the MTP draft first and asks the suffix index only when MTP has
none, so a bound agent chat never drafts a copy span (file contents, earlier tool arguments). Would suffix
precedence at a long match pay?

Instrument. Under `diagnostics.spec_trace` every bound request step also asks the suffix index for a 32-token
draft (`kSuffixShadowDrafts`, `src/runtime/suffix_shadow.h`) and logs, once the tokens are known,
`spec-sfx-shadow id=I pos=P n=N T=T mtp=nd:m:e sfx=len:match:agree`: `match` is the backward match behind the
draft (`SuffixDraftIndex::draft` reports it, capped at `speculative.max_match`), `agree` the leading draft
tokens equal to what the request emitted from `pos` on. The step itself is unchanged.

Workload. `tools/analysis/agent_turn_replay.py` replays recorded agent sessions (Anthropic block format) against
`/v1/messages`: request t sends the recorded messages before assistant turn t and reads the model's own turn,
the next request continues from the recorded turn, so the prefix cache hits like in an agent loop (cached
95.5-97.3 % of the prompt tokens) and no tool runs. Recorded familiar SWE-bench sessions (55 sessions, 4427
turns: thinking 75.2 %, tool calls 20.9 %, text 4.0 % of the assistant characters, `recorded_mix.txt`),
Qwen3.8-27B-NVFP4-vllm server at defaults + `kv_cache.host_spill_mb=16384`, `speculative.max_match=32`,
`max_tokens` 4096, think budget 2048, tools rebuilt from the recorded calls, no system prompt.

Serial run (6 sessions x 60 turns, every request decodes alone; `client_serial.txt`, `shadow_serial.txt`):
360 requests, 166353 output tokens, reasoning 75.2 %, 165.47 tok/s decode, 60599 steps at 16.324 ms.

| match | steps | share % | MTP tok/step | suffix agree | agree >= 8 % | agree >= 16 % | suffix tok/step at MTP depth |
|---|---|---|---|---|---|---|---|
| none | 52091 | 86.0 | 2.52 | 0.00 | 0.0 | 0.0 | 1.00 |
| below 8 | 1930 | 3.2 | 3.73 | 4.08 | 15.9 | 6.4 | 2.85 |
| 8-11 | 2029 | 3.3 | 3.78 | 5.83 | 28.0 | 10.4 | 3.29 |
| 12-15 | 1055 | 1.7 | 4.28 | 8.06 | 39.0 | 14.5 | 3.94 |
| 16-23 | 1186 | 2.0 | 4.44 | 9.75 | 42.5 | 22.6 | 4.00 |
| 24-31 | 655 | 1.1 | 3.98 | 10.76 | 42.0 | 29.3 | 3.93 |
| 32+ | 1653 | 2.7 | 4.69 | 19.45 | 71.9 | 57.8 | 4.58 |

Where the suffix index matches, MTP already fills its 4 drafts (3.73-4.69 of 5 tokens per step): a suffix draft
at MTP depth emits less in every bucket. Only rows past the MTP depth can pay.

Policy replay (`tools/analysis/sfx_shadow.py`, a calculation, no A/B): a step with match >= M runs the suffix
draft at K rows and emits min(agree, K) + 1 tokens, every other step is the recorded MTP step; a row past the
measured row count costs 0.27 ms per request in the step (one row count per step: the other requests pad).
Gain against the recorded run, 1 request (168.16 tok/s):

| M | K 4 | K 8 | K 16 | K 32 |
|---|---|---|---|---|
| 6 | -1.04 % | +1.80 % | +2.87 % | +2.23 % |
| 8 | -0.55 % | +1.95 % | +3.08 % | +2.96 % |
| 12 | -0.26 % | +1.76 % | +2.75 % | +2.90 % |
| 16 | -0.15 % | +1.45 % | +2.21 % | +2.51 % |
| 24 | -0.02 % | +1.13 % | +1.68 % | +2.02 % |
| 32 | -0.05 % | +0.85 % | +1.34 % | +1.64 % |

Mixed run (1 chat x 60 turns, then 4 chats x 60 turns, 190118 tokens, row cost 0.284 ms fitted over 4 step
shapes; `client_mixed.txt`, `shadow_mixed.txt`): 1 request best +1.31 % (M 8, K 16); 4 requests lose at 23 of
24 points (M 12: K 8 -1.77 %, K 16 -5.00 %, K 32 -11.96 %), the pad rows of the 3 other requests cost more than
the copy returns.

Decision: not built. A deep suffix step needs factor-row folds past 3 rows (`kGdnFactorFoldMaxRows`), cap rows
and graph shapes up to 17 rows and a ragged row count per request to stop losing at 4 chats, for a replayed
+3.08 % at 1 request. The trace and both tools stay as the instrument. Gates: `SuffixShadow.*`
(`tests/test_suffix_shadow.cpp`, 5 tests), 6 of 6 mutants red (`mutants.txt`); 1 greedy chat x 20k, main, branch and
branch + trace md5-equal with 460 drafted, 266 accepted each, 115 shadow lines in the trace arm; `make verify-fast`
OK (`gates.txt`). Evidence
`docs/audit/qwen38_suffix_shadow_2026_11_11/`.

The same directory holds the profile behind ledger row B-69.2 (`head_level_*.txt`): the fourth head chain
level costs 1.32 ms at 4 requests; 3 matrix kernels hold 12.376 of 17.043 ms exclusive per step at 1 chat, every
other kernel stays below 0.7 ms. No lever identified, no arm built.

## Coupled MTP drafts: the draft draws what its verify row draws (2026-11-11, B-71)

A sampled bound request draws verify row j with the seed of its output index (`compute_step_seed`,
`src/runtime/engine_internal.h`). An argmax draft is accepted only where that draw lands on the head's top
token: in the off arm of the paired run below 29.2 % of the steps reject draft 1 and 33.1 % accept all 4.

| Piece | Where |
|---|---|
| Draft row per request and level: the request's `top_k` / `top_p` / temperature, the seed of output index `size() + level`; a greedy request's rows take `top_k` 1 | `engine_internal::draft_row_draws` |
| Rows staged and uploaded before the head pass, drawn inside it (graph key carries `draw`, graph cap 48 -> 96) | `GraphExecutor::stage_draft_draws`, `draw_uploaded_rows` (`src/compute/sampling_verify_rows.cpp`), `MtpHeadGraphKey` |
| Scratch for 32 rows, 2.02 MiB, allocated once at prewarm | `GraphExecutor::kDraftDrawRows` |
| Unseeded request: one sampler base per request, fixed at its first draw (was `hash(id) ^ clock` per step) | `Request::draw_base` |
| Sampler: exponential race `argmin(-log u / p)` with `u` hashed from (seed, token id), in the 3 top-k / top-p tails | `sampler_race_pick` (`src/compute/sampling_internal.cuh`) |

The emitted token stays the verify row's draw: with `runtime.deterministic=true` the text md5 of the on and
the off arm are equal (3 of 3 chats, 48176 tokens).

Sampler choice, measured on experiment builds of the branch (Qwen3.8-27B-NVFP4-vllm, recorded familiar
SWE-bench sessions via `tools/analysis/agent_turn_replay.py --serial --seed 1000`, 3 chats x 30 turns in turn,
`runtime.deterministic=true`, `speculative.mtp_k=4`, `speculative.ngram=false`; texts md5-equal within a
row; tokens per step / tok/s decode per chat):

| Sampler | drafts argmax | coupled, draft temperature x 0.7 | x 1.0 | x 1.3 |
|---|---|---|---|---|
| inverse CDF over the sorted candidates, one uniform per seed (the sampler before this change) | 2.787 / 165.71 | 2.833 / 168.26 | 2.818 / 167.34 | 2.673 / 158.79 |
| race keyed on (seed, token id) | 2.847 / 168.54 | 3.276 / 193.42 | 3.340 / 197.15 | 3.253 / 191.94 |

One uniform through a sorted CDF couples ranks, and head and target order their candidates differently: +1.5 % tok/s
at best. Keyed on the token, the two rows pick the same token wherever their probabilities agree. The draft
temperature scale and the switch between the samplers were arms only; neither ships.

Shipped tree (`imp:crn`), on = default, off = `speculative.mtp_draft_coupled=false`:

| Load | tokens per step | steps accepting all drafts | tok/s decode per chat | tok/s wall |
|---|---|---|---|---|
| 3 chats x 30 turns in turn, deterministic, texts md5-equal (16158 -> 13917 steps) | 2.981 -> 3.461 | 33.1 % -> 43.7 % | 176.13 -> 203.81 | 166.99 -> 191.62 |
| the same, default mode, off / on / off / on | 2.813 / 3.234 / 2.813 / 3.223 | 28.2 % / 35.7 % / 28.2 % / 35.4 % | 168.43 / 194.79 / 168.71 / 193.70 | 162.30 / 186.80 / 162.62 / 185.85 |
| 2 chats x 30 turns overlapping, deterministic (steps at 2 requests) | 3.102 -> 3.578 | 36.0 % -> 47.2 % | 157.23 -> 179.08 | 217.13 -> 253.76 |
| 4 chats x 20 turns overlapping, deterministic (steps at 4 requests, 3 drafts) | 2.761 -> 3.124 | 45.3 % -> 57.8 % | 123.91 -> 138.71 | 393.20 -> 421.46 |

Activity: `[mtp-head-graph] captured ... draw=<top_k>` lines, off arms 0, on arms 5 / 14 / 38 (1 / 2 / 4
chats). The overlapping rows are not paired (the texts differ between the arms).

Repeat across server starts, seed 1000, 3 chats x 30 turns in turn:

| Mode | drafts argmax | coupled |
|---|---|---|
| default | 2 starts: 3 of 3 texts md5-equal, 17921 of 17921 trace steps equal | 2 starts: 2 of 3 texts md5-equal, first differing step 5627 of 15858 (4 against 0 drafts accepted at one position) |
| `runtime.deterministic=true` | | 3 starts: 13917 of 13917 trace steps equal |

In default mode a token's logits depend on its row in the verify chunk (on and off arm give different texts there), so a draft that differs between two starts moves
the text; [`DETERMINISM.md`](../DETERMINISM.md) carries the row.

Gates: `SampleVerifyRows.CoupledDraftFollowsTheTokenNotTheRank` (503 of 512 rows),
`SampleVerifyRows.CoupledDraftMatchesTheVerifyDrawMoreOftenThanTheArgmax` (511 against 259 of 512 rows),
`SampleVerifyRows.DraftRowDrawsWhatTheVerifyRowDrawsOnTheSameLogits`, `DraftRowDraws.*`,
`StepSeed.UnseededRequestKeepsOneBase`; 5 of 5 mutants red, control 39 + 4 tests green (`mutants.txt`); `make verify-fast`
and the degeneration suite on the shipped tree in `gates.txt`. Evidence
`docs/audit/qwen38_coupled_draft_2026_11_11/`.

## Verify depth ladder after a full accept (2026-11-11, B-72, `speculative.mtp_batch_verify_k_streak`, default 7)

A step in which a request accepts all N drafts is followed by a step at N + 2 drafts, up to the key: 4 -> 6 -> 7.
A step without a full accept returns to `speculative.mtp_batch_verify_k`. Evidence:
`docs/audit/qwen38_verify_ladder_2026_11_11/`.

Why a ladder and not a deeper default: the deeper drafts hold only after a full accept. Qwen3.8-27B-NVFP4-vllm,
recorded familiar SWE-bench sessions (`tools/analysis/agent_turn_replay.py --serial --seed 1000`, 3 chats x 30
turns in turn, `runtime.deterministic=true`, `speculative.ngram=false`), share of steps whose draft l is accepted
(`survival.txt`):

| Step before | draft 4 | draft 5 | draft 6 | draft 7 | draft 8 |
|---|---|---|---|---|---|
| accepted fewer than its 4 drafts (5154 steps) | 28.4 % | | | | |
| accepted 4 of 4 (2223 steps) | 44.2 % | 35.2 % | 29.1 % | | |
| accepted 6 of 6 (645 steps) | 58.9 % | 52.2 % | 46.0 % | 40.5 % | 34.1 % |
| accepted 8 of 8 (551 steps) | 80.6 % | 76.2 % | 71.1 % | 63.0 % | 58.8 % |

| Piece | Rule |
|---|---|
| Next depth | `mtp_bound_verify_streak_next(full_drafts, streak_depth)`: the most drafts a request of the step accepted in full, plus `kMtpBoundVerifyStreakStep` (2), capped at the key; 0 = no full accept (`src/runtime/spec_batch_verify_state.h`) |
| Row tile | `mtp_bound_verify_tile_depth(depth, n, streak)`: a streak never moves the n x (depth + 1) rows of a step into the next 8-row tile of the small-M GEMM (`kMtpBoundVerifyRowTile`, `src/quant/nvfp4_gemm_smallm_ws.cu`): 1 or 2 requests reach 7 drafts (8, 16 rows), 3 requests stay at 4 (15 rows) |
| Wide tier | 4 requests keep `speculative.mtp_batch_verify_k_wide` |
| Depth cap | `kMtpBoundVerifyMaxDepth` 4 -> 8, `kGdnFactorFoldMaxRows` 3 -> 7, head chain levels `kMtpPassChainLevels` 7 (`MtpPassTables::chain_pos[l]`), `kMtpHeadGraphMax` 96 -> 384 |
| `mtp_k` | `speculative.mtp_k=auto` on a concurrent GDN-hybrid server resolves to the deepest step (7); an explicit `mtp_k` below the key caps the ladder (`mtp_verify_depth_follows_k`) |
| Factor rows, conv taps | 48 layers x 4 groups x 6 drafts = 54.8 MiB + 22.5 MiB (27.4 + 11.2 at 4 drafts); KV plan 17089 -> 16977 blocks (4806 -> 4775 MiB), server defaults with `kv_cache.host_spill_mb=16384`, 32 GB |
| `diagnostics.mtp_verify_pin=N` | every step below 4 requests stages N draft rows per request (pad rows past the drafts): the instrument of the paired comparison below, 0 = off |

Row tile, forward + argmax per step in ms by rows of the step (pin arms, drafts held at 4, `row_sweep.txt`):

| Requests | rows: ms |
|---|---|
| 1 | 5: 13.072, 6: 12.957, 7: 13.350, 8: 13.461, 9: 14.501 |
| 2 | 10: 14.687, 14: 15.057, 16: 15.180, 18: 16.195 |
| 3 | 15: 15.254, 21: 16.566, 24: 16.726, 27: 17.902 |

Paired comparison at 1 chat. The text of a chat depends on the rows of its steps, so two depth policies emit
different texts and a tok/s difference below 4 % is not readable from them. With the pin at 8 every policy stages
9 rows per step and the texts are md5-equal (4 policies, 3 of 3 chats, 40259 output tokens,
`pinned_clients.txt`). The pinned run gives the step count per class (drafts of the step, full accept or not), the
unpinned run of the same policy gives forward and head ms per class (`paired_cost.txt`):

| Policy | steps | tokens per step | tokens per ms | against 4 drafts |
|---|---|---|---|---|
| 4 drafts every step (`mtp_batch_verify_k_streak=0`) | 11974 | 3.3620 | 0.19893 | |
| ladder to 6 | 11234 | 3.5835 | 0.20288 | +1.99 % |
| ladder to 7 (default) | 11096 | 3.6281 | 0.20457 | +2.84 % |
| ladder to 8 (9 rows) | 11034 | 3.6485 | 0.20314 | +2.12 % |

Head pass per step class, ladder to 7: 4.009 ms after a partial step at 4 drafts, 6.005 ms after 4 of 4 (6 drafts
next), 7.082 ms after 6 of 6, 7.110 ms after 7 of 7.

Unpaired runs (the texts differ between the arms):

| Load | 4 drafts | ladder to 7 |
|---|---|---|
| 1 chat at a time, default mode (`mtp_k=auto` 4 / 7), tok/s decode per chat; tokens per ms | 193.79; 0.1944 | 200.95; 0.2017 |
| 1 chat at a time, deterministic, tok/s decode per chat | 203.79 | 206.92, 207.02 |
| 2 chats x 30 turns overlapping, deterministic: tok/s decode per chat; tok/s wall; tokens per ms of the steps at 2 requests | 179.05; 253.68; 0.3718 | 180.71, 184.10; 265.15, 263.81; 0.3807, 0.3892 |
| 3 chats x 30 turns overlapping, deterministic, tile rule (3958 of 3958 steps at 3 requests stage 4 drafts): tok/s decode per chat; tok/s wall; tokens per ms of the steps at 3 requests | 159.18; 356.39; 0.5096 | 157.66; 352.38; 0.5007 |
| the same without the tile rule (steps at 3 requests on 15 / 21 / 24 rows: 15.366 / 16.862 / 17.220 ms forward) | 158.26; 374.66; 0.5055 | 156.61; 362.38; 0.5035 |

At 3 requests a ladder step costs up to 1.85 ms more forward and gained nothing in tokens per ms (0.5035 against
0.5055), so the tile rule holds those steps at 4 drafts; the 0.5007 to 0.5096 spread is 3 runs of the same policy at
3 requests. Files: `overlap_shipped_steps.txt`, `overlap_shipped_clients.txt`, `overlap_per_requests.txt`,
`overlap_no_tile_clamp.txt`, `real_clients_pre_clamp.txt`.

Gates (`mutants.txt`, `gates.txt`): `BatchVerifyPredicates.StepDepthFollowsTheRequestsInTheStep` and `MtpAuto.*`
turn red on 13 of 13 mutants of the depth rules (tile clamp, tile rounding, step 2, pin, full accept at a finished
request, wide tier, factor-row condition, reset, default, `mtp_k` cap, auto `mtp_k`);
`GdnVerifyTwoDraftsTest.ScanFactorRowsFoldOneToSevenDraftsOfEight` (f32, BF16) and
`ConvFactorRowsFoldFourToSevenDraftsOfEight` cover the fold at 7 rows (`gdn_factor_fold`, `ssm_conv_tap_fold`),
red on 2 of 2 kernel mutants (fold capped at 3 rows, first tap skipped at 4 or more taps).

Not built: a second tile on a long streak. After 8 of 8 the next step accepts 8 of 8 again in 58.8 % of 551 steps,
and a 9th row costs 1.04 ms forward plus about 1.0 ms head per chain level.

Draft head format (B-73 lead, `draft_head_fp8_vs_nvfp4.txt`, pin 4, texts md5-equal, 48176 output tokens): FP8
draft head over rows [0, 98304) (server default) 4.084 ms head per step, 203.59 tok/s decode per chat; the full
NVFP4 copy (`speculative.mtp_nvfp4_head=true`, 682 MiB) 4.553 ms, 199.31 tok/s. The full copy stays off. Open: an
NVFP4 copy over the first `speculative.mtp_draft_vocab` rows only.

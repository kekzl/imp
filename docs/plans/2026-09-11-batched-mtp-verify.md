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

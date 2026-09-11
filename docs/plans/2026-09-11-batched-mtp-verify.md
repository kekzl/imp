# Batched MTP verify on the GDN hybrid (the concurrent-serving lever)

Status (2026-09-11): in progress, branch `perf/batched-mtp-verify`. Stage
table at the end carries the state of every stage.

## Why

Qwen3.8-27B-NVFP4-vllm at 32 streams: step 14.75-14.98 ms = 1977-2005 tok/s,
of which 12.06 ms is the DRAM floor (weights 13.7 GB + LM head 0.72 + GDN
state 4.83 r+w + KV 0.4 = 19.6 GB at 1628 GB/s, `docs/roadmap.md` 2026-09-08).
Every remaining lever inside that step is worth 1-3 %. The only way past the
floor is tokens per weight sweep: the MTP head accepts 83.5-84.5 % at depth 1
(`docs/roadmap.md`, "MTP acceptance gap"), so a step that forwards 2 rows per
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

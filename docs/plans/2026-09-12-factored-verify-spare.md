<!--
layer: L2
audience: kernel-devs
verified: 2026-09-12
commit: d28a5d48
-->

# Factored spare state for the batched speculative verify

## The price being paid

`speculative.batch_verify` reserves one spare recurrent slot per batch slot, held for a
request's lifetime because accept swaps slots instead of copying
(`engine_spec_batch_verify.cpp`). The spare is an exact duplicate of the state pool.
Qwen3.8-27B-NVFP4-vllm at `runtime.max_batch_size=32`, measured 2026-09-12:

| `speculative.batch_verify` | SSM/GDN state | KV pool | batch |
|---|---:|---:|---:|
| off | 2544 MiB (32 slots) | 2339 blocks, `max_model_len` 131 072 | 32 |
| on | 5088 MiB (32 slots) | 1429 blocks, `max_model_len` 33 680 | clamped to 18 |

79.5 MiB per slot, of which 72 MiB is the recurrent `h_state` and the rest the conv window.
The clamp lands in the regime where the verify is already a loss (32 streams
1966 -> 1159 tok/s, [2026-09-11-batched-mtp-verify.md](2026-09-11-batched-mtp-verify.md)).

## Why a triple is enough

The delta rule's per-token update in `gdn_scan_fused_kernel` is a scalar decay plus a rank-1
outer product:

    H_new[s][d] = g * H[s][d] + k[s] * delta[d]

`g` is per head, `k` the L2-normalised key (`state_size`), `delta = (v - g * H^T k) * beta`
(`head_dim`). So the state after the drafted row is fully determined by the state before it
plus `(g, k, delta)`: `1 + state_size + head_dim` floats per head against
`state_size * head_dim`. On this model (48 value heads, SS = HD = 128, 48 GDN layers) that is
48 KiB per layer against 1.5 MiB, and 2.3 MiB per slot against 72 MiB.

Same direction as TreeWY (arXiv 2608.20961) and Bole (arXiv 2608.01651), which report 82-99x
lower transient state memory for tree verification on linear-attention models. The rank-1 form
here is read off imp's own kernel, not ported.

## Where the work lands

`compute/gdn_factor.cuh` holds the layout and the two device helpers. Per (slot, head):

    [0]                      g, and EXACTLY 0 is the "no pending row" sentinel - g is
                             expf(fmaxf(A*dt, -20)) and cannot reach 0, so no side array
    [1 .. 4)                 padding, keeps k 16-byte aligned
    [4 + s]                  k[s], state_size of them
    [4 + SS + d]             delta[d], head_dim of them

`gdn_scan_fused_kernel` gained three optional parameters. `fac_out` receives the row for
`real_n` **instead of** the full state copy `out_slots` used to take; `fac_in` carries a
pending row applied to `H_reg` right after the state load, before any token is processed, so
applying it costs no extra state traffic at all - the next step reads and writes the state
regardless. Rows are indexed by the recurrent **slot**, not the batch position: a request keeps
its slot across steps but moves within the batch.

Invariant the kernel enforces: with `fac_out` set, the row is always defined. A row is written
only when there is a drafted row past the snapshot (`snap_n >= 1 && real_n > snap_n`);
otherwise the full state goes to `h_out` as before and `g` is set to the 0 sentinel, so a stale
row from an earlier step cannot survive to be applied twice.

## Status

| stage | state |
|---|---|
| `gdn_factor.cuh`, kernel emit + apply, F32 and BF16 launchers | built |
| `GdnBatchedScanTest.FactoredSpareReproducesTheFullSpareAfterTheNextToken` | GREEN, bit-exact; red under a mutated delta column |
| engine wiring: buffers, verify emits, accept/reject bookkeeping, next-step apply (`speculative.factored_spare`) | built, opt-in |
| drop the spare slots from `batch_verify_spare_slots` and the planner | built, follows the flag |
| conv window: `ssm_conv_tap.cu`, stash + apply, `SsmConvTapTest` | built |
| quality gate on the factored path (`degen_suite.py`, 50 checks) | GREEN |

The test is the gate on the algebra: it runs the real accept sequence both ways (full spare,
then factored) and compares the state **after the following token**, which is what the engine
actually consumes. F32 state makes it exact, so it carries no tolerance - a tolerance there
would hide a wrong `k` or `delta` index.

## The conv window is a blocker, not an extra (built 2026-09-12)

First reading had the conv window as an optional follow-up. It is not. A slot is one
contiguous block holding every GDN layer's conv window AND `h_state`
(`memory/ssm_state_size.h`, `ssm_bytes_per_slot`), and the spare exists because accept swaps
whole slots. Factoring `h_state` alone leaves the drafted conv window with nowhere to live, so
dropping the spare slot needs the conv factored too: the drafted row shifts the 4-tap window
by one, so the carried form is the single new tap (40 KiB per layer against 160). That path
has its own commit kernel (`ssm_conv1d_commit_kernel`) and its own race history (#1976, the
prefill commit race), so it gets its own stage and its own test.

The alternative is to keep slot-shaped spares but give the spare region a different geometry,
which means `SsmStateCache` stops having one uniform slot stride. That is the larger change of
the two.

`compute/ssm_conv_tap.cu` carries it: `ssm_conv_tap_stash` saves the drafted row's conv input
per slot (channels halfs, 20 KiB per layer at 10240 channels against 160 for the window) and
`ssm_conv_tap_apply` advances a slot's window by that tap, which is the single-row form of
`ssm_conv1d_commit_kernel`. Its own translation unit because `ssm.cu` sits at 596 of the
600-code-LOC kernel ceiling.

The wiring does not need a "suppress the commit" flag: committing the conv at the SNAPSHOT
length and stashing the tap for the drafted row is the same thing as committing at both
lengths into two slots, because `d_real_n` bounds only the commit, never the conv output rows
the scan consumes.

Spare per slot with both halves factored: 2.3 MiB of recurrent factors plus 0.96 MiB of conv
taps against 79.5 MiB, so 32 slots cost about 105 MiB against 2544.

## How the wiring is shaped (2026-09-12)

`speculative.factored_spare` is opt-in on purpose: with it off the verify keeps swapping slots,
with it on the same binary carries the drafted row as factors. That makes the two paths an A/B
against each other, which is the evidence the flip of the default needs.

With the flag on, `batch_verify_spare_slots` returns 0, so the planner stops charging a second
slot per batch slot. The verify then sets `out_slots` to null and snapshots in place, emits the
recurrent row into `bv_.d_fac` and the conv tap into `bv_.d_tap`, both indexed by the live
slot. The conv is committed at the SNAPSHOT length rather than the chunk length, which is the
same window the two-slot form left in the live slot.

Applying costs nothing extra on either half: the conv tap advance runs once per layer before
the conv reads the window, and the recurrent row is folded into `H_reg` at the state load the
scan performs anyway.

## What it buys, measured 2026-09-12

Qwen3.8-27B-NVFP4-vllm, one image, `speculative.batch_verify=true` and
`speculative.mtp_k=1` in both arms, 32 concurrent streams x 200 tokens with `ignore_eos`,
two rounds, idle card:

| arm | batch | KV pool | tok/s |
|---|---|---:|---:|
| slot-swapping spare | clamped 32 -> 18 | 298 blocks | 1117.1 / 1104.6 |
| factored spare | 32, no clamp | 1429 blocks | 1761.1 / 1898.3 |

The clamp is the whole difference: the spare pool is an exact duplicate, so enabling the
verify used to cost the batch AND the KV pool, and the factored form gives both back. At 16
slots the state pool reads 2544 MiB against 1272, exactly the halving the arithmetic predicts.
`degen_suite.py` is 50/50 on the factored arm.

For context, the verify-off arm at 32 streams measured 1966 tok/s on 2026-09-11
([2026-09-11-batched-mtp-verify.md](2026-09-11-batched-mtp-verify.md)), so the batched verify
is no longer a disaster at this concurrency but is still not obviously a win over leaving it
off. That is why the flag stays opt-in.

## Byte equality is not an available oracle here

The obvious check - same prompts, greedy, slot-swapping arm against factored arm, identical
tokens - cannot work. The batched verify needs concurrency, and under concurrency which
requests carry a draft on a given step is a timing race, so the two arms take different step
sequences: measured, the first batched verify of one arm had 8 drafts and the other 1. Greedy
text then diverges for reasons that have nothing to do with the state (SETTLED D-2, and the P7
finding of 2026-09-10 that greedy under foreign GPU load is a race).

Two arms with different memory plans are worse still: left to itself the factored arm keeps a
larger batch and a six times larger KV pool, and batch shape alone moves greedy output.
`tools/analysis/factored_spare_equiv.sh` therefore pins `runtime.max_batch_size` and
`kv_cache.max_blocks` in both arms, and it still cannot assert equality.

What does discriminate is the acceptance rate: the verify accepts a draft only when the
model's own next token matches it, so a wrong recurrent state or conv window collapses
acceptance towards zero. Measured with the geometry pinned, slot-swapping 73.4% against
factored 64.3% - the same band, on different step counts.
`tools/analysis/factored_spare_accept.sh` is that check.

## Bookkeeping the wiring stage has to get right

The scan self-clears in steady state: a verify step applies the row left by the previous one
at the state load and writes a new row at the commit row, same launch. Explicit clears are
needed exactly where that chain breaks, and each one is a way to advance a request's state by
a token it never accepted:

| event | why the row must be cleared |
|---|---|
| draft rejected | the next step would apply a row for a token that was not accepted |
| request finished | same, for whatever request takes the slot next |
| slot (re)assigned | a new request would inherit the previous tenant's row |
| a non-verify step runs for that slot | nothing consumes the row, and it outlives its state |

The 0 sentinel in `g` is what a clear writes, and the kernel already maintains "with `fac_out`
set, the row is always defined" for the no-draft shape. `gdn_factor_clear` writes it for a list
of slots; the list is filled from the verify's own rejects and from
`release_recurrent_slot_`, and flushed once per verify after every group has been classified.
The conv half needs no clear because its apply is driven by the slot list rather than by the
row content.

A second gate sits in front of all of it: `ssm_fac_in` and `ssm_tap_in` are null unless the
previous verify accepted at least one draft, so a step with nothing pending cannot apply
anything at all.

## Correction: the 32-stream numbers drafted for one request per step (2026-09-12)

`engine_spec_mtp.cpp` sized the MTP draft KV pool from `batch_verify_spare_slots() > 0`, which
this plan sets to 0 for the factored form. One KV slot, one bound request, every other row a
pad. Fixed by keying on the verify (`mtp_draft_kv_slots`, `tests/test_batch_verify_predicates.cpp`).

Qwen3.8-27B-NVFP4-vllm, `mtp_k=2 ngram=false verify_smallm=true`, `--think-budget 0`, 24 streams
x 300 tokens with `ignore_eos` (`tools/analysis/conc_arms_ab.sh`), 3 trials x 3 waves:

| geometry | arm | tok/s (medians) | verify counters |
|---|---|---|---|
| mbs 32, seq 4096, before the fix | `batch_verify=false` | 1402.3 / 1435.4 / 1440.4 | - |
| mbs 32, seq 4096, before the fix | `batch_verify=true factored_spare=true` | 1014.2 / 1023.7 / 1016.4 | 127 steps, 128 drafted, 24.28 tok/verify, 24.61 ms |
| mbs 24, seq 1024, after the fix | `batch_verify=false` | 1493.4 / 1483.3 / 1455.1 | - |
| mbs 24, seq 1024, after the fix | `batch_verify=true factored_spare=true` | 1226.4 / 1265.2 / 1237.8 | 410 steps, 8030 drafted, 60.2 % accept, 33.85 tok/verify, 26.12 ms |

| finding | number | consequence |
|---|---|---|
| draft KV at 24-32 slots | `kv_cap=10922 x 24 slot(s)`; at mbs 32 / seq 4096 the plan cuts KV 8192 -> 544 blocks and clamps 32 -> 30 | the draft pool is not priced before the KV pool; the geometry above leaves room for both |
| step price | 26.12 ms for 48 rows vs ~16.3 ms for a 24-row decode step (1.60x) | break-even needs 1.60 x 24 = 38.4 tokens per step |
| step yield | 33.85 tokens: 19.6 of 24 requests draft, 60.2 % accepted | the gap is draft coverage and acceptance (78.0 % at 8 streams, stage 3), not the kernels |

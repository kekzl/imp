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
| engine wiring: factor buffer, verify emits, accept/reject bookkeeping, next-step apply | not built |
| drop the spare slots from `batch_verify_spare_slots` and the planner | not built |
| conv window: `ssm_conv_tap.cu`, stash + apply, `SsmConvTapTest` | built |
| numerics: deterministic PPL, `DegenerationTest`, state diff against the full-spare arm | not built |

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
set, the row is always defined" for the no-draft shape.

# Qwen3.8-27B-NVFP4 MTP: adaptive draft depth in the batched verify, 2026-11-04

REFUTED. Default `imp-server`, RTX 5090 (WSL2). Main 1346f2c4 (`imp:mtpdepth-main`) vs `adaptive_depth.diff.txt` on it (`imp:mtpdepth-br`).
Candidate: `speculative.mtp_adaptive_k` (AIMD, the single verify chunk's rule) in the bound batched verify: a rejection drafts 1 token next (its row a pad, the chain head pass skipped), a full accept drafts 2 again (ceiling `speculative.mtp_batch_verify_k` = 2).
Prompts: `tools/analysis/mtp_longctx_client.py` over the #3050 frozen corpus (`git archive b3c0087d docs/internals src/runtime src/compute`), greedy, 384 tokens, `ignore_eos`.

| File | What |
|---|---|
| `adaptive_depth.diff.txt` | the candidate, plus `[bv-verify]` / `[bv-step]` lines under `diagnostics.spec_trace` and the n=1 row of `spec-batch price` |
| `explore_fix_vs_ada.txt` | branch image, `speculative.mtp_adaptive_k=false` (fix) vs default (ada), spec_trace on, 1 chat 8k / 20k / 80k |
| `price_trace.txt` | `tools/analysis/mtp_bv_depth_price.py` on those traces: accept distribution per depth and per previous outcome, step price per depth, AIMD estimate |
| `ab_c1_long.txt` | A/B, 3 alternating pairs, 1 chat 8k / 20k / 80k |
| `ab_c1_reps.txt` | A/B, 2 pairs x 4 prompts (`REPS=4`) per context: each arm repeats its bytes across pairs, so the spread is per prompt |
| `ab_c2_long.txt` | A/B, 3 pairs, 2 overlapping chats 20k / 40k / 80k |
| `ab_short_c1_c4_c32.txt` | A/B, 3 pairs, `tools/analysis/mtp_concurrent_ab.sh` (6 chats x 400, c4, c32) |

## Price (fixed depth 2, lone request)

| ctx | accept m0 / m1 / m2 | m2 after full / part / zero | depth 1 saves (fwd + head) | fixed vs AIMD estimate at 2.0 ms saved |
|---|---|---|---|---|
| 7696 | 0.146 / 0.185 / 0.669 | 0.752 / 0.519 / 0.455 | 0.15 + 1.45 = 1.60 ms of 15.76 | 159.63 -> 156.84 tok/s |
| 19773 | 0.153 / 0.153 / 0.693 | 0.750 / 0.565 / 0.545 | 0.33 + 1.77 = 2.10 ms of 16.89 | 150.05 -> 146.17 |
| 80043 | 0.135 / 0.155 / 0.709 | 0.752 / 0.727 / 0.450 | 0.31 + 2.00 = 2.31 ms of 17.60 | 146.15 -> 142.55 |

A second draft after a rejection is still accepted 45-73 % of the time: one token (6.2-6.8 ms of step) against 1.6-2.3 ms saved. Break-even: 3.0 ms saved per depth-1 step at 8k (159.57 vs 159.63), above 3.0 ms at 20k / 80k.

## A/B (main -> branch)

| Regime | main | branch |
|---|---|---|
| 1 chat 8k / 20k / 80k, 4 prompts per context x 2 pairs (mean decode tok/s) | 166.68 / 147.16 / 145.60 | 165.04 / 143.48 / 139.50 |
| same, the #3050 prompt, 3 pairs | 159.78 / 148.49 / 145.75 | 167.76 / 143.25 / 130.99 |
| 2 overlapping chats 20k / 40k / 80k, per stream | 138.80 / 130.17 / 124.95 | 131.45 / 125.99 / 121.54 |
| same, both streams (AGG) | 267.68 / 253.15 / 245.00 | 260.68 / 246.10 / 238.08 |
| 6 chats x 400, c1 | 154.61-155.07 | 148.07-148.38 |
| c4 / c32 | 550.9-561.6 / 2024.5-2036.5 | 485.8-540.3 / 2020.3-2039.5 |

- Branch wins 3 of the 12 prompt x context cells (8k rep 0 = the #3050 prompt, 20k rep 1, 80k rep 2); c32 is neutral (pairs -14.2 / +1.8 / +15.0 tok/s); every other row loses in every pair.
- A pad instead of a draft row changes the greedy text (mechanism not bisected): `fix` = main byte for byte (same md5 per prompt), the branch differs per prompt, so the A/B compares trajectories; hence 4 prompts per context.

## Next candidates (measured here, not built)

| Candidate | Measured |
|---|---|
| Lone head pass | 3.05-4.05 ms of a 15.3-17.6 ms step (8k -> 80k); the chain alone 1.45-2.00 ms for one row |
| Depth 3 after a full accept | m2 after full 0.75 at every context; needs a third spare recurrent slot and snapshot |

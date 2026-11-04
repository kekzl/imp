# Qwen3.8-27B-NVFP4 MTP at long context, 2026-11-04

Default `imp-server`, RTX 5090 (WSL2). Main b3c0087d (`imp:mtplong-main`) vs this branch (`imp:mtplong-br`, tree 17c80274).
Prompts: `tools/analysis/mtp_longctx_client.py` over a frozen corpus (`git archive b3c0087d docs/internals src/runtime src/compute`), greedy, 384 tokens, `ignore_eos`.

| File | What |
|---|---|
| `step1_main_baseline.txt` | main: drafting stops above 16384 (`prompt exceeds the head KV capacity`), 4 x 16384-row slots |
| `window_arms_c1.txt` | head KV ring 16384 / 4096 vs full attention (131072 rows, 1 slot) on the single verify chunk; corpus not frozen yet (prompts 7705-79671) |
| `nsys_ring_classic_verify_20k.txt` | last 2.5 s of decode at 20k on the single verify chunk: `fmha_sm120_fa2_kernel` 41.6 %, 0.99 ms per call (16 per verify) |
| `solo_vs_chunk_explore.txt` | dev build, lone request: batched verify (`solo1`) vs the single verify chunk (`solo0`), 1k-40k |
| `identity_vs_mtp_off.txt`, `divergence_vs_mtp_off.txt` | dev build, `solo1` / `solo0` (= main's verify path) / MTP off (`off`): common greedy prefix vs MTP off |
| `ab_c1_long.txt` | A/B, 3 alternating pairs, 1 chat at 8k / 20k / 40k / 80k |
| `ab_c2_long.txt` | A/B, 3 pairs, 2 overlapping chats at 20k / 40k / 80k (`AGG` = both streams' tokens over their delta span) |
| `ab_short_c1_c4_c32.txt` | A/B, 3 pairs, `tools/analysis/mtp_concurrent_ab.sh` (6 chat prompts x 400, c4, c32) |
| `gates.txt` | `degen_suite.py --quick`, `niah_check.py` 32k / 64k / 100k on the branch |
| `identity_short_main_vs_solo_off.txt` | main vs branch with `speculative.mtp_batch_verify_solo=false`, 1k / 4k / 8k |

## Refuted on the way

| Candidate | Measured | Verdict |
|---|---|---|
| Head KV sized to max_seq_len per slot | 4 slots x 131072 rows = 2 GiB; the multi-slot scan holds scores in smem, 25k rows max | not needed: ring 16384 accepts 222 / 276 at 40k, full attention 222 / 277 |
| Ring 4096 | 20k / 40k 209 / 284, 215 / 282 accepted vs 221 / 276, 222 / 276 at 16384 | ring 16384 kept (same 256 MiB as main) |
| Ring alone (single verify chunk) | 20k / 40k 66.48 / 68.14 tok/s vs 93.84 / 92.54 drafting off | verify chunk attention walks the context per CTA; drafting past 16k loses without the batched verify |

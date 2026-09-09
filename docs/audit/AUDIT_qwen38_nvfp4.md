# AUDIT: Qwen3.8-27B-NVFP4-vllm, harden the checkpoint that already runs

Record, append-only. Started 2026-09-09. Base commit `0da8d55e` (main `8456782a` + #clamp commit `2ed5ef15`). Scouts read-only; every phase ends in a reproduced defect or a written "attack run, nothing found".

## 0. Regression floor (measured 2026-09-09 09:27-09:32, image imp:test from 0da8d55e, host quiet except CPU builds)

| Metric | Command | Value | Dispatch floor |
|---|---|---|---|
| ppl | `imp-cli --perplexity tools/analysis/ppl_corpus_45k.txt --set runtime.deterministic=true --set speculative.mtp_k=0 --set speculative.ngram=false` | 4.6242 (n=13811, mean_nll 1.5313) | 4.6158 (older corpus revision: file dated 2026-09-06, 13811 tokens vs 13537 in the skill text; 4.6158 is not reproducible on the current file, floor fixed at 4.6242) |
| degen suite | `tools/analysis/degen_suite.py` vs imp-server (`runtime.deterministic=true`) | 50 checks, 0 FAIL, 0 skipped (50 s) | 45/45 (suite grew) |
| 74-turn replay, run 1 (after degen suite, same server) | `multiturn_deep.py --filler 60 --max-tokens 260 / 600` | 74/74 clean at both | doc: several empty at 260 |
| 74-turn replay, run 2 (fresh server per sweep, `runtime.deterministic=true`) | max_tokens 120 / 150 / 200 / 260 | 5 / 1 / 0 / 3 FAIL of 74 | reproduced: 150 turn 5 empty content finish=length reasoning_chars=586 completion_tok=150 (budget spent in reasoning, P3 DEFECT-4); 120 turn 18 empty content finish=stop completion_tok=64 (in-think stop suppression grace path); 120 turns 26/42/58/68 early abort 5-8 words |
| tg128 / pp512 | `imp-cli --bench --bench-pp 512 --bench-reps 3 --max-tokens 128 --set speculative.ngram=false --set speculative.mtp_k=0`, 3 process trials, host load 0.51 | tg128 92.23 / 91.95 / 93.31 (median 92.23), pp512 10495 / 10423 / 10223 (median 10423) | >= 87 / >= 7500; MTP auto arm 93.09 (head not taken: the CLI bench resolves max_batch_size > 1) |
| weights | server log | approx_weights 18.8 GB, 18.36 GiB advised away after upload | <= 18.0 GiB |

Server-side resolved config (auto): max_batch_size 28, KV NVFP4 4596 blocks = 73 536 tokens = 1292.6 MiB (16/64 attn layers, kv_heads 4, head_dim 256, block 16), pool copy bandwidth 1570 GB/s (resident), SSM state 48 x 28 = 2226 MiB (BF16), RecurrentSnapshotStore 3/3 device slots x 79.5 MiB + 25 host slots, max_seq_len auto 131072 (model 262144, auto_cap), prefill graph OFF (NVFP4 KV append host sync + 2425 MiB lm_head > 512 MiB dequant cap), MTP head present but not loaded (server: max_batch_size > 1).

Stderr gate hits on the baseline (`falling back`): 10 x `gemm: cublasLtMatmul failed (status 7) M=32 K=5120 N=48 after algo reselect, falling back to cublasGemmEx` (linear_attn.in_proj_a/b, N=48, batched M=32 warmup). Logged, not silent; costed under P4.

## P7 finding 1: greedy output depends on server history (baseline image, `runtime.deterministic=true`, temperature 0, one 97-token prompt, max_tokens 260)

| Arm (fresh server each) | Requests | completion_tokens / reasoning chars / content prefix |
|---|---|---|
| default, prompt cold x2, full degen suite, prompt x2 | 4 | 112/269 "Confirmed. I have:" all four identical |
| ngram off, same order | 4 | identical to above |
| prefix cache off, same order | 4 | identical to above |
| default, full degen suite FIRST, then prompt x2 (repro_run1) | 2 | 249/742 "Confirmed. I have " both |
| prefix cache off, suite first, prompt x2 | 2 | 249/742 both (prefix cache is not the cause) |
| ngram off, suite first, prompt x2 | 2 | 112/188 "Got it. Here are t" then 226/504 same prefix: two consecutive identical requests differ |

| N-run probe (6 x A with logprobs after the full suite) | default / graphs never / ngram off + graphs never | 226/504 in all 18 runs (per-server stable, differs from the 249 plain run) |
| bisect: suite halves (repetition,think-leak,special-tokens,adherence,long-context) and (kv-growth,multi-turn,stream,constrained,anthropic-thinking) | each half then A plain / A logprobs / A plain | both halves: 249 / 226 / 249; no suite: 112 / 112 / 112 |
| one unrelated short request B (64 tok) before A, arms default / prefill_batch off / smallm off / chunkpar off / graphs never | B, A, A, B(260), A | A = 112 in all 15 asks; B(260) = 106 tok with graphs, 105 with graphs never |

Verdict: DEFECT, determinism claim (docs/determinism.md: byte-deterministic at temp 0 with runtime.deterministic=true) does not hold across server history; prefix cache excluded, n-gram speculation changes the outcome but does not explain it. Baseline stderr shows `cublasLtMatmul failed (status 7) M=32 K=5120 N=48 after algo reselect, falling back to cublasGemmEx` x10 per start (in_proj_a/b, N=48). Next: N-run divergence probe with `runtime.cuda_graphs=never` and first-divergent-token position.

## P3 correction (2026-09-09, baseline server)

| Probe | Result |
|---|---|
| single request, max_tokens 64, "Explain why the sky is blue" | finish=length, reasoning 126 chars (32 tokens = the 0.5 budget), content 186 chars: the forced `</think>` fires on the server |
| single request, max_tokens 16, 3581-token prompt | reasoning 32 chars (8 tokens), content 25 chars |
| 74-turn replay max_tokens 150, turn 5 | finish=length, reasoning 586 chars, content empty, completion 150 |

Reading (validator P3): tools/imp-server/handlers_chat_core.cpp already sets `started_in_think = ctx.snap.enable_thinking` on main; the engine-side seed (scout DEFECT-4) only covers the library/CLI path. The multi-turn empty answers have another cause (hypothesis: the model re-opens `<think>` after the forced close; last-close-wins split leaves content empty). Raw replay (mt5_dump.py, deterministic server, 74 turns with filler 60, plain requests; dumps mt5_150_plain.json / mt5_120_plain.json):

| max_tokens | empty turns | shape |
|---|---|---|
| 150 | turn 38 (topic) | finish=stop, completion 13, reasoning = `Human: Write a Python function that returns the nth Fibonacci number.` (turn 4's user prompt regurgitated inside think, then stop): context/state class, prefix-cache suspect |
| 120 | turns 18, 24, 50 (topic) | finish=stop, completion 64 = 60 (0.5 budget) + 4, reasoning cut mid-sentence at the forced `</think>`, then EOS within 4 tokens: forced close followed by an immediate stop, no answer |
| 150 / 120 with `logprobs:true` (14 base turns) | none | every turn answered, budget visibly engaged |

Same replay with `server.prefix_cache=false`: 150 -> 0 empty turns; 120 -> 1 empty (turn 37, class 1 shape: 64 = 60 + 4, forced close then EOS). Trajectories differ from turn 5 on between cache on/off (chunk-split numerics, documented in tests/test_prefix_cache_e2e.cpp), so the turn-38 regurgitation is not yet attributed; direct replay of turn 38 cold / after turn 37 / after turns 30-37 queued.

P2 finding (test-driven, small hybrid Qwen3.5-4B-mxfp4): `snapshot_boundary` saved the recurrent snapshot at the full length of a block-aligned prompt while `hybrid_prefix_reuse_limit_` caps reuse at (n-1)/bs blocks, so every block-aligned prompt got zero reuse (warm cached_tokens 0 at 512 tokens). Fixed in src/runtime/snapshot_boundary.h: boundary = ((n-1)/bs)*bs, CPU test pins 512 -> 496, 256 -> 0, 257 -> 256. The SWA twin (`maybe_save_swa_snapshot_span_`) still floors the full span: gap, not fixed here. After the fix (GPU lane, Qwen3.5-4B-mxfp4): HybridSnapshotRestoreMatchesFresh OK (aligned 512 -> cached 496, unaligned, warm+40 arms token-identical to cold), GdnGraphBucketTest 2/2 OK with the permutation oracle (n=2,4,8 distinct prompts, reversed slot order). The original oracle "bucket n == n=1" fails at token 16 for n=4 and n=8 on that model: batch-shape numerics, rows inside a bucket agree.

Direct replay of the turn-38 message list (dump contents as history) on a fresh server: cold, after turn 37, and turns 30-38 in sequence with cache hits (cached 2384..3056) all answer normally (125 tokens, "A circuit breaker is..."). Turns 1-40 in sequence on a fresh baseline server (dump contents as history): turn 38 REPRODUCED, `cached=2976`, reasoning `Human: Write a Python function that returns the nth Fibonacci number.`, content empty, finish=stop. In the direct replays the restore was at 3056 (a snapshot of turn 38's own prompt existed) and the answer was sane. Shape: restore from turn 37's snapshot (2983-token prompt -> boundary 2976) plus a 93-token continuation prefill = the "snapshot shorter than the KV match" arm. Class: confident garbage after a recurrent restore, exactly the P2 fear. Same sequence on the P2-branch image (84555ff7): turn 38 answers normally (cached=2976, 125 tokens); the P2 image resolves a different KV pool (growable 3561 blocks, ceiling 13258) than the baseline (4596). Class-1 empties move to turns 18 and 24 (ct=79 = 75 budget + 4, finish=stop). Baseline arms gdn.chunkpar_scan=false, runtime.prefill_batch=false, kv_cache.max_blocks=3561, server.recurrent_snapshot_mb=0: turn 38 sane in all four, each with a different reasoning length (169 / 346 / 348 / 358 chars): different trajectories, no attribution from the knobs.

Top-5 logprobs at turn 38 on the default baseline (sequence 1-37 first, restore at 2976): pos 0 chosen `<|endoftext|>` -0.448 (64 %), then `The` -1.748, `I` -2.475, `</think>` -3.733; pos 1 `Human` -0.710; pos 2 `:` -0.176. Mechanism (src/runtime/engine_sampling_stop.cpp ~100-115): a stop token inside the think block is suppressed as an "implicit </think>", the EOS token stays in the context, generation continues and the model starts a new document (`Human: Write a Python function ...` = turn 4's prompt), then stops: empty content, finish=stop, no signal. `<|endoftext|>` 248044 is an EOS in generation_config.json (eos_token_id [248046, 248044]) and is on the banned-token list at warmup, yet it is chosen. Class-1 empties (forced close, EOS within 4 tokens) share the "EOS right after a (forced) close" shape. Fix direction (P3 follow-up): mask stop tokens inside the think block and for the grace window after a forced close instead of suppress-and-continue; the budget guarantees termination. Cold turn 38 (no history, full prefill) pos 0: `The` -0.208, `We` -2.037, `I` -3.008, `User` -6.227; `<|endoftext|>` below -6.3. After the 1-37 chain with restore at 2976: `<|endoftext|>` -0.448. Not a rounding flip: the state after 37 chained restore-and-continue prefills differs grossly from a cold prefill of the same 3069 tokens. HybridSnapshotRestoreMatchesFresh (one restore + 100 tokens) is token-identical on the 27B, so a single restore is exact; the chain is the suspect (BF16 state, chunk splits). Oracle P(`<|endoftext|>` at pos 0 of turn 38) after the 1-37 chain, baseline image:

| Arm | pos 0 top-1 | `<|endoftext|>` logprob | turn 38 |
|---|---|---|---|
| default (BF16 GDN state, chained restores) | `<|endoftext|>` -0.448 | -0.448 | `Human: ...`, empty |
| kv_cache.max_blocks=3561 (same chain) | `<|endoftext|>` -0.448 | -0.448 | identical garbage (earlier run of this arm answered: cross-instance nondeterminism, see P7) |
| server.recurrent_snapshot_mb=0 (no restores, full prefill each turn) | `The` -0.157 | < -5.7 | sane |
| server.prefix_cache=false | `The` -0.157 (identical to the row above) | < -5.7 | sane |
| gdn.state_bf16=false (chained restores, FP32 state) | `The` -0.130 | < -5.0 | sane |

Verdict: DEFECT (P2, S0 class). Chained hybrid prefix restores with the default BF16 recurrent state corrupt the state over a multi-turn session (37 turns here): the model puts 64 % on EOS at the start of the think block and regurgitates an earlier user turn. One restore is exact (HybridSnapshotRestoreMatchesFresh on the 27B), the chain is not. With FP32 state the chain is healthy. Second arm set (same chain, BF16 state): gdn.chunkpar_scan=false -> `The` -0.188 healthy; runtime.prefill_batch=false -> `The` -0.026 healthy; runtime.cuda_graphs=never -> `<|endoftext|>` -0.448 identical corruption. So the corruption needs BF16 state AND the chunk-parallel scan AND the ragged (batched) prefill path; graphs are not involved and the reproduction is bit-stable. FP32 state, 74-turn replay at max_tokens 120: 2 empties, both class 1 (forced close then EOS), no regurgitation. Hypothesis: the ragged + chunkpar continuation path mishandles the BF16 state at a non-zero offset (dtype of the initial-state load or the final store), a per-turn error that compounds over the chain; one restore + 100 tokens stays token-identical (test), so the per-step error is small.

P3 finding (builder fix round): `engine_graph_decode.cpp` computed the graph-path think limit from the fraction only (max_tokens 4096: 2048 vs the host rule's 3072); one `think_logic::think_limit()` now feeds both paths.

### P2b verdict (builder, commit 3e0ae96c on the P2 branch, clean card): the BF16-chain hypothesis is REFUTED

| Measurement | Result |
|---|---|
| turns 1-37 replayed in order at HEAD 3e0ae96c, turn 38 pos 0 | BF16 chain `The` -0.023; FP32 chain `The` -0.112; prefix cache off `The` -0.157; 74/74 clean at 600 on both dtypes |
| `HybridRestoreChainStateStaysClose` (30 turns, 28 restores, 3299 tokens, h-state rel-L2 vs cold) | 4B: chain@30 0.218 vs equally-chunked cold 0.224; 27B: 0.337 vs 0.329; dtype does not move it; cold-vs-cold 0.0 |
| cold prefill, prefix cache off, only `runtime.prefill_chunk_size` varied (27B, 3069 tokens) | 2048 -> 1024 alone: 0.338 rel-L2 and a greedy flip; flat 0.33-0.35 down to chunk 32; turn-38 pos 0: chunk 2048 `The` -0.043, 128 `The` -0.261, 112 `The` -0.448, 96 `</think>` -0.923, 64 `The` -0.009 |

Reading: the `<|endoftext|>` -0.448 run was measured on the baseline image (snapshot at the full aligned length); 84555ff7 moved the restore point and the chain at HEAD is healthy. The 27B sits at a near-tie between opening and closing the think block at ~3000 tokens of history; any chunk shape (restore point, chunk size) flips it. This is the P7 class (greedy depends on chunk shape), not a state defect. Recorded REFUTED in docs/audit/SETTLED.md. Also found and fixed by the builder: `imp_context_reset()` evicts the prefix cache only when a request is active (src/api/imp_api.cpp), so the E2E test's cold arms were not cold (two restores compared); `go_cold()` now drops blocks and snapshots, and arm C pins the chunk size to the restore point (without the pin 16/16 tokens differ on Qwen3.5-4B). Branch hygiene job was red since de654d60 (absolute path in tests/test_memory_plan.cpp), fixed.

Still open from P3: an in-think EOS is suppressed and kept in the context (engine_sampling_stop.cpp ~100-115); when the near-tie lands on `<|endoftext|>` the model continues as a new document. Masking stop tokens inside the think block, and after a forced close until content appears, remains the fix direction.

## P6 calibration (P6 worktree f4aac640, GPU run 2026-09-09)

| Step | Command / result |
|---|---|
| calibration | `imp-cli --model Qwen3.8-27B-NVFP4-vllm --perplexity calib_corpus.txt (150 000 bytes, 37 235 samples) --calibrate qwen38-27b-calib.bin`: 496 entries, 13.6 MB, 34 s |
| export BD | `imp-quantize --model Qwen3.8-27B (BF16) --out Qwen3.8-27B-nvfp4-awq-bd --format vllm --calib ... --calib-groups BD`: 9 min 53 s, 20 GB; `AWQ: 128 groups scaled, 0 kept RTN, 128 disabled (BD), 3830 norm channels clamped`; quant_report.json: worst 0.1350 max-rel, mean 0.0600, mean MSE 2.865e-06 over 496 tensors |
| ppl of the export | 4.6286 (n=13811, deterministic) vs RTN 4.6242: +0.10 %, no win |

| export BDEG | 10 min 8 s; 224 groups scaled, 32 disabled, 3838 channels clamped | ppl 4.6136 (-0.23 %) |
| export ABCD | 9 min 40 s; 160 groups scaled, 96 disabled, 3830 clamped | ppl 4.5986 (-0.55 %) |

Conclusion: the model card's reason (unit-offset norm) was wrong; a calibrated export is REACHABLE (allowlist + layer-prefix blockers removed, offset-aware fold with the BF16 clamp) and it wins: ABCD -0.55 % PPL against RTN on the deterministic 45k corpus (4.5986 vs 4.6242), BDEG -0.23 %, BD +0.10 %. ABCDEG (all six groups): 12 min 51 s, 256 groups scaled, worst 0.1307 max-rel, mean 0.0496, ppl 4.5995 (-0.53 %). Best: ABCD. ABCD export checks (imp:baseline): degen suite 50/50, 74-turn replay 260 and 600 both 74/74, bench tg128 90.96 / pp512 8965 INVALID: a second 27B server (the P2b builder's test server) shared the card during this bench (32 039 / 32 607 MiB used, WDDM oversubscription); the degen suite and replay results are correctness-only and stand, the perf number must be re-measured alone on the card. Not yet published; the model card update is P8.

## Phase log

(one entry per phase: reproduction or "attack run, nothing found", commit, floor deltas)

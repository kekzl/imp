# KV eviction (roadmap row 48): TriAttention vs the H2O memo

Roadmap row 48 "more streams fit by evicting KV". Reference: [TensorRT-LLM #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957) (merged 2026-08-04), paper [arXiv 2604.04921](https://arxiv.org/abs/2604.04921). Order: accuracy proof first, kernels only after it passes.

## 1. Why the H2O memo shelved it (2026-05-17)

Memo: `docs/archive/plans-closed/k5_h2o_eviction_design_2026_05_17.md` (removed from HEAD in #698, `git show d3f4eb9e^:docs/archive/plans-closed/k5_h2o_eviction_design_2026_05_17.md`).

| Reason | Memo statement |
|---|---|
| No VRAM pressure | NVFP4 KV 3.9x (Qwen3-8B 140 -> 36 KiB/token) already opens 32k-80k on every dense model |
| Better alternative | K8 CPU offload keeps every token, cost is PCIe only |
| Retrieval loss | H2O evicts the needle before the query arrives: RULER NIAH 32k -20 %, multi-hop QA -20 % at 5 % retention |
| Cost | multi-week, plus no NIAH/RULER harness in the repo (the long pole) |
| Re-eval triggers | >= 100k context that NVFP4 KV cannot open; or a successor at <= 1 % RULER 32k loss without imp-side retraining; or a workload that does not need retrieval |

## 2. What TriAttention changes

| Item | H2O (memo) | TriAttention (paper + TRT-LLM #16957) |
|---|---|---|
| Score source | accumulated post-softmax attention mass | no attention read: pre-RoPE Q/K concentration around per-head centres |
| Score formula | sum of past probabilities | `S(k, D) = sum_f |E q_f| |k_f| cos(w_f D + phi_f)` (trig distance preference, Eq. 6) `+ sum_f (E|q_f| - |E q_f|) |k_f|` (norm, Eq. 8), averaged over future offsets `D + {1, 2, 4, ..., 2^16}` (Eq. 11) |
| Query statistics | none | offline per-head calibration (`calibrate.py`, 32k text); TRT-LLM loads the `.pt` |
| Head handling | per head | z-normalize per query head, max over the GQA group (Eq. 12-13); TRT modes `union` (one set, default), `per_head`, `per_layer_perhead` |
| What is kept | top-K heavy hitters + recent | top `budget` (default 2048) of the decode region; TRT keeps the whole prompt on top |
| Cadence | every step | every `beta` = 128 generated tokens (TRT default) |
| Paged blocks | n/a | token-level selection, K/V compacted in place, tail pages released |
| Prefix cache | n/a | TRT preserves the committed prompt prefix, compacts only the generated suffix |
| Claims | MMLU/XSum only | AIME25 Qwen3-8B 32.9 vs full 40.8 at budget 2048 (40.8 at 3072); LongBench 50 % KV 48.1 vs full 47.2; RULER 4k 66.1 (no full-attention number); no NIAH in the paper |

Consequence: TRT-LLM targets long generations (reasoning), not long prompts. At a fixed prompt context it frees nothing, because the prompt is kept. Row 48 asks for streams at a fixed context, so imp would have to evict prompt tokens, which is the regime the paper does not measure.

## 3. Where imp would hook it

| Piece | imp place | Decision |
|---|---|---|
| Granularity | `kKVBlockSize` 16, one block table per sequence for all layers (`src/memory/kv_cache_manager.h`) | whole blocks, one set per sequence (TRT `union` mode); per-layer sets do not free a block |
| Freeing | `KVCacheManager::evict_blocks` (new), same -1 hole scheme as StreamingLLM `evict_middle_blocks` | physical: blocks return to the pool, table length and logical positions unchanged |
| Attention | `kv_evict_compact_tables` per decode attention call (`src/exec/kv_evict.cu`) | every row reads a compacted table, `ctx' = ctx - holes * 16`; no paged kernel changes (the FP8 tile kernels clamp a -1 to block 0, they never skip it) |
| Score | `kv_evict_score_pass` on the paged keys (F16 / FP8 / NVFP4) | post-RoPE keys: `S_trig(k, D) = rot(E q, t + d) . k_post`, so the offset mean folds into one vector `u` per head; the norm term uses `|k_f|` (rotation invariant) |
| Query centre | `kv_evict_qstats_accumulate` in the single-sequence prefill attention | self-calibration from the request's own prompt queries (RoPE undone), instead of an offline `.pt`; count must equal the prompt length or nothing is evicted |
| When | after the last prefill chunk (`Engine::kv_evict_after_prefill_`) | prompt eviction (SnapKV placement), no decode-time periodic eviction in the harness |
| Prefix cache | evicted and later blocks leave the hash table; shared / pinned blocks are never evicted | compatible, eviction stops at shared prefix blocks |
| Excluded | hybrid (GDN, Mamba), SWA, MLA, M-RoPE / YaRN / partial RoPE, INT8/INT4/MXFP4 KV, speculation (verify chunks may take the prefill path), ragged prefill, sparse decode attention, residual KV, StreamingLLM | refused at init with a WARN, key reset to 0 |

## 4. Accuracy harness (step 2)

| Item | Setting |
|---|---|
| Key | `kv_cache.evict_keep_ratio` (0 = off), `evict_sink_tokens` 64, `evict_recent_tokens` 256, `evict_min_tokens` 4096 |
| Model | Qwen3-8B-NVFP4-cortecs (36 layers, 32 Q / 8 KV heads, hd 128, F16 KV, 40960 positions) |
| Probes | `tools/analysis/kv_evict_niah.py`: NIAH 5 depths x 4 seeds, continuation recall (quote the next sentence, word F1) 5 depths x 4 seeds, per length 8k / 16k / 32k, text = Einstein "Relativity" (`tools/eval/niah/data/filler.txt`) |
| Arms | `tools/analysis/kv_evict_ab.sh`: keep 1.0 (off), 0.5, 0.25; prefix cache, ragged prefill, prefill graphs, speculation, sparse attention off in all arms |
| Acceptance | NIAH hits equal to baseline at 0.5 on every length, continuation F1 drop <= 2 % |

## 5. Results (2026-11-01, RTX 5090, build-dev of this branch)

Positive control: 120 `kv evict: req` lines per eviction arm, 0 statistics mismatches; 32k probe (30281 prompt tokens) freed 946 / 1893 blocks at 0.5, 1419 / 1893 at 0.25. Max prompt tokens per length: 7877 / 15421 / 30701.

| length | NIAH off | NIAH 0.5 | NIAH 0.25 | cont F1 off | cont F1 0.5 | cont F1 0.25 |
|---|---|---|---|---|---|---|
| 8k | 20/20 | 20/20 | 15/20 | 0.3028 | 0.3424 (+13.1 %) | 0.2496 (-17.6 %) |
| 16k | 20/20 | 20/20 | 17/20 | 0.4554 | 0.3107 (-31.8 %) | 0.2467 (-45.8 %) |
| 32k | 20/20 | 20/20 | 19/20 | 0.2646 | 0.2554 (-3.5 %) | 0.2217 (-16.2 %) |
| all | 60/60 | 60/60 | 51/60 | 0.3410 | 0.3029 (-11.2 %) | 0.2393 (-29.8 %) |

Paired continuation probes at 0.5 (worse / better / same vs off): 8k 8 / 9 / 3, 16k 12 / 6 / 2, 32k 7 / 6 / 7. Off-arm F1 0.26-0.46: the metric is noisy, the sign test alone does not separate 0.5 from off; the gate is a mean bound and it is missed by 5.6x.

Default path: off-key build equals `main` c97c51fe byte for byte (6 greedy prompts, 9-151 completion tokens, `runtime.deterministic=true`); degen_suite --quick 52 checks, 0 FAIL.

## 6. Verdict: REFUTED (gate missed on the second metric)

| Gate | Result |
|---|---|
| NIAH at 0.5 equal to off, every length | met: 20/20 x 3 |
| continuation recall drop <= 2 % at 0.5 | missed: -11.2 % pooled, -31.8 % at 16k |
| Step 3 (capacity, 32-stream tok/s) | not run: the gate stops it |

What would have to change to reopen row 48:

| Change | Why |
|---|---|
| A lower-noise second metric (teacher-forced decode NLL on the continuation) at <= 2 % | the F1 probe has 0.26-0.46 off-arm scores; a real loss and probe noise are not separable at n = 20 per length |
| Offline calibration (TriAttention `.pt`) instead of the prompt's own queries | the harness centres on the prompt's mean query; the paper's centres come from 32k of calibration text |
| Generation-side eviction (TRT-LLM keeps the prompt, compacts every 128 generated tokens) | the regime the paper measures (AIME, 32k generations); frees nothing at a fixed prompt context, so it does not serve row 48's goal |

## ROADMAP CLOSED (2026-11-01)

Row 48 closed REFUTED. The opt-in harness stays (`kv_cache.evict_keep_ratio`, default 0) for a reopen measurement.

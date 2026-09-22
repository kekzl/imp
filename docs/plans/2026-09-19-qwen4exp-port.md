# Qwen3.8-Flash-Next (qwen4_exp) port

Record, 2026-09-19. Branch `feat/qwen4-exp`. The checkpoint is
`nvidia/Qwen3.8-Flash-Next-NVFP4` (123 GiB: 10 shards of 73.3 GiB for the
transformer, `model-fp8-mtp-ple.safetensors` 50 GiB for the n-gram embedding
and the MTP experts), modelopt MIXED_PRECISION: NVFP4 (group 16, `input_scale`)
on `mlp.experts` only, everything else BF16.

## What the config says (not the name)

| Field | Value |
|---|---|
| `model_type` / class | `qwen4_exp` / `Qwen4ExpForConditionalGeneration` |
| Parameters | 125 B transformer (6 B active) + 51 B n-gram embedding + 4 B MTP |
| Layers | 48: 36 `linear_attention` (Gated DeltaNet) + 12 `full_attention` (QSA) at 3, 7, ..., 47 |
| GDN | 48 V heads, 16 QK heads, head dim 128, conv 4; `output_gate_type=sigmoid` |
| Attention | 24 Q heads, 2 KV heads, head_dim 256, rotary 64, gated q_proj `[12288, 2560]` |
| MoE | 512 experts x 10 + 1 shared, intermediate 640, top-k softmax, `norm_topk_prob` |
| Gated residual | `hc_count=4`, `hc_lowrank=320` |
| PLE | `ple_layer_ids=[2]`, checked as `layer_idx + 1`: the block sits at layer 1 |
| Indexer | 4 Q heads, 1 K head, head dim 128, budget 2048 tokens = 512 blocks of 4 |

## Blocks, status

| Block | Tensors | Status |
|---|---|---|
| Registry, detection, template family, GDN head layout, `(1 + W)` norm offset, KV allowlists | | done, `165e45f2` |
| GDN | `linear_attn.*` | resolves through the Qwen3.5/3.6 names; sigmoid gate added |
| MoE experts, 512 x 3 x 48 | modelopt layout, same as `Qwen3-30B-A3B-NVFP4-Modelopt` | resolves; routing needed the >= 256-expert fix (see "Reference comparison"); 56 GiB stay host-resident (`moe.force_host_experts=48`, LRU 48 x 67 slots x 0.88 MiB) |
| Gated residual | `attn_/mlp_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}`, `hyper_connection_mixer.*` | loader, kernels (`compute/gated_residual.cu`), forward wiring (`exec/executor_gated_residual.cu`), BF16 upload: done, not yet run end to end |
| PLE (layer 1) | `ple.{key_proj, value_proj, conv1d, norm_key, norm_query, norm_conv}`, `ple_embedding.{layer_multipliers [3], ngram_heads_offsets [16], ngram_heads_vocab_sizes [16], ngram_embedding.shard_0..127 F8 [2500012, 160], weight_scale BF16 [1]}` | built, runs (see "PLE, as built"); reference comparison pending; previously: not started |
| QSA indexer (12 layers) | `self_attn.indexer.{index_qk_proj [640, 2560], q_layernorm [128], k_layernorm [128]}` | built (see "QSA indexer, as built"); `attention.qsa=false` runs dense |
| MTP | `mtp.*` in the FP8 shard | not started (`speculative.mtp_k=0`) |
| VL tower | `model.visual.*` | dropped |

## Gated residual, exact

Residual stream `[T, hc*d]`, initialised as `embed.repeat(hc)`. Per block:

```
normed = grouped_rmsnorm(stream, w)            one RMS per (row, stream), out * (1 + w)
low    = silu(normed @ down^T / hc)             [T, 320]
mixw   = sigmoid(low @ up^T)                    [T, hc*d]
mixed  = mean_s(mixw[:, s] * normed[:, s])      [T, d]      -> block input, NO further pre-norm
inj    = 2 * sigmoid(normed @ inject^T / hc)    [T, hc]
stream[:, s] += block_out * inj[:, s]
```

The final `hyper_connection_mixer` is the same read without `inject`; its
`mixed` goes into the output norm and the LM head.

imp keeps the block convention (`hidden_` in, `hidden_ + out` out): `hc_read_`
writes `mixed` into `hidden_[n]`, the block's pre-norm weight is null and
`rmsnorm_for_smallm_` / `run_moe_ffn` treat that as the identity, `hc_write_`
recovers `out = hidden_ - mixed` and injects it. That subtraction is FP16; if
the layer diff against a reference shows drift, keep `mixed` and `out` in FP32.

## The bring-up crash

Every run died with an illegal memory access reported by the first GEMM of
layer 0 (`M=57 K=2560 N=10240`, `in_proj_qkv`), also under
`CUDA_LAUNCH_BLOCKING=1`. Sync checkpoints placed the fault between the
embedding ("no error") and the CUTLASS activation quant ("illegal memory
access" already set): the model has no `input_layernorm`, so `ly.attn_norm.data`
was null and `rmsnorm()` (no null guard, `pdl::launch` without a launch check)
read it. The fix is not a guard but the gated residual: the block input is the
mixed stream, the pre-norm lives in `hc_norm`.

## PLE (layer 1), exact

Per token, 16 lookups into the n-gram table: heads 0-7 are bigrams
(positions 0, 1), heads 8-15 trigrams (positions 0, 1, 2). Context is the
`ngram_size - 1 = 2` previous tokens, EOS-filled at sequence start; shifts do
not cross an EOS (`_shift_right_ignore_eos`).

```
mixed_ids = XOR_p( shift_p(tokens) * layer_multipliers[p] )        int64, p over the n-gram
id_h      = mixed_ids mod ngram_heads_vocab_sizes[h] + ngram_heads_offsets[h]
emb       = concat_h( table[id_h] )                                  [T, 16 * 160 = 2560], F8 * weight_scale
key       = grouped_rmsnorm(emb @ key_proj^T)     [T, hc, d]
value     = emb @ value_proj^T                    [T, d]
q         = grouped_rmsnorm(stream)               [T, hc, d]
gate      = sum_j(key * q) / sqrt(d)              [T, hc, 1];  gate = sqrt(|gate|) * sign(gate), |gate| >= 1e-6
gv        = sigmoid(gate) * value                 [T, hc, d]
out       = gv + silu(depthwise_conv1d(grouped_rmsnorm(gv)))   kernel 4, dilation 3, causal, state 9
stream   += out                                   before attn_hyper_connection
```

`table` is 128 shards of `[2500012, 160]` F8_E4M3 (320 M rows, 51 GiB) in the
FP8 shard: host mmap, 16 x 160 bytes gathered per token, never resident in
VRAM. The multipliers, offsets and vocab sizes come from the checkpoint; the
prime search in the modeling code is not needed.

## Reference

The n-gram table never needs to be resident: llama.cpp (PR #27742, merged
2026-08-27, arch `qwen4exp`) hashes host-side ("the splitmix64-derived
multipliers reach 2^45, so the products need 64-bit integers and an xor,
neither of which ggml has") and gathers rows with `ggml_get_rows` from the
mmap'd `per_layer_token_embd` tensor. `-ot "per_layer_token_embd=CPU"` keeps it
off the GPU; the page cache serves the hot n-grams. Reported: RTX 3080 Ti
16 GB + 64 GB RAM with experts 6-46 in RAM and the table on NVMe, pp ~40 and
tg 8-10 tok/s; M1 Max 64 GB with the table on SSD, tg 17.6 tok/s at 45.8 GB
resident. 16 rows of 160 bytes per token is ~3 MB/s of random reads at
36 tok/s, well inside one NVMe's budget.

So the oracle on this host is llama.cpp in the `ghcr.io/ggml-org/llama.cpp`
CUDA container with `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL` (104 GiB,
4 shards): experts in RAM, table on NVMe, dense part + KV on the card, and
`llama-eval-callback` for per-layer dumps against `diagnostics.dump_hidden_dir`
+ `tools/analysis/layer_diff.py`. Quantisation differs between the two, so
the comparison bounds drift rather than proving equality; token-level
agreement on greedy decode is the first gate, perplexity on
`ppl_corpus_45k.txt` the second. imp's own PLE path is the same design: host
hash, host mmap, 16 gathers per token, nothing of the table in VRAM.

## PLE, as built

| Piece | Where | Fact |
|---|---|---|
| Table + hash buffers | `src/model/ngram_table.{h,cpp}`, `Model::ngram_table()` | opened from the index by `.layers.<i>.ple.ple_embedding.*`; FP8 file mmapped `MAP_PRIVATE` without populate + `MADV_RANDOM`, 128 shards x [2500012, 160] = 320001536 rows, 47.7 GiB; `weight_scale` BF16 = 0.00019932; I64 buffers by `pread` (the standard loader drops I64 as unservable, `safetensors_loader.cpp` dtype table) |
| Loader | `llm_compressor_loader.cpp` `name_is_unused`, `safetensors_loader.cpp` 6a | every `.ple.ple_embedding.` name is unused, so the 50 GiB shard is skipped (3201 tensors) as long as MTP is off; PLE weights without a table refuse the load |
| Hash | `ngram_hash()` | history = 2 context tokens ++ chunk; shift s is EOS when any of `[pos-s, pos-1]` is EOS; `uint64` products + XOR, Python modulo; `tests/test_ngram_table.cpp` |
| Gather | `NGramTable::gather` | `MADV_WILLNEED` per row first, then F8 -> float x scale -> FP16 into pinned staging (`max_tokens x 2560 x 2` B, 10 MiB at 2048) |
| Device | `src/compute/ple.{h,cu}`, `src/exec/executor_ple.cu` | key/value GEMMs + `hc_grouped_rmsnorm` reuse; `ple_gate_value` (one block per (token, stream), in place over q); `ple_conv_add` (thread per channel, 64 rows per block) + `ple_conv_state` (9 rows carried); scratch = the free `hc_*` buffers, own VRAM only the conv state |
| Sequence state | `ple_ctx_` (2 tokens), `ple_conv_state_` | reset when the chunk's first position is 0, carried otherwise. Not modelled: batched decode (logged once), prefix-cache resume, chunked prefill across a prefix hit |
| Graphs | none needed today | the D2H of token ids sits outside capture because experts on host already demote graphs (`ExpertsOnHost`); a device-resident model needs the pinned-staging-as-graph-input pattern |
| MTP | open | `speculative.mtp_k>0` would make the FP8 shard "used" and `MAP_POPULATE` 50 GiB; the MTP milestone needs the shard split or a lazy map |

## Reference comparison (2026-09-19, llama.cpp UD-Q4_K_XL vs imp NVFP4)

Method: teacher-forced NLL on the same text, `llama-perplexity -c 256 --chunks 1`
scores positions 128..255 of the first chunk; imp `--perplexity` with
`diagnostics.ppl_dump=full` gives every position, averaged over the same window.
Both tokenize identically (748 / 1301 tokens). Control on Qwen3-30B-A3B: llama
Q4_K_M 11.48, imp on the same GGUF 11.29, imp NVFP4-Modelopt 12.78.

| Text | Window | llama.cpp | imp before | imp after |
|---|---|---|---|---|
| `novel_prose_2026.txt` (unseen prose, 748 tok) | 128..255 | 5.54 | 14.25 | 6.00 |
| `prose_5500.txt` (Gutenberg, 1301 tok) | 256..511 | 1.4238 | 4.03 | 1.422 |

The gap was NOT the port. `llama-eval-callback` per-block fingerprints (token 0
matched through layer 23, later tokens diverged from the layer-0 FFN on) led to
the router: logits matched llama.cpp, the selected experts did not. Token 1's
best expert is 338; imp picked 68. `topk_gating_kernel` gave each of its 256
threads exactly one expert (`tid < n_experts`), so experts >= 256 were never
candidates. Every previous MoE checkpoint had <= 256 experts; Qwen3.8-Flash-Next
has 512. Fix: strided slots per thread (`kTopkSlotsPerThread = 4096 / 256`),
same for the fused decode gating kernel; regression test
`MoERoutingWideTest.ExpertsAbove256AreCandidates` (fails on the old kernel).
PLE-off ablation on the unseen text: 24.8 (before the fix), so the PLE block
carries roughly half of the model's quality on plain prose.

## QSA indexer, what it does and when it matters

`Qwen4ExpTextQSAIndexer` (modeling_qwen4_exp.py:672-816): per query, the visible
tokens are cut into complete blocks of `indexer_compress_ratio=4`; each block key
is the mean of the 4 raw keys (`index_qk_proj` K half, 1 head x 128), then
`k_layernorm`, then RoPE at the block's first position; the query (4 heads x 128,
`q_layernorm`, RoPE) scores every block with `relu(q . k)` summed over heads /
sqrt(128); the top `indexer_budget / 4 = 512` blocks plus the incomplete tail
block stay visible, everything else is masked out of the ordinary attention.

`topk(min(512, num_complete_blocks))`: with at most 512 complete blocks, i.e.
`visible <= 2048 + 3` tokens, every block is selected and the mask is all-true.
Dense attention is therefore EXACT up to 2048 tokens of context and only diverges
beyond. imp's existing `sparse_attn_select.cu` is a Quest-class page heuristic on
the paged decode kernels, not this learned block selection; the indexer needs its
own key cache ([ctx, 128] per QSA layer), a block-pooling + scoring kernel and a
token-mask consumer in both the prefill FMHA and the paged decode path.

Indexer tensors (`self_attn.indexer.{index_qk_proj [640, 2560], q_layernorm [128],
k_layernorm [128]}`) now map to `qsa_index_qk / qsa_index_q_norm / qsa_index_k_norm`
and are uploaded raw (norms are `1 + w`). RoPE on the indexer is the attention's
`apply_rotary_pos_emb`: NeoX `rotate_half` over the first `rotary_dim = 0.25 x 256 = 64`
dims of each 128-dim indexer head, the other 64 untouched; the block key takes the
position of the block's first token.

Plan (milestone 4), exact at every context length:

| Step | Design |
|---|---|
| State per QSA layer | raw index keys `[max_ctx, 128]` FP16 (pre-norm, pre-RoPE) and block keys `[max_ctx/4, 128]` (mean of 4 raw keys -> `k_layernorm` -> RoPE at block start); block keys are query-independent, so they are built once per completed block |
| Per forward | `index_qk_proj` GEMM `[n, 640]`; q: `q_layernorm` + RoPE (4 heads); k: write raw keys; pool the newly completed blocks |
| Scoring | per query row: `relu(q_h . blk)` summed over the 4 heads / sqrt(128) for every complete block `< (pos+1)/4`; top-512 blocks (vocab top-k machinery in `compute/sampling_topk_topp.cu` as template) + the incomplete tail = a token index list `[n_q, 2051]`, -1 padded, ascending |
| Attention consumer | one "selected-token" attention kernel over the paged KV (F16 first) taking the index list: decode (`n_q = 1`) and prefill rows alike; rows with `pos < 2048` get `[0..pos]`, so the kernel is exact everywhere and FA2 stays the fast path for chunks entirely below 2048 |
| Gate | `attention.qsa_force=true` runs the selected path below 2048 too: must match dense FA2 (the correctness A/B); above 2048 the reference is llama.cpp's qwen4exp indexer |
| Not reused | `sparse_attn_select.cu` (Quest page heuristic, KV-page granularity `kv_cache.block_size`, not 4-token blocks) |

## QSA indexer, as built (2026-09-20)

| Piece | Where | Fact |
|---|---|---|
| Kernels | `compute/qsa_indexer.{h,cu}` | `qsa_prep_queries` (q: (1+w) norm, NeoX RoPE on 64 of 128 dims at the row's position; raw key written at the position), `qsa_pool_blocks` (mean of 4 raw keys -> fp16 -> (1+w) norm -> RoPE at 4b; decode derives the block from `positions[0]` on the device), `qsa_select` (scores: one warp per block key over two waves of CTAs; then one 1024-thread CTA per row: radix select of the k-th largest float key, compaction ascending, ties by lowest index, tail appended), `qsa_gather_kv` (selected rows from the paged cache into a scratch paged cache, row r owns 129 blocks) |
| Orchestration | `exec/executor_qsa.cu` | decode (one sequence): index GEMM -> prep -> pool -> select -> gather -> `paged_attention_decode` on the scratch, in place of the dense kernel, all device-driven so the captured step replays; prefill: dense FA2 for the chunk, then rows with position >= 2051 recomputed in passes of `attention.qsa_rows` (16); `attention.qsa_force` recomputes every row |
| State | lazily sized from the KV cache on the first forward | per QSA layer raw keys `[ctx, 128]` + block keys `[ctx/4, 128]` FP16 (16k context: 60 MiB for 12 layers), scratch K/V 16 rows x 129 blocks (64.5 MiB), scores `[16, ctx/4]` |
| Exactness | `tests/test_qsa_indexer.cu` (test-attention, 5 GPU tests) | select + gather + paged on the scratch is byte-identical to paged on the original cache (stale bytes in the scratch); select vs CPU top-k incl. ties; prep vs CPU norm+RoPE |
| In situ | `attention.qsa_debug` | layer 3, every 16-row pass: max abs diff FA2 vs paged-on-cache 0.007-0.023 (values ~3.3), paged-on-cache vs selected 0 |

Measurements (config of the speed table below, `max_seq_len` 8192):

| Probe | dense (`attention.qsa=false`) | indexer |
|---|---|---|
| PPL window 2048..4095 of `ppl_corpus_45k.txt` (imp dump indices; llama.cpp `-c 4096 --chunks 1` on the same text with its own QSA top-k, UD-Q4_K_XL: 4.5021) | 4.6978 | 4.7558 |
| PPL 0..2047 of the same run | 5.3653 | 5.3653 (identical, the selection is all-true there) |
| `attention.qsa_force` on `prose_5500.txt` (1301 tokens, all-true selection through the paged kernel) | 1.6930 | 1.7004 (+0.44 %, FA2-vs-paged numerics amplified by the MoE routing; skill band +-0.5 %) |
| Needle at 4503 tokens context (`ZEBRA-9134`, greedy) | found | found |
| Decode at 4.5k context | 54.1 tok/s | 50.6 tok/s (31.7 before the split-K scratch was handed to the selected path) |
| pp4503 | 912 tok/s | 650 tok/s (the row passes; batching more rows per pass is the lever) |

Not modelled: prefix-cache resume (a chunk starting past the valid keys makes the sequence
dense with one warning), batched decode (one sequence's key caches; the PLE context has the
same limit), non-F16 KV caches (dense). The indexer arm sits 1.2 % above dense on the 2k
window while both sit above llama.cpp; the quant gap (NVFP4 experts vs Q4_K_XL) was +8 % on
unseen prose yesterday, so the reference cannot separate indexer from quant here. What the
build proves: the kernels match the reference math, the selection reads the right bytes, and
retrieval past the budget works.

Default flipped to `attention.qsa=false` on 2026-09-21. On short prompts the decode path runs
selection + gather + the paged kernel where dense would do, and the paged-vs-FA2 re-rounding
flips MoE routing: `degen_suite.py` 48/50 with the indexer (a counting prompt derails,
"What is the capital of France" answers `7391`), 50/50 without, reproduced on three server
starts. At 4503 tokens it is also slower on both axes and 1.2 % above dense on PPL, so nothing
measured here pays for it. It earns its place above ~8k context, where the dense KV read
dominates; the kernels and the gate stay.

### The above-8k expectation, measured and refuted (2026-09-22)

The 2026-09-21 note left the default off but expected the indexer to pay above ~8k. It does
not, on this model. One server per arm, `imp:test` at 9ae7ad2a + the fixes below, greedy,
seed 42, needle recall plus one 1024-token generation from an ~11k-token prompt:

| Probe (prompt tokens) | `qsa=false` | `qsa=true` |
|---|---|---|
| needle @ 4883 | found, 3.01 tok/s | found, 2.35 |
| needle @ 11283 | found, 1.63 | found, 0.96 |
| needle @ 23084 | found, 0.79 | found, 0.50 |
| 1024 generated @ 11283 | 18.14 tok/s | 15.88 (-12.5 %) |
| `degen_suite.py` | 50/50 | 47-48/50 |

Retrieval is intact at every length; the cost is throughput. The reason is the layer budget,
not the kernels: 12 of 48 layers carry attention and each holds 2 KiB of KV per token, so the
whole dense KV read is 24 KiB/token. At 12k context that is 288 MB, ~0.19 ms at 1.5 TB/s,
against a 63 ms decode step (15.88 tok/s) whose time belongs to the host-resident experts.
The indexer replaces 0.3 % of the step and adds an index GEMM, a selection CTA per row, a
gather and a second paged launch on each of those 12 layers. `qsa_select_kernel` runs one CTA
per query row and walks `ctx/4` block keys serially on a single SM, so its cost grows with
context while the dense read it saves does not.

It also holds VRAM whether or not it engages: `qsa_ensure_ctx_` sizes the key caches from
`ceiling_blocks()`, so a 131072-token ceiling reserves 480 MiB of key caches plus 64.5 MiB of
scratch on first forward, while the KV pool itself starts at 1675 blocks and grows. On this
card that is ~13 expert-cache slots per layer.

The 2026-09-21 cause ("the paged-vs-FA2 re-rounding flips MoE routing") is not established.
Two facts against it: `attention.qsa_debug` over 88 passes reads max |paged-on-cache -
selected| of 0 or one half ULP (0.0009766 against values ~3.5), and no `degen_suite.py` prompt
reaches 2051 tokens (the long-context check tops out near 1700), so the selection there is
every token in order and never sparse. One real defect did sit under that budget and is fixed
below, but fixing it moved the suite from 48/50 to 47/50 — the residual is unexplained and
tracks the known order-dependence classes (`kv-growth`, `stream`, cf. 10f2f18b), not the
selection.

Reopen trigger: a QSA model whose attention layers carry the decode (a dense-attention
architecture, or KV per token an order up), or a selection kernel that scales across SMs.
Measured to 23k tokens only; the 131k end is untested.

### Reopened: nsys says the kernels, not the model (2026-09-22, second pass)

The "0.19 ms of a 63 ms step" above is wrong twice: the 63 ms included prefill, and the dense
decode attention measures 81.6 us per layer, not 16. `nsys --cuda-graph-trace=node`, imp-cli,
13863-token prompt (`ppl_corpus_45k.txt`), tg256, per-kernel sums:

| Kernel (13863 context) | before | after |
|---|---|---|
| dense decode attention, per layer-step | 81.6 us | unchanged (same dispatch) |
| QSA decode select, per layer-step | 128.4 us (1 CTA) | 9.1 us (score 2.6 + select 6.5) |
| QSA decode attention on 2051 tokens, per layer-step | 16.3 us | 15.1 us |
| QSA prefill attend, 16-row passes | 9233 ms, `paged_attention_gqa_kernel` 1041 us x 8868 | 1588 ms split-K, 133 us; gqa 24 calls |
| QSA prefill select | 1130 ms | 128 ms (select 67 + score 61) |

Two fixes: `qsa_select` scores with one warp per block key across the card, then selects in a
1024-thread CTA (ballot ranks, one warp finds the radix digit). `paged_attention_decode` split
on `batch * n_heads` = 384 CTAs at 16 rows and skipped split-K, but its no-split kernel for
GQA ratios above 8 (no multitok) launches `batch * n_kv_heads` = 32; it now splits on that
count when the first says "enough". Reaches F16 KV with ratio 9..16 only: Flash-Next
(24/2), Nemotron with `kv_cache.dtype=fp16` (defaults to FP8 KV, a different kernel).

e2e, imp-cli, one process per run, 13863-token prompt, tg512, two trials (tok/s):

| Arm | pp | tg |
|---|---|---|
| main ece705a4, `qsa=false` | 213.38 / 209.71 | 55.41 / 53.22 |
| main ece705a4, `qsa=true` | 185.64 / 185.21 | 57.59 / 58.31 |
| this change, `qsa=false` | 215.12 / 220.15 | 57.10 / 55.99 |
| this change, `qsa=true` | 203.65 / 197.65 | 63.84 / 62.28 |

At 13.9k the indexer now gains 11-12 % decode and costs 5-10 % prefill. Default stays
`false`: `degen_suite.py` 47-48/50 unexplained, 480 MiB of key caches reserved at a 131k
ceiling. Open: prefill passes, 16 rows each, 741 per layer on a 13.9k prompt ((11952 - 3060) / 12 select launches).

## Speed on the 32 GB card (2026-09-20)

56 GiB of experts stay on the host; every decode token streams its misses over PCIe.
Config for every row: `moe.force_host_experts=48 moe.pin_host_experts=true
moe.expert_cache_budget_pct=45 speculative.ngram=false speculative.mtp_k=0
runtime.warmup=false`, greedy, "history of Paris" prompt, tg96; the text is
byte-identical across all rows.

| Step | tg96 tok/s | What moved |
|---|---|---|
| mmap experts, 15 % budget (62 slots/layer) | 5.81 | per miss: 2 `cudaMemcpyAsync` + a 4-byte scale H2D from a stack float (pageable, syncs the stream first) |
| `pin_host_experts` | 8.10 | DMA-able source |
| scales + slot indices as kernel params | 14.35 | no pageable copy, no drain per miss |
| ngram spec off, budget 45 % (186 slots/layer, 71 % hits) | 20.6 (under nsys) | the one n-gram verify step cost 2.9 s for 4 tokens through the legacy prefill |
| one `cudaMemcpyBatchAsync` per layer | 27.2 | 31-56 us host time per memcpyAsync; the batch call issues 60 copies in 0.05 ms at 54 GB/s (`tools/analysis/h2d_gather_probe.cu`) |
| device expert cache (`exec/expert_cache_device.{h,cu}`) | 33.2 | resolve kernel (LRU tables on the device) + gather kernel from mapped pinned (51 GB/s); no D2H, no sync per layer |
| captured decode (per-step graph pool) | 53.0 (tg512: 65.9, 81.9 % hits) | ~2600 launches per token were the host-side floor; PLE host half moved to `prepare_decode_step_host` |

End to end, `imp-cli --bench --bench-pp 512 --bench-reps 2`, host experts at 45 % budget,
`imp:ab-dc05d8b8` (main before) vs `imp:test` (after), alternating A B B A:

| | before | after |
|---|---|---|
| pp512 | 395.98 / 395.05 tok/s | 859.64 / 852.90 tok/s |
| tg8192 | 67.85 / 67.85 tok/s | 102.03 / 101.37 tok/s |

Prefill from touched-only staging, decode from `attention.qsa=false`: at 8704 tokens of
context, where the indexer is fully active, it costs a third of decode throughput.

Expert cache budget is at its ceiling: 60 % raises the hit rate (98.8 -> 99.2 %) and drops
tg8192 to 78.26 tok/s. The allocation succeeds and the bytes spill (WSL2/WDDM, #1103).

The 54 vs 102 tok/s gap between a 120-token server request and tg8192 is not a cold cache,
it is the routing: `--bench` generates on a repetitive synthetic prompt and hits 98.8 %,
real text hits 62.7 % (300 tokens, `diagnostics.moe_expert_trace`).

A better eviction policy is not the lever. `tools/analysis/expert_cache_sim.py` replays the
trace; it reproduces the engine's own 62.7 % at the shipped 186 slots/layer, which is what
makes its other rows worth reading:

| slots/layer | LRU | pin hottest half, LRU rest |
|---|---|---|
| 93 | 43.5 % | 40.4 % |
| 186 (shipped) | 62.0 % | 59.4 % |
| 279 | 72.0 % | 70.7 % |
| 372 | 78.9 % | 77.9 % |
| 512 | 85.0 % | 85.3 % |

Pinning by frequency loses to plain recency at every budget the card can hold. The routing is
too flat for it: layer 0 touches 384 of 512 experts in 300 tokens, and its 64 most frequent
experts carry only 41 % of the activations. Only more slots raise the hit rate, and more slots
spill. What is left for decode is hiding the transfer behind compute, or not transferring at
all: 62.7 % hits leave ~470 MiB of misses per token over PCIe against ~5.5 ms of GPU work.

Computing those misses on the CPU instead, where the bytes already are, has far less headroom
than it sounds. `tools/analysis/cpu_expert_gemv_probe.cpp` reads random 0.88 MiB experts out
of a 4 GiB slab on a 9800X3D, 16 threads:

| | ms per token (537 misses) | GB/s |
|---|---|---|
| read only, memory-bound ceiling | 8.25 | 60.0 |
| dequant + GEMV, scalar | 28.61 | 17.3 |
| the same bytes over PCIe | 9.52 | 52.0 |

Host RAM for this access pattern is 58-60 GB/s, 13 % above the link, not the 2x the DDR5
figure suggests. So a perfect AVX-512 NVFP4 GEMV sitting exactly on the memory roofline saves
1.2 ms of 9.5 on that component; the scalar version is 3.5x off that roofline. The case for a
CPU path is overlap (the CPU's misses running while the GPU does its hits), not bandwidth, and
it has to be written against a 8.3 ms floor.

TTFT (2026-09-21, `moe.stage_touched_only`): prefill staged all 512 experts of every layer
regardless of routing, 63 GB per prefill, so TTFT was flat at ~1.27 s whatever the prompt
length. A gather kernel reading `expert_offsets` stages only the touched experts:

| prompt tokens | all experts | touched only |
|---|---|---|
| 67 | 1267 ms | 499 ms |
| 328 | 1287 ms | 811 ms |
| 1022 | 1328 ms | 1038 ms |

Decode unchanged (~54 tok/s), PPL on `ppl_corpus_45k.txt` 4.6326 vs 4.6306 (+0.04 %). The
ratio at 67 tokens says the routing is skewed: a uniform top-10 over 512 experts would touch
73 % of them, the measured cost falls to 39 %.

Prefill: `moe.staged_cutlass_prefill=true` needed the >256-expert fixes in
`compact_alpha_active`, `compute_sfa_offsets`, `build_grouped_3x_staging_kernel`;
pp62 2.49 -> 1.68 s. The PCIe floor for a full 512-expert layer set is 65 GB per
chunk at 45 GB/s = 1.4 s, minus cache hits. A 70 % cache budget over-commits VRAM
(418 MiB free against the 1629 MiB headroom, no tokens); 45 % is the measured
ceiling at `max_seq_len 4096`.

## Findings on the way

- cuBLASLt on sm_120 (`tools/analysis/cublaslt_grouped_probe.cu`, CUDA 13.4.1,
  cuBLASLt 13.7.0): FP16 plain 8 algos, FP16 grouped 0, NVFP4 plain 0,
  NVFP4 grouped 0 with `VEC16_UE4M3` block scales. CUTLASS stays the only
  NVFP4 GEMM path.
- `gpu-busy-check.sh` sampled 317 MiB in the gap between two `songforge` runs
  while that session held the card; a peer's START/ENDE outranks the sample.

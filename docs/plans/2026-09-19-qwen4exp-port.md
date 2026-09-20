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
| QSA indexer (12 layers) | `self_attn.indexer.{index_qk_proj [640, 2560], q_layernorm [128], k_layernorm [128]}` | not started; attention runs dense without it |
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

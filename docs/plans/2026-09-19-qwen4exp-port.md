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
| MoE experts, 512 x 3 x 48 | modelopt layout, same as `Qwen3-30B-A3B-NVFP4-Modelopt` | resolves; 56 GiB stay host-resident (`moe.force_host_experts=48`, LRU 48 x 67 slots x 0.88 MiB) |
| Gated residual | `attn_/mlp_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}`, `hyper_connection_mixer.*` | loader, kernels (`compute/gated_residual.cu`), forward wiring (`exec/executor_gated_residual.cu`), BF16 upload: done, not yet run end to end |
| PLE (layer 1) | `ple.{key_proj, value_proj, conv1d, norm_key, norm_query, norm_conv}`, `ple_embedding.{layer_multipliers [3], ngram_heads_offsets [16], ngram_heads_vocab_sizes [16], ngram_embedding.shard_0..127 F8 [2500012, 160], weight_scale BF16 [1]}` | not started |
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

HF transformers needs the BF16 checkpoint in RAM (335 GiB). The workable
oracle on this host is llama.cpp with `unsloth/Qwen3.8-Flash-Next-GGUF`
`UD-Q4_K_XL` (104 GiB, experts on CPU) and `llama-eval-callback` for per-layer
dumps against `diagnostics.dump_hidden_dir` + `tools/analysis/layer_diff.py`.
Quantisation differs between the two, so the comparison bounds drift rather
than proving equality; token-level agreement on greedy decode is the first
gate, perplexity on `ppl_corpus_45k.txt` the second.

## Findings on the way

- cuBLASLt on sm_120 (`tools/analysis/cublaslt_grouped_probe.cu`, CUDA 13.4.1,
  cuBLASLt 13.7.0): FP16 plain 8 algos, FP16 grouped 0, NVFP4 plain 0,
  NVFP4 grouped 0 with `VEC16_UE4M3` block scales. CUTLASS stays the only
  NVFP4 GEMM path.
- `gpu-busy-check.sh` sampled 317 MiB in the gap between two `songforge` runs
  while that session held the card; a peer's START/ENDE outranks the sample.

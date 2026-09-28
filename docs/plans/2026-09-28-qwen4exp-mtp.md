# Qwen4Exp MTP head: reference semantics (A2 step 0)

Pins the draft-layer forward of Qwen3.8-Flash-Next (`model_type` `qwen4_exp_text`) from the
upstream reference before imp implements it. Loader state: `src/model/mtp_head.h`
(`MtpLayout::Qwen4Exp`), `dispatch_mtp_head()` in `src/model/safetensors_loader.cpp`.

## Sources

| Tag | File | Revision |
| --- | --- | --- |
| T | `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` | transformers `v5.16.0` |
| V | `vllm/models/qwen4_exp/nvidia/mtp.py` | vLLM `60ad959b` (merge of PR #55513) |
| VM | `vllm/models/qwen4_exp/nvidia/model.py` | same |
| VH | `vllm/models/qwen4_exp/nvidia/hyperconnection.py` | same |
| VQ | `vllm/models/qwen4_exp/nvidia/qsa.py` | same |
| VS | `vllm/config/speculative.py` | same |
| VP | `vllm/v1/spec_decode/llm_base_proposer.py` | same |

- T has no MTP module: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` (T:1256, T:1535). The
  MTP forward exists only in V. T is cited for the shared building blocks.
- Commit `d4d703ca` is PR #54882 (FP8 PLE loading, `ple_layer.py`), not #55513. #55513 merged
  as `60ad959b` and touches `nvidia/mtp.py` (quantized_layers remap) and `modelopt.py`.

## Checkpoint (headers of Qwen3.8-Flash-Next-NVFP4)

| Group | Tensors | dtype | Shape |
| --- | --- | --- | --- |
| `mtp.fc_embedding`, `mtp.fc_hidden` | 2 | BF16 | [2560, 2560] |
| `mtp.pre_fc_norm_embedding` / `_hidden` | 2 | BF16 | [2560] / [10240] |
| `mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` | 3 | BF16 | [10240], [320, 10240], [10240, 320] |
| `mtp.layers.0.{attn,mlp}_hyper_connection.*` | 8 | BF16 | as mixer + `block_inject_weight` [4, 10240] |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj` | 4 | BF16 | [12288, 2560], [512, 2560], [512, 2560], [2560, 6144] |
| `mtp.layers.0.self_attn.{q,k}_norm` | 2 | BF16 | [256] |
| `mtp.layers.0.self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}` | 3 | BF16 | [640, 2560], [128], [128] |
| `mtp.layers.0.mlp.{gate, shared_expert.*, shared_expert_gate}` | 5 | BF16 | [512, 2560], 640-wide SwiGLU, [1, 2560] |
| `mtp.layers.0.mlp.experts.{0..511}.{gate,up,down}_proj.weight` | 1536 | F8_E4M3 | [640, 2560] / [2560, 640] |
| same `.weight_scale_inv` | 1536 | BF16 | [5, 20] / [20, 5] (128x128 blocks) |

- 3101 tensors: 29 in shards 9-10, 3072 in `model-fp8-mtp-ple.safetensors` (50 GiB file, the
  experts sit at byte 51.2e9 to 53.6e9 behind 129 PLE table tensors).
- `hf_quant_config.json`: `quantized_layers["mtp.layers.0.mlp.experts"] = {FP8_BLOCK_SCALES,
  group_size 128}`; every other mtp tensor is unquantized BF16. V dispatches it to
  `Fp8MoEMethod` with `weight_block_size [128, 128]`, dynamic activation scale (PR #55513
  `modelopt.py`, test `test_modelopt_mixed_precision_dispatches_block_fp8_moe`).
- config.json: `mtp_num_hidden_layers 1`, `mtp.layer_types ["full_attention"]`,
  `mtp_use_dedicated_embeddings false`, `tie_word_embeddings false`, `hc_count 4`,
  `hc_lowrank 320`, `ple_layer_ids [2]`, no `index_share_for_mtp_iteration`.

## Forward, one draft step (d = 2560, hc = 4, T tokens)

Notation: `norm_g(x, w)` = RMSNorm with `(1 + w)` gain (T:170-177, V uses `GemmaRMSNorm`).

1. **Embedding branch.** `e = embed_tokens(t)` shared with the target (V:277-278, remap V:102-103);
   `e = pre_fc_norm_embedding(e)` [T, d] (V:299); `e = fc_embedding(e)` [T, d] (V:300).
2. **Hidden branch.** `h` is the target's hc stream [T, hc*d] (V:302-307). `pre_fc_norm_hidden` is
   ONE RMSNorm over all 10240 elements, not per stream (V:230-232 size `hidden_size * hc_count`,
   V:308 applies it to `flatten(-2)`). Then `fc_hidden` per stream on the [T, hc, d] view with one
   shared weight (V:204-212, V:311), flattened back to [T, hc*d] (V:312).
3. **Fusion.** `prev_block_output = e`, `prev_injection = None` (V:313, V:322-323): the layer's first
   hc combine adds `e` to every one of the 4 streams with unit weight (VH:163-164, VH:166-173).
   So `x_i = fc_hidden(norm(h))_i + e` for i in 0..3.
4. **Layer type.** One `Qwen4ExpDecoderLayer`, `layer_type="full_attention"` (V:213-220), index
   `num_hidden_layers + 0 = 48` (V:179, V:217). PLE is off: `49 not in ple_layer_ids [2]` (VM:197,
   T:1202). MoE layer (T:1200; VM:242-243).
5. **Attention hc read.** `attn_hyper_connection.combine_and_mix(x, e, None)` (VM:309): combine
   (step 3), then `xn = hc_norm(x)` grouped per d (T:947, VH:85-90), `lora = silu(down(xn) / hc)`
   (T:960), `gate = sigmoid(up(lora))` (T:961), `block_input = mean_i(gate_i * xn_i)` [T, d]
   (T:962-965), `injection = 2 * sigmoid(block_inject(xn) / hc)` [T, hc] (T:968).
6. **Attention.** QSA sparse attention because `indexer_n_heads` is set (VM:219). Gated output:
   q_proj emits [24, 2 x 256] split into query and gate (T:805-808, VQ:229-231), per-head
   `q_norm`/`k_norm` (T:810-811), 2 KV heads of 256, `attn * sigmoid(gate)` then `o_proj`
   (T:836-838, VQ:424-426). Indexer: `index_qk_proj(block_input)` (VQ:414), budget 2048 tokens,
   compress ratio 4, block top-k 512 (T:620-622). The draft attention owns its indexer and top-k buffer
   (VQ:305-331, V:251-260).
7. **Attention hc inject + MLP hc read.** `mlp_hyper_connection.combine_and_mix(x, attn_out,
   injection)` (VM:326-328): `x = x + attn_out * injection_i` per stream (T:1236-1237), then the
   step 5 mix with the mlp module's weights.
8. **MoE.** Router `softmax(gate(x))` in FP32, top-10 of 512, renormalized (`norm_topk_prob`)
   (T:907-916). Experts SwiGLU `down(silu(gate(x)) * up(x))` scaled by the routing weight
   (T:889-892). Shared expert SwiGLU 640 wide times `sigmoid(shared_expert_gate(x))`, added
   (T:930, T:934-936).
9. **Output mixer.** `hyper_connection_mixer.combine_and_mix(x, mlp_out, injection)` (V:341-345):
   the combine `x + mlp_out * injection_i` yields `multi_hidden` [T, hc*d]; the mix (step 5
   without `block_inject`, `use_combine=False`, V:242-246, VH:108-117) yields
   `sample_hidden` [T, d]. The checkpoint ships no `block_inject_weight` for the mixer; V maps it
   to None if present (V:356-358). There is no final RMSNorm after the mixer (T:1330, T:1430).
10. **Logits.** `lm_head(sample_hidden)` (V:441). `tie_word_embeddings` is false, so the draft
    loads the target's `lm_head.weight` (V:112-117, V:399-406): shared, not a separate head.

## h_prev per step

| Step | `hidden_states` input | Reference |
| --- | --- | --- |
| k = 0 | target's final-mixer `multi_hidden` [T, hc*d]: the post-combine, pre-mix hc stream of layer 47 | VM:540-548, V:302-305 |
| k >= 1 | the previous draft step's `multi_hidden` (second tuple element) | V:337-346, VP:603, VP:760 |

- VS:839-846 sets `hc_mult = hc_count` so the proposer's hidden buffer is hc*d wide.
- QSA index sharing across steps (`set_skip_topk`) is off for this checkpoint: the flag
  `index_share_for_mtp_iteration` is absent, default False (VS:836-838, VP:579, VP:607). Each
  step runs its own indexer top-k.

## imp mapping

| Reference | MtpHead field |
| --- | --- |
| `fc_embedding`, `fc_hidden` | `fc_embedding`, `fc_hidden` (`fc` stays null) |
| `pre_fc_norm_{embedding,hidden}` | same names; `hc_count` = 10240 / 2560 = 4 from their shapes |
| `attn_hyper_connection`, `mlp_hyper_connection` | `attn_hc`, `mlp_hc` (`MtpHyperConnection`) |
| `hyper_connection_mixer` | `final_mixer` (`block_inject` null) |
| `self_attn.*`, `indexer.*` | `q/k/v/o_proj`, `q/k_norm`, `indexer_qk_proj`, `indexer_{q,k}_norm` (host only) |
| `mlp.gate`, `shared_expert*` | `router`, `shared_expert_*` |
| experts + `weight_scale_inv` | `experts_fp8[512]` host views; device: `fp8_{gate_up,down}_tab` pointer tables, FP32 `fp8_*_scales` |

- Forward: `src/compute/mtp_forward_qwen4exp.cu` (steps 1-9), attention shared with the Qwen
  layout (`mtp_attention_row`), experts `gemv_fp8_block_moe` (E4M3 bytes unchanged, 2400 MiB,
  one allocation per expert). The QSA indexer is not run: see resolved item 3.
- h_prev: `GraphExecutor::view_mtp_hidden` = `hc_hidden_` rows, the post-combine stream after
  layer 47 (the final mixer only reads it). Chain steps read `ws.d_hc_x` (`mtp_chain_hidden`).
- `diagnostics.mtp_prenorm_h` does not apply: the reference feeds the stream un-normed (V:302-308).
- Verify with host-resident experts: rows 2..8 run the n == 1 host-expert path per row
  (`run_moe_decode_rows_host_`); n-gram and token recycling stay off there (`spec_drafter_state_`).
- `speculative.mtp_k=auto` resolves to 1 on this head (`mtp_auto_k_cap`).
- `model-fp8-mtp-ple.safetensors` maps sparse when the head is requested: no `MAP_POPULATE`,
  129 PLE table tensors skipped, `MADV_WILLNEED` on the 3072 expert tensors only.

## Resolved (were OPEN)

Extra sources, same revision `60ad959b`: VU = `vllm/v1/spec_decode/utils.py`,
VF = `vllm/model_executor/layers/quantization/utils/fp8_utils.py`.

| Item | Answer | Reference |
| --- | --- | --- |
| Draft positions / RoPE per step | Pair (t_{i+1}, h_i) sits at position i, the target position of h_i; step k >= 1 at i + k. imp: `ws.mtp_pos` = pair index, chain steps append at +1. | VP:848 "Simply rotate the input ids and leave the positions unchanged", VP:856 `self.input_ids[: num_tokens - 1] = target_token_ids[1:]`, VP:864 `self._set_positions(num_tokens, target_positions)`; later steps VU:62 `new_position = position + 1` |
| FP8 block-scale orientation | W[n, k] = fp8(W[n, k]) * scale[n / 128][k / 128], scale rows follow weight rows (N), columns K. Activation stays FP16 in imp (V also quantizes it per 128-group to FP8: a kernel detail, not model math). | VF:933 `_Bsf = _Bs.repeat_interleave(_bn, dim=0).repeat_interleave(_bk, dim=1)[:_N, :_K]`, VF:934 `_out = (_Af * _Asf) @ (_Bf * _Bsf).t()` |
| Draft KV / indexer under verify rollback | V writes draft K/V by position (slot from position), so a rejected position is overwritten when the next round re-drafts it; seq_lens drop the rejected tail. imp: chained appends roll back to `pos_after`, the verify feed re-appends accepted pairs at their positions (`mtp_post_verify_update_`). The draft KV is separate from the target's GDN snapshot. Indexer: the draft attends densely; QSA selects every block below 512 complete blocks (2048 + 3 positions), so draft math equals V there; beyond it the draft attends to more than V (acceptance only, verify stays exact). | VU:74 `slot_id = block_id * block_size + (clamped_position % block_size)`; VP:678-683 "In padded drafter batch ... `common_attn_metadata.seq_lens -= num_rejected_tokens_gpu`"; `compute/qsa_indexer.h` lines 3-7 |

## Verification

`tools/analysis/mtp_qwen4exp_reference.py` computes two chained draft steps (tokens 9707, 1234;
positions 0, 1; h_prev = hash k / 1024) in FP64 numpy from the checkpoint and writes
`tests/data/mtp_qwen4exp_ref.txt`; `tests/test_mtp_qwen4exp_reference.cpp` runs imp's draft step
on the same input. Band per logit: 4 x the script's FP16-storage noise + 2^-7.

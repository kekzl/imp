# MTP Wiring — Design Spec (2026-05-14)

## Background

Qwen3.6-NVFP4 (and other DeepSeek-V3-family models) ship a Multi-Token-Predictor head (`model_mtp.safetensors`, ~1.6 GB for Qwen3.6) that imp currently skips at load time (`src/model/llm_compressor_loader.cpp:96`):

```cpp
return ... || starts_with(in, "mtp.") || starts_with(in, "model.mtp.");
```

Memory file `spec_decode_qwen36_broken_2026_05_02` documents that self-speculative decoding on stock pretrained models has ≈0% acceptance because the LM head was trained on the final layer, not intermediate layers. The MTP head IS a trained multi-token-predictor — wiring it gives the only viable speculative-decode path on these models.

## MTP file structure (Qwen3.6-NVFP4 reference)

19 BF16 tensors, all under `mtp.*` prefix. Effectively a single Qwen3.6 Transformer layer + the FC projection that conditions the layer on (prev_hidden_state ⊕ embedding):

```
mtp.fc.weight                                     [2048, 4096]  BF16
mtp.pre_fc_norm_embedding.weight                  [2048]
mtp.pre_fc_norm_hidden.weight                     [2048]
mtp.layers.0.input_layernorm.weight               [2048]
mtp.layers.0.post_attention_layernorm.weight      [2048]
mtp.layers.0.self_attn.q_proj.weight              [8192, 2048]
mtp.layers.0.self_attn.k_proj.weight              [512, 2048]
mtp.layers.0.self_attn.v_proj.weight              [512, 2048]
mtp.layers.0.self_attn.o_proj.weight              [2048, 4096]
mtp.layers.0.self_attn.q_norm.weight              [256]
mtp.layers.0.self_attn.k_norm.weight              [256]
mtp.layers.0.mlp.gate.weight                      [256, 2048]
mtp.layers.0.mlp.experts.gate_up_proj             [256, 1024, 2048]  # 256 experts × (2×512, 2048)
mtp.layers.0.mlp.experts.down_proj                [256, 2048, 512]
mtp.layers.0.mlp.shared_expert.gate_proj.weight   [512, 2048]
mtp.layers.0.mlp.shared_expert.up_proj.weight     [512, 2048]
mtp.layers.0.mlp.shared_expert.down_proj.weight   [2048, 512]
mtp.layers.0.mlp.shared_expert_gate.weight        [1, 2048]
mtp.norm.weight                                   [2048]
```

`mtp.layers.0` mirrors the architecture of a regular Qwen3.6 layer (GQA attention with q_norm/k_norm + shared+experts MoE). The MTP-specific addition is the `mtp.fc` 4096→2048 projection that maps `concat(embedding(token), prev_hidden_state)` into the layer's hidden dim.

LM head is **shared** with the main model (`model.lm_head.weight` is reused).

## Forward pass (DeepSeek-V3 / Qwen3.6 MTP)

```
# Given prev tokens t_0..t_{n-1} and main model's final-layer hidden state h_{n-1}:
mtp_input = concat(embedding(t_{n-1}), h_{n-1})            # [n, 2*hidden_dim]
mtp_input = normalize(mtp_input) via pre_fc_norm_*
hidden    = mtp.fc(mtp_input)                              # [n, hidden_dim]
hidden    = mtp.layers.0(hidden)                            # 1 transformer layer
hidden    = mtp.norm(hidden)
draft_logits = model.lm_head(hidden)                        # [n, vocab_size]
draft_token  = argmax(draft_logits)                         # or sampling
```

This yields ONE draft token per call. For K-step speculation (K=2 typical), the MTP head is invoked K times sequentially, each time feeding back the previously drafted token as the new "last token".

## Phases

### Phase 1 — Foundation (this PR scope)
**1.A. Detection + scaffolding** (THIS SESSION, ~2-3h):
- Add `MtpHead` struct in new file `src/model/mtp_head.h`
- Add `std::optional<MtpHead> mtp_` field on `Model`
- Detect `model_mtp.safetensors` next to `model.safetensors` at load time
- Log when detected; no load yet
- Acceptance: `imp-server --model /path/with/mtp` logs "MTP head detected (X.X GB), not yet wired"

**1.B. Weight loading** (future session, ~1-2 days):
- Stop skipping `mtp.*` prefix in `llm_compressor_loader.cpp:96`
- Route `mtp.*` tensors into a separate `weight_map` namespace
- Use existing BF16→FP16 upload path
- VRAM accounting: add ~1.6 GB to total budget

**1.C. Quantization decision** (future session, ~1 day):
- BF16 storage: simplest, 1.6 GB VRAM cost
- FP16 storage: same size, better compute compat
- NVFP4 storage: cuts to ~400 MiB, but needs per-tensor quantization pass at load
- Decision: ship BF16 first (no quant), defer NVFP4 to performance phase

### Phase 2 — MTP forward kernel (future, ~3-5 days)
- Implement the FC projection + 1 transformer layer + final norm
- Reuse existing attention/MLP kernels (Qwen3.6 layer is already supported)
- Add LM-head invocation that reuses main model's `out_proj_`

### Phase 3 — Verify-loop integration (future, ~2-3 days)
- Plumb through `src/runtime/self_speculative.cpp` (or new `src/runtime/mtp_speculative.cpp`)
- Draft tokens via MTP head, verify via full forward in paged-attention decode mode
- KV state handling: MTP layer has its own attention → needs its own KV cache slot
- (Critical: avoid the prefill-mode KV divergence documented in `turbodraft.md`)

### Phase 4 — Engine plumbing (future, ~1-2 days)
- New CLI flag `--mtp-spec-decode K` (K = draft length)
- Config field `runtime.mtp_spec_decode`
- VRAM budgeting for MTP weights + draft KV slot

### Phase 5 — Acceptance-rate validation (future, ~3-5 days)
- Smoke test on Qwen3.6-NVFP4 with multiple prompt classes (factual / verbose-think / code / instruction-following)
- Measure: tok/s vs baseline, acceptance rate, per-K-value scaling
- Validate output coherence (no degeneration vs baseline)
- Gate default-on/off based on measured net win

## Risks + open questions

1. **MTP file is not NVFP4** — weights are BF16. Either keep BF16 (1.6 GB extra VRAM) or quantize at load. Decision deferred to Phase 1.C.

2. **GDN/SSM models** — Qwen3.6 has 24 GDN layers in the main model. MTP forward only runs ONE attention layer, so GDN state doesn't apply. But verify-mode still runs the full main model including GDN — Failure 3 from `spec_decode_qwen36_broken_2026_05_02` may not fully apply because we're not doing layer-skip drafting; we're using a separate MTP head.

3. **NVFP4 dequant-graph crash** — addressed by PR #121 (workspace + capture-guard). cuBLAS GEMM under capture status=14 risk still open for fallback paths but not on the main NVFP4 path.

4. **Acceptance rate is unknown** — DeepSeek-V3 paper claims ~85% acceptance on `most prompts`. Real-world on imp's Qwen3.6 NVFP4 quantized weights with imp's sampling: not measured. Phase 5 will determine if MTP justifies its 1.6 GB cost.

5. **Multimodal models** — Qwen3.6-VL has the MTP head conditioned on multimodal features. Audio/image tokens may require different handling. Defer until pure-text Qwen3.6 wired.

## Phase 1.A acceptance criteria

- Building Qwen3.6-NVFP4 model loads cleanly (no regression)
- `imp-server --model /home/kekz/models/Qwen3.6-35B-A3B-NVFP4` logs `MTP head detected: 1.69 GB`
- All existing tests pass
- `MtpHead` struct compiles with all expected fields
- `Model::mtp_` field is `std::optional` to signal absence cleanly

## Out-of-scope for Phase 1.A

- Loading any MTP tensors (still skipped at safetensors_loader level)
- VRAM budget changes
- Any forward-pass code
- CLI flag wiring
- Production usage

## Cross-references

- Memory: `spec_decode_qwen36_broken_2026_05_02` — why current self-spec fails
- Memory: `turbodraft.md` — KV-divergence pitfalls in spec-decode
- DeepSeek-V3 paper: arXiv 2412.19437, sec. 4.5 (MTP architecture)
- Qwen3.6 model card on HF: confirms `model_mtp.safetensors` is the trained MTP head

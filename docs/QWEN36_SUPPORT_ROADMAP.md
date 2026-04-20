# Qwen 3.6 Support — Roadmap

## Context

Qwen3.6-35B-A3B was released 2026-04-14 (Apache 2.0, unsloth GGUF + official FP8). It is a **GDN + Gated Attention + MoE hybrid** — the same GDN-Attention interleaving as Qwen 3.5 (3:1 ratio) but with **MoE on every layer** instead of dense FFN.

Architecture summary:
- 40 layers: 10 × (3 × GDN + 1 × Attention)
- MoE on all 40 layers (not just attention layers)
- 35B total / 3B active parameters, 128 experts
- 262k native context, extensible to 1M
- Focus: agentic coding, frontend/repo-level reasoning

**vs Qwen3.5 (already supported):**
- Qwen3.5: 24 GDN + 8 Attn + 32 dense FFN
- Qwen3.6: 30 GDN + 10 Attn + 40 MoE (all layers)

Both use the same GDN kernels (scan + fused RMSNormGated+SiLU + attention output gate + partial RoPE) and MoE kernels. What's new: the combination.

## Current State (2026-04-20, second pass after initial download)

**Verified on unsloth/Qwen3.6-35B-A3B-GGUF (Q4_K_M variant, 22 GB):**
- GGUF ships as `general.architecture = "qwen35moe"` — unsloth reuses the
  Qwen 3.5 MoE arch name. imp's existing `qwen35moe` parser picks it up;
  the `QWEN36_MOE` enum I added earlier is therefore dead scaffolding.
- Layer census correctly detects 10 attention + 30 GDN + 40 MoE + 40
  shared experts + 30 SSM state slots.
- Tensor names match imp's existing `attn_gate → gdn_gate` mapping.

**Shipped fixes (main branch):**
- PR / commit `dc0be95`: `run_moe_ffn` FFN input norm fallback now includes
  `post_attn_norm` (Qwen's single-norm variant stores the FFN norm under
  `post_attention_norm`, not `ffn_norm`). Without this, residual stream
  explodes: logits L2=108k, semantic garbage.

**Shipped, commit `b3c300b`:**
- `ffn_gate_inp_shexp.weight [2048]` now mapped →
  `TransformerLayer::shared_expert_gate_inp`. Uploaded as FP16 (matches the
  other norm-weight path). Applied in `run_moe_ffn` via the new
  `shared_expert_gate_scale` device kernel: per-row dot product,
  sigmoid, elementwise scale of the shared-expert output.
- 733/733 tensors load (0 skipped). No more NaN cascade.

**Still blocked — output still garbage, next architecture gap:**

Attention layers (blk.3, blk.7, ..., blk.39 in the 3:1 GDN:attn pattern)
ship a **fused Q + attention-output-gate** tensor:

```
blk.3.attn_q.weight shape=[2048, 8192]   # 2× expected Q size
blk.3.attn_k.weight shape=[2048, 512]    # 2 kv heads × 256 head_dim
blk.3.attn_v.weight shape=[2048, 512]
blk.3.attn_output.weight shape=[4096, 2048]  # 4096 in, 2048 out
```

Reference (llama.cpp `qwen3next.cpp::build_layer_attn`):
```
Qcur_full = cur @ wq         // [n, 8192]
Qcur, gate = split(Qcur_full)  // each [n, 4096]
Kcur = cur @ wk              // [n, 512]
Vcur = cur @ wv              // [n, 512]
Qcur = q_norm(Qcur); Kcur = k_norm(Kcur)   // per-head RMSNorm
Qcur = RoPE(Qcur); Kcur = RoPE(Kcur)        # n_rot = 64 (partial)
attn_out = attention(Q, K, V)    // [n, 4096]
gate = sigmoid(gate)
attn_out *= gate
cur = attn_out @ w_output       // [n, 2048]
```

imp's current `run_attention` does not handle the fused Q+gate split or
the post-attention sigmoid multiply. With the 8192-wide Q, imp currently
interprets all of it as Q heads (2× the real head count), producing
invalid attention outputs that are then multiplied by the wrong
`attn_output` shape.

GDN layers (blk.0, blk.1, blk.2, ...) have `attn_qkv [2048, 8192]` and a
separate `attn_gate [2048, 4096]`. These go through `run_gdn`, not
`run_attention`, but the same Q+gate structure likely applies for the
delta-rule path. imp's existing `gdn_gate` mapping may already handle
this correctly for GDN (needs verification); the blocker is specifically
the attention-layer path.

**Remaining work (approximate, ~300-500 LoC):**
1. Shape-detect fused Q in `run_attention`: if `attn_q.shape[1] == 2 *
   n_heads * head_dim`, split.
2. Cache the gate after Q projection; apply `out *= sigmoid(gate)` after
   the attention computation, before `attn_output` GEMM.
3. Verify partial RoPE matches: GGUF metadata says `rope_dim=64` (not
   full `head_dim=256`). Current imp logs "Partial RoPE: rope_dim=64 (full
   head_dim=256)" so this looks plumbed, but needs tensor-level
   comparison against a reference forward pass.
4. E2E test once output is coherent.
- Still TODO: 1M-context RoPE extension if needed; real E2E test in
  `tests/test_e2e_models.cpp`.

## Remaining Work

### 1. Model metadata parsing
Verify which string unsloth uses for `general.architecture` in the GGUF (likely `qwen36moe` based on Qwen naming convention — need to `hexdump` a real file). If different, add to `parse_model_arch()`.

For SafeTensors: `config.json` "model_type" field; check official `Qwen/Qwen3.6-35B-A3B` repo.

### 2. Layer config validation
Current GGUF loader detects GDN vs attention structurally (from tensor names). Qwen 3.6 uses the same tensor names as Qwen 3.5, so the detection should work. Verify:
- Layer census correctly reports `30 GDN + 10 attn + 40 MoE` when Qwen3.6-35B-A3B loads
- MoE is detected for ALL 40 layers (not just the 10 attention layers — Qwen 3.5 only had MoE on attention blocks in the MoE variant)

### 3. Forward pass dispatch
Most of the forward path should Just Work because:
- GDN layers: same kernels as Qwen 3.5 (already wired)
- Attention layers: standard Gated Attention (already wired for Qwen 3.5)
- MoE FFN: already wired for Qwen3-MoE / Qwen3-Coder-30B-A3B

Risk: does `executor_ssm_gdn.cu` correctly call MoE after the GDN layer? Currently Qwen 3.5 GDN layers use dense FFN. Likely need a branch: `if (arch == QWEN36_MOE) run_moe_ffn(); else run_dense_ffn();` after each GDN block.

### 4. RoPE / context extension
Qwen 3.6 has 262k native + 1M extended context. Verify RoPE scaling params (likely YaRN or similar) are parsed from metadata. Current Qwen 3.5 RoPE handling should cover native; 1M extension may need extra work.

### 5. E2E test
- Download `unsloth/Qwen3.6-35B-A3B-GGUF` (Q4_K_M or IQ4_XS variant) — ~18-22 GB
- Add `Qwen36ModelTest.GenerateCoherentOutput` in `tests/test_e2e_models.cpp`
- Benchmark: target tg256 ≥ 130 tok/s (similar to Qwen3.5-9B-GDN)
- Compare against llama.cpp on same quant

## Hardware fit (RTX 5090 32GB)

At Q4_K_M (~18 GB): fits with headroom for 8-12k KV context. Q5_K_M (~22 GB) tighter. Q8_0 (~35 GB) **does not fit** on 32 GB.

## Open Questions

- Does `config.json` contain an explicit `layer_types` array indicating which layers are GDN vs Attention? If yes, parse it directly rather than relying on structural tensor-name detection.
- Does the official FP8 checkpoint use Model Optimizer NVFP4 prequant format (like Qwen3-Coder-30B-A3B-FP4)? If yes, it can use the existing NVFP4 MoE path (and benefit from the pending CUTLASS 3.x work in PR #22).
- "Reasoning context from historical messages": is this a model-side behavior or does it require explicit KV-cache reuse API changes?

## Priority vs other work

Depends on:
- **Blocker**: need a local GGUF download (~20 GB) before we can test
- **Parallelizable**: PR #22 (CUTLASS 3.x NVFP4 MoE) — if FP8 checkpoint is NVFP4-prequant, #22 completion blocks fastest Qwen 3.6 path
- **Orthogonal**: native Q4_K_M GEMV wiring (low value) — unrelated to Qwen 3.6

## References
- Qwen blog (2026-04-14): https://qwen.ai/blog?id=qwen3.6
- unsloth GGUF: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- Official FP8: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- Reference to Qwen 3.5 implementation: `src/graph/executor_ssm_gdn.cu`, `src/compute/gdn.cu`, memory `qwen35_gdn.md`

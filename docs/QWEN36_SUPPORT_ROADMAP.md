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

## Current State (2026-04-20)

**Scaffold added on branch `qwen36-support-scaffolding`:**
- `ModelArch::QWEN36_MOE` enum + `IMP_ARCH_QWEN36_MOE = 13` in C API
- `parse_model_arch()` recognizes `"qwen36moe"`, `"qwen3.6_moe"`, `"qwen3.6moe"` strings
- `ArchRegistry` entry with ChatML chat template + Qwen sampling defaults (T=0.6, top_p=0.95, top_k=20)
- `chat_template.cpp` routes QWEN36_MOE → CHATML family

**Not yet implemented:**
- GGUF config parsing for Qwen 3.6 metadata
- Per-layer dispatch verifying MoE-on-all-layers (not MoE-on-attention-only)
- SafeTensors loader extension for the FP8 variant
- E2E test

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

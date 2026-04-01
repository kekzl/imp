# imp Model Abstraction Layer — Roadmap (Post-Implementation)

**Date:** 2026-04-01 (updated after implementation)

All three tiers have been implemented. This document summarizes what was done and what remains.

---

## Tier 1 — "Jedes neue Modell bricht ohne das" — COMPLETED

### 1.1 config.json Parser — DONE
- `src/model/hf_config_loader.{h,cpp}` — reads 20+ fields from `config.json`
- HF architecture mapping: 18 class names → ModelArch
- `model_type` fallback for configs without `architectures` field
- Rope scaling: linear, yarn, longrope, dynamic
- MoE, softcapping, activation type detection

### 1.2 Sharded SafeTensors — DONE
- `load_sharded()` in `safetensors_loader.cpp` — parses `model.safetensors.index.json`
- Multi-shard mmap with proper cleanup via `split_mmaps_`
- Directory mode: tries sharded first, falls back to single file

### 1.3 Multiple EOS Token IDs — DONE
- `eos_ids_` vector in `tokenizer.h`
- GGUF: reads `eot_token_id` + `eog_token_id` in addition to primary
- `generation_config.json`: handles array `eos_token_id`
- Engine: `is_eos()` checks all IDs

### 1.4 Weight Mappings — DONE
- WeightMap extended for: GDN, SSM, shared experts, attention biases, QK-norm, post-layer norms, GPTQ

---

## Tier 2 — "Wichtig für Production-Nutzung" — COMPLETED

### 2.1 Tool Calling via Jinja2 — DONE
- `apply_with_tools()` passes tools to Jinja2 context
- `ToolFunction` struct for type-safe tool definitions
- JSON parameters parsed into Jinja2 objects for template access
- Fallback to text-based `build_tool_prompt()` when Jinja2 doesn't handle tools

### 2.2 logit_bias — DONE
- Parsed from request JSON (string key → float value)
- Stored in `Request::logit_bias`
- Applied in executor before sampling (H2D per-entry)

### 2.3 tokenizer_config.json + generation_config.json — DONE
- `load_chat_template()` — extracts Jinja2 template string
- `load_added_tokens()` — extracts id, content, special flag
- `load_generation_config()` — extracts EOS token IDs (scalar or array)
- `load_gptq_config()` — extracts bits, group_size, desc_act

### 2.4 Architecture Defaults from config.json — DONE
- `hidden_act` → FFNActivation mapping
- `rope_scaling` → all variants (linear, yarn, longrope, dynamic)
- `apply_arch_defaults()` remains as GGUF-path fallback

---

## Tier 3 — "Nice-to-have für Developer Experience" — COMPLETED

### 3.1 HF Hub Download — DONE
- `resolve_model_path()` — checks local, HF cache, then `huggingface-cli download`
- `resolve_model_gguf()` — convenience for GGUF files
- `--revision` flag in CLI and server
- Integrated into both `imp-cli` and `imp-server`

### 3.2 tokenizer.json Support — DONE
- `Tokenizer::load()` — full JSON parser for HF tokenizer.json
- Supports BPE, Unigram, WordPiece model types
- Pre-tokenizer detection: ByteLevel, Metaspace, Sequence
- Added tokens with special flag → CONTROL type
- Auto-detection of BOS/EOS from token content

### 3.3 Pre-tokenizer Support — DONE
- `tokenizer.ggml.pre` read from GGUF
- `llama3_pre_tokenize()` — individual digits, contraction splitting
- Default `gpt2_pre_tokenize()` — space-attached, digit groups

### 3.4 Model Registry Pattern — DONE
- Static `unordered_map` in `parse_model_arch()` — 31 entries (15 GGUF + 16 HF)
- Single lookup, falls back to GENERIC

### 3.5 Tokenizer Correctness Tests — DONE
- `tests/generate_tokenizer_golden.py` — generates golden data from `transformers.AutoTokenizer`
- `tests/test_tokenizer_compat.cpp` — GTest comparing imp vs HF output

### 3.6 Additional Items — DONE
- NFC normalization (100 Latin pairs, fast-path)
- GPTQ 4-bit dequant kernel + upload integration
- `strftime_now()` Jinja2 builtin
- Vision fallback ID removal
- `default_family_for_arch` QWEN35/QWEN35_MOE entries
- `n > 1` completions (sequential, 1-4)

---

## Future Work (not in original audit)

These are potential improvements identified during implementation but not part of the original gap list:

1. **AWQ quantization support** — separate kernel needed, lower priority than GPTQ
2. **Full regex pre-tokenizer** — would cover all model-specific patterns, significant effort
3. **HF token authentication** — `--hf-token` flag for gated models
4. **GPTQ 8-bit kernel** — straightforward extension of 4-bit kernel
5. **Streaming n > 1** — multiplexed SSE, complex state management
6. **Vision encoder SafeTensors loading** — SigLIP weights from SafeTensors (currently GGUF mmproj only)

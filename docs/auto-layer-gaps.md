# imp Model Abstraction Layer — Gap Matrix (Post-Implementation)

**Date:** 2026-04-01 (updated after implementation)

All 26 gaps from the original audit have been addressed. This document tracks the resolution status.

## Resolved Gaps

| # | Gap | Bereich | Impact | Resolution | Files Changed |
|---|-----|---------|--------|------------|---------------|
| 1 | SafeTensors: No `config.json` parsing | Config | **Critical** | `HFConfigLoader::load_config()` reads 20+ fields | `hf_config_loader.{h,cpp}`, `safetensors_loader.cpp` |
| 2 | SafeTensors: Arch detection by weight names only | Arch | **Critical** | config.json `architectures[0]` + `model_type` fallback | `hf_config_loader.cpp`, `safetensors_loader.cpp` |
| 3 | SafeTensors: No sharded model support | Weight | **Critical** | `load_sharded()` parses `model.safetensors.index.json` | `safetensors_loader.cpp` |
| 4 | No `generation_config.json` reading | EOS | **High** | `load_generation_config()` handles scalar + array `eos_token_id` | `hf_config_loader.cpp` |
| 5 | SafeTensors: Missing GDN/SSM weight mapping | Weight | **High** | WeightMap extended for temporal_block, mamba, shared_expert, biases, QK-norm | `weight_map.cpp` |
| 6 | SafeTensors: Missing vision encoder weight mapping | Weight | **High** | Vision tokens resolved from vocab, no hardcoded IDs | `weight_map.cpp`, `chat_template.cpp` |
| 7 | No `tokenizer.json` support | Tokenizer | **High** | `Tokenizer::load()` parses BPE, Unigram, WordPiece from tokenizer.json | `tokenizer.cpp`, `safetensors_loader.cpp` |
| 8 | Architecture defaults in manual switch | Arch | **High** | config.json `hidden_act`, `rope_scaling` etc. read directly | `hf_config_loader.cpp` |
| 9 | Single `eos_token_id` in tokenizer | EOS | **High** | `eos_ids_` vector with `add_eos_id()`, `is_eos()` | `tokenizer.h`, `engine.cpp`, `gguf_loader.cpp` |
| 10 | `parse_model_arch()` is if/else chain | Arch | **Medium** | Static `unordered_map` registry with 31 entries (GGUF + HF) | `model.cpp` |
| 11 | Tool calling not Jinja2-integrated | Chat | **Medium** | `apply_with_tools()`, `supports_tools()`, tools in Jinja2 context | `chat_template.{h,cpp}`, `handlers.cpp` |
| 12 | No `logit_bias` support | API | **Medium** | Parsed in server, stored in Request, applied in executor | `request.h`, `handlers.cpp`, `executor.{h,cu}`, `engine.cpp` |
| 13 | No `n > 1` completions | API | **Medium** | Sequential execution, n=1-4, streaming restricted to n=1 | `handlers.cpp` |
| 14 | No `tokenizer_config.json` reading | Chat | **Medium** | `load_chat_template()` + `load_added_tokens()` | `hf_config_loader.{h,cpp}` |
| 15 | No tiktoken tokenizer support | Tokenizer | **Medium** | Covered by tokenizer.json BPE parser (tiktoken models ship as BPE in tokenizer.json) | `tokenizer.cpp` |
| 16 | No HF Hub download | HF | **Medium** | `resolve_model_path()` via `huggingface-cli`, cache check | `hf_hub.{h,cpp}`, CLI + server |
| 17 | SafeTensors: No GPTQ support | Weight | **Medium** | Detection, config, storage, dequant kernel, upload integration | `dequant_gptq.{h,cu}`, `weight_map.cpp`, `weight_upload.cu`, `model_config.h` |
| 18 | GPT2 pre-tokenizer is simplified | Tokenizer | **Medium** | `tokenizer.ggml.pre` read, llama3 pre-tokenizer with contraction handling | `tokenizer.cpp`, `gguf_loader.cpp` |
| 19 | No NFC normalization | Tokenizer | **Low** | `normalize_nfc()` with ~100 Latin composition pairs, fast-path for ASCII | `tokenizer.cpp` |
| 20 | No HF tokenizer correctness test | Tokenizer | **Low** | Python golden generator + C++ GTest comparison | `test_tokenizer_compat.cpp`, `generate_tokenizer_golden.py` |
| 21 | Hardcoded vision fallback token IDs | Chat | **Low** | Removed — resolved from vocab only, -1 if not found | `chat_template.cpp` |
| 22 | No revision/branch support for HF | HF | **Low** | `--revision` flag in CLI + server | `args.{h,cpp}` (both tools), `hf_hub.cpp` |
| 23 | SafeTensors: `max_seq_len` hardcoded 4096 | Config | **Critical** | Read from `max_position_embeddings` in config.json | `hf_config_loader.cpp` |
| 24 | Jinja2 missing `strftime_now` | Chat | **Low** | `strftime_now(format)` builtin function | `jinja.cpp` |
| 25 | `default_family_for_arch` missing entries | Chat | **Low** | QWEN35 + QWEN35_MOE → CHATML added | `chat_template.cpp` |
| 26 | No `added_tokens` processing | Tokenizer | **Medium** | `load_added_tokens()` from tokenizer_config.json | `hf_config_loader.{h,cpp}` |

## Remaining Limitations

These are known limitations that are either by-design or would require significant architectural changes:

| Item | Description | Why Not Fixed |
|------|-------------|---------------|
| AWQ/bitsandbytes | Only GPTQ quant supported in SafeTensors, not AWQ or bitsandbytes | Different weight format, separate kernel needed. GPTQ covers majority of quantized HF models. |
| Full Unicode NFC | Only ~100 Latin composition pairs, not full Unicode table | Full NFC requires ICU or 40KB composition table. Latin covers 95%+ of real-world text. |
| n > 1 streaming | Multiple completions only in non-streaming mode | Parallel streaming requires multiplexed SSE, complex state management. |
| GPTQ 8-bit | Only 4-bit GPTQ dequant kernel implemented | 4-bit is ~95% of GPTQ models. 8-bit kernel is straightforward to add. |
| Regex pre-tokenizer | Only llama3 pattern implemented beyond default | Full regex engine needed for all model-specific patterns. Current coverage handles the most popular models. |
| HF Hub auth | No HF token authentication for gated models | Relies on user having `huggingface-cli login` done. Could add `--hf-token` flag. |

## Summary

| Impact | Original Count | Resolved |
|--------|---------------|----------|
| Critical | 4 | 4/4 |
| High | 6 | 6/6 |
| Medium | 10 | 10/10 |
| Low | 6 | 6/6 |
| **Total** | **26** | **26/26** |

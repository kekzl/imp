# imp Model Abstraction Layer — Full Audit

**Date:** 2026-04-01 (post-implementation update)
**Scope:** Complete inventory of all model abstraction, tokenizer, config parsing, architecture routing, EOS handling, weight loading, API server, and HF integration capabilities.

---

## 1.1 Tokenizer Stack

### How is the tokenizer loaded?

**Three loading paths:**

| Path | Source | Supported Types | Trigger |
|------|--------|----------------|---------|
| GGUF metadata | `src/model/gguf_loader.cpp:1106-1202` | SPM, GPT2 BPE | GGUF model load |
| `tokenizer.json` | `src/model/tokenizer.cpp:447-643` | BPE, Unigram, WordPiece | SafeTensors directory |
| GGUF metadata (embedded) | same as above | same | always for GGUF |

**SafeTensors integration:** `src/model/safetensors_loader.cpp:722-727` — automatically checks for `{model_dir}/tokenizer.json` and loads it via `Tokenizer::load()`.

### Tokenizer types supported

- **SPM** (SentencePiece/Unigram): Score-based BPE merging, byte fallback via `<0xHH>` tokens
- **GPT2** (Byte-level BPE): Merge-rank based, `BYTE_TO_CODEPOINT[256]` encoding table
- **Unigram**: Loaded from tokenizer.json `model.type == "Unigram"` → uses SPM encoder with scores
- **WordPiece**: Recognized from tokenizer.json, treated as generic BPE

### tokenizer_config.json / added_tokens

- `tokenizer_config.json` → `chat_template` field: **read** via `HFConfigLoader::load_chat_template()` (`src/model/hf_config_loader.cpp:485-498`)
- `added_tokens_decoder`: **read** via `HFConfigLoader::load_added_tokens()` (`src/model/hf_config_loader.cpp:502-529`) — returns id, content, special flag
- `tokenizer.json` → `added_tokens` array: **read** during `Tokenizer::load()` (`src/model/tokenizer.cpp:530-560`) — marks special tokens as CONTROL type, auto-detects BOS/EOS

### Pre/Post-Processing

- **NFC normalization**: Implemented. `normalize_nfc()` in `src/model/tokenizer.cpp:1215-1258`. ~100 Latin composition pairs, fast-path skips ASCII-only text. Called from `encode()` at line 1264.
- **SPM space prefix**: `▁` (U+2581) prepended, controlled by `add_space_prefix_`
- **Byte fallback**: Unmapped SPM symbols → `<0xHH>` byte tokens
- **GPT2 byte encoding**: 256-entry `BYTE_TO_CODEPOINT` table
- **Pre-tokenizer dispatch**: `tokenizer.ggml.pre` field read from GGUF (`src/model/gguf_loader.cpp:1118-1121`), used in `encode_gpt2()` to select split pattern:
  - `"llama3"` / `"llama-v3"` / `"llama-bpe"` → `llama3_pre_tokenize()` (individual digits, contraction handling)
  - Default → `gpt2_pre_tokenize()` (space-attached-to-next, digit groups of 3)
- **tokenizer.json pre-tokenizer**: `ByteLevel`, `Metaspace`, `Sequence` types auto-detected (`src/model/tokenizer.cpp:562-605`)

### Special token recognition

- **Primary**: `token_type` metadata from GGUF (CONTROL=3) — `src/model/tokenizer.h:77-86`
- **tokenizer.json**: `added_tokens[].special == true` → marked as CONTROL in `token_types_`
- **Fallback heuristic**: Pattern matching `<|...|>`, known names (`<pad>`, `<unk>`, etc.) — `src/runtime/engine.cpp:846-861`

### HF tokenizer comparison test

**Implemented.** `tests/test_tokenizer_compat.cpp` (GTest) + `tests/generate_tokenizer_golden.py` (Python).
- Python script generates golden output using `transformers.AutoTokenizer` for 20 test strings
- C++ test loads GGUF model + golden file, compares encode output, expects ≥80% match
- Controlled via `IMP_TEST_MODEL` and `IMP_TEST_GOLDEN` env vars

### How llama.cpp handles it

llama.cpp reads GGUF metadata + `tokenizer.ggml.pre` for regex-based pre-tokenizer selection. No tokenizer.json support. imp now covers both paths.

### How vLLM handles it

vLLM delegates to HuggingFace `tokenizers` Rust library via Python. imp now has native tokenizer.json parsing covering the same model types.

---

## 1.2 Chat Template Engine

### Template application method

**Dual path — Jinja2 primary, hardcoded fallback.**

- `apply()` at `src/model/chat_template.cpp:241-265` — dispatches to Jinja2 if available, falls back to family-specific method
- `apply_with_tools()` at `src/model/chat_template.h:58-62` — passes `tools` and `tool_choice` into Jinja2 context
- `apply_with_image()` at `src/model/chat_template.h:66-69` — Gemma-3 vision token injection

### Jinja2 template source

1. GGUF metadata `tokenizer.chat_template` (primary)
2. `tokenizer_config.json` → `chat_template` field via `HFConfigLoader::load_chat_template()` (SafeTensors path)

### Hardcoded template families (fallback)

8 families in `src/model/chat_template.h:13-22`: RAW, CHATML, LLAMA2, LLAMA3, NEMOTRON, GEMMA, DEEPSEEK_R1, PHI.

`default_family_for_arch()` at `src/model/chat_template.cpp:47-62` — maps all 12 ModelArch values including QWEN35 and QWEN35_MOE (→ CHATML).

### Tool-call handling

**Jinja2-integrated** via `apply_with_tools()` (`src/model/chat_template.cpp:907-959`):
- Builds `tools` array as Jinja2 Value objects (OpenAI format: `{type, function: {name, description, parameters}}`)
- Parses `parameters_json` into proper Jinja2 objects for template access
- `tool_choice` passed as context variable
- `supports_tools()` returns true when Jinja2 engine is active

**Fallback**: Server-side `build_tool_prompt()` string injection (`tools/imp-server/tool_call.cpp:3-56`) — used when template doesn't handle tools natively.

### bos_token, eos_token

- Read from GGUF metadata (`tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`)
- Passed to Jinja2 context as `bos_token` and `eos_token` strings
- For SafeTensors: auto-detected from `added_tokens` in tokenizer.json

### System prompt support

- **Gemma**: System merged into first user turn
- **Phi**: System treated as user role
- **DeepSeek R1**: System as plain text after BOS
- All families support system messages in some form

### Vision token IDs

**No hardcoded fallbacks.** Resolved from vocabulary via `find_token()` only (`src/model/chat_template.cpp:172-174`). Stays -1 if not found, which disables vision.

### Jinja2 builtins

`strftime_now(format)` — `src/model/jinja.cpp:1869-1879`. Returns formatted current time. Used by some chat templates.

---

## 1.3 Architecture Detection & Model Routing

### Forward pass routing

**Runtime tensor presence detection** — no registry needed for forward pass.

`src/graph/executor_forward.cu:1844-1893` — per-layer checks:
- `layer_has_gdn()` → `gdn_gate.data != nullptr`
- `layer_has_ssm()` → `ssm_in.data != nullptr`
- `layer_has_attention()` → `wq.data != nullptr`
- `layer_has_moe()` → `moe_gate.data != nullptr`
- `layer_has_dense_ffn()` → `w_up.data != nullptr && moe_gate.data == nullptr`

Naturally handles hybrid architectures (Qwen3.5, Nemotron-H).

### Architecture string recognition

**Static registry** (not if/else) at `src/model/model.cpp:82-119`:

```cpp
static const std::unordered_map<std::string, ModelArch> registry = {
    // 15 GGUF strings + 16 HuggingFace class names
};
```

GGUF: `llama`, `mistral`, `mixtral`, `deepseek`, `deepseek2`, `nemotron_h_moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `gemma3`, `gemma`, `gemma2`, `llama4`, `qwen2`, `phi3`

HF: `LlamaForCausalLM`, `MistralForCausalLM`, `MixtralForCausalLM`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`, `Gemma2ForCausalLM`, `GemmaForCausalLM`, `Gemma3ForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `PhiForCausalLM`, `Phi3ForCausalLM`, `Phi3SmallForCausalLM`, `InternLM2ForCausalLM`, `Starcoder2ForCausalLM`, `CohereForCausalLM`

**SafeTensors**: `config.json` → `architectures[0]` read via `HFConfigLoader::load_config()`. Falls back to weight-name heuristics (`detect_arch_from_weights()`).

### Tensor name mapping

Two paths, both comprehensive:

1. **GGUF**: `src/model/gguf_loader.cpp:327-486` — `assign_tensor()` with GGUF names (`blk.{i}.attn_q`)
2. **SafeTensors**: `src/model/weight_map.cpp:60-460` — `WeightMap` with HF names (`model.layers.{i}.self_attn.q_proj.weight`)

Covers: attention (QKVO + biases + QK-norm), dense FFN, MoE (Mixtral + DeepSeek styles), shared experts, SSM (Mamba2), GDN (Qwen3.5), post-layer norms (Gemma-3), GPTQ quantized weights.

---

## 1.4 Model Config Parsing

### GGUF path

`src/model/gguf_loader.cpp:740-890` — reads ~40 hyperparameters with architecture-prefixed keys. Comprehensive.

### SafeTensors / HuggingFace path

`src/model/hf_config_loader.cpp:299-451` — `HFConfigLoader::load_config()` reads from `config.json`:

| Category | Fields Read |
|----------|------------|
| Core | `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `max_position_embeddings`, `head_dim` |
| Norm | `rms_norm_eps`, `layer_norm_eps` |
| RoPE | `rope_theta`, `rope_scaling` (linear, yarn, longrope, dynamic — with factor, attn_factor, beta_fast/slow, per-dim arrays) |
| Attention | `sliding_window`, `attn_logit_softcapping`, `final_logit_softcapping` |
| Activation | `hidden_act` / `hidden_activation` → SWIGLU or GEGLU |
| MoE | `num_local_experts`, `num_experts`, `num_experts_per_tok` |
| Architecture | `architectures[0]`, `model_type` (fallback) |

**Shape inference** (`infer_config()` at `src/model/safetensors_loader.cpp:297-358`) only fills fields still at zero — config.json values are never overwritten.

### Fallback defaults

- `max_seq_len`: 4096 (only if not in config.json AND not inferable)
- `n_kv_heads`: defaults to `n_heads` (MHA) when not specified
- `rope_theta`: 10000.0 (correct for vanilla Llama, overridden by config.json for modern models)

### Unknown fields

Silently ignored (logged at DEBUG/WARN level).

---

## 1.5 Stopping & EOS Handling

### EOS token recognition

**Three sources** in `src/runtime/engine.cpp:169-181`:
1. `tok->is_eos(token)` — checks against `eos_ids_` vector (multiple EOS)
2. `chat_template_.stop_token_ids()` — template-defined stop tokens
3. `banned_token_ids_` — degeneration-detected banned tokens

### Multiple EOS tokens

**Fully supported.** `src/model/tokenizer.h:57-66`:
- `eos_ids_` is a `std::vector<int32_t>` (default: `{2}`)
- `add_eos_id(int32_t id)` — deduplicates
- `is_eos(int32_t id)` — linear scan of vector
- `eos_ids()` — returns full vector

**GGUF**: Reads `tokenizer.ggml.eos_token_id` (primary) + `tokenizer.ggml.eot_token_id` and `tokenizer.ggml.eog_token_id` (additional) at `src/model/gguf_loader.cpp:1186-1193`.

**generation_config.json**: `HFConfigLoader::load_generation_config()` at `src/model/hf_config_loader.cpp:455-481` — handles both scalar and array `eos_token_id`.

### stop_sequences (string-based)

Supported in API server. `tools/imp-server/handlers.cpp:486-501` — tokenizes stop strings, buffered matching in streaming mode.

### Speculative decoding + EOS

Correct. EOS checked post-acceptance on verified tokens, not draft tokens.

### Think block handling

`track_think_state()` toggles `req.in_think_block`. `should_stop()` suppresses stop tokens inside `<think>...</think>`.

---

## 1.6 Weight Loading & Tensor Mapping

### SafeTensors loading

- **mmap'd** (PROT_READ, MAP_PRIVATE)
- **Directory support**: `src/model/safetensors_loader.cpp:590` — accepts both files and directories
- **Sharded**: `load_sharded()` at line 529-580 — parses `model.safetensors.index.json`, loads all referenced shard files
- **Single file**: Falls back to `model.safetensors` in directory

### Weight name mapping

`WeightMap` at `src/model/weight_map.cpp:60-460` covers:
- Standard attention (QKVO), norms, dense FFN
- MoE: Mixtral (`block_sparse_moe.experts`) + DeepSeek (`mlp.experts`) styles
- Shared experts (`mlp.shared_expert`)
- Attention biases (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`)
- QK-norm (`q_norm.weight`, `k_norm.weight`)
- Post-layer norms (`post_feedforward_layernorm`, `pre_feedforward_layernorm`)
- MoE router bias (`mlp.gate.bias`)
- SSM/Mamba (`mamba.in_proj`, `mamba.out_proj`, `mamba.conv1d`, `mamba.A_log`, `mamba.D`, `mamba.norm`)
- GDN/DeltaNet (`temporal_block.gate_proj`, `temporal_block.alpha`, `temporal_block.beta`)
- GPTQ (`qweight`, `qzeros`, `scales`, `g_idx` for all attention + FFN projections)

### GPTQ quantization support

- **Detection**: `detect_arch_from_weights()` checks for `.qweight` tensors
- **Config**: `quantize_config.json` → `bits`, `group_size`, `desc_act` via `HFConfigLoader::load_gptq_config()`
- **Storage**: `GPTQWeight` struct in `TransformerLayer` (7 instances: q/k/v/o + gate/up/down)
- **Dequant kernel**: `src/quant/dequant_gptq.cu` — CUDA kernel for 4-bit GPTQ → FP16
- **Upload integration**: `upload_gptq_weight()` in `src/model/weight_upload.cu` — uploads, dequants, frees temporaries

### tie_word_embeddings

Auto-detected in both GGUF and SafeTensors paths. If `out_proj_.data == nullptr`, shares `tok_emb_`.

---

## 1.7 API Server Compliance

| Field | Status | Location |
|-------|--------|----------|
| `temperature`, `top_p`, `top_k` | Supported | `handlers.cpp:465-467` |
| `max_tokens` / `max_completion_tokens` | Supported | `handlers.cpp:468` |
| `seed` | Supported | `handlers.cpp:469` |
| `frequency_penalty` | Supported | `handlers.cpp:474` |
| `presence_penalty` | Supported | `handlers.cpp:475` |
| `repetition_penalty` | Supported (extension) | `handlers.cpp:473` |
| `logprobs` / `top_logprobs` | Supported (0-20) | `handlers.cpp:504-507` |
| `tools` / `tool_choice` | Supported (Jinja2 + fallback) | `handlers.cpp:535-538, 709-769` |
| `response_format` (json_object) | Supported | `handlers.cpp:514` |
| `response_format` (json_schema) | Supported | `handlers.cpp:518-523` |
| `stop` | Supported (string array) | `handlers.cpp:486-501` |
| `stream` / `stream_options` | Supported | `handlers.cpp:530-532` |
| `n` | **Supported (1-4)** | `handlers.cpp:434-435` — sequential execution, streaming restricted to n=1 |
| `logit_bias` | **Supported** | `handlers.cpp:550-557` — parsed, applied in executor |
| Token usage | Correct | prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens |

### Structured output

Fully implemented: `JsonConstrainer` (json_object mode) and `SchemaConstrainer` (json_schema mode) with schema caching.

---

## 1.8 HuggingFace Hub Integration

### Direct HF repo loading

**Supported.** `src/model/hf_hub.cpp:50-115` — `resolve_model_path()`:
1. Checks if path exists locally
2. If repo ID (contains `/`): checks HF cache (`~/.cache/huggingface/hub/models--{org}--{model}/snapshots/`)
3. Falls back to `huggingface-cli download` with output capture
4. Supports `$HUGGINGFACE_HUB_CACHE`, `$HF_HOME`, `$HOME` env vars

`resolve_model_gguf()` at line 141-168 — combines path resolution + directory scanning for `.gguf` files.

### Revision/branch/commit

**Supported.** `--revision` flag in both CLI (`tools/imp-cli/args.cpp`) and server (`tools/imp-server/args.cpp`). Passed to `huggingface-cli download --revision`.

### Auto-config discovery

**Fully implemented** for SafeTensors:
- `config.json` → `HFConfigLoader::load_config()`
- `tokenizer.json` → `Tokenizer::load()`
- `tokenizer_config.json` → `HFConfigLoader::load_chat_template()` + `load_added_tokens()`
- `generation_config.json` → `HFConfigLoader::load_generation_config()`
- `quantize_config.json` → `HFConfigLoader::load_gptq_config()`

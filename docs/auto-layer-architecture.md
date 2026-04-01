# imp Model Abstraction Layer — Architektur (Post-Implementation)

**Date:** 2026-04-01 (updated to reflect implemented state)

---

## Implemented Components

### 1. ConfigLoader (`src/model/hf_config_loader.{h,cpp}`)

```cpp
struct HFConfigLoader {
    static bool load_config(const std::string& model_dir, ModelConfig& cfg);
    static bool load_generation_config(const std::string& model_dir,
                                       std::vector<int32_t>& eos_token_ids);
    static std::string load_chat_template(const std::string& model_dir);
    static std::vector<AddedToken> load_added_tokens(const std::string& model_dir);
    static bool load_gptq_config(const std::string& model_dir, GPTQConfig& cfg);
    static ModelArch map_architecture(const std::string& hf_arch);
};
```

**Integration:** Called from `safetensors_loader.cpp` before weight assignment. config.json values are authoritative; `infer_config()` only fills fields still at zero.

**Adding a new model:** Add the HF architecture class name to `map_architecture()` (1 line in the static map).

### 2. ModelArchRegistry (`src/model/model.cpp`)

```cpp
ModelArch parse_model_arch(const std::string& s) {
    static const std::unordered_map<std::string, ModelArch> registry = {
        // 15 GGUF + 16 HF entries
    };
    auto it = registry.find(s);
    return (it != registry.end()) ? it->second : ModelArch::GENERIC;
}
```

**Adding a new architecture:**
1. Add enum value to `ModelArch` in `model_arch.h`
2. Add GGUF string + HF class name to the registry map (2 lines)
3. Add string representation in `model_arch_name()` (1 line)
4. Optionally add defaults in `apply_arch_defaults()` if needed

### 3. TokenizerFactory (distributed across `tokenizer.cpp` + `safetensors_loader.cpp`)

Loading priority in SafeTensors path:
```
1. tokenizer.json (in model directory) → Tokenizer::load()
2. GGUF metadata (for GGUF models) → gguf_loader.cpp
3. Minimal tokenizer for chat template only → safetensors_loader.cpp
```

```cpp
// src/model/tokenizer.cpp
bool Tokenizer::load(const std::string& path);
// Parses HF tokenizer.json: vocab, merges, added_tokens, pre_tokenizer type

// src/model/safetensors_loader.cpp (step 9)
if (fs::exists(model_dir + "/tokenizer.json")) {
    auto tok = std::make_unique<Tokenizer>();
    tok->load(tok_json_path);
    model->set_tokenizer(std::move(tok));
}
```

**Adding a new tokenizer type:** Extend `Tokenizer::load()` with a new `model.type` case.

### 4. ChatTemplateEngine (`src/model/chat_template.{h,cpp}`)

```cpp
class ChatTemplate {
    // Detection
    static ChatTemplateFamily detect_family(const std::string& jinja2_str);
    static ChatTemplateFamily default_family_for_arch(ModelArch arch);

    // Application
    std::vector<int32_t> apply(const Tokenizer& tok,
                               const std::vector<ChatMessage>& messages, ...) const;
    std::vector<int32_t> apply_with_tools(const Tokenizer& tok,
                                           const std::vector<ChatMessage>& messages,
                                           const std::vector<ToolFunction>& tools, ...) const;
    std::vector<int32_t> apply_with_image(const Tokenizer& tok,
                                           const std::vector<ChatMessage>& messages,
                                           int n_image_tokens, ...) const;
    bool supports_tools() const;
};
```

**Resolution order:**
1. Jinja2 template from GGUF `tokenizer.chat_template` (primary)
2. Jinja2 template from `tokenizer_config.json` (SafeTensors)
3. Hardcoded family based on architecture (fallback)
4. RAW (no template)

**Adding a new template family:** Usually not needed — Jinja2 handles most templates. If needed:
1. Add enum value to `ChatTemplateFamily`
2. Add detection pattern in `detect_family()`
3. Implement `apply_<family>()` method
4. Add case in `init()` for special token resolution

### 5. WeightMap (`src/model/weight_map.cpp`)

```cpp
class WeightMap {
    WeightMap(ModelArch arch);
    bool apply_weights(Model& model,
                       const std::unordered_map<std::string, Tensor>& tensors);
    std::string map_name(const std::string& name) const;
};
```

Pattern-based matching on HuggingFace weight names. Covers all standard + advanced patterns.

**Adding a new weight pattern:** Add a matching block in `apply_weights()` (5-10 lines).

### 6. HF Hub (`src/model/hf_hub.{h,cpp}`)

```cpp
std::string resolve_model_path(const std::string& model_id,
                                const std::string& revision = "");
std::string resolve_model_gguf(const std::string& model_id,
                                const std::string& revision = "");
bool hf_cli_available();
std::string find_gguf_in_dir(const std::string& dir);
```

**Integration:** Called from `imp-cli/main.cpp` and `imp-server/main.cpp` before model loading.

### 7. GPTQ Dequantization (`src/quant/dequant_gptq.{h,cu}`)

```cpp
void dequant_gptq4(half* out, const int32_t* qweight, const int32_t* qzeros,
                    const half* scales, const int32_t* g_idx,
                    int N, int K, int group_size, cudaStream_t stream = nullptr);
```

**Integration:** Called from `upload_gptq_weight()` in `weight_upload.cu` during model upload.

---

## Interaction with GraphExecutor

The GraphExecutor's tensor-presence-based dispatch remains unchanged:

```
Model Loading:
  GGUF → gguf_loader.cpp → Model (config + tensors + tokenizer)
  SafeTensors → safetensors_loader.cpp
                ├─ HFConfigLoader::load_config()  → ModelConfig
                ├─ Tokenizer::load()               → Tokenizer
                ├─ WeightMap::apply_weights()       → Tensors in Model
                └─ upload_gptq_weight()             → GPTQ → FP16

Runtime:
  GraphExecutor reads Model.layers_[i]
  Per-layer dispatch via tensor presence (unchanged):
    layer_has_attention() → run_attention()
    layer_has_gdn()       → run_gdn()
    layer_has_ssm()       → run_ssm()
    layer_has_moe()       → run_moe_ffn()
    layer_has_dense_ffn() → run_ffn()
```

The abstraction layer (ConfigLoader, WeightMap, etc.) ensures that `Model.layers_[i]` is correctly populated regardless of source format. The GraphExecutor doesn't need to know whether the model came from GGUF or SafeTensors.

---

## What New Models Need (Current State)

### GGUF models
**Nothing.** GGUF embeds all metadata. Architecture routing happens automatically via tensor presence.

### SafeTensors models (standard HF format)
1. Architecture class name in `parse_model_arch()` registry — **1 line** (if not already listed)
2. That's it for standard transformer variants (Llama-like, Mistral-like, etc.)

### SafeTensors models with novel layer types
1. Architecture entry (1 line)
2. Weight name patterns in `WeightMap::apply_weights()` (5-10 lines)
3. If truly novel layer type: new `run_*()` method in GraphExecutor + CUDA kernel (this is the actual compute work, not abstraction overhead)

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/model/hf_config_loader.h` | 46 | Config loader interface |
| `src/model/hf_config_loader.cpp` | 549 | config.json, generation_config, tokenizer_config, quantize_config |
| `src/model/hf_hub.h` | 31 | HF Hub resolution interface |
| `src/model/hf_hub.cpp` | 168 | HF cache check, CLI download, GGUF finder |
| `src/quant/dequant_gptq.h` | 18 | GPTQ dequant interface |
| `src/quant/dequant_gptq.cu` | ~80 | GPTQ 4-bit dequant kernel |
| `tests/test_tokenizer_compat.cpp` | ~120 | Tokenizer golden comparison test |
| `tests/generate_tokenizer_golden.py` | ~75 | HF tokenizer golden data generator |

**Total new code:** ~1090 lines across 8 new files.
**Modified code:** ~1170 lines changed across 17 existing files.

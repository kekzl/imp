# SafeTensors Loading Audit (Phase 1)

**Date:** 2026-05-07
**Scope:** Read-only audit of imp's SafeTensors loading pipeline. Goal is zero-config local model loading on RTX 5090 (sm_120a). Each finding cites `file:line`. Gaps and partial-support items are the actionable output for Phase 2+.

**Audit method:** five parallel `Explore` agents (one per subsection) cross-referenced against the running CLAUDE.md notes.

> **Status (2026-05-07):** Phase 2 has landed via PR #116 — 15 of 18 actionable items are now closed; 3 (#16-18) ship detection + warnings only and are tracked for separate follow-up PRs. See [Phase 2 outcomes](#phase-2-outcomes) below for the full mapping. Body of the audit is the original Phase-1 snapshot — line numbers were accurate at audit time but have drifted by ~50-100 lines for files modified by Phase 2.

---

## 1.1 Model Detection

### Detection flow

CLI entry → arch dispatch:
- `tools/imp-cli/main.cpp:32-67` (model path) → `src/api/imp_api.cpp:165-166` → `src/model/safetensors_loader.cpp:779-794`.
- Format auto-pick is presence-based: `model.safetensors.index.json` (sharded) or `model.safetensors`.
- `src/model/hf_config_loader.cpp:56-103` parses `config.json`. Architecture priority:
  1. `architectures[0]` only — additional entries ignored silently (`hf_config_loader.cpp:67`).
  2. `model_type` fallback (lines 71-102).
  3. Unknown → log warn → `ModelArch::GENERIC` (lines 50, 99-100).
- If `config.json` is missing or arch is `GENERIC`, the loader falls back to **tensor-name heuristics** at `src/model/safetensors_loader.cpp:322-356` (e.g., `mlp.experts` → DEEPSEEK, `mamba`/`ssm` → NEMOTRON_H_MOE, `block_sparse_moe` → MIXTRAL).
- Hyperparams missing from config are inferred from tensor shapes; per-arch defaults applied via registry at `src/model/model.cpp:117-200, 202-224`.

### Supported HF architecture classes

Mapped at `src/model/hf_config_loader.cpp:18-44`:

| imp enum            | HF class names                                                                                                                                          |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLAMA`             | `LlamaForCausalLM`, `PhiForCausalLM`, `Phi3ForCausalLM`, `Phi3SmallForCausalLM`, `InternLM2ForCausalLM`, `Starcoder2ForCausalLM`, `CohereForCausalLM`    |
| `MISTRAL`           | `MistralForCausalLM`, `Mistral3ForConditionalGeneration`                                                                                                |
| `MIXTRAL`           | `MixtralForCausalLM`                                                                                                                                    |
| `QWEN3`             | `Qwen2ForCausalLM`, `Qwen3ForCausalLM`                                                                                                                  |
| `QWEN3_MOE`         | `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM`                                                                                                            |
| `QWEN36_MOE`        | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`                                                                                           |
| `GEMMA3`            | `Gemma2ForCausalLM`, `GemmaForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`                                                          |
| `GEMMA4`            | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`                                                                                                   |
| `DEEPSEEK`          | `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`                                                                                                        |
| `NEMOTRON_H_MOE`    | `NemotronHForCausalLM`                                                                                                                                  |
| `GENERIC`           | unknown / fallback                                                                                                                                      |

### Gaps

- **`QWEN35` and `QWEN35_MOE`** enums exist (`src/model/model_arch.h:15-16`) but have **no HF class mapping** for SafeTensors. Only the GGUF path uses them. Real Qwen3.5/3.6 GDN SafeTensors models advertising `Qwen3_5MoeForCausalLM` are routed through `QWEN36_MOE`.
- **`LLAMA4`** exists in the enum (`model_arch.h:20`) but has no HF mapping; GGUF-only.
- **GLM** is not in the SafeTensors mapping at all — GGUF-only.
- **Unknown `architectures[0]`** silently downgrades to `GENERIC` and relies on tensor-name heuristics. Multi-element `architectures` arrays drop entries 1..N silently.
- No alias for community/finetune-style class names (`<Family>ForCausalLM` variants outside the enumerated list).

---

## 1.2 Tokenizer Loading

### What's loaded

SafeTensors models read tokenizer state from the model directory at `src/model/safetensors_loader.cpp:873-887` and `src/model/tokenizer.cpp:758-970`.

| File                          | Status   | Where                                                       |
|-------------------------------|----------|-------------------------------------------------------------|
| `tokenizer.json` (HF fast)    | ✅ native | `tokenizer.cpp:16-400` (custom JSON), `758-970` (orchestr.) |
| `tokenizer_config.json`       | ✅ native | `hf_config_loader.cpp:502-684`                              |
| `special_tokens_map.json`     | ✅ native | `safetensors_loader.cpp:933-960`                            |
| `added_tokens.json` / `_decoder` | ✅ native | `tokenizer.cpp:840-888`, `hf_config_loader.cpp:572-601`     |
| `chat_template.jinja` (standalone) | ✅ native | `hf_config_loader.cpp:504-568`                          |
| `tokenizer.model` (SentencePiece) | ❌ **missing** for SafeTensors path | only via GGUF (`gguf_loader.cpp:1559-1667`) |
| Tiktoken                      | ❌ missing | —                                                           |

### BPE / pre-tokenization

- Merge table parsed in `tokenizer.cpp:992-998` (both `"a b"` string and `["a","b"]` array forms).
- Encoders: `encode_gpt2()` (rank-priority heap, `tokenizer.cpp:1628-1680`) and `encode_spm()` (score-priority).
- Pre-tokenization regex variants: `gpt2_pre_tokenize()` (default) and `llama3_pre_tokenize()` (CJK-aware) — dispatched in `tokenizer.cpp:1587-1591` based on `pre_tokenizer.type` from `tokenizer.json` or `tokenizer_config.json`.
- Byte-level encoding for GPT-2 (`byte_to_gpt2()` 0xFF→Ā mapping); SPM keeps UTF-8 raw.

### Special tokens & added_tokens

- BOS/EOS auto-detected against a hardcoded list (`tokenizer.cpp:872-884`): `<s>`, `<|begin_of_text|>`, `<|startoftext|>`, `</s>`, `<|end_of_text|>`, `<|endoftext|>`, `<|eot_id|>`.
- Control tokens (`<|im_start|>`, `<start_of_turn>`, `[INST]` …) collected at `tokenizer.cpp:1087-1119` and pre-split during encode so they round-trip as a single token.
- `added_tokens.json` extends vocab; `special_tokens_map.json` is authoritative for `additional_special_tokens` and patches missing CONTROL flags via `mark_as_control()`.

### Chat template (Jinja2)

- Custom 2629-line mini-Jinja2 engine: `src/model/jinja.h`, `jinja.cpp`. Supports variables, `{% for %}`, `{% if %}`, `{% set %}`, filters, operators, `namespace()`. Template is parsed once and cached in `ChatTemplate::jinja_tpl_` (`chat_template.h:129`).
- Source priority (`hf_config_loader.cpp:504-568`): `tokenizer_config.json:chat_template` → array variant `[{name,template}]` → standalone `chat_template.jinja`.
- Parse failure → warn + fall back to a hardcoded family (CHATML / LLAMA3 / MISTRAL_V3 / DEEPSEEK_R1) selected by substring match (`chat_template.cpp:128-145`) or by arch default (`chat_template.cpp:71-102`).
- `use_default_system_prompt` honored (`tokenizer.h:47-53`, `hf_config_loader.cpp:658-684`) — necessary for Mistral-Small-3.2 (~600-token default system prompt).

### Failure modes

| Scenario                                                | Behaviour                                                                                  |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `tokenizer.json` missing                                | Returns failure; model loads but `tokenizer_` is null. **No SentencePiece fallback.**     |
| Invalid JSON / missing `model` object                   | Warn (`tokenizer.cpp:778, 785`), return false                                              |
| `vocab_size` config vs tokenizer mismatch               | Inferred from `tok_emb_` shape if config absent (`safetensors_loader.cpp:916-918`)         |
| `chat_template` absent                                  | Hardcoded family fallback; no error                                                        |
| Special token absent from vocab                         | Silent skip (`find_token() == -1`)                                                         |
| Tokenizer-config flags absent                           | Defaults: `add_bos=true` (gpt2 → false), `add_prefix_space=true`, `use_default_system=true` |

### Gaps

- **No SentencePiece (`.model`) parser on the SafeTensors path.** Older / community Llama-style SafeTensors models that ship only `tokenizer.model` cannot be loaded via SafeTensors today; must convert to GGUF or have a `tokenizer.json` available.
- **No Tiktoken parser.** Not a common HF artefact for the supported model families, but a gap for some OAI-style releases.
- BOS/EOS auto-detect is a hardcoded literal list — unusual templates (e.g., new Phi-4 turn markers) require code edits.

---

## 1.3 Weight Loading

### SafeTensors I/O

- **Native parser**, no third-party dependency. Custom JSON AST in `JsonParser` at `src/model/safetensors_loader.cpp:43-277`. Handles 8-byte size prefix + JSON metadata + binary tensor block.
- **Sharded loading** via `model.safetensors.index.json` is fully supported (`safetensors_loader.cpp:589-695`, `load_sharded()`). Per-shard parallel mmap+parse via thread pool. Unused shards skipped (vision-only / MTP-only when spec-decode off).
- **mmap path:** `mmap(PROT_READ, MAP_PRIVATE | MAP_POPULATE)` (`safetensors_loader.cpp:491`), with `MAP_POPULATE` removed on `ENOSYS` (line 498). `madvise(MADV_WILLNEED, MADV_SEQUENTIAL)` at lines 503-504. Tensors stay in mmap'd host memory; engine uploads at infer prep — no staging buffer.

### Name remapping

- Centralized at `src/model/weight_map.cpp:272-1094` (`apply_weights()`); `name_map_` for top-level and per-layer pattern matching at `weight_map.cpp:77-270`.
- Architecture-specific prefix strips and translations:
  - Gemma-4: strips `model.language_model.` (`weight_map.cpp:291-305`); fused `experts.gate_up_proj` 3D tensor (`[n_exp, 2*moe_ff, d]`) split during GPU upload (`weight_upload.cu:495-502`).
  - Nemotron-H: `backbone.layers.N.mixer.*` translation (`weight_map.cpp:315-372`).
- MoE expert weights are **not** fused at load-time — per-expert tensors:
  - Mixtral: `block_sparse_moe.experts.{e}.w{1,3,2}.weight` → `expert_w_{gate,up,down}[e]` (lines 622-638).
  - DeepSeek: `mlp.experts.{e}.{gate,up,down}_proj.weight` → indexed slots (lines 665-681).
  - Gemma-4: stays packed as 3D, split on upload.

### Layout transforms

- Q/K/V loaded as separate tensors (`q_proj`, `k_proj`, `v_proj`) — no QKV fusion at load (lines 438-450).
- No transpose/permute during load. SafeTensors shape is preserved.
- **Tied embeddings detected by null-check, not config flag:** `safetensors_loader.cpp:854-856` ties `out_proj` to `tok_emb` if `lm_head.weight` was missing. The `tie_word_embeddings` flag from `config.json` is parsed (`hf_config_loader.cpp:446-449`) but not enforced in the weight pipeline.

### Bias handling

- Presence-inferred. If `self_attn.{q,k,v}_proj.bias` is in the file it's loaded; else the slot stays null (`weight_map.cpp:800-812`). Compute paths null-check before use.
- Config flags like `attention_bias` / `mlp_bias` are **not** parsed. Risk: a config that promises bias but a SafeTensors export that omits it produces a silent null without warning.
- Dense MLP bias slots not represented in `TransformerLayer`; expert down bias not mapped (`weight_map.cpp:664-681`).

### Edge cases

- **Missing tensor with config-promised bias:** silent null. No validation pass.
- **Unknown SafeTensors dtypes:** `safetensors_dtype()` (`safetensors_loader.cpp:291-318`) warns and downcasts (F64→F32, I64→I32, F8_E5M2→FP8_E4M3). No precision validation against compute expectations.
- **Extra tensors** (vision, MTP, …): silently skipped — at shard level via the llm-compressor translator (lines 627-637) or per-tensor.
- **Unrecognized layer weights:** warn + counter (`weight_map.cpp:419, 1053`); load continues.
- **No streaming / progress callback.** All shards mmap'd in parallel and merged. With `MADV_WILLNEED`, working set can spike on a 32 GB host while loading large multi-shard models.

### Gaps

- No config-driven bias presence check (silent null on omitted-bias models).
- No checksum / safetensors-header sanity validation pass beyond JSON well-formedness.
- No progress reporting during multi-shard load (UX gap, not correctness).

---

## 1.4 Quantization Auto-Detection

### Detection sources

1. **`quantize_config.json`** (GPTQ) — `hf_config_loader.cpp:688-705`. Fields: `bits`, `group_size`, `desc_act`.
2. **`hf_quant_config.json`** (Modelopt NVFP4) — `hf_config_loader.cpp:718-769`. Fields: `quantization.quant_algo == "NVFP4"`, `group_size`, optional `kv_cache_quant_algo`.
3. **`recipe.yaml`** (llm-compressor) — same loader, parses `QuantizationModifier` → `scheme: NVFP4` or infers from `config_groups.weights.{num_bits:4, type:float}`.
4. **SafeTensors wire dtype** — `safetensors_dtype()` at `safetensors_loader.cpp:291-318`.
5. **Tensor-name heuristics** — `.qweight` suffix logged as GPTQ at `safetensors_loader.cpp:344`.

Dispatch wire-up in `load_safetensors()` at `safetensors_loader.cpp:699-851` (sets `cfg.is_nvfp4_prequant`, `cfg.is_llm_compressor_nvfp4`, populates per-layer `gptq_*` weight structs).

### Format support matrix

| Format                              | Detection                                                  | Status       | Notes                                                              |
|-------------------------------------|------------------------------------------------------------|--------------|--------------------------------------------------------------------|
| FP16                                | wire `"F16"`                                               | ✅            | `safetensors_loader.cpp:294`                                       |
| BF16                                | wire `"BF16"`                                              | ✅            | `safetensors_loader.cpp:296`                                       |
| FP8 E4M3                            | wire `"F8_E4M3"`                                           | ✅            | `safetensors_loader.cpp:312-313`                                   |
| FP8 E5M2                            | wire `"F8_E5M2"`                                           | ⚠ proxy       | mapped to E4M3 (line 314-315), lossy                                |
| INT8 / INT8 weight-only             | wire `"I8"`, `"U8"`                                        | ✅            | `safetensors_loader.cpp:300-303`                                   |
| INT4 GPTQ                           | `quantize_config.json` `bits:4`                            | ✅            | `dequant_gptq.cu`                                                  |
| INT4 AWQ                            | —                                                          | ❌ missing    | no `awq_config.json` parsing                                       |
| NVFP4 (Modelopt)                    | `hf_quant_config.json` + FP8 micro-scales + FP32 tensor scale | ✅          | `executor_pre_dequant.cu:387-475`, CUTLASS sm_120 fast-path         |
| NVFP4 (llm-compressor)              | `recipe.yaml`                                              | ✅            | `is_llm_compressor_nvfp4` flag, divisor-style scale (`model_config.h:83-85`) |
| MXFP4 (GPT-OSS / Qwen3.5-mxfp4)     | GGUF wire type 31                                          | ⚠ GGUF only   | **No SafeTensors detection path**                                  |
| Mixed precision per-layer           | per-tensor dtype + arch-specific layer types               | ✅            | Qwen3.5 GDN, Nemotron-H hybrid, Gemma-4 alt SWA                     |

### Format quirks

- **Modelopt vs llm-compressor NVFP4:** different scale conventions. Modelopt: `val = fp4 * weight_scale_fp8 * weight_scale_2`. llm-compressor: `val = fp4 * weight_scale_fp8 / weight_global_scale` (divisor). Runtime flag set at `safetensors_loader.cpp:848`.
- **`input_scale` loaded but not applied** at inference (`executor_pre_dequant.cu:382`). Memory cross-ref: see `llm_compressor_input_scale_dead_end_2026_05_07.md` — the scale absorption hypothesis was refuted A/B; the chunked-prefill bug is a separate issue and is now fixed via single-chunk default.
- **Modelopt `kv_cache_quant_algo == "FP8"`** is parsed and stored (`hf_config_loader.cpp:745-747`) but **not enforced** — engine respects user `--kv-fp8` / `--kv-nvfp4` flags instead. Auto-FP8 KV from model metadata is missing.
- **GPTQ `desc_act`** parsed (line 699-700) but no evidence of consumption in dequant kernels.
- **llm-compressor naming translation:** `.weight_packed` → `.weight`, `.weight_global_scale` → `.weight_scale_2`, `.input_global_scale` → `.input_scale`, plus prefix strips and vision-tower skip (`llm_compressor_loader.cpp:99-158`). Skip-guard rationale documented in memory (`llm_compressor_cutlass_skip_2026_05_05.md`).

### Unsupported-format behaviour

- Missing GPTQ/NVFP4 config → tensors load with raw dtype, no quant struct. Inference may proceed in FP16/FP8 but without the quantization metadata the kernels need.
- llm-compressor `recipe.yaml` with non-NVFP4 scheme → **hard error** at `llm_compressor_loader.cpp:300-302` (returns false, blocks load).
- Phase-1 dequant-to-FP16 fallback exists only for GGML block-quant types (`dequant_gpu.cu:11-28`: Q4_0/1, Q5_0/1, Q2/3/4/5/6/8_K). SafeTensors-only formats (AWQ, MXFP4-via-SafeTensors) have no fallback path.

### Format → kernel dispatch

- NVFP4: Phase 0 promotes scale sidecars (`executor_pre_dequant.cu:211-383`); Phase 0b builds CUTLASS NVFP4 cache or dequant→cuBLAS fallback (lines 387-475). Decode dispatch at `executor_forward.cu:677, 703-704`.
- MXFP4: GGUF-only wire decode + `mxfp4.linear_scales` runtime check at `executor_forward.cu:583, 663, 677`.
- FP8: Phase 2 cache build (`executor_pre_dequant.cu:491-523`) + cuBLASLt dispatch at `executor_forward.cu:755-759`.
- GPTQ: per-layer struct populated at `safetensors_loader.cpp:825-836`; kernel `dequant_gptq.cu`.

### Gaps

1. **MXFP4 SafeTensors detection missing.** Models exported as SafeTensors with MXFP4 weights load as raw FP16, losing the quantization. Needs `quantization_config` parsing parallel to the NVFP4 path.
2. **AWQ unimplemented.** No config parsing, no kernel dispatch.
3. **FP8 KV-cache auto-negotiation absent.** Modelopt metadata is read but ignored.
4. **`input_scale` absorption / SmoothQuant.** Loaded but not used.
5. **`desc_act` GPTQ ordering.** Parsed but not applied; risk of quality drop on `desc_act:true` models.

---

## 1.5 Architecture Quirks

Per-feature support map. ✅ supported / ⚠ partial / ❌ missing. "Auto" = config-driven, "HC" = hardcoded per-arch.

| Feature                              | Status | Trigger     | Source                                                                                       |
|--------------------------------------|--------|-------------|----------------------------------------------------------------------------------------------|
| RoPE linear scaling                  | ✅      | Auto        | `hf_config_loader.cpp:168-169`                                                              |
| RoPE dynamic (NTK)                   | ✅      | Auto        | `hf_config_loader.cpp:197-199`                                                              |
| RoPE YaRN                            | ✅      | Auto        | `hf_config_loader.cpp:170-177`, `model_config.h:50-54`                                       |
| RoPE Llama-3 scaling                 | ❌      | —           | no `low_freq_factor` / `high_freq_factor` parsing                                            |
| RoPE LongRoPE (Phi-4)                | ✅      | Auto        | `hf_config_loader.cpp:178-195` (short_factor/long_factor)                                    |
| Partial RoPE                         | ✅      | Auto        | `hf_config_loader.cpp:145-152` (`partial_rotary_factor`)                                     |
| Per-layer rope_freqs (Gemma-4)       | ✅      | HC          | `model.cpp:202-224` + `gguf_loader.cpp:1503-1539` (rope_freqs.weight)                        |
| MHA / GQA / MQA                      | ✅      | Auto        | `num_key_value_heads` (`hf_config_loader.cpp:121-123`)                                       |
| MLA (DeepSeek V2/V3 latent attn)     | ❌      | —           | no MLA-specific fields parsed; `DeepseekV*ForCausalLM` mapped to standard DEEPSEEK arch       |
| Sliding window (global)              | ✅      | Auto        | `hf_config_loader.cpp:203-204`                                                              |
| Per-layer SWA (Gemma-4 alternating)  | ✅      | Auto        | `layer_types[]` → `swa_layers[]` (`hf_config_loader.cpp:403-418`)                            |
| Attention sinks (StreamingLLM)       | ⚠      | HC          | `streaming_kv_n_sinks` in EngineConfig (`engine.h:88-96`); not parsed from model config       |
| Logits soft-cap (Gemma)              | ✅      | Auto        | `attn_logit_softcapping`, `final_logit_softcapping` (`hf_config_loader.cpp:207-210`)         |
| MoE top-k routing                    | ✅      | Auto        | `num_experts_per_tok` (`hf_config_loader.cpp:227-228`)                                       |
| MoE shared experts                   | ✅      | Auto        | `n_shared_experts` (`hf_config_loader.cpp:330-334`)                                          |
| MoE router precision                 | ⚠      | HC          | FP32 router for Gemma-4 hardcoded; no config field                                            |
| MoE expert biases / aux-loss         | ⚠      | partial     | `moe_router_bias` loaded, `norm_topk_prob` parsed (`hf_config_loader.cpp:339-342`); aux-loss coeffs not exposed |
| Per-expert vs fused expert layout    | ✅      | Auto+HC     | per-expert (Mixtral, DeepSeek), fused 3D (Gemma-4 — `expert_{gate,up,down}_packed`)           |
| Mamba2 / GDN hybrid                  | ✅      | Auto        | `linear_*` fields, `layer_types[]`, `hybrid_override_pattern` (`hf_config_loader.cpp:234-376`) |
| GDN head layout (grouped vs tiled)   | ⚠      | HC          | grouped=true for SafeTensors, false for GGUF (`hf_config_loader.cpp:247-252`)                 |
| Vision tower (SigLIP)                | ⚠      | HC          | mmproj GGUF only (`vision_loader.cpp:368-396`); no `vision_config` from `config.json`         |
| Vision (Qwen-VL / Llava / Pixtral)   | ❌      | —           | no SafeTensors loaders                                                                       |
| Tied embeddings                      | ⚠      | inferred    | `tie_word_embeddings` parsed but not enforced; ties via null-check (`safetensors_loader.cpp:854-856`) |
| RMSNorm `1+W` (Gemma-3)              | ✅      | HC          | `norm_weight_offset = 1.0f` (`model.cpp:214-215`); Gemma-4 uses raw weights                   |
| QK-norm (per-head RMS)               | ✅      | tensor pres | `attn_q_norm`, `attn_k_norm`; fused with RoPE (`compute/rope.h:29-34`)                       |
| LayerNorm fallback                   | ❌      | —           | RMSNorm only                                                                                 |
| Pre/post-norm placement              | ⚠      | HC          | per-arch in `apply_arch_defaults()` (`model.cpp:218-219`)                                    |
| Activation detection                 | ✅      | Auto        | `hidden_act` → SWIGLU / GEGLU (`hf_config_loader.cpp:213-220`)                                |
| Activation: ReLU² (Nemotron-H)       | ✅      | HC          | `apply_arch_defaults()` (`model.cpp:220-221`)                                                |

Tested model coverage (from CLAUDE.md memory): Gemma-4 (NVFP4 + Q4/Q5/Q8), Qwen3 / Qwen3.5 GDN / Qwen3.6 MoE, Nemotron-H NVFP4, Phi-4 LongRoPE, Mistral-3.2-NVFP4, Llama-3.2-3B Q8.

---

## Cross-cutting findings & actionable items

Ordered by user-impact for "any local SafeTensors model just works":

### Hard gaps — block unsupported model classes

1. **DeepSeek MLA (V2/V3).** Class names map to `DEEPSEEK` but no MLA-specific attention is implemented. Real DeepSeek-V2/V3 SafeTensors checkpoints will load but produce wrong outputs.
2. **AWQ INT4.** No config or kernel path. Common community quantization, especially for Llama/Qwen finetunes.
3. **MXFP4 on SafeTensors.** Detection only on the GGUF wire format. SafeTensors MXFP4 exports degrade silently to FP16 storage.
4. **SentencePiece tokenizer (`tokenizer.model`)** on SafeTensors path. Many older Llama/Mistral checkpoints ship only this file.
5. **RoPE Llama-3 scaling.** `low_freq_factor` / `high_freq_factor` not read; long-context Llama-3.x quality regression.
6. **Multimodal SafeTensors loaders** beyond mmproj-GGUF SigLIP (Qwen-VL, Llava, Pixtral, Gemma-3 vision-from-SafeTensors).
7. **GLM, Llama4, Qwen3.5 (non-MoE)** — enums or HF mappings missing on SafeTensors path.

### Soft gaps — silent quality drops

8. **`tie_word_embeddings` flag ignored.** Detection is null-check based; a model with `tie=false` but missing `lm_head.weight` will silently tie.
9. **GPTQ `desc_act` parsed but unused.** Models with `desc_act:true` may degrade.
10. **`input_scale` (NVFP4) loaded but not applied.** No SmoothQuant integration. (Hypothesis already separately refuted as cause of long-context bug — see memory `llm_compressor_input_scale_dead_end_2026_05_07.md`.)
11. **FP8 E5M2 → E4M3 silent downcast.**
12. **Bias presence not validated against `attention_bias` / `mlp_bias` config flags.** Silent null on omitted-bias exports.
13. **GDN head layout grouped/tiled** chosen by loader (SafeTensors vs GGUF), not by model metadata. Cross-converted models can mis-route.

### UX / observability gaps

14. **No multi-shard load progress reporting.** Memory pressure visible only in `dmesg` if working set spikes.
15. **Unknown `architectures[0]` quietly downgrades to `GENERIC`** with weight-name heuristics. No machine-readable signal to the caller that detection failed.
16. **Multi-element `architectures` arrays drop entries silently.**
17. **`recipe.yaml` non-NVFP4 schemes hard-error.** Should warn and fall back to dequant-to-FP16 where possible.

### Auto-detection gaps that should be cheap to close

18. Auto-FP8 KV from Modelopt `kv_cache_quant_algo`.
19. RoPE Llama-3 scaling parser.
20. AWQ config parser (kernel can land separately).
21. MXFP4 from `quantization_config` on SafeTensors.
22. Native SentencePiece parser (third-party-free implementation similar to `tokenizer.cpp` is feasible — ~few hundred LoC).

---

## Reference: file map

```
src/api/imp_api.cpp                    # CLI → loader entry
src/model/model_arch.h                 # ModelArch enum
src/model/model.cpp                    # arch registry, apply_arch_defaults()
src/model/hf_config_loader.{h,cpp}     # config.json / quantize_config / recipe.yaml / chat_template
src/model/safetensors_loader.cpp       # JsonParser, mmap shard load, dispatch
src/model/weight_map.cpp               # name remapping, MoE expert indexing
src/model/llm_compressor_loader.cpp    # llm-compressor name translation
src/model/tokenizer.{h,cpp}            # native tokenizer.json + BPE/SPM encoders
src/model/jinja.{h,cpp}                # mini-Jinja2 (2629 lines)
src/model/chat_template.{h,cpp}        # template family fallback
src/graph/executor_pre_dequant.cu      # NVFP4/FP8/GGML phase build
src/graph/executor_forward*.cu         # runtime dispatch
src/quant/dequant_*.cu                 # GGML & GPTQ dequant kernels
src/vision/vision_loader.cpp           # mmproj SigLIP loader
```

---

## Phase 2 outcomes

Status of the 22 actionable items (1–17 are unique; 18–22 in the Phase-1 list were "cheap auto-detect" duplicates of items above) after PR #116:

### ✅ Fully resolved (15)

| #  | Item                                                              | Commit         | Notes                                                                                                                          |
|----|-------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| 1  | RoPE Llama-3 scaling parser                                       | `2a52bc0`      | Per-pair factor table built at parse time, reuses LongRoPE infra. Unit test for HF-published Llama-3.1-8B values.              |
| 2  | Auto-FP8 KV from Modelopt metadata                                | `2a52bc0`+`2aade68` | Hint surfaced (`cfg.kv_cache_quant_hint`); engine logs author intent but does not auto-flip (correctness varies by family). |
| 3  | Enforce `tie_word_embeddings` flag                                | `2a52bc0`+`2aade68` | Tri-state parsed; loader cross-checks vs `lm_head.weight` presence and warns on mismatch.                                     |
| 4  | Warn on FP8 E5M2 → E4M3 silent proxy                              | `2aade68`      | One-shot WARN at first occurrence (avoids per-tensor spam).                                                                    |
| 5  | Validate `attention_bias` / `mlp_bias` config flags vs tensors    | `2a52bc0`+`2aade68` | Tri-state parsed; per-layer null-check + summary WARN at end of load.                                                          |
| 6  | Surface unknown-architecture detection                            | `2a52bc0`+`2aade68` | `arch_inferred_fallback` flag + actionable WARN with "add a class mapping" advice.                                            |
| 7  | Warn on multi-element `architectures` arrays                      | `2a52bc0`      | Lists dropped entries explicitly.                                                                                              |
| 8  | Warn-and-fallback on non-NVFP4 `recipe.yaml` schemes              | `a0e9734`      | Soft-fail returns false from `load_nvfp4_config()`; SafeTensors loader proceeds with wire dtype.                              |
| 9  | Multi-shard load progress reporting                               | `2aade68`      | Atomic counter + `[i/N] mmap'd shard …` per worker.                                                                            |
| 10 | Plumb GPTQ `desc_act` through dequant                             | `28814e6`      | Flag propagated to `GPTQWeight`; warn if `desc_act:true` and `g_idx` tensor absent (silent miscompute path).                  |
| 11 | NVFP4 `input_scale` decision                                      | `4e8b923`      | Documented as audit-only (refuted as long-context-bug cause); skip prod GPU upload, save VRAM.                                |
| 12 | GDN head layout from metadata                                     | `01fdcd8`      | Default still loader-path-driven; `IMP_GDN_LAYOUT=tiled\|grouped` env override for cross-converted checkpoints.               |
| 13 | MXFP4 detection on SafeTensors                                    | `bda0c3d`      | `quantization_config.quant_method == "mxfp4"` parsed → flag + WARN. **Decode path is GGUF-only — future work.**               |
| 14 | SentencePiece `tokenizer.model` path                              | `da229ad`      | Actionable IMP_LOG_ERROR with the conversion recipe, instead of null-tokenizer crash. **Native parser is future work.**       |
| 15 | Map Llama-4 / Qwen3.5 non-MoE on SafeTensors                      | `bda0c3d`      | `Llama4ForCausalLM` → `LLAMA4`, `Qwen3_5ForCausalLM` → `QWEN35`. **GLM intentionally not mapped** (no real impl path).         |

### ⚠️ Detection + warning only (3) — implementations are future work

| #  | Item                          | Commit     | What landed                                                                                                                              | What's missing                                                                                          |
|----|-------------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 16 | AWQ INT4                      | `7c0b8c8`  | `HFConfigLoader::load_awq_config()` covers HF-standard nesting + AutoAWQ legacy field names; loader emits WARN naming bits/group_size.   | Native AWQ dequant kernel (or dequant-to-FP16 fallback). Direction users to GPTQ / NVFP4 for now.       |
| 17 | DeepSeek MLA (V2/V3)          | `52e0ef0`  | `kv_lora_rank > 0` or `q_lora_rank > 0` → load-time WARN that DEEPSEEK forward path uses MHA and produces incorrect outputs.             | MLA-aware attention path (`q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`). Multi-week effort. |
| 18 | Multimodal SafeTensors loaders | `52e0ef0` | `vision_config` block presence → WARN naming `model_type`. Vision tower silently skipped today.                                          | Per-family loaders (Qwen-VL, Llava, Pixtral, Gemma-3 vision-from-SafeTensors), vision encoder, prefix-injection wiring. |

### Resolution method (where to look)

- All Phase-2 commits land on branch `chore/safetensors-audit-phase-2` (PR #116). Cherry-picked from the original `fix/gemma4-long-context-chunked-prefill` branch.
- New unit tests in `tests/test_hf_config_loader.cpp` cover Llama-3 RoPE, tri-state flags, arch fallback, MXFP4 / AWQ / MLA / vision detections, and the newly-mapped HF arch class names. 9 new tests, all green.
- The chunked-prefill default-fix that originally lived on the same dev branch is shipping separately as PR #117.

### Items that remain truly unresolved

- **GLM** — class not mapped, intentionally. Adding `GlmForCausalLM` → `LLAMA` would silently produce wrong outputs because GLM's architecture (especially in earlier ChatGLM variants) diverges from LLAMA enough to matter. Real fix: add a `GLM` enum entry + dedicated forward path.
- **Native SentencePiece (.model) parser** — protobuf decoder + Unigram model decoder + byte-fallback handling, ~few hundred LoC. Workaround documented in the WARN.
- **AWQ dequant kernel** — packing convention differs from GPTQ (column-packed + interleave permutation). MVP path: dequant-to-FP16 + cuBLAS.
- **DeepSeek MLA attention** — proper multi-head latent attention path. Multi-week effort.
- **Multimodal SafeTensors loaders** — per-family work (Qwen-VL, Llava, Pixtral, Gemma-3 vision).
- **Tiktoken parser** — uncommon in supported families; ignored.

### Implementation snapshot — verification

- 322 unit tests pass on the audit branch (1 skipped baseline). 9 new tests added.
- Pre-push hook `verify-fast` (build + filtered tests + perf gate + Qwen3-4B Q8_0 smoke prompt distinct=8 contains 'Paris') passes cleanly on each push.
- No changes to compute kernels, no perf-baseline impact (`tests/perf_baseline.json` unchanged).

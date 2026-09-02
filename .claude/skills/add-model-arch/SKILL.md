---
name: add-model-arch
description: Use when adding support for a new model architecture to imp, porting a model family, or debugging a model that loads but produces wrong output - "add support for <model>", "new arch", loader detection, chat template, tokenizer parity, RoPE variant, "outputs garbage", "prompt-blind", "digits scrambled", "NaN logits", "describes a different picture", "does it fit in VRAM". Do NOT use for kernel performance (sm120-cuda-expert) or quant-format questions (quant-formats).
---

# Adding a Model Architecture - imp

## First: is it a new arch at all?

Diff `config.json` against a supported sibling before estimating. Qwen3.8 shipped with zero enum members (loads as `QWEN35`); the work was tokenizer parity, template goldens, KV dtype default and MTP head (#1750). HF reference values come from `curl`, never from memory.

## Integration checklist (gpt-oss #572 is the reference PR)

| Step | Where | Notes |
|---|---|---|
| 1. Enum + registry | `ModelArch` in `src/model/model_arch.h`; `parse_model_arch` (GGUF `general.architecture` in `gguf_loader.cpp`, HF `architectures`/`model_type` in `hf_config_loader.cpp`), `model_arch_name`, `apply_arch_defaults`, sampling defaults in `src/model/model.cpp` | then `ModelProfile` (`src/model/model_profile.h/.cpp`, SSoT since #622/#623; `AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE, MLA }`). No new `cfg.arch == X` in hot paths. Add the arch to the KV-dtype lists `kv_nvfp4_default_safe`, `kv_fp8_hint_default_safe`, `kv_fp8_no_hint_default_safe` (evidence per family in `model.cpp`): a missing entry silently gets FP16 KV, which on a GDN hybrid gates context |
| 2. Loader | `src/model/tensor_kind_matcher.cpp`, `weight_map.cpp`; SafeTensors `safetensors_loader.cpp`; NVFP4 via `llm_compressor_loader.cpp` (compressed-tensors) or Modelopt (`hf_quant_config.json`) | the two NVFP4 layouts have RECIPROCAL tensor scales (quant-formats) |
| 3. Arch config | `model_config.h` + `apply_arch_defaults` | RoPE pair layout (NeoX vs GPT-J), YaRN/`rope_freq_scale`, SWA layer pattern, NoPE, sinks, softcap, norm placement, MoE router |
| 4. Chat template | family in `src/model/chat_template_families.cpp` (`ChatTemplateFamily` in `chat_template.h`), rendering in `chat_template.cpp` (+ `jinja.cpp`) | a new family needs a golden pin (`make chat-goldens`, `tests/refs/chat_template_goldens.h`, nine families since #1721, three more Jinja gaps fixed in #1701; Jinja fails SILENTLY). `reasoning_effort` must reach `ChatTemplate::apply*`/`render_jinja` + the server snapshot: identical prompt-token counts across efforts = not threaded (#1750: 67/67 before, 41/11/53 after) |
| 5. Tokenizer parity | template `tests/test_tokenizer_qwen38.cpp` (32/32 encode+decode vs HF), `tests/test_qwen38_chat_template.cpp` | BERT-family GGUFs use SPM, not WordPiece |
| 6. Kernels | only for genuinely new ops; check `src/exec/` + `src/compute/` first | new RoPE variants go into `src/compute/rope_yarn.cuh` (shared with MTP heads, #913) |
| 7. Verify | loads -> greedy coherent (check-degeneration) -> `imp-cli --perplexity` vs HF (within ~10-20%, often closer: gpt-oss 4.68 vs bf16 4.607, #663) -> decode/prefill sanity (benchmark-cuda) | PPL with `runtime.deterministic=true`, `speculative.mtp_k=0`, `ppl_corpus_45k.txt` |
| 8. Docs | `docs/MODELS.md` row (+ `docs/BENCHMARKS.md` if hero); known gaps to `docs/LIMITATIONS.md`; perf baseline entry if gated | |

A new checkpoint is UNTRUSTED INPUT: SafeTensors/`tokenizer.json` parsers are hardened (#1660, #1694); fuzz targets under `fuzz/`; no parsing shortcuts around the bounds checks.

## Wrong-output fingerprints

| Symptom | Root-cause class | Case |
|---|---|---|
| Fluent but ignores the prompt ("prompt-blind") | RoPE pair layout: HF SafeTensors need `rope_neox=true`, GGUF pre-permutes Q/K | SafeTensors Llama/Mistral, #503 |
| Words fine, digits scrambled | position encoding (NoPE layer as RoPE or vice versa) | Nemotron-H `rope_attn_disabled`, #518 |
| Argmax always token 0 | NaN logits upstream (residual overflow, bad scale) | gpt-oss FP16 residual |
| Coherent to ~1k ctx, then garbage | YaRN/`rope_freq_scale` inverted or fused-rope path without YaRN | gpt-oss 1024x error, #572 |
| Wrong only with chunked prefill at long ctx | continuation-chunk path | #553 |
| Wrong language / valid-but-wrong tokens | weight upload / dequant layout (MoE: `weight_upload.cu` expert promotion first) | Qwen3.6-35B NVFP4, #925 |
| Garbage from token 0 (`!!!`) | silent VRAM-alloc failure in a decode fallback | MXFP4 GDN hybrids, #935 |
| Multimodal: describes a DIFFERENT picture | M-RoPE per-token (t,h,w) layout, `src/model/mrope_positions.cpp` | Qwen3-VL |
| Vision fluent but generic | tower loaded partly or embeddings never reach the sequence: `tools/analysis/vision_sight_check.py` | |
| Drift only at very long positions | YaRN float trap: `__sinf/__cosf` on an unreduced argument; long-ctx tests run `ext_factor=0` and cannot see it | #1704 |
| Coherent-ish, drifts vs HF | MLA/YaRN `rope_mscale` on the wrong base (1.261x); pin the transformers oracle (4.44.2 for MLA) | #880 |
| Draft accept collapses, output correct | MTP head RoPE differs from the target (NeoX vs YaRN) | #913 |
| CLI fine, server broken | not arch: server-api | |
| Which block diverges | `diagnostics.dump_hidden_dir` + `tools/analysis/layer_diff.py` (vs llama.cpp `llama-eval-callback`) or `layer_ab_diff.py` (two imp runs); `dump_gdn_state_dir`, `dump_logits_dir` | 0 non-finite GDN states over a 46579-token prefill on Qwen3.8 |

## Known traps

- `rope_neox`: GGUF converters pre-permute Q/K; HF SafeTensors do not.
- `swa_layers` was Gemma-only hardcoded once; verify per-layer attention type on any interleaved-SWA arch.
- The fused rope+KV-write kernel must apply the same YaRN scaling as the standalone path.
- Arch control tokens (Harmony channels) must not land on the banned list; the spec-verify argmax applies the mask since #1796.
- Gemma-4: per-layer `rope_freqs` for non-SWA layers, `n_rot=hd`.
- GDN/hybrid state: BF16 storage + FP32 arithmetic is the default (`gdn.state_bf16`, #1776/#1778); FP16 state NaNs at depth; the old "must be FP32" was a layout bug.
- Multimodal checkpoints wrap the LM under `model.language_model.*` (Qwen3.5-VL, #647): strip in the loader or every tensor "is missing".
- Encoder/embedding archs are supported (nomic-bert #867, cosine 0.999 vs HF).
- VRAM arithmetic: card 32 607 MiB; CUDA primary context ~1680 MiB; library reserve is a per-model MEASURED value cached in `src/memory/library_reserve_cache.h` (#1119; `vram.library_reserve_cache`, override `vram.library_reserve_mb`); the `kMeasuredLibraryReserveBytes` ~3900 MiB constant in `src/memory/plan.h` is the first-run fallback (measured 0 MiB on Qwen3-4B-IQ4_NL, 7460 on Qwen3-8B-Q8_0). An MTP head adds ~0.79 GiB when `speculative.mtp_k` auto takes it. Peaks per config: `docs/internals/MEMORY.md`.
- Getting a checkpoint onto the NVFP4 path: `scripts/stage-model.sh` (download + `imp-quantize` in one command; imp fetches nothing itself).

## After it works

`make verify-fast`; a `DegenerationTest`-compatible probe prompt; think-channel archs through `tools/analysis/degen_suite.py` (think-leak); vision through `vision_sight_check.py` and `make test-vision`.

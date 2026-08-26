---
name: add-model-arch
description: Use when adding support for a new model architecture to imp, porting a model family, or debugging a model that loads but produces wrong output — "add support for <model>", "new arch", loader detection, chat template, RoPE variant, "outputs garbage", "prompt-blind", "digits scrambled", "NaN logits". Do NOT use for kernel performance (sm120-cuda-expert) or quant-format questions (quant-formats).
---

# Adding a Model Architecture — imp

## Integration checklist (gpt-oss PR #572 is the reference example)

1. **Enum + registry**: add to `ModelArch` in `src/model/model_arch.h`; wire `parse_model_arch` (GGUF `general.architecture`, `gguf_loader.cpp`) and/or HF detection (`hf_config_loader.cpp` - `architectures` array, `model_type` fallback), `model_arch_name`, `apply_arch_defaults`, sampling defaults in `src/model/model.cpp` (registry + `parse_model_arch`/`apply_arch_defaults` live there; `src/model/model_arch.h` holds the enum + decls). **Then register the arch's traits in `ModelProfile`** (`src/model/model_profile.h`/`.cpp` - single source of truth since PRs #622/#623; `AttnVariant` is now `{STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE, MLA}`). Never add new `cfg.arch == X` checks in hot-path code - the profile is what dispatch reads. **And add the arch to the KV-dtype safety lists** declared in `model_arch.h` (`kv_nvfp4_default_safe`, `kv_fp8_hint_default_safe`, `kv_fp8_no_hint_default_safe`, evidence per family in `model.cpp`) - a missing entry silently gets FP16 KV, which on a GDN hybrid gates context length. First check whether it is a new arch AT ALL: Qwen3.8 shipped with zero enum members added (loads as `QWEN35`; diff `config.json` before estimating).
2. **Loader**: tensor-name mapping in `src/model/tensor_kind_matcher.cpp` / `weight_map.cpp`; SafeTensors path in `safetensors_loader.cpp` (NVFP4 prequant via `llm_compressor_loader.cpp` if applicable).
3. **Arch config**: RoPE variant (NeoX vs GPT-J pair layout! see traps), YaRN/`rope_freq_scale`, SWA layer pattern, attention quirks (NoPE, sinks, softcap), norm placement, MoE router type — in `model_config.h` + `apply_arch_defaults`.
4. **Chat template**: family registration in `src/model/chat_template_families.cpp` (`ChatTemplateFamily` enum in `chat_template.h`), rendering in `chat_template.cpp` (+ `jinja.cpp` if templated); think/reasoning channel handling if applicable. **A new family needs a golden pin** (#1721 pinned nine families and exposed two silent Jinja gaps; #1701 fixed three more - Jinja fails SILENTLY, so a render that "works" is not evidence). If the model reads `reasoning_effort`, thread it through `ChatTemplate::apply`/`apply_with_tools`/`apply_with_image`/`render_jinja` + the server snapshot; the symptom of missing it is identical prompt-token counts across efforts (#1750: 67/67 before, 41/11/53 after).
5. **Kernels** only if genuinely new ops (sinks, new gating) — check `src/exec/` + `src/compute/` for an existing path first.
6. **Verify** (in order): loads → coherent greedy output (run `check-degeneration` battery) → **perplexity vs HF reference** (`imp-cli --perplexity`; expect within ~10-20% of HF — often much closer, e.g. gpt-oss imp 4.68 vs HF bf16 4.607, #663: the residual elevation is model-intrinsic) → decode/prefill sanity (`benchmark-cuda`).
7. **Docs**: row in `docs/MODELS.md` (+ docs/BENCHMARKS.md if hero-class); what is known NOT to work goes to `docs/LIMITATIONS.md`; perf baseline entry if it becomes a gated model.

A new-arch checkpoint is UNTRUSTED INPUT: the SafeTensors/tokenizer.json parsers were hardened against OOB and attacker-sized allocations (#1660, #1694) - don't add parsing shortcuts that bypass the bounds checks, and fuzz targets exist under `fuzz/` for new parser surface.

## Diagnostic fingerprints (wrong-output triage)

| Symptom | Root-cause class | Historical case |
|---|---|---|
| Fluent text but ignores the prompt ("prompt-blind") | **RoPE pair layout** — HF SafeTensors need `rope_neox=true`; GGUF pre-permutes Q/K | whole SafeTensors Llama/Mistral family, PR #503 |
| Words fine, digits/numbers scrambled | Position encoding bug (NoPE layer treated as RoPE, or vice versa) | Nemotron-H `rope_attn_disabled`, PR #518 |
| Argmax always token 0 | NaN logits upstream (residual overflow, bad scale) | gpt-oss FP16 residual overflow |
| Coherent until ~1k ctx, then garbage | YaRN/`rope_freq_scale` inverted or fused-rope path missing YaRN | gpt-oss: inverted scale = 1024× error, PR #572 |
| Long-context wrong only with chunked prefill | continuation-chunk path | PR #553 |
| Wrong language / valid-but-wrong tokens | weight upload / dequant layout, not the arch code (MoE: check `weight_upload.cu` expert promotion first) | Qwen3.6-35B NVFP4, PR #925 |
| Garbage from token 0 (`!!!…`) | silent VRAM-alloc failure in a decode fallback, not arch code | MXFP4 GDN hybrids, PR #935 |
| Multimodal: describes a DIFFERENT picture, no crash | M-RoPE per-token (t,h,w) position layout wrong - `src/model/mrope_positions.cpp` (its header states this fingerprint verbatim) | Qwen3-VL port |
| Correct output that drifts only at very long positions | YaRN float-precision trap: `__sinf/__cosf` on an unreduced argument; long-context tests run `ext_factor=0` (linear branch) and cannot see it | #1704 |
| CLI fine, server broken | not an arch bug — see `server-api` skill |

## Known traps (each cost a debugging session)

- **`rope_neox`**: GGUF converters pre-permute Q/K; HF SafeTensors do NOT. Llama-family SafeTensors without `rope_neox=true` = prompt-blind.
- **SWA layer masks**: the `swa_layers` pattern was Gemma-only hardcoded once — verify per-layer attention type for any interleaved-SWA arch.
- **Fused-rope-KV vs YaRN**: the fused rope+KV-write kernel must apply the same YaRN scaling as the standalone path.
- **Banned-token list vs channel tokens**: arch-specific control tokens (Harmony channels) must not land on the banned list.
- **Per-layer rope_freqs** (Gemma-4): non-SWA layers need their own freqs, `n_rot=hd`.
- **h_state precision** (GDN/hybrid): FP16 state NaNs at depth (subnormal truncation). BF16 storage with FP32 arithmetic is the shipped default (`gdn.state_bf16`, #1776/#1778); the old "must be FP32" note was a layout bug, not numerics - see sm120-cuda-expert known-issues.
- **HF tensor-name prefixes**: multimodal checkpoints wrap the LM under e.g. `model.language_model.*` (Qwen3.5-VL, PR #647) — strip the prefix in the loader or every tensor "is missing".
- **MLA/YaRN `rope_mscale`** (#880): the mscale ratio applied to the wrong base inflated RoPE by 1.261× — coherent-ish output that drifts vs HF. When comparing against a transformers oracle, PIN the transformers version (4.44.2 was the validated MLA oracle).
- **Draft/MTP heads must share the main model's exact RoPE math** — an MTP head computing plain NeoX while the target uses YaRN drifts the drafter (accept rate collapses, output stays correct). Shared impl: `src/compute/rope_yarn.cuh` (PR #913). Any new RoPE variant goes there, not into per-kernel copies.
- **Encoder/embedding archs are supported** (nomic-bert, PR #867 — cosine 0.999 vs HF). Gotcha: BERT-family GGUFs use an SPM tokenizer, not WordPiece-as-expected.
- Model too big? Do the arithmetic instead of trusting a remembered ceiling: the card is 32 607 MiB, the CUDA primary context takes ~1680 MiB before imp allocates anything, and the library reserve (cuBLAS/CUTLASS) is a **per-model measured value cached in `src/memory/library_reserve_cache.h`** since #1119 - the `kMeasuredLibraryReserveBytes` ~3900 MiB constant in `src/memory/plan.h` is only the first-run fallback and is wrong in both directions (measured 0 MiB on Qwen3-4B-IQ4_NL, 7460 on Qwen3-8B-Q8_0; accounting 82.5% with the constant vs 98.3% measured). Override: `library_reserve_mb`. So "~26 GiB for weights" holds only for the FIRST start on an unseen model. Numbers and measured per-config peaks: `docs/internals/MEMORY.md`.

## After it works

Re-run `make verify-fast`, add the model to local model notes, and consider a `DegenerationTest`-compatible probe prompt. New archs with think-channels: validate via `tools/analysis/degen_suite.py` against a running server (think-leak category).

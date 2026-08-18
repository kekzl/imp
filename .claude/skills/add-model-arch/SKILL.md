---
name: add-model-arch
description: Use when adding support for a new model architecture to imp, porting a model family, or debugging a model that loads but produces wrong output — "add support for <model>", "new arch", loader detection, chat template, RoPE variant, "outputs garbage", "prompt-blind", "digits scrambled", "NaN logits". Do NOT use for kernel performance (sm120-cuda-expert) or quant-format questions (quant-formats).
---

# Adding a Model Architecture — imp

## Integration checklist (gpt-oss PR #572 is the reference example)

1. **Enum + registry**: add to `ModelArch` in `src/model/model_arch.h`; wire `parse_model_arch` (GGUF `general.architecture`, `gguf_loader.cpp`) and/or HF detection (`hf_config_loader.cpp` — `architectures` array, `model_type` fallback), `model_arch_name`, `apply_arch_defaults`, sampling defaults in `src/model/model.cpp` (registry + `parse_model_arch`/`apply_arch_defaults` live there; `src/model/model_arch.h` holds only the enum + decls). **Then register the arch's traits in `ModelProfile`** (`src/model/model_profile.h` — single source of truth since PRs #622/#623, incl. the `AttnVariant` SWA/NoPE dispatch enum). Never add new `cfg.arch == X` checks in hot-path code — the profile is what dispatch reads.
2. **Loader**: tensor-name mapping in `src/model/tensor_kind_matcher.cpp` / `weight_map.cpp`; SafeTensors path in `safetensors_loader.cpp` (NVFP4 prequant via `llm_compressor_loader.cpp` if applicable).
3. **Arch config**: RoPE variant (NeoX vs GPT-J pair layout! see traps), YaRN/`rope_freq_scale`, SWA layer pattern, attention quirks (NoPE, sinks, softcap), norm placement, MoE router type — in `model_config.h` + `apply_arch_defaults`.
4. **Chat template**: `src/model/chat_template.cpp` (+ `jinja.cpp` if templated); think/reasoning channel handling if applicable.
5. **Kernels** only if genuinely new ops (sinks, new gating) — check `src/exec/` + `src/compute/` for an existing path first.
6. **Verify** (in order): loads → coherent greedy output (run `check-degeneration` battery) → **perplexity vs HF reference** (`imp-cli --perplexity`; expect within ~10-20% of HF — often much closer, e.g. gpt-oss imp 4.68 vs HF bf16 4.607, #663: the residual elevation is model-intrinsic) → decode/prefill sanity (`benchmark-cuda`).
7. **Docs**: row in `docs/MODELS.md` (+ docs/BENCHMARKS.md if hero-class); perf baseline entry if it becomes a gated model.

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
| CLI fine, server broken | not an arch bug — see `server-api` skill |

## Known traps (each cost a debugging session)

- **`rope_neox`**: GGUF converters pre-permute Q/K; HF SafeTensors do NOT. Llama-family SafeTensors without `rope_neox=true` = prompt-blind.
- **SWA layer masks**: the `swa_layers` pattern was Gemma-only hardcoded once — verify per-layer attention type for any interleaved-SWA arch.
- **Fused-rope-KV vs YaRN**: the fused rope+KV-write kernel must apply the same YaRN scaling as the standalone path.
- **Banned-token list vs channel tokens**: arch-specific control tokens (Harmony channels) must not land on the banned list.
- **Per-layer rope_freqs** (Gemma-4): non-SWA layers need their own freqs, `n_rot=hd`.
- **h_state precision** (GDN/hybrid): FP32 gate required — FP16 NaNs at depth.
- **HF tensor-name prefixes**: multimodal checkpoints wrap the LM under e.g. `model.language_model.*` (Qwen3.5-VL, PR #647) — strip the prefix in the loader or every tensor "is missing".
- **MLA/YaRN `rope_mscale`** (#880): the mscale ratio applied to the wrong base inflated RoPE by 1.261× — coherent-ish output that drifts vs HF. When comparing against a transformers oracle, PIN the transformers version (4.44.2 was the validated MLA oracle).
- **Draft/MTP heads must share the main model's exact RoPE math** — an MTP head computing plain NeoX while the target uses YaRN drifts the drafter (accept rate collapses, output stays correct). Shared impl: `src/compute/rope_yarn.cuh` (PR #913). Any new RoPE variant goes there, not into per-kernel copies.
- **Encoder/embedding archs are supported** (nomic-bert, PR #867 — cosine 0.999 vs HF). Gotcha: BERT-family GGUFs use an SPM tokenizer, not WordPiece-as-expected.
- **`attention_k_eq_v`** (gemma-4-31B-style) is NOT implemented — such archs need real work, not config.
- Model too big? Do the arithmetic instead of trusting a remembered ceiling: the card is 32 607 MiB, the CUDA primary context takes ~1680 MiB before imp allocates anything, and cuBLAS/CUTLASS claim a measured ~3900 MiB constant on the first forward pass (`kMeasuredLibraryReserveBytes`, `src/memory/plan.h`). That leaves ~26 GiB for weights *before* the KV pool and workspaces — so gemma-4-31B NVFP4 never fits, and the old "~29 GiB of weights" figure was never reachable. Numbers and measured per-config peaks: `docs/internals/MEMORY.md`.

## After it works

Re-run `make verify-fast`, add the model to local model notes, and consider a `DegenerationTest`-compatible probe prompt. New archs with think-channels: validate via `tools/analysis/degen_suite.py` against a running server (think-leak category).

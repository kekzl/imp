# Phase 3 — SafeTensors Coverage Sweep Results

Builds on `30_safetensors_matrix.md`. The four (architecture × precision) cells with local NVFP4 weights were validated; FP8 KV path was exercised on the two architectures that support it; Qwen3.6 NVFP4 decode was confirmed broken (Phase 1 load-only per PR #71).

## Validation method

- **NVFP4 baseline:** load via SafeTensors loader, run `imp-cli --temperature 0 --seed 0 --max-tokens 32` on a deterministic prompt, confirm the output is coherent (i.e. contains expected ground truth or at least is non-degenerate). Three cells (Mistral, Gemma-4, Qwen3-Coder) were already validated by `LlmCompressorE2E` in Phase 2.
- **FP8 KV variant:** same model, same prompt, with `--kv-fp8` to engage `kv_cache.dtype=fp8` + `attention.fp8_prefill=auto` + `attention.fp8_fmha=auto`.
- **Reference:** the LlmCompressorE2E tests use a hard "Paris" substring check; the orchestrator extends this to FP8 KV variants where supported.

## Final matrix (NVFP4 + FP8 columns)

| Architecture | NVFP4 | FP8 KV (--kv-fp8) | Notes |
|---|---|---|---|
| MISTRAL (Mistral-Small-3.2-24B-Instruct-2506-NVFP4, 15 GB) | ✅ PASS — `LlmCompressorE2E.MistralSmall_LoadsAndGeneratesCoherent` returns "Paris" | ✅ PASS — `--kv-fp8 --chat-template none` → "Paris. It is the capital of France. ..." | Chat-template default applies a ~600-token system prompt that hits the NVFP4 long-context regression (memo `nvfp4_long_context_regression_2026_04_28`); use `--chat-template none` for raw completion or rely on PR #78's author-flag default. |
| GEMMA4 (Gemma-4-26B-A4B-it-NVFP4, 16 GB) | ✅ PASS — `LlmCompressorE2E.Gemma4_LoadsAndGeneratesCoherent` returns "Paris" via chat template | ⚠ KNOWN_LIMITATION | FP8 KV unsupported today — `engine.cpp:547` hardcodes FP16 for Gemma-4 because KV write/read kernels need per-layer head_dim awareness (memo `gemma4_fp8_kv_2026_04_29`). Allocator side is done; kernel side not. |
| QWEN3_MOE (Qwen3-Coder-30B-A3B-FP4, ModelOpt, 17 GB) | ✅ PASS — `LlmCompressorE2E.Modelopt_QwenCoder30B_StillWorks` loads + generates | ✅ PASS — `--kv-fp8 --no-cuda-graphs --chat-template none` → "Paris ... The capital of Belgium is Brussels." (39.7 tok/s decode) | `--no-cuda-graphs` required: D2H routing memcpy in MoE prefill is incompatible with graph capture (CLAUDE.md). `expert_overhead_pct` defaults to 10 (auto path); host-fall-back gated on VRAM. |
| QWEN36_MOE (Qwen3.6-35B-A3B-NVFP4, GDN+MoE, 24 GB) | ⚠ KNOWN_LIMITATION — load succeeds, generation degenerates to `<\|im_start\|>` repetition | n/a — base path broken | PR #71 added Phase 1 SafeTensors plumbing as **load-only**. Decode coherence requires Phase 2/3 work (likely 1–2 days) — wire NVFP4 weights through fused QKV/gate-up + GDN scan with the right shape semantics. CUDA graphs additionally fail capture on this hybrid even with non-MoE attention layers. |

### Other 9 architectures (LLAMA, MIXTRAL, DEEPSEEK, NEMOTRON_H_MOE, QWEN3, QWEN35, QWEN35_MOE, GEMMA3, LLAMA4)
All `NO_WEIGHTS` — no local NVFP4 SafeTensors checkpoint exists. Loader code paths exist (covered by unit tests in Phase 2 — `test_hf_config_loader`, `test_llm_compressor_loader`), but no end-to-end NVFP4 generation can be executed without downloading multi-tens-of-GB weights, which the bringup excludes.

## Artifacts (per cell)

| File | Cell |
|---|---|
| `phase3/30_qwen36_nvfp4.log` | Qwen3.6 NVFP4 baseline (degenerate + CUDA graph capture errors) |
| `phase3/30_qwen36_nvfp4_nograph.log` | Qwen3.6 NVFP4 with `--no-cuda-graphs` (still degenerate) |
| `phase3/31_mistral_fp8kv.log` | Mistral FP8 KV with default chat template (degenerate due to ~600-token system prompt + NVFP4 long-context bug) |
| `phase3/31b_mistral_fp8kv_raw.log` | Mistral FP8 KV with `--chat-template none` ✅ |
| `phase3/32_qwencoder_fp8kv.log` | Qwen3-Coder FP8 KV ✅ |

## Phase 3 status

- **3 of 4 NVFP4 cells PASS.** One KNOWN_LIMITATION (Qwen3.6 — upstream Phase 2/3 work pending).
- **2 of 4 FP8 KV cells PASS.** One KNOWN_LIMITATION (Gemma-4 — engine.cpp:547 hardcode); one cascades from the NVFP4 limitation (Qwen3.6).
- **0 SafeTensors-load bugs.** The earlier LlmCompressorE2E "failures" were a bind-mount problem, fixed by symlinking real weights into `models/`.
- **0 unmapped HF architectures on disk.**


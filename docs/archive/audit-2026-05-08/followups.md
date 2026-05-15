# Deferred Roadmap Items — 2026-05-08

Every item listed in `docs/audit/roadmap_inventory_2026-05.md` that did not enter the master plan, with the reason it was deferred. Quality Gate is non-negotiable; deferral is the correct outcome when an item can't be landed at full quality in this run.

## Format

`<id> <one-line title>` then `Reason:` and `Pre-conditions to revisit:`.

---

### L1 — FP8 KV cache: Gemma-4 carve-out

Reason: Per-layer head_dim awareness needed in KV write/read kernels. Single-stride assumption is hard-coded across the FP8 paged-attention path. Multi-day kernel work + multi-model regression sweep, and the workaround (default `--kv-fp16` for Gemma-4) ships today.
Pre-conditions: dedicated branch, full Gemma-4 long-context regression baseline, kernel rewrite with per-layer stride tables.

### L2 — Chunked prefill: paged-prefill kernel pending

Reason: Mitigation already shipped (PR #114 single-chunk default at `engine.cpp:1644`). The proper fix is a paged-prefill kernel reading K/V from cache during chunked attention — separate, larger work. Cannot be done within this run's Quality Gate.
Pre-conditions: paged-prefill kernel design + multi-tenant decode-latency test fleet.

### L3 — NVFP4 SmoothQuant `input_scale`

Reason: Direct absorption refuted by A/B (`llm_compressor_input_scale_dead_end_2026_05_07.md`). Real fix needs per-channel scaling vector applied during activation quantization, not a scalar alpha modifier. Single test model (Mistral-3.2-NVFP4); workaround PR #78 (default-system-prompt skip) lives today.
Pre-conditions: extra calibrated test models with SmoothQuant scaling, per-channel activation quant path.

### L4 — Qwen3.5-27B MXFP4 fails at load

Reason: 12 GiB MXFP4 + 48 GiB FP16 fallback oversubscribes VRAM. Real fix needs host-dequant + StoragePlanner — large change for a single-model unlock. PR #60 already added a clear diagnostic.
Pre-conditions: StoragePlanner extension, host-dequant kernel, layer-tier policy.

### L5 — Gemma-4 Q4_K_M code-gen drift

Reason: Accumulated FP16 drift across 30 layers is precision-architectural. Quant rework, no clean low-risk lever. Workaround "use Q5_K_M / Q8_0" documented.
Pre-conditions: per-layer FP32 accumulator audit + selective dequant policy.

### L6 — MoE expert offload disables CUDA Graphs

Reason: Needs device-side LRU prefetch with async pipeline. Significant runtime kernel work. Tip `IMP_EXPERT_OVERHEAD_PCT` ships today.
Pre-conditions: async expert prefetch design + multi-model regression coverage.

### P1 — Closing TurboQuant–FP8 gap

Reason: "Algorithm-inherent" per roadmap. Needs QJL → MXFP4-K direction redesign.
Pre-conditions: MXFP4-K-direction kernel.

### P2 — pp=512 on large dense models

Reason: cuBLAS autotune variance; "not gating any user" per roadmap.
Pre-conditions: deterministic-cuBLAS audit on these specific shapes.

### P3 — Speculative decoding

Reason: Investigated and shelved per roadmap. CLI flags removed in `7380ea8`. OBSOLETE.

### R1–R10 — Research-interest items (CUDA 13.2 features, PTX 9.2, KV-compression research)

Reason: Each is a multi-day-to-multi-week kernel project with no in-tree benchmark/regression scaffold appropriate for one-run sign-off. Most are dead-ends or zero-net (R5 already shipped, R7 quality-risky, Lv5 SFU-exp2 net-zero).
Pre-conditions: dedicated dev cycle per item.

### Lv3 — CLC-persistent kernel for continuous batching

Reason: Listed payoff is multi-tenant only; imp is "single-author single-target experiment" (`docs/roadmap.md`). No multi-tenant validation harness.
Pre-conditions: multi-tenant decode benchmark + workload model.

### AU1 — GLM architecture

Reason: GLM diverges from LLAMA enough that mapping `GlmForCausalLM → LLAMA` would silently produce wrong outputs. Real fix needs `GLM` enum + dedicated forward path. Multi-week.
Pre-conditions: GLM model in test fleet, dedicated forward path.

### AU2 — Native SentencePiece (`.model`) parser — `closed-in-27a7a7a`

Status: closed. In-tree wire-format protobuf decoder + ModelProto/TrainerSpec field extraction + integration with imp's existing SPM-style encoder shipped as `feat/sentencepiece-loader`. 10 new unit tests, no new third-party deps. SafeTensors checkpoints with only `tokenizer.model` (older Llama 1/2, some Mistral variants) now load directly without the previous Python-conversion workaround.

### AU3 — AWQ INT4 dequant kernel

Reason: Two coupled blockers, both fixture-related:

1. **No AWQ checkpoint in the local model fleet.** `find $IMP_MODELS_DIR $HOME/.cache -iname "*awq*"` returns nothing as of 2026-05-08. Implementing the dequant kernel without a real-world checkpoint to integration-test against would ship a synthetic-only-validated kernel — the Quality Gate's "anti-pattern: stubs/mocks/placeholders behind a flag" forbids that even when synthetic tests pass.

2. **AWQ packing convention has subtle differences from GPTQ.** AWQ exports column-pack the INT4 weights with an interleave permutation (qweight[i,j] high/low nibble for sequential rows in a packing block), whereas GPTQ row-packs. Misinterpreting the packing produces output that compiles fine, runs without crash, and silently drifts. Cross-validating against a Python `autoawq` reference is the only way to be sure — that requires a real model file.

Detection is already in place: PR #116 commit `7c0b8c8` parses `awq_config.json` and emits a load-time WARN with bits/group_size. So users get a clear "this model needs work" signal today; they're directed to GPTQ or NVFP4 variants.

**Acquisition path for next session:**
- Download a small AWQ checkpoint (~2-4 GB), e.g. `TheBloke/Llama-2-7B-Chat-AWQ` (HF Hub) or `casperhansen/llama-3-8b-instruct-awq`. Place under `$IMP_MODELS_DIR/`.
- Cross-reference output against Python `autoawq`'s `AutoAWQForCausalLM` on a fixed prompt with `temperature=0`. Use the existing `scripts/validate_safetensors.py` harness pattern.
- Implement dequant-to-FP16 path first (~150 LoC kernel + cuBLAS dispatch); native AWQ-INT4×FP16 GEMV is a follow-up.

**LoC estimate (post-fixture):** 150-300 LoC kernel + 100 LoC loader integration + 200 LoC tests. ~1 session.

Pre-conditions: AWQ test model with golden FP32 reference from `autoawq`.

### AU4 — DeepSeek MLA attention

Reason: Multi-head latent attention path is "multi-week effort" per audit. Affects `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` across attention layout. Out of scope.
Pre-conditions: DeepSeek-V2/V3 SafeTensors test model, MLA-specific attention kernel.

### AU5 — Multimodal SafeTensors loaders

Reason: Per-family work (Qwen-VL, Llava, Pixtral, Gemma-3 vision-from-SafeTensors), each with its own vision-tower loader, prefix-injection wiring, encoder. Multi-week.
Pre-conditions: per-family test models + dedicated vision development cycle.

### AU6 — Tiktoken parser

Reason: Uncommon in supported families per audit; explicit `ignored`.
Pre-conditions: User-driven request from a Tiktoken-only family.

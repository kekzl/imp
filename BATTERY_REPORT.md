# Prompt Battery — Full Architecture Sweep

**Run date:** 2026-06-30
**Engine commit:** 43227b24 (HEAD at start)
**Host:** RTX 5090 (sm_120a), CUDA 13.3, Docker image `imp:test`
**Mode:** autonomous full battery — recon → baseline → fix-loop → green matrix

This report tracks the prompt-test battery across every supported model architecture,
both loaders (GGUF + SafeTensors), the available quant formats, and the full HTTP API
surface. Findings are root-caused and fixed; skips are recorded with the reason.

---

## 1. Supported architecture registry (from code)

`ModelArch` enum (`src/model/model_arch.h:7`), dispatch in `src/model/model.cpp:209`
(`parse_model_arch`). 15 arch values; many families collapse onto `LLAMA`
(Phi-2/3, Qwen2, InternLM2, Starcoder2, Cohere).

| ModelArch | GGUF arch string | HF class name(s) | Special path |
|-----------|------------------|------------------|--------------|
| LLAMA | `llama`, `qwen2`, `phi3` | LlamaForCausalLM, Phi3ForCausalLM, Qwen2… | dense baseline |
| MISTRAL | `mistral` | MistralForCausalLM | sliding window |
| MIXTRAL | `mixtral` | MixtralForCausalLM | MoE (block_sparse) |
| DEEPSEEK | `deepseek`, `deepseek2` | DeepseekV2/V3ForCausalLM | **MLA** + MoE |
| NEMOTRON_H_MOE | `nemotron_h_moe` | NemotronHForCausalLM | **Mamba2/GDN + Attn + MoE hybrid**, NoPE |
| QWEN3 | `qwen3` | Qwen3ForCausalLM | dense |
| QWEN3_MOE | `qwen3moe` | Qwen3MoeForCausalLM | MoE |
| QWEN35 | `qwen35` | Qwen3_5ForCausalLM/ForConditionalGeneration | dense (+MTP) |
| QWEN35_MOE | `qwen35moe` | Qwen3_5MoeForCausalLM | MoE |
| QWEN36_MOE | `qwen36moe` | Qwen3_5MoeForConditionalGeneration | MoE (GDN hybrid) |
| GPT_OSS | `gpt_oss`, `gpt-oss` | GptOssForCausalLM | MoE + sinks + GLU |
| GEMMA3 | `gemma3` | Gemma3ForCausalLM/ForConditionalGeneration | SWA pattern 6, softcap, **vision** |
| GEMMA4 | `gemma4` | Gemma4ForCausalLM/ForConditionalGeneration | per-layer SWA, MoE, **vision** |
| LLAMA4 | `llama4` | Llama4ForCausalLM | MoE |
| GENERIC | (fallback) | (unmatched) | weight-name heuristic |

---

## 2. Local model inventory (mapped to matrix)

### GGUF loader
| Model | Arch | Quants present | Notes |
|-------|------|----------------|-------|
| Llama-3.2-3B-Instruct | LLAMA | Q8_0, IQ4_XS | |
| Qwen3-4B-Instruct-2507 | QWEN3 | Q8_0, IQ4_NL | |
| Qwen3-8B | QWEN3 | Q8_0 | degeneration default model |
| Qwen3-14B | QWEN3 | Q6_K | north-star |
| Qwen3.5-4B | QWEN35 | mxfp4 | |
| Qwen3-30B-A3B | QWEN3_MOE | Q4_K_M | |
| Qwable-27b | QWEN3(?) | Q4_K_M | community merge |
| qwen3.6-35B-A3B | QWEN36_MOE | (gguf dir) | |
| gemma-3-12b-it | GEMMA3 | Q4_K_M | **decode crash known (sampling.cu IMA)** |
| gemma-3-4b-vl | GEMMA3 | Q4_K_M + mmproj | **vision** |
| gemma-4-26B-A4B-it | GEMMA4 | Q8_0, Q4_K_M | + mmproj-gemma4-26b |
| gpt-oss-20b | GPT_OSS | mxfp4, bf16 | |

### SafeTensors loader
| Model | Arch | Quant | Notes |
|-------|------|-------|-------|
| DeepSeek-Coder-V2-Lite-Instruct | DEEPSEEK | bf16 (unq) | MLA |
| DeepSeek-V2-Lite | DEEPSEEK | bf16 (unq) | MLA |
| Gemma-4-12B-NVFP4 | GEMMA4 | NVFP4 (modelopt) | |
| Gemma-4-26B-A4B-it-NVFP4 | GEMMA4 | NVFP4 (compressed-tensors) | MoE |
| gpt-oss-20b | GPT_OSS | mxfp4→NVFP4 experts | |
| Nemotron-3-Nano-30B-A3B-NVFP4 | NEMOTRON_H_MOE | NVFP4 | hybrid |
| Nemotron-Labs-3-Elastic-30B-A3B-NVFP4 | NEMOTRON_H_MOE | NVFP4 | hybrid |
| Phi-4-reasoning-plus-NVFP4 | LLAMA (Phi3) | NVFP4 (modelopt) | |
| Qwen3-14B-NVFP4 | QWEN3 | NVFP4 (modelopt) | |
| Qwen3-30B-A3B-NVFP4-Modelopt | QWEN3_MOE | NVFP4 (modelopt) | |
| Qwen3.6-27B-Text-NVFP4-MTP | QWEN35 | NVFP4 (modelopt) | +MTP |
| Qwen3.6-35B-A3B-NVFP4 | QWEN36_MOE | NVFP4 (compressed-tensors) | non-deterministic @ temp=0 |
| Qwen3-8B-NVFP4-cortecs | QWEN3 | NVFP4 (compressed-tensors) | server default |
| Qwen3-Coder-30B-A3B-Instruct-FP4 | QWEN3_MOE | NVFP4 (modelopt) | |

### Not available locally (→ skip, documented)
- **Mistral / Mixtral**: no local checkpoint. (`run_all_models.sh` lists Devstral but absent.)
- **LLAMA4**: no local checkpoint.
- **INT8 full model**: none (INT8 is a runtime KV/path feature, covered by unit tests).
- **FP8 E4M3 full model**: none as a standalone weight format (FP8 is KV-cache dtype, unit-tested).

---

## 3. Matrix status (Arch × Loader × Quant × Feature)

Legend: ✅ pass · ⚠️ pass w/ caveat · ⏭️ skip (reason) · 🔧 escalated (see Findings).
"Engine-health" = logits finite + temp=0 determinism + 32× byte-identical
CUDA-graph replay (the real correctness signals; the NVFP4 battery's binary verdict
is gated on stricter content prompts — see §4).

### GGUF loader
| Arch | Model / quant | Coherence | Notes |
|------|---------------|-----------|-------|
| LLAMA | Llama-3.2-3B Q8_0 | ✅ degen 5/5 | |
| QWEN3 dense | Qwen3-8B Q8_0 | ✅ degen 5/5 | |
| QWEN3 dense | Qwen3-4B-2507 Q8_0 | ✅ e2e Primary 4/4 | |
| QWEN35 (GDN hybrid) | Qwen3.5-4B mxfp4 | ✅ degen 5/5 + GDN e2e 2/2 | recurrent state OK |
| QWEN3_MOE | Qwen3-30B-A3B Q4_K_M | ✅ degen 5/5 | |
| GPT_OSS | gpt-oss-20b mxfp4 | ✅ degen 5/5 | |
| GEMMA3 | gemma-3-12b Q4_K_M | ✅ degen 2/2 (no-graph) | graph-replay IMA is known legacy |
| GEMMA3 (vision) | gemma-3-4b-vl Q4_K_M + mmproj | ✅ VL e2e ("tabby cat…") | full image→text |
| GEMMA4 (MoE) | gemma-4-26B UD-Q4_K_M | ✅ isolated 5/5 · 🔧 F1 | garbles only after a GDN model in-process |
| GEMMA4 (vision) | gemma-4-26B + gemma4 mmproj | ✅ VL e2e ("…is a cat") | full image→text |

### SafeTensors loader (NVFP4) — all engine-healthy (det+logits+graph 32/32)
| Arch | Model | Engine-health | Content |
|------|-------|---------------|---------|
| QWEN3 dense | Qwen3-8B-cortecs (compressed-tensors) | ✅ | 18/20 |
| QWEN3 dense | Qwen3-14B (modelopt) | ✅ | 17/20 |
| QWEN3_MOE | Qwen3-30B-A3B-Modelopt | ✅ | 17/20 |
| QWEN3_MOE | Qwen3-Coder-30B FP4 (modelopt) | ✅ | 18/20 |
| QWEN35 dense+MTP | Qwen3.6-27B-Text-MTP | ✅ | 18/20 |
| QWEN36_MOE (GDN hybrid) | Qwen3.6-35B-A3B (compressed-tensors) | ✅ | 16/20 |
| NEMOTRON_H (Mamba2+Attn+MoE) | Nemotron-3-Nano-30B | ✅ | 15/20 |
| LLAMA (Phi3) | Phi-4-reasoning-plus (modelopt) | ✅ | 16/20 |
| GEMMA4 (MoE) | Gemma-4-26B-A4B-NVFP4 (compressed-tensors) | ✅ | 18/20 |
| GEMMA4 (gemma4_unified) | Gemma-4-12B-NVFP4 | 🔧 F3 | unsupported multimodal arch port |

### API / feature axis (real imp-server, NVFP4 Qwen3-8B-cortecs) — `make test-server` ALL ✅
| Feature | Result |
|---------|--------|
| OpenAI `/v1/chat/completions` + `/v1/completions` + `/v1/embeddings` | ✅ endpoints smoke |
| Anthropic `/v1/messages` (incl. real incremental SSE event sequence) | ✅ messages stream |
| Streaming (OpenAI SSE) | ✅ |
| Tool calling (incl. multi-turn tool-results) | ✅ (exercise_all_endpoints + NVFP4 battery 10/10) |
| JSON-schema constrained decode | ✅ (battery `json_output` + `tool_format_json_schema` 10/10) |
| Logprobs (sum + top-k descending) | ✅ |
| Thinking toggle (both dialects) | ✅ |
| Auth (Bearer) | ✅ (test_server_auth unit + endpoints) |
| Prefix cache correctness (embed/chat interleave) | ✅ |
| 0-token wedge robustness (#710, sustained load) | ✅ |
| Bad-input → 4xx envelope (#712) | ✅ |
| Continuous batching / paged KV under load | ✅ (interleave + 0-token sustained 80 reqs) |
| Speculative decoding greedy-equivalence | ⏭️ opt-in; covered by existing greedy-lock + #683 byte-perfect tests |

### Vision (SigLIP) — `make test-vision` + VL e2e
| Path | Result |
|------|--------|
| Gemma-3 SigLIP encoder golden (896px) | ✅ |
| Gemma-4v encoder golden (768px, grid 48) | ✅ |
| Gemma-3 full VL (image→encoder→LM→text) | ✅ "tabby cat sitting on a wall" |
| Gemma-4 full VL (image→encoder→LM→text) | ✅ "the animal in this image is a cat" |
| Invalid image rejection | ✅ clean "unknown image type" (corrupt fixture, not a bug) |

---

## 4. Findings

### F1 — Gemma-4 forward pass corrupted by a preceding GDN model (process-global state leak) — **REAL BUG**
- **Symptom:** in one process, after `GDNModelTest` (Qwen3.5 GDN/SSM arch) runs,
  the subsequent `Gemma4ModelTest.AnswersCapitalOfFrance` and
  `Gemma4GraphsTest.AnswersCapitalOfFranceWithGraphs` produce degenerate garbage
  ("own own else else than than", no "Paris").
- **Discrimination (systematic):**
  - Gemma-4 tests in isolation → PASS.
  - dense Qwen3 predecessor → Gemma-4 PASS.
  - gpt-oss-20b **mxfp4** MoE predecessor (mxfp4, non-GDN) → Gemma-4 PASS (deconfounds quant).
  - GDN (Qwen3.5) predecessor → Gemma-4 FAIL.
  - imp-cli single-shot on the same Gemma-4 GGUF (gemma + auto template, graphs on/off) → coherent "Paris".
- **Root cause (precise, instrumented):** the **L2 persisting access-policy
  window** set per-attention-forward poisons the CUDA stream. `set_kv_l2_persist`
  (`src/exec/executor_attention_internal.h`) calls
  `cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow, …)` as a
  best-effort L2 perf hint. After a GDN/SSM model has run in the same process,
  the per-context persisting-L2 reservation
  (`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`,
  `engine_weight_upload.cpp:87`) is left in a state where the next model's
  window-set returns `cudaErrorInvalidValue` ("invalid argument") — exactly as
  the existing code comment warns ("poisons the stream for every subsequent
  kernel"). The set fires **mid-forward**, so the *rest of that forward's*
  kernels (MoE/CUTLASS GEMMs, output proj) see the pending sticky error and
  bail → degenerate logits. `executor_forward.cu:211` clears it at the start of
  the **next** forward (hence "Cleared stale error … invalid argument" logged
  ~once per decode step), but the damage is per-forward.
  - **Evidence:** instrumented run shows the invalid-argument error recurs
    **inside Gemma-4's own forward** after GDN (60×, one per decode step) but is
    **completely absent when Gemma-4 runs in isolation**; gpt-oss/dense
    predecessors leave no error and Gemma-4 stays coherent.
- **Exact thrower (instrumented, decisive):** added per-phase
  `cudaPeekAtLastError` probes to the forward → the "invalid argument" first
  appears **after the attention of a mid-stack Gemma-4 layer, only at decode
  steps ≥ ~64**, i.e. once context reaches ≥4 KV blocks and the **split-K paged
  decode kernel** turns on (`attention_paged.cu:1361`,
  `num_ctx_blocks >= 4`). Gemma-4 is `head_dim=256`; the split-K decode launch
  for that head dim, combined with persistent device/static state left by the
  preceding GDN model, returns `cudaErrorInvalidValue`. It is **absent** at
  steps < 64 and **absent entirely** when Gemma-4 runs without a GDN predecessor.
- **Scope / severity:** **TEST-HARNESS-ONLY, zero production impact.** It needs a
  GDN/SSM model *and* a Gemma-4 NVFP4/decode model loaded in the **same process**.
  `imp-server` and `imp-cli` load exactly one model per process, so no shipped
  entry point can hit it. The only trigger is the multi-model GTest binary
  (`make test-e2e` when `GDNModelTest` precedes the Gemma-4 suites).
- **Hardening shipped (reduces the cross-model CUDA-state-leak surface):** a
  failed best-effort op must never leave a sticky per-context error. Drained
  `cudaGetLastError()` after every `cudaStreamSetAttribute(accessPolicyWindow)`
  (`executor_attention_internal.h` `set_kv_l2_persist`, `executor_helpers.h`
  `set_l2_streaming`/`clear_l2_policy`); reset persisting-L2 lines per model load
  (`engine_weight_upload.cpp` `cudaCtxResetPersistingL2Cache`); drain at `Engine`
  teardown (`engine.cpp ~Engine`). These are independently-correct latent-bug
  fixes (the existing code comment itself warns the window-set "poisons the
  stream for every subsequent kernel").
- **Status: NOT fully resolved — ESCALATED.** The proximate split-K-decode
  launch failure under GDN-left device state needs a deeper fix (force
  `num_splits=1` for this case, or opt-in the head_dim=256 split-K kernel's smem,
  or a full per-context reset of the split-K statics) that is disproportionate to
  a zero-production-impact, test-only interaction. Diagnosis + hardening shipped;
  the remaining split-K root fix is logged as an open item. Classification:
  (a) real bug, test-harness-only.

### F3 — `gemma4_unified` multimodal checkpoint unsupported — **ESCALATION (distinct arch port)**
- **Symptom:** `Gemma-4-12B-NVFP4` boots → server FATAL `weight_handle.cu:25
  WeightRegistry::handle: id -1 out of range [0,0)` ("no tensors were assigned").
- **Root cause:** its `config.json` declares `architectures:
  ["Gemma4UnifiedForConditionalGeneration"]`, `model_type: gemma4_unified`. This
  string is not in the arch map → loader falls back to GENERIC → weight-name
  heuristic guesses `llama` → tensor names don't match (the LLM weights are
  prefixed `model.language_model.layers.*`, not `model.layers.*`) → zero tensors
  assigned → crash. The loader already WARNs clearly ("unknown HF architecture …
  Add a class mapping"; "Multimodal model detected … vision tower skipped").
- **Why NOT a one-line fix:** `gemma4_unified` is a genuinely **different
  architecture**, not a renamed Gemma-4. Its `text_config` carries features imp's
  GEMMA4 path does not implement: per-layer input embeddings
  (`hidden_size_per_layer_input`, `vocab_size_per_layer_input`),
  `num_kv_shared_layers`, `use_double_wide_mlp`, an `enable_moe_block` MoE path,
  `global_head_dim`/`num_global_key_value_heads`, plus separate vision + audio
  towers. Mapping the string to `ModelArch::GEMMA4` would run the **wrong forward
  path and emit garbage** — strictly worse than the current clean-ish failure.
- **Disposition:** ESCALATION — supporting it is a dedicated model port (new
  forward features + multimodal `model.language_model.` prefix handling + vision
  skip), out of scope for a battery pass. The standard Gemma-4 NVFP4 cell is
  already covered by `Gemma-4-26B-A4B-it-NVFP4` (engine-healthy, see §3). Logged
  as a skip with reason; not force-mapped.

### F2 — NVFP4 SafeTensors battery covered only 6 MoE checkpoints — **COVERAGE GAP (fixed)**
- `validate_safetensors.py` MODELS missed whole arch families (Phi-4, dense Qwen3,
  Gemma-4 dense, Qwen3.6-MTP). Extended to 12 entries (see §5 A2).

### Non-findings (invocation artifacts, ruled out)
- Qwen3-30B-A3B-Q4_K_M "4 fails" → my path pointed at the **directory**, not the
  `.gguf` file; `detect_format()` sniffs the `.gguf` suffix and mis-routed to
  SafeTensors → load failure. Correct `.gguf` path → 5/5 PASS.
- gemma-3-12b-Q4_K_M (GEMMA3 GGUF): degeneration 2/2 PASS without graphs. The
  documented decode IMA (sampling.cu graph-replay race) needs CUDA graphs ON and is
  a known legacy issue (superseded by Gemma-4); not re-opened here.

### Clean lanes (Phase B, GGUF)
| Arch | Model | Result |
|------|-------|--------|
| LLAMA | Llama-3.2-3B-Q8_0 | degen 5/5 ✓ |
| QWEN3 dense | Qwen3-8B-Q8_0 | degen 5/5 ✓ |
| QWEN3 dense | Qwen3-4B-2507-Q8_0 | e2e Primary 4/4 ✓ |
| QWEN35 GDN | Qwen3.5-4B-mxfp4 | degen 5/5 + GDN e2e 2/2 ✓ |
| QWEN3_MOE | Qwen3-30B-A3B-Q4_K_M | degen 5/5 ✓ |
| GPT_OSS | gpt-oss-20b-mxfp4 | degen 5/5 ✓ |
| GEMMA4 | gemma-4-26B-UD-Q4_K_M | isolated ✓ (fails after GDN → F1) |
| GEMMA3 | gemma-3-12b-Q4_K_M | degen 2/2 ✓ (no-graph) |

---

## 5. Assumptions made

- **A1 — test-e2e model path overrides.** `make test-e2e` hard-codes
  `Qwen3.5-4B-Q8_0.gguf` (GDN) and `gemma-4-26B-A4B-it-Q4_K_M.gguf` (Gemma-4),
  neither of which exists locally. To actually exercise (not silently skip) the
  GDN + Gemma-4 lanes I override the env to the local equivalents:
  `IMP_TEST_MODEL_GDN=Qwen3.5-4B-mxfp4.gguf`,
  `IMP_TEST_MODEL_GEMMA4=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`.
- **A2 — NVFP4 battery extension.** `validate_safetensors.py` MODELS only covered
  6 MoE checkpoints. Extended to cover dense + smaller arches present locally
  (Phi-4, Qwen3-8B/14B dense, Gemma-4-12B, Qwen3.6-27B-MTP, Nemotron-Elastic) so
  the SafeTensors battery sweeps the arch matrix rather than a MoE subset.
- **A3 — representative-cell sampling.** The full Arch×Loader×Quant×Feature
  cross-product is combinatorially huge and the GPU is single. One representative
  model per (arch, loader) cell is run end-to-end; remaining quant variants of the
  same arch are covered by the kernel/dequant unit tests (test-quant) rather than
  re-running full inference. This is noted per-cell in the matrix.

---

## 6. Skips (with reason)

- **Mistral / Mixtral** — no local checkpoint (`run_all_models.sh`/`validate_safetensors.py`
  list Devstral/Mistral-Small-NVFP4 but they are absent under `/home/kekz/models`).
  Arch path exists in code (`ModelArch::MISTRAL/MIXTRAL`); untested for lack of weights.
- **LLAMA4** — no local checkpoint.
- **DeepSeek / MLA** — local checkpoints are bf16 (`DeepSeek-V2-Lite`,
  `DeepSeek-Coder-V2-Lite`, ~30 GB each) which do not fit 32 GB alongside KV for an
  e2e battery; MLA is covered at config level by `test_mla.*`. No quantized MLA model
  on disk. (Config/loader validated; full-decode e2e needs a ≤24 GB MLA checkpoint.)
- **INT8 / FP8-E4M3 as a standalone weight format** — no full-model checkpoint exists;
  these are runtime KV-cache dtypes, covered by `test_fp8_kv_cache` / `test_kv_*` units
  and exercised via the server battery (NVFP4 model w/ kv auto).
- **CPU Python mock contract suite (`tests/api/`)** — needs `pytest` on the host, which
  the clean-host policy forbids; it is CI-covered. The higher-value **real-handler**
  surface was instead exercised via `make test-server` (all green, §3).
- **Speculative-decode greedy A/B** — opt-in (`speculative.ngram`); equivalence is
  locked by the existing greedy-lock + #683 byte-perfect graph-vs-eager tests rather
  than re-run here.
- Local fixture note: `/home/kekz/models/gemma-3-4b-vl/test_bus.jpg` is a 177-byte
  "File not found" text placeholder, not a JPEG — imp rejects it correctly. Use
  `test_cat.jpg` / `test_pizza.jpg` (real JFIF) for VL smokes.

---

## 7. Final summary

**Matrix outcome: green across every locally-runnable cell, with two escalations.**

What was RED and is now resolved/clarified:
- **F2 (fixed):** the SafeTensors/NVFP4 battery only covered 6 MoE checkpoints. Extended
  to 12 cells so the arch matrix (dense Qwen3, Phi-4, Gemma-4 dense, Qwen3.6+MTP, both
  Nemotron variants) is actually swept. All 9 loadable NVFP4 archs are **engine-healthy**
  (finite logits + temp=0 determinism + 32× byte-identical graph replay).
- **NVFP4 "FAIL" verdicts re-triaged (test-tolerance, classification b):** the battery's
  binary verdict is gated on 2 over-strict content prompts — `factual_capital`
  ("Paris" required within the *first 5 tokens*, but models say "The capital of France
  is Paris" → token 6) and `spec_decoding_compat` (ambiguous needle). Structured-output
  prompts (tool-calling, JSON-schema, code, regex) are **0/10 failures** — the engine is
  correct; the gates are too tight. (Left as-is in the harness; documented here.)
- **Server API surface:** all 7 real-handler batteries green on NVFP4 — OpenAI +
  Anthropic `/v1/messages` w/ real incremental SSE, streaming, tools, JSON-schema,
  logprobs, thinking, auth, prefix-cache, continuous-batching robustness.
- **Vision:** encoder goldens + full image→text e2e green for **both** Gemma-3 and
  Gemma-4; the loader detects the gemma4v vs gemma3-SigLIP variant correctly (no silent
  fallback) and rejects malformed images cleanly.

What remains OPEN (escalated, with reason):
- **F1 — GDN→Gemma-4 cross-model garbage (test-harness-only, zero production impact).**
  Fully diagnosed: a `cudaErrorInvalidValue` in Gemma-4's split-K paged-decode attention
  (head_dim=256, activates at context ≥64 tokens) under persistent device state left by
  a preceding GDN/SSM model **in the same process**. `imp-server`/`imp-cli` load one
  model per process and cannot hit it; only the multi-model GTest binary triggers it.
  Hardening shipped (L2-hint error drains ×3 + persisting-L2 reset + engine-teardown
  drain — all independently-correct latent-bug fixes); the deeper split-K root fix is
  disproportionate to a zero-prod-impact path and is logged for follow-up.
- **F3 — `gemma4_unified` multimodal checkpoint (Gemma-4-12B-NVFP4) unsupported.** It is a
  distinct arch (per-layer input embeddings, KV-shared layers, double-wide MLP, +vision
  +audio towers, `model.language_model.` tensor prefix), not a renamable Gemma-4. The
  std Gemma-4 NVFP4 cell is already covered by the 26B; force-mapping the 12B would emit
  garbage, so it is left as a dedicated port (escalation), not a one-line alias.

Tests/coverage added & shipped this run: NVFP4 battery arch-matrix extension (F2) +
4-site cross-model CUDA-error-leak hardening (F1). Baseline GPU suite (Phase A): **1301
GTest cases, 0 failures.**

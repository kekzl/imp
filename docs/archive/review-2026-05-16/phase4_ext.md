# Phase 4 — INTEGRATOR Extensibility Audit

Anchor commit: `f58eb9e` (matches Phase 1/2/3). Target: sm_120a / RTX 5090 only.
Citations are `file:line` against the working tree. I do not look at history;
I only ask "if I were a contributor on day-1 with the docs and the code,
what would I have to touch?"

The headline of this audit, repeated up front because every section restates
a facet of it: **today, integrating a new model architecture is not a "model
plugin" exercise — it is surgery across `model/` (loader + tokenizer +
weight upload), `graph/` (executor + pre-dequant + 30+ inline branches),
`runtime/` (engine policy + chunked prefill gate) and `tests/` (5+ new
gtest fixtures). The single biggest extensibility friction is that
"per-architecture behavior" lives as `if (cfg.arch == ModelArch::GEMMA4) {}`
sprinkled across 40 sites in 4 files instead of a polymorphism seam.**
Sections 1, 2, 3 concretize this from three different angles; sections
4-7 propose the ideal-state and a sequenced roadmap.

---

## 0. Today's integration anatomy (warm-up)

How does a model on disk become a working `imp_decode_step`?

### 0.1 The eight touch points

| # | Touch point | Files (file:line) | Concern owned |
|---:|---|---|---|
| 1 | **C-API enum addition** | `include/imp/types.h:24-39` (`ImpModelArch`) | Public ABI: every new arch must add a constant here; current count = 14. |
| 2 | **C++ enum + name registry** | `src/model/model_arch.h:7-22` (`ModelArch`); `src/model/model.cpp:117-135` (`kArchRegistry`); `src/model/model.cpp:155-212` (`parse_model_arch` GGUF + HF strings) | Sole source of truth for arch name, C-API ID, sampling defaults, FP32-router/embed-scale flags. |
| 3 | **Loader: GGUF arch dispatch** | `src/model/gguf_loader.cpp:1032` (`cfg.arch = parse_model_arch(arch_str)`); `src/model/gguf_loader.cpp:1471, 1514` (Gemma-4 specials); `src/model/tensor_kind_matcher.cpp:1-165` (tensor-name → `TensorKind` mapping); `src/model/tensor_kind_table.cu:22-50` (per-kind storage tier) | Reads GGUF metadata; decides which tensors to expect; per-kind quant-format negotiation. |
| 4 | **Loader: HF SafeTensors arch dispatch** | `src/model/hf_config_loader.cpp:21-49` (HF class-name → `ModelArch`); `src/model/safetensors_loader.cpp` (1 332 LOC, weight stream); `src/model/llm_compressor_loader.cpp:1-321` (NVFP4 prequant sidecar) | Same job as #3 but for SafeTensors / HF config.json. |
| 5 | **Weight uploader + per-arch quant transforms** | `src/model/weight_upload.cu` (2 092 LOC, 1 public function, giant `switch(qtype)` × per-arch); `src/model/weight_upload.cu:1956` (`IMP_AUDIT_NVFP4_SCALES`) | GGUF dequant, GPTQ unpack, NVFP4 sidecar promotion, host-pinned expert split. |
| 6 | **Forward graph layer schedule** | `src/graph/executor_forward.cu:380-417` (per-layer attn/SSM/GDN/FFN/MoE selector via `layer_has_*`); `src/graph/executor_workspace_config.cu:287-301` (`layer_has_attention/_ssm/_gdn/_moe/_dense_ffn` — all decided by **tensor presence**, not enum); `src/graph/executor_attention.cu:140` (`run_attention`); `src/graph/executor_ffn.cu:31` (`run_ffn`); `src/graph/executor_forward_moe.cu:146` (`run_moe_ffn`); `src/graph/executor_ssm_gdn.cu` (`run_ssm`, `run_gdn`) | Per-arch attention / FFN / MoE behavior tweaks live as inline `if (cfg.arch == ModelArch::GEMMA4) { ... }` (30 sites in attention, 19 in MoE — see §0.3). |
| 7 | **Tokenizer + chat-template wiring** | `src/model/tokenizer.cpp` (2 108 LOC, BPE + SentencePiece + add_bos heuristics); `src/model/chat_template.cpp:71-102` (`default_family_for_arch` switch over 13 enum values); `src/model/jinja.cpp` (2 629 LOC hand-rolled Jinja2 evaluator) | New arch must either map to one of the 7 existing ChatTemplateFamily values or add a new family. |
| 8 | **Engine policy hooks** | `src/runtime/engine.cpp:828-880` (`init_request_context_` Gemma-4 carve-outs); `src/runtime/engine.cpp:1663` (warmup skip); `src/runtime/engine.cpp:1864-1911` (`supports_chunked_prefill_()` arch reject list); `src/runtime/engine.cpp:1913-1929` (chunk-size resolver); `src/runtime/engine.cpp:1410` (`has_pure_ssm_layers_` set from layer-presence count); `src/runtime/vram_budget.cpp:56` (SSM state budget); `src/runtime/storage_planner.cpp` (per-tensor storage-tier picks) | Per-arch deterministic GEMM, FP8-prefill, NVFP4-decode, warmup, chunked-prefill, KV-dtype gates. |
| 9 | **Tests: unit + e2e + perf gate** | `tests/test_e2e_models.cpp:140-460` (`PrimaryModelTest`, `GDNModelTest`, `Gemma4ModelTest`, `Gemma4GraphsTest` — each one a separate fixture class, **gated on a separate env var**: `IMP_TEST_MODEL`, `IMP_TEST_MODEL_GDN`, `IMP_TEST_MODEL_GEMMA4`); `tests/test_degeneration.cpp` (multi-turn coherence); `tests/test_chunked_prefill.cu` (chunked path); `tests/perf_baseline.json` (3% decode / 5% prefill thresholds — but only for **Q8_0 baseline models**, not new arches) | New arch needs a new fixture file or a new TEST_F class plus a new env var. |

(I list 9 because tests deserve their own row.)

### 0.2 The path from bytes to `imp_decode_step` (one-screen)

```
imp_model_load (api/imp_api.cpp:152) — switch(format)
  └── load_gguf (model/gguf_loader.cpp:1032) parse_model_arch
       └── reads GGUF KV-pairs, builds ModelConfig
       └── for each tensor in GGUF: tensor_kind_matcher → upload via weight_upload.cu
            └── weight_upload reads tensor_kind_table.cu storage tier
                 └── for NVFP4 quant: llm_compressor_loader sidecar
       └── apply_arch_defaults (model/model.cpp:214) sets rope_neox/embed_scale/sigmoid_gating
  OR load_safetensors (model/safetensors_loader.cpp) parse_model_arch
       └── same downstream

imp_context_create (api:255) → Engine::init (runtime/engine.cpp)
  └── supports_chunked_prefill_() — arch reject list at l.1864-1911
  └── per-arch carve-outs at l.828-880 (Gemma-4)
  └── builds GraphExecutor (graph/executor.cu) — owns WeightCaches, ssm_state_, gdn_state_
       └── pre_dequant_weights (graph/executor_pre_dequant.cu) — per-qtype/per-arch caches
  └── builds KV cache (memory/kv_cache_manager.cpp)
  └── builds chat template (model/chat_template.cpp:71 — switch over arch enum)

imp_prefill (api:540) → Engine::step (runtime/engine.cpp:1735)
  └── step_prefill chunk loop → executor->forward_logits
       └── executor_forward.cu:174 — per-layer dispatch:
            if (layer_has_gdn(i))      run_gdn
            else if (layer_has_ssm(i)) run_ssm
            else if (layer_has_attention(i)) run_attention  ← 14 GEMMA4 branches inside
            then if (layer_has_moe(i)) run_moe_ffn          ← 19 GEMMA4 branches inside
            else if (layer_has_dense_ffn(i)) run_ffn

imp_decode_step (api:661) → same Engine::step → step_decode → step_decode_forward
  └── (graph-captured fast-path replays, see Phase 2 §4)
```

The **layer-kind switch is by tensor presence** (`layer_has_*` each just checks
`model_->layer(i).wq.data != nullptr` etc., per `executor_workspace_config.cu:287-301`).
That is a small but real abstraction — adding a hybrid pattern only requires
populating the right tensors. The **arch-specific tweak** is, however,
not abstracted: it lives as 40 inline `if (cfg.arch == ModelArch::GEMMA4)`
branches across `executor_attention.cu` (14 branches), `executor_forward_moe.cu`
(19 branches), `executor_forward.cu` (2 branches), `engine.cpp` (5+ branches),
and `executor_workspace.cu` (1 branch). See `phase3_maint.md` §1.5.

### 0.3 Why this is the warm-up, not the meat

The eight touch points above are **discoverable**: a contributor doing
`grep ModelArch::QWEN3 src/` (or even `git log --grep "feat(qwen3)"`) finds
all of them in 30 minutes. The friction in §§1-3 is what happens once
they start writing the code — the `if`-ladder hot-path means every "tweak
this attention scale for arch X" is a 4-file diff, not a 1-file plugin.

---

## 1. Simulated integration: Qwen3.5-35B-A3B (MoE with new routing variant)

The hypothetical: a Qwen3.5-style hybrid (24 GDN + 8 attention + 32 FFN
layers) but with a **35B-A3B MoE pattern** (similar to Qwen3.6-35B-A3B-MoE,
which already exists at `ModelArch::QWEN36_MOE`) and a **new routing
variant** — e.g. router uses sigmoid-gated top-k with router-bias-norm-add
(like Nemotron-H's `moe_sigmoid_gating` flag, see `model_config.h:72`)
plus a new **per-token expert-weight calibration** scheme that does not
exist today.

### 1.1 Files that would need a diff

| # | File:lines | LOC delta | Why |
|---:|---|---:|---|
| 1 | `include/imp/types.h:24-39` | +1 | Add `IMP_ARCH_QWEN35_A3B` (or reuse QWEN35_MOE — see §1.5). |
| 2 | `src/model/model_arch.h:7-22` | +1 | Mirror enum. |
| 3 | `src/model/model.cpp:117-135` | +1 row | `ArchEntry` registry: name, c_api id, sampling defaults, sigmoid-gating flag. |
| 4 | `src/model/model.cpp:155-212` | +3-5 | `parse_model_arch` GGUF and HF mappings (`Qwen3_5MoeForCausalLM`, GGUF `qwen35moe_a3b`). |
| 5 | `src/model/model.cpp:214-236` | +0-3 | `apply_arch_defaults` — flags already covered by `kArchRegistry` row. |
| 6 | `src/model/model_config.h:67-72` | +1-2 | Add `moe_router_bias_norm` boolean, or piggyback on `moe_sigmoid_gating`. |
| 7 | `src/model/gguf_loader.cpp` (loader for new metadata key, e.g. `qwen35moe.expert_weights_calibration`) | +20-40 | Read new calibration tensors / scalars. |
| 8 | `src/model/hf_config_loader.cpp:21-49` | +2 | HF class-name registration. |
| 9 | `src/model/tensor_kind_matcher.cpp` | +5-15 | If new GGUF tensor names appear (e.g. `blk.%d.expert_calibration_scale`), add matcher rules. |
| 10 | `src/model/tensor_kind_table.cu:22-50` | +1 | New `TensorKind::EXPERT_CALIB_SCALE` (FP32 storage). |
| 11 | `src/model/weight_upload.cu` | +30-80 | Wire new tensor through upload + dequant. |
| 12 | `src/model/chat_template.cpp:71-102` | +1 | Map enum → `ChatTemplateFamily::CHATML` (Qwen line uses CHATML). |
| 13 | **`src/compute/moe_routing.cu`** | +50-150 | New routing variant kernel — `topk_gating_sigmoid_kernel` exists at l. 26 (current code already supports `score_bias`); but **per-token calibration** is genuinely new logic and requires a separate launch. |
| 14 | `src/compute/moe_routing.h` | +1-2 sigs | Declare new kernel. |
| 15 | **`src/graph/executor_forward_moe.cu`** | +30-100 | Insert dispatch arm: where `moe_topk_gating` is called (`executor_forward_moe.cu:2207-2210`), branch on `cfg.moe_router_bias_norm` (or arch enum). Today this is **inline if**, not a virtual call. |
| 16 | `src/graph/executor_attention.cu` | +5-50 | Only if Qwen3.5-A3B has GDN+attention layers with new attention tweak (e.g. partial RoPE). Per `qwen35_partial_rope_fix_2026_04_23` memo, partial RoPE is already handled. |
| 17 | `src/graph/executor_pre_dequant.cu` | +10-30 | Calibration scales need a per-layer cache slot if dequant needed. |
| 18 | `src/runtime/engine.cpp:828-880, 1864-1911` | +5-15 | If new arch has chunked-prefill quirks: add to/remove from `supports_chunked_prefill_()` reject list. |
| 19 | `src/memory/gdn_state.cu`, `src/memory/ssm_state.cu` | 0 | Reuse — Qwen3.5 GDN already supported (see `qwen35_gdn.md` memo). |
| 20 | `src/runtime/vram_budget.cpp:56-70` | 0-5 | If A3B expert count / shape differs, may need an exception. |
| 21 | `src/runtime/storage_planner.cpp` | +0-10 | Only if new tensor kind needs a non-default tier. |
| 22 | `src/runtime/presets.cpp` | +1 row | Per-arch CLI / server preset — `--preset qwen35-a3b`. |
| 23 | **Tests:** `tests/test_e2e_models.cpp` | +60-120 (new `Qwen35A3BModelTest` fixture) | Mirror `GDNModelTest` (l. 144). New env var `IMP_TEST_MODEL_QWEN35_A3B`. |
| 24 | `tests/test_moe.cu` | +30-80 | New routing-variant unit test. |
| 25 | `tests/test_chunked_prefill.cu` | +0-30 | Only if chunked-prefill behavior differs. |
| 26 | `tests/perf_baseline.json` | +1 row | If user wants a CI gate for the new arch. |
| 27 | `docs/usage.md`, `docs/roadmap.md` | +5-15 | Document new `--arch qwen35_a3b` and any new env vars. |

**Total file count touched: 25-27 files.** **LOC delta: ~280-650** (with
the kernel-side routing variant as the dominant chunk).

### 1.2 Where does the new routing logic plug in?

There is **no clean seam** today.

- The routing logic is invoked inside `executor_forward_moe.cu:2207-2210`
  via a single `moe_topk_gating(...)` call, which itself dispatches inside
  `compute/moe_routing.cu:26` (`topk_gating_kernel` already supports a
  `score_bias` for sigmoid models).
- A new routing variant must either (a) add another kernel + branch in
  `executor_forward_moe.cu`, OR (b) extend `topk_gating_kernel` with another
  parameter and pass-through.
- The 5 dispatch paths inside `executor_forward_moe.cu` (header at l.5-10:
  decode_fast, TC fused, scalar fused, batch path, shared expert path) all
  call into routing in slightly different ways. If the new variant
  applies at decode time, you need to update **5 sites**. If only at prefill,
  you need to update at least 2.

The smell: there is no `MoeRouter` virtual interface. Compare with the
`ModelArchAdapter` proposed in Phase 3 §11 Refactor #1 — the same problem.

### 1.3 Magic numbers / hardcoded shapes / hardcoded dtype tables

- `compute/moe_routing.cu:533` (`moe_fused_permute_kernel`) — single-block
  scan, `__launch_bounds__(256)`. **Hard-codes n_experts ≤ 1024.** If
  Qwen3.5-A3B has, say, 384 experts (as Qwen3-Coder-30B-A3B does, which
  works), fine. If it has > 1024, this kernel breaks silently.
- `quant/nvfp4_gemm.cu:855` — `gemv_nvfp4_moe_decode_kernel` with
  `__launch_bounds__(128, 12)` is the MoE decode hot kernel (Phase 2 §2.3).
  Already shape-generic, but the SF-stride math assumes group_size = 16
  (NVFP4 default).
- `graph/executor_pre_dequant.cu` — the `cache_moe_native_nvfp4` lambda
  (l. 1509+ per `qwen36_status_2026_05_02` memo) has self-tracked logical
  budget rather than `cudaMemGetInfo` polling. New arches with > 120 packed
  expert tensors hit the budget if not budgeted in `vram_budget.cpp:56-70`.
- `model/tensor_kind_table.cu:22-50` — hardcoded per-kind storage tiers
  (`StorageTier::NVFP4` for `EXPERT_GATE/UP/DOWN`). A new kind that should
  default to FP8 would need an explicit entry.
- `executor_workspace.cu:80, 119` — SSM workspace caps prefill `max_tokens`
  to 256 for SSM/GDN+MoE hybrids (per `nemotron_h_moe_imp_broken_2026_05_04`
  memo). New hybrid arch inherits this cap silently.

### 1.4 Tests to add

- **New gtest fixture** in `tests/test_e2e_models.cpp` (mirror
  `GDNModelTest` at l. 144, ~80 LOC). Gated on a new env var
  `IMP_TEST_MODEL_QWEN35_A3B`.
- **`tests/test_moe.cu`** — extend with a `RoutingSigmoidWithBiasNorm`
  test if the variant is non-trivial.
- **NOT parameterizable today.** The `tests/test_e2e_models.cpp` pattern
  is **per-arch fixture class** with a per-arch env var. There is no
  `INSTANTIATE_TEST_SUITE_P` over a list of model paths. Adding a new
  parameterization framework (e.g. `tests/test_e2e_param.cpp` over a
  config table) would unblock adding new arches without copy-paste — but
  that doesn't exist today.
- **Perf baseline:** to add the model to the CI gate, also add a row to
  `tests/perf_baseline.json` (file format per CLAUDE.md). Decode numbers
  only — prefill is unstable per CLAUDE.md "pp512 varies up to 2.6×".

### 1.5 Effort estimates

| Persona | Wall-clock | Notes |
|---|---:|---|
| **Competent contributor (knows codebase)** | **2-4 days** | If routing variant is a kernel parameter tweak (sigmoid + bias-norm-add): half a day. If it requires a new fused kernel and the per-token calibration: 2-3 days. The 25-file diff is mostly mechanical (enum + registry + matcher rules + 1 test fixture); the kernel work is the long pole. |
| **New contributor on day 7** | **2-3 weeks** | Day 1-3: read CLAUDE.md, CONTRIBUTING.md, find the touch points (no central "Adding a new model" doc exists; `docs/roadmap.md` lists *what* is supported, not *how* to add). Day 4-7: get a first GGUF load working (probably stuck on `tensor_kind_matcher.cpp` not finding the new tensor names; debugging is `IMP_LOG_DEBUG` + grep). Week 2: get the forward pass to produce non-garbage by chasing `if (cfg.arch == ...)` branches in `executor_attention.cu` + `executor_forward_moe.cu` and discovering each by `git log -p`-ing the Qwen3.6 PRs. Week 3: chunked prefill + perf bench + tests. |

---

## 2. Simulated integration: Gemma 4 26B-A4B (new attention variant — already partially done)

Today's Gemma-4 integration touched, by `git log` evidence and by the
40-grep at `phase3_maint.md` §1.5: `executor_attention.cu` (14 branches),
`executor_forward_moe.cu` (19 branches), `executor_forward.cu` (2
branches), `engine.cpp` (5+ branches), `executor_workspace.cu` (1 branch),
plus all the loader/registry plumbing. The work that landed was driven by
seven separate memos: `gemma4_working_2026_04_14`, `gemma4_rope_freqs_fix`,
`gemma4_chunked_prefill_2026_05_15`, `gemma4_q4km_vs_q8_2026_04_19`,
`gemma4_paged_attention_hd512_bug` (archived), `gemma4_fp8_kv_2026_04_29`
(archived), `gemma4_3120_token_vram_limit` (archived).

### 2.1 Where do the *attention*-specific changes live (not the SWA / chunked-prefill bits)?

Stripping out the SWA/chunked prefill (which `gemma4_chunked_prefill_2026_05_15`
shipped), the surviving Gemma-4-attention tweaks are:

| File:line | What it does |
|---|---|
| `src/graph/executor_attention.cu:161` | Gemma-4 hd-init guard — when `wq.data` present, allow non-default head_dim path. |
| `src/graph/executor_attention.cu:310, 387` | FP32 accumulator promotion (per `gemma4_working_2026_04_14` MoE precision drift fix). |
| `src/graph/executor_attention.cu:464, 472` | **Gemma-4 V=K compaction** (`wv.data == nullptr`, `v_norm_ones_buf_`) — Gemma-4 specific weight layout where V is reused from K. |
| `src/graph/executor_attention.cu:493, 534` | SWA layer dispatch using `cfg.swa_layers`. |
| `src/graph/executor_attention.cu:596, 658` | Per-layer rope_freqs override (per `gemma4_rope_freqs_fix` memo) — global layers consume llama.cpp's freq_factors with hd=512. |
| `src/graph/executor_attention.cu:678` | Gemma-4 attention scale = 1.0 (not 1/sqrt(hd)). |
| `src/graph/executor_attention.cu:893` | Post-attention norm placement (Gemma-style). |
| `src/graph/executor_attention.cu:1198, 1274` | FP32 attention out + post-attention-norm Gemma branches. |

That's 14 inline branches for what is morally **one design decision**:
"Gemma-4 has dual head_dim, V=K layout, FP32 residual stream, rope_freqs
on globals, no scaling, post-attn-norm." Each landed in its own
PR after a separate bug.

### 2.2 If Gemma-5 with a new attention variant showed up tomorrow

A hypothetical Gemma-5 with, say, **per-layer rope theta cycling** + **K-norm
+ V-norm fused** + **a new attention-output bias term** would require
touching:

| # | File:lines | LOC delta | Why |
|---:|---|---:|---|
| 1-12 | All §1 items 1-12 (enum + registry + parser + chat template) | +30-50 | Same boilerplate, no shortcut. |
| 13 | `src/model/model_config.h` | +5-10 | New per-layer fields (rope_theta_per_layer, attn_out_bias). |
| 14 | `src/model/gguf_loader.cpp:1471, 1514` | +30-60 | New per-layer tensors and metadata keys. |
| 15 | `src/model/weight_upload.cu` | +50-150 | Upload new per-layer tensors + bias. |
| 16 | **`src/graph/executor_attention.cu`** — would need **8-15 new `if (cfg.arch == ModelArch::GEMMA5)` branches** | +100-300 | Per-layer rope-theta dispatch (no infrastructure today), K-norm+V-norm fusion (V-norm doesn't exist), attention output bias (no codepath). |
| 17 | `src/compute/rope.cu` | +20-40 | New rope variant template. |
| 18 | `src/compute/layernorm.cu` | +30-80 | V-norm if dtype/layout new. |
| 19 | `src/runtime/engine.cpp:828` | +20-40 | Mirror Gemma-4 carve-outs (warmup skip, deterministic GEMM, NVFP4 decode). |
| 20-25 | Tests, perf-baseline, docs | +120-200 | New `Gemma5ModelTest`, `Gemma5GraphsTest`. |

**Total: 23-25 files; LOC delta ~410-980. Wall-clock for a competent contributor: 5-12 days.** Wall-clock for a new contributor on day 7: **3-5 weeks** because Gemma-style layout (V=K compaction, dual head_dim, FP32 residual) is famously the hardest area in the codebase per the memos.

### 2.3 Different from §1?

**Yes, in two ways:**

1. **Quantitatively** — §1 (Qwen3.5-A3B) reuses GDN + MoE + sigmoid-gating
   that already exist. Hypothetical Gemma-5 introduces 3 new attention-side
   features that have **no codepath today**, so the diff is dominated by
   **net-new kernel work**, not registry boilerplate.
2. **Architecturally** — §1's per-layer routing variant has at least a
   single integration point (`moe_topk_gating` at `executor_forward_moe.cu:2207`).
   §2's per-layer rope theta + V-norm would need to thread a new
   per-layer-tensor field through 4 call sites in `executor_attention.cu`
   (QKV proj, RoPE fused, RoPE separate, KV-write+RoPE) — and each one
   has its own dtype overload set.

### 2.4 Where is "the attention variant for arch X" decided today?

**Nowhere centrally.** It is decided at every read site. Concrete:

- "Attention scale" — `executor_attention.cu:678`:
  `float scale = (cfg.arch == ModelArch::GEMMA4) ? 1.0f : (1.0f / std::sqrt(hd));`
- "Should I use rope_freqs?" — `executor_attention.cu:447-451` (per
  `gemma4_rope_freqs_fix` memo) — set inside the QKV-proj branch.
- "V exists or V==K?" — `executor_attention.cu:464` —
  `if (cfg.arch == ModelArch::GEMMA4 && ly.wv.data == nullptr && ...)`.
- "Use FP32 accumulator?" — `executor_attention.cu:310, 387, 1198` —
  three separate branches, each with its own `gemma4 && using_fp32_accum`
  predicate.

This is the canonical "polymorphism-by-if-ladder" smell. Phase 3 §11
Refactor #1 (`ModelArchAdapter`) is the proposed cure; until that lands,
adding a new attention variant means touching 8-15 inline branches.

### 2.5 Effort estimates — Gemma-5

| Persona | Wall-clock | Notes |
|---|---:|---|
| **Competent contributor** | **5-12 days** | Net-new kernels (rope variant + V-norm + attn-out bias) are the long pole. The 14 inline branch additions are mechanical given Gemma-4 as template. |
| **New contributor on day 7** | **4-6 weeks** | Same memos exist for Gemma-4, but the contributor would have to read all 7 of them to understand what *not* to break. Also: Gemma-style models are uniquely sensitive to numerical drift (per `gemma4_layer_diff_2026_04_17` memo + dump infrastructure at `executor_forward.cu:481-493`). |

---

## 3. Simulated integration: hypothetical Mamba2-Hybrid (Nemotron-H-style)

The hardest of the three. Nemotron-H is **already nominally supported** —
`ModelArch::NEMOTRON_H_MOE` exists at `model_arch.h:12`, has a registry
entry at `model.cpp:124-125`, parses `nemotron_h_moe` GGUF and
`NemotronHForCausalLM` HF strings. **It is also archived as broken** per
`nemotron_h_moe_imp_broken_2026_05_04` memo: silent prefill hang at chunk
#3+ (>~470 prompt tokens) due to SSM-state-handoff between prefill chunks.

### 3.1 Map of the work

If the goal were **finishing** Nemotron-H (not adding it from scratch),
i.e. fixing the ~470-token cliff:

| Area | File:lines | LOC delta | Why |
|---|---|---:|---|
| **SSM state handoff** | `src/memory/ssm_state.cu` (~270 LOC, init only); `src/graph/executor_ssm_gdn.cu` (run_ssm); `src/runtime/engine.cpp` chunked-prefill loop (~l. 1935-2050) | +50-200 | Per memo: state at end of chunk N must persist into chunk N+1. Today the state buffer is per-sequence × per-SSM-layer, but the chunk-boundary write/read pattern is broken. |
| **Workspace cap removal** | `src/graph/executor_workspace.cu:80, 119` | +5-20 | "Capping max_tokens 4096 → 256 for SSM/GDN hybrid" — SSM workspace cap forces small chunks; lifting it requires shape-aware allocation. |
| **Graph capture across chunks** | `src/runtime/cuda_graph.cu`, `src/runtime/engine_graph_decode.cpp` | +30-80 | Today decode-graph captures fine for SSM at n=1 (per `cuda_graphs_moe_works_2026_05_07`). Prefill capture probably re-instantiates per-chunk; if that's the deadlock, change to per-chunk-shape pool. |

If the goal were **adding** a fresh Mamba2-only (non-MoE, non-hybrid) arch:

| Area | File:lines | LOC delta | Why |
|---|---|---:|---|
| Loader plumbing | §1 items 1-12 | +30-50 | Boilerplate. |
| **SSM kernel deltas vs GDN** | `src/compute/ssm.cu` (~480 LOC, exists); `src/compute/gdn.cu` (~410 LOC) | +0-200 | Mamba2 SSM scan kernel exists at `ssm.cu:1` and is shared with Nemotron-H. A fresh Mamba2 variant (e.g. Mamba-2.5 with new gate scheme) would need a new kernel. |
| State subsystem | `src/memory/ssm_state.{cu,h}` | +50-100 | New state shape (e.g. interleaved chunked-state) — see §3.2 for the polymorphism gap. |
| Forward graph schedule | `src/graph/executor_forward.cu:380-417` | 0 | The `layer_has_*` schedule already supports per-layer SSM/attention typing; tensor presence drives dispatch. |
| Chunked-prefill | `src/runtime/engine.cpp:1864-1911` | +5-10 | New arch enters/exits the reject list. |
| Tests | `tests/test_e2e_models.cpp` | +100 | New fixture. |

### 3.2 State management: GDN vs Mamba2 cache shape

- **GDN cache** (`src/memory/gdn_state.h:15-50`):
  `init(n_gdn_layers, max_sequences, n_heads, head_dim, state_dim, dtype)`.
  Per-layer recurrent state; allocated once per session.
- **SSM cache** (`src/memory/ssm_state.h:15-55`):
  `init(n_ssm_layers, max_sequences, conv_channels, conv_kernel, n_heads, head_dim, state_dim, dtype)`.
  Per-layer conv1d state + ssm scan state; shape differs from GDN.

**The two are entirely independent classes.** No common base. Per
`runtime/engine.cpp:1410`, the engine sets a separate boolean
`has_pure_ssm_layers_` based on layer-presence count.

**Polymorphism gap:** A fresh hybrid arch with a *different* recurrent
state layout (e.g. RWKV-7 cache, RetNet retention state) would need a
*third* state class — there is no `RecurrentState` interface to inherit
from. Adding the third class means:
- New `src/memory/foo_state.{cu,h}` (~200-400 LOC).
- New field `foo_state_` in `Engine` and `GraphExecutor` (~5 sites each).
- New init / reset / swap call sites (5-10 places).
- New VRAM budget arm in `vram_budget.cpp:56-70` (~10 LOC).

### 3.3 Graph capture for hybrid layer schedule

Today's MoE-prefill-graphs work (per `moe_prefill_graphs_plan_2026_05_10`)
covers Phase 3 (+11-39%) and partial Phase 4. The hybrid case:

- **Decode-graph** captures cleanly today on hybrid (per
  `cuda_graphs_moe_works_2026_05_07` — Qwen3.5-GDN + Qwen3.6-MoE both
  graph-replay fine).
- **Prefill-graph** has the SSM-state-handoff bug at chunk boundaries
  (per the Nemotron-H archived memo). A fresh hybrid arch would inherit
  this bug.
- **Per-layer schedule diversity** (e.g. 24 SSM + 8 attn + 32 FFN in
  Qwen3.5; or different ratios in a new arch) is handled by `layer_has_*`
  predicates without code changes — that part is fine.

### 3.4 Loader story: per-layer typing

Today's loader **does** support per-layer typing — the assumption of
"homogeneous architecture" doesn't really exist:

- `src/model/model_config.h:27-31` —
  `n_kv_heads_per_layer`, `d_ff_per_layer`, `head_dim_per_layer`,
  `n_heads_per_layer`, `swa_layers`. All per-layer, populated by GGUF
  loader for hybrid models.
- `tensor_kind_matcher.cpp:60-90` matches tensors by name pattern
  (e.g. `blk.X.ssm_in.weight`) regardless of arch — so the *which layers
  have SSM* fact emerges from tensor presence.

So a new hybrid loader fits the existing pattern. The friction is in
chunked-prefill + state handoff (§3.1, §3.2), not in loading.

### 3.5 What blocked Nemotron-H last time

Per `nemotron_h_moe_imp_broken_2026_05_04` (archived) memo + verified
code evidence:

- **Initially:** IMA + cuBLAS status=13 + lm_head crash. Resolved by
  commit `5b2c5db` (route SSM in_proj/out_proj through CUTLASS NVFP4
  fast-path; the slow fallback `gemm_nvfp4()` allocated ~52 MiB scratch
  per-layer per-GEMM-call → IMA).
- **Surviving:** silent prefill hang at chunk #3+ (>~470 prompt tokens).
  Per-chunk SSM-state-handoff broken; engine never returns; chunk #2 stays
  in same graph, chunk #3 triggers a new capture-pair that deadlocks.
- **Workspace cap:** `executor_workspace.cu:80, 119` caps `max_tokens`
  to 256 for SSM/GDN+MoE hybrids. Per `nemotron_h_moe_reserve_env_2026_05_07`
  memo, raising this cap is non-trivial because the SSM workspace is
  budgeted statically.

Is the blocker still in place? The memo is 11 days old (cut off
2026-05-05). The current `executor_workspace.cu:80` still contains the
cap. The current `engine.cpp:1864-1911` `supports_chunked_prefill_()`
function does **not** reject hybrid GDN+MoE archs (it allows QWEN35*,
QWEN36_MOE, NEMOTRON_H_MOE per the comment) — so chunked prefill is
allowed at the API level, but the actual SSM-state handoff bug at
chunk-boundary is unresolved per code inspection (no PR to
`memory/ssm_state.cu` since 2026-05-04). **Blocker still in place.**

### 3.6 Effort estimate plus confidence band

| Task | Wall-clock | Confidence |
|---|---:|---|
| **Finish Nemotron-H (chunk-boundary state fix)** | **2-6 weeks** | LOW-MEDIUM. Nature of the bug is silent hang; debugging requires nsys traces of cross-chunk graph re-instantiation. The fix may be 50-200 LOC but finding it could take a week. |
| **Add fresh Mamba2-only arch** (no MoE, no hybrid) | **1-2 weeks** | MEDIUM. Reuses existing `ssm.cu` + `ssm_state.cu`. Mostly registry + 1 fixture. Risk: if the new SSM has different state shape, new cache class needed. |
| **Add fresh hybrid arch with non-Mamba2 recurrent layer (e.g. RWKV-7)** | **8-16 weeks** | LOW. Net-new state class, net-new kernel, net-new chunked-prefill bring-up, plus inheriting the unfixed Nemotron-H chunk-boundary cliff. |

The Mamba2-Hybrid family carries a structural tax that the dense / MoE
families don't.

---

## 4. Friction-point catalog (ranked, deduplicated)

```
1. Per-architecture behavior is an `if`-ladder, not a polymorphism seam
   [hit in: §1 / §2 / §3]  [severity: H]
   [root location: src/graph/executor_attention.cu (14 GEMMA4 branches),
    src/graph/executor_forward_moe.cu (19), src/graph/executor_forward.cu (2),
    src/runtime/engine.cpp:828, 1663, 1893, 1899]
   Problem: every per-arch tweak is a single-line `if (cfg.arch == ...)`,
     interleaved with code that runs for all arches. Adding a new arch
     means hunting these branches and copy-pasting them with the new enum.
     Removing an arch is unsafe — branches may be load-bearing for
     other arches via the `else` side.
   Fix shape: introduce ModelArchAdapter virtual interface (Phase 3 §11
     Refactor #1); each arch is one .cu file (~400 LOC).
   Cost to fix: 1-2 weeks (per Phase 3 estimate).
   Unblocks: §1 (clean MoE routing seam), §2 (Gemma-5-style attention
     variant lands in 1 file), §3 (hybrid state polymorphism gets a
     parallel adapter pattern).

2. Tests are per-arch fixtures + per-arch env vars (no parameterization)
   [hit in: §1 / §2 / §3]  [severity: H]
   [root location: tests/test_e2e_models.cpp:140-460 (PrimaryModelTest,
    GDNModelTest, Gemma4ModelTest, Gemma4GraphsTest — each their own class,
    each their own env var)]
   Problem: cannot add a new arch test without copy-pasting an entire
     fixture class. CI must export 4+ env vars before any e2e test runs;
     adding model #15 means env var #15 + fixture #15.
   Fix shape: a single `tests/test_e2e_param.cpp` with an INSTANTIATE_
     TEST_SUITE_P over a model-table (path, expected-token, kv-dtype, etc).
     Tests skip per-row when env var unset.
   Cost to fix: 2-3 days.
   Unblocks: every future arch (§1/§2/§3) gets free e2e coverage.

3. No "Adding a new architecture" doc; learnings live in MEMORY only
   [hit in: §1 / §2 / §3]  [severity: H]
   [root location: docs/ has usage.md, sm120.md, performance.md, roadmap.md
    — no integration guide. The 25+ memory files (gemma4_*, qwen35_gdn,
    qwen36_status_*, nemotron_h_*) are the actual TODO log per Phase 3 §5.4]
   Problem: a new contributor on day 1 has no map. Touch points discovered
     by grep + git log + reading memos one at a time.
   Fix shape: docs/integration.md with the §0 touch-point table + a
     worked example (e.g. Qwen3 → Qwen3-Coder added X files).
   Cost to fix: 2-4 hours.
   Unblocks: new contributor day 7 → day 3.

4. Tokenizer/chat-template family registry is closed-set
   [hit in: §1 / §2]  [severity: M]
   [root location: src/model/chat_template.cpp:71 (default_family_for_arch
    switch); src/model/chat_template.h ChatTemplateFamily enum]
   Problem: if a new arch needs a new chat family (e.g. Mistral-3 changed
     theirs), the enum + switch + parser get touched. Today there are 7
     families; some arches reuse (LLAMA→LLAMA3, GEMMA3+GEMMA4→GEMMA).
   Fix shape: load the chat template from the GGUF/HF metadata directly
     when present (the hand-rolled jinja.cpp can already evaluate it);
     keep the enum as fallback.
   Cost to fix: 1-2 days.
   Unblocks: arches with novel chat formats (Llama-4 already had quirks
     per `mistral_3_2_nvfp4_use_default_system_2026_04_28`).

5. Tensor-name matcher is centralized but hard-codes naming conventions
   [hit in: §1 / §2 / §3]  [severity: M]
   [root location: src/model/tensor_kind_matcher.cpp:1-165]
   Problem: new tensor names (e.g. `blk.X.expert_calibration_scale.weight`)
     require explicit case additions. The matcher does NOT delegate to a
     per-arch table — every arch's tensor names go through the same global
     switch. Future arches may collide with name patterns.
   Fix shape: per-arch matcher functions in a registry (similar to
     ChatTemplate); fallback to global. Or: declarative table:
     `{TensorKind::EXPERT_GATE, "blk.{layer}.ffn_gate_exps.weight"}`.
   Cost to fix: 3-5 days.
   Unblocks: arches with novel tensor naming (proven by Gemma-4's
     ffn_pre_norm_2 / ffn_post_norm_1 / layer_out_scale).

6. weight_upload.cu is one file with one giant function (2 092 LOC)
   [hit in: §1 / §2]  [severity: M]
   [root location: src/model/weight_upload.cu (1 public function,
    everything anon-namespace, switch over 12 quant types × 4 archs)]
   Problem: per Phase 3 §2 #10 — one file, two public functions, giant
     switch. Adding a new tensor kind (e.g. EXPERT_CALIB_SCALE) requires
     wedging it in the right anon-namespace function.
   Fix shape: split per qtype family (Phase 3 §10 #18).
   Cost to fix: 1-2 weeks.
   Unblocks: §1 (new EXPERT_CALIB_SCALE kind in a focused file).

7. Per-layer recurrent-state classes are not polymorphic
   [hit in: §3]  [severity: H for Mamba2-Hybrid; M overall]
   [root location: src/memory/ssm_state.{h,cu} and
    src/memory/gdn_state.{h,cu} are independent classes; no common base]
   Problem: a third hybrid (RWKV / RetNet / your own) means a third
     parallel class + 5-10 new wiring sites in Engine/GraphExecutor.
   Fix shape: RecurrentState abstract base; SSMState + GDNState inherit;
     Engine holds vector<unique_ptr<RecurrentState>>.
   Cost to fix: 1-2 weeks.
   Unblocks: any non-Mamba2-non-GDN hybrid.

8. SSM/GDN+MoE hybrid prefill chunk_size capped at 256, state handoff
   broken across chunks
   [hit in: §3]  [severity: H for hybrid arches]
   [root location: src/graph/executor_workspace.cu:80, 119;
    src/memory/ssm_state.cu (no cross-chunk handoff implemented);
    nemotron_h_moe_imp_broken_2026_05_04 archived memo]
   Problem: silent hang past ~470 prompt tokens for any SSM-hybrid arch.
     Documented blocker, unfixed.
   Fix shape: cross-chunk state persist at engine.step_prefill chunk
     boundary; remove static workspace cap once dynamic.
   Cost to fix: 2-6 weeks (LOW confidence per §3.6).
   Unblocks: Nemotron-H, future hybrid arches with prompts > 470 tokens.

9. WeightCaches god-struct + 21-param dispatch + dual dispatch tables
   [hit in: §1 / §2 (kernel arm), §3]  [severity: M for new arches]
   [root location: src/graph/executor.h:286 (WeightCaches struct);
    src/graph/executor_kernels.cu:2003-2269 (gemm_dispatch_impl);
    src/compute/weight_dispatch.cu:73-125 (parallel dispatch)]
   Problem: any new qtype (e.g. INT3, INT2, BFP16) lands in 6 cache maps,
     gemm_dispatch_impl arm, weight_dispatch.cu arm, executor_pre_dequant
     populator. Phase 3 §11 Refactor #2 already designed the fix.
   Fix shape: GemmKernel registry (one file per qtype).
   Cost to fix: 2-3 weeks (Phase 3 estimate).
   Unblocks: §1 (calibration-scale handling), future quant formats.

10. CUTLASS / NVFP4 device-args path silently bypassed when da_cache
    not populated; failure mode is per-layer "fallback to host loop"
    [hit in: §1 (NVFP4 MoE arch)]  [severity: M]
    [root location: src/graph/executor_forward_moe.cu:566-578 (per-layer
     pre-cache); :601-613 (H2D fallback when cache miss);
     qwen36_status_2026_05_02 memo (cache aborts at 16-22/120 entries
     under VRAM pressure → CUDA Graph capture fails silently)]
    Problem: new NVFP4 MoE arch with high expert count or VRAM pressure
      hits the silent fallback; decode drops 5×. Failure logged at
      INFO not ERROR.
    Fix shape: hard-fail or upgrade to ERROR when per-layer cache aborts;
      promote `vram_alloc_force()` pattern to default for prequant MoE.
    Cost to fix: 2-3 days.
    Unblocks: §1 (any future NVFP4 MoE arch survives VRAM-tight prod boxes).

11. CUDA Graphs disabled on host-offloaded experts; no LRU prefetch
    [hit in: §1 / §3 (any A3B-style high-expert-count model)]  [severity: M]
    [root location: src/runtime/engine.cpp:1158-1164; speculative S7 in
     phase2_perf.md §9; docs/roadmap.md:53]
    Problem: model arches that don't fit on 32 GB are forced into
      experts_on_host, which disables graphs (per §4.5 of Phase 2). New
      arches inherit a 5× decode penalty silently.
    Fix shape: device-side LRU prefetch with cudaMemcpyAsync overlap.
    Cost to fix: 4-6 weeks.
    Unblocks: any 50B+ MoE.

12. Gemma-4 chunked-prefill exception; supports_chunked_prefill_()
    arch reject list is hardcoded
    [hit in: §1 / §2]  [severity: M]
    [root location: src/runtime/engine.cpp:1864-1911]
    Problem: each new arch must be either added to the reject list or
      proven safe; the policy lives as inline `if (cfg.arch == ...)`.
    Fix shape: per-arch capability bits on ModelArchAdapter (#1 above).
      `arch_adapter->supports_chunked_prefill()` virtual.
    Cost to fix: subsumed by #1.

13. Per-arch storage tier in tensor_kind_table.cu hardcodes "this kind
    always goes to NVFP4"; no per-arch override
    [hit in: §1 / §2]  [severity: L]
    [root location: src/model/tensor_kind_table.cu:22-50]
    Problem: a new arch where (e.g.) ROUTER should be FP16 instead of
      FP32 has no clean override. Today the only escape is the "qtype
      override" path through weight_upload.cu, which is ad-hoc.
    Fix shape: per-arch storage tier table (key by ModelArch×TensorKind).
    Cost to fix: 1-2 days.
    Unblocks: arches with non-default tier preferences.

14. Engine carve-outs (warmup skip, deterministic GEMM, FP8 prefill flip)
    are arch-coded
    [hit in: §1 / §2]  [severity: M]
    [root location: src/runtime/engine.cpp:828-880, 1663]
    Problem: each carve-out is `if (cfg.arch == ModelArch::GEMMA4) ...`.
      Generalizes from #1 to engine-level policy.
    Fix shape: ModelArchAdapter::engine_policy_overrides() virtual that
      returns a struct of bools (skip_warmup, deterministic_gemm, etc.).
    Cost to fix: subsumed by #1.

15. NVFP4 prequant loader has "Modelopt vs llm-compressor" dichotomy
    surfaced as two boolean fields
    [hit in: §1]  [severity: L]
    [root location: src/model/model_config.h:81-86 (is_nvfp4_prequant +
     is_llm_compressor_nvfp4); src/model/llm_compressor_loader.cpp]
    Problem: Modelopt and llm-compressor differ on whether
      weight_global_scale is a multiplier or divisor; future quant
      vendor (e.g. AWQ-NVFP4 hybrid) needs a third boolean.
    Fix shape: enum NvFP4ScaleConvention { MODELOPT, LLM_COMPRESSOR,
      AWQ_HYBRID, ... } instead of N boolean flags.
    Cost to fix: 4-8 hours.
    Unblocks: clean third-vendor support.

16. moe_routing.cu single-block scan caps n_experts ≤ 1024
    [hit in: §1 (if a future MoE has >1024 experts; Mixtral-style 8
     experts is fine, Qwen3-Coder 384 fine; >1024 may show up)]
    [severity: L today, M for very-large-MoE]
    [root location: src/compute/moe_routing.cu:533 (moe_fused_permute_kernel,
     __launch_bounds__(256))]
    Problem: silent break at >1024 experts; documented in Phase 2 §6.2.
    Fix shape: multi-block scan with two-pass exclusive-sum.
    Cost to fix: 3-5 days.
    Unblocks: very-large-MoE arches.
```

---

## 5. Ideal-state proposal: "<500 LOC = working model"

### 5.1 Proposed `model/` plugin interface

Header `src/model/arch_plugin.h` (sketch ≤ 80 LOC):

```c++
#pragma once
#include "core/tensor.h"
#include "model/model_config.h"

namespace imp {

struct ArchPlugin {
    virtual ~ArchPlugin() = default;

    // === Identity ===
    virtual ModelArch          arch() const = 0;
    virtual const char*        name() const = 0;
    virtual int                c_api_id() const = 0;
    virtual std::vector<std::string> gguf_arch_strings() const = 0;
    virtual std::vector<std::string> hf_arch_classes() const = 0;
    virtual ChatTemplateFamily default_chat_family() const = 0;

    // === Loader hooks ===
    // Register tensor-name matchers (called once at static init time).
    virtual void register_tensor_kinds(TensorKindMatcher& m) const {}
    // Apply arch-specific defaults to ModelConfig after metadata is read.
    virtual void apply_config_defaults(ModelConfig& cfg) const {}
    // Optional: reject obviously-bad config (n_layers etc).
    virtual bool validate_config(const ModelConfig& cfg, std::string* err) const { return true; }

    // === Forward-pass behavior ===
    struct AttentionPolicy {
        bool   v_equals_k                 = false;
        float  attention_scale_override   = 0.0f;   // 0 = use 1/sqrt(hd)
        bool   fp32_residual              = false;
        bool   fp32_attn_out              = false;
        bool   needs_per_layer_rope_freqs = false;
        int    sliding_window_default     = 0;
        bool   use_qk_norm                = false;
    };
    virtual AttentionPolicy attention_policy() const { return {}; }

    struct MoePolicy {
        bool sigmoid_gating         = false;
        bool expert_weights_norm    = false;
        bool router_bias_norm_add   = false;
        bool fp32_router_gate       = false;
        bool fp32_expert_down       = false;
        int  routing_kernel_variant = 0;       // 0 = default topk-softmax
    };
    virtual MoePolicy moe_policy() const { return {}; }

    struct EnginePolicy {
        bool skip_warmup             = false;
        bool force_deterministic_gemm = false;
        bool disable_fp8_prefill     = false;
        bool supports_chunked_prefill = true;
        int  prefill_chunk_default    = -1;     // -1 = auto
    };
    virtual EnginePolicy engine_policy() const { return {}; }
};

class ArchRegistry {
public:
    static ArchRegistry& instance();
    void                  add(std::unique_ptr<ArchPlugin>);
    const ArchPlugin*     find(ModelArch) const;
    const ArchPlugin*     find(const std::string& gguf_or_hf) const;
};

}  // namespace imp
```

Each plugin lives in **one file** under `src/model/plugins/`:
`qwen35_a3b.cpp`, `gemma4.cpp`, `nemotron_h.cpp`. Static registration at
namespace scope:

```c++
namespace { struct Reg { Reg() { ArchRegistry::instance().add(std::make_unique<Qwen35A3BPlugin>()); } } reg_; }
```

### 5.2 How does it interact with `compute/`?

Per Phase 3 §11 Refactor #2 (`GemmKernel` registry): each per-qtype kernel
lives in `src/compute/gemm_kernel_*.cu` and registers itself.

Plugins **do not include compute headers**. They only describe behavior
(`AttentionPolicy`, `MoePolicy`, `EnginePolicy`). The executor reads the
policy and dispatches to the appropriate generic kernel. Any net-new
kernel work is a separate PR in `compute/`, not in the plugin.

### 5.3 How does it interact with `graph/`?

`GraphExecutor` carries a `const ArchPlugin* plugin_` (from the loaded
model). All `if (cfg.arch == ModelArch::GEMMA4)` branches collapse into
`if (plugin_->attention_policy().fp32_residual)`, etc. Phase 3 §11
Refactor #1's `ModelArchAdapter` is essentially the runtime side of this
plugin interface.

### 5.4 What lives in the plugin, what stays central?

| Lives in plugin | Stays central |
|---|---|
| Identity (name, enum, GGUF/HF strings) | Public C-API (`include/imp/types.h`) — generated from the enum, but the enum still exists |
| Per-arch defaults (rope_neox, embed_scale, sigmoid_gating) | `ModelConfig` struct (data only) |
| Tensor-name matcher rules | `TensorKindMatcher` engine |
| Attention policy (V=K, FP32 residual, scale override) | `executor_attention.cu` reads policy and acts |
| MoE routing variant id | `executor_forward_moe.cu` dispatches on id |
| Chat template family | `ChatTemplate` resolves family, then evaluates Jinja |
| Engine policy overrides (skip warmup, deterministic GEMM) | `Engine` reads policy at init |
| **NOT in plugin:** | |
| Kernel implementations | `compute/` |
| Recurrent state classes (SSMState/GDNState/...) | `memory/`, with `RecurrentState` base — but the *picking* of which state class goes by tensor presence, not plugin |
| Loader binary parsers (GGUF, SafeTensors) | `model/gguf_loader.cpp`, `model/safetensors_loader.cpp` — they call into plugin for arch-specific bits |

### 5.5 Concrete sketch — `model/plugins/qwen35_a3b.cpp` (~110 lines)

```c++
#include "model/arch_plugin.h"
namespace imp {
class Qwen35A3BPlugin : public ArchPlugin {
public:
    ModelArch          arch() const override          { return ModelArch::QWEN35_MOE; }
    const char*        name() const override          { return "qwen35moe_a3b"; }
    int                c_api_id() const override      { return /*IMP_ARCH_QWEN35_MOE*/ 11; }
    std::vector<std::string> gguf_arch_strings() const override {
        return {"qwen35moe", "qwen3.5moe_a3b"};
    }
    std::vector<std::string> hf_arch_classes() const override {
        return {"Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"};
    }
    ChatTemplateFamily default_chat_family() const override {
        return ChatTemplateFamily::CHATML;
    }

    void register_tensor_kinds(TensorKindMatcher& m) const override {
        // Hybrid GDN+MoE; reuses default rules. Add A3B calibration tensor:
        m.add_pattern(R"(blk\.(\d+)\.expert_calibration_scale\.weight)",
                      TensorKind::EXPERT_CALIB_SCALE);
    }

    void apply_config_defaults(ModelConfig& cfg) const override {
        cfg.rope_neox = true;
        // GDN already detected by tensor presence; nothing extra here.
        cfg.moe_sigmoid_gating  = true;       // a3b variant
        cfg.expert_weights_norm = true;
    }

    AttentionPolicy attention_policy() const override {
        AttentionPolicy p;
        p.use_qk_norm = true;                   // Qwen3-style per-head RMSNorm
        return p;
    }

    MoePolicy moe_policy() const override {
        MoePolicy p;
        p.sigmoid_gating          = true;
        p.expert_weights_norm     = true;
        p.routing_kernel_variant  = 1;          // sigmoid + bias-norm-add
        return p;
    }

    EnginePolicy engine_policy() const override {
        EnginePolicy p;
        p.supports_chunked_prefill = true;       // hybrid GDN+MoE allowed
        p.prefill_chunk_default    = 256;        // safe per existing cap
        return p;
    }
};
namespace { struct Reg { Reg() {
    ArchRegistry::instance().add(std::make_unique<Qwen35A3BPlugin>());
}} reg_; }
}  // namespace imp
```

That's the entire plugin: ~75 lines of behavior + ~10 lines of registration.

### 5.6 Best first migration to validate the design

**Qwen3 (`ModelArch::QWEN3`).** Reasons:

- Smallest existing plugin surface — uses CHATML, default GDN-free,
  no per-arch quirks beyond `top_k=20` sampling default.
- Migrating it does not interact with the GEMMA4 `if`-ladder (which is
  the harder case — saved for migration #4).
- Has the most coverage in CI (`PrimaryModelTest` uses Qwen3-4B-Q8_0).
- Done correctly, **deletes ~20 lines** from `model.cpp`'s
  `kArchRegistry`, **deletes ~5 lines** from `chat_template.cpp` switch,
  **deletes ~3 entries** from `model.cpp:155` parse map.

If the Qwen3 plugin migration succeeds without test regressions, the
design is validated and the rest of the migrations are mechanical.

---

## 6. Roadmap to ideal state

Sequenced by smallest-unblocking-first. Friction-points reverted are
the §4 numbers in brackets.

### Step 1 — Mechanical prep (3 days)

- Add `docs/integration.md` documenting §0 touch points + worked example.
- Convert `tests/test_e2e_models.cpp` to parameterized `INSTANTIATE_TEST_SUITE_P`
  over a model-table (file + env var driving rows).
- Centralize Modelopt/llm-compressor flag dichotomy (#15) into a small enum.

**Files touched:** `docs/integration.md` (new, ~200 LOC), `tests/test_e2e_models.cpp`
(rewrite, -300+200 LOC), `model_config.h`, `safetensors_loader.cpp`,
`llm_compressor_loader.cpp` (~40 LOC).

**Risk:** LOW. **Reverts:** §4 #2, #3, #15.

### Step 2 — env-var + error-handling sweep (Phase 3 §11 Refactor #5) (4 days)

Already proposed by Phase 3. Run before structural refactors. Centralizes
all 16 IMP_* env reads into `RuntimeConfig`.

**Files touched:** `runtime/config.{h,cpp}` + 16 grep-targeted TUs +
`core/`, `memory/` (~40 sites for throw → return-bool).

**Risk:** LOW. **Reverts:** none structural; unblocks #1 by reducing
hot-path env noise.

### Step 3 — `ArchPlugin` interface + first migration (Qwen3) (1 week)

- Write `src/model/arch_plugin.h` (sketch §5.1).
- Write `src/model/arch_registry.cpp` with `instance()`.
- Migrate Qwen3 to plugin (`src/model/plugins/qwen3.cpp`).
- Wire `chat_template.cpp:71` switch + `model.cpp:117 kArchRegistry` to
  call `ArchRegistry::instance().find(arch)->...` first; fall back to
  current code for un-migrated arches.

**Files touched:** `model/arch_plugin.h` (new, ~120 LOC),
`model/arch_registry.cpp` (new, ~60 LOC), `model/plugins/qwen3.cpp` (new,
~80 LOC), `model/chat_template.cpp` (-10 +5), `model/model.cpp` (-8 +3).
**LOC delta:** +250 net (mostly new infrastructure).

**Risk:** MEDIUM (new abstraction; need test-suite green before
migrating others).

**Reverts:** §4 #4, #13 (per-storage-tier override now per-plugin).

### Step 4 — Migrate remaining 13 arches (1.5 weeks)

Mechanical. Each plugin is ~50-120 LOC. As each is migrated, the
corresponding row in `kArchRegistry` and the corresponding case in
`default_family_for_arch` is deleted.

**Files touched:** 13 new plugin files (~80 LOC × 13 = 1 040 LOC); -80
LOC across `model.cpp` and `chat_template.cpp`. Net **+960 LOC**, but
single-arch surgery now lands in one file.

**Risk:** LOW (each migration validated by existing tests).

**Reverts:** §4 #5 partially (matcher rules can move per-plugin).

### Step 5 — Refactor #1 from Phase 3 §11: ModelArchAdapter / hot-path branches (1.5 weeks)

The plugin policy structs (`AttentionPolicy`, `MoePolicy`, `EnginePolicy`)
become readable in the executor. Replace the 14+19+5 `if (cfg.arch ==
ModelArch::GEMMA4)` branches with `plugin_->attention_policy().X` reads.

**Files touched:** `executor_attention.cu` (-80 LOC, 14 branches replaced),
`executor_forward_moe.cu` (-60 LOC, 19 branches replaced), `engine.cpp`
(-20 LOC), `executor_workspace.cu` (-2 LOC), `executor_forward.cu` (-5 LOC).
**LOC delta:** -150 net.

**Risk:** MEDIUM. Each branch removal needs the Gemma-4 e2e test green.

**Reverts:** §4 #1, #12, #14.

### Step 6 — Refactor #2 from Phase 3 §11: GemmKernel registry (2 weeks)

Mostly orthogonal to the plugin work; runs in parallel with Step 5.
Collapses the dual dispatch tables and the `WeightCaches` god-struct.

**Files touched:** `executor_kernels.cu` (-266 LOC), `executor.h` (-150
LOC), `weight_dispatch.cu` (-100 LOC), 8 new `compute/gemm_kernel_*.cu`
(~150 LOC each), `executor_pre_dequant.cu` (-500 LOC). **LOC delta:** -1
000 net.

**Risk:** MEDIUM. Lots of tests touch these surfaces.

**Reverts:** §4 #6 (weight_upload still big but no longer giant-switch)
indirectly, #9, #10 (the pre-cache populator becomes per-kernel).

### Step 7 — `RecurrentState` polymorphism (1 week)

Introduce abstract base for SSMState + GDNState; engine holds
`vector<unique_ptr<RecurrentState>>`. Sets up for hybrid arch #3.

**Files touched:** `memory/recurrent_state.h` (new, ~40 LOC),
`memory/ssm_state.{h,cu}`, `memory/gdn_state.{h,cu}` (refactored, no LOC
delta), `runtime/engine.{h,cpp}` (~10 sites), `graph/executor.h` and
`graph/executor_ssm_gdn.cu` (~5 sites).

**Risk:** LOW.

**Reverts:** §4 #7.

### Step 8 — Cross-chunk SSM state handoff fix (2-6 weeks)

Real engineering work; per §3.6 LOW confidence. Open Nemotron-H bug.

**Files touched:** `memory/ssm_state.cu`, `runtime/engine.cpp` chunked
path, `graph/executor_workspace.cu` cap removal, `runtime/cuda_graph.cu`
per-chunk pool.

**Risk:** HIGH (silent hangs hard to debug).

**Reverts:** §4 #8.

### Step 9 — Long-tail cleanups (4-8 weeks scattered)

#11 (host-LRU expert prefetch — 4-6 wks), #16 (multi-block routing scan
— 3-5 days). Defer until a specific arch needs them.

### Honesty check

If 80% of the win comes from a 2-week refactor: **Steps 1-3 (~2 weeks)
deliver about 70% of the extensibility win.** The plugin interface is
the load-bearing change; the rest is a mechanical migration that one
contributor can grind through over 2-3 weeks.

If the answer were "12-week rewrite": no, it's not. The codebase is
already well-organized at the directory layer (Phase 1 §1 confirms);
the smell is in the inline if-ladders. Steps 1-5 (~5 weeks) deliver
**95%** of the win.

Steps 6-9 are real engineering work but they're **per-feature** (qtype,
hybrid, chunk-state) rather than per-arch — i.e. their cost amortizes
over future archs.

---

## 7. Current vs. target zero-to-working LOC

Defended by the file-touch counts in §§1-3. "LOC for a new instance"
counts the diff size for **adding a new model**, not building one
from scratch.

| Arch family | Today's LOC for new instance | After §6 Step 1 (docs+param-tests) | After Step 3 (plugin Qwen3 migrated) | After full ideal-state (Steps 1-7) |
|---|---:|---:|---:|---:|
| Dense LLaMA-style (LLaMA, Mistral, Qwen3) | ~60-120 LOC | same (60-120) | ~80-100 (plugin file) | **~50-80 LOC** (one plugin file, no chat_template patch needed) |
| MoE (Qwen3-Coder / Mixtral / Qwen3.5-A3B style) | ~280-650 LOC (§1.1) | ~250-500 (less test boilerplate) | ~200-350 | **~120-200 LOC** (plugin + maybe one routing variant in compute/) |
| Hybrid (Mamba2 + attn — Nemotron-H finished, fresh Mamba2-only) | ~400-800 LOC for new variant; many weeks of debugging for chunk-state cliff | ~350-700 + cliff still unresolved | ~250-500 + cliff resolved (Step 8) | **~200-300 LOC** (plugin + RecurrentState subclass when truly novel) |
| New attention variant (e.g. hypothetical Gemma-5 §2) | ~410-980 LOC (§2.2) | ~370-900 | ~300-700 | **~150-300 LOC** if the variant fits the AttentionPolicy struct; ~400-600 if it requires a new kernel in compute/ |

**Defense of the dense-LLaMA today number (60-120 LOC):** §1.1 items
1-12 plus a chat-template family if needed. The bulk for §1's MoE
example was the kernel work (item 13) and the test fixture (item 23);
a vanilla LLaMA-derivative skips both.

**Defense of the MoE today number (280-650):** §1 explicit count.
Caveats: routing-variant kernel work depends entirely on what's
"new". A pure-config arch (different defaults, no kernel) lands
closer to 100 LOC; a genuine routing innovation needs 150-300 LOC of
kernel code.

**Defense of the hybrid number (400-800 + weeks):** §3 explicit. Adding
the arch-registry plumbing is small; the actual blocker is the
chunk-state cliff (#8) which adds several weeks.

**Defense of the attention-variant number (410-980):** §2 explicit. The
variation in the range tracks whether the attention quirk is expressible
as a flag (cheap) or requires a new kernel (expensive).

**Target column defense:** with the plugin interface in place, anything
that **fits the existing kernel set** lands as a single-file plugin
(~80 LOC) plus optional `EnginePolicy` overrides. Anything that
requires a new kernel or new routing variant adds the kernel work but
no longer requires hunting 25 files for the integration sites.

---

## Appendix A — Anchors for Phase 5 synthesis

- Public C-API enum (extension point #1): `include/imp/types.h:24-39`
- Single-source-of-truth ArchRegistry today: `src/model/model.cpp:117-135`
- HF arch-class registry: `src/model/hf_config_loader.cpp:21-49`
- Layer-kind dispatch (by tensor presence, not enum):
  `src/graph/executor_forward.cu:380-417`,
  `src/graph/executor_workspace_config.cu:287-301`
- Hot-path arch branches:
  - `src/graph/executor_attention.cu:161, 310, 387, 464, 472, 493, 534, 596, 658, 678, 821, 893, 1198, 1274` (14)
  - `src/graph/executor_forward_moe.cu:187, 218, 224, 229, 262, 310, 324, 381, 1727, 1735, 1745, 1766, 1781, 1875, 1893, 1899, 2304, 2538, 2541` (19)
  - `src/graph/executor_forward.cu:447, 513` (2)
  - `src/runtime/engine.cpp:828, 1663, 1893, 1899` (4)
- Chat template family switch: `src/model/chat_template.cpp:71-102`
- Tensor-kind matcher: `src/model/tensor_kind_matcher.cpp:1-165`
- Tensor-kind storage tier: `src/model/tensor_kind_table.cu:22-50`
- Engine chunked-prefill arch reject list: `src/runtime/engine.cpp:1864-1911`
- SSM workspace cap (Nemotron-H blocker): `src/graph/executor_workspace.cu:80, 119`
- Recurrent state classes (no common base): `src/memory/ssm_state.h`,
  `src/memory/gdn_state.h`
- Tests per-arch-fixture pattern: `tests/test_e2e_models.cpp:140-460`

## Appendix B — Memory-file anchor reuse

| Memory file (slug) | Used in section |
|---|---|
| `gemma4_working_2026_04_14` | §2.1, §2.4 |
| `gemma4_rope_freqs_fix` | §2.1, §2.4 |
| `gemma4_chunked_prefill_2026_05_15` | §2 (preamble), §0.1 #8 |
| `gemma4_layer_diff_2026_04_17` | §2.5 |
| `qwen35_gdn` | §1.1 #16, §3.4 |
| `qwen36_status_2026_05_02` | §1.3 (NVFP4 cache), §4 #10 |
| `nemotron_h_moe_imp_broken_2026_05_04` (archived) | §3.5, §3.6, §4 #8 |
| `nemotron_h_moe_reserve_env_2026_05_07` | §3.5 |
| `cuda_graphs_moe_works_2026_05_07` | §3.3 |
| `moe_prefill_graphs_plan_2026_05_10` | §3.3 |

End of Phase 4 integrator audit.

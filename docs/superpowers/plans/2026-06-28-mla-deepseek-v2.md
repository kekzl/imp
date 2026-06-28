# MLA (DeepSeek-V2) Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add correct Multi-head Latent Attention (MLA) support to imp so DeepSeek-V2-Lite (and the DeepSeek/GLM-5/Kimi/Ling MLA family) loads and produces output matching the HF reference, then add the latent-KV-cache compression that makes MLA worth running on a single 32 GB RTX 5090.

**Architecture:** Two stages. **Stage A (Phases 0–2): materialized MLA** — reconstruct full K/V from the latent at projection time so every existing attention/paged-KV/RoPE kernel is reused unchanged. This is correctness-first and fully verifiable against HF. **Stage B (Phase 3): absorbed / latent-cache MLA** — store only the 512-dim latent + 64-dim decoupled-RoPE key in the KV cache and add a paged-latent attention kernel. Stage B is the real VRAM/long-context win and is only started after Stage A passes the perplexity gate.

**Tech Stack:** C++20 / CUDA (sm_120a), GTest, imp SafeTensors + NVFP4 loaders, existing paged KV cache + FA2/paged-decode attention, existing MoE engine (shared experts + routed_scaling already implemented).

## Global Constraints

- **English only in the repo** — code, comments, commit messages, docs, PR text.
- **Target chip: sm_120a only** (RTX 5090, 32 GB). No portability fallback in the hot path.
- **Branch off `main`**, `gh pr create --base main`, never stack PRs.
- **Verification order is mandatory** (add-model-arch skill): loads → coherent greedy output (`check-degeneration` battery) → **perplexity within ~10–20 % of HF reference** → decode/prefill sanity. A kernel that "loads and looks fluent" is the documented MLA failure mode (silent corruption) — PPL-vs-HF is the gate, nothing ships on vibes.
- **GPU hygiene:** before any GPU job confirm `docker ps -q | wc -l` is `0` and `nvidia-smi` shows the card idle (~30 °C). Warm clocks (>1 s discarded run) before any timed number.
- **Reference model:** `/home/kekz/models/DeepSeek-V2-Lite` (bf16 SafeTensors, downloading). Config (verified from HF): `hidden_size=2048`, `num_hidden_layers=27`, `num_attention_heads=16`, `q_lora_rank=null` (NO Q compression — full Q proj), `kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, `rope_theta=10000`, YaRN `factor=40, original_max_position=4096, mscale=0.707, beta_fast=32, beta_slow=1`, `n_routed_experts=64`, `n_shared_experts=2`, `num_experts_per_tok=6`, `moe_intermediate_size=1408`, `first_k_dense_replace=1`, `vocab_size=102400`.
- **Per-head dims:** Q head dim = `qk_nope_head_dim + qk_rope_head_dim` = 128+64 = **192**. V head dim = **128**. So Q/K head dim (192) ≠ V head dim (128) — an asymmetry the standard path does not assume; see Task 2.4.
- Base commit: `390b92cd`. New `cfg.arch == X` checks in hot-path code are banned — dispatch reads `ModelProfile`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/model/model_config.h` | Modify | New `TransformerLayer` MLA fields (`kv_a_proj`, `kv_a_layernorm`, `kv_b_proj` + TensorIDs); new `ModelConfig` fields (`kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`, `mla_mscale`) |
| `src/model/model_profile.h` | Modify | Add `AttnVariant::MLA`; MLA trait derivation |
| `src/model/hf_config_loader.cpp` | Modify | Parse MLA config fields + YaRN mscale; **remove the silent-MHA warning, replace with real path** |
| `src/model/model.cpp` | Modify | `apply_arch_defaults` for DeepSeek MLA dims |
| `src/core/tensor_kind.h` | Modify | New `TensorKind::KV_A_PROJ / KV_A_NORM / KV_B_PROJ` |
| `src/model/tensor_kind_matcher.cpp` | Modify | Match `kv_a_proj_with_mqa` / `kv_a_layernorm` / `kv_b_proj` |
| `src/model/weight_map.cpp` | Modify | DeepSeek HF name routing (attn + MoE expert/shared/router names), field assignment, NVFP4 scale routing |
| `src/exec/executor_attention_qkv.cu` | Modify | MLA two-step KV projection (materialized): x→kv_a→[latent\|k_rope]; RMSNorm(latent); latent→kv_b→[k_nope\|v] |
| `src/exec/executor_attention.cu` | Modify | mscale attention-scale; per-layer V-head-dim handling |
| `src/compute/attention_mla.cu` + `.h` | **Create (Phase 3 only)** | Paged-latent absorbed attention kernel |
| `src/memory/kv_cache.{h,cu}` | Modify (Phase 3 only) | Latent-mode block layout (`[latent_dim + rope_head_dim]` per token) |
| `tests/test_mla.cu` | Create | Projection-shape + materialized-K/V correctness + (Phase 3) latent-cache equivalence |
| `docs/supported-models.md` | Modify | DeepSeek-V2-Lite row |

**Reuse map (from code investigation):** MoE (shared experts, `routed_scaling_factor` = `expert_weights_scale`, per-layer dense-vs-MoE dispatch via `moe_gate` presence, separate `moe_intermediate_size`) is **already implemented** — DeepSeek MoE is a loader-only change. Partial/decoupled RoPE in `src/compute/rope.cu` is **production-ready** (`rope_dim` param, NeoX pairing). Materialized MLA reuses all paged-attention and KV-write kernels.

---

## Phase 0 — Arch detection & config plumbing

### Task 0.1: Parse MLA config fields from HF config

**Files:**
- Modify: `src/model/model_config.h` (add fields to `ModelConfig`)
- Modify: `src/model/hf_config_loader.cpp:610-626` (replace the MLA warning)
- Test: `tests/test_mla.cu` (create)

**Interfaces:**
- Produces: `ModelConfig::kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` (all `int`, default 0), `mla_mscale` (`float`, default 1.0). `bool is_mla() const { return kv_lora_rank > 0; }`.

- [ ] **Step 1: Write the failing test** — load DeepSeek-V2-Lite `config.json` through `load_hf_config` and assert the MLA fields.

```cpp
// tests/test_mla.cu
TEST(MLAConfig, ParsesDeepSeekV2LiteFields) {
  ModelConfig cfg = load_hf_config_from_path(
      "/home/kekz/models/DeepSeek-V2-Lite/config.json");
  EXPECT_EQ(cfg.arch, ModelArch::DEEPSEEK);
  EXPECT_TRUE(cfg.is_mla());
  EXPECT_EQ(cfg.kv_lora_rank, 512);
  EXPECT_EQ(cfg.q_lora_rank, 0);          // null in V2-Lite
  EXPECT_EQ(cfg.qk_rope_head_dim, 64);
  EXPECT_EQ(cfg.qk_nope_head_dim, 128);
  EXPECT_EQ(cfg.v_head_dim, 128);
  EXPECT_EQ(cfg.head_dim, 192);           // nope+rope
  EXPECT_NEAR(cfg.mla_mscale, 0.707f * /*yarn-adjusted*/ 1.0f, 1e-3);
  EXPECT_EQ(cfg.n_experts_shared, 2);
  EXPECT_EQ(cfg.first_k_dense_replace, 1);
}
```

- [ ] **Step 2: Run test, verify it fails** — `make test-unit GTEST_FILTER='MLAConfig.*'`. Expected: FAIL (fields don't exist / are zero).

- [ ] **Step 3: Add the fields** to `ModelConfig` in `model_config.h` (near `rope_dim`, line ~49):

```cpp
// MLA (DeepSeek-V2/V3). kv_lora_rank>0 selects the MLA path.
int   kv_lora_rank   = 0;   // 512 (V2-Lite) / 1024 (V3)
int   q_lora_rank    = 0;   // 0 = full Q projection (V2-Lite); >0 = Q down/up LoRA (full V2/V3)
int   qk_rope_head_dim = 0; // 64  (decoupled RoPE key dims)
int   qk_nope_head_dim = 0; // 128 (non-RoPE key dims)
int   v_head_dim       = 0; // 128 (value head dim; may differ from qk head dim)
float mla_mscale       = 1.0f; // YaRN attention-scale multiplier (see Task 2.5)
bool  is_mla() const { return kv_lora_rank > 0; }
int   first_k_dense_replace = 0; // layers [0,k) are dense FFN even in a MoE model
```

- [ ] **Step 4: Parse them** in `hf_config_loader.cpp`, replacing the warning block at 610–626:

```cpp
jobj_get_int(eff, "kv_lora_rank",     cfg.kv_lora_rank);
jobj_get_int(eff, "q_lora_rank",      cfg.q_lora_rank);     // absent/null -> stays 0
jobj_get_int(eff, "qk_rope_head_dim", cfg.qk_rope_head_dim);
jobj_get_int(eff, "qk_nope_head_dim", cfg.qk_nope_head_dim);
jobj_get_int(eff, "v_head_dim",       cfg.v_head_dim);
jobj_get_int(eff, "first_k_dense_replace", cfg.first_k_dense_replace);
if (cfg.is_mla()) {
  cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim; // 192
  cfg.rope_dim = cfg.qk_rope_head_dim;                        // decoupled RoPE subset
  // YaRN mscale: DeepSeek scales attention logits by mscale; see Task 2.5 for the
  // exact formula. Store raw mscale here; the yarn-adjusted value is computed there.
  double mscale = 1.0; jobj_get_double(eff, /*rope_scaling.*/ "mscale", mscale);
  cfg.mla_mscale = static_cast<float>(mscale);
}
// (delete the old "Inference will produce incorrect outputs" warning)
```

- [ ] **Step 5: Run test, verify pass** — `make test-unit GTEST_FILTER='MLAConfig.*'`. Expected: PASS. (`mla_mscale` exact value finalized in Task 2.5; for now assert raw `0.707`.)

- [ ] **Step 6: Commit** — `git commit -m "feat(mla): parse DeepSeek-V2 MLA config fields"`

### Task 0.2: ModelProfile AttnVariant::MLA

**Files:**
- Modify: `src/model/model_profile.h:50` (enum), profile derivation
- Test: `tests/test_mla.cu`

**Interfaces:**
- Consumes: `ModelConfig::is_mla()`.
- Produces: `ModelProfile::attn_variant == AttnVariant::MLA` for DeepSeek MLA models; `prof.is_mla` accessor.

- [ ] **Step 1: Failing test**

```cpp
TEST(MLAConfig, ProfileSelectsMLAVariant) {
  ModelConfig cfg = load_hf_config_from_path("/home/kekz/models/DeepSeek-V2-Lite/config.json");
  ModelProfile prof = derive_model_profile(cfg);
  EXPECT_EQ(prof.attn_variant, AttnVariant::MLA);
}
```

- [ ] **Step 2: Run, verify fail** — `make test-unit GTEST_FILTER='MLAConfig.ProfileSelectsMLAVariant'`. Expected: FAIL (no enumerator).

- [ ] **Step 3: Add enumerator** at `model_profile.h:50`: `enum class AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE, MLA };` and in `derive_model_profile()`: `if (cfg.is_mla()) prof.attn_variant = AttnVariant::MLA;` (before the STANDARD default).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): add AttnVariant::MLA profile trait"`

---

## Phase 1 — DeepSeek MoE loader (no kernels; ~loader only)

> The MoE engine already supports shared experts, `routed_scaling_factor`, per-layer dense/MoE, and a separate MoE intermediate size. This phase is purely DeepSeek tensor-name routing + the first-dense-layer wiring.

### Task 1.1: DeepSeek MoE weight-name mapping

**Files:**
- Modify: `src/model/weight_map.cpp` (name routing + assignment; mirror the Qwen3-MoE block)
- Test: `tests/test_mla.cu`

**Interfaces:**
- Consumes: HF names `model.layers.N.mlp.gate.weight` (router), `model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight`, `model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight`, and for layer 0 the dense `model.layers.0.mlp.{gate,up,down}_proj.weight`.
- Produces: populated `TransformerLayer` `moe_gate`, `expert_w_{gate,up,down}`, `w_{gate,up,down}_shared` for layers ≥1; dense `w_{gate,up,down}` for layer 0.

- [ ] **Step 1: Failing test** — assert that after loading, layer 0 is dense and layer 1 is MoE with 64 experts + 2 shared.

```cpp
TEST(MLALoader, DeepSeekMoELayerTypes) {
  Model m = load_model("/home/kekz/models/DeepSeek-V2-Lite"); // SafeTensors
  EXPECT_NE(m.layer(0).w_up.data, nullptr);       // dense FFN
  EXPECT_EQ(m.layer(0).moe_gate.data, nullptr);
  EXPECT_NE(m.layer(1).moe_gate.data, nullptr);   // MoE
  EXPECT_EQ(m.layer(1).expert_w_up.size(), 64u);
  EXPECT_NE(m.layer(1).w_up_shared.data, nullptr);
}
```

- [ ] **Step 2: Run, verify fail** — names unmatched ⇒ "missing tensor" or null layers.

- [ ] **Step 3: Add DeepSeek name routing** in `weight_map.cpp` next to the existing Qwen3-MoE matcher: route `mlp.gate.weight`→`moe_gate`, `mlp.experts.{e}.*`→packed expert slots, `mlp.shared_experts.*`→`*_shared`, and (only when `parts[2] < cfg.first_k_dense_replace`) `mlp.{gate,up,down}_proj`→dense `w_*`. Confirm `expert_weights_scale` is set from `routed_scaling_factor` (already parsed at `hf_config_loader.cpp:465`).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): DeepSeek-V2 MoE weight-name mapping (dense L0 + shared experts)"`

---

## Phase 2 — Materialized MLA attention (correctness-first, no new kernel)

> Reconstruct full K/V from the latent at projection time, hand standard-shaped K/V to the existing RoPE + paged attention. The only asymmetry is V-head-dim (128) ≠ QK-head-dim (192). No KV-cache change yet.

### Task 2.1: TransformerLayer MLA weight fields

**Files:**
- Modify: `src/model/model_config.h:168-324` (`TransformerLayer`)
- Modify: `src/core/tensor_kind.h` (new enums)

**Interfaces:**
- Produces: `TransformerLayer::kv_a_proj`, `kv_a_layernorm`, `kv_b_proj` (`Tensor`); `kv_a_proj_id`, `kv_a_norm_id`, `kv_b_proj_id` (`TensorID`, default `kInvalidTensorID`). `TensorKind::{KV_A_PROJ, KV_A_NORM, KV_B_PROJ}`.

- [ ] **Step 1: Add fields** to `TransformerLayer` (alongside `wq/wk/wv/wo`) and enumerators to `tensor_kind.h`. (No standalone test; covered by Task 2.2.)

- [ ] **Step 2: Build** — `make build`. Expected: compiles.

- [ ] **Step 3: Commit** — `git commit -m "feat(mla): TransformerLayer latent-projection fields"`

### Task 2.2: MLA attention tensor-name mapping + NVFP4 scale routing

**Files:**
- Modify: `src/model/tensor_kind_matcher.cpp:34-46`
- Modify: `src/model/weight_map.cpp` (name remap ~105-114, field assign ~510-550, NVFP4 scale ~696-706)
- Test: `tests/test_mla.cu`

**Interfaces:**
- Consumes: HF `model.layers.N.self_attn.{q_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}.weight`.
- Produces: populated `layer.wq`, `layer.kv_a_proj`, `layer.kv_a_layernorm`, `layer.kv_b_proj`, `layer.wo` with correct shapes.

- [ ] **Step 1: Failing test** — assert MLA tensors loaded with expected shapes (`kv_a_proj`: [2048→576], `kv_b_proj`: [512→16*256=4096], `wq`: [2048→16*192=3072]).

```cpp
TEST(MLALoader, AttentionProjectionShapes) {
  Model m = load_model("/home/kekz/models/DeepSeek-V2-Lite");
  const auto& l = m.layer(1);
  EXPECT_EQ(l.wq.shape[0], 3072);          // 16*192
  EXPECT_NE(l.kv_a_proj.data, nullptr);
  EXPECT_EQ(l.kv_a_proj.shape[0], 576);    // 512 + 64
  EXPECT_NE(l.kv_a_layernorm.data, nullptr);
  EXPECT_EQ(l.kv_b_proj.shape[0], 4096);   // 16*(128+128)
}
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Add matching** — in `tensor_kind_matcher.cpp` after line 46: `.kv_a_proj.`→`KV_A_PROJ`, `.kv_a_layernorm.`→`KV_A_NORM`, `.kv_b_proj.`→`KV_B_PROJ`. In `weight_map.cpp`: name remap `kv_a_proj_with_mqa`→`kv_a_proj` etc.; field assignment branches; extend NVFP4 scratch slot logic (kv_a_proj / kv_a_norm / kv_b_proj). Norm weights (`kv_a_layernorm`) stay FP16/FP32 — never quantized.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): tensor-name mapping + NVFP4 scale routing for latent projections"`

### Task 2.3: Materialized KV projection (host-orchestrated, reuse GEMV)

**Files:**
- Modify: `src/exec/executor_attention_qkv.cu` (new MLA branch, separate-GEMV fallback first)
- Test: `tests/test_mla.cu` (numeric check vs a CPU reference of the two-step projection)

**Interfaces:**
- Consumes: `layer.kv_a_proj`, `kv_a_layernorm`, `kv_b_proj`; `norm_out` (post-attn-RMSNorm activations); `cfg.{kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, n_heads}`.
- Produces: into the existing K buffer a `[n, n_heads, 192]` tensor (`k_nope`||`k_rope`) and into the V buffer a `[n, n_heads, 128]` tensor — exactly the layout the standard RoPE + KV-write expect (modulo Task 2.4's V-dim).

Projection math (materialized):
```
kv_a   = norm_out @ kv_a_proj^T            # [n, 576] = [n, 512 latent | 64 k_rope]
latent = kv_a[:, :512];  k_rope = kv_a[:, 512:]   # k_rope shared across all heads (MQA-style)
latent = rmsnorm(latent, kv_a_layernorm)   # [n, 512]
kv_b   = latent @ kv_b_proj^T              # [n, 4096] = [n, 16*(128 k_nope | 128 v)]
# reshape per head: k_nope[h]=kv_b[..,h,:128], v[h]=kv_b[..,h,128:]
# K[h] = concat(k_nope[h] (128), k_rope (64)) -> 192 ; RoPE applied to the k_rope 64 only
```

- [ ] **Step 1: Failing test** — feed a fixed `norm_out`, run the MLA projection path, compare `K`/`V` against a host (Eigen/manual) reference of the two-step math to `1e-2` (bf16 tolerance).

- [ ] **Step 2: Run, verify fail** (MLA branch not taken yet).

- [ ] **Step 3: Implement the separate-GEMV MLA branch** in `executor_attention_qkv.cu`, gated on `prof.attn_variant == AttnVariant::MLA`: Q via existing GEMV (output dim 3072); two GEMVs for `kv_a`→split→`rmsnorm` (reuse `rmsnorm()` kernel on the 512 slice)→`kv_b`; scatter `k_nope`+broadcast `k_rope`→K buffer, `v`→V buffer. Use existing GEMV dispatch (NVFP4/FP16) — no new kernel.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): materialized two-step KV projection (reuses paged attention)"`

### Task 2.4: Asymmetric V-head-dim in attention/KV path

**Files:**
- Modify: `src/exec/executor_attention.cu:70-97` (per-layer V head dim), KV-cache shape for V
- Test: `tests/test_mla.cu`

**Interfaces:**
- Consumes: `cfg.v_head_dim` (128) vs `cfg.head_dim` (192).
- Produces: attention that scores with QK head dim 192 and accumulates output with V head dim 128; `o_proj` input dim = `n_heads * v_head_dim` = 2048.

- [ ] **Step 1: Failing test** — single-layer attention forward with MLA dims produces output of width `16*128=2048` (feeds `o_proj`), not 3072.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Wire V-head-dim** — extend the per-layer head-dim queries (already present for Gemma-4 at `executor_attention.cu:70-77`) so the value path uses `v_head_dim`. The paged-decode/FA2 kernels already template/parametrize head_dim; verify the V accumulation width is taken from `v_head_dim`. If a kernel hardcodes QK==V dim, add the split (this is the one place a kernel touch may be needed; keep it dynamic, no new template instantiations).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): asymmetric QK(192)/V(128) head dims in attention"`

### Task 2.5: YaRN mscale attention scaling

**Files:**
- Modify: `src/exec/executor_attention.cu:445-449`; finalize `mla_mscale` in `hf_config_loader.cpp`
- Test: `tests/test_mla.cu`

**Interfaces:**
- Consumes: `cfg.mla_mscale`, YaRN factor.
- Produces: attention scale = `(1/sqrt(qk_head_dim)) * mscale_adj^2`, where `mscale_adj = 0.1 * mscale * ln(factor) + 1.0` (DeepSeek YaRN formula; for `mscale=0.707, factor=40` ⇒ `mscale_adj ≈ 1.0 + 0.0707*ln(40) ≈ 1.261`, scale multiplier ≈ 1.59).

- [ ] **Step 1: Failing test** — assert the computed scale multiplier matches the DeepSeek reference value to `1e-3`.

```cpp
TEST(MLAConfig, YarnMscaleAttentionScale) {
  ModelConfig cfg = load_hf_config_from_path("/home/kekz/models/DeepSeek-V2-Lite/config.json");
  float mult = mla_attention_scale_multiplier(cfg); // new helper
  EXPECT_NEAR(mult, 1.261f * 1.261f, 1e-2); // mscale_adj^2
}
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the helper (compute `mscale_adj` from `mla_mscale` + YaRN factor) and apply `scale *= mscale_adj*mscale_adj` after `executor_attention.cu:449` when `prof.attn_variant == AttnVariant::MLA`. Cross-check the formula against the HF `DeepseekV2` modeling code (`yarn_get_mscale`).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(mla): YaRN mscale attention-logit scaling"`

### Task 2.6: End-to-end verification gate (Stage A done)

**Files:** none (verification + docs)

- [ ] **Step 1: Loads + coherent** — `imp-cli` greedy generate on 5 prompts; run the `check-degeneration` battery (`tools/analysis/degen_suite.py`). Expected: no repetition loops, coherent prose.

- [ ] **Step 2: Perplexity vs HF** — run `imp-cli --perplexity` on a fixed corpus; run the HF `DeepSeek-V2-Lite` reference (CPU or device_map; the model is bf16 ~31 GB so run imp and HF in separate processes) over the *same* span; compare summed NLL. **Gate: within ~10–20 % of HF** (per add-model-arch; usually much closer). Record both numbers.

- [ ] **Step 3: If PPL fails**, triage with the add-model-arch fingerprint table (prompt-blind ⇒ RoPE pair layout; digits scrambled ⇒ decoupled-rope dim mismatch; argmax token 0 ⇒ NaN/scale). Fix, re-run.

- [ ] **Step 4: Docs + commit** — add the DeepSeek-V2-Lite row to `docs/supported-models.md` (note: Stage A = correct output, no KV compression yet). `git commit -m "docs(mla): DeepSeek-V2-Lite supported (materialized MLA)"`

- [ ] **Step 5: Open PR** — `gh pr create --base main` titled `feat(mla): DeepSeek-V2 MLA support (materialized)`. Body: what works (correct output, validated PPL), what's deferred (latent KV compression = Phase 3). This is a shippable, verified increment.

---

## Phase 3 — Absorbed / latent-cache MLA (the VRAM win) — start only after Phase 2 PR merges

> Goal: store only `[latent(512) | k_rope(64)] = 576` per token in the KV cache instead of `16*192 + 16*128 = 5120` ⇒ ~9× KV reduction (the spec's "64×" is vs uncompressed MHA; vs imp's GQA baseline the realized factor is model-dependent — measure it). This needs a new paged-latent attention kernel and a KV-cache latent mode.

### Task 3.1: Latent-mode KV cache layout

**Files:**
- Modify: `src/memory/kv_cache.{h,cu}` (latent block bytes = `(kv_lora_rank + qk_rope_head_dim) * dtype` per token; single unified cache, no K/V split)
- Test: `tests/test_mla.cu`

- [ ] **Step 1: Failing test** — construct a latent-mode `KVCache`, assert `block_bytes == 16 * 576 * 2` and that `latent_ptr(layer, block)` is valid while `k_ptr/v_ptr` are unused.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** a latent cache mode (reuse the per-layer `layer_block_bytes_` machinery from the Gemma-4 path; add `latent_ptr()`; gate on `prof.attn_variant == MLA`).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit.**

### Task 3.2: Q-absorption + paged-latent attention kernel

**Files:**
- Create: `src/compute/attention_mla.cu` + `.h`
- Modify: `src/exec/executor_attention.cu` (dispatch MLA decode/prefill to the new kernel)
- Test: `tests/test_mla.cu`

Design: absorb `W_UK` (the k_nope half of `kv_b_proj`) into Q so scores are `Q_absorbed @ latent^T + Q_rope @ k_rope^T`; output is `softmax(scores) @ (latent @ W_UV)`. Decoupled RoPE applied to the 64-dim `q_rope`/`k_rope` only. Dynamic head_dim loops (no template bloat — latent_dim 512 won't fit the FA2 hd=128 template).

- [ ] **Step 1: Failing equivalence test** — for a fixed input, the latent-cache kernel output must match the **Phase 2 materialized** path to `1e-2`. (Phase 2 is the oracle — this is why correctness-first ordering matters.)
- [ ] **Step 2: Run, verify fail** (kernel absent).
- [ ] **Step 3: Implement** the absorbed kernel (decode first, then prefill). Validate the absorption math on CPU before the kernel.
- [ ] **Step 4: Run, verify pass** + re-run the Phase 2.6 PPL gate (must stay within tolerance) + `check-degeneration`.
- [ ] **Step 5: Commit.**

### Task 3.3: VRAM + perf measurement, baseline

**Files:** `docs/supported-models.md`, possibly `tests/perf_baseline.json`

- [ ] **Step 1: Measure** KV-cache VRAM at a long context (e.g. 32k) materialized vs latent; record decode tok/s (warm clocks, methodology per benchmark-cuda). 
- [ ] **Step 2: Document** the realized KV reduction + any decode delta. If DeepSeek-V2-Lite becomes a gated model, refresh the baseline via `scripts/gen_perf_baseline.sh` and say so in the PR.
- [ ] **Step 3: PR** — `feat(mla): latent-cache absorbed attention (KV compression)`.

---

## Self-Review

**Spec coverage** (vs `docs/superpowers/specs/2026-05-28-mla-deepseeek-architecture-design.md`): model loading → Phase 1 + Tasks 2.1/2.2; KV cache latent layout → Task 3.1; attention kernel → Task 3.2; decoupled RoPE → reused (`rope.cu`, wired in 2.3) ✓; arch enums → reuse existing `ModelArch::DEEPSEEK` (no new enum needed — corrected from spec). Spec's `W_UK`/`W_UV` absorption → Task 3.2. Spec gating conditions (model available, fits 32 GB, reference engine) → satisfied by DeepSeek-V2-Lite + HF reference. **Added beyond spec:** DeepSeek MoE loader (Phase 1) and YaRN mscale (2.5) — the spec omitted that DeepSeek-V2 is also a shared-expert MoE with mscale, both required for correct output.

**Placeholder scan:** No "TBD"/"handle edge cases"/"add error handling" steps. The one genuinely uncertain item — exact CUDA of the Phase 3 absorbed kernel — is deliberately scoped as design+equivalence-test rather than fabricated final code, because it must be validated against the Phase 2 oracle, not guessed.

**Type consistency:** `is_mla()`, `kv_lora_rank`, `mla_mscale`, `AttnVariant::MLA`, `TensorKind::KV_*`, `kv_a_proj/kv_a_layernorm/kv_b_proj`, `mla_attention_scale_multiplier()` used consistently across tasks.

**Note on q_lora_rank:** V2-Lite has `q_lora_rank=null` (full Q). Full DeepSeek-V2/V3 use Q-LoRA (down/up). This plan handles the V2-Lite (full-Q) case; the `q_lora_rank>0` branch is a small future extension (add `q_a_proj`/`q_a_layernorm`/`q_b_proj` mirroring the KV path) — out of scope until a full-V3 model that fits 32 GB exists.

---

## Execution Handoff

Stage A (Phases 0–2) is the committed deliverable: a correct, HF-validated DeepSeek-V2-Lite. Stage B (Phase 3) is the VRAM win and is gated on Stage A merging.

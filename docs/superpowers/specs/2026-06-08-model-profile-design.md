# ModelProfile — one place for architecture-derived facts (D1, design)

Date: 2026-06-08
Status: approved → implementation
Audit basis: `docs/audit/structural_debt_2026_06_08.md` (D1).

## Problem

Architecture-derived facts are recomputed inline across the codebase instead of
decided once and read. This is the scattered-`if(arch==…)` class that caused
#514/#516 (a path forgets a carve-out) and that the rejected FP16-gate diff hit.
Verified instances:
- **GDN/SSM-hybrid detection** recomputed at ≥6 sites (loop layers, check
  `gdn_gate.data`/`ssm_in.data != nullptr`): `engine_init_resolver.cpp:128/198`,
  `engine_kv_cache_init.cpp:155/174/210`, `vram_budget.cpp:60`,
  `engine_weight_upload.cpp:142`.
- **Attention variant** branched inline 8+ times in `executor_attention.cu`
  (`cfg.arch == GEMMA4/GPT_OSS`): SWA routing (565/576/613), qk-norm (202),
  V=K (535/543), fp32-accum (367/445).
- **FFN-norm selection** (per-layer chain) repeated in every MoE/FFN site, e.g.
  `executor_forward_moe.cu:163`.

`ModelConfig` holds static metadata (arch, rope params, n_experts, swa_layers)
plus a few derived flags (is_nvfp4_prequant, gdn_grouped_head_layout). The core
classification (is_gdn/is_moe/is_hybrid, attn/ffn variant) is the missing piece.

## Goal & constraints

- **Strictly behaviour-neutral**: each migrated call-site computes the same value
  it did inline. Verified per step.
- **One decision, then read**: derive once at init, never recompute downstream.
- **Gated**: per-step, behind the arch-coverage canaries below.
- Branch off `main` (no stacking on the VRAM PR #621).

## Architecture — two levels

### Level 1: global `ModelProfile` (new, src/model/model_profile.h)

```cpp
struct ModelProfile {
    // classification
    bool is_moe = false;     // n_experts > 0
    bool is_gdn = false;     // any layer has gdn_gate
    bool is_ssm = false;     // any layer has ssm_in
    bool is_hybrid = false;  // recurrent (gdn/ssm) AND attention layers coexist
    bool is_dense = false;   // !is_moe

    // attention variant + flags (drives executor_attention dispatch)
    enum class AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE };
    AttnVariant attn_variant = AttnVariant::STANDARD;
    bool attn_qk_norm = false;            // gemma-4 per-head q/k RMSNorm
    bool attn_v_eq_k = false;             // gemma-4 V=K layers (wv absent)
    bool attn_fp32_accum_gemma4 = false;  // gemma-4 fp32 attention accumulation

    // eligibility (centralizes what engine_init_resolver already decides)
    bool fp8_eligible = false;
    bool graphs_eligible = false;
};

// Pure, no side effects, no allocation. Reads Model layers + ModelConfig once.
ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg);
```

Held by `Model` (alongside `ModelConfig`), exposed via `model.profile()`. Filled
once at model-load / engine-init, before any forward pass.

### Level 2: per-layer resolution (no new state)

The ffn-norm chain is a *per-layer* tensor choice, not a global fact. Centralize
the chain in one helper instead of a global field:

```cpp
// the (gemma4 → ffn_pre_norm_2 : ffn_norm : post_attn_norm : attn_norm) chain,
// in one place
const Tensor& effective_ffn_norm(const TransformerLayer& L, const ModelProfile& p);
```

The tensors already exist on `TransformerLayer`; only the selection logic moves.

## Migration order (incremental, each its own commit, each gated)

1. **Plumbing**: add `ModelProfile` + `derive_model_profile` + `Model::profile()`,
   filled at init. Add a one-time debug log of the derived profile. **Zero
   behaviour change** — nothing reads it yet.
2. **C — GDN/SSM classification** (most isolated, real bug history): the ≥6
   detection loops → `profile.is_gdn` / `is_ssm` / `is_hybrid`.
3. **B — attention variant**: the 8+ inline arch checks in executor_attention →
   `profile.attn_variant` + flags. (Biggest single-file win.)
4. **A — ffn-norm helper**: the repeated chain → `effective_ffn_norm()`.
5. **Eligibility**: fold the `fp8_eligible`/`graphs_eligible` decisions into the
   profile (touches engine_init_resolver — do last, after VRAM PR #621 settles).

## Verification (per step)

The critical axis is ARCHITECTURE COVERAGE — behaviour must not shift for any
arch. Canaries:
- **gemma-4** (Gemma-4-26B-A4B-NVFP4) — the densest arch-carve-out user.
- **GDN hybrid** — Nemotron-3-Nano-30B-NVFP4 + Qwen3.6-35B-A3B (GGUF + NVFP4).
- **gpt-oss** (gpt-oss-20b) — SWA variant.
- **dense** — Qwen3-8B Q8_0.
Each: coherent output (the check-degeneration battery), no IMA/NaN, plus
`make verify-fast`. Strictly behaviour-neutral; greedy output should be
unchanged for deterministic models.

## Out of scope

- D2 (GraphExecutor split), D3 (god-functions), D4 (flag sweep) — separate.
- Any change to how the variants BEHAVE — only WHERE the decision is made.

## Risks

- **A missed carve-out flips behaviour for one arch.** Mitigation: per-step
  migration + the four-arch canary set; derive_model_profile reproduces each
  inline expression exactly (diff the old expression against the new field at
  each call-site during migration).
- **Hybrid edge cases** (a model that is both GDN and MoE, e.g. Qwen3.6-35B):
  `is_hybrid` must be the coexistence test, not either-or. Covered by the
  Qwen3.6 canary.

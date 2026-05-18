# Chunked Prefill — Hybrid GDN+MoE / Mamba2+MoE Arch Support

**Status**: design accepted, implementation pending
**Roadmap link**: `docs/roadmap.md` known-limitation "Chunked prefill scope (full-attention + FP16/FP8 KV)" — out-of-scope item "Hybrid models with non-attention layers (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)" (L28)
**Predecessor**: PR #149 (chunked prefill on NVFP4 KV + correct attn_scores guard) and `2026-05-08-paged-prefill-kernel-design.md` (chunked attention past-KV read for full-attention)

## Problem

Hybrid models (`QWEN35`, `QWEN35_MOE`, `QWEN36_MOE`, `NEMOTRON_H_MOE`) are blanket-excluded from chunked prefill at `src/runtime/engine.cpp:1697-1700`. Long prompts (`prompt_len > effective_chunk`) on these archs are rejected with `RequestStatus::CANCELLED` at `engine.cpp:1772`.

Practical impact: Qwen3.5/3.6 GDN and Nemotron-H Mamba2 cannot ingest prompts beyond `executor_->max_tokens()` (256 for SSM/MoE hybrids, 512 for dense GDN). Server users hit the cancellation; CLI users hit it on any document of moderate length.

Roadmap framing said the fix needs a *"per-layer-shape-aware paged-prefill kernel"*. Code reading shows that framing was Gemma-4 specific (dual head_dim 256/512). For the four hybrid archs in scope:

- All attention layers within a given hybrid model share one `(nkv, hd)` geometry — no `n_kv_heads_per_layer` / `head_dim_per_layer` overrides
- `executor_attention.cu`'s existing chunked path (post PR #149) supports them iff the carve-out is removed
- The SSM/GDN/conv kernels in `src/compute/` are mostly chunk-safe today (state buffers persist across calls), with one exception (Mamba2 plain conv kernel)

So the actual gap is small: one CUDA-kernel patch + carve-out removal in the engine.

## Scope

**In scope:**

- `QWEN35` (Qwen3.5 GDN dense — 4B, 9B)
- `QWEN35_MOE` (Qwen3.5 GDN+MoE)
- `QWEN36_MOE` (Qwen3.6 35B-A3B GDN+MoE; both Q4_K_M and prequant NVFP4 variants)
- `NEMOTRON_H_MOE` (Nemotron-3-Nano-30B-A3B Mamba2+MoE; both Q*_K and NVFP4 variants)
- KV dtypes: FP16, FP8_E4M3, NVFP4 — same set already supported for full-attention chunked prefill
- Default `prefill_chunk_size` enabled (resolves to `512` for hybrids, clamped to executor `max_tokens` of 256/512)

**Out of scope (separate work items):**

- Gemma-3 / Gemma-4 — SWA + (Gemma-4) dual head_dim. Different problem: SWA-aware mask in chunked attention.
- Llama-4 — MoE + SWA, untested.
- Sub-byte KV gather kernels other than NVFP4 (INT4, TurboQuant, TurboQuant Lite).
- The full-attention `per_layer_shapes=true` branch — only Gemma-4 hits this.

## Approach

Two-part change. Each is independently small.

### Part 1 — Mamba2 conv kernel chunked-prefill fix

`src/compute/ssm.cu:196-233` `ssm_conv1d_prefill_kernel` (used by Nemotron-H Mamba2 path at `executor_ssm_gdn.cu:128`) zero-pads input when `src_t < 0`:

```cuda
for (int k = 0; k < kernel_size; k++) {
    int src_t = token - (kernel_size - 1) + k;
    float val = 0.0f;
    if (src_t >= 0) {
        val = __half2float(x_in[src_t * channels + ch]);
    }
    sum += val * __half2float(weight[ch * kernel_size + k]);
}
```

For chunk N>0 this loses the last `kernel_size-1` tokens of the previous chunk. The fused `ssm_conv1d_prefill_f32_silu_kernel` (used by GDN path) already has the correct logic at `ssm.cu:267-273`:

```cuda
} else if (conv_state) {
    int state_idx = src_t + kernel_size;  // maps to [1..K-1]
    val = (state_idx >= 0 && state_idx < kernel_size)
        ? conv_state[ch * kernel_size + state_idx] : 0.0f;
}
```

Port this branch into `ssm_conv1d_prefill_kernel`. The kernel's last-token state-write logic (`ssm.cu:225-231`) is correct as-is — it writes the last `kernel_size` tokens of the current chunk back into `conv_state`, which becomes the trailing context for chunk N+1.

### Part 2 — Engine carve-out removal

`src/runtime/engine.cpp:1689-1709` `Engine::supports_chunked_prefill_()`:

- Drop returns for `QWEN35`, `QWEN35_MOE`, `QWEN36_MOE`, `NEMOTRON_H_MOE`.
- Keep `GEMMA3`, `GEMMA4`, `LLAMA4` returns (out of scope).
- Keep KV-dtype check (FP16 / FP8 / NVFP4 only).

`src/runtime/engine.cpp:1761+` `Engine::step_prefill_one`:

- The cancellation block at `engine.cpp:1772-1779` becomes dead code for hybrid archs (since `supports_chunked_prefill_()` now returns true). It still guards Gemma-3/4/Llama-4 — unchanged.
- The `attn_scores_` capacity clamp at `engine.cpp:1786-1796` already correctly handles hybrid models (s_cap drives the effective_chunk).
- The `effective_chunk = min(effective_chunk, executor_->max_tokens())` clamp at `engine.cpp:1742` already keeps per-chunk size within the SSM workspace bound (256 for SSM/MoE hybrids, 512 for dense GDN).

### Why no per-layer-shape work is needed

`executor_attention.cu:699` guards on `per_layer_shapes`. None of the four in-scope hybrid archs populate `n_kv_heads_per_layer` / `head_dim_per_layer` — verified by grep:

```bash
grep -n "n_kv_heads_per_layer\|head_dim_per_layer" src/model/*.cpp
```

Only `gemma4_loader.cpp` writes per-layer shape arrays. Hybrid models route attention layers through the standard `nh / nkv / hd` triplet.

The `kv_layer_map_` (engine.cpp:985+) handles non-attention layer indices: `get_kv_layer(kv_layer_map_, layer)` returns `-1` for GDN/Mamba2 layers, so `write_kv_cache` skips them. This is already in place for non-chunked prefill and continues to work in chunked mode.

## Components

### Files modified

| File | Change |
|---|---|
| `src/compute/ssm.cu:196-233` | Add `conv_state` read branch for `src_t < 0` to `ssm_conv1d_prefill_kernel` (~10 LoC) |
| `src/runtime/engine.cpp:1689-1709` | Drop four hybrid arch returns from `supports_chunked_prefill_()` (~4 LoC removed) |
| `tests/test_ssm_chunked.cu` | New: chunked vs single-chunk equivalence test for Mamba2 conv (small fixture) |
| `tests/test_engine_chunked_dispatch.cu` (new or appended to existing) | Asserts `supports_chunked_prefill_()` is true for hybrid archs |
| `tests/perf_baseline.json` | Refresh hybrid model baselines (Qwen3.5-4B, Qwen3.5-9B, Qwen3.6 if available) — chunked may be slightly slower than single-chunk pp for prompts that fit in one chunk; add long-prompt perf entry |
| `docs/roadmap.md:28` | Update "Hybrid models" out-of-scope line — move to "shipped" or remove from carve-out list |

### Existing components — no change required

- `src/compute/ssm.cu:250+` `ssm_conv1d_prefill_f32_silu_kernel` — already chunked-correct (used by GDN)
- `src/compute/ssm.cu:333+` `ssm_scan_kernel` — reads h_state at start of each token, writes back; chunked-natural
- `src/compute/gdn.cu:30+` `gdn_scan_fused_kernel` — loads h_state into registers at start, stores back at end; chunked-natural
- `src/graph/executor_attention.cu:692-781` chunked prefill path — already supports kvt_ok && !sliding && !per_layer_shapes (which all four hybrid archs satisfy)
- `src/graph/executor_kv_write.cu` — uses `state.positions` (offset-aware via prefill_offset)
- `attn_scores_` capacity guard (PR #149) — already correctly checks `n × ctx_len ≤ s_cap²`

## Testing

### Unit / kernel correctness

- **`test_ssm_chunked.cu`** — pure kernel test:
  1. Generate random `[N, channels]` input where N is large (e.g., 768).
  2. Run `ssm_conv1d_prefill_kernel` once on the full input. Capture output and final conv_state.
  3. Run the same kernel in three chunks of 256 tokens each, threading `conv_state` between calls.
  4. Compare outputs element-wise (FP16 abs tol ≤ 1e-3) and final conv_state.
  5. Repeat for `ssm_conv1d_prefill_f32_silu_kernel` as a regression-detection witness (it should already pass).

### Engine-level smoke

- Add a non-fixture-dependent gtest that constructs a `ModelConfig` with `arch=QWEN35` / `QWEN35_MOE` / `QWEN36_MOE` / `NEMOTRON_H_MOE` (no weight load) and verifies `Engine::supports_chunked_prefill_()` returns true and `resolve_prefill_chunk_size_()` returns >0 with default config. Per-chunk loop behavior is validated by the end-to-end run on real models below — synthetic engine-level prefill_one fixtures don't exist for hybrid archs and are out of scope to add here.

### End-to-end on real models

- **`scripts/validate_safetensors.py`** — re-run the existing battery on:
  - Qwen3.5-4B-Q8_0.gguf (existing baseline)
  - Qwen3.5-9B-Q8_0.gguf (existing baseline)
  - Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  - Nemotron-3-Nano-30B-A3B-NVFP4 (if present locally)
- Specifically: `long_context_recall` prompt and any prompt > 256 tokens must complete coherent (was previously CANCELLED or hit `n_tokens exceeds max_tokens` abort).

### Performance gate

- **`scripts/gen_perf_baseline.sh`** then `make verify`:
  - Existing tg256/pp512 numbers for Qwen3.5-4B-Q8_0 / Qwen3.5-9B-Q8_0 must hold within the existing 3% decode / 5% prefill thresholds. Long prompts (pp >256) get a new non-regression-gated baseline number entered fresh.
  - Spot-check: pp4096 on Qwen3.5-9B-Q8_0 (was CANCELLED) should now produce a sensible tok/s number and coherent output.

### Risks I'm not pre-validating

- SSM-state precision drift across many chunks: the H-register state is FP32 inside the kernel and persists as FP32 or FP16 in `h_state` (per `ssm_state->h_dtype()`). Chunk boundaries don't introduce additional precision loss vs single-shot prefill of the same length, since the persisted state was already going to be written/read at chunk boundaries during the regular forward pass. Coherence test on long prompts will surface any anomaly.
- Per-chunk launch overhead on Mamba2/GDN scan kernels: not measured. Chunked prefill on a 4K prompt with chunk=256 = 16 launches per layer per chunk (conv + scan + ssm_in/ssm_out projections + group_rmsnorm). Acceptable cost vs cancellation.

## Risks

| Risk | Mitigation |
|---|---|
| SSM state silently drifts across chunk boundaries on long prompts | `validate_safetensors.py long_context_recall` catches incoherence ≥768 tokens (existing harness). |
| Mamba2 conv `conv_state` read at chunk boundary writes the wrong stride | Unit test directly compares full-prefill vs chunked-prefill output element-wise. |
| Decode regression on hybrid models due to default chunk=512 changing prefill behavior | Decode is unaffected (decode runs after prefill is complete; chunk size only affects prefill ingest path). Existing `tg256` perf gate catches any regression. |
| Per-chunk launch overhead noticeably regresses pp on prompts that DO fit in one chunk | Short prompts (`prompt_len ≤ effective_chunk`) still complete in a single iteration of the `step_prefill_one` chunk loop — chunking only triggers when `total_input > effective_chunk`. Existing `pp512` baselines verify no regression on small prompts. |
| Nemotron-H NVFP4 specifically fails the SSM in/out_proj NVFP4 fast-path under chunked load | Already routed through CUTLASS NVFP4 cache by `5b2c5db`. No new code path. |

## Out of scope — explicit follow-ups

These are tracked separately and NOT addressed by this PR:

1. Gemma-3/Gemma-4 SWA-aware chunked prefill (different problem class)
2. Llama-4 MoE+SWA chunked prefill
3. Per-chunk perf optimization for hybrid (e.g., persistent chunk-aware kernel scheduling) — current scope is correctness-only, defaults conservatively
4. INT4 / TurboQuant / TurboQuant-Lite KV gather kernels for chunked attention path (not specific to hybrid)

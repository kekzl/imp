# imp — Architecture

This document is the canonical narrative companion to [`architecture.svg`](architecture.svg).
The SVG shows the structural overview; this file explains each phase in
prose and points at the source files that implement it.

If the code and this document disagree, the code wins — but a disagreement
is a bug in this document and should be fixed.

## At a glance

imp runs LLM inference end-to-end in four phases:

1. **Load** — read a GGUF file or a Hugging-Face SafeTensors directory into a
   `Model` object with a `WeightMap`, a `Tokenizer`, and a `ModelConfig`.
2. **Engine init** — resolve runtime config, upload weights to VRAM, build
   the paged KV cache, allocate workspaces, capture CUDA graphs for decode.
3. **Prefill** — run the prompt through the per-layer forward pass (chunked
   if the architecture supports it), producing the first-token logits.
4. **Decode** — replay the captured CUDA graph per token: attention → FFN →
   LM head → penalties → sampler → stop check, looping until EOS or limit.

See [`architecture.svg`](architecture.svg) for the full graph including
the attention dispatcher, memory subsystem, and kernel subsystem.

## Phase 1 — Load (one-time, `src/model/`)

Entry: `imp_model_load(path) → ImpModel` (`include/imp/imp.h`, dispatched
via `src/api/imp_api.cpp`).

Format detection inspects the path: a `.gguf` file routes to
`src/model/gguf_loader.cpp`; a directory containing `config.json` and
`*.safetensors` routes to `src/model/safetensors_loader.cpp` with optional
LLM-Compressor recipe handling in `src/model/llm_compressor_loader.cpp`.

Both loaders produce:

- A **WeightMap** (`src/model/weight_map.cpp`) — tensor name → role.
- A **Tokenizer** (`src/model/tokenizer.cpp` + `chat_template.cpp` +
  `jinja.cpp` + optional `sentencepiece_loader.cpp`).
- A **ModelConfig** + **Model** object (`src/model/model.cpp`,
  `src/model/model_arch.h`).

## Phase 2 — Engine init (one-time, `src/runtime/engine.cpp`)

Entry: `imp_context_create(model, ImpConfig) → ImpContext`.

`Engine::init()` orchestrates the init pipeline. The major steps are
distinct private methods on `Engine`:

| Step | Method | Notes |
|---|---|---|
| Load runtime config | `RuntimeConfig::load()` | `imp.conf` + `--config` CLI + legacy env-var seeds (`src/runtime/config.cpp`) |
| Resolve quant/KV/SSM dtypes | `init_resolve_*` group | `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_` |
| Compute max sequence length | `init_compute_max_seq_len_` | VRAM budget → max context (`src/runtime/vram_budget.cpp`) |
| Upload weights | `init_weights` | `upload_weight` + `upload_expert_weights` in `src/model/weight_upload.cu`; pre-dequant orchestrated by `src/exec/executor_pre_dequant.cu` (calls per-phase TUs `pre_dequant_phase*.cu`: `pre_dequant_phase0_nvfp4_loader.cu`, `pre_dequant_phase1_fp16_cache.cu`, `pre_dequant_phase2_fp8_cache.cu`, `pre_dequant_phase3_nvfp4_decode.cu`, `pre_dequant_phase3c_mxfp4.cu`, `pre_dequant_phase4_tensor_registry.cu`, `phase4b` drop-source + VRAM reclamation, `phase4c` second-pass FP8 using reclaimed VRAM) |
| Init KV cache | `init_kv_cache` | Paged blocks (block_size=16); dtype is FP16 / FP8 / INT8 / INT4 / NVFP4 / MXFP4 |
| Allocate workspaces | `init_features` | MMVQ scratch, cuBLAS S-matrix (~384 MiB default `attention.attn_scores_mib` — see Known limitations), FP8 activation scratch, split-K attn scratch |
| Warm up | `warmup()` | Captures CUDA graph for decode (`src/runtime/cuda_graph.cu`) |

The Engine façade (`engine.cpp`, ~570 LOC) delegates to 6 per-subsystem
TUs by concern (resolver, weight upload, KV cache, workspaces, scheduler,
sampling/stop).

## Phase 3 — Prefill (per request, `Engine::step_prefill`)

Entry: `imp_prefill_with_params(tokens, n) → status`.

Per-chunk loop in `src/exec/executor_forward.cu` (or
`executor_forward_moe.cu` for MoE architectures). Each layer runs:

```
RMSNorm → QKV GEMM + RoPE + KV-cache write → Attention → O proj →
RMSNorm + residual → FFN (dense SwiGLU or MoE top-k grouped GEMM)
```

After the last layer of the last chunk: final RMSNorm + LM head → logits.

### Attention dispatcher (the central choice)

`executor_attention_prefill.cu` decides which attention kernel to call. Since #687
the prefill gate is **FA2-first**:

```
const bool force_cublas_attn = per_layer_shapes || attn_sinks != nullptr;
const bool s_matrix_fits      = attn_scores_buf_ != nullptr && n <= attn_scores_.shape[1];
const bool prefer_fmha        = !force_cublas_attn && n >= fmha_prefill_threshold;

if (!force_cublas_attn && try_fa2_fp16qk_prefill(...))      // hd==128: O(n) memory, no S-matrix
    /* handled by FA2 f16-QK */;
else if (s_matrix_fits && !prefer_fmha)
    attention_cublas_prefill(...);                          // legacy materialized fallback
else
    attention_prefill_dispatch(...);                        // per-dtype FMHA family
```

The FP16-QK FA2 kernel is the primary path for hd=128 and — since #930/#932
(`attention.fa2_hd256`, default on) — hd=256 at every length (at-or-above the
materialized cuBLAS path: ~parity pp512, +24% pp1024, +52% pp2048) and needs no
S-matrix. `attention_cublas_prefill` (cuBLAS QK^T → ~384 MiB S-matrix → causal
softmax → cuBLAS PV) stays the fallback for the configs FA2 declines — `hd ∉
{128, 256}`, hd=256 with `fa2_hd256=false`, and `force_cublas_attn` (learned
sinks / truly heterogeneous per-layer shapes; uniform GDN/Mamba2-hybrid shapes
are FA2-servable since #932).
Everything else falls through to `attention_prefill_dispatch`, which selects among
the per-dtype FMHA kernels. Full coverage matrix: [`attention-dispatch.md`](attention-dispatch.md).

`force_cublas_attn` is set per-layer for Gemma-4 hd=512 global layers
where FMHA OOMs the 100 KiB smem cap.

Decode attention uses a separate switch on `cache_dtype` further down
in the same file, dispatching to one of the paged kernels
(INT4 / NVFP4 ± TC / MXFP4-KV / INT8 / FP8 / FP16-paged).

## Phase 4 — Decode loop (per token, `Engine::step_decode_forward`)

Entry: `imp_decode_step(params) → next_token` (or the streaming wrapper
`imp_generate_streaming`).

Per token:

1. **Replay** the captured CUDA graph (`src/runtime/cuda_graph.cu`).
   Graph capture is enabled for most architectures; non-Gemma-4 MoE
   disables it because of host-side routing.
2. **Paged attention decode** — kernel chosen by KV dtype.
3. **FFN GEMV** — dp4a / mma.sync / NVFP4 variants in
   `executor_ffn.cu`.
4. **LM head GEMV** → logits.
5. **Apply penalties** (repeat / freq / presence / DRY) —
   `src/compute/sampling.cu` (kernels) called from `src/exec/executor.cu`.
   Parameters are declared in `src/runtime/request.h`.
6. **Sampler** (temp / top-p / top-k / min-p / typical / mirostat) —
   `src/compute/sampling.{h,cu}` (`sample_greedy`, `sample_topk_topp`,
   `sample_mirostat_v2`, `apply_typical_p`).
7. **Stop check** — EOS, max_tokens, stop strings.
8. **(Optional) speculative decoding** — batch-1 greedy requests verify
   drafts as teacher-forced continuation chunks
   (`src/runtime/engine_spec_ngram.cpp`). Draft sources: the suffix index /
   n-gram matcher (`src/runtime/suffix_draft.{h,cpp}`,
   `src/runtime/ngram_draft.h`), and — opt-in — the trained MTP head
   (`src/runtime/engine_spec_mtp.cpp`, forward in
   `src/compute/mtp_forward.cu`). Hybrid (SSM/GDN) models participate via a
   recurrent-state slab snapshot around the verify chunk (a partial
   acceptance restores the slab and re-forwards the accepted prefix);
   `speculative.hybrid` in `imp.conf` gates it, imp-cli `--bench` pins it
   off.

## Subsystems referenced across phases

- **Memory** — `src/memory/vram_allocator.cu`, `src/memory/kv_cache.cu`,
  `src/memory/kv_cache_manager.cpp`, `src/memory/layer_offload.cu`,
  `src/runtime/vram_budget.cpp`, `src/runtime/storage_planner.cpp`.
  Prefix caching (content-addressed KV block reuse) works on hybrid SSM/GDN
  models via `src/memory/recurrent_snapshot_store.cpp`: one recurrent-state
  slab per prefill is snapshotted at the largest block-aligned prompt
  position and restored on a prefix hit — KV blocks alone cannot skip
  prefill for a recurrent model.
- **Kernels** — `src/compute/` (attention, GEMM, RMSNorm, RoPE, SwiGLU,
  softmax, sampling) and `src/quant/` (dequant, FP8 quant, NVFP4 quant).
- **Constrained decoding** — four grammars, one contract. Each owns a host-side
  FSM and exposes the same `apply_mask(logits, vocab, stream)`:
  `JsonConstrainer` (any valid JSON), `SchemaConstrainer` (a JSON Schema, plus
  the tool-call envelopes), `RegexConstrainer` (`RegexNfa`, shared with
  JSON-Schema `pattern`), and `GrammarConstrainer` (GBNF — a nondeterministic
  pushdown simulator in `src/compute/gbnf_grammar.cpp`, so recursive and
  bracket-balanced formats are expressible where a regex is not).
  `ConstraintManager` (`src/runtime/constraint_manager.h`) picks at most one per
  request and is pooled across requests. The mask is applied in
  `src/exec/executor.cu` — through a single `apply_constraint_mask` helper,
  because the sampling paths that must not bypass it are easy to miss. The
  scheduler routes any constrained request through the pipelined constrained
  decode (`Engine::step_constrained_pipeline`), gated on one `needs_constrained`
  flag — a second flag for the same question is how regex requests ended up
  taking a different path than the JSON ones for no reason.
- **Public C API** — `include/imp/{imp,types,error,config}.h`,
  implemented in `src/api/imp_api.cpp`. ABI-stable per CONTRIBUTING.md.

## Known limitations

- **The cuBLAS attention path allocates ~384 MiB of S-matrix workspace** (default `attention.attn_scores_mib`), which caps maximum context length for that legacy path; FA2 is the primary prefill kernel for hd=128 (#687) and hd=256 (#932), and on FA2-served configs the S-matrix is skipped entirely at init — so this only applies to the remaining cuBLAS fallback configs (heterogeneous shapes, learned sinks, opted-out hd=256).
- **`process_diag` is a process-wide config snapshot.** `RuntimeConfig` itself is per-Engine since Phase 5 Track D, but the leaf-utility diagnostics cache (`src/runtime/process_diag.h`) is seeded once per process — two Engines with *different* diagnostics/attention-variant settings in one process would fight over it.

## Re-rendering the diagram

```bash
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tsvg /d/architecture.dot -o /d/architecture.svg
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tpng /d/architecture.dot -o /d/architecture.png
```

Edit `architecture.dot` first, then regenerate both raster forms.

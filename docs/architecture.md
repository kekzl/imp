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
| Allocate workspaces | `init_features` | MMVQ scratch, cuBLAS S-matrix (~1 GiB — see Known limitations), FP8 activation scratch, split-K attn scratch |
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

`executor_attention.cu` decides which attention kernel to call. The
post-Phase-2 prefill gate is a two-branch switch:

```
const bool s_matrix_fits = attn_scores_buf_ != nullptr &&
                           n <= attn_scores_.shape[1];
const bool non_gemma4_sliding = !force_cublas_attn && sliding_active;

if (s_matrix_fits && !non_gemma4_sliding)
    attention_cublas_prefill(...);
else
    attention_prefill_dispatch(...);   // FMHA fallback
```

When the S-matrix workspace can hold `[nh, n, n]` AND the layer is
either non-sliding or is Gemma-4 (which uses the cuBLAS sliding mask
via chunked prefill), prefill goes through `attention_cublas_prefill`:
cuBLAS QK^T → ~1 GiB S-matrix buffer → causal softmax → cuBLAS PV.
Otherwise it falls through to `attention_prefill_dispatch`
(`attention_dispatch.cu:30`), which selects among the per-dtype FMHA
kernels.

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
8. **(Optional) MTP spec-decode draft** — `src/runtime/mtp_forward.cu`.

## Subsystems referenced across phases

- **Memory** — `src/memory/vram_allocator.cu`, `src/memory/kv_cache.cu`,
  `src/memory/kv_cache_manager.cpp`, `src/memory/layer_offload.cu`,
  `src/runtime/vram_budget.cpp`, `src/runtime/storage_planner.cpp`.
- **Kernels** — `src/compute/` (attention, GEMM, RMSNorm, RoPE, SwiGLU,
  softmax, sampling) and `src/quant/` (dequant, FP8 quant, NVFP4 quant).
- **Public C API** — `include/imp/{imp,types,error,config}.h`,
  implemented in `src/api/imp_api.cpp`. ABI-stable per CONTRIBUTING.md.

## Known limitations

- **The cuBLAS attention path allocates ~1 GiB of S-matrix workspace**, which caps maximum context length; a tiled streaming softmax or FMHA-default-switch would eliminate this.
- **`RuntimeConfig::current()` is a global singleton** rather than per-Engine, preventing true multi-engine use in a single process.

## Re-rendering the diagram

```bash
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tsvg /d/architecture.dot -o /d/architecture.svg
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tpng /d/architecture.dot -o /d/architecture.png
```

Edit `architecture.dot` first, then regenerate both raster forms.

<!--
layer: L2
audience: kernel-devs
verified: 2026-08-28
commit: be825e4a
-->

# imp - Architecture

Canonical narrative companion to [`architecture.svg`](../architecture.svg). The SVG shows the structural overview; this file explains each phase and points at the source files that implement it. If the code and this document disagree, the code wins, and the disagreement is a bug in this document.

## Target architecture

**This is the one place in the repository that states what consumer Blackwell has and lacks.** Every other document links here rather than restating it; the `docs_lint.py` forbidden-token check keeps it that way, because the same delimitation used to appear in eight files and a reader could not tell which one was maintained.

imp compiles for **`sm_120a` exclusively**, emitting raw SASS via direct gencode, with a `compute_120f` PTX fallback for the other consumer Blackwell SKUs. No portability layer, no second target.

**Consumer Blackwell is not a smaller datacenter Blackwell.** `sm_120a` does *not* have:

| absent on `sm_120a` | where it does exist | consequence for imp |
|---|---|---|
| `tcgen05` async MMA | datacenter Blackwell (`sm_100`, B200) | the MMA always blocks the issuing warp, so a producer/consumer pipeline cannot be built around it |
| TMEM | `sm_100` | no tensor-memory accumulator ring; accumulators live in registers |
| `wgmma` | Hopper and `sm_100` | the tensor-core path is register-based `mma.sync`, and the FA4-style warpgroup split is not expressible |

What `sm_120a` *does* have, and imp uses:

- **NVFP4 block-scaled `mma.sync` with `kind::mxf4nvf4`**, the FP4 path. FlashAttention-2-style block scaling rather than a B200 kernel design.
- **FP8 MMA `kind::f8f6f4`**, enabled by the `f` family-feature suffix, used for attention scores.
- **TMA bulk-tensor loads.** `src/compute/gemm_grouped_nvfp4_smallM.cu:65` wraps `cp.async.bulk.tensor.2d...` and emits `UTMALDG`. Whether the CUTLASS *warp-specialized grouped GEMM tactic* is selectable on this arch is **unresolved** and deliberately not claimed either way; see [`OPEN_QUESTIONS.md`](../audit/docs-rewrite/OPEN_QUESTIONS.md) Q1.

**The practical consequence, stated once so nobody re-derives it:** kernel designs published for B200 or Hopper do not port. When a paper or a competing engine reports a large FP4 win, check which architecture it was measured on before treating it as a lever here.

Deeper hardware notes, MMA shapes and the measured ceilings: [`SM120.md`](SM120.md).

## At a glance

imp runs LLM inference end-to-end in four phases:

1. **Load** - read a GGUF file or a Hugging-Face SafeTensors directory into a `Model` object with a `WeightMap`, a `Tokenizer`, and a `ModelConfig`.
2. **Engine init** - resolve runtime config, upload weights to VRAM, build the paged KV cache, allocate workspaces, capture CUDA graphs for decode.
3. **Prefill** - run the prompt through the per-layer forward pass (chunked if the architecture supports it), producing the first-token logits.
4. **Decode** - replay the captured CUDA graph per token: attention → FFN → LM head → penalties → sampler → stop check, looping until EOS or limit.

See [`architecture.svg`](../architecture.svg) for the full graph including the attention dispatcher, memory subsystem, and kernel subsystem.

## Phase 1 - Load (one-time, `src/model/`)

Entry: `imp_model_load(path) → ImpModel` (`include/imp/imp.h`, dispatched via `src/api/imp_api.cpp`).

Format detection inspects the path: a `.gguf` file routes to `src/model/gguf_loader.cpp`; a directory containing `config.json` and `*.safetensors` routes to `src/model/safetensors_loader.cpp`, with optional LLM-Compressor recipe handling in `src/model/llm_compressor_loader.cpp`.

Both loaders produce:

- A **WeightMap** (`src/model/weight_map.cpp`) - tensor name → role.
- A **Tokenizer** (`src/model/tokenizer.cpp` + `chat_template.cpp` + `jinja.cpp` + optional `sentencepiece_loader.cpp`).
- A **ModelConfig** + **Model** object (`src/model/model.cpp`, `src/model/model_arch.h`).

## Phase 2 - Engine init (one-time, `src/runtime/engine.cpp`)

Entry: `imp_context_create(model, ImpConfig) → ImpContext`.

`Engine::init()` orchestrates the init pipeline; the major steps are distinct private methods on `Engine`:

| Step | Method | Notes |
|---|---|---|
| Load runtime config | `RuntimeConfig::load()` | `imp.conf` + `--config` CLI + legacy env-var seeds (`src/runtime/config.cpp`) |
| Resolve quant/KV/SSM dtypes | `init_resolve_*` group | `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_` |
| Compute max sequence length | `init_compute_max_seq_len_` | VRAM budget → max context (`src/runtime/vram_budget.cpp`) |
| Upload weights | `init_weights` | `upload_weight` + `upload_expert_weights` in `src/model/weight_upload.cu`; pre-dequant orchestrated by `src/exec/executor_pre_dequant.cu` (calls per-phase TUs `pre_dequant_phase*.cu`: `pre_dequant_phase0_nvfp4_loader.cu`, `pre_dequant_phase1_fp16_cache.cu`, `pre_dequant_phase2_fp8_cache.cu`, `pre_dequant_phase3_nvfp4_decode.cu`, `pre_dequant_phase3c_mxfp4.cu`, `pre_dequant_phase4_tensor_registry.cu`, `phase4b` drop-source + VRAM reclamation, `phase4c` second-pass FP8 using reclaimed VRAM) |
| Open the T2 arena + graph slot pool | `engine_arena_open`, `graph_slot_pool_open_for` | Engine-persistent tier, sized from `exec_t2_demand()`. Opened BEFORE the first tenant: the arena acquires its region here, which is what reserves those bytes against everything allocated later |
| Init KV cache | `init_kv_cache` | **Weight caches are built first, then the pool takes the measured residual.** The reverse order sized the pool from an *estimate* of cache demand and starved the caches when the estimate was low - that is #1103, and it cost ~7x decode on gpt-oss-20b. Paged blocks (block_size=16); dtype is FP16 / FP8 / INT8 / INT4 / NVFP4 / MXFP4 |
| Allocate workspaces | `init_features` | MMVQ scratch, cuBLAS S-matrix (~384 MiB default `attention.attn_scores_mib` - see Known limitations), FP8 activation scratch, split-K attn scratch |
| Warm up | `warmup()` | Captures CUDA graph for decode (`src/runtime/cuda_graph.cu`) |
| Pre-size speculative scratch | `prewarm_spec_scratch_` | Last step before the allocation-phase guard arms. The verify path's one-shot capacity resolutions happen here rather than at first use, so serving allocates nothing |

The Engine façade (`engine.cpp`, ~570 LOC) delegates to 6 per-subsystem TUs by concern (resolver, weight upload, KV cache, workspaces, scheduler, sampling/stop).

## Phase 3 - Prefill (per request, `Engine::step_prefill`)

Entry: `imp_prefill_with_params(tokens, n) → status`.

Per-chunk loop in `src/exec/executor_forward.cu` (or `executor_forward_moe.cu` for MoE architectures). Each layer runs:

```
RMSNorm → QKV GEMM + RoPE + KV-cache write → Attention → O proj →
RMSNorm + residual → FFN (dense SwiGLU or MoE top-k grouped GEMM)
```

After the last layer of the last chunk: final RMSNorm + LM head → logits.

With an image in the request, the encoder has already run and its merged embeddings replace the expanded `<|image_pad|>` positions before the first layer; on Qwen3-VL the first few layers additionally get DeepStack taps added at those same positions (see **Vision** under Subsystems).

### Attention dispatcher (the central choice)

`executor_attention_prefill.cu` decides which attention kernel to call. Since #687 the prefill gate is **FA2-first**:

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

The FP16-QK FA2 kernel is the primary path for hd=128 and - since #930/#932 (`attention.fa2_hd256`, default on) - hd=256 at every length (at-or-above the materialized cuBLAS path: ~parity pp512, +24% pp1024, +52% pp2048), and needs no S-matrix. `attention_cublas_prefill` (cuBLAS QK^T → ~384 MiB S-matrix → causal softmax → cuBLAS PV) stays the fallback for the configs FA2 declines: `hd ∉ {128, 256}`, hd=256 with `fa2_hd256=false`, and `force_cublas_attn` (learned sinks / truly heterogeneous per-layer shapes; uniform GDN/Mamba2-hybrid shapes are FA2-servable since #932). Everything else falls through to `attention_prefill_dispatch`, which selects among the per-dtype FMHA kernels. Full coverage matrix: [`attention-dispatch.md`](ATTENTION_DISPATCH.md).

`force_cublas_attn` is set per-layer for Gemma-4 hd=512 global layers, where FMHA OOMs the 100 KiB smem cap.

Decode attention uses a separate switch on `cache_dtype` further down in the same file, dispatching to one of the paged kernels (INT4 / NVFP4 ± TC / MXFP4-KV / INT8 / FP8 / FP16-paged).

## Phase 4 - Decode loop (per token, `Engine::step_decode_forward`)

Entry: `imp_decode_step(params) → next_token` (or the streaming wrapper `imp_generate_streaming`).

Per token:

1. **Replay** the captured CUDA graph (`src/runtime/cuda_graph.cu`). Graph capture is enabled for most architectures; non-Gemma-4 MoE disables it because of host-side routing.
2. **Paged attention decode** - kernel chosen by KV dtype.
3. **FFN GEMV** - dp4a / mma.sync / NVFP4 variants in `executor_ffn.cu`.
4. **LM head GEMV** → logits.
5. **Apply penalties** (repeat / freq / presence / DRY) - `src/compute/sampling.cu` (kernels) called from `src/exec/executor.cu`. Parameters are declared in `src/runtime/request.h`.
6. **Sampler** (temp / top-p / top-k / min-p / typical / mirostat) - `src/compute/sampling.{h,cu}` (`sample_greedy`, `sample_topk_topp`, `sample_mirostat_v2`, `apply_typical_p`).
7. **Stop check** - EOS, max_tokens, stop strings.
8. **(Optional) speculative decoding** - batch-1 greedy requests verify drafts as teacher-forced continuation chunks (`src/runtime/engine_spec_ngram.cpp`). Draft sources: the suffix index / n-gram matcher (`src/runtime/suffix_draft.{h,cpp}`, `src/runtime/ngram_draft.h`) and - opt-in - the trained MTP head (`src/runtime/engine_spec_mtp.cpp`, forward in `src/compute/mtp_forward.cu`). Hybrid (SSM/GDN) models participate via a recurrent-state slab snapshot around the verify chunk (a partial acceptance restores the slab and re-forwards the accepted prefix); `speculative.hybrid` in `imp.conf` gates it, imp-cli `--bench` pins it off.

## Subsystems referenced across phases

- **Memory** - has its own design document: [`docs/internals/MEMORY.md`](MEMORY.md) is canonical for anything about ownership, lifetime or capacity, and [`AUDIT.md`](../../AUDIT.md) records what was measured on the way (including the negative results). The short version: five lifetime tiers (T1 model-resident, T2 engine-persistent, T3 pooled fixed-block, T4 forward-scratch, T5 host staging - split into T5a transient and T5b engine-persistent pinned, because a buffer pinned once and reused every decode step cannot obey "load only") over a three-layer stack. `src/memory/backend.{h,cpp}` is the only code that talks to the driver about *device* memory and `memory/host_pinned.{h,cpp}` is its host-side counterpart; `arena` / `block_pool` / `scratch_stack` / `graph_slots` are the tier allocators; `StableSpan` vs `DeviceSpan` encodes in the type system which memory a captured CUDA graph may bake an address into. `src/memory/plan.cpp` plans capacity without querying the device; `src/runtime/vram_budget.cpp` is still the live pass it shadows. Older, still-live pieces: `src/memory/vram_allocator.cu`, `src/memory/kv_cache.cu`, `src/memory/kv_cache_manager.cpp`, `src/memory/layer_offload.cu`, `src/runtime/storage_planner.cpp`. Prefix caching (content-addressed KV block reuse) works on hybrid SSM/GDN models via `src/memory/recurrent_snapshot_store.cpp`: one recurrent-state slab per prefill is snapshotted at the largest block-aligned prompt position and restored on a prefix hit - KV blocks alone cannot skip prefill for a recurrent model.
- **Kernels** - `src/compute/` (attention, GEMM, RMSNorm, RoPE, SwiGLU, softmax, sampling) and `src/quant/` (dequant, FP8 quant, NVFP4 quant).
- **Constrained decoding** - four grammars, one contract. Each owns a host-side FSM and exposes the same `apply_mask(logits, vocab, stream)`: `JsonConstrainer` (any valid JSON), `SchemaConstrainer` (a JSON Schema, plus the tool-call envelopes), `RegexConstrainer` (`RegexNfa`, shared with JSON-Schema `pattern`), and `GrammarConstrainer` (GBNF - a nondeterministic pushdown simulator in `src/compute/gbnf_grammar.cpp`, so recursive and bracket-balanced formats are expressible where a regex is not). `ConstraintManager` (`src/runtime/constraint_manager.h`) picks at most one per request and is pooled across requests. The mask is applied in `src/exec/executor.cu` through a single `apply_constraint_mask` helper, because the sampling paths that must not bypass it are easy to miss. The scheduler routes any constrained request through the pipelined constrained decode (`Engine::step_constrained_pipeline`), gated on one `needs_constrained` flag - a second flag for the same question is how regex requests ended up taking a different path than the JSON ones for no reason.
- **Vision** (`src/vision/`) - two shapes, one seam. The Gemma path (SigLIP / gemma4v) loads its encoder from a separate `mmproj.gguf` and produces a fixed token count from a fixed `image_size`. The Qwen3-VL path loads its tower from the checkpoint itself and is *dynamic*: `smart_resize` + `patchify` derive a per-image grid, so the token count varies with the picture, and every workspace is sized from a patch budget (`runtime.vision_max_patches`) rather than an image size - which makes the budget a ceiling, so a larger image is scaled down rather than refused. Both hand the LM a merged embedding, but Qwen3-VL touches the text forward in two more places: `deepstack_inject.cu` adds encoder taps after each of the first few LM layers at the image-token positions (`executor_forward.cu`), and positions are three-axis M-RoPE (`model/mrope_positions.cpp` builds per-token (t,h,w); `compute/rope.cu` applies the section split). One image per request today. The seam is the `<|image_pad|>` placeholder: `model/image_placeholders.cpp` expands it to the encoder's real token count and salts the prefix-cache hash with the image content, since every image token otherwise carries the same id and two different pictures would share a prefix.
- **Public C API** - `include/imp/{imp,types,error,config}.h`, implemented in `src/api/imp_api.cpp`. ABI-stable per CONTRIBUTING.md.

## Known limitations

- **The cuBLAS attention path allocates ~384 MiB of S-matrix workspace** (default `attention.attn_scores_mib`), which caps maximum context length for that legacy path. FA2 is the primary prefill kernel for hd=128 (#687) and hd=256 (#932), and on FA2-served configs the S-matrix is skipped entirely at init - so this only applies to the remaining cuBLAS fallback configs (heterogeneous shapes, learned sinks, opted-out hd=256).
- **`process_diag` is a process-wide config snapshot.** `RuntimeConfig` itself is per-Engine since Phase 5 Track D, but the leaf-utility diagnostics cache (`src/runtime/process_diag.h`) is seeded once per process - two Engines with *different* diagnostics/attention-variant settings in one process would fight over it.

## Re-rendering the diagram

```bash
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tsvg /d/architecture.dot -o /d/architecture.svg
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tpng /d/architecture.dot -o /d/architecture.png
```

Edit `architecture.dot` first, then regenerate both raster forms.

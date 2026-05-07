# Roadmap

Open work and known limitations. Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

This is a single-author single-target experiment, so "roadmap" is more "current focus" than "schedule." Items here are ordered by impact, not by ETA.

## Known limitations

### FP8 KV cache: Gemma-4 carve-out

Gemma-4's dual head_dim layout (256 SWA / 512 global) doesn't fit the FP8 KV write/read kernel's single-stride assumption, so Gemma-4 force-falls-back to FP16 KV. Lifting the carve-out needs per-layer head_dim awareness in the KV write/read kernels — separate, larger work item.

Default KV dtype is FP16; FP8 is opt-in via `--kv-fp8` (or `kv_cache.dtype = "fp8"` in `imp.conf`). Coherent on Qwen3 dense, Qwen3.5/3.6 GDN, and Llama-3.2.

### NVFP4 long-context recall (model-inherent, not NVFP4-specific)

The 2048-token sentinel-recall test in `tests/fixtures/battery_prompts.json` fails on multiple model+format combinations:

- Gemma-4-26B-A4B-it-NVFP4 (llm-compressor format) — fails: emits "the moon could read its spine" (regurgitates document content)
- gemma-4-26B-A4B-it-Q8_0 GGUF (≈2× higher precision per weight) — same failure mode: emits "thought\nThe library was old…"
- Qwen3-30B-A3B-NVFP4-Modelopt (Modelopt format) — also fails

Since the failure reproduces on Q8_0 (no NVFP4 anywhere) and on Modelopt (different scale convention), this is **not** NVFP4 format- or scale-specific. It's a copy-from-context attention/recall limitation at long context. Any fix likely belongs in attention/KV cache or chat-template handling, not weight quantization. The roadmap previously listed this as _"llm-compressor NVFP4: degenerate output past ~30 tokens"_ — empirical bracketing 2026-05-07 (see `memory/llm_compressor_input_scale_dead_end_2026_05_07.md`) refutes the NVFP4-specific framing.

### NVFP4 SmoothQuant input_scale (Mistral-3.2 NVFP4)

Mistral-Small-3.2-NVFP4 was calibrated with SmoothQuant 0.9, which records a per-Linear `input_scale` in SafeTensors. imp loads `input_scale` into `nvfp4_scratch_` but does not consume it at inference. PR #78 worked around long-prompt drift by disabling the 600-token default system prompt.

Direct absorption of `input_scale` as a per-tensor scalar GEMM alpha modifier (in either direction) was tested 2026-05-07 on Gemma-4-NVFP4 and refuted: phase4 18/20 → 4/20 (full degeneration). A real fix likely needs the per-channel SmoothQuant scaling vector applied during activation quantization, not a scalar alpha modifier — testable only against Mistral-3.2-NVFP4 (not present in default model set).

### Qwen3.5-27B MXFP4 fails at load

12 GiB of MXFP4 weights plus the 48 GiB FP16 fallback oversubscribes 32 GB of VRAM. PR #60 added a clear diagnostic. A real fix needs host-dequant + a storage planner. Workarounds: 9B Q8_0, 35B-A3B Q4_K_M.

### Gemma-4 Q4_K_M code-gen drift

Q4_K_M decodes coherent for chat but degenerates on complex code-gen prompts (Fibonacci → backtick loop). Cause is accumulated FP16 drift over 30 layers. Practical fix: use Q5_K_M or Q8_0 when output quality matters.

### MoE D2H routing blocks CUDA Graphs (non-fast-path)

Non-Gemma-4 / non-NVFP4-prequant MoE decode falls through the legacy expert-routing path with a D2H sync per layer per token, so CUDA Graphs are disabled for these models. Gemma-4 and NVFP4-prequant MoE (Qwen3.6, Gemma-4 llm-compressor) capture cleanly via the decode fast-path. Generalising the fast-path to GGUF MoE would restore Graphs.

### Reasoning models + JSON schema — preamble pass-through

Reasoning models (Qwen3.6, DeepSeek-R1, Gemma-4-thinking) emit `<think>...</think>` before every response. Strict JSON / JSON-Schema enforcement starting at token 0 masks the `<think>` opener, leaving the model with no valid token to sample. Auto-detected via the tokenizer (presence of `<think>` + `</think>` special tokens) and handled by `PreambleGate` (`src/compute/preamble_gate.h`): the gate lets all tokens pass until the close marker, an `{` / `[` is observed, or a budget cap is hit, then strict enforcement kicks in.

Open follow-ups: tool-calling response-format combinations (`tools` + `response_format=json_schema`) aren't covered yet; lenient grammar prefixes for non-reasoning preambles like markdown fences (` ```json `) would generalise the same pattern.

## Performance work

### Native MXFP4 GGUF weight format

Native MXFP4 weights would feed directly into Blackwell tensor cores via CUTLASS — zero dequant overhead, expected 2–4× prefill speedup vs Q4_K_M.

CUTLASS MXFP4 GEMM is fully implemented today (`attention.mxfp4 = "always"`), but only triggers when an NVFP4 cache exists as source data. Native MXFP4 GGUF would remove that dependency. Required pieces: GGUF type extension + loader, Python converter (SafeTensors → block-Hadamard → MXFP4 → GGUF), GPU-side weight upload, MXFP4 GEMV for decode, and MR-GPTQ calibration. Round-to-nearest MXFP4 sits at +5–15% perplexity vs Q8_0 — worse than Q4_K_M's +2.2% — so calibration is effectively required to ship.

### Closing the TurboQuant–FP8 gap

TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode (191 vs 248 tok/s). The gap is algorithm-inherent — QJL sketch computation adds per-token overhead. Closing it would need to drop QJL and switch to MXFP4 K directions with group micro-scales.

### 1024→2048 prefill cliff on small dense models

Qwen3-4B Q8_0 drops from 27k to 19k tok/s at the dispatch boundary where cuBLAS attention hands off to FP8 FMHA. Output stays correct; the FP8-FMHA kernel is just less tuned for the small-model regime. Options: raise the cuBLAS cap past 1024, or tune FP8-FMHA occupancy / Bq.

### `pp=512` on large dense models

Qwen3-32B Q4_K_M and Mistral-24B Q6_K sit at ~0.5–0.6× llama.cpp at `pp=512`. Suspected cuBLAS autotuning variance + launch-overhead-bound regime. Output is correct; not gating any user.

### Speculative decoding — investigated and shelved

EAGLE-3, self-speculative, DFlash, PPM-based TurboDraft, and n-gram speculation were all investigated. None paid off on a single RTX 5090: decode is bandwidth-bound, and the variants tested either failed to amortise weight reads (EAGLE-3, self-spec at 56–50% of baseline) or had unacceptable acceptance rates (PPM 0% on real text). Spec-decode CLI flags were removed in `7380ea8`.

## Research interest

These are upstream features that would unlock real wins but haven't been integrated yet.

### CUDA 13.2 / CCCL 3.2 features

- **Grouped GEMM with CUDA Graphs + device-side shapes** (`cublasLtMatmulGrouped` with NVFP4, `sm_120` since CUDA 13.2 Update 1) — host-sync-free MoE expert dispatch, direct unblock for the general MoE D2H limitation above.
- **`cub::DeviceTopK`** (AIR) — 5× faster top-k for `top_k > 128`.
- **`cub::DeviceSegmentedReduce`** (fixed-size variant) — up to 66× speedup for small uniform segments. Useful for per-head reductions.
- **`cudaMemcpyWithAttributesAsync`** — L2 persistence hints on individual transfers; prefix-cache pinning without batched API.
- **`add.f32x2` native PTX** (Blackwell) — softmax / accumulation instruction-count reduction.

### PTX ISA 9.2

- **`cp.async.bulk` with `.ignore_oob`** — eliminates bounds-checking in TMA descriptors for variable seq lengths. Simplifies paged attention with partial last blocks.
- **`st.async.b128`** — 16-byte async stores for KV cache writeback.
- **`cvt .bf16x2` ↔ narrow** (`.e2m1x2`, `.e4m3x2`) — packed FP4/FP8 pair conversion, 2× throughput for KV cache quant.
- **`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA — finer scale granularity (1 per 4 elements vs per block).
- **`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`** — operand layouts decoded byte-exact in PR #55, integration open.

### Long-context KV memory

Decode is bandwidth-bound and every sub-byte KV quantization tested so far regresses decode. The next class of win comes from reducing the *token count*, not the element precision:

- **K2 MLA (DeepSeek)** — latent vector replaces full K/V, ~–90% KV VRAM.
- **K5 Token eviction (H2O)** — drop 50–70% of tokens by attention score.
- **K8 CPU offload** — async prefetch for cold tokens, enables 100K+ context.

### KV cache compression research

- **BitDecoding** ([arxiv:2503.18773](https://arxiv.org/abs/2503.18773)) — 8.6× vs FP16 FlashDecoding on Blackwell via MXFP4 KV. Builds on existing MXFP4 infrastructure.
- **DeltaKV** ([arxiv:2602.08005](https://arxiv.org/abs/2602.08005)) — residual-based KV compression, 187 tok/s at 128K context on Blackwell PRO 6000. Orthogonal to weight quant.

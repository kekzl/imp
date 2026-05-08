# Roadmap

Open work and known limitations. Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

This is a single-author single-target experiment, so "roadmap" is more "current focus" than "schedule." Items here are ordered by impact, not by ETA.

## Known limitations

### FP8 KV cache: Gemma-4 carve-out

Gemma-4's dual head_dim layout (256 SWA / 512 global) doesn't fit the FP8 KV write/read kernel's single-stride assumption, so Gemma-4 force-falls-back to FP16 KV. Lifting the carve-out needs per-layer head_dim awareness in the KV write/read kernels — separate, larger work item.

Default KV dtype is FP16; FP8 is opt-in via `--kv-fp8` (or `kv_cache.dtype = "fp8"` in `imp.conf`). Coherent on Qwen3 dense, Qwen3.5/3.6 GDN, and Llama-3.2.

### Chunked prefill scope (full-attention + FP16/FP8 KV)

Default `prefill_chunk_size = 512` for full-attention models (Qwen3, Llama, Mistral) with FP16 or FP8 KV cache. Past chunks' K/V are read from the paged cache via `paged_kv_gather_*` and concatenated with the current chunk before a rectangular `attention_cublas_prefill` with `q_offset`-aware causal masking. PR #114 mitigation (default `prefill_chunk_size = 0`) is replaced by `Engine::resolve_prefill_chunk_size_()` which clamps to 0 for out-of-scope archs.

**Out-of-scope** — stay at `prefill_chunk_size = 0` via per-arch default; explicit `--prefill-chunk-size N` is logged + clamped to 0:

- Gemma-4 (SWA + dual head_dim 256/512)
- Hybrid models with non-attention layers (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)
- Sub-byte KV cache dtypes (INT4, NVFP4, TurboQuant variants)

Each excluded class is a separate larger work item (paged-prefill kernel with SWA-aware mask / dual-head_dim support / sub-byte dequant during gather).

### `d_pf_block_tables_` undersized for prompts ≥ max_seq_len

When a single prompt exceeds `max_seq_len`, the engine's pre-allocated device buffer `d_pf_block_tables_` (sized `max_seq_len / block_size`) overflows during `cudaMemcpyAsync`. Discovered while validating chunked-prefill at >4096-token prompts. Mitigation: configure `max_seq_len` to actual maximum prompt length. Real fix: size `d_pf_block_tables_` from `max_kv_blocks` instead.

### NVFP4 SmoothQuant input_scale (Mistral-3.2 NVFP4)

Mistral-Small-3.2-NVFP4 was calibrated with SmoothQuant 0.9, which records a per-Linear `input_scale` in SafeTensors. imp loads `input_scale` into `nvfp4_scratch_` but does not consume it at inference. PR #78 worked around long-prompt drift by disabling the 600-token default system prompt.

Direct absorption of `input_scale` as a per-tensor scalar GEMM alpha modifier (in either direction) was tested 2026-05-07 on Gemma-4-NVFP4 and refuted: phase4 18/20 → 4/20 (full degeneration). A real fix likely needs the per-channel SmoothQuant scaling vector applied during activation quantization, not a scalar alpha modifier — testable only against Mistral-3.2-NVFP4 (not present in default model set).

### Qwen3.5-27B MXFP4 fails at load

12 GiB of MXFP4 weights plus the 48 GiB FP16 fallback oversubscribes 32 GB of VRAM. PR #60 added a clear diagnostic. A real fix needs host-dequant + a storage planner. Workarounds: 9B Q8_0, 35B-A3B Q4_K_M.

### Gemma-4 Q4_K_M code-gen drift

Q4_K_M decodes coherent for chat but degenerates on complex code-gen prompts (Fibonacci → backtick loop). Cause is accumulated FP16 drift over 30 layers. Practical fix: use Q5_K_M or Q8_0 when output quality matters.

### MoE expert offload disables CUDA Graphs

Decode fast-path (`src/graph/executor_forward_moe.cu:524`) handles all device-resident MoE quants — Q6_K, Q8_0, Q4_0, Q4_K, Q5_K, Q2_K, Q3_K, Q5_1, NVFP4 — fully device-side (no D2H memcpy of routing or expert offsets), so CUDA Graphs capture cleanly. Verified A/B 2026-05-07: Qwen3-Coder Q6_K tg128 117 → 232 tok/s (+97%), Gemma-4 Q4_K_M tg128 65 → 179 tok/s (+177%).

The remaining limitation is **host-offloaded experts**: when the model + KV doesn't fit in VRAM, `experts_on_host_=true` triggers per-layer H2D staging via `expert_cache_` LRU at `executor_forward_moe.cu:1517`, which inserts a host pointer dereference + `cudaMemcpyAsync` per expert per token. `engine.cpp:936` disables CUDA Graphs in that mode. Tip: bumping `IMP_EXPERT_OVERHEAD_PCT` from 30 to 10 trades VRAM headroom for full on-device experts and unlocks +180% decode on Qwen3.6-35B Q4_K_M. Generalising the LRU prefetch to be device-side / async-pipelined would restore Graphs while keeping host-offload available.

### Reasoning models + JSON schema — preamble pass-through

Reasoning models (Qwen3.6, DeepSeek-R1, Gemma-4-thinking) emit `<think>...</think>` before every response. Strict JSON / JSON-Schema enforcement starting at token 0 masks the `<think>` opener, leaving the model with no valid token to sample. Auto-detected via the tokenizer (presence of `<think>` + `</think>` special tokens) and handled by `PreambleGate` (`src/compute/preamble_gate.h`): the gate lets all tokens pass until the close marker, an `{` / `[` is observed, or a budget cap is hit, then strict enforcement kicks in.

Non-reasoning models (Llama-3.2 etc.) get the same gate in budget-only mode (8-token slack) so markdown-fence preambles like ` ```json ` and short verbal openers ("Sure! ") pass through cleanly.

When a request sets both `tools` and `response_format=json_schema`/`json_object`, the engine-side `PreambleGate` enters tool-aware mode. It bypasses the schema mask through the entire tool-call body (delimited by single-token tags for ChatML/Hermes/Mistral/Gemma, or `<function=`/`</function>` char-prefix/suffix for Llama3) and stays unmasked for the rest of the generation, supporting parallel tool calls. If the model emits free-text JSON instead, the schema mask kicks in normally on the first `{`/`[`. Tool argument validation continues to flow through each tool's own `parameters` schema (post-hoc, not in-stream).

## Performance work

### Closing the TurboQuant–FP8 gap

TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode (191 vs 248 tok/s). The gap is algorithm-inherent — QJL sketch computation adds per-token overhead. Closing it would need to drop QJL and switch to MXFP4 K directions with group micro-scales.

### `pp=512` on large dense models

Qwen3-32B Q4_K_M and Mistral-24B Q6_K sit at ~0.5–0.6× llama.cpp at `pp=512`. Suspected cuBLAS autotuning variance + launch-overhead-bound regime. Output is correct; not gating any user.

### Speculative decoding — investigated and shelved

EAGLE-3, self-speculative, DFlash, PPM-based TurboDraft, and n-gram speculation were all investigated. None paid off on a single RTX 5090: decode is bandwidth-bound, and the variants tested either failed to amortise weight reads (EAGLE-3, self-spec at 56–50% of baseline) or had unacceptable acceptance rates (PPM 0% on real text). Spec-decode CLI flags were removed in `7380ea8`.

## Research interest

These are upstream features that would unlock real wins but haven't been integrated yet.

### CUDA 13.2 / CCCL 3.2 features

- ~~**Grouped GEMM with CUDA Graphs + device-side shapes**~~ — re-tested 2026-05-08 against cuBLAS 13.4.0.1 (`tools/analysis/probe_cublaslt_grouped.cu`): zero algorithms returned for FP16/BF16/FP8/NVFP4 on sm_120. Grouped layout API still marked Experimental in `cublasLt.h` 13.4 and only supported on datacenter Blackwell (SM100/B200), not consumer SM120. Re-run probe on each new cuBLAS release.
- ~~**`cub::DeviceTopK`**~~ — already wired in production (`src/compute/sampling.cu:834`, `cub::DeviceTopK::MaxPairs` for the `top_k > MAX_TOP_K=128` path with a small follow-up `DeviceRadixSort` over just the top-k results for top-p ordering).
- ~~**`cub::DeviceSegmentedReduce`**~~ — re-evaluated 2026-05-08, no applicable use case in imp. The 66× speedup claim applies to host-launched many-small-segments patterns (e.g. CUB benchmarks reducing thousands of fixed-size rows in one call). imp's per-head reductions are all already inside their owning kernel as warp-/block-level shuffle reductions (RMSNorm, attention softmax, MoE gate norm) — fused, optimal, and unrelated to DeviceSegmentedReduce's regime.
- **`cudaMemcpyWithAttributesAsync`** — L2 persistence hints on individual transfers; prefix-cache pinning without batched API.
- **`add.f32x2` native PTX** (Blackwell) — softmax / accumulation instruction-count reduction.

### PTX ISA 9.2

- **`cp.async.bulk` with `.ignore_oob`** — eliminates bounds-checking in TMA descriptors for variable seq lengths. Simplifies paged attention with partial last blocks.
- **`st.async.b128`** — 16-byte async stores for KV cache writeback.
- **`cvt .bf16x2` ↔ narrow** (`.e2m1x2`, `.e4m3x2`) — packed FP4/FP8 pair conversion, 2× throughput for KV cache quant.
- **`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA — finer scale granularity (1 per 4 elements vs per block).
- **`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`** — QKT path shipped in PR #56 (commit `b51788e`, default-on via `attention.fmha_blockscale = "auto"`); per-16-element UE4M3 SFA/SFB feed real `q_scales_fp8` / `k_scales_fp8`. Measured +1.8% Qwen3-4B MXFP4 at HD=128 (Phase 1 MMA is only ~15% of FMHA wall time, so 2.5× raw MMA → small visible delta). Open lever is **FP4 PV** (Phase 3 P×V): same MMA op for the second GEMM in attention, ~+13% additional upside but quality-risky (FP4-quant of post-softmax probabilities — needs SageAttention3-style two-level accumulator). 200-300 LoC, prereq is a PV-only A/B test harness. HD=256 models (Qwen3.5 GDN, Gemma-4 globals) would also see a larger Phase-1 fraction and better visible speedup, but no clean MXFP4 HD=256 model is currently in the test set.

### Long-context KV memory

Decode is bandwidth-bound and every sub-byte KV quantization tested so far regresses decode. The next class of win comes from reducing the *token count*, not the element precision:

- **K2 MLA (DeepSeek)** — latent vector replaces full K/V, ~–90% KV VRAM.
- **K5 Token eviction (H2O)** — drop 50–70% of tokens by attention score.
- **K8 CPU offload** — async prefetch for cold tokens, enables 100K+ context.

### KV cache compression research

- **BitDecoding** ([arxiv:2503.18773](https://arxiv.org/abs/2503.18773)) — 8.6× vs FP16 FlashDecoding on Blackwell via MXFP4 KV. Builds on existing MXFP4 infrastructure.
- **DeltaKV** ([arxiv:2602.08005](https://arxiv.org/abs/2602.08005)) — residual-based KV compression, 187 tok/s at 128K context on Blackwell PRO 6000. Orthogonal to weight quant.

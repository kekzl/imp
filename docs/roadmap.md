# Roadmap

Single-author, single-GPU experiment -- "roadmap" means "current focus," not "schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

## Direction: local inference for AI agents

The goal is making imp the fastest local engine for AI agent workloads on consumer Blackwell. Agents generate far more tokens per session (20k-100k+), accumulate context fast, and often run in parallel. This demands long context, concurrent request handling, and high decode throughput.

### Phase 1 -- Long context ✓

**Shipped PR #453.** FMHA kernels got `q_offset` for chunked prefill. S-matrix shrunk 1024→256 MiB. Auto `fmha_prefill_threshold` routes long sequences to FMHA. Context ceiling moved from ~4-6k to 32k+.

### Phase 2 -- Concurrent requests ✓

**Shipped PR #454.** Server handles up to 4 concurrent decode requests. Removed single-request cancellation guard. Added `runtime.max_batch_size` config. SSM/GDN models stay batch=1.

### Phase 3 -- KV streaming for long sessions ✓

**Shipped PR #455.** Auto-enables StreamingLLM when KV cache >90% full. Graceful degradation: sink tokens + sliding window, middle blocks freed. Agent sessions effectively unlimited.

## Open performance work

- **Q4_K_M prefill gap** (-38% vs llama.cpp) -- needs a custom in-SMEM Q4_K MMQ kernel with FP16 HMMA. Design spec: [`specs/2026-05-28-q4k-mmq-kernel-design.md`](superpowers/specs/2026-05-28-q4k-mmq-kernel-design.md). Prior INT8 IMMA and FP16 HMMA v2 approaches were both refuted. Estimated 2-3 weeks.

- **Sawtooth Wavefront Reordering** (PR #456) -- alternate KV scan direction per Q tile for L2 locality. Implemented in both FMHA kernels. Expected +5-15% prefill at 32k+ context.

## Architecture support

- **MLA (DeepSeek-V2/V3)** -- latent-vector KV for 64x compression. Design spec: [`specs/2026-05-28-mla-deepseeek-architecture-design.md`](superpowers/specs/2026-05-28-mla-deepseeek-architecture-design.md). Blocked on no local MLA model. Estimated 3-4 weeks.

## Known limitations

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Blackwell only.** No Hopper, Ada, Ampere. No AMD, Intel, Apple, CPU.
- **Gemma-4 Q4_K_M code-gen drift** -- accumulated FP16 rounding. Use Q5_K_M or Q8_0.
- **Qwen3.5-27B MXFP4 fails at load** -- blocked on no public MXFP4 GGUF + NaN bug.

## Investigated and shelved

- **Speculative decoding** -- none amortize weight reads on single bandwidth-bound GPU.
- **FFN contextual sparsity** -- warp-cooperative layout masks the skip. +0-1% measured.
- **BitDecoding (TC KV decode)** -- decode is weight-bound, not attention-bound. 0% gain.
- **NVFP4 GEMV tuning** -- 6 approaches refuted. 157 tok/s is 64-73% HBM peak.
- **FMHA rewrites** -- cluster, TMA bulk, long-context heuristic all A/B tested. cuBLAS wins.
- **MoE offload + CUDA Graphs** -- `expert_overhead_pct=10` default keeps most models on-device. Full kernel-driven slot resolution deferred (multi-week, marginal user impact).

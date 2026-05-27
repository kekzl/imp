# Roadmap

Single-author, single-GPU experiment -- "roadmap" means "current focus," not "schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md).

## Known limitations

- **Chunked prefill scope** -- works for full-attention models (Qwen3, Llama, Mistral), hybrid GDN/Mamba2+MoE (Qwen3.5/3.6, Nemotron-H), and Gemma-4 across FP16/FP8/NVFP4/INT4 KV. Out of scope: Gemma-3 (unverified, no test model), Llama-4 (MoE+SWA), and TurboQuant KV dtypes (needs sketch-aware gather).

- **Qwen3.5-27B MXFP4 fails at load** -- needs host-side dequant to fit in 32 GB VRAM. Blocked on two external issues: no public MXFP4 GGUF exists for this model, and a separate NaN bug in the N=48 alpha/beta MXFP4 GEMV path. Workaround: use Qwen3.5-9B Q8_0 or Qwen3.5-35B-A3B Q4_K_M.

- **Gemma-4 Q4_K_M code-gen drift** -- accumulated FP16 rounding over 30 layers causes degeneration on complex prompts. Use Q5_K_M or Q8_0 when output quality matters.

- **MoE expert offload disables CUDA Graphs** -- when the model doesn't fit in VRAM entirely, host-offloaded experts require per-layer H2D staging that prevents graph capture. Workaround: set `moe.expert_overhead_pct=10` to keep experts on-device and unlock graphs (+97-234% decode). A proper fix (kernel-driven slot resolution so captured graphs adapt to per-token routing) is a multi-week effort.

- **Reasoning models + JSON schema** -- models that emit `<think>...</think>` before responding would break strict JSON enforcement at token 0. Auto-detected and handled by `PreambleGate`: lets tokens pass until the close marker or a `{`/`[`, then strict enforcement kicks in.

## Open performance work

- **Q4_K_M prefill gap vs llama.cpp** -- imp sits at -48-59% on large dense Q4_K_M prefill. Closing the gap needs a custom tiled MMQ kernel; an INT8 IMMA prototype was built and deferred (plateaued at 4.3% of raw MMA peak, 3.8x slower e2e than dequant-to-cuBLAS).

## Investigated and shelved

Things that were tried and didn't pan out, so you don't have to wonder:

- **Speculative decoding** -- EAGLE-3, self-speculative, PPM-based TurboDraft, and n-gram speculation all investigated. None amortise weight reads on a single bandwidth-bound GPU. MTP on Qwen3.6 lands at 22-30% acceptance -- below the ~50% needed to break even.

- **FFN contextual sparsity** -- 25-52% theoretical sparsity confirmed, but the warp-cooperative GEMV layout means wallclock tracks the slowest warp, not the average skip rate. Measured +0-1% end-to-end. Code stays as opt-in research artifact.

- **BitDecoding (Tensor-Core KV decode)** -- all three phases shipped (Q-K WMMA, P-V WMMA, residual FP16 cache). 0% gain at any context length tested -- decode is weight-bandwidth-bound, not attention-math-bound on consumer Blackwell.

- **NVFP4 GEMV kernel tuning for 175 tok/s north-star** -- software prefetch, scale-batch4, silu tanh-form, GDN scan occupancy/split-K, kernel fusion paths A and B, LM-head GEMV swap all individually benchmarked and refuted. Top-3 GEMVs already run at 64-73% HBM peak. The 157 tok/s plateau on Qwen3-14B Q6_K is near-roofline for this hardware.

- **FMHA rewrites** -- cluster-launch, TMA bulk, and long-context heuristic all A/B tested. cuBLAS attention wins or ties on sm_120 at all tested shapes.

## Research interest

Worth tracking but not actively being worked on:

- **Sawtooth Wavefront Reordering** ([arxiv:2601.16032](https://arxiv.org/abs/2601.16032)) -- L2-locality technique for flash attention, portable to sm_120. Estimated +15-30% prefill at context >= 64K, but that workload is niche on 32 GB consumer VRAM. Re-eval when a real user workload regularly hits 64K+ context.

- **KV CPU offload** -- async prefetch for cold KV blocks to enable 100K+ context. Not trivially un-bottlenecked: full-attention over a 100K cold tail touches thousands of blocks per token (~40 ms vs 5 ms decode budget). Only pays off with attention regimes that naturally skip the cold tail (SWA, eviction, retrieval).

- **MLA (DeepSeek)** -- latent vector replaces full K/V for ~93% KV VRAM reduction. Gates on adding DeepSeek-V2/V3 architecture support to imp; no MLA-arch model in scope today.

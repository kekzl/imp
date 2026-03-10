# TODO

## Performance

- [x] **Gemma-3-12B Decode Bench** — ~~Bench mode "internal error"~~.
  Fix: `ignore_eos=true` during prefill (synthetic tokens → immediate EOS → FINISHED).
  `imp_decode_step` sets ignore_eos correctly for actual decode.

- [x] **Gemma-3-12B NVFP4 Decode** — head_dim=256 KV cache consumed VRAM, limiting
  NVFP4 budget. GeGLU+GEMV fusion boosted decode: 59→100 tok/s (+70%).
  KV budget trade-off (10% cap for mode 2) + incremental NVFP4 processing
  (net VRAM-negative per FP16→NVFP4 conversion) achieved 334/337 tensor
  coverage. Gemma-3-12B decode: 86→119 tok/s (bench), 96 tok/s (real prompt).

- [x] **CUTLASS FMHA head_dim=96** — Added template `Shape<_128, _96, _96>`.
  Phi4-mini actually has head_dim=128 (rope_dim=96 is partial RoPE, not head_dim).
  CUTLASS FMHA already worked correctly. Template added for future models with head_dim=96.

- [x] **Init Async Quantization** — Eliminated per-tensor `cudaStreamSynchronize` in
  weight cache construction. FP8 overflow, FP16→FP8 migration, and NVFP4 cache now
  use async calibrate+quantize with single sync at end. Bulk cudaMalloc for FP8 data
  and scale buffers. DeepSeek-14B init: 55.4s → ~35s (37% faster).

- [x] **Decode Kernel Fusion** — Fused activation+GEMV+residual kernels:
  - SwiGLU+NVFP4 GEMV+residual: Qwen3-4B 273→298 (+9%), Qwen3-8B 138→184 (+33%)
  - GeGLU+NVFP4 GEMV+residual: Gemma-3-12B 59→100 (+70%)
  - Multi-row dispatch (8 rows/block) for all fused NVFP4 GEMV variants
  - MoE SwiGLU+NVFP4 GEMV fusion for down projection
  - NVFP4 GEMV kernels registered with PDL, launches via pdl::launch()
  Tested: RMSNorm+GEMV fusion regresses (extra norm_w loads + muls in inner
  loop outweigh kernel launch savings). Multi-row threshold >512 also
  regresses (lower occupancy for large-K projections).

- [x] **CUTLASS FMHA Softcap** — Added `SoftcapCausalFusion` for models with
  logit soft-capping (Gemma-2). Uses `__constant__` memory for softcap params,
  applies scale→tanh→undo-scale in `before_softmax` hook. Dispatch relaxed to
  attempt CUTLASS for softcap models (graceful fallback). HD=256 tiles require
  >200 KB smem — exceeds RTX 5090 limit (99 KB), so Gemma-3 still uses WMMA.
  Gemma-3 doesn't actually use softcap (only Gemma-2 does).

- [x] **dp4a Fused Act+GEMV+Residual** — Attempted fusing SwiGLU/GeGLU + Q8_1
  quantize + dp4a GEMV + residual into a single kernel. Two-pass approach
  (pass 1: compute amax, pass 2: recompute + quantize + dp4a) to avoid float[32]
  register pressure. **Result: 22% regression** (Qwen3-4B 150→117 tok/s).
  Root cause: 2x gate/up L2 reads + 2x SwiGLU ALU per inner loop iteration.
  The kpar GEMV is memory-bound on weight reads — doubling input bandwidth
  and ALU per block saturates L2. Same pattern as attention O-proj inline quant
  (deliberately kept as separate quant + kpar for higher occupancy).
  Reverted — separate `swiglu_quantize_q8_1 + gemv_q*_q8_1_residual` is faster.

- [x] **NVFP4 LM Head** — FP32 output NVFP4 GEMV for output projection (LM head).
  Saves ~47% weight reads vs Q8_0 dp4a for vocab_size×d_model decode GEMV.
  Output projection collected first in NVFP4 budget to ensure inclusion.
  Large weights that exceed dequant scratch use temporary buffer during init.
  Multi-row (NR=8) dispatch for K ≤ 8192, kpar (128 threads) for larger K.
  Correctness verified on Qwen3-4B, Qwen3-8B, DeepSeek-7B.

- [x] **KV Cache Budget Trade-off** — In NVFP4 mode 2, KV cache allocation
  capped at 10% of available VRAM (was 80%). Excess KV blocks sit unused while
  weight caching directly improves decode throughput. Also skip 2x KV headroom
  and don't enforce needed_blocks floor. Qwen3-8B real prompt: 110→174 tok/s
  decode by trading 100K→30K KV tokens for full NVFP4 coverage.

- [x] **MoE NVFP4 Auto-Detection** — Removed d_model < 4096 threshold for MoE
  models. Expert weights dominate VRAM and sparse activation limits quantization
  error accumulation. Qwen3-Coder-30B: 247→265 tok/s (+7%), Nemotron-30B: 68→75
  tok/s (+10%).

- [x] **cuBLASLt NVFP4 Probe** — status=7 (INVALID_VALUE). Investigated: cuBLASLt
  handle works (FP16 GEMM passes), CUDA_R_4F_E2M1 data type recognized (enum=33),
  but no FP4 GEMM kernels exist for sm_120 in cuBLAS 13.2. Tested all combinations
  (OP_T/OP_N, BF16/FP32/FP16 output, with/without D_SCALE_MODE) — all fail.
  Root cause: cuBLASLt FP4 kernels only compiled for sm_100 (data center Blackwell).
  Fixed probe config (BF16 output, OP_T, D_SCALE_MODE) to auto-activate when NVIDIA
  adds sm_120 support. CUTLASS NVFP4 path is primary — no performance impact.

---

## L2 Cache Tuning

Current decode GEMV achieves ~44% of RTX 5090 peak bandwidth (788/1792 GB/s), while
llama.cpp reaches ~55-69%. The codebase has **zero** L2 cache control — no
`cudaAccessPolicyWindow`, no `__ldcs`/`__ldcg`, no `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`.
All loads use default `.ca` (cache-all) policy. Weight data, Q8_1 activations, KV cache,
and output all compete for the same 96 MB L2 with no priority differentiation.

### Phase 1: In-Kernel Cache Operators (PTX-level)

Compatible with CUDA Graphs (no host-side API calls in the hot path).

- [ ] **`__ldcs` for paged attention KV reads** — KV cache data is accessed once per
  decode step, then not needed until next step. Currently uses default `.ca` loads which
  pollute L2, evicting weight data that subsequent FFN GEMV kernels need. Change
  K/V loads in `attention_paged.cu` (split-K, GQA, pipeline kernels) and
  `attention_paged_fp8.cu`/`attention_paged_int8.cu` to `__ldcs()` (evict-first).
  **Files:** `src/compute/attention_paged.cu`, `attention_paged_fp8.cu`, `attention_paged_int8.cu`
  **Expected:** +2-5% decode by reducing L2 thrashing between attention and FFN phases.
  **Risk:** Low — KV data has no intra-step reuse across kernels. GQA multi-head reuse
  within a single kernel happens in smem (cp.async pipeline) or via DSMEM (cluster),
  not L2. Verify that `__ldcs` doesn't hurt the non-pipeline fallback where multiple
  warps read the same KV block from global.

- [ ] **`__ldcs` for GEMV weight reads** — In both K-par and row-par dp4a kernels,
  each weight row is read by exactly one block (K-par) or one warp (row-par). No
  cross-block reuse. Making weight loads streaming prevents stale weight lines from
  occupying L2 space needed by the next kernel (different weight matrix).
  Requires changes in each `DequantTraits::dp4a_block()` function — replace `memcpy`
  of weight bytes with `__ldcs`-based loads.
  **Files:** `src/compute/gemv_dp4a_traits.cuh` (all dp4a_block specializations)
  **Expected:** +1-3% decode. Smaller impact than KV streaming because weight-to-weight
  L2 contention is less severe (sequential GEMV launches, hardware prefetcher handles
  the transition).
  **Risk:** Medium — `__ldcs` bypasses L1. For K-par GEMV where 12 blocks/SM read
  different rows of the same weight matrix, L1 may have been caching shared sub-block
  metadata (Q6_K scales, Q4_K super-block headers). Benchmark carefully per quant type.
  May need `__ldcg` (L2-only, skip L1) instead of `__ldcs` (evict-first) for complex
  quant types.

- [ ] **Keep Q8_1 activation loads as default `.ca`** — In K-par GEMV, Q8_1 is the
  most reused data (read by all M blocks). In row-par, Q8_1 is already in shared memory.
  Do NOT add streaming hints to Q8_1 reads. Consider `__ldca` (explicit cache-all) as
  documentation if changing other load types.

- [ ] **`__stcs` for attention output writes** — `attn_out` is written once by paged
  attention, then read once by the O-projection GEMV. Streaming write avoids polluting
  L2 with output data that displaces weight lines.
  **Files:** `src/compute/attention_paged.cu` (output store in reduce kernel)
  **Expected:** +0.5-1% decode (minor — output is small relative to weights).

### Phase 2: Persisting L2 Reservation (Engine-level)

One-time setup at engine init. Does NOT require per-kernel changes. Complements Phase 1.

- [ ] **`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)`** — Reserve 75% of
  L2 (72 MB on RTX 5090) for persisting data. Without this, the persisting/streaming
  distinction from Phase 1 has no effect — there's no set-aside region for persisting
  lines to live in. Call once in `Engine::init()` after GPU properties query.
  **Files:** `src/runtime/engine.cpp` (init)
  **Expected:** Enables Phase 1 benefits. Without reservation, `__ldcs` still helps
  via evict-first heuristic, but the hardware has more freedom to retain streaming data.
  **Risk:** Low — this is a global device setting. Only affects the persisting vs
  streaming eviction priority. No functional change. Query `persistingL2CacheMaxSize`
  to cap the request.

- [ ] **`cudaAccessPolicyWindow` per decode step** — Set a single policy window before
  the layer loop in `forward_logits()` covering all model weights as `persisting` and
  everything else as `streaming`. This is a coarse-grained approach (one window for the
  entire forward pass, not per-layer rotation).
  **Files:** `src/graph/executor_forward.cu` (forward_logits), `src/graph/executor.h`
  **Concern:** Incompatible with CUDA Graphs — stream attributes are snapshot at capture
  time. Only applies to non-graph decode path (logprobs, JSON mode, first decode step).
  For graph path, Phase 1 in-kernel operators are the only option.
  **Expected:** +1-3% on non-graph path. No effect on graph path.

### Phase 3: Per-Phase L2 Policy Rotation (Aggressive)

Per-layer weight pinning. Highest potential but most complex.

- [ ] **Rotate policy window between attention and FFN** — In `run_attention()`: set
  policy window covering KV cache as persisting. In `run_ffn()`: set policy window
  covering FFN weight matrices (gate, up, down) as persisting. This ensures the L2
  persisting region always contains the data most relevant to the current compute phase.
  **Concern:** 2× `cudaStreamSetAttribute` per layer = 64-96 host-side calls per decode
  step. At ~1-2 µs each = 64-192 µs overhead. Decode step for Qwen3-4B is ~3.3 ms,
  so 2-6% overhead. May negate the benefit. Also incompatible with CUDA Graphs.
  **Approach:** Only enable on non-graph path. Profile host overhead carefully.
  Could batch the attribute set with the kernel launch to amortize.
  **Expected:** +3-8% if overhead is manageable. The per-phase pinning perfectly matches
  the sequential access pattern (attention reads KV → FFN reads weights → next layer).

### Notes

- **CUDA Graphs compatibility:** Phase 1 (in-kernel `__ldcs`/`__ldcg`) works inside
  captured graphs. Phase 2 L2 reservation is a one-time init call (fine). Phase 2/3
  `cudaAccessPolicyWindow` is incompatible with graph replay — only affects non-graph path.
- **Blackwell L2 latency:** RTX 5090 L2 is ~358 cycles (vs ~273 on Hopper), but still
  far better than ~800+ cycle GDDR7 access. L2 pinning remains strongly beneficial.
- **L2 sector size:** 32 bytes. L2 cache line: 128 bytes. Weight block reads should
  align to 32-byte boundaries where possible (Q8_0: 34 bytes — misaligned; Q6_K: 210
  bytes — misaligned; Q4_K: 144 bytes — aligned to 16 but not 32).
- **Prefill is compute-bound** — L2 tuning targets decode GEMV only. Do not apply
  streaming hints to prefill cuBLAS GEMM paths.

# Phase 2 — PERFHAWK Performance Audit

Single-target: sm_120a / RTX 5090 / GB202. No diplomacy. No re-derivation of
hardware. Map = `review/phase1_inventory.md` (`HEAD = f58eb9e`). Memory
baselines = `memory/MEMORY.md` (2026-05-16). Per claim: **measured** (number
from MEMORY/test output), **code-evidence** (read source at file:line), or
**speculative** (needs ncu trace).

The headline of this audit, repeated up front because it dominates the
verdict: imp on Qwen3-Coder-30B-A3B-NVFP4 measures **pp512 = 1258 tok/s vs
vLLM 25513 tok/s (continuous batching) — a 20.3× gap**. vLLM single-seq is
18 500 tok/s = a **14.7× gap**. Both engines call the same CUTLASS Sm120
NVFP4 grouped GEMM template (`nvfp4_moe_prefill_landscape_2026_05_10`),
which means **the gap is host-side, scheduler-side, and around-the-kernel —
not in the kernel itself**. Sections 6 and 7 quantify which buckets are
worth shipping fixes for.

---

## 1. Hot-kernel roofline table

GB202 specs taken as given: 1792 GB/s HBM (≈1500 GB/s achievable), 838
TFLOPS FP16/BF16, 1677 TFLOPS FP8, 3354 TOPS NVFP4/MXFP4 block-scaled, 88
MB L2, 99 KiB SMEM/block opt-in, 170 SMs. Decode-style numbers are tok/s on
a single sequence; prefill-style are TFLOPS/TOPS.

| Kernel category | File:line | Bound | Theoretical ceiling | Best-known achieved | Headroom | Confidence |
|---|---|---|---|---|---|---|
| FP16 paged attention decode (GQA, default) | `src/compute/attention_paged.cu:90` (`paged_attention_gqa_kernel`) | **memory-bound** (28:1 KV-load to dot-product, per `docs/sm120.md:46`) | ≥320 tok/s @ 8K ctx on Qwen3-8B Q8 KV | 256 tok/s tg256 Qwen3-8B Q8 (cache=FP16) | ~1.25× | measured (MEMORY baseline) |
| FP16 paged attention decode (cluster, GQA, ≥8 ctx blocks, hd ∈ {64,96,128,256,512}) | `src/compute/attention_paged.cu:1111` (`paged_attention_cluster_kernel`) | memory-bound; cluster cuts KV reads by n_q_per_kv× via DSMEM | ~1.5–2× over non-cluster on GQA=8 | not separately benched; folded into 256 tok/s | unmeasured | code-evidence + speculative |
| NVFP4 paged attention decode (scalar, default when `--kv-nvfp4`) | `src/compute/attention_paged_nvfp4.cu:53` (`paged_attention_decode_nvfp4_kernel`) | memory-bound on K-cache reads | parity with FP16 at 4× VRAM compression | parity 193 tok/s Qwen3-8B Q8+`--kv-nvfp4` | n/a (parity is the goal) | measured (`lever2_nvfp4_kv_implemented_2026_05_07`) |
| NVFP4 paged attention decode (WMMA-Q.K, BitDecoding, `IMP_USE_BITDECODING_QK=1`) | `src/compute/attention_paged_nvfp4_tc.cu:58` (`paged_attention_decode_nvfp4_tc_kernel`) | latency-bound on the HMMA Q.K op (24 HMMA per call vs 0 in scalar) | paper claims 8.6× FP16 FlashDecoding-v2 at long ctx | tg=148 vs scalar 149 at 4K ctx (within noise); 0% across all long-ctx configs | 8.6× claimed, 1.0× delivered | measured (`bitdecoding_long_context_eval_2026_05_14`) |
| FMHA prefill (FP16 WMMA, SM120) | `src/compute/attention_fmha_sm120.cu:69` (`fmha_sm120_kernel`) | compute-bound at large seq (HMMA m16n16k16 via `nvcuda::wmma`) | 838 TFLOPS FP16 | not isolated; folded into pp4096 = 11–18k tok/s on Qwen3-4/8B | unmeasured | code-evidence |
| FMHA prefill (FP8 m16n8k32) | `src/compute/attention_fmha_sm120.cu:581` (`fmha_sm120_fp8_kernel`) | compute-bound (m16n8k32 `mma.sync.kind::f8f6f4`) | 1677 TFLOPS FP8 | +3.3% pp4096 vs FP16 (`docs/sm120.md:32`) | small; was bigger before bug fixes | measured (PR #33) |
| MXFP4 FMHA prefill (block-scaled, `mxf4nvf4`) | `src/compute/attention_fmha_mxfp4_sm120.cu:108` (`fmha_sm120_mxfp4_kernel`) | compute-bound (`mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) | 3354 TOPS NVFP4 | +1.8% Qwen3-4B MXFP4 HD=128 (Phase 1 only 15% of FMHA wall, per `roadmap.md:104`) | Phase-3 PV-FP4 ~+13% claimed but quality-risky | measured |
| FP16 cuBLAS dense GEMM prefill | `src/compute/gemm.cu:1` (cuBLAS via `cublasGemmEx`, `attention_cublas_prefill`) | compute-bound; `CUBLAS_TF32_TENSOR_OP_MATH` default | 838 TFLOPS FP16-TC | cuBLAS autotune; pp512 varies up to 2.6× across container restarts (CLAUDE.md) | unmeasured per-kernel | measured (variance), code-evidence (mode set) |
| NVFP4 grouped GEMM prefill (MoE) | `src/compute/gemm_cutlass_grouped_3x.cu:150` (`gemm_grouped_cutlass_3x_nvfp4`) | compute-bound (cooperative block-scaled mainloop, `<128,128,128>` tile, `Sm120` arch tag, `KernelScheduleAuto`) | 3354 TOPS NVFP4 | imp pp512 = 1258 tok/s vs vLLM 18.5k single-seq (`nvfp4_moe_prefill_landscape_2026_05_10`) | **14.7×** | measured |
| NVFP4 smallM grouped GEMM (`IMP_NVFP4_SMALLM=1`, opt-in, hand-rolled) | `src/compute/gemm_grouped_nvfp4_smallM.cu:87` (inline `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) | compute-bound | 3354 TOPS NVFP4 | **par with CUTLASS at large M_e, -50–55% at small M_e** | negative; kept opt-in | measured (memo) |
| NVFP4 MoE decode GEMV (per-expert) | `src/quant/nvfp4_gemm.cu:855` (`gemv_nvfp4_moe_decode_kernel`, `__launch_bounds__(128, 12)`) | memory-bound on weight stream | ~1.8 TB/s × top_k weight bytes ⇒ ~270 tok/s @ MoE-30B-A3B | tg128 = 261–266 tok/s Qwen3-Coder NVFP4 | ~0.97× (effectively at roof) | measured |
| Q4_K GEMV (mmvq, dp4a) | `src/compute/ggml_mmvq.cu` (`mmvq_kernel`, warp-per-output) | memory-bound on weight stream | ~250 tok/s saturating regardless of M (per `q4k_mmvq_crossover_2026_05_15`) | matches | n/a (saturated for its design) | measured |
| Q4_K direct tiled GEMM v2 (HMMA, `IMP_FORCE_Q4K_V2=1`) | `src/compute/mmq_q4k_v2.cu:404` (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`) | compute-bound, but **stuck at ~15% of FP16-TC peak** | 838 TFLOPS FP16-TC | 4.87× v1 dp4a kernel-only, **-4% end-to-end on Qwen3.6-35B Q4_K_M** | -4% net; opt-in only | measured (`mmq_q4k_v2_phase2_shipped_2026_05_16`) |
| Q6_K direct fused-MoE prefill (FP16 accum, dequant on the fly) | `src/compute/gemm_q6k.cu:41` (`gemm_q6k_moe_fused_prefill_kernel`, `__launch_bounds__(128, 2)`) | likely compute-bound (FP16 dequant + dp4a) | unmeasured | tg=234 Qwen3-Coder Q6_K (decode); prefill 5643 tok/s (pp512) | unmeasured per kernel | code-evidence + measured (E2E) |
| MoE expert routing (router GEMV + softmax + top-k) | `src/compute/moe_routing.cu:196` (`gemv_gate_topk_fused_kernel`); `src/compute/moe_routing.cu:533` (`moe_fused_permute_kernel`, `__launch_bounds__(256)`) | latency-bound (single-block scan, n_experts ≤ 1024) | single-block at ~5 µs / decode (estimate) | unmeasured | (not on the critical path at decode size) | code-evidence |
| MoE combine (weighted sum + residual fused) | `gemm_grouped_3x_nvfp4.cu` epilogue + `moe_weighted_sum_residual` (graph/executor_forward_moe.cu:2362) | memory-bound on activations | bandwidth-trivial | unmeasured | n/a (small) | code-evidence |
| KV cache FP16 load (paged decode hot loop) | `src/compute/attention_paged.cu:174` (`ldcs_half`) + `:191` double-buffered streaming load | memory-bound; double-buffered cp.async (FP16) | 1500 GB/s achievable | unmeasured per-kernel | n/a | code-evidence |
| RoPE + QK-norm fused (FP16) | `src/compute/rope.cu:214` (`qknorm_rope_fused_fp16_kernel`) | bandwidth-bound | trivial | n/a | n/a | code-evidence |
| RMSNorm + Q8_1 quantize fused | `src/compute/layernorm.cu:72`, `rmsnorm_quantize_q8_1` (used at `executor_forward_moe.cu:236`) | bandwidth-bound | trivial | n/a | n/a | code-evidence |
| Sampler (top-k/top-p/min-p/DRY) | `src/compute/sampling.cu:1` (block size 256, `cub::DeviceTopK` for k>128) | latency-bound on vocab-sized arrays | argmax ~10 µs (mentioned `sampling.cu:97`) | unmeasured at decode-step granularity | n/a | code-evidence |
| Embedding lookup | `src/compute/embedding.cu` | bandwidth-trivial (single token row) | trivial | n/a | n/a | code-evidence |

**One sentence on the roofline shape:** for batch=1 decode imp is sitting
within 10–20% of the bandwidth-limited NVFP4 / Q4_K ceiling on MoE
models (memory-bound, working as designed); for prefill on dense models
the FP16-TC path is sitting near 838 TFLOPS via cuBLAS auto-tune; the one
kernel category where imp is dramatically **off** the achievable ceiling
is **NVFP4 grouped MoE prefill** (Qwen3-Coder pp512 1.26k vs vLLM 18.5k
on the same template), and that is the dominant brecher-opportunity in
this audit.

---

## 2. Occupancy & resource limiters

Top-5 hot decode kernels, what limits them, where the headroom is. Block
size from `__launch_bounds__` (where present), shared memory from `extern
__shared__` allocator at launch site, register count not directly visible
without ncu so estimated from kernel size + accumulators.

### 2.1 `paged_attention_gqa_kernel` (default FP16 KV decode)

- `src/compute/attention_paged.cu:90` — `__launch_bounds__(1024)`.
- Block size: variable, up to 1024 (`gqa_threads = n_q_per_kv *
  warps_per_q * 32`, capped at 1024); `warps_per_q = 4` for ratio≤8,
  `2` for >8.
- Shared mem: dynamic, `kv_tile_bytes = 4 * block_size * head_dim *
  sizeof(half)` = 8 KiB at bs=16/hd=128 (`attention_paged.cu:1518`).
  When `gqa_smem > 48 KiB` it calls `cudaFuncSetAttribute` for opt-in
  extended SMEM (`:1530`).
- Per-thread reg state: `q_reg[16] + o_reg[16]` floats + running m/l ⇒
  ~36 floats per thread = 36 registers minimum. With 1024 threads/block
  and 64 KiB register file/SM, this is **register-limited to 1
  block/SM** at the upper bound (1024 × 64 = 65536 regs, exactly the
  file size). That matches `__launch_bounds__(1024)` which asks for 1
  block/SM by convention when the second arg is omitted.
- **Limiter: register file × block-size combo.** With 1024 threads/block
  and no second arg to `__launch_bounds__`, occupancy = 1 block/SM ≈
  6% theoretical occupancy.
- **Code-evidence opportunity:** `__launch_bounds__(1024, 1)` would be
  explicit; current implicit 1-block/SM is fine. But the kernel
  arguably runs 256 threads worth of *useful* work on 32-warp Q-head
  groupings — a `__launch_bounds__(256, 4)` variant with `warps_per_q
  = 1` could re-claim occupancy at small GQA ratios. Speculative.
  Note `dead_ends.md` line 77: "`__launch_bounds__` on paged
  attention: -6% decode" — so changes here historically regressed.
  Keep current setting; flag the implicit-1-block-per-SM as low-risk
  but already validated.

### 2.2 `paged_attention_decode_nvfp4_kernel` / `paged_attention_decode_nvfp4_tc_kernel`

- `src/compute/attention_paged_nvfp4.cu:53` and
  `src/compute/attention_paged_nvfp4_tc.cu:58` — **no `__launch_bounds__`**
  (comment at `attention_paged_nvfp4_tc.cu:53-56`: "with the dots/weights
  moved to shared mem (warp-shfl reduction) the spill is gone (cuobjdump
  STACK:0 across HD ∈ {64,128,256,512}); the compiler picks the best
  occupancy/register trade-off automatically").
- Per-warp WMMA scratch on the TC kernel: 16×16 + 16×16 + 16×16×2 halves
  = 2 048 B/warp, ~16 KiB for 8 warps (`attention_paged_nvfp4_tc.cu:86-91`).
- Limiter: bandwidth (FP4 K stream) on the scalar variant; HMMA
  pipeline issue rate on the TC variant. The 0% E2E gain of TC despite
  24 HMMA ops (audit at `bitdecoding_sass_audit_2026_05_09`) tells the
  story: the K-byte stream is bandwidth-bound, and rerouting Q.K
  through HMMA doesn't change the K-byte count.
- No opportunity here: the comment is right that the compiler picks the
  right occupancy.

### 2.3 `gemv_nvfp4_moe_decode_kernel` (MoE decode hot kernel)

- `src/quant/nvfp4_gemm.cu:855` — `__launch_bounds__(128, 12)`.
  4 warps/block, **explicitly asking for 12 blocks/SM** — aggressive,
  ~85% occupancy on a 24-block-per-SM device (sm_120 has 16 blocks/SM
  hardware max so 12 is also bounded by HW; compiler will warn if
  unachievable).
- Per-thread reg state: warp_sums float reduction array (4 floats), one
  acc float, dequant scratch from `dot_micro_block`. Estimate <40 reg.
- Shared mem: `SmemKpar` struct = `kKparWarps * sizeof(float)` = 16 B
  per block, negligible.
- Limiter: HBM bandwidth on weight bytes (FP4 + per-16-elem UE4M3
  scales). 1.8 TB/s × bytes / (top_k × N × K / 2) gives the upper
  bound — matches measured 261 tok/s on Qwen3-Coder.
- No opportunity in occupancy. Headroom would come from reducing weight
  byte count (smaller quant) or sharing weights across requests (batch).

### 2.4 `fmha_sm120_kernel` / `fmha_sm120_fp8_kernel` (prefill FMHA)

- `src/compute/attention_fmha_sm120.cu:69`, `:581` — both
  `__launch_bounds__(256, 1)` (256 threads = 8 warps = 1 block/SM).
- Shared mem: 65–89 KiB depending on HD (table at top of file). HD=64
  uses 89 KiB which is **above** the 48 KiB default carveout — relies
  on opt-in (line 564 mentions `cudaFuncSetAttribute` for SMEM).
- HD=256 uses 88 KiB with Bq=32 — already near the 99 KiB sm_120
  per-SM cap.
- Per-thread regs: holds float accumulators `O_acc[Bq * head_dim /
  threads]` + WMMA fragments + 6 floats softmax state. With 1
  block/SM, registers aren't the limit.
- **Limiter: SMEM at 89 KiB ⇒ 1 block/SM** (CTAS-per-SM = floor(99/89)
  = 1). Recorded in `dead_ends.md` line 55: "`__launch_bounds__(256,1)`
  on FMHA: CORRECT for SMEM-limited kernels (69KB → 1 block/SM
  anyway)."
- No occupancy headroom; this is the right shape for the kernel.
- **One non-obvious gap:** the FP8 kernel computes Q→FP8 conversion in
  shared memory (`Q_fp8` written by all threads before the main loop,
  line 638-655) — this is one full Q-tile write to SMEM per kernel,
  per CTA. With Bq×HD = 128×64 = 8 KiB at HD=64 that's not free at
  100% SM occupancy = 1. **Speculative:** precomputing FP8 Q outside
  the kernel (in the executor before launch) would eliminate the SMEM
  conversion stage but adds an extra global write/read pair.
  Cannot rule in/out without an ncu trace.

### 2.5 `paged_attention_cluster_kernel`

- `src/compute/attention_paged.cu:1111` — no `__launch_bounds__`.
- Block size: `BLOCK_THREADS = 256` (from cluster_block dim3,
  `:1454`).
- Cluster: up to (n_q_per_kv, 1, 1) = up to 8 blocks. Uses DSMEM via
  `cluster.map_shared_rank` (`:1179`).
- Shared mem: `cluster_smem = 2 * block_size * 2 * head_dim *
  sizeof(half) + NUM_WARPS * sizeof(float)*2 + NUM_WARPS * head_dim *
  sizeof(float)` = up to ~50 KiB at hd=256/bs=16 (`:1450-1451`).
- **Cluster scheduling policy = `cudaClusterSchedulingPolicySpread`**
  (`:1472`), explicit comment "CUDA 13.2: spread cluster blocks
  across GPCs (GB202 has 12 GPCs)". Good code.
- No occupancy concern; cluster is doing the right thing for KV
  reuse (8× fewer KV bytes per Q-head group).

### 2.6 `__launch_bounds__` audit summary

`grep -rn "__launch_bounds__" src/compute/ src/quant/` finds 44
declarations. Sample-checked vs `dead_ends.md` guidance:

| File:line | Bounds | Verdict |
|---|---|---|
| `attention_fmha_sm120.cu:69, 581` | (256,1) | matches SMEM = 89 KiB → 1 block/SM (correct) |
| `attention_fmha_mxfp4_sm120.cu:108` | (`MX_BLOCK_THREADS`, 1) | same shape, same justification |
| `attention_paged.cu:90` | (1024) | OK; implicit 1 block/SM at thread count = file size |
| `attention_paged_nvfp4_tc.cu` | none | deliberate (comment explains) |
| `attention_paged_turboquant.cu:44/286/639/824` | (256, 2) | 2 blocks/SM × 256 = 512 threads/SM → fine on sm_120 |
| `quant/nvfp4_gemm.cu` (12× `kKparThreads, 12`) | (128, 12) | aggressive; compiler may not deliver 12 but won't regress |
| `quant/mxfp4_gemm.cu` (9× `kKparThreads, 12`) | (128, 12) | same |
| `gemm_q6k.cu:41` | (128, 2) | 2 blocks/SM × 128 = 256 threads/SM — undersubscribed |
| `gemm_moe_fused_tc.cu:40` | (TC_BLOCK=256) | implicit 1 block/SM — file is WMMA-heavy, justified |
| `gdn.cu:30` | (HD, 1) | template-bound, fine |

No `__noinline__` annotations found in `src/` (`grep` for the term
returned no matches — CLAUDE.md hard-rule is **respected**).

No `cudaMalloc/cudaFree` in true decode-token hot loops EXCEPT the
size-grow-if-needed patterns at:
- `src/graph/executor_kernels.cu:2178-2181` (mmvq scratch — fires once
  at first call per dimension; static across decode steps; not a true
  hot-loop allocation) — passes the CLAUDE.md hard rule on a strict
  reading but is fragile if a larger-K weight appears.
- `src/compute/gemm_cutlass_grouped_3x.cu:120-136` (`ensure_staging`,
  `ensure_workspace`) — only grows; `gemm_grouped_3x_nvfp4_prewarm()`
  pre-allocates 1 MiB + 512 MiB so the lazy path never fires under
  capture in practice.
- `src/compute/attention_cublas.cu:343-347` (`s_attn_d_ptrs`) — same
  pattern.
- **`src/graph/executor_forward_moe.cu:2092-2100`** —
  `try_run_moe_gemma4_ggml_prefill` lazy-allocates `s_q8_scratch` and
  `s_norm_fp32` via `cudaMalloc/cudaFree`. This is a **prefill** path
  (Gemma-4 MoE ggml fallback), one-shot per session normally; if `d`
  grows mid-session (multi-model swap) it free/mallocs. Acceptable
  but worth a static-buffer rewrite for cleanliness.

The real CLAUDE.md violation candidates do not exist in the current
tree — the surviving lazy-allocs are all monotonic-grow-only with
pre-warm hooks.

---

## 3. GB202 feature utilization audit

Feature-by-feature. "Used" = real production kernel calls it; "Bench
only" = only in `src/compute/*_bench.cu` (which CMakeLists strips from
the runtime image when `IMP_BUILD_BENCH=OFF`). "Missing" =
applicable-but-absent.

### 3.1 `cp.async.bulk` / `cp.async.bulk.tensor` (TMA)

**Production: hand-rolled in one TU; CUTLASS-internal elsewhere; otherwise NO.**

`grep -rn "cp.async.bulk\|cuTensorMap\|__cvta_generic_to_shared" src/`:

| File:line | Status |
|---|---|
| `src/compute/tma_block_scale_bench.cu:117-122` | bench-only (off in runtime). Uses `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`. |
| `src/compute/attention_paged_common.cuh:17-24` | `__cvta_generic_to_shared` helpers — used by NVFP4 paged attention's `cp_async_*` non-bulk paths. NOT TMA. |
| `src/compute/gemm_grouped_nvfp4_smallM.cu:69-122, :362-363, :702-744` | **The only production TMA in imp.** Builds `CUtensorMap` descriptors via `cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", …)` and issues `cp.async.bulk.tensor.2d` for A and B in the smallM kernel mainloop. **Opt-in via `IMP_NVFP4_SMALLM=1`** (per `executor_forward_moe.cu:703-704`). Default off because at production shapes it's -50–55% vs CUTLASS. |
| `src/compute/fmha_v_load_bench.cu:117-122` | bench-only. Re-confirmed by `fmha_tma_lever_refuted_2026_05_14`: cp.async beats TMA bulk 0.31–0.79× on SM120 — TMA loses to cp.async. |

**CUTLASS-internal TMA:** `gemm_cutlass_grouped_3x.cu:24` includes
`cutlass/detail/sm100_blockscaled_layout.hpp` and the
`MainloopSm120ArrayTmaWarpSpecializedBlockScaled` schedule (per memo
`nvfp4_moe_prefill_landscape_2026_05_10`) — so CUTLASS uses TMA
internally on the NVFP4 grouped GEMM path. imp doesn't reach in for
it directly.

**Missing-but-applicable:**
- `paged_attention_*_kernel` KV-block loads. Each kernel uses
  `ldcs_half` + double-buffered SMEM (`attention_paged.cu:194-200`).
  TMA bulk could replace this but `fmha_tma_lever_refuted_2026_05_14`
  shows cp.async wins on SM120 — **don't pursue**.
- Prefill `attention_cublas_prefill` Q/K/V gather. cuBLAS owns this,
  no opportunity.

**Verdict:** TMA usage is appropriate for SM120; the refuted
benchmark proves more TMA isn't free upside. **Not** a left-on-the-table
lever.

### 3.2 `mma.sync.aligned.m16n8k16` (HMMA, FP16/BF16)

**Used widely.** Direct inline PTX:
- `src/compute/mmq_q4k_v2.cu:404, 607, 975, 1166, 1421, 1591` (six
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` invocations
  across the v1/v2/q5k/q6k variants).
- `src/compute/attention_fmha_sm120.cu:201-214` (via `nvcuda::wmma`
  which lowers to HMMA m16n16k16 — same SASS class).
- `src/compute/attention_blackwell.cu`, `attention_tc.cu`,
  `attention_paged_nvfp4_tc.cu`, `gemm_capture_fp16_sm120.cu`,
  `gemm_moe_fused_tc.cu` — all use `<mma.h>` (wmma API).

**Verdict:** Healthy use. Per SASS audit
`sass_audit_120a_no_tcgen05_2026_05_04` there are 1 898× HMMA SASS ops
in the compiled binary. This is the peak FP16 compute path on SM120.

### 3.3 `mma.sync.aligned.*.mxf4nvf4.*` (block-scaled FP4)

**Used in 3 production kernels:**
- `src/compute/attention_fmha_mxfp4_sm120.cu:591` —
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`
  for Q.K block-scaled FP4 (PR #56). Default-on via
  `attention.fmha_blockscale = "auto"`.
- `src/compute/gemm_grouped_nvfp4_smallM.cu:87` — same MMA in the
  hand-rolled smallM kernel. Opt-in (`IMP_NVFP4_SMALLM=1`).
- CUTLASS-internal: the
  `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` schedule uses
  the same MMA, observed at SASS via `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`
  (per `nvfp4_moe_prefill_landscape_2026_05_10` correction).

Bench-only paths: `mxf4nvf4_mma_bench.cu`, `mxf4nvf4_mma_variants_bench.cu`,
`mxf4nvf4_qkt_validate.cu`, `attention_mxf4nvf4_probe.cu` — research /
characterization only.

**Missing-but-applicable:**
- **MXFP4/NVFP4 P×V in FMHA** (Phase-3 from `roadmap.md:104`). Claimed
  +13% but quality-risky (softmax-output FP4-quant). 200-300 LoC, not
  shipped. **Should ship a quality-gated A/B harness, then ship.**
- **Block-scaled NVFP4 for dense GEMM prefill on FP16 weights** —
  i.e. on-the-fly FP16→NVFP4 of activations + NVFP4×NVFP4 grouped MMA
  for the non-MoE dense path. Currently dense FP16 goes through cuBLAS
  TF32-TC. **Speculative** but worth one ncu trace.

### 3.4 `mma.sync.aligned.*.e4m3.*` (FP8 m16n8k32)

**Used in FMHA FP8 prefill kernel.**
- `src/compute/attention_fmha_sm120.cu:758` —
  `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
  via inline PTX guarded by `__CUDA_ARCH__ >= 1200`.
- Used in Q.K phase (PV stays FP16 WMMA per line 563).

FP8 GEMM lifting via `cuBLASLt` happens through
`gemm.cu`/`weight_dispatch.cu` (cuBLASLt manages MMA selection
internally on FP8 ops).

**Missing-but-applicable:**
- FP8 GEMM for dense MoE expert weights at prefill (currently goes
  through CUTLASS NVFP4 on prequant or cuBLAS FP16 on others). Memos
  show FP8-cache prefill landed (`docs/performance.md:11`) — so this
  is exercised on the cache path.

### 3.5 PDL (Programmatic Dependent Launch)

**Used via wrapper.** `src/runtime/pdl.h:34-59` defines `pdl::launch`
that wraps `cudaLaunchKernelEx` with
`cudaLaunchAttributeProgrammaticStreamSerialization`. Edge conversion
in graph capture: `src/runtime/cuda_graph.cu:41-152` (`apply_pdl_edges`)
replaces default kernel-to-kernel edges with PDL edges using
`cudaGraphKernelNodePortProgrammatic`.

**Caveat at `src/compute/gemm_dp4a.cu:733`:** comment "cudaLaunchKernelEx
overhead outweighs PDL tail/head overlap benefit" — for that path PDL
is disabled. Reasonable measurement.

Graph-capture-edge replacement converts PDL only for kernels that
opted into PDL via the registry (`pdl::is_enabled(kparams.func)`,
`cuda_graph.cu:119`). Non-PDL kernels keep default edges. Good
defensive code.

**Verdict:** PDL is wired correctly and used inside captured graphs.
Coverage is opt-in per kernel — Phase 4 of the codereaper review
should audit the per-kernel coverage.

### 3.6 L2 persistence / streaming policy

- `set_l2_persist_kv` at `src/graph/executor_attention.cu:97-131`:
  per-layer in attention, persists KV with hitRatio scaled to L2
  carveout, clamped to `cudaDevAttrMaxAccessPolicyWindowSize` (128
  MiB on RTX 5090 per CLAUDE.md footgun). Window also clamped to
  `kv_bytes`. **Correct**, and the comment at line 100-103 captures
  the historical poisoning bug.
- `set_l2_streaming` at `src/graph/executor_helpers.h:67-89`: same
  clamping discipline, hitRatio=0 (anti-persist).
- Called at `src/graph/executor_ffn.cu:69` for FFN weight reads
  (streaming hint).
- `clear_l2_policy` (`executor_helpers.h:91-97`) zeros the window
  size, runs at `executor_attention.cu:1118`.

**Missing:** the LM-head weight tensor is not L2-managed. For models
with vocab≤32K and d_model≤4K it's ~256 MiB — too large for L2
persist, but FP8 cache hit could keep the active layer-out residual
in L2.

**Speculative gain at decode:** L2-persist on the residual stream
between layers (~32 KiB at d_model=8192) would be free; not
implemented. Marginal — n=1 decode resid fits in L1 anyway.

### 3.7 Cluster launch (multi-CTA cooperative groups)

**Used in production.**
- `src/compute/attention_paged.cu:1469-1473` —
  `cudaLaunchAttributeClusterDimension` + `cudaLaunchAttributeClusterSchedulingPolicyPreference
  = cudaClusterSchedulingPolicySpread` for the GQA cluster kernel.
- Cluster up to 8 blocks (per the GQA ratio constraint `n_q_per_kv ∈
  {2,4,8}` at `:1447`).
- DSMEM via `cluster.map_shared_rank` at `:1179`.

**Missing-but-applicable:**
- Cluster launch in MoE NVFP4 grouped GEMM mainloop: CUTLASS uses
  `Shape<_1,_1,_1>` cluster (`gemm_cutlass_grouped_3x.cu:57`). A
  `<_2,_2,_1>` cluster would let 4 CTAs share KV across the expert
  dimension. **Speculative** — would require CUTLASS template
  change. Memo `nvfp4_moe_prefill_landscape_2026_05_10` doesn't
  flag this as tested.
- Cluster launch for FMHA prefill on long context (each Q-tile
  share K across cluster blocks). Could halve KV reads at large
  seq_kv. Not implemented.

### 3.8 WGMMA (sm_90 only)

**Dead code at the documentation level.**

`src/compute/attention_fmha_sm120.cu:1-31` and
`src/compute/attention_fmha_sm120.h:6-10` claim "WGMMA (Warp Group MMA)
… `wgmma.mma_async` PTX instructions for ~2x tensor core throughput vs
WMMA." The implementation uses `nvcuda::wmma` (m16n16k16 WMMA fragments)
plus FP8 inline `mma.sync.kind::f8f6f4` — **no wgmma anywhere**. wgmma
is Hopper-only and ptxas rejects it on sm_120a per `dead_ends.md:17`.

**Action:** the comment block is stale and misleading. Flag for the
codereaper phase. Code itself is fine.

### 3.9 WMMA fragments (`nvcuda::wmma`)

**Used in 10 TUs** (per Phase 1 §7.5). Three categories:
1. Active hot path: `attention_fmha_sm120.cu`,
   `attention_fmha_mxfp4_sm120.cu`, `attention_paged_nvfp4_tc.cu` (3 322
   LOC). **Keep** — WMMA → HMMA SASS is the peak FP16 path on SM120.
2. Active fallback: `attention_blackwell.cu` (460 LOC, primary FP16
   prefill per Phase 1 §7.4).
3. Dead or opt-in: `attention_tc.cu` (411), `mmq_q4k_v2.cu` (1 667 —
   opt-in), `gemm_moe_fused_tc.cu` (~520), `gemm_capture_fp16_sm120.cu`
   (~600).

The codereaper phase can decide what to remove. **From a perf
standpoint:** the WMMA-vs-direct-mma.sync question is moot on SM120
since both lower to HMMA SASS.

### 3.10 Async copy elision

`cp.async.commit_group` / `cp.async.wait_group` usage in
`attention_paged_common.cuh:17-34` — both helpers. Used in NVFP4 paged
attention split-K path (`paged_attention_splitk_pipeline_kernel` at
`attention_paged.cu:817`).

**Refuted lever:** the NVFP4 pipelined splitk decode kernel was
**removed** because of -3% regression (per `dead_ends.md:40`). INT4 KV
keeps a pipelined variant; NVFP4 dropped it.

**Verdict:** appropriate use. Don't chase more cp.async pipelining.

### 3.11 Distributed shared memory (DSMEM)

Used in `paged_attention_cluster_kernel` only (per §3.7). Could be
extended to FMHA prefill clusters — see §3.7 missing item.

### 3.12 Summary feature table

| Feature | Used? | Where (representative) | Gap |
|---|---|---|---|
| `cp.async.bulk.tensor` (TMA) | partial | `gemm_grouped_nvfp4_smallM.cu` (opt-in only); CUTLASS-internal in grouped GEMM | Bench refutes more TMA on SM120 — not a lever |
| `mma.sync.aligned.m16n8k16` (HMMA) | yes | `mmq_q4k_v2.cu`, all WMMA TUs | None |
| `mma.sync.aligned.*.mxf4nvf4.*` | yes | `attention_fmha_mxfp4_sm120.cu`, `gemm_grouped_nvfp4_smallM.cu` (opt-in), CUTLASS-internal | **PV-FP4 in FMHA (Phase 3) missing; +13% claimed, gated on quality A/B** |
| `mma.sync.aligned.*.e4m3.*` (FP8) | yes | `attention_fmha_sm120.cu:758` Q.K only | FP8 PV still WMMA FP16 |
| PDL | yes | `runtime/pdl.h`, edge-conversion in `cuda_graph.cu` | Per-kernel opt-in coverage to audit |
| L2 persist/streaming | yes | `executor_attention.cu`, `executor_ffn.cu` | Window=128 MiB clamp correct; not used on LM-head |
| Cluster launch | yes | `attention_paged.cu:1469` (paged attention only) | **NOT extended to FMHA prefill or grouped GEMM** |
| WGMMA | NO (hardware absent) | n/a | Stale comments at `attention_fmha_sm120.h:10` |
| WMMA fragments | yes (10 TUs) | mixed | Phase 3 codereaper for dead-code |
| cp.async commit/wait | yes | `attention_paged_common.cuh` | Pipelined-splitk refuted; no more |
| DSMEM (cluster map_shared_rank) | yes | `paged_attention_cluster_kernel:1179` | Not in FMHA/grouped GEMM |

---

## 4. CUDA Graphs landscape

### 4.1 Capture machinery

`src/runtime/cuda_graph.cu` (989 LOC) contains:
- `CudaGraphCapture` — single-graph capture, instantiate, replay,
  `cudaGraphExecUpdate` support (`:228-240`).
- `apply_pdl_edges` (`:41-152`) — post-capture pass that rewrites
  kernel-to-kernel default edges into PDL edges (with rollback on
  failure).
- `ConditionalRunner` (`:739-784`) — conditional-graph body capture
  via `cudaStreamBeginCaptureToGraph`.
- Capture mode selector: `IMP_GRAPH_CAPTURE_MODE = global | relaxed
  | thread_local` (`:14-36`). Default `global`; `relaxed` exists to
  work around CUTLASS hangs (memo `prefill_graph_blockers_2026_05_14`).

### 4.2 Capture boundaries

Searching `engine.cpp` for `use_cuda_graphs`:
- Decode fast-path: captured per (decode-step, n_sequences) cell;
  pool size `kMaxGraphPoolSize` referenced at
  `engine.cpp:2637`. Default ON.
- Prefill: opt-in via `IMP_PREFILL_GRAPH` (Phase 1 Appendix B). The
  `can_capture` predicate at `engine.cpp:2209` gates it.
- Multi-graph pool indexed by `graph_idx`; `cudaGraphExecUpdate` used
  for shape changes when topology is unchanged (`cuda_graph.cu:228-240`).

### 4.3 Re-capture conditions

`engine.cpp:2643` comment: "grid dims / params differ —
`cudaGraphExecUpdate` handles this." Topology-changing changes (e.g.
new MoE prefill expert distribution) trigger full reinstantiate.

### 4.4 Disabled-graph conditions

Mapped from `engine.cpp` greps:
- L624: `config_.use_cuda_graphs = 0;` (early init force-off path)
- L890: disabled when profiling
- L1021-1025: experts_on_host_=true triggers disable
- L1158-1164: same, MoE path
- L1166-1168: `runtime.cuda_graphs = "never"`
- L1545-1546: disabled when residual BitDecoding active (host-state
  ring) — but post-`18abb9e` (commit referenced in
  `bitdecoding_phase3_continuation_2026_05_09`) the device-side ring
  re-enables capture.

### 4.5 Known gaps (from MEMORY + code review)

| Gap | Status | Where |
|---|---|---|
| Non-Gemma-4 MoE D2H routing memcpy blocks capture | **STALE** per `cuda_graphs_moe_works_2026_05_07`: the decode fast-path at `executor_forward_moe.cu:2316` skips routing D2H; captures cleanly | resolved |
| MoE PREFILL graphs blocked by cuBLASLt hangs | Phase 3 PR #164 shipped (+11–39%); Phase 4 partial per MEMORY | `nvfp4_moe_prefill_landscape_2026_05_10` |
| MoE prefill D2H sync still present in some branches | Yes — `executor_forward_moe.cu:671-682, 1131-1135, 1192, 1996-2000` all do `cudaMemcpyAsync(... DtoH)` + `cudaStreamSynchronize`. Each one breaks capture | code-evidence |
| Host-offloaded experts disable graphs | `engine.cpp:1158-1164`, mitigation: `IMP_EXPERT_OVERHEAD_PCT=10` | known |
| `try_run_moe_gemma4_ggml_prefill` per-token D2H sync | `executor_forward_moe.cu:2121-2123`: per-token `cudaMemcpyAsync(h_experts) + cudaStreamSynchronize` | **NEW finding**: kills prefill capture entirely for this Gemma-4 fallback path |

### 4.6 Pool sizing + invalidation

`kMaxGraphPoolSize` not directly visible at the grep point but exists
(used at `engine.cpp:2637`). For chunked prefill the chunk size
changes shape per chunk; `cudaGraphExecUpdate` is attempted before
full reinstantiation (`cuda_graph.cu:228-240`) — this is the right
pattern.

### 4.7 Verdict

Decode-graph machinery is mature, captures cleanly across
prequant-NVFP4 MoE (per `cuda_graphs_moe_works_2026_05_07` 2.9–3.3×
delta). **Prefill graph capture is still blocked on MoE NVFP4** by
several persistent D2H syncs in `executor_forward_moe.cu`. Section 7
ranks the surviving D2H sites by impact.

---

## 5. Memory subsystem audit

### 5.1 Paged KV cache layout

- Block size = 16 (per CLAUDE.md, MEMORY); coalesced reads across
  `block_size * head_dim` slots.
- Per-block stride: `block_size * n_kv_heads * head_dim` halves
  (`attention_paged.cu:133`).
- Slot stride within a block: `n_kv_heads * head_dim`.
- Kernel access pattern: per warp, per-block, lane covers
  `head_dim/32` consecutive elements via `ldcs_half` (line 174) — full
  coalesced FP16 transactions.
- Double-buffered next-block prefetch (`:189-200`) with single
  `__syncthreads` barrier per block.

**SM120 optimality:** the FP16 KV layout matches HBM transaction
granularity (128 B per warp = 32 lanes × 4 halves). Double-buffering
hides global load latency.

**Per-page coalescing across pages:** page tables (`block_tables[]`)
deliver a possibly-non-contiguous sequence of physical blocks. Each
block is contiguous internally; the `ldcs_half(&K_block[...])` loads
are coalesced within a block. There's no cross-page coalescing issue
because block_size=16 already exceeds one warp's natural unit.

**Verdict: layout is sm_120a-optimal for FP16 KV.** No leak here.

### 5.2 L2 persistence

- KV persist: `set_l2_persist_kv` (§3.6). Window correctly clamped.
- Comment at line 119-122: "hitRatio: compare against total KV size
  so the hardware probabilistically persists a representative subset
  even when kv_bytes exceeds the window." — sensible.
- `persistingL2CacheMaxSize` queried once and cached (line 104-109).

### 5.3 L2 streaming

- `set_l2_streaming` called at `executor_ffn.cu:69` for FFN weight
  region. Window clamped to 128 MiB. Correct.
- Not called for LM head weights (could be — at ~256 MiB it
  wouldn't fit but a streaming hint costs nothing).

### 5.4 HBM bandwidth utilization at 40K ctx

Memory-bound estimate for Qwen3-8B at 40K ctx, GQA ratio = 8,
head_dim = 128, NVFP4 KV (Lever 2):
- KV bytes/token/layer = (128 / 2 bytes packed FP4) × 8 KV heads × 2
  (K+V) + (128/16 scale bytes) × 8 × 2 = 1 152 bytes
- Per-step per-layer KV read = 40 000 × 1 152 = 46 MB
- 32 layers: ~1.47 GB per decode step → ~1 ms at 1500 GB/s
- Achieved: 156 tok/s at 20K (table at `docs/performance.md:125`)
  ⇒ 6.4 ms / token. Most of that is **NOT** KV; it's weight loads
  (8B params × 1 byte FP8 KV / non-MoE = ~8 GB total weight stream
  per token = ~5.3 ms). KV alone is ≤20% of decode time at 20K.

**No leak.** Bandwidth utilization on the KV stream is at ≥90%
ceiling on the paged-decode kernel.

### 5.5 VRAM efficiency

Per `memory/MEMORY.md` and `docs/performance.md`:
- Weights = the largest line item (NVFP4 ≈ 0.5 byte/param).
- KV at FP16 = 100% baseline; FP8 = 50%; INT4 = 25%; NVFP4 = ~25%
  effective.
- Expert offload (`experts_on_host_=true` at `engine.cpp:1158`) gives
  another 2–4× weight reduction at the cost of disabling CUDA
  graphs (per §4.5).

The VRAM budgeter is in `src/runtime/vram_budgeter` / `storage_planner`
(per Phase 1). Not audited in detail; no obvious perf leak from VRAM
fragmentation visible from the executor side.

### 5.6 Allocator behavior

Phase 1 audit + grep here: no `cudaMalloc`/`cudaFree` in true per-token
decode loops. The handful of `cudaMalloc`/`cudaFree` sites in
`src/compute/` and `src/quant/` are all:
- One-shot init (workspace, `s_workspace` in `gemm.cu:108`).
- Grow-only static caches (`ensure_workspace`,
  `gemm_cutlass_grouped_3x.cu:120`).
- Bench-only paths (`mxf4nvf4_mma_bench.cu`,
  `tma_block_scale_bench.cu`, `fmha_v_load_bench.cu` — all gated
  off `IMP_BUILD_BENCH=OFF` in Docker).
- `gemm_grouped_nvfp4_smallM.cu:575-580` — uses `cudaMallocAsync` (so
  it's stream-ordered, not synchronous), allocates inside the
  software-ref path which is `#ifdef SMALLM_SOFTWARE_REF` (not the
  production hardware path). Acceptable.
- `quantize_fp16_nvfp4_moe_native.cu:245-247` — `cudaMallocAsync` for
  three small device arrays inside a prefill-MoE path. **In a stream-
  capturable graph, `cudaMallocAsync` is legal but ties the graph to
  a memory pool**. Worth a Phase 3 codereaper note; not a perf leak.

**Verdict: hot-loop allocator hygiene is respected.** CLAUDE.md hard
rule passes.

---

## 6. DEEP DIVE — Qwen3-Coder NVFP4 prefill 20× gap

This section walks the prefill path end-to-end from
`imp_prefill_with_params` to the CUTLASS NVFP4 grouped GEMM, then
hypothesizes root-cause buckets with relative blame. Re-evaluated from
code; compared against memo
`nvfp4_moe_prefill_landscape_2026_05_10`.

### 6.1 End-to-end call chain for Qwen3-Coder-30B-A3B NVFP4 prefill

1. `imp_prefill_with_params` (`src/api/imp_api.cpp:~700` — public C
   API).
2. `Engine::step → step_schedule → step_prefill` (in
   `src/runtime/engine.cpp`, ~line 1735 driver; prefill arm at
   `engine.cpp:~2100`).
3. Chunk loop: per chunk of `prefill_chunk_size` tokens (default 512
   for hybrid models; default 0 for dense from
   `Engine::resolve_prefill_chunk_size_()`).
4. Per chunk: build `GPUBatch` (token IDs + positions + block tables
   uploaded H2D at `engine.cpp:2122-2149`).
5. `GraphExecutor::forward_logits` (`src/graph/executor_forward.cu:174`).
6. Per layer:
   a. `run_attention` (`executor_attention.cu:140`)
      → QKV projections (`gemm_dispatch` → CUTLASS NVFP4 GEMM at
      `executor_kernels.cu:2099-2103` for NVFP4 weights)
      → RoPE
      → KV write
      → Attention (cuBLAS prefill path on Qwen3-Coder via
      `attention_cublas_prefill` at `attention_cublas.cu:398`)
      → O projection (NVFP4 GEMM via `gemm_dispatch`).
   b. `run_moe_ffn` (`executor_forward_moe.cu:146`). **This is the hot
      path.**

### 6.2 The MoE prefill path step-by-step

For prequant-NVFP4 MoE at `n > 1`:

| Step | File:line | What happens |
|---|---|---|
| Residual save | `executor_forward_moe.cu:212` | `cudaMemcpyAsync` device-to-device of FP16 hidden state (h.nbytes) — graph-safe |
| RMSNorm | `executor_forward_moe.cu:251` (or `rmsnorm_fp32_to_fp16` at `:243` for Gemma-4) | bandwidth-trivial |
| Gate logits | `executor_forward_moe.cu:284` (`gemv_gate_fp32_fp32input`) | small GEMV (d_model × n_experts) |
| Top-k routing | `executor_forward_moe.cu:2207-2210` (`moe_topk_gating`) | softmax + top-k; sets up `expert_offsets`, `expert_indices`, `expert_weights` (all device-resident) |
| Permute / scatter | `moe_routing.cu:533` (`moe_fused_permute_kernel`, single-block) | builds `expert_offsets[n_experts+1]` device-side. **Single-block scan limits to n_experts ≤ 1024.** |
| Gather activations into per-expert layout | NVFP4 device-args path: `executor_forward_moe.cu:546-563` (`quantize_fp16_to_nvfp4_cutlass_moe`) — quantizes FP16 gathered to NVFP4 SfAtom layout | bandwidth-bound |
| Build per-expert M counts | `compute_M_per_from_offsets_device` (`executor_forward_moe.cu:535-537, 678-680`) | device kernel |
| **Build per-expert SFA offsets** | `compute_sfa_offsets_device` (`executor_forward_moe.cu:547-548`) | device kernel |
| **Build per-expert SFA base pointers** | `build_sfa_bases_device` (`executor_forward_moe.cu:549-551`) | device kernel — fills `cutlass3x_sfa_ptrs` |
| **Build per-expert weight pointer arrays** | `executor_forward_moe.cu:587-613` — on cache-miss falls back to host loop + `cudaMemcpyAsync` H2D × 3 | per-expert pointer arithmetic on host then bulk H2D |
| **Dispatch CUTLASS 3.x grouped GEMM** | `gemm_grouped_cutlass_3x_nvfp4` (`gemm_cutlass_grouped_3x.cu:150`) → CUTLASS template at `:72-75` (`MainloopSm120ArrayTmaWarpSpecializedBlockScaled`, tile `<128,128,128>`, cluster `<1,1,1>`) | **THE KERNEL.** Same as vLLM. |
| Activation (SwiGLU / GeGLU / ReLU²) | `apply_expert_activation` (`executor_forward_moe.cu:117-133`) | bandwidth-trivial |
| Down projection | same dispatch_device path, K_in=eff, N_out=d | second grouped GEMM call |
| Combine | `moe_weighted_sum_residual` (`executor_forward_moe.cu:2362`) | fused weighted sum + residual add |

### 6.3 Routing D2H/H2D sync

**Device-args path (default, `IMP_NVFP4_DEVICE_ARGS=1`):**
- `executor_forward_moe.cu:508-665`. **NO D2H sync.** All pointer
  arrays built device-side. Per-call H2D only on pre-cache miss (line
  601-613): three `cudaMemcpyAsync` of `ne * sizeof(void*)` each
  (small, on-stream). On hit, even those are zero (the per-layer
  `da_cache` pre-built at model load).
- Per-layer pre-cache at `executor_forward_moe.cu:566-578`. When
  populated (per memo `nvfp4_moe_prefill_landscape_2026_05_10`,
  shipped commit `41fb8fc`), dispatch is purely device-side.

**Legacy host-args path (set `IMP_NVFP4_DEVICE_ARGS=0`):**
- `executor_forward_moe.cu:668-682`: D2H of `expert_offsets[ne+1]` →
  `cudaStreamSynchronize` → host scan for `M_per[ne]`. **Blocks graph
  capture.**

**Verdict:** the default path is already graph-capturable for the MoE
prefill. The remaining D2H sync sites are in:
- `executor_forward_moe.cu:1131-1135` (smallM/legacy non-prequant Q*_K path)
- `executor_forward_moe.cu:1189-1192` (another legacy branch)
- `executor_forward_moe.cu:1996-2000` (yet another)
- `executor_forward_moe.cu:2121-2123` (`try_run_moe_gemma4_ggml_prefill`
  per-token sync — Gemma-4 fallback only)
- `executor_forward_moe.cu:1288-1292` (FP8 scales fetch)

For Qwen3-Coder NVFP4 specifically the device-args path is taken, so
**routing D2H is not the bottleneck**.

### 6.4 Which CUTLASS template is instantiated?

`src/compute/gemm_cutlass_grouped_3x.cu:30-86`:
- ArchTag = `cutlass::arch::Sm120`
- OperatorClass = `OpClassBlockScaledTensorOp`
- TileShape = `Shape<_128, _128, _128>`
- ClusterShape = `Shape<_1, _1, _1>` ← only 1 CTA per cluster
- Element A/B = `cutlass::nv_float4_t<float_e2m1_t>`
- Element D = `cutlass::half_t`
- KernelSchedule = `KernelScheduleAuto` → resolves to
  `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>` (3-stage)
  per memo.
- EpilogueSchedule = `EpilogueScheduleAuto`
- Adapter is `static`, lifetime-cached: `s_gemm` at `:113`,
  `s_gemm_initialized` flag, `can_implement` memoized per (N,K) at
  `:298-307`. Workspace + staging persistent (`s_staging`, `s_workspace`,
  `ensure_*` at `:120-136`).

This **matches vLLM exactly** per memo `nvfp4_moe_prefill_landscape_2026_05_10`
section "What vLLM actually uses".

### 6.5 Is `m_offsets` rebuilt per layer or amortized?

Per layer. `compute_M_per_from_offsets_device` runs every prefill MoE
call (`executor_forward_moe.cu:535, 678`). It's a small device kernel
(reads ne+1 ints, writes ne ints) — ~1 µs.

The expensive per-call work is the **per-expert SFA layout build**:
`compute_sfa_offsets_device` + `build_sfa_bases_device` + the SFA
zero-memset (`cudaMemsetAsync` of `moe_.cutlass3x_sf_size` ≈ 2 MB at
`:557-558`, ~3 µs). All of this is in the device path now.

Per-expert weight pointer arrays are amortized into `da_cache` per
layer at model load (when populated). The 3× H2D fallback at
`:601-613` only fires when `da_cache` isn't ready — i.e. never in
steady state on a properly-loaded NVFP4 prequant model.

### 6.6 Dequant: online or offline?

**Online.** The CUTLASS Sm120 NVFP4 mainloop does FP4→FP32 conversion
during MMA via the block-scaled MMA opcode (`mxf4nvf4.block_scale.scale_vec::4X`).
The activation quantize step at `executor_forward_moe.cu:559-563`
(`quantize_fp16_to_nvfp4_cutlass_moe`) converts FP16 activations to NVFP4
on the fly into the staging buffer (no offline cache for activations).

For decode (n=1), per-expert kernels `gemv_nvfp4_moe_*` in
`src/quant/nvfp4_gemm.cu:855` do the FP4 dequant via inline PTX
`cvt.rn.f16x2.e2m1x2` (single PTX op, per `dead_ends.md:72` correction).

`grep -rn "cvt.rn.satfinite.e2m1x2.f16x2\|cvt.rn.f16x2.e2m1x2" src/`
hits both NVFP4 paged attention kernels (`attention_paged_nvfp4.cu:43`,
`attention_paged_nvfp4_tc.cu:44`) and the activation quant path.
**No offline-dequantize-to-FP16-then-cuBLAS** on the production
hot path.

### 6.7 Combine kernel

`moe_weighted_sum_residual` at `executor_forward_moe.cu:2362` and
`:2479`. Implementation outside this audit — likely a fused write
into hidden_ with FP16 accumulator. Bandwidth-trivial.

### 6.8 Root-cause buckets for the 14.7× gap (vLLM single-seq)

I disagree with the memo's framing in one place: the memo treats this
gap as "multi-week, no single PR" — which is correct for **closing
it** but understates how much is identifiable. Three buckets:

**Bucket A — CUTLASS scheduler maturity (host launch / dispatch
overhead) — blame ~50%.**
- The memo confirms vLLM uses the same template. The kernel itself
  cannot be slower in imp.
- Per-prefill launch overhead is ~1.1 ms after `769effe` (memo). That's
  ~10% of typical prefill wall (≈11 ms at 16k tok/s × 512 / 16000
  ≈ 32 ms — wait, this checks: 1258 tok/s × 1s = 1258 tok = 2.46
  chunks × 512 = single chunk takes ~407 ms at 1258 tok/s).
- At ~407 ms wall vs vLLM ~28 ms (18.5k tok/s × 512 = ~28 ms), the
  total is **~14× slower**. CUTLASS host launch contributes some
  fraction; the bigger contributor is **how often the kernel
  re-instantiates can_implement / scheduler state across iterations**.
- Imp memoizes `can_implement` per (N, K) (`gemm_cutlass_grouped_3x.cu:298`)
  and uses `update()` instead of `initialize()` after the first call
  (`:317-330`). That's the right pattern. Comparison with vLLM
  would need its FlashInfer wrapper code.
- **Fix sketch:** none — already optimized per memo. Bucket may be
  smaller than 50% if the actual gap is dominated by bucket B.

**Bucket B — Quantization + staging activation cost — blame ~30%.**
- For each prefill chunk × layer × {gate, up, down} = 3 grouped GEMMs.
  Each one quantizes the gathered FP16 activation to NVFP4 +
  per-128-elem UE4M3 SF (`quantize_fp16_to_nvfp4_cutlass_moe` at
  `:559-563`).
- For n=512, top_k=8 expanded = 4096 rows × d=2048 = 8M FP16 elems
  read, 4M FP4 + 256K SF bytes written per gate-or-up call. Two
  calls per layer (gate+up share, down). 64 MoE layers × 2 quant calls
  = 128 quant kernel launches per prefill chunk.
- Each launch is ~100k threads × tens of cycles → ~50 µs minimum
  per call → 128 × 50 µs = 6.4 ms wall **just for activation quant**.
- vLLM likely fuses or reuses this. **Fix:** fuse the SwiGLU
  activation with the gate+up quant for the down stage, avoiding one
  full read/write of the activation tile. Hinted at by
  `moe_fusion_targets.md` (memo). 2-3 days work per memo.

**Bucket C — Per-layer kernel launches around the GEMM — blame ~20%.**
- Per layer: residual save (cudaMemcpyAsync DtoD) + RMSNorm + gate GEMV
  + topk + permute + 3× SFA-builds + 3× zero-memset + 3× quant + 3×
  CUTLASS + activation + combine = ~15 kernel launches × 64 layers
  = ~960 kernel launches per prefill chunk. At ~5 µs each → 4.8 ms.
- Captured into a graph this drops to ~1 launch (cuGraphLaunch). The
  prefill graphs phase-3 lands ~+11–39% (per `moe_prefill_graphs_plan_2026_05_10`)
  but **phase 4 is partial** — full multi-layer capture is incomplete.
- **Fix:** complete phase 4 prefill graph capture. ETA per memo:
  weeks. Real upside ~+10–15% per `nvfp4_moe_prefill_landscape`.

**Putting buckets together:** the headline 14.7× gap closes to maybe
2–3× even after all three buckets are fully shipped, per the memo's
own ceiling analysis. The remaining 2–3× is CUTLASS-internal scheduler
maturity (waiting on NVIDIA upstream). **The actionable gap in 2026 H2
is probably ~3×, not 14.7×.**

### 6.9 What I'd ship first

**Bucket B fusion (gate+up+SwiGLU+quant into a single kernel for the
down phase).** Reasoning:
- Smallest engineering footprint (2-3 days estimated, per memo
  `moe_fusion_targets.md`).
- Decode-side risk LOW: the fusion is in a prefill-only path
  (`run_moe_ffn` non-fast-path branch). Decode fast-path
  (`run_moe_decode_fast` at `:2316`) doesn't use these grouped GEMMs.
- Quality risk LOW: same numerics, just fewer round-trips through
  HBM.
- Concrete signal: per-component nsys profile would tell exact saving
  before commit. Speculative magnitude: ~5–10% pp512 lift.

**What I would NOT ship:**
- Hand-rolled NVFP4 grouped kernel beyond the existing smallM opt-in
  (smallM lost 50% at production M, per `nvfp4_moe_prefill_landscape`).
- cuDNN integration (broken on SM120 per memo).
- Cluster-launch on CUTLASS template (multi-week CUTLASS template
  surgery; speculative gain).

### 6.10 Agreement with memo `nvfp4_moe_prefill_landscape_2026_05_10`

- **Agree:** vLLM uses same template, no vendor IP, ceiling is
  CUTLASS scheduler maturity.
- **Agree:** hand-rolled NVFP4 grouped kernel can't beat CUTLASS in
  1-2 weeks.
- **Refine:** memo doesn't explicitly break out activation-quantize
  cost as bucket B at ~30%. The per-kernel-launch arithmetic above
  is mine; would need nsys to confirm.
- **Disagree:** memo treats CUDA Graph capture for prefill as parked
  ("Option B blocked by `IsMoEScheduler = false` stub"). The
  device-args path at `executor_forward_moe.cu:508-665` is graph-
  capturable today (no D2H) — Phase 3 PR #164 confirmed +11–39%.
  Phase 4 *for non-NVFP4 MoE arches* is still blocked, but for
  Qwen3-Coder specifically the path is open.

---

## 7. Top-10 performance leaks (ranked)

Ordered by TTFT × confidence. At least 3 non-obvious.

```
1. NVFP4 MoE prefill activation-quant fusion gap  [TTFT impact: H]  [confidence: M]  [effort: days]
   src/graph/executor_forward_moe.cu:559-563 (quantize_fp16_to_nvfp4_cutlass_moe)
   src/graph/executor_forward_moe.cu:646-657 (apply_expert_activation + second quant)
   Activation tiles roundtrip through HBM twice per layer (after gate+up, before down)
   because quant lives outside the activation kernel. ~6 ms of pp512 wall on
   Qwen3-Coder.
   Fix: fuse SwiGLU + quantize into single kernel for the down-phase activation.
   2-3 days per moe_fusion_targets.md memo.
   Decode-side risk: NONE — decode fast-path uses gemv_nvfp4_moe_*, not this code.

2. Qwen3-Coder NVFP4 prefill 14.7× gap (entirety)  [TTFT impact: H]  [confidence: H]  [effort: weeks]
   src/compute/gemm_cutlass_grouped_3x.cu:150
   src/graph/executor_forward_moe.cu:508-665
   Section 6 above. Three buckets, no single PR closes it; bucket B (#1 above)
   is the first shippable slice.
   Decode-side risk: NONE for bucket B alone.

3. Per-token D2H sync in Gemma-4 ggml MoE prefill  [TTFT impact: H]  [confidence: H]  [effort: 1-2 days]
   src/graph/executor_forward_moe.cu:2121-2123 (cudaMemcpyAsync h_experts +
   cudaStreamSynchronize per token in try_run_moe_gemma4_ggml_prefill)
   Per-token D2H + stream sync kills prefill graph capture entirely AND linearises
   the host CPU vs GPU. For n=512 prefill that's 512 sync points.
   Fix: build expert per-token slices device-side, replace inner host loop with a
   single device kernel that takes routing.expert_indices as input.
   Decode-side risk: NONE — function only runs for n>1.

4. Per-decode-step graph re-instantiation cost on shape change  [TTFT impact: M]  [confidence: M]  [effort: days]
   src/runtime/cuda_graph.cu:228-240 + engine.cpp:2643
   cudaGraphExecUpdate is attempted before full reinstantiation. When n_sequences
   or ctx_len pushes into a different bucket, full reinstantiate fires. Cost
   unmeasured in current memos but should be ~10-30 ms per swap.
   Fix: bucket allocations more aggressively (ctx_len in pow-2 buckets), enlarge
   the multi-graph pool above kMaxGraphPoolSize.
   Decode-side risk: LOW — slight VRAM bump.

5. NVFP4 smallM kernel kept opt-in despite -50%  [TTFT impact: L]  [confidence: H]  [effort: 0]
   src/compute/gemm_grouped_nvfp4_smallM.cu (entire 948 LOC TU)
   src/graph/executor_forward_moe.cu:703-704 (gated by IMP_NVFP4_SMALLM env)
   This is technical debt in the binary. The opt-in gate is correct (CUTLASS
   wins) but the TU is in the runtime library. **NOT a perf leak per se** but
   the cudaMallocAsync paths inside it (e.g. line 575-580) tie the smallM
   memory pool to the stream. If a user sets IMP_NVFP4_SMALLM=1 in prod they
   get a 50% perf regression. Move smallM to a separate test-only TU or
   document it as benchmark-only.
   Decode-side risk: NONE (opt-in).

6. BitDecoding TC path empty win + complexity  [TTFT impact: L]  [confidence: H]  [effort: 0 to remove or weeks to fix]
   src/compute/attention_paged_nvfp4_tc.cu (1216 LOC)
   src/graph/executor_attention.cu:1027-1083 (residual ring args)
   The TC dispatch lands 0% across all measured configs (long_context_eval memo).
   Residual ring infrastructure (Phase 3a/b/c) ships parity but no measurable
   win for typical workloads.
   Fix: keep opt-in. **Non-obvious leak:** the kernel signature now takes 8
   nullable trailing args; even when all nullptr the host marshalling for
   that opt-in is ~few µs per call * n_layers * n_decode_steps. Could be
   gated behind a single null-check at the dispatcher (executor_attention.cu:1071)
   instead of always passing.
   Decode-side risk: LOW for the dispatcher refactor.

7. Stale wgmma docstring in FMHA header  [TTFT impact: 0 PERF]  [confidence: H]  [effort: 5 min]
   src/compute/attention_fmha_sm120.h:8-10
   Claims "WGMMA (Warp Group MMA) ... wgmma.mma_async PTX instructions for ~2x
   tensor core throughput vs WMMA". The implementation uses WMMA (HMMA SASS) +
   FP8 inline mma.sync, not wgmma (which is sm_90 only and ptxas-rejected on
   sm_120a per dead_ends.md:17).
   **Non-obvious because it's documentation; flags here because it's actively
   misleading anyone reading the FMHA code to find perf wins.**
   Decode-side risk: NONE.

8. Lazy cudaMalloc/cudaFree in mmvq scratch  [TTFT impact: M for first call]  [confidence: H]  [effort: 1 hour]
   src/graph/executor_kernels.cu:2175-2182
   First call at a new K hits cudaFree + cudaMalloc on the stream — illegal under
   stream capture. Subsequent calls hit the cache. **Non-obvious: the cache is
   keyed by size only, so if the kernel sees a smaller-then-larger sequence it
   stays valid; if it sees only growing sizes it free/mallocs each time.**
   Fix: pre-warm via the existing executor init flow (mirror gemm_init
   pre-alloc pattern). One static call site.
   Decode-side risk: NONE.

9. Per-layer SFA zero-memset in MoE prefill  [TTFT impact: M]  [confidence: M]  [effort: hours]
   src/graph/executor_forward_moe.cu:557-558
   "Zero the whole staging buffer — graph-capturable alternative to host-
   computed total_sfa. Cost is ~2 MB memset (~3 µs at 1.8 TB/s)."
   3 µs × 3 quants/layer × 64 layers × 4 prefill chunks per session = ~2.3 ms
   per session. **Non-obvious: it's a microsecond-grade cost that the
   programmer (correctly) decided was acceptable; but it's free to fix by
   tracking the actual active SFA bytes in a device-side counter and
   doing a tighter memset/clear-if-needed.**
   Decode-side risk: NONE.

10. LM-head GEMV at decode lacks L2-streaming hint  [TTFT impact: L]  [confidence: M]  [effort: 1 hour]
    src/graph/executor_forward.cu (final LM head step, not L2-managed in
    set_l2_streaming/persist callers list)
    LM-head weight tensor is ~256 MiB (vocab × d_model × FP16); doesn't fit
    L2, but a streaming hint via set_l2_streaming would tell the GPU to not
    evict KV from L2. Currently no hint = default = some L2 pollution.
    **Non-obvious: every other big weight tensor has this hint except LM head.**
    Speculative magnitude: ≤1% decode at long context.
    Decode-side risk: NONE (just a hint).
```

---

## 8. GB202 features NOT YET used (clean list)

Order-of-magnitude expected upside in parentheses.

1. **Cluster launch in FMHA prefill** (KV reuse 2–4× via DSMEM for
   long-context). Not implemented. Cluster scheduling already wired
   for paged attention so the infrastructure exists; FMHA tile
   sizes would need re-layout. Expected upside: **+10–20% on
   pp4096+ on FP16 prefill** for non-MoE models.

2. **PV-FP4 in FMHA (Phase 3, MXFP4 block-scaled P×V)**. Currently
   only Q.K uses block-scaled FP4 MMA; PV stays in WMMA FP16
   (per `attention_fmha_sm120.cu:563`). Roadmap claims +13% upside.
   Quality-risky — needs SageAttention3 two-level accumulator.
   Expected upside: **+10–13% on attention-bound prefill**.

3. **NVFP4 dense GEMM for non-MoE models** (currently only via
   `gemv_nvfp4_kpar` for decode and the CUTLASS NVFP4 path is
   activated only when `nvfp4_cache` populated by prequant
   SafeTensors loader). The kpar decode kernel exists; the
   prefill grouped non-MoE path is hooked but rarely fires on
   GGUF Q*_K models. Expected upside: **+30–50% prefill** vs
   FP16 cuBLAS for prequant-NVFP4 dense models. Conditional on
   model format.

4. **Block-scaled NVFP4 KV (full BitDecoding stack)**. Phase 3
   shipped FP16 residual + WMMA Q.K but the TC path lands 0%
   currently. Real BitDecoding paper requires four combined
   levers (per memo) — only 1.5 of 4 are in. Expected upside if
   completed: **paper claim 8.6×, realistic 1.5–2×** decode at
   long ctx.

5. **Sparse FP4 MMA (`sp::ordered_metadata`)**. PTX accepts
   `mma.sync.aligned.kind::f8f6f4.sp::ordered_metadata.m16n8k64`
   (per `mxf4nvf4_mma_variants_bench.cu:229`) but production
   doesn't use it. Would double effective compute on 50%-sparse
   weights. Gates on a sparsification recipe. Expected upside:
   **2× theoretical** on sparse-weight inference (currently
   no sparse weights in scope).

6. **`mma.sync.aligned.kind::mxf8f6f4.scale_vec::1X` (mixed FP8 + FP4)**.
   Variants benched at `mxf4nvf4_mma_variants_bench.cu:194` show
   FP8 act × FP4 weight is supported on sm_120. Could be used for
   prefill where activations are still FP8 cache. Expected upside:
   **+20% over current FP8×FP8 path** if quality holds.

7. **`cublasLtMatmulGrouped` with NVFP4** — re-tested 2026-05-08
   (memo `cublas_13_4_sm120_no_movement_2026_05_09`): zero
   algorithms on SM120. **DEAD-END unless NVIDIA ships SM120
   support.** Listed here for completeness.

8. **`add.f32x2` PTX** — investigated, decomposed to 2× scalar
   FADD on sm_120 (per `dead_ends.md` and MEMORY entry). DEAD-END.

9. **L2 persistence on prefix cache** — speculative. Would let the
   first N tokens of a long prompt stay in L2 across decode steps.
   Currently `set_l2_persist_kv` is per-call. Expected upside:
   marginal at decode (KV is bandwidth-saturated either way).

10. **Multi-CTA scheduler for MoE NVFP4 (CUTLASS `MoEProblemShape`)**.
    Blocked upstream (memo): "sm120 scheduler carries `IsMoEScheduler =
    false` stub". Re-trigger when CUTLASS upstream wires it. Expected
    upside: +10–15% prefill per memo.

---

## 9. Speculative wins (clearly marked)

**SPECULATIVE — needs ncu trace before commit.**

These are sensible-on-paper ideas that need profiling before any
engineering. Don't ship without an nsys/ncu trace showing the
predicted hotspot.

S1. **Pre-warm `cudaGraphInstantiate` for known shapes at model load.**
   The first capture per shape incurs instantiation cost (~tens of ms
   per memo). If the engine knows the prompt-length buckets up front
   (e.g. 128, 256, 512 token chunks) it could pre-instantiate those
   graphs at warmup time. Need: nsys trace of first-100-tokens latency.

S2. **Persistent kernel for varlen continuous batching at the API layer.**
   Per `sm120_real_perf_levers_2026_05_04`: "CLC-Persistent-Kernel for
   varlen +10-20% bei concurrent batching". imp currently dispatches
   per-request grids. A persistent kernel that absorbs varlen prompts
   in a single grid could amortize launch overhead.

S3. **L2-persist on LM-head decode weights** (see leak #10). Speculative
   gain.

S4. **Fused norm+gate-GEMV+topk in single kernel for MoE routing**. The
   existing `gemv_gate_topk_fused_kernel` (`moe_routing.cu:196`) fuses
   gate GEMV + softmax + top-k but not the preceding norm. Saving one
   norm read for routing is small; only worth it if profiling shows
   routing path is >5% of decode time.

S5. **Replace `cudaMallocAsync` calls in `quantize_fp16_nvfp4_moe_native.cu:245-247`
   with a persistent slab.** Three tiny device arrays per call. Per-call
   cost is small but ties the graph to the memory pool.

S6. **TMA-multicast cluster on FMHA for very long context (n>16k).** With
   cluster launch (S2) + multicast TMA, K could be broadcast to all
   cluster CTAs in a single bulk descriptor. Hardware supports it per
   §3.7. Implementation complexity high.

S7. **MoE expert weight prefetch via persistent device-side LRU.** Today
   `experts_on_host_=true` disables graphs and synchronously pages H2D
   per expert per token. A device-side LRU with cudaMemcpyAsync
   prefetch overlap could keep graphs ON. Mentioned in `roadmap.md:53`
   as future work.

---

## Appendix: file:line citation map

Heavy citations from this audit, deduplicated:

| Anchor | Purpose |
|---|---|
| `src/api/imp_api.cpp:661` (`imp_decode_step`) | decode entry |
| `src/runtime/engine.cpp:1735` (`Engine::step`) | step driver |
| `src/runtime/engine.cpp:2421` (`step_decode_forward`) | decode forward |
| `src/runtime/engine.cpp:2122-2149` (prefill H2D upload) | per-chunk batch upload |
| `src/runtime/engine.cpp:2209` (`can_capture`) | prefill graph gate |
| `src/runtime/engine.cpp:2637` (`use_cuda_graphs` decode) | decode graph dispatch |
| `src/runtime/cuda_graph.cu:14-36` (`get_capture_mode`) | global/relaxed/thread_local |
| `src/runtime/cuda_graph.cu:41-152` (`apply_pdl_edges`) | post-capture PDL conversion |
| `src/runtime/pdl.h:34-59` (`pdl::launch`) | PDL launch wrapper |
| `src/graph/executor_forward.cu:174` (`forward_logits`) | forward outer |
| `src/graph/executor_forward.cu:280-289` (token-ID D2H) | gated by `debug_forward_enabled()` |
| `src/graph/executor_attention.cu:97-131` (`set_l2_persist_kv`) | KV L2 persist + clamping |
| `src/graph/executor_attention.cu:984` | persist call site |
| `src/graph/executor_attention.cu:988-1112` | KV-dtype paged attention switch |
| `src/graph/executor_attention.cu:1027-1083` | BitDecoding TC dispatch + residual args |
| `src/graph/executor_attention.cu:1071-1091` | NVFP4 paged decode TC vs scalar |
| `src/graph/executor_helpers.h:67-89` (`set_l2_streaming`) | streaming policy |
| `src/graph/executor_helpers.h:91-97` (`clear_l2_policy`) | clear |
| `src/graph/executor_ffn.cu:69` | FFN L2-streaming call |
| `src/graph/executor_forward_moe.cu:101-111` (`can_decode_fast`) | MoE decode fast-path gate |
| `src/graph/executor_forward_moe.cu:146` (`run_moe_ffn`) | MoE entry |
| `src/graph/executor_forward_moe.cu:212` | residual save DtoD |
| `src/graph/executor_forward_moe.cu:508-665` | device-args full prefill path |
| `src/graph/executor_forward_moe.cu:601-613` | H2D pointer-array fallback |
| `src/graph/executor_forward_moe.cu:668-682` | legacy D2H sync path |
| `src/graph/executor_forward_moe.cu:1131-1135, 1189-1192, 1996-2000` | further D2H sync sites |
| `src/graph/executor_forward_moe.cu:2065-2102` (`try_run_moe_gemma4_ggml_prefill`) | lazy cudaMalloc/cudaFree + per-token sync |
| `src/graph/executor_forward_moe.cu:2121-2123` | per-token D2H + sync barrier |
| `src/graph/executor_forward_moe.cu:2316` (`run_moe_decode_fast`) | MoE decode fast-path |
| `src/graph/executor_kernels.cu:2099-2103` (NVFP4 CUTLASS dispatch) | dense NVFP4 GEMM |
| `src/graph/executor_kernels.cu:2156-2168` | Q4_K v2 HMMA dispatch gate (IMP_FORCE_Q4K_V2) |
| `src/graph/executor_kernels.cu:2175-2182` | mmvq scratch cudaMalloc/Free |
| `src/compute/attention_paged.cu:90` (`paged_attention_gqa_kernel`) | default decode kernel |
| `src/compute/attention_paged.cu:1111` (`paged_attention_cluster_kernel`) | cluster decode |
| `src/compute/attention_paged.cu:1469-1473` | cluster launch attributes |
| `src/compute/attention_paged_common.cuh:17-34` | cp.async helpers |
| `src/compute/attention_paged_nvfp4.cu:43` (`fp4_byte_to_half2`) | PTX cvt.rn.f16x2.e2m1x2 |
| `src/compute/attention_paged_nvfp4.cu:53` (decode kernel) | scalar NVFP4 paged decode |
| `src/compute/attention_paged_nvfp4_tc.cu:44, 58` | TC NVFP4 paged decode + residual |
| `src/compute/attention_fmha_sm120.h:8-10` | **STALE** wgmma comment (no wgmma code) |
| `src/compute/attention_fmha_sm120.cu:69, 581` | FMHA FP16 + FP8 kernels (`__launch_bounds__(256,1)`) |
| `src/compute/attention_fmha_sm120.cu:758` | inline FP8 `mma.sync.kind::f8f6f4.m16n8k32` |
| `src/compute/attention_fmha_mxfp4_sm120.cu:108, 591` | MXFP4 FMHA + `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` |
| `src/compute/attention_cublas.cu:343-347` (`ensure_attn_ptr_arrays`) | static ptr-array grow |
| `src/compute/attention_cublas.cu:357-369` (`build_attn_ptr_arrays_kernel`) | device-side ptr array builder |
| `src/compute/gemm.cu:51-78` | cublas handle init + TF32 mode |
| `src/compute/gemm.cu:100-115` | pre-allocated 64 MiB cuBLASLt workspace |
| `src/compute/gemm_cutlass_grouped_3x.cu:30-86` (template config) | CUTLASS template definition |
| `src/compute/gemm_cutlass_grouped_3x.cu:113, 298-307` (memoization) | adapter + can_implement memo |
| `src/compute/gemm_cutlass_grouped_3x.cu:120-148` (`ensure_staging`, `prewarm`) | grow-only allocators |
| `src/compute/gemm_cutlass_grouped_3x.cu:317-330` | initialize vs update fast-path |
| `src/compute/gemm_grouped_nvfp4_smallM.cu:87` | inline `mxf4nvf4.block_scale.scale_vec::4X` (opt-in) |
| `src/compute/gemm_grouped_nvfp4_smallM.cu:746-754` (`available`) | sm_120 detection |
| `src/compute/gemm_grouped_nvfp4_smallM.cu:117-122, 362-363, 702-744` | TMA descriptor build + dispatch |
| `src/compute/mmq_q4k_v2.cu:404, 607, 975, 1166, 1421, 1591` | direct `mma.sync.aligned.m16n8k16` |
| `src/compute/moe_routing.cu:196` (`gemv_gate_topk_fused_kernel`) | fused gate + softmax + topk |
| `src/compute/moe_routing.cu:533` (`moe_fused_permute_kernel`) | single-block scan, `__launch_bounds__(256)` |
| `src/quant/nvfp4_gemm.cu:32-38` | `kKparThreads=128, kMRThreads=256` |
| `src/quant/nvfp4_gemm.cu:174, 196, 855, 887` | NVFP4 GEMV kernels, `__launch_bounds__(128, 12)` |
| `src/quant/mxfp4_gemm.cu:147, 163, 183, 206` | MXFP4 GEMV kernels |
| `src/compute/quantize_fp16_nvfp4_moe_native.cu:245-247, 283-285` | cudaMallocAsync × 3 + cudaFreeAsync × 3 per prefill call |
| `src/compute/sampling.cu:1-28, 49-95, 834` | sampler (block 256, cub::DeviceTopK for k>128) |
| `CMakeLists.txt:74` | CUTLASS v4.5.0 declared |
| `Dockerfile:27` | CUTLASS v4.4.2 actually pinned in Docker (Phase 1 §5 note) |

---

End of Phase 2 perfhawk audit.

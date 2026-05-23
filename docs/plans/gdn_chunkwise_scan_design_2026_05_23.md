# GDN Mamba2 Chunkwise (SSD) Scan — Qwen3.6 Prefill Lever
*2026-05-23 · multi-week design doc · not yet implemented*

## Mission

Replace `gdn_scan_fused_kernel<128,128,half>`'s **sequential token loop** with the Mamba2 chunkwise (SSD: Structured State-space Duality) algorithm to unlock prefill throughput on Qwen3.6-35B-A3B-NVFP4. The current kernel is **36.6 % of prefill kernel time** at long contexts. Parallel scan via matmul-based chunkwise SSD has known O(n / log n) speedup over sequential scan.

Realistic upside: **+20-30 % prefill wall** on Qwen3.6, lifting pp512 from 11 k → ~14-15 k tok/s and pp2048 from 10.8 k → ~14 k tok/s.

## Why this is the first non-refuted lever

The session refuted 13 hypotheses on Qwen3.6 (`MEMORY.md` Workflow feedback section), but every one of them targeted **decode** or **engine micro-optimisation** on existing kernels. The Qwen3.6 prefill profile (this session) reveals:

```
36.6 %  gdn_scan_fused_kernel<128,128,half>     ← sequential token loop
23.9 %  cutlass GroupProblemShape Universal     ← MoE grouped GEMM (efficient)
8.4 %   cutlass_80 f16 64x64_32x6               ← FP16 GEMM
4.6 %   cutlass_80 f16 128x64_64x4              ← FP16 GEMM
2.0 %   ssm_conv1d_prefill_f32_silu             ← Mamba2 conv1d
...
```

The GDN scan kernel was profiled before for DECODE and was 4.1 % of decode kernel time. The Phase-0 cost analysis (`gdn_scan_splitk_phase0_refuted_2026_05_23.md`) refuted split-K for decode because the cross-block sync overhead (4 × 7 µs = 28 µs per token-step) exceeded the 90 µs/decode-token achievable win.

**Prefill physics are different**: per-call kernel time is ~467 µs (avg over the prefill profile) vs decode's 5.9 µs. The split-K sync cost is the same in absolute µs but the relative overhead drops 80×. **Split-K for prefill might pencil out.** But there's a MUCH stronger algorithmic angle.

## The Mamba2 chunkwise / SSD algorithm

The sequential GDN scan computes:

```
For each token t in [0, n_tokens):
    H_new[t] = g_t · H[t-1] + k_norm[t] · delta[t]
    y[t] = H_new[t]^T · q_norm[t]
```

This is a 1D recurrent dependency with O(n_tokens) sequential steps. On 30 layers × 4096 tokens = ~123 k sequential iterations per forward.

The Mamba2 paper (Dao + Gu 2024) restructures this as:

1. **Chunk the sequence** into blocks of size `B` (typically 64 tokens).
2. **Within each chunk**, the recurrence becomes a **lower-triangular matrix multiply**:
   - Compute `D[i,j] = g[i] · g[i-1] · ... · g[j+1]` for all i ≥ j in the chunk (lower-triangular decay matrix)
   - Compute `y_chunk = (Q_chunk · K_chunk^T · D) · V_chunk` — this is **structurally a masked attention** computation, runnable on Tensor Cores
3. **Across chunks**, propagate the chunk-end state via a smaller scan: `H[chunk_end] = scan over chunks of (g_chunk · H[prev_chunk_end] + chunk_residual)`. With chunk_size=64, this is `n_tokens/64 = 64` cross-chunk steps instead of 4096 sequential steps.

**Net complexity**: O(n_tokens × chunk_size) Tensor-Core matmul ops + O(n_tokens / chunk_size) sequential. For n_tokens=4096, chunk=64: the inner work is 4096 × 64 = 262 k matmul ops (TC-accelerated) + 64 sequential cross-chunk steps. Vs current: 4096 × ~10 µs sequential operations.

Practical speedup on H100 in published benchmarks: **3-5× over sequential scan** for n_tokens = 1k-4k. On sm_120 (consumer Blackwell, no tcgen05) the speedup may be lower because the Tensor Core utilisation is bounded by FP16 HMMA rates (838 TFLOPS) rather than tcgen05 MMA (more on B200/SM100). Realistic sm_120 ceiling: **2-3× on the scan kernel** = +18-25 % prefill wall.

## Why this isn't a 1-week project

The SSD reformulation is mathematically elegant but the kernel implementation is non-trivial:

1. **Lower-triangular masked attention compute** — needs to handle the causal mask without doing 2× the FLOPs of unmasked
2. **Chunk-boundary state propagation** — requires a separate small-grain kernel or inline pass; needs FP32 precision to avoid GDN-state precision drift over chunks
3. **K/Q normalization** — the per-token L2 norm of K/Q is still per-token; needs to interleave with the chunk-matmul pipeline
4. **Variable-token batches** in chunked prefill — chunks aren't always full; need clean handling at boundaries
5. **Decode fallback** — for n_tokens=1, chunkwise is overkill; the sequential kernel is fine. Need clean dispatch.

Reference implementations:
- **Mamba2 official** (`state-spaces/mamba` GitHub) — Triton kernel, useful as algorithm reference but not directly portable to sm_120 CUDA C++.
- **Flash-Linear-Attention** (`fla-org/flash-linear-attention`) — CUTLASS-style implementation, closer to imp's style. Could be a starting point.
- **Mamba2 paper Algorithm 3** — pseudocode for the SSD chunkwise scan.

## Implementation phases

### Phase 0 — Verify the bottleneck breakdown (1 day)

- [ ] ncu with full memory + compute breakdown on `gdn_scan_fused_kernel` at PREFILL shape (n_tokens=4096, large grid)
- [ ] Classify: HBM-bound? SM-throughput-bound? Latency-bound on the sequential dependency?
- [ ] If HBM-bound, the chunkwise rewrite won't help (same total bytes). If compute or latency-bound, chunkwise can win.

### Phase 1 — Numerical reference implementation (3-5 days)

- [ ] Standalone CUDA prototype of the chunkwise SSD algorithm at chunk_size=64
- [ ] Pure FP32 first (no NVFP4/FP8 quantisation), correctness reference
- [ ] Bit-equivalent (to numerical tolerance) output vs `gdn_scan_fused_kernel` at chunk_size=64 boundary
- [ ] Microbench: compare per-token throughput on the standalone kernel

### Phase 2 — Production kernel integration (5-7 days)

- [ ] Add `gdn_scan_chunkwise_kernel<HD, SS, CHUNK_SIZE, YOut>` in `src/compute/gdn.cu`
- [ ] Use CUTLASS / cute tile descriptors for the chunk-internal lower-triangular MMA
- [ ] FP32 accumulation for cross-chunk state propagation (precision preservation)
- [ ] Dispatch from `gdn_scan_fused_f32_*` host launchers when `n_tokens >= CHUNK_SIZE` (decode falls through to existing sequential path)
- [ ] Gate behind `gdn.chunkwise_scan = false` config flag (off by default until validated)

### Phase 3 — Numerical validation across context lengths (2-3 days)

- [ ] Coherence smoke tests on Qwen3.6-35B-A3B-NVFP4 at ctx={128, 1024, 4096, 16384}
- [ ] NIAH retrieval at ctx={4K, 16K, 32K} — sensitive to GDN state precision drift
- [ ] If any context shows degradation, debug FP32-accumulation precision in the cross-chunk propagation

### Phase 4 — Perf bench + ship decision (1 day)

- [ ] Cold-median bench Qwen3.6 + Qwen3.5-9B-GDN + Qwen3.5-4B-GDN (all GDN models) with chunkwise scan on
- [ ] Compare per-pp wall vs sequential baseline
- [ ] If ≥ +10 % prefill wall on Qwen3.6 AND no quality regression: flip default

## Risks

- **Sequential kernel is already FP32-accumulation-correct.** The chunkwise variant inherits FP32 accumulation across chunks but has more FP32-to-FP16-and-back conversions. Precision audit needed.
- **Tensor Core utilisation on sm_120 capped at FP16 HMMA (838 TFLOPS).** B200 / SM100 has tcgen05 MMA at much higher TFLOPS for SSD-shaped tile patterns. The sm_120 ceiling is lower than the published H100/B200 benchmarks.
- **The Mamba2 algorithm assumes constant `g` within a chunk.** GDN has per-token `g_t = exp(A_h * dt_val)` — variable per token. The standard SSD formulation handles this via the D matrix (which is per-token-pair); confirm imp's formulation matches.
- **Chunked-prefill interaction** — imp's chunked prefill (default chunk_size=512) already breaks long prefills into 512-token chunks. The SSD chunk_size (64) is INSIDE each prefill chunk. Need careful state handoff at prefill-chunk boundaries.

## Don't repeat

- ❌ **Refactoring the scan kernel without an algorithmic angle.** Split-K Phase 0 refuted the "sync-based parallel scan" approach because cross-block sync overhead exceeded the gain. Chunkwise SSD is a DIFFERENT algorithm — uses Tensor Core matmuls within a chunk, not cross-block sync.
- ❌ **Testing on decode-only.** Chunkwise SSD has NO benefit for n_tokens=1. The dispatch must route decode through the existing sequential kernel.
- ❌ **Naive FP16 cross-chunk propagation.** GDN state drift over many chunks (e.g., 4096 tokens / 64 chunks = 64 chunks) is the same precision risk that motivates FP32 scan today. Keep FP32 cross-chunk accumulators.

## Estimate

- **Phase 0**: 1 day measure-only
- **Phase 1-2 implementation**: 8-12 days
- **Phase 3-4 validation + ship**: 3-4 days
- **Total**: 12-17 days of focused work

Realistic delivered wall savings:
- **+15-25 % prefill wall on Qwen3.6** (pp512 11 k → ~13-14 k; pp2048 10.8 k → ~13-14 k)
- **Decode unchanged** (sequential kernel still used for n_tokens=1)
- **Cross-model**: Qwen3.5-4B-GDN and Qwen3.5-9B-GDN would also benefit; pure-attention models (Qwen3-Coder, Qwen3-14B) unaffected

## Re-evaluation triggers

Re-open this plan when:
- A new GDN model lands and prefill becomes the bottleneck
- sm_120 ships tcgen05-equivalent primitives for SSD-shaped tiles
- The current sequential `gdn_scan_fused` time share grows past 50 % of prefill (currently 36.6 %)

---

*The 14th lever investigation this session — but the FIRST one that's not refuted. The remaining 13 were all kernel-micro-optimisation or naive-recipe model-level levers; this one is **algorithmic** and uses Tensor Cores in a way the current kernel doesn't. Multi-week work but real upside.*

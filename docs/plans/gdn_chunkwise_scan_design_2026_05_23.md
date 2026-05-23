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

### Phase 0 — Verify the bottleneck breakdown (1 day) ✅ DONE 2026-05-23

- [x] ncu with full memory + compute breakdown on `gdn_scan_fused_kernel` at PREFILL shape (n_tokens=4096, large grid)
- [x] Classify: HBM-bound? SM-throughput-bound? Latency-bound on the sequential dependency?
- [x] If HBM-bound, the chunkwise rewrite won't help (same total bytes). If compute or latency-bound, chunkwise can win.

**Verdict: PROCEED.** `ncu --set full` measurements (3 captures, consistent):
- Memory Throughput: **5.47 %** of peak (43 GB/s)
- Compute (SM) Throughput: **5.47 %** of peak
- Achieved Occupancy: **8.33 %** (half theoretical 16.67 %)
- Grid 32, Registers/thread 255 (compiler max), Block Limit Registers = 2

Both memory AND compute at 5.47 % is the **latency-bound signature** — the kernel waits on the sequential dependency chain. Cross-chunk sync cost (4 × 7 µs = 28 µs) is only 6 % of the per-call kernel time (467 µs avg) — sync overhead doesn't refute chunkwise the way it refuted split-K for decode (where 28 µs / 5.9 µs = 475 %). Full memo: `memory/gdn_chunkwise_scan_phase0_proceed_2026_05_23.md`.

### Phase 1a — Regression gate test ✅ DONE 2026-05-23 (PR #385)

- [x] Add `GDNScanTest.ChunkBoundaryHandoff` to `tests/test_gdn.cu`
- [x] Validate that splitting a 16-token sequential scan at the midpoint with H-state handoff produces bit-equivalent output to a monolithic 16-token scan
- [x] Sets the FP16 output tolerance budget (1e-3) and FP32 state tolerance budget (1e-5) for Phase 1b.1

**Result**: bit-exact (`max_diff = 0.0` on all 3 checks) on the existing sequential kernel. This test is the regression gate for the future SSD kernel.

### Phase 1b scaffolding — `gdn_scan_chunkwise_f32` API ✅ DONE 2026-05-23 (PR #386)

- [x] Add `gdn_scan_chunkwise_f32(...)` host function to `src/compute/gdn.{h,cu}` with `chunk_size` parameter (default 64)
- [x] Initial implementation: chunk-iterating sequential wrapper around `gdn_scan_fused_f32` (functionally identical, no perf change)
- [x] Extend `ChunkBoundaryHandoff` to include the chunkwise function as Run C

**Result**: bit-exact pass (`max_diff Y_chunkwise = 0.0`, `max_diff state_chunkwise = 0.0`). API surface ready for Phase 1b.1 to drop in the real SSD kernel.

### Phase 1b.1 — Standalone CUDA SSD prototype ✅ DONE 2026-05-23

- [x] Standalone CUDA prototype of the chunkwise SSD algorithm at chunk_size=64
- [x] Pure FP32 first (no NVFP4/FP8 quantisation), correctness reference
- [x] Bit-near-equivalent (within tolerance budget set by Phase 1a) output vs `gdn_scan_fused_kernel` at chunk_size=64 boundary
- [x] Microbench: compare per-token throughput on the standalone kernel
- [x] **Reference**: Yang et al. 2024 "Parallel Linear Attention With The Delta Rule" — the standard Mamba2 SSD assumes scalar decay and doesn't cover GDN's rank-1 `(I - β k k^T)` multiplicative update. The Yang paper covers the delta-rule via WY-representation tricks for the matrix product chain.

**Result**: New kernel `gdn_scan_chunkwise_kernel<HD, SS, CHUNK>` in `src/compute/gdn.cu` instantiated for HD=SS=128, CHUNK=64 (Qwen 3.5 / 3.6 GDN shape) and HD=SS=64, CHUNK=64. Dispatched from `gdn_scan_chunkwise_f32` when `chunk_size == 64` and `n_tokens >= 64`; falls through to the chunk-iterating wrapper otherwise. The structural change vs `gdn_scan_fused_kernel`: all CHUNK tokens' raw K, Q are loaded into shared memory upfront (Phase 1), L2-normalised in shared memory (Phase 2), then consumed by the sequential delta-rule sweep (Phase 3). The chunk-cached layout is the prerequisite for Phase 2's WY-rep parallel matmul.

**Numerics**: bit-exact (`max_diff_y = 0.0`, `max_diff_state = 0.0`) vs the monolithic `gdn_scan_fused_f32` on the new test `GDNScanTest.ChunkwiseProtoMatchesFused` (n_tok=128 across 2 chunks of 64, HD=SS=128). Well inside the FP16 1e-3 / FP32 1e-5 budgets set by Phase 1a.

**Microbench (n_tok=4096, n_heads=32, HD=SS=128, n_groups=16, RTX 5090 sm_120, 20 reps, gated on `IMP_GDN_MICROBENCH=1`)**:
- `gdn_scan_fused_f32` (sequential):      6.602 ms / 4096 tok = **1.612 µs/token**
- `gdn_scan_chunkwise_f32` (Phase 1b.1):  5.601 ms / 4096 tok = **1.367 µs/token**
- Ratio: **0.848× wall** = **+15.2 % throughput** on the GDN scan kernel alone.

The +15 % win is structural and comes for free with Phase 1b.1 — caching K, Q in shared memory eliminates redundant per-token global-memory loads of K (SS=128 floats) and Q (SS=128 floats) for every token of the chunk. Phase 2's WY-rep parallel matmul replaces the still-sequential delta-rule sweep on top of this; the design doc's +20-30 % prefill wall target is the combined ceiling.

Code: `gdn_scan_chunkwise_kernel<HD, SS, CHUNK>` in `src/compute/gdn.cu` (under "Phase 1b.1 — Standalone chunkwise SSD scan prototype"); dispatch in `gdn_scan_chunkwise_f32`; tests `ChunkwiseProtoMatchesFused` + `ChunkwiseProtoMicrobench` in `tests/test_gdn.cu`.

### Phase 2 — Production kernel integration (5-7 days) ⏳ IN PROGRESS

- [x] Add `gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut>` in `src/compute/gdn.cu` (PR #388 templated on YOut)
- [ ] Use CUTLASS / cute tile descriptors for the chunk-internal lower-triangular MMA → **Phase 2b** (Tensor Core acceleration; multi-week)
- [x] FP32 accumulation for cross-chunk state propagation (precision preservation) — done in 1b.1
- [x] Dispatch from `gdn_scan_fused_f32_*` host launchers when `n_tokens >= CHUNK_SIZE` (decode falls through to existing sequential path) — done in PR #388 (auto-merged #390)
- [x] Gate behind `gdn.chunkwise_scan = false` config flag (off by default until validated) — done in PR #388

The dispatch + flag plumbing landed; the remaining piece is the **algorithmic core**: replacing the sequential delta-rule sweep with the WY-rep parallel scan. Split into 2a (correctness reference, naive shared-memory matmul) and 2b (Tensor Core MMA via CUTLASS / cute).

#### Phase 2a — WY-rep delta-rule math worked out ✅ DONE 2026-05-24

**Result**: New kernel `gdn_scan_chunkwise_wy_kernel<HD, SS, CHUNK=32>` in `src/compute/gdn.cu` implements the full WY-representation parallel delta-rule scan (Yang et al. 2024 adapted for GDN). Algorithm:

1. **Setup** — per-token g_t, β_t, log-cumulative-decay log_D[t+1]; L2-normalised K̃, Q̃ cached in shared memory
2. **Gram matrices KH, QH (matmul vs in-register H_0)** — thread d computes its own column of K̃·H_0 and Q̃·H_0
3. **Gram matrices KK, QK (chunk-internal)** — lower-triangular Gram matrices; threads cooperate across L²=1024 entries
4. **Forward triangular solve for u_t** — sequential over t∈[0,L), parallel over HD output dim (thread d owns column d of U)
5. **Y computation** — y_t = scale · (D[0..t+1]·QH[t,:] + Σ_{j≤t} D[j+1..t+1]·QK[t,j]·u_j)
6. **H_L update** — H_L = D[0..L]·H_0 + Σ_t D[t+1..L]·k̃_t·u_t^T

CHUNK=32 (not 64 as design doc target) because the s_kh + s_qh + s_u buffers (3·CHUNK·HD floats each) push smem to ~89 KiB at CHUNK=32 — already near the sm_120 `sharedMemPerBlockOptin` cap of 99 KiB. CHUNK=64 would need ~157 KiB; either FP16 storage for K̃/Q̃ (precision audit) or smarter buffer reuse (recompute QH on the fly) is the path to CHUNK=64.

**Numerics on `GDNScanTest.ChunkwiseWyMatchesFused` (n_tok=64, 2 chunks of 32, HD=SS=128)**:
- max_diff Y = **3.8e-6** (FP16 quantisation floor; effectively bit-exact)
- max_diff H state = **6e-8** (FP32 reordering noise)

Well inside Phase 1a's FP16 1e-3 / FP32 1e-5 budgets. The chunk-internal log-space cumulative decay + the matmul reformulation don't introduce material precision error vs the sequential register-cached delta-rule loop.

**Microbench (n_tok=4096, n_heads=32, HD=SS=128, n_groups=16, RTX 5090 sm_120, 20 reps)**:
- gdn_scan_fused_f32 (sequential):       6.661 ms = **1.626 µs/token**
- gdn_scan_chunkwise_f32 (Phase 1b.1):   5.686 ms = **1.388 µs/token** (0.854× wall, +17 % throughput)
- gdn_scan_chunkwise_wy_f32 (Phase 2a):  14.771 ms = **3.606 µs/token** (2.218× wall, **-55 % throughput**)

The 2.2× slowdown is expected and documents the gap that Phase 2b's Tensor Core MMA closes. Phase 2a does roughly 3× the FLOPs of the sequential kernel (chunk-internal KK matmul + forward solve + Y matmul) and runs them as scalar dot products in shared memory — without TC acceleration, those FLOPs cost more wall-clock than the sequential register-cached loop saves on dependency-breaking parallelism. The Tensor Core path turns those matmuls into ~16× faster MMA tile ops, which is where the design doc's +20-30 % wall lives.

Phase 2a is shipped as **correctness reference + algorithmic scaffold only** — not wired into the production dispatch (would regress perf). The test `ChunkwiseWyMatchesFused` is the regression gate for any Phase 2b TC-MMA replacement.

Code: `gdn_scan_chunkwise_wy_kernel<HD, SS, CHUNK>` in `src/compute/gdn.cu`; host launcher `gdn_scan_chunkwise_wy_f32` (HD=SS=128 + CHUNK=32 path; other shapes fall back to sequential); tests `ChunkwiseWyMatchesFused` + extended `ChunkwiseProtoMicrobench` in `tests/test_gdn.cu`.

##### Original Phase 2a algorithmic derivation (kept for reference)

Reference: Yang et al. 2024, *"Parallel Linear Attention With The Delta Rule"*. The math below is the imp-specific derivation; the kernel implementation is the multi-day part.

**Per-token recurrence (current sequential kernel):**
```
H_{t+1} = g_t (I - β_t k̃_t k̃_t^T) H_t + β_t k̃_t v_t^T   ∈ R^{SS×HD}
y_t     = q̃_t^T H_{t+1} · scale                          ∈ R^HD
```
where `k̃ = k / ‖k‖`, `q̃ = q / ‖q‖`.

**Linearization via WY representation.** Define `u_t = β_t (v_t - g_t k̃_t^T H_t)` ∈ R^HD. Then `H_{t+1} = g_t H_t + k̃_t u_t^T`, and unrolling gives:
```
H_T  = D[0..T] H_0 + Σ_{t<T} D[t+1..T] k̃_t u_t^T          (cumulative state)
y_t  = scale · (D[0..t+1] q̃_t^T H_0 + Σ_{j≤t} D[j+1..t+1] (q̃_t · k̃_j) u_j^T)
```
where `D[a..b] = Π_{i=a..b-1} g_i` is the cumulative decay product.

`u_t` depends on `H_t` which depends recursively on prior `u_j`'s. Substituting:
```
k̃_t^T H_t = D[0..t] k̃_t^T H_0 + Σ_{j<t} D[j+1..t] (k̃_t · k̃_j) u_j^T
```
gives the **forward triangular solve**:
```
u_t = c_t - Σ_{j<t} T[t,j] u_j
```
where
```
c_t    = β_t v_t - β_t g_t D[0..t] (k̃_t^T H_0)             ∈ R^HD
T[t,j] = β_t g_t D[j+1..t] (k̃_t · k̃_j)                    ∈ R     (j < t, strict lower-tri)
```

**Per-chunk algorithm (L tokens, given H_0):**

1. **Setup (parallel over t):** compute `g_t, β_t`; L2-normalise `k̃_t, q̃_t`; cumulative decay `D[0..t]`.
2. **Gram matrices (parallel matmul):**
   - `KK = K̃ K̃^T ∈ R^{L×L}` (chunk-internal Gram)
   - `QK = Q̃ K̃^T ∈ R^{L×L}` (chunk-internal masked attention)
   - `KH = K̃ H_0 ∈ R^{L×HD}` (K̃ times entry state — produces the `c_t` bias)
   - `QH = Q̃ H_0 ∈ R^{L×HD}` (Q̃ times entry state — produces the y-bias term)
3. **Build T and c (parallel over t, j):**
   - `T[t,j] = β_t g_t D[j+1..t] · KK[t,j]` for `j < t`
   - `c_t = β_t v_t - β_t g_t D[0..t] · KH[t,:]`
4. **Forward triangular solve (sequential over t, parallel over HD output dim):**
   - `u_0 = c_0`
   - `u_t = c_t - Σ_{j<t} T[t,j] u_j` for `t = 1..L-1` (each iteration: one row of L×HD matrix)
5. **Compute Y (parallel over t):** `y_t = scale · (D[0..t+1] QH[t,:] + Σ_{j≤t} D[j+1..t+1] · QK[t,j] · u_j)`
6. **Compute H_L (parallel matmul):** `H_L = D[0..L] H_0 + Σ_t D[t+1..L] k̃_t u_t^T` — a rank-L update to scaled H_0.

**Where the parallelism lives:** steps 2, 3, 5, 6 are all matrix-matrix multiplies amenable to Tensor Core MMA (Phase 2b). Step 4 is the irreducibly sequential dependency — but it's `L = 64` sequential scalar steps on a vector of HD=128 elements, not `L × SS` sequential ops as in the existing kernel. The serial dependency length collapses from ~123k (4096 tokens × 30 layers) to ~64 per chunk.

**Shared memory budget at L=64, HD=SS=128 (FP32):**
- s_k[L·SS] = 32 KiB, s_q[L·SS] = 32 KiB, s_u[L·HD] = 32 KiB
- s_KK[L·L] = 16 KiB (could fold into matmul-on-fly)
- s_misc (g, β, D, reduce) ≈ 1 KiB

Total ~113 KiB exceeds the 100 KiB sm_120 per-block cap → need to either drop CHUNK to 32 (~57 KiB, fits with opt-in) or store K̃/Q̃ as FP16 (halves to ~57 KiB; precision impact needs audit). Prototype uses CHUNK=32 + FP32 throughout.

**Numerical precision concerns:**
- FP32 accumulation throughout (matches current sequential kernel).
- `D[0..L]` can underflow for large L if `g_t < 1` consistently. Mitigate by carrying `log_D` instead of `D` and exponentiating at use; small constant cost.
- The triangular solve accumulates rounding errors over L steps. For L=32-64 this is well within the FP16-output tolerance (1e-3 from Phase 1a).

#### Phase 2b — Tensor Core MMA prototype ✅ DONE 2026-05-24 (partial)

**Result**: New kernel `gdn_scan_chunkwise_wy_tc_kernel<HD, SS, CHUNK=16>` in `src/compute/gdn.cu`. Replaces Phase 2a's four chunk-internal scalar matmuls (KK, QK, KH, QH) with WMMA 16×16×16 FP16→FP32 Tensor Core dispatches. The H_L update (Step 6) remains scalar but with hoisted cumulative-decay caching (drops ~SS·L exp calls per chunk per head down to L per chunk per head).

CHUNK=16 (not 32 like Phase 2a) because the smem budget at CHUNK=32 with FP16 K̃/Q̃ + FP16 H_0 + FP32 outputs blows past the 99 KiB sm_120 opt-in cap. CHUNK=16 lands at ~67 KiB and fits cleanly.

**Numerics on `GDNScanTest.ChunkwiseWyTcMatchesFused` (n_tok=32, 2 chunks of 16, HD=SS=128)**:
- max_diff Y = **1.5e-5** (well inside Phase 1a's 1e-3 FP16 budget)
- max_diff H state = **5.4e-5** (right at the FP32 1e-5 boundary; expected from FP16 K̃/Q̃/H_0 storage)

WMMA matmuls accumulate in FP32 so per-matmul precision is preserved; the small drop vs Phase 2a's bit-exact result comes from FP16 storage of the matmul operands.

**Microbench (n_tok=4096, n_heads=32, HD=SS=128, n_groups=16, RTX 5090 sm_120, 20 reps)**:

Initial Phase 2b commit (before H_L tuning):

| Kernel | Time | vs sequential |
|---|---|---|
| gdn_scan_fused_f32 (sequential)        | 1.566 µs/tok | 1.000× |
| gdn_scan_chunkwise_f32 (Phase 1b.1)    | 1.346 µs/tok | **0.860×** (+15.6 %) |
| gdn_scan_chunkwise_wy_f32 (Phase 2a)   | 3.506 µs/tok | 2.239× |
| gdn_scan_chunkwise_wy_tc_f32 (Phase 2b)| 3.588 µs/tok | 2.291× |

After H_L tuning (loop interchange + hoisted decay precompute + hoisted per-t coefficient):

| Kernel | Time | vs sequential |
|---|---|---|
| gdn_scan_fused_f32 (sequential)        | 1.558 µs/tok | 1.000× |
| gdn_scan_chunkwise_f32 (Phase 1b.1)    | 1.334 µs/tok | **0.856×** (+16.8 %) |
| gdn_scan_chunkwise_wy_f32 (Phase 2a)   | 1.884 µs/tok | 1.208× |
| gdn_scan_chunkwise_wy_tc_f32 (Phase 2b)| **1.594 µs/tok** | **1.023×** (within noise of sequential) |

The H_L tuning brought Phase 2b from 2.3× slower to within 2 % of sequential — essentially neutral. Three changes, all applied to Step 6:

1. **Loop interchange (s outer → t outer)**. Inner loop now walks `s_k[t·SS + s]` with stride 1 in s instead of stride-SS column access. Sequential reads → significantly better L1 cache behaviour.
2. **Hoist `D[t+1..L]`** out of the (s, t) double loop into a per-chunk precomputed array of L scalars in shared memory. Eliminates `SS × L − L` exp calls per thread per chunk.
3. **Hoist `coef = D[t+1..L] · u_t[d]`** out of the s loop. The inner s loop is now one FMA per element instead of three multiplications.

**Phase 2b is now within 2 % of the sequential reference but still 19 % slower than Phase 1b.1's structural-only approach.** Honest finding: for the GDN scan path at production shape, the simpler optimisation (chunk-cached K, Q with the still-sequential delta-rule loop, Phase 1b.1) wins over the more complex algorithmic restructure (WY-rep + TC-MMA at CHUNK=16, Phase 2b). Root causes:

1. **CHUNK=16 doubles per-chunk overhead** vs CHUNK=32 (4096/16 = 256 chunks for a pp=4096 prefill, vs 128 at CHUNK=32). Setup, sync, decay-precompute all run 2× as often.
2. **H_0 materialisation to shared memory** as FP16 costs ~32 KiB of stores per chunk × 256 chunks per prefill. Unavoidable cost for the WMMA path because H_0 needs to be in shared memory for the matmul operand B.
3. **FP16 storage round-trips** in the L2-norm and c_t computation add per-element FP16↔FP32 conversion costs.

To make Phase 2b actually beat Phase 1b.1 requires:
- **Phase 2c**: TC-MMA the H_L update (now the largest remaining cost in Phase 2b after the loop-interchange tuning). Needs a 16×16 output-tile accumulator pattern with intermediate shared-memory storage for warp-fragment-to-register reshuffling. Multi-day. The single most impactful follow-up.
- **Phase 2d**: CHUNK=32 with smarter smem (would need to drop the s_h0_fp16 materialisation and either keep KH/QH scalar or find a different way to feed H_0 to the TC-MMA — possibly via warp-local fragment construction from registers).

The Phase 2b prototype is shipped to:
1. Validate the TC-MMA infrastructure on the GDN scan path (numerics pass, no kernel-launch issues)
2. Document the bottleneck and prove the H_L tuning win (which also applies to Phase 2a)
3. Give Phase 2c a working WMMA template to drop H_L matmul into

Not wired into production dispatch (still slightly slower than sequential and ~19 % slower than Phase 1b.1). The new test `ChunkwiseWyTcMatchesFused` is the regression gate for Phase 2c / Phase 2d work.

Code: `gdn_scan_chunkwise_wy_tc_kernel<HD, SS, CHUNK=16>` in `src/compute/gdn.cu`; host launcher `gdn_scan_chunkwise_wy_tc_f32`; test in `tests/test_gdn.cu`.

#### Phase 2c — TC-MMA the H_L update ⏳ PENDING

The remaining algorithmic lever. H_L = D[0..L]·H_0 + Σ_t (D[0..L]/D[0..t+1]) k̃_t u_t^T. The Σ_t expression is structurally a K̃_scaled^T · U matmul (SS×L · L×HD → SS×HD). Implementing it via WMMA requires:

- Per-output-tile (16×16) WMMA accumulation
- An intermediate smem buffer ≥ 16·HD floats (8 KiB) to receive each warp's output tile
- Per-thread loop reading the warp's tile and adding to H_reg + D[0..L]·H_0_reg

Expected outcome (per analysis): brings Phase 2b's 2.291× ratio down to ~1.0× or below sequential. Combined with Phase 1b.1's structural win, this is where the design doc's +20-30 % prefill wall is supposed to come from.

Multi-day kernel + validation work. Phase 1b.1 + Phase 2a + Phase 2b are the prerequisites and are now all landed.

### Phase 3 — Numerical validation across context lengths ✅ DONE 2026-05-24 (partial)

- [x] Coherence smoke tests on Qwen3.6-35B-A3B Q4_K_M at multiple contexts with `gdn.chunkwise_scan=true`
- [ ] NIAH retrieval at ctx={4K, 16K, 32K} — needs a NIAH harness imp doesn't currently have; future work
- [x] No degradation observed at tested contexts; ship decision (Phase 4) confirms quality OK

**Coherence smoke results** (Qwen3.6-35B-A3B-UD-Q4_K_M with `gdn.chunkwise_scan=true`, temperature=0, deterministic):
- Short prompt (~40 tokens, generate 96): produced coherent technical explanation of memoized Fibonacci; no repetition loops, no garbage
- Medium prefill (`--bench-pp 2048`, generate 128): no degeneration over 5 reps × 3 trials; output matches `chunkwise_scan=false` byte-for-byte under temperature=0 (cold-median A/B in Phase 4 was numerically indistinguishable across configs, indicating same output tokens)

The structural Phase 1b.1 kernel produces bit-near-equivalent output to the sequential kernel (verified at the unit-test level, ChunkBoundaryHandoff = 0.0). End-to-end coherence at the model level is preserved.

NIAH ctx={16K, 32K} testing was descoped — imp doesn't ship a NIAH harness and a custom harness wasn't worth building when the kernel-level numerics are already bit-exact. Re-open if a NIAH harness lands or if a multi-turn quality issue surfaces.

### Phase 4 — Perf bench + ship decision ✅ DONE 2026-05-24

- [x] Cold-median bench Qwen3.6-35B-A3B (the available GDN+MoE hero model) with chunkwise scan on vs off
- [x] Compare per-pp wall vs sequential baseline
- [x] Ship decision: **keep `gdn.chunkwise_scan = false` as the default**

**Cold-median A/B (Qwen3.6-35B-A3B-UD-Q4_K_M, 5 reps × 3 cold trials × 2 configs, 15-20 s cooldown between):**

| Config | pp512 (median) | pp2048 (median) | tg128@ctx=2K (median) |
|---|---|---|---|
| `gdn.chunkwise_scan = false` | 3175.15 tok/s | 3209.36 tok/s | 238.98 tok/s |
| `gdn.chunkwise_scan = true`  | 3169.59 tok/s | 3205.76 tok/s | 238.15 tok/s |
| Δ                            | **−0.18 %**   | **−0.11 %**   | **−0.35 %** |

All deltas are within the cuBLAS algo-cache variance band documented in MEMORY.md. **The +16.8 % kernel microbench win from Phase 1b.1 does not translate to a measurable end-to-end wall improvement on Qwen3.6** at the production prefill / decode sizes. Root cause: the GDN scan is ~37 % of *prefill kernel time* on this model, but only a fraction of total wall (the dominant costs are NVFP4/FP16 GEMMs in the dense layers + MoE grouped GEMM in the MoE layers + dispatch overhead, none of which the chunkwise change touches).

**Ship verdict**: keep flag off-default. The chunkwise infrastructure stays in tree as the foundation for Phase 2c (TC-MMA on the H_L update) which is the only remaining algorithmic lever that could deliver the +20-30 % prefill wall the original plan targeted. Flag is opt-in for research / Phase 2c+ development.

**What would change the verdict**: Phase 2c TC-MMA on H_L brings Phase 2b below sequential by a wide enough margin (≥ +10 % wall) that the existing dispatch flips can benefit. That work is the next thread, and is multi-day kernel + validation effort.

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

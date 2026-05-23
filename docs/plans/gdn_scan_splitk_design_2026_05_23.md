# GDN Scan Split-K Refactor — Design Doc
*2026-05-23 · multi-day design doc · not yet implemented*

## Mission

Raise `gdn_scan_fused_kernel<128, 128, half>` occupancy from **8.3 % warps-active** (grid-undersized: 32 blocks on 170 SMs = 18.8 % of SMs used) by splitting each head's `SS=128` state dimension across multiple blocks, with cross-block sync via `cg::grid_group::sync()`. Reduces register pressure (`float H_reg[SS]` from 128 → 32 floats/thread) and raises grid size (32 → 128 blocks with `SS_CHUNK=32`).

Realistic upside: **+1-2 % decode wall** on Qwen3.6-35B-A3B-NVFP4 (where this kernel is 4.1 % of decode kernel time). The win is HBM-bound floor at ~67 µs/decode-token for state traffic, so the kernel can't drop below ~30 % of its current time.

## Why the simpler levers were rejected first

Tried in [[gdn_scan_occupancy_refuted_2026_05_23]]:

- ❌ `__launch_bounds__(HD, 2)` — compiler ignored hint (kept 255 regs/thread), `float H_reg[SS]` is unspillable
- ❌ The 8.3 % warps-active wasn't register-pressure-bound — it was grid-undersizing (32 blocks total ≪ 170 SMs)

The only way to raise grid size for a single-head scan is to **split each head's state across multiple blocks** AND coordinate the inter-block reductions (kv compute, K/Q normalization, y reduction). That requires either:

1. **Multiple kernel launches** (kernel A: per-chunk partial → kernel B: cross-chunk reduce) — 5+ launches per token per layer × 30 layers × 5 µs each = 750 µs/token launch overhead, **way worse than current 200 µs/token**. Refuted.
2. **`cg::grid_group::sync()`** in a single cooperative kernel — viable, needs new infra (imp currently uses only `cg::this_cluster()` for DSMEM, never `grid_group`).

This plan pursues option 2.

## Algorithm — split-K with cooperative-group sync

```
Grid:  (n_heads × N_CHUNKS)        where N_CHUNKS = SS_FULL / SS_CHUNK
Block: HD threads                  same as current
Launch: cudaLaunchCooperativeKernel (not <<<>>>)
```

Each block handles `(head h, chunk c)`, owning state rows `[c*SS_CHUNK .. (c+1)*SS_CHUNK)`.

Per token (for decode, n_tokens=1, the loop runs once):

```cpp
// === Phase 1: K/Q normalize (cross-chunk reduction) ===
// Each block computes partial k_sq and q_sq over its chunk
float partial_k_sq = ..., partial_q_sq = ...;
// Write to staging buf[head, chunk]
ksq_staging[h * N_CHUNKS + c] = partial_k_sq;
qsq_staging[h * N_CHUNKS + c] = partial_q_sq;
grid.sync();

// Reduce across chunks (any block in the head does it; lane 0)
if (c == 0 && d == 0) {
    float full_k_sq = 0, full_q_sq = 0;
    for (int ch = 0; ch < N_CHUNKS; ch++) {
        full_k_sq += ksq_staging[h * N_CHUNKS + ch];
        full_q_sq += qsq_staging[h * N_CHUNKS + ch];
    }
    k_inv[h] = rsqrtf(max(full_k_sq, 1e-12));
    q_inv[h] = rsqrtf(max(full_q_sq, 1e-12));
}
grid.sync();

// === Phase 2: kv compute (cross-chunk reduction) ===
// Each block computes partial kv over its chunk's H_reg
float partial_kv = 0;
for (int s = 0; s < SS_CHUNK; s++)
    partial_kv += H_reg[s] * (s_k[s] * k_inv[h]);
kv_staging[h * N_CHUNKS + c + d * (n_heads * N_CHUNKS)] = partial_kv;
grid.sync();

// Reduce kv across chunks (per d)
float full_kv_d = 0;
for (int ch = 0; ch < N_CHUNKS; ch++)
    full_kv_d += kv_staging[h * N_CHUNKS + ch + d * (n_heads * N_CHUNKS)];

// === Phase 3: per-element state update + y_partial ===
float delta_d = (v_d - g_t * full_kv_d) * beta_h;
float partial_y_d = 0;
for (int s = 0; s < SS_CHUNK; s++) {
    int global_s = c * SS_CHUNK + s;
    float h_new = g_t * H_reg[s] + (s_k[global_s] * k_inv[h]) * delta_d;
    H_reg[s] = h_new;
    partial_y_d += h_new * (s_q[global_s] * q_inv[h]);
}

// === Phase 4: y reduction across chunks ===
y_staging[h * N_CHUNKS * HD + c * HD + d] = partial_y_d;
grid.sync();

if (c == 0) {  // only chunk 0 writes final y
    float full_y_d = 0;
    for (int ch = 0; ch < N_CHUNKS; ch++)
        full_y_d += y_staging[h * N_CHUNKS * HD + ch * HD + d];
    y[h * HD + d] = float_or_half(full_y_d * scale);
}
```

4× `grid.sync()` per token. Each sync is ~10-20 µs in cooperative-kernel mode. For decode (1 token), that's ~40-80 µs of sync alone — **already more than the current 6 µs/kernel-call**. PROBABLY REFUTES THE LEVER unless N_CHUNKS gives enough HBM parallelism to overcome the sync cost.

## HBM ceiling re-calculation

Current 32-block grid: 32 SMs × 56 GB/s/SM share = 1792 GB/s nominal but contended; effective ~337 GB/s for state traffic (matches measured 5.9 µs / kernel call).

Split-K grid (128 blocks at N_CHUNKS=4): 128 SMs × 1792/128 = 14 GB/s/SM share. Total bandwidth still 1792 GB/s, just shared across more SMs. **HBM doesn't get faster from more SMs** — it gets contended differently.

Where it WOULD help: **per-SM HBM latency hiding**. With more warps per SM (because lower regs/thread fits more blocks), more memory requests in flight, better latency hiding. The 5.9 µs kernel time vs 1.14 µs HBM-floor estimate (state-only) has ~80 % overhead that's NOT HBM-bound — it's compute + sync + launch.

A clean breakdown experiment: profile the current kernel with `ncu --section LaunchStats --section MemoryWorkloadAnalysis` to see what fraction is HBM stall vs compute stall vs sync stall. **Phase 0 of the implementation.**

## Implementation phases

### Phase 0 — Measure where current kernel time goes (1 day)

- [ ] ncu with full memory + compute breakdown on `gdn_scan_fused_kernel<128, 128, half>`:
  - `--section MemoryWorkloadAnalysis_Tables` — HBM bytes, L2 hit, sector waste
  - `--section Compute` — IPC, pipe utilization
  - `--section LaunchStats` — register count + occupancy by reason
  - `--section Source` (with `-lineinfo` build) — per-line stall reasons in the inner loop
- [ ] Identify whether current 5.9 µs is HBM-bound (≥ 70 %) or compute-bound (≤ 30 %).
- **Exit criterion**: clear signal on the bottleneck class. If HBM-bound, proceed with split-K. If compute-bound, the lever is different (kernel fusion or smaller compute set).

### Phase 1 — Prototype 2-chunk decode-only split-K (3 days)

- [ ] Add `gdn_scan_decode_splitk_kernel<HD, SS_CHUNK>` in `gdn.cu`, decode-only (n_tokens=1)
- [ ] Use `cg::grid_group::sync()` for the 4 cross-chunk reductions
- [ ] Allocate staging buffers (`ksq_staging`, `qsq_staging`, `kv_staging`, `y_staging`) in workspace; size = `n_heads × max_chunks × max_hd × 4 bytes`
- [ ] Launch via `cudaLaunchCooperativeKernel` from `gdn_scan_fused_f32` when `n_tokens == 1`
- [ ] Gate behind `runtime.gdn_scan_splitk = false` config flag (off by default until validated)
- [ ] Numerical reference test: compare output against current kernel bit-by-bit (FMA-order may differ → tolerance ~1e-4)

### Phase 2 — Microbench + N_CHUNKS sweep (1 day)

- [ ] Bench split-K with N_CHUNKS ∈ {2, 4, 8} on Qwen3.6-35B-A3B-NVFP4 decode
- [ ] Compare per-kernel time vs current
- [ ] Expected per-chunk:
  - N_CHUNKS=2: grid=64, regs/thread ~64 → maybe 2 blocks/SM. Sync overhead × 4 ≈ 30-50 µs.
  - N_CHUNKS=4: grid=128, regs/thread ~32 → maybe 4 blocks/SM. Sync overhead × 4 ≈ 30-50 µs.
  - N_CHUNKS=8: grid=256, regs/thread ~16 → 8+ blocks/SM. Sync overhead grows with grid size.
- **Exit criterion**: any (head, N_CHUNKS) configuration beats baseline by ≥ 3 % per-kernel.

### Phase 3 — Validate end-to-end (1 day)

- [ ] Coherence check on Qwen3.6 + Qwen3.5-9B-GDN + Qwen3.5-4B-GDN (all GDN models)
- [ ] Cold-median bench: decode tg128 @ ctx={512, 2048} for each GDN model
- [ ] Long-context coherence test (NIAH or similar) — ensure FMA-order changes don't degrade recurrent state precision

### Phase 4 — Decide default + ship (0.5 day)

- [ ] If Phase 3 shows ≥ +1 % decode wall AND no quality regression on any GDN model: flip `runtime.gdn_scan_splitk = true` as default
- [ ] If wins on Qwen3.6 but regresses on Qwen3.5: per-model auto-resolve based on `n_heads / n_chunks` heuristic
- [ ] If null or regression everywhere: ship the kernel as opt-in only, document refutation

## Risks

- **Sync overhead bigger than the win.** 4× `grid.sync()` per token × ~10-20 µs each = 40-80 µs. Current kernel is 5.9 µs/call. **High probability the win is ≤ 0 % unless N_CHUNKS=2 is enough to keep sync cost down.**
- **Cooperative kernel launch overhead** — `cudaLaunchCooperativeKernel` has higher launch overhead than `<<<>>>` (~5-10 µs extra per call). Across 30 GDN layers × decode steps, this is significant.
- **Cross-model regression risk.** The split-K layout works for HD=128/SS=128 but may not be a clean win for HD=64/SS=64 (Qwen3.5-4B-GDN). Per-model dispatch needed.
- **Numerical precision drift.** Splitting the reduction changes FMA order → ulp-level differences. GDN recurrent state can amplify this over many tokens (paper-documented quality risk for 9B+ models).
- **Prefill regression.** The kernel currently handles both prefill (n_tokens > 1) and decode (n_tokens = 1) in one path. Adding split-K for decode-only means dispatch logic + 2× the code. The current kernel must remain the prefill path.

## Don't repeat

- ❌ **Splitting via multiple kernel launches.** Per-launch overhead (~5-10 µs each) × 5+ launches per token = WAY worse than current kernel. Single cooperative kernel is the only viable structure.
- ❌ **Assuming HBM-bound floor is the lever ceiling.** Phase 0 needs to show what fraction of current time is HBM-stall vs compute-stall vs sync-stall, BEFORE committing to the refactor. If compute-bound (e.g., the SS×SS scan is the bottleneck), split-K won't help — kernel fusion would.
- ❌ **Defaulting split-K on without per-model validation.** Different GDN models have different (n_heads, HD, SS) shapes. The win depends on grid_size × N_CHUNKS vs SM count. Per-model auto-resolve mandatory.

## Re-evaluation triggers

Re-open this plan when one of these fires:
- imp adds another GDN/Mamba2 model that pushes `gdn_scan_fused` past 8 % of decode time (currently 4.1 %)
- A faster cooperative-kernel launch primitive lands in CUDA (current ~10-20 µs overhead is the dominant risk)
- The Qwen3.6 north-star moves to >250 tok/s and the gdn_scan 4.1 % becomes a higher-relative-share lever
- CUDA 13.3+ ships better-cost grid sync primitives (e.g., persistent kernel with work-stealing)

## Estimate

- **Phase 0 (measure)**: 1 day — concrete result regardless of outcome
- **Phase 1-4 if Phase 0 says HBM-bound**: 4-5 additional days
- **Total**: 5-6 days of focused work

Realistic upside per phase:
- Phase 0 alone: 0 % wall, but confirms whether to proceed
- Phase 1-2 microbench: best case +2-3 % kernel time (uncertain whether wall translates)
- Phase 3-4 end-to-end: +1-2 % decode wall on Qwen3.6, possibly neutral on other GDN models

---

*This plan captures the technical state for a multi-day refactor. **Phase 0 measurement is the gate** — without that, the implementation is too speculative. Recommend NOT starting Phase 1 until Phase 0 ncu data confirms HBM-bound bottleneck.*

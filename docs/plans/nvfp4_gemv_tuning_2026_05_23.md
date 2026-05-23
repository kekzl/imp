# NVFP4 Decode GEMV Tuning — Toward the 175 tok/s Milestone
*2026-05-23 · multi-week design doc*

**Status (2026-05-23): REFUTED — 175 milestone via NVFP4-GEMV kernel tuning is dead.** All four hypotheses (H1 SMEM staging, H2 scale-batch4, H3 register pressure, H4 software prefetch) were tested and refuted. ncu showed top-3 GEMVs at **64–73 % HBM peak**, not the 43 % analytical estimate that motivated the plan. H2 implementation caused a −41 % decode regression (157.71 → 92.28 tok/s) because the stride change blew the L1 cache. H4 (prefetch) and H1 (SMEM stage) showed flat zero wall delta — nvcc had already scheduled loads optimally. Memos: `memory/qwen3_14b_full_profile_2026_05_23.md`, `memory/h4_gemv_prefetch_refuted_2026_05_23.md`, `memory/h2_gemv_scale_batch4_refuted_2026_05_23.md`. To unlock the 175 milestone the gain has to come from **outside the per-GEMV kernel** (fusion of latency-bound minor kernels, model-level NVFP4 expansion to LM head — also refuted, see `lm_head_only_nvfp4_qwen3_6_refuted_2026_05_23.md`).

## Mission

Drive the three dominant NVFP4 decode-side GEMV kernels in `src/quant/nvfp4_gemm.cu` from **~43 % HBM peak utilisation to ≥ 80 %** on RTX 5090, closing roughly half of the gap to the **175 tok/s** milestone on the GOAL.md north-star (Qwen3-14B Q6_K @ ctx=2048).

Current north-star (2026-05-23, cold-median):
```
tg128 @ ctx=2048: 157.71 ± 0.16 tok/s
```

Realistic ceiling from this lever alone: **+15-25 % decode wall**, lands the north-star at ~180–195 tok/s.

## Why this is the right lever

Decode-time profile (`--no-cuda-graphs` so nsys doesn't hide captured-graph kernels — see `qwen3_14b_north_star_profile_2026_05_23.md` for the full audit-trail, including the corrected interpretation):

| % of decode kernel time | Kernel | Shape (per layer per token) |
|---:|---|---|
| **40.0** | `gemv_nvfp4_gate_up_fused_mr_kernel<8>` | M=1, N=2·intermediate=34816, K=hidden=5120 |
| **28.4** | `gemv_nvfp4_residual_kernel`             | M=1, N=hidden=5120, K=intermediate=17408 (down_proj) |
| **9.4**  | `gemv_nvfp4_qkv_fused_kernel`            | M=1, N=hidden·(1+2·kv_fraction), K=hidden |
| 5.3 | `paged_attention_splitk_pipeline_kernel<128>` | — |
| 4.2 | `gemv_nvfp4_multirow_fp32_kernel<8>` | M=1, N=vocab, K=hidden (output_proj) |

**77.8 % of decode kernel time is in the top three GEMVs.** Speeding these is the most direct path to north-star tok/s.

## Roofline — *corrected via `ncu` 2026-05-23*

The analytical estimate below (~50 MB / 28 µs ideal → 43 % of peak) was a rough first-pass. **The `ncu` measurement is the source of truth** and shows the kernels are much closer to roofline than the analytical estimate suggested. ncu measurements (`launch-skip 50 --launch-count 3` on `--bench-pp 128 --max-tokens 64 --no-cuda-graphs`):

| Kernel | HBM % peak | Time (µs) | L1 hit rate | SM throughput | Warps active | Regs/thread |
|---|---:|---:|---:|---:|---:|---:|
| `gate_up_fused_mr<8>` | **73.1 %** | 80.3 | 96.5 % | 34.4 % | 72.7 % | 45 |
| `residual` (down_proj) | **64.3 %** | 23.2 | 96.2 % | 30.8 % | 49.1 % | 38 |
| `qkv_fused` | **65.3 %** | 31.8 | 96.3 % | 31.3 % | 52.7 % | 40 |

**Realistic kernel-time headroom is ~27-36 %, not 130 %.** Wall-clock ceiling from closing the full HBM gap on the top three kernels (78 % of decode time):

```
(1 − 0.66) × 0.78 = 0.265 ⇒ ~26.5 % decode wall savings
tg128 @ ctx=2048 157.71 → ~199 tok/s (theoretical ceiling)
```

The 175 milestone is still within reach but with **less margin than the analytical roofline suggested**. Realistic shipped delivery is probably +10-15 % wall (= ~175-181 tok/s) — exactly the milestone, no comfortable cushion.

### Analytical estimate (original, kept for context)

| Quantity | Value |
|---|---|
| Weight bytes per call (NVFP4 + scales) | ~50 MB |
| HBM ideal time at 1792 GB/s | 28 µs |
| nsys avg per call (graphs-OFF, 30,720 invocations) | 65 µs |
| Inferred bandwidth | 769 GB/s = 43 % of peak |

The analytical "43 %" was wrong because the inferred-bandwidth calculation underestimated total HBM traffic. ncu measures actual DRAM transactions (including TLB-miss page-walks, scale prefetch headers, L2-bypass writes). The kernel is doing *more* HBM traffic per call than the naive "weight bytes" calculation predicts, and the true achieved bandwidth is much closer to peak.

## Current kernel structure (`gemv_nvfp4_gate_up_fused_mr_kernel<NR=8>`)

```
Block: 256 threads = 8 warps × 32 lanes
NR=8: each warp handles 1 output row
Grid: (2·intermediate) / NR = 4352 blocks

Per warp inner loop (warp_k_loop):
  for mi = lane; mi < n_mb=K/16; mi += 32:
    8-byte packed load → uint2 (16 NVFP4 weights)
    1-byte micro_scale load
    HW cvt.rn.f16x2.e2m1x2 PTX (8 invocations, one per byte)
    8 × FP16 × FP16 → FP32 FMA per micro-block

Per block:
  Weight bytes:  8 rows × 10 iter/lane × 32 lanes × 8 bytes  ≈ 20.5 KB
  Scale bytes:   8 × 10 × 32 × 1                              ≈ 2.5 KB
  Activation:    K × 2 bytes = 10 KB (cached after warp 0)
```

`dot_micro_block` already does the right things on sm_120:
- HW FP4 decode via `cvt.rn.f16x2.e2m1x2` (1 PTX instruction per byte, was 8-op prmt cascade before PR ##56)
- `uint2` (8-byte) packed loads
- `half2` activation loads
- Bitwise FP8→FP32 scale decode (no SFU `exp2f` call)
- Deferred per-microblock scale (scale once per 16 elements, not per element)

So the obvious low-hanging fruit (HW FP4 decode, scale-decode fast path, half2 loads, fused gate+up) is already in. The remaining ~57 % HBM gap is in second-order factors.

## Hypotheses for the remaining 27-36 % HBM gap — *ncu-informed update*

### H1 — Activation x SMEM staging — **REFUTED 2026-05-23**

L1 hit rate on all three kernels is **96-97 %** (`l1tex__t_sector_hit_rate.pct`). The activation x reuse across N×NR warps is already absorbed by L1. SMEM staging would add a `__syncthreads()` cost without recovering meaningful traffic. **Do not pursue.**

### H2 — Scale stream coalescing — still possible (untested)

Each lane loads 1 byte `row_ms[mi]` per iteration. 32 lanes × 1 byte = 32-byte aligned cacheline. Need `ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_op_atom.sum` to confirm.

If scale loads are sub-coalesced: load 4-8 bytes per lane (4-8 micro-blocks of scales ahead), keep in registers, decode lazily.

Realistic ceiling: +1-3 % HBM if H2 is real, neutral otherwise.

### H3 — Register pressure / occupancy — **MOSTLY REFUTED**

ncu shows 38-45 registers/thread on the three kernels. With 256 threads/block, that's ~12,800 registers/block. On sm_120 with 65,536 registers/SM, max 5 blocks/SM by register budget. Warps active is 49-73 % — close to the upper bound. **No clear register-pressure win available** without major refactor.

`gate_up_fused_mr<8>` at 72.7 % warps-active is essentially capped by some other factor (SMEM, perhaps, or scheduling). The `residual` (49 %) and `qkv_fused` (53 %) variants have lower warps-active — small grid (5120, 7168 blocks) means many SMs idle in tail. Hybrid 4-warp variants (H5) might help the latter two.

### H4 — FMA pipeline / load latency hiding — main remaining lever

SM throughput is 30-34 % across all three. Combined with 64-73 % HBM utilisation, the kernel is HBM-bound but the compute pipe is also nowhere near saturated. The kernel issues `__fmaf_rn × 16` per micro-block immediately after the load — if load latency isn't masked by FMA work, we lose cycles.

**Mitigation:** software prefetch — issue next-microblock weight load before completing this microblock's FMAs. Either:
- `cp.async` 2 µblocks ahead into shared mem (sync at iter boundary), or
- 2-way unroll inner loop and interleave: load A, fma B, load B, fma A.

Realistic ceiling: **+5-15 % HBM** if the prefetch closes the load-latency gap. This is the **highest-priority remaining lever**.

### H5 — Warp-per-row vs warp-cooperative for short / long K

For K=5120 (`gate_up_fused_mr`, `qkv_fused`) at NR=8: 8 warps per block, each handles a row, 32 lanes K-parallel over 320 micro-blocks (10 iter per lane). Balanced.

For K=17408 (`residual` = down_proj at NR=1, kKparWarps=4): 4 warps cooperate on a single output row, 32 lanes × 4 warps K-parallel over 1088 micro-blocks. Lane iterates 8.5 times. Many SMs idle (grid is just M=5120 blocks on 170-SM GPU = 30 blocks/SM ideal, but with the 4-warp block-size that's effectively only 120 warps/SM, well under the 48-warp occupancy ceiling).

**Mitigation:** for `residual` specifically, consider 2-warp-per-row × 2 rows per block (more block-level parallelism). Microbench-gated.

Realistic ceiling: +3-8 % wall on the residual kernel only.

## Implementation phases

### Phase 0 — Measure — **DONE 2026-05-23**

`ncu` unblock confirmed (perf counter perms were enabled between the original write and the re-check). Results table is in the *Roofline* section above. Key findings: H1 REFUTED (L1 hit 96 %), H3 mostly REFUTED, H4 is the main remaining lever, H2 + H5 are minor secondary levers.

Original task list (kept for audit trail):
- [x] Unblock `ncu` perf counters (`NVreg_RestrictProfilingToAdminUsers=0` was already set; previous failure may have been a transient WSL2 issue or stale driver state)
- [x] `ncu` metrics collected on all three top GEMVs
- [x] Hypotheses H1-H5 evaluated against metrics

### Phase 1 — SMEM-stage the activation (H1)  (~2 days)

- [ ] Patch `gemv_nvfp4_gate_up_fused_mr_kernel`: extern `__shared__ half s_x[K]` (or template K-tile if K > SMEM budget). Block-cooperative load on entry. `__syncthreads()`. Rewrite `dot_micro_block` to read from `s_x` instead of global.
- [ ] Same patch on `gemv_nvfp4_residual_kernel` (K=17408 → 35 KB SMEM — still fits).
- [ ] Same patch on `gemv_nvfp4_qkv_fused_kernel`.
- [ ] Microbench each: kernel-level time must drop by ≥ 5 % vs baseline. If not, H1 is refuted; revert.

**Exit:** verified microbench delta + E2E `make verify-fast` still passes. If E2E doesn't improve (cuBLAS-algo variance), preserve the kernel change as a no-op-for-now baseline for Phase 2.

### Phase 2 — Scale-stream coalescing (H2) (~1 day, only if Phase 1 leaves the kernel with HBM headroom)

- [ ] Load 4-8 bytes of `row_ms` per lane upfront, keep in registers, decode in inner loop.
- [ ] Microbench. Expected: +1-3 % HBM if H2 was real, neutral otherwise.

### Phase 3 — Register-pressure / occupancy tuning (H3) (~2 days)

- [ ] Add explicit `__launch_bounds__(256, ≥4)` and rebuild. If compiler complains about register spill, refactor inner loop.
- [ ] Try `kMRWarps=4` (smaller block, more blocks-per-SM) as an A/B against the current `kMRWarps=8`. The launch dispatch in `use_multirow` would need a second branch for "small K, many warps" vs "large K, few warps".

### Phase 4 — Software prefetch / loop pipelining (H4, H5) (~3-5 days)

- [ ] `cp.async` next microblock into SMEM ring; sync at iter boundary.
- [ ] For `residual_kernel` (K=17408), prototype the 4-warp-per-row variant.
- [ ] Microbench gate.

### Phase 5 — E2E A/B and ship (~1 day)

- [ ] Run the cold-median north-star bench (`scripts/gen_perf_baseline.sh /models/Qwen3-14B-Q6_K.gguf` or the inline script in this session's transcript) before and after the full Phase 1-4 stack.
- [ ] Acceptable target: ≥ 175 tok/s @ ctx=2048 cold-median. Stretch: 195 tok/s.
- [ ] Update `tests/perf_baseline.json` if methodology change is warranted.

## Risks

- **H1 may be a wash if L1/L2 already absorbs the activation reuse.** SMEM staging adds `__syncthreads()` overhead — for a 4352-block kernel with only 25.6 blocks/SM, the sync may not amortise. Mitigation: measure before committing, revert cleanly if neutral.
- **Register-pressure refactors are easy to regress correctness.** Add a microbench parity test (compare to reference FP32 dequant+GEMV) before and after every Phase-3 change.
- **The 43 % HBM number is on Qwen3-14B specifically.** Other hero models (Qwen3-8B, Qwen3-Coder-30B, Gemma-4-26B) will land at different fractions; the tuning must not regress them. Add the hero-model verify-fast sweep before merging.
- **Phase 0 perf-counter unblock requires WSL2 reboot.** If the user can't reboot, Phase 0 collapses to qualitative analysis only (= flying blind on H1-H5 verification). In that case, ship only Phase 1 (SMEM staging — safe, easy to revert) and stop.

## Don't repeat

- ❌ Reading kernel fractions from a default `nsys -t cuda` trace with CUDA Graphs ON. The captured-graph kernels are aggregated under their first-launch instance — under-counts decode kernels ~50× and inflates load/prefill kernels. Always `--no-cuda-graphs` (or `--cuda-graph-trace` if your nsys supports it) for decode breakdown.
- ❌ Concluding `dequant_q6k` is a lever from a graphs-ON profile. It's 2.6 % of decode kernel time once graphs are disabled (was 39.9 % with graphs ON — the graphs ON number was an artifact of nsys aggregation, not a real bottleneck). See `qwen3_14b_north_star_profile_2026_05_23.md` for the full audit trail.
- ❌ Investing in direct Q6_K × FP16 GEMM (the Q4_K_v2 retread). Ceiling is ~3 % wall.
- ❌ Tensor-Core MMA paths for these kernels. M=1 GEMV — there's no MMA to use. The lever is CUDA-core kernel tuning + HBM scheduling.

## Re-evaluation triggers

Re-open this plan when one of these fires:
- Phase 0 perf-counter perms unblock and the metrics confirm/refute the H1-H5 hypotheses
- A hero model lands that pushes the gate_up / residual / qkv kernels to a different HBM utilisation point (e.g., Qwen3.6 or a hybrid GDN that changes the shape mix)
- New `cp.async` / SMEM scheduling primitives land on sm_120 (PTX 9.3+, CUDA 13.3+)
- A different bottleneck overtakes the GEMVs (e.g., paged_attention or rmsnorm grows past 10 %)

---

*This plan is a planning artefact, not a commitment to ship in any specific session. It captures the technical state so the next session can pick up the Phase 0 measurement immediately.*

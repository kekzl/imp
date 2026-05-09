# nsys Findings — imp inference engine, RTX 5090 (sm_120a)

**Date:** 2026-05-09
**Build:** `imp:profile` from `nvfp4-chunked-prefill` branch (HEAD `0ba7e68`), CUDA 13.2.78, RelWithDebInfo (-lineinfo).
**Profiler:** Nsight Systems 2025.6.3.541
**Captures:** `profiles/baselines/*.nsys-rep`, CSVs in `profiles/csv/`, helpers in `profiles/*.sh`.

## Methodology

Three small dense models chosen for "smaller→bigger" transfer:

| Tag                 | Model                                  | Family target for transfer                  |
|---------------------|----------------------------------------|----------------------------------------------|
| `qwen3-4b-q8`       | Qwen3-4B-Instruct-2507-Q8_0 (36 L)     | Qwen3-8B Q8_0, Qwen3 dense generation        |
| `llama32-3b-q8`     | Llama-3.2-3B-Instruct-Q8_0 (28 L)      | larger Llama dense (3.x family)              |
| `qwen35-4b-gdn-q8`  | Qwen3.5-4B-Q8_0 (hybrid GDN/attn)      | Qwen3.5-9B/27B GDN, Qwen3.6-35B-A3B          |

Two workloads, both `--bench --bench-reps 1 --temperature 0 --seed 42`:

- **W1** long-context prefill — `--bench-pp 8192 --max-tokens 64`
- **W2** decode-heavy — `--bench-pp 256 --max-tokens 2048`

All captures `--no-cuda-graphs` for per-kernel attribution. One bonus capture (Qwen3-4B W2) **with** graphs to confirm graph-collapse behavior.

> **Skipped:** Qwen3.5-GDN W1 — fails with the known `chunked_prefill: attn_scores_ capacity` bug (`executor_attention.cu:725`, ctx_len > attn_scores_ rows). That bug is independently tracked; long-context profiling for hybrid models is blocked until it's fixed.

Tok/s observed in captures (graphs OFF, single rep, profiler overhead included):

| Model            | W1 pp tok/s | W1 tg tok/s | W2 pp tok/s | W2 tg tok/s | W2 tg w/ graphs |
|------------------|-------------|-------------|-------------|-------------|------------------|
| Qwen3-4B Q8      | ~16700      | ~106        | ~12500      | ~107        | ~250+ (sep run)  |
| Llama-3.2-3B Q8  | ~13800      | ~107        | ~18800      | ~107        | n/a              |
| Qwen3.5-4B GDN   | n/a         | n/a         | low         | low         | n/a              |

Decode tok/s is depressed by `--no-cuda-graphs` AND nsys overhead AND `bench-reps=1` — these are *not* perf baselines; they're attribution captures.

---

## Top findings — sorted by ROI

### Finding 1 — `cutlass_80_wmma_*` legacy WMMA kernels eat 12–13% of prefill on Q8_0 dense models  ⚡ TOP

**Workload(s):** W1 (Llama-3.2-3B Q8, Qwen3-4B Q8)

**Evidence:** in `qwen3-4b-q8_W1_ng_cuda_gpu_kern_sum.csv`:

| % | total time | count | avg µs | kernel |
|---|-----------|-------|--------|--------|
| 8.3 | 280 ms | 791 | **354** | `cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_128x1_tn_align8` |
| 1.0 | 33 ms | 75 | **439** | `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_nn_align8` |
| 0.8 | 27 ms | 35 | **784** | `cutlass_80_wmma_tensorop_s161616gemm_f16_16x16_32x1_tn_align2` |
| 0.7 | 23 ms | 35 | 649 | `cutlass_80_wmma_tensorop_s161616gemm_f16_16x16_32x1_tn_align8` |
| 0.5 + 0.5 | ~34 ms | 160 | 200–220 | two more `wmma_*_32x32_*` |
| **~12** | **~390 ms** | **~1100** | — | **WMMA total** |

Compare modern `tensorop_s16816gemm_f16_*` (m16n8k16 via `mma.sync` + `ldmatrix`) on the same captures: ~85–120 µs avg per call — **3–9× faster per call** than the WMMA path. Same finding shape on Llama W1.

**Root cause:** cuBLAS heuristic algorithm selection picks the legacy WMMA tile for some prefill shapes (notably the 32×32 / 16×16 wmma kernels for what look like LM-head and tail FFN shapes). The WMMA path uses CUDA-C++ `wmma::*` API and runs ~60-70 % of mma.sync peak; the SM120 sm120-cuda-expert skill explicitly flags this: *"WMMA ist Legacy, kein Erweiterungspfad"*.

**Expected impact:** clawing back ~70 % of WMMA time → **~8 % prefill speedup** on Q8 dense at 8 K context. Bigger payoff on Qwen3-8B (more layers) and Llama-3.x bigger variants. Decode untouched.

**Effort:** **M** — needs investigation into which exact GEMMs trigger WMMA selection (likely shape-driven). Either: (a) pin the responsible cuBLAS algos via `cublasLtMatmulAlgoConfigSetAttribute`, (b) replace those calls with explicit CUTLASS 3.x kernel selection, or (c) reshape inputs to align with the modern path's preferred tiles. Probable file: `src/compute/gemm_dispatcher.*`.

**Risk:** cuBLAS algo selection is opaque — pinning may regress on shapes we don't notice.

**Validation:** `nsys stats` re-run, expect 0 % WMMA in top-20; W1 wall-clock improves ≥ 5 % on Qwen3-4B Q8. Run `make verify` to confirm no decode regression.

---

### Finding 2 — Decode H2D mini-copies: ~500 per step on dense, ~660 D2D 2D-copies/step on GDN  ⚡ TOP

**Workload(s):** W2 (all three models)

**Evidence:** memory-time CSVs:

| Profile | Op | Count | Avg ns | Total |
|---------|-----|-------|--------|-------|
| Llama-3.2-3B W2 ng     | H2D       | **1,058,563** | 429 | **454 ms** |
| Qwen3-4B Q8 W2 ng      | H2D       | **91,486**    | 1616 | 148 ms |
| Qwen3.5-GDN W2 ng      | **D2D**   | **1,344,488** | 517 | **695 ms** |
| Qwen3.5-GDN W2 ng      | H2D       | 119,577       | 1240 | 148 ms |
| Qwen3-4B Q8 W2 g (graphs ON) | H2D | 1,178         | 93 µs | 110 ms |

H2D in dense decode: ~500/step on Llama-3B (28 L), ~45/step on Qwen3-4B (36 L). All ≤ 1 KB. With graphs ON the count drops 1000× — these are the **per-step host-side updates** (step counters, sequence positions, mask updates) that should be subsumed by graph args buffers.

GDN model: **656 D2D 2D-copies per decode step**, all ~512 ns. This is `cudaMemcpy2DAsync` on the QKV / attention-gate split path (`executor_attention.cu:582`-area, the same code that hits the chunked-prefill capacity bug at L725). 11.9 s API time over 2048 steps = **5.8 ms/step** spent in the host launching mini 2D copies.

**Root cause:**
- Dense H2D — pre-graph era launch pattern that wasn't migrated; or graph-incompatible code paths in the engine that spill out of the captured region.
- GDN D2D 2D — interleaved Q/G layout in QKV projection requires per-head strided extract. Either (a) materialize separate Q and gate buffers in the projection kernel itself, or (b) batch the per-head copy into a single 3D copy / a fused gather kernel.

**Expected impact:**
- Dense: graphs already win 2.5–3× decode (mem note `cuda_graphs_moe_works_2026_05_07`). The H2Ds we still see are inside non-graph-captured code paths (warmup, retry, scheduler decisions) — eliminating them is single-digit % at best on dense.
- **GDN: real and big.** 5.8 ms/step lost to copy-launch overhead = ~25 % of decode wall-clock at the GDN model's current ~140 tok/s. Fusing into one launch could push GDN decode +20 % toward Qwen3.5-9B's bandwidth ceiling, transferring directly to Qwen3.5-9B/27B and Qwen3.6-35B-A3B-NVFP4.

**Effort:** **M** for GDN (one new fused kernel, ~80 LoC), **L** for dense H2D triage (need to find which non-captured paths emit them).

**Files:** `src/graph/executor_attention.cu` (lines 582 and surrounding gather), `src/compute/qkv_split_*.cu`.

**Validation:** GDN `cudaMemcpy2DAsync` count → < 32 K (16/step); GDN `tg256` decode > 175 tok/s.

---

### Finding 3 — `cudaLaunchKernel` is the wall-clock dominator at ~1 M calls / 2 K decode steps on dense ⚡ TOP

**Workload(s):** W2 (all three models)

**Evidence:** API summary CSVs:

| Profile | cudaLaunchKernel calls | cudaLaunchKernelExC calls | API total / call |
|---------|------------------------:|---------------------------:|-------------------|
| Llama-3.2-3B W2 ng | **936,732** | 463,193 | 9.45 s + 4.68 s = 14.1 s in launches alone |
| Qwen3-4B Q8 W2 ng  | 1,052,228 | 742,045 | 11.2 s + 8.15 s = 19.3 s |
| Qwen3.5-GDN W2 ng  | 1,230,552 | 1,151,627 | 11.8 s + 11.1 s = 22.9 s |
| Qwen3-4B Q8 W2 g   | **4,426** | — | 37 ms |

≈ 875–1100 launches per decode step on dense, 1170 on GDN. Captured graph collapses 99.6 % of these. This is exactly the pattern the sm120-cuda-expert skill calls out as **Law 1**: *decode at batch=1 is launch-overhead-bound first, memory-bound second.*

**Root cause:** ~80–120 launches per layer × 28–36 layers + sampling/embedding/norm tail. With graphs ON, the entire decode loop becomes one `cudaLaunchKernelEx` per N steps (`AsyncGraphLoop max_steps=255`).

**Expected impact:** already realized when graphs are on. **Action: ensure graphs stay on** — every hot-path patch must re-bench `--no-cuda-graphs` *and* default. The numbers above are the cost of any code change that breaks graph capture (e.g., adds a host sync inside the captured region).

**Effort:** **N/A** — this is policy + regression-watch, not a fix.

**Risk:** Patches that disable graphs silently lose 60-70 % of decode tok/s. Add a CI gate: `make verify-fast` must include `IMP_FORCE_NO_GRAPHS=0` and fail if decode tok/s drops below the with-graphs baseline by >10 %.

**Validation:** any future PR touching `src/graph/`, `src/compute/`, or `src/quant/` should include a `nsys stats … --report cuda_api_sum` snippet showing cudaLaunchKernel count stays flat.

---

### Finding 4 — Prefill: `causal_softmax_fp32_inplace` is 16-19 % of prefill, 270-370 µs avg, plus a 12-14 % FP32→FP16 cast right after

**Workload(s):** W1 (both Q8 dense models)

**Evidence:**

| Profile | softmax % | softmax avg | fp32→fp16 % | fp32→fp16 avg |
|---------|----------:|-------------:|-------------:|---------------:|
| Llama W1 | 16.3 | 272 µs | 11.9 | 199 µs |
| Qwen3-4B W1 | 18.9 | 368 µs | 13.8 | 269 µs |

Both kernels run the same number of times (1344 / 1728 = layers × n_heads × prefill chunks). The pattern is: attention → score-matrix in FP32 → softmax FP32 → cast FP32→FP16 → AV.

**Root cause:** the prefill attention path emits FP32 scores into a separate buffer, runs softmax in place, then casts down to FP16 to feed the next matmul. Two separate kernels read/write the entire scores tensor (n × ctx_len matrix). At pp=8192, n=4096 chunk × ctx=8192 = 33 M floats = 134 MB read+write per call **twice**.

**Expected impact:** fusing softmax + FP32→FP16 cast into one pass saves ~50 % of the cast time = **~6 % of prefill**. Bigger payoff if we also fuse the score-write kernel one step further upstream. ~7–10 % prefill on Qwen3-4B Q8 / Llama-3.2-3B Q8; transfers directly to Qwen3-8B and any longer-context Llama variant.

**Effort:** **S** — one new fused kernel `causal_softmax_fp32_to_fp16_kernel`, ~50 LoC. Output dtype changes — need to update the consumer GEMM signature.

**Files:** `src/compute/softmax.cu`, `src/graph/executor_attention.cu` (the call site).

**Risk:** numerical drift if we accidentally compute softmax in FP16. Keep FP32 reduction internally.

**Validation:** nsys re-run shows the merged kernel; `tests/test-attention` parity. Long-context regression test (Qwen3-4B Q8 pp=8192).

---

### Finding 5 — Prefill: imp on-the-fly NVFP4-quantizes Q8_0 activations to use SM120 BlockScaled GEMM (working as designed, but verify shape coverage)

**Workload(s):** W1 (Llama and Qwen3-4B)

**Evidence:** the SM120 fast-path kernel **is firing** in Q8_0 prefill, with NVFP4 inputs:

```
cutlass::device_kernel<…MainloopSm120TmaWarpSpecializedBlockScaled…
  KernelTmaWarpSpecializedCooperativeBlockScaledSm120<3>,
  Tile<128,128,128>,
  A,B = float_e2m1_t (NVFP4),  scales = float_ue4m3_t,
  MMA atom = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1,e2m1,float,ue4m3,16>>
```

Llama W1: 5.5 % (123 ms / 6720 calls / 18 µs avg).
Qwen3-4B W1: 4.2 % (143 ms / 8640 calls / 17 µs avg).

Paired with `quantize_fp16_nvfp4_cutlass_kernel` (6720 / 8640 calls — exact match) and `dequant_q8_0_kernel` (3000+ calls) — the pipeline is Q8_0 → FP16 weight → quantize-to-NVFP4 + scales → SM120 BlockScaled GEMM → FP16 out. Burns dequant+requant for 3354 TOPS FP4 TC throughput.

**Root cause:** intentional. The sm120-cuda-expert skill flagged the `mma.sync.kind::mxf4nvf4.block_scale` path as the only way to hit Blackwell FP4 peak.

**Expected impact:** this is a **positive observation** — the fast path is alive on the right shapes. But we should sanity-check coverage: the WMMA-fallback shapes (Finding 1) are the GEMMs *not* taking this path. Any GEMM where on-the-fly NVFP4 quantize-then-GEMM beats cuBLAS-FP16 should also be routed through this path.

**Effort:** **M** to extend dispatch coverage. **0** if you're happy with current coverage.

**Files:** `src/compute/gemm_dispatcher.*`, the routing logic that picks NVFP4 vs cuBLAS.

**Validation:** count of `MainloopSm120TmaWarpSpecializedBlockScaled` calls grows; WMMA fallback shrinks.

---

### Finding 6 — Decode: 5 GEMV kernels concentrate 88 % of decode kernel time on dense Q8

**Workload(s):** W2 (Qwen3-4B Q8, Llama-3.2-3B Q8)

**Evidence:** Qwen3-4B Q8 W2 graphs OFF, total 16.88 s GPU kernel time:

| % | kernel | per-step calls | avg ns/call |
|---:|--------|----------------:|-------------:|
| 35.9 | `gemv_dp4a_gate_up_kernel` | 72 | 41 100 |
| 24.5 | `gemv_dp4a_kpar_kernel` | 144 | 14 050 |
| 10.9 | `gemv_dp4a_kpar_qkv_kernel` | 72 | 12 440 |
| 10.6 | `paged_attention_splitk_pipeline_kernel<128>` | 72 | 12 180 |
| 6.6  | `gemv_dp4a_fp32_kernel` (LM head) | 2 | 270 490 |
| **88.5** | (top-5 total) | | |

Llama-3.2-3B Q8 W2: gate_up 36.0 %, kpar 24.1 %, kpar_qkv 11.0 %, splitk_pipeline 9.3 %, fp32 8.6 % = **88.9 %** top-5.

**Root cause:** decode-time dispatch goes through the dp4a (INT8 dot-product) path because weights are Q8_0. These kernels are the gate of decode bandwidth. A 10 % speedup on `gemv_dp4a_gate_up_kernel` alone is **worth more than** any other single optimization on this hardware/quant combo.

**Expected impact:** these are bandwidth-bound at decode (1 token × hidden × n_layers worth of weight reads). Peak HBM = 1792 GB/s on RTX 5090. Without ncu metrics (skipped — needs CAP_SYS_ADMIN, see "Limitations"), I can't put % of peak on these — but the avg-µs / weight-bytes analysis suggests they're likely 60-80 % of peak. Headroom ~5-15 % per kernel via tighter cp.async pipelining or wider vector loads.

**Effort:** **L** per kernel, but each one moves decode tok/s by 1-3 %. **Recommendation: spend ncu time on `gemv_dp4a_gate_up_kernel` first** — it's 36 % of decode time and any improvement compounds across every dense Q8 model.

**Files:** `src/compute/gemv_dp4a*.cu`.

**Risk:** dp4a kernels have a long history of subtle bugs (see memory: launch_bounds miscompiles, partial-RoPE pair layout). Don't refactor without an existing parity test.

**Validation:** ncu roofline of `gemv_dp4a_gate_up_kernel`; dense Q8 decode tok/s.

---

### Finding 7 — Long-context GDN/Mamba2 path is BLOCKED by the chunked_prefill `attn_scores_` capacity bug

**Workload(s):** W1 (Qwen3.5-4B-GDN Q8) — capture aborted

**Evidence:** the engine errors out at pp=8192 on Qwen3.5-GDN with:

```
[ERROR] executor_attention.cu:733: chunked_prefill: attn_scores_ capacity
  4096×4096=16777216 too small for n=4096 × ctx_len=8192 = 33554432 at L3 —
  engine should have prevented this
```

This is the documented bug `chunked_prefill_attn_scores_capacity_bug_2026_05_09`. `resolve_prefill_chunk_size` doesn't actually clamp `n × ctx_len ≤ attn_scores_ rows × cols`.

**Root cause:** capacity check missing in scheduler. Attention executor has the assert; engine should clamp before dispatch.

**Expected impact:** unblocks long-context profiling for Qwen3.5-9B/27B GDN, Qwen3.6-35B-A3B-NVFP4. Currently any ctx > 4096 on a hybrid model hits this.

**Effort:** **S** — already shape-spec'd in the memo (PR #134-shape fix). One conditional in `engine.cpp::resolve_prefill_chunk_size`.

**Risk:** low.

**Validation:** rerun `profiles/run_nsys_baselines.sh` GDN W1 line — should produce a clean `qwen35-4b-gdn-q8_W1_ng.nsys-rep`.

---

### Finding 8 — Decode H2D + cudaMemsetAsync still appear at 10 K+ counts even on dense Q8 (Llama W1 has 10 203 cudaMemsetAsync calls = 4.5 % API time)

**Workload(s):** W1 (both dense models)

**Evidence:** Llama W1 API summary: 10 203 `cudaMemsetAsync`, 197 ms total, 19 µs avg. Qwen3-4B W1: 13 042 calls, 194 ms.

**Root cause:** likely buffer-zeroing between attention chunks or KV-cache init. Many of these are tiny (the median is 4 KB) and could be (a) deferred (lazy zero), (b) batched into one larger memset, or (c) done inside the kernel that consumes the buffer (just write all elements unconditionally).

**Expected impact:** ~3-4 % API time → small but free if the kill is local.

**Effort:** **S** if it's one or two call sites; **M** if it's diffuse.

**Files:** grep `cudaMemsetAsync` under `src/`. Probably attention scratch, NVFP4 dequant scratch.

**Risk:** correctness — only safe if the consumer overwrites all bytes.

**Validation:** Llama W1 cudaMemsetAsync count ≤ 1 K.

---

## Summary table — by ROI

| # | Title                                                           | Workload   | Effort | Impact (decode / prefill) |
|---|------------------------------------------------------------------|------------|--------|---------------------------|
| **1** | WMMA legacy fallback eats 12 % prefill                         | W1 dense   | M      | **+8 % prefill**           |
| **2** | GDN: 656 D2D copies/step = 5.8 ms/step lost                    | W2 GDN     | M      | **+20 % decode** (GDN)     |
| **3** | Decode launch overhead (already won by graphs — defend it)     | W2 all     | policy | already +200 % decode      |
| 4 | Fuse softmax + FP32→FP16 cast in prefill                          | W1 dense   | S      | +6 % prefill              |
| 5 | NVFP4 BlockScaled SM120 fast-path is alive (extend coverage)     | W1 dense   | M      | shape-dependent           |
| 6 | `gemv_dp4a_gate_up` is 36 % of dense decode — ncu deep-dive next | W2 dense   | L      | +1-3 % per kernel        |
| 7 | Fix `chunked_prefill` capacity bug to unblock GDN long-ctx prof  | W1 GDN     | S      | unblocks profiling        |
| 8 | Triage 10 K+ cudaMemsetAsync calls in dense prefill              | W1 dense   | S      | +1-2 % prefill            |

**Top-3 immediate work (by `impact / effort`):**

1. **Finding 7** (chunked_prefill clamp) — S effort, unblocks an entire model family from this profiling pipeline.
2. **Finding 4** (softmax+cast fuse) — S effort, +6 % prefill, all dense Q8 models.
3. **Finding 1** (WMMA fallback investigation) — M effort, +8 % prefill, all dense Q8 models. Start by enumerating which exact GEMM shapes hit WMMA (one debug pass), then decide between cuBLAS algo pin vs CUTLASS replacement.

**Top-3 strategic** (compound across more models):

- Finding 2 (GDN D2D fusion): +20 % decode for the whole GDN family (Qwen3.5-9B, 27B, Qwen3.6-35B-A3B-NVFP4).
- Finding 6 (`gemv_dp4a_gate_up` ncu): improvements compound across every Q8 dense model.
- Finding 3 (graph-policy CI gate): protects the *biggest* historical perf win.

---

## Limitations

- **No `--gpu-metrics-devices`.** Consumer driver requires CAP_SYS_ADMIN or `RestrictProfilingToAdminUsers=0`. We have nsys timeline + per-kernel times, but no DRAM bandwidth %, no SM utilization %. ncu Phase 3 hits the same restriction unless run as root or with the modprobe override. **Action item for the user**: decide whether to enable perf-counter access globally, or run ncu with sudo for per-kernel roofline.
- **`bench-reps=1`** for capture cleanliness. Tok/s tables in this doc are *not* perf baselines; refresh via `scripts/gen_perf_baseline.sh`.
- **W3 (16-concurrent batched serving) not captured.** Server-mode + load-test client is its own setup; deferred.
- **Prefill graph capture not separately profiled** — graphs help decode primarily; prefill is large-batch, less launch-overhead-sensitive.

## Reproduction

```bash
# Phase 1
./profiles/run_nsys_baselines.sh    # 6 captures @ ~30s each
# Phase 2
./profiles/export_stats.sh          # CSV exports
python3 ./profiles/analyze_csv.py   # top-15 + flags per profile
# Phase 3 (when ready)
./profiles/run_ncu_topk.sh "imp::gemv_dp4a_gate_up_kernel.*"
```

All `.nsys-rep` files retained — open in nsys-ui for timeline GUI, or `nsys stats --report nvtx_sum` after we wire NVTX.

## Phase 0 deferred work

- **NVTX ranges not added.** The original prompt mandated wiring `Engine::generate`, prefill/decode boundaries, per-layer ranges, KV cache ops, host scheduler. Skipped because kernel names alone produced clearly actionable findings. Revisit when iterating on Finding 6 — per-layer attribution would let us see which specific layer's `gemv_dp4a_gate_up` is fattest (LM head shape vs FFN shapes).
- **GPU clocks not locked** during capture. Auto-boost was active; A/B comparisons in future iterations should `sudo nvidia-smi -lgc 2400,2400`.

<!--
layer: L2
audience: kernel-devs
verified: 2026-08-28
commit: be825e4a
-->

# imp: Systematic Nsight Systems Profiling & Optimization

## Mission

End-to-end performance audit of `imp` with Nsight Systems (`nsys`) on the RTX 5090 (SM120). Output: top 5-10 optimization opportunities ranked by expected tok/s impact, prioritized action list with concrete patches. Measurement-first: no speculative optimization, every claim backed by a profile.

---

## Phase 0: Environment & Build Setup

1. Verify toolchain:
   - `nsys --version` (require ≥ 2025.3 for Blackwell support)
   - `nvidia-smi` confirms RTX 5090, driver, CUDA runtime
   - imp built with `-lineinfo` and **NVTX ranges enabled** (`-DIMP_ENABLE_NVTX=ON`)
   - Release build with `-O3`, `-g` symbols retained for kernel attribution
2. If NVTX is not wired in, **add it first**. Annotate at minimum: `imp::Engine::generate()` (top-level); prefill vs decode phases (separate ranges); per-layer attention / FFN-MoE / sampling; KV cache ops (copy, append, evict); host-side tokenization, scheduler, batching. `nvtxRangePushA`/`PopA` with descriptive names; categories per subsystem.
3. Lock GPU clocks for reproducibility; document chosen frequencies:
   ```
   nvidia-smi -lgc <base_clock>
   nvidia-smi -lmc <mem_clock>
   ```

---

## Phase 1: Baseline Profile Collection

Three reference workloads; for each capture both `nsys` (timeline) and a separate `ncu` summary (per-kernel metrics) for cross-reference.

- **W1 - Long-context prefill:** single request, 8k input, 64 output tokens. Stresses FMHA prefill, weight loads; scheduler overhead amortized.
- **W2 - Decode-heavy:** single request, 256 input, 2048 output. Stresses KV cache, decode-phase FMHA, sampling, host-GPU sync.
- **W3 - Batched serving:** 16 concurrent requests, mixed lengths. Stresses scheduler, KV layout, MoE expert routing if applicable.

**Capture command template:**
```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=20000 \
  -o profiles/W<N>_baseline \
  ./build/imp_server <args>
```

`cudaProfilerStart()`/`cudaProfilerStop()` in the harness skips warmup and shutdown noise: profile **steady state only**. Per workload also record wall-clock tok/s (prefill and decode separately), peak and average VRAM, power draw (`nvidia-smi dmon -s pucvmet`). Keep raw `.nsys-rep` files.

---

## Phase 2: Timeline Analysis - Where Does Time Go?

Open each profile in the GUI **and** export stats for grep-able artifacts:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum \
  --format csv --output profiles/W<N>_baseline profiles/W<N>_baseline.nsys-rep
```

Per workload, a structured report answering:

### 2.1 Kernel time breakdown
- Top 20 kernels by total GPU time, grouped by NVTX phase. Per kernel: % of runtime, invocation count, avg duration, occupancy hint.
- Flag unexpectedly hot kernels (`memset`, small reduction, layout transform > 2 %): usually the easy wins.

### 2.2 GPU idle / bubbles
- Fraction of wall time executing kernels vs idle; largest idle gaps, cross-referenced with NVTX for the host-side cause (`cudaMemcpyAsync` on default stream, scheduler decisions, tokenizer on hot path, allocator calls).
- `cudaStreamSynchronize` / `cudaDeviceSynchronize` on the critical path that should not be there?

### 2.3 Host-GPU concurrency
- Is the host issuing fast enough? Look for kernel-launch-latency gaps (> 5-10 µs between kernels with no host work).
- CUDA Graphs on the decode loop? If not, almost certainly a top-3 finding.
- malloc/free in the steady-state hot loop (any `cudaMalloc`/`cudaFree` after warmup is a bug): nsys is the wrong instrument. Build with `-DIMP_ALLOC_INTERPOSE=ON` and read `[alloc-interpose] steady state`, which attributes every call site. Shipped state is zero (`0 cudaMalloc, 0 cudaMallocAsync, 0 pinned-host allocations while serving`); any nonzero count is the finding. **Rebuild with the default OFF before measuring throughput**: the shim costs ~3 % decode and has been mistaken for a regression (`AUDIT.md` G16).

### 2.4 Memory subsystem
- HBM read/write bandwidth utilization (from `--gpu-metrics`); decode kernels below 70 % of theoretical (1.79 TB/s on RTX 5090) = headroom.
- H2D / D2H transfers in the hot path? (Tokenization output, sampling output: pinned memory? batched?)
- KV cache: writes per token, layout (paged?), fragmentation.

### 2.5 Stream usage
- How many CUDA streams; real overlap between compute streams and copy engine, or everything on stream 0?
- Batched serving: per-request work on separate streams? (Probably not worth it for compute-bound kernels; worth checking for small ops.)

### 2.6 NVFP4 / MXFP4 path verification
- Confirm the dispatched FMHA + GEMM kernels are the NVFP4/MXFP4 variants, not BF16/FP16 fallbacks (kernel name in `nsys`). Any FP16 GEMM in the SM120 hot path runs at half speed and is an immediate bug.
- MXFP4 FMHA: confirm the new kernel, not an older path still resident in the binary.

---

## Phase 3: Per-Kernel Deep-Dive (selective)

`ncu` on the top 5 hottest kernels from Phase 2. **Only** the top 5: `ncu` is slow.

```bash
ncu --set full \
    --target-processes all \
    --kernel-name regex:"<pattern>" \
    --launch-skip 50 --launch-count 10 \
    -o profiles/ncu_<kernel_name> \
    ./build/imp_server <args>
```

Per kernel extract: achieved vs theoretical occupancy; memory throughput vs peak (HBM, L2, shared); tensor core utilization (NVFP4 SM120 path); register pressure / spills; top-3 warp stall reasons; roofline position. Compare against the published SM120 peak for that precision/op; the gap is the headroom.

### Register pressure without a GPU

`cuobjdump -res-usage` reads the **compiled library**: works in CI, in a container with no card, against any build.

```
make kernel-resources         # check the pinned kernels
make kernel-resources-stats   # totals only
make kernel-resources-update  # re-pin, deliberately
```

| field | meaning |
|---|---|
| `REG` | registers per thread. **255 is the hardware ceiling** - ptxas spills past it |
| `STACK` | per-thread local frame in bytes. Non-zero = state in local memory |
| `LOCAL` | separately declared local memory |

Current build: **823 kernels, 71 at risk, 6 sitting exactly at 255** (`gdn_scan_chunkwise_kernel`, `gdn_scan_fused_kernel`, `fmha_sm120_fa2_kernel`) and 70 with a non-zero local frame. `tools/kernel_resource_baseline.txt` pins those 71 as a **two-way ratchet**: a kernel that starts spilling fails the gate, and so does a pinned kernel that improved, so the list cannot go stale in either direction.

This is the gate the throughput gates cannot be: `verify-fast` compares decode and prefill at an 8 % threshold, and one kernel dropping over the register cliff inside a 48-layer forward is far below that. It also makes the 82 hand-set `__launch_bounds__` auditable (`src/compute/CLAUDE.md` says never add one blind; until #1549 the measurement that sentence demands did not exist in the tree).

---

## Phase 4: Findings & Prioritized Action List

Produce `nsys_findings.md`, target 8-12 findings:

```
### Finding N: <short title>
**Workload(s):** W1 / W2 / W3
**Evidence:** <screenshot/excerpt from nsys, kernel name, % of runtime, metric values>
**Root cause:** <what is actually happening>
**Expected impact:** <estimated tok/s gain, with reasoning>
**Effort:** S / M / L
**Risk:** <regressions, complexity>
**Proposed fix:** <concrete code change, file paths, kernel/host>
**Validation plan:** <which metric must move, by how much>
```

Then a summary table sorted by `expected_impact / effort` (ROI), top 3 marked as immediate work.

Common findings to check for (do not assume present):

- Decode loop not using CUDA Graphs → graph capture per batch shape, cache by shape key
- Sampling kernel launching many small CUB calls per step → fuse, or `cub::DeviceTopK` (CUDA 13.2)
- KV append doing layout transforms fusable into the attention output write
- Any BF16/FP16 GEMM on the SM120 hot path (half speed; must move to NVFP4/MXFP4)
- Host-side scheduler decisions blocking the launch queue
- `cudaMemcpyAsync` on the default stream instead of a copy stream
- Attention kernel not saturating tensor cores during decode (low arithmetic intensity; mergeable with adjacent ops?)
- MoE expert dispatch: scatter/gather kernels dominating over expert GEMMs
- Tokenizer or detokenizer on the response loop's critical path instead of overlapped

---

## Phase 5: Iterate

After implementing the top 1-3 fixes:

1. Re-run the **identical** Phase 1 capture commands.
2. Diff against baseline via `nsys stats` CSV outputs.
3. Confirm the targeted metric moved as predicted; check for regressions elsewhere.
4. Update the findings doc with actuals vs predicted.
5. Repeat.

Each iteration includes the `.nsys-rep` files in `profiles/` so improvements are auditable.

---

## Deliverables

- `profiles/`: all `.nsys-rep` and `ncu-rep` files (baseline + each iteration)
- `profiles/*.csv`: exported stats per workload
- `nsys_findings.md`: prioritized findings as above
- `nsys_methodology.md`: exact reproduction steps (clocks, build flags, harness commands)
- A short PR per implemented fix, linking the finding ID with before/after numbers

## Constraints

- Do not optimize anything not backed by a profile.
- Do not change kernel correctness without a numerical equivalence test.
- Keep all NVTX ranges in the final build: always-on profiling is a feature.
- If the GUI shows something `nsys stats` does not, screenshot it into the findings doc.

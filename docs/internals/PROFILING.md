<!--
layer: L2
audience: kernel-devs
verified: 2026-08-13
commit: 81ffa573
-->

# imp: Systematic Nsight Systems Profiling & Optimization

## Mission

Conduct a rigorous end-to-end performance audit of `imp` using Nsight Systems (`nsys`) on the RTX 5090 (SM120). Identify the top 5–10 optimization opportunities ranked by expected tok/s impact, and produce a prioritized action list with concrete patches.

This is a **measurement-first** workflow. No speculative optimization. Every claim is backed by a profile.

---

## Phase 0: Environment & Build Setup

1. Verify toolchain:
   - `nsys --version` (require ≥ 2025.3 for Blackwell support)
   - `nvidia-smi` confirms RTX 5090, driver, CUDA runtime
   - imp built with `-lineinfo` and **NVTX ranges enabled** (`-DIMP_ENABLE_NVTX=ON`)
   - Release build with `-O3`, but `-g` symbols retained for kernel attribution

2. If NVTX is not yet wired in, **add it first** before profiling. Annotate at minimum:
   - `imp::Engine::generate()` (top-level)
   - Prefill vs. decode phases (separate ranges)
   - Per-layer: attention, FFN/MoE, sampling
   - KV cache ops (copy, append, evict)
   - Host-side: tokenization, scheduler, batching
   - Use `nvtxRangePushA`/`PopA` with descriptive names; categories per subsystem

3. Lock GPU clocks for reproducibility:
   ```
   nvidia-smi -lgc <base_clock>
   nvidia-smi -lmc <mem_clock>
   ```
   Document chosen frequencies.

---

## Phase 1: Baseline Profile Collection

Collect three reference workloads. For each, capture both `nsys` (timeline) and a separate `ncu` summary (per-kernel metrics) so we can cross-reference.

**Workloads:**
- **W1 — Long-context prefill:** single request, 8k input, 64 output tokens. Stresses FMHA prefill, weight loads, scheduler overhead is amortized.
- **W2 — Decode-heavy:** single request, 256 input, 2048 output. Stresses KV cache, decode-phase FMHA, sampling, host-GPU sync.
- **W3 — Batched serving:** 16 concurrent requests, mixed lengths. Stresses scheduler, KV layout, MoE expert routing if applicable.

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

Use `cudaProfilerStart()`/`cudaProfilerStop()` in the test harness to skip warmup and exclude shutdown noise. Profile **steady state only**.

For each workload also record:
- Wall-clock tok/s (prefill tok/s and decode tok/s separately)
- Peak and average VRAM
- Power draw (`nvidia-smi dmon -s pucvmet`)

Save raw `.nsys-rep` files; do not delete after analysis.

---

## Phase 2: Timeline Analysis — Where Does Time Go?

Open each profile in the Nsight Systems GUI **and** export stats via CLI for grep-able artifacts:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum \
  --format csv --output profiles/W<N>_baseline profiles/W<N>_baseline.nsys-rep
```

For each workload, produce a structured report answering:

### 2.1 Kernel time breakdown
- Top 20 kernels by total GPU time. Group by logical phase using NVTX ranges.
- For each: % of total runtime, invocation count, avg duration, occupancy hint.
- Flag any kernel that is unexpectedly hot (e.g., a `memset`, a small reduction, a layout transform consuming >2% — these are usually the easy wins).

### 2.2 GPU idle / bubbles
- What fraction of wall time is the GPU actually executing kernels vs. idle?
- Where are the largest idle gaps on the timeline? Cross-reference with NVTX to identify the host-side cause (e.g., `cudaMemcpyAsync` on default stream stalling, scheduler decisions, tokenizer on hot path, allocator calls).
- Are there `cudaStreamSynchronize` / `cudaDeviceSynchronize` calls on the critical path that should not be there?

### 2.3 Host-GPU concurrency
- Is the host issuing work fast enough to keep the GPU saturated? Look for "kernel launch latency" gaps (>5–10 µs between kernels with no host work).
- Are CUDA Graphs being used for the decode loop? If not, this is almost certainly a top-3 finding.
- Check for malloc/free in the steady-state hot loop (any `cudaMalloc`/`cudaFree` after warmup is a bug). nsys is the wrong instrument for this one — build with `-DIMP_ALLOC_INTERPOSE=ON` and read `[alloc-interpose] steady state`, which attributes every call site. The shipped state is zero (`0 cudaMalloc, 0 cudaMallocAsync, 0 pinned-host allocations while serving`), so any nonzero count is the finding. **Rebuild with the default OFF before measuring throughput** — the shim costs ~3% decode and has already been mistaken for a regression (`AUDIT.md` G16).

### 2.4 Memory subsystem
- HBM read/write bandwidth utilization (from `--gpu-metrics`). If decode kernels are below 70% of theoretical (1.79 TB/s on RTX 5090), there's headroom.
- H2D / D2H transfers: any in the hot path? (Tokenization output, sampling output — pinned memory? batched?)
- KV cache: pattern of writes per token, layout (paged?), fragmentation.

### 2.5 Stream usage
- How many CUDA streams? Is there real overlap between compute streams and copy engine, or is everything on stream 0?
- For batched serving: is per-request work on separate streams to enable kernel overlap? (Probably not worth it for compute-bound kernels, but worth checking for small ops.)

### 2.6 NVFP4 / MXFP4 path verification
- Confirm the FMHA + GEMM kernels actually being dispatched are the NVFP4/MXFP4 variants, not BF16/FP16 fallbacks. Kernel name in `nsys` should make this obvious; if there's any FP16 GEMM in the hot path on SM120 it's running at half speed and is an immediate bug.
- For the MXFP4 FMHA: confirm it's the new kernel and not an older path still resident in the binary.

---

## Phase 3: Per-Kernel Deep-Dive (selective)

For the top 5 hottest kernels identified in Phase 2, run `ncu` for detailed metrics. **Only** the top 5 — `ncu` is slow, don't profile everything.

```bash
ncu --set full \
    --target-processes all \
    --kernel-name regex:"<pattern>" \
    --launch-skip 50 --launch-count 10 \
    -o profiles/ncu_<kernel_name> \
    ./build/imp_server <args>
```

For each kernel, extract:
- Achieved occupancy vs. theoretical
- Memory throughput vs. peak (HBM, L2, shared)
- Tensor core utilization (NVFP4 SM120 path)
- Register pressure / spills
- Warp stall reasons (top 3)
- Roofline position: compute-bound or memory-bound?

Compare against the published peak for SM120 for that precision/op. The gap is the headroom.

---

## Phase 4: Findings & Prioritized Action List

Produce `nsys_findings.md` with the following structure:

For each finding (target 8–12 findings):

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

Then a summary table sorted by `expected_impact / effort` (ROI), with the top 3 marked as immediate work.

Common findings to specifically check for (do not assume present, but look):
- Decode loop not using CUDA Graphs → graph capture per batch shape, cache by shape key
- Sampling kernel launching many small CUB calls per step → fuse, or use `cub::DeviceTopK` (CUDA 13.2)
- KV append pattern doing layout transforms that could be fused into attention output write
- Any BF16/FP16 GEMM on SM120 hot path (running at half speed — must move to NVFP4/MXFP4)
- Host-side scheduler decisions blocking the launch queue
- `cudaMemcpyAsync` on the default stream when it should be on a copy stream
- Attention kernel not saturating tensor cores during decode (low arithmetic intensity — can it be merged with adjacent ops?)
- MoE expert dispatch: scatter/gather kernels dominating over expert GEMMs
- Tokenizer or detokenizer running on the critical path of the response loop instead of overlapped

---

## Phase 5: Iterate

After implementing the top 1–3 fixes:
1. Re-run the **identical** capture commands from Phase 1.
2. Diff against baseline using `nsys stats` CSV outputs.
3. Confirm: did the targeted metric move as predicted? Any regressions elsewhere?
4. Update the findings doc with actuals vs. predicted.
5. Repeat.

Each iteration must include the `.nsys-rep` files in `profiles/` so improvements are auditable.

---

## Deliverables

- `profiles/` — all `.nsys-rep` and `ncu-rep` files (baseline + each iteration)
- `profiles/*.csv` — exported stats per workload
- `nsys_findings.md` — prioritized findings as described above
- `nsys_methodology.md` — exact reproduction steps (clocks, build flags, harness commands)
- A short PR per implemented fix, each linking to the finding ID and showing before/after numbers

## Constraints

- Do not optimize anything that isn't backed by a profile.
- Do not change kernel correctness without a numerical equivalence test.
- Keep all NVTX ranges in the final build — always-on profiling is a feature.
- If the GUI shows something `nsys stats` doesn't, screenshot it and put it in the findings doc.

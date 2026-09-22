<!--
layer: L2
audience: kernel-devs
verified: 2026-09-22
commit: 9cbb8004
-->

# Profiling commands

`nsys`/`ncu` reference. Campaign mission, phase plan, findings template and deliverables: [`docs/plans/2026-06-profiling-campaign.md`](../plans/2026-06-profiling-campaign.md).

## Setup

```bash
nsys --version                 # require >= 2025.3 for Blackwell support
nvidia-smi                     # confirm RTX 5090, driver, CUDA runtime
nvidia-smi -lgc <base_clock>   # lock GPU clocks for reproducibility
nvidia-smi -lmc <mem_clock>
```

Build with `-lineinfo`, `-DIMP_ENABLE_NVTX=ON`, `-O3` with `-g` symbols retained for kernel attribution.

## nsys: capture a profile

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

`cudaProfilerStart()`/`cudaProfilerStop()` in the harness skips warmup/shutdown noise - profile steady state only.

## nsys: export stats to CSV

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum \
  --format csv --output profiles/W<N>_baseline profiles/W<N>_baseline.nsys-rep
```

## ncu: per-kernel deep-dive

Top 5 hottest kernels only - `ncu` is slow.

```bash
ncu --set full \
    --target-processes all \
    --kernel-name regex:"<pattern>" \
    --launch-skip 50 --launch-count 10 \
    -o profiles/ncu_<kernel_name> \
    ./build/imp_server <args>
```

Extract: achieved vs theoretical occupancy, memory throughput vs peak (HBM/L2/shared), tensor-core utilization, register pressure/spills, top-3 warp stall reasons, roofline position vs the published SM120 peak for that precision/op.

## Register pressure without a GPU

`cuobjdump -res-usage` reads the compiled library: works in CI, in a container with no card, against any build.

```bash
make kernel-resources         # check the pinned kernels
make kernel-resources-stats   # totals only
make kernel-resources-update  # re-pin, deliberately
```

| Field | Meaning |
|---|---|
| `REG` | registers per thread; 255 is the hardware ceiling, ptxas spills past it |
| `STACK` | per-thread local frame in bytes; nonzero means state in local memory |
| `LOCAL` | separately declared local memory |

`tools/kernel_resource_baseline.txt` pins every kernel at or above REG 240 or with a nonzero local frame as a two-way ratchet: a kernel that starts spilling fails the gate, and so does a pinned kernel that improved (the list cannot go stale in either direction). This is the gate the throughput gates (`verify-fast`, 8 % threshold) cannot be: one kernel crossing the register cliff inside a 48-layer forward is far below that threshold. It also makes hand-set `__launch_bounds__` auditable (`src/compute/CLAUDE.md`).

## Common findings to check for

- Decode loop not using CUDA Graphs
- Sampling kernel launching many small CUB calls per step
- KV append doing layout transforms fusable into the attention output write
- Any BF16/FP16 GEMM on the SM120 hot path (half speed; must be NVFP4/MXFP4)
- Host-side scheduler decisions blocking the launch queue
- `cudaMemcpyAsync` on the default stream instead of a copy stream
- Attention kernel not saturating tensor cores during decode
- MoE expert dispatch: scatter/gather kernels dominating over expert GEMMs
- Tokenizer/detokenizer on the response loop's critical path instead of overlapped

## Allocations while serving

Any `cudaMalloc`/`cudaFree` after warmup is a bug; nsys is the wrong instrument. Build with `-DIMP_ALLOC_INTERPOSE=ON` and read `[alloc-interpose] steady state`, which attributes every call site. Shipped state is zero. Rebuild with the default OFF before measuring throughput: the shim costs ~3 % decode.

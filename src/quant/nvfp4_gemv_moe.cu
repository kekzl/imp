// MoE NVFP4 GEMV kernels (per-expert decode / gate+up / swiglu) + launchers.
// Split out of nvfp4_gemm.cu (kernel .cu size gate). All kernels and launchers
// MOVED VERBATIM — hot-path numeric code, must stay bit-identical. Shared device
// helpers + tuning constants live in nvfp4_gemm_internal.cuh.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "quant/nvfp4_quant.h"
#include "runtime/pdl.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// MoE NVFP4 GEMV: per-expert decode projections.
// Grid: top_k * rows blocks, 128 threads.  Each block computes one row of
// one expert's output.  FP16 input (no Q8_1 pre-quantization).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_moe_decode_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales,
    const float* __restrict__ tensor_scales, const int32_t* __restrict__ expert_indices,
    const half* __restrict__ x, half* __restrict__ y, int rows, int K, size_t expert_stride_packed,
    size_t expert_stride_ms, int x_stride, int blocks_per_expert) {
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int row = blockIdx.x % blocks_per_expert;
    if (row >= rows)
        return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];

    const uint8_t* W = packed_data + (size_t)expert_id * expert_stride_packed;
    const uint8_t* MS = micro_scales + (size_t)expert_id * expert_stride_ms;
    float ts = tensor_scales[expert_id];
    const half* xi = x + (size_t)expert_slot * x_stride;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(W + (size_t)row * (K / 2), MS + (size_t)row * n_mb, ts, xi, n_mb, tid);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[(size_t)expert_slot * rows + row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE NVFP4 GEMV.
// Grid: dim3(top_k * rows, 2).  blockIdx.y=0 → gate, blockIdx.y=1 → up.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_moe_gate_up_fused_kernel(
    const uint8_t* __restrict__ gate_packed, const uint8_t* __restrict__ gate_ms,
    const float* __restrict__ gate_ts, const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_ms, const float* __restrict__ up_ts,
    const int32_t* __restrict__ expert_indices, const half* __restrict__ x, half* __restrict__ y_gate,
    half* __restrict__ y_up, int rows, int K, size_t expert_stride_packed, size_t expert_stride_ms,
    int blocks_per_expert) {
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int row = blockIdx.x % blocks_per_expert;
    if (row >= rows)
        return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];
    const bool is_up = (blockIdx.y == 1);

    const uint8_t* W = is_up ? (up_packed + (size_t)expert_id * expert_stride_packed)
                             : (gate_packed + (size_t)expert_id * expert_stride_packed);
    const uint8_t* MS = is_up ? (up_ms + (size_t)expert_id * expert_stride_ms)
                              : (gate_ms + (size_t)expert_id * expert_stride_ms);
    float ts = is_up ? up_ts[expert_id] : gate_ts[expert_id];
    half* out = is_up ? y_up : y_gate;

    // x_stride = 0 (shared input for gate+up)
    const half* xi = x;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(W + (size_t)row * (K / 2), MS + (size_t)row * n_mb, ts, xi, n_mb, tid);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        out[(size_t)expert_slot * rows + row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Multi-row MoE NVFP4 kernels: 8 warps (256 threads), NR rows per block.
// Truly fused gate+up: one block computes both gate and up dot products.
// Dramatically reduces block count for small K (e.g., d_model=2048).
// ---------------------------------------------------------------------------

// Multi-row MoE gate+up with blockIdx.y split (gate=0, up=1).
// One warp per row; threads-per-block = NR * 32 set by the launcher.
// Note: no __launch_bounds__ — per sm120 perf testing the override costs
// -4.5 % to -20 % on GEMV/attention paths (see sm120-cuda-expert skill).
template <int NR>
__global__ void gemv_nvfp4_moe_gate_up_mr_kernel(
    const uint8_t* __restrict__ gate_packed, const uint8_t* __restrict__ gate_ms,
    const float* __restrict__ gate_ts, const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_ms, const float* __restrict__ up_ts,
    const int32_t* __restrict__ expert_indices, const half* __restrict__ x, half* __restrict__ y_gate,
    half* __restrict__ y_up, int rows, int K, size_t expert_stride_packed, size_t expert_stride_ms,
    int blocks_per_expert) {
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int row = local_block * NR + warp_id;
    if (row >= rows || warp_id >= NR)
        return;

    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];
    const bool is_up = (blockIdx.y == 1);

    const uint8_t* W = is_up ? (up_packed + (size_t)expert_id * expert_stride_packed)
                             : (gate_packed + (size_t)expert_id * expert_stride_packed);
    const uint8_t* MS = is_up ? (up_ms + (size_t)expert_id * expert_stride_ms)
                              : (gate_ms + (size_t)expert_id * expert_stride_ms);
    float ts = is_up ? up_ts[expert_id] : gate_ts[expert_id];
    half* out = is_up ? y_up : y_gate;

    const uint8_t* row_packed = W + (size_t)row * (K / 2);
    const uint8_t* row_ms = MS + (size_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, ts, n_mb, lane, [&] __device__(const uint8_t* pb, int off) {
        return dot_micro_block(pb, x, off);
    });

    acc = warp_reduce(acc);
    if (lane == 0)
        out[(size_t)expert_slot * rows + row] = __float2half(acc);
}

// Multi-row MoE NVFP4 decode (single weight matrix, e.g., non-gated up or down).
template <int NR>
__global__ void gemv_nvfp4_moe_decode_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales,
    const float* __restrict__ tensor_scales, const int32_t* __restrict__ expert_indices,
    const half* __restrict__ x, half* __restrict__ y, int rows, int K, size_t expert_stride_packed,
    size_t expert_stride_ms, int x_stride, int blocks_per_expert) {
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int row = local_block * NR + warp_id;
    if (row >= rows || warp_id >= NR)
        return;

    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];

    const uint8_t* W = packed_data + (size_t)expert_id * expert_stride_packed + (size_t)row * (K / 2);
    const uint8_t* MS = micro_scales + (size_t)expert_id * expert_stride_ms + (size_t)row * n_mb;
    float ts = tensor_scales[expert_id];
    const half* xi = x + (size_t)expert_slot * x_stride;

    float acc = warp_k_loop(W, MS, ts, n_mb, lane, [&] __device__(const uint8_t* pb, int off) {
        return dot_micro_block(pb, xi, off);
    });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[(size_t)expert_slot * rows + row] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// MoE NVFP4 host launchers
// ---------------------------------------------------------------------------

// Multi-row dispatch: use MR kernel when K is small enough for warp-level
// reduction (n_mb <= 512, i.e., K <= 8192).
static bool use_moe_multirow(int K) { return (K / kMicroBlockSize) <= 512; }

// Tile-tuning knob: read NR from RuntimeConfig, clamp to {4, 8, 16, 32}.
// Default 8 = legacy behavior; 4/16/32 are A/B candidates for the perf
// regression recovery work (see PR opening MoE NVFP4 decode tile sweep).
static int resolve_mr_nr() {
    int v = imp::process_diag_moe_mr_nr();
    if (v == 4 || v == 8 || v == 16 || v == 32) return v;
    return 8;
}

// Dispatch a single-output MR-decode launch for the requested NR (one warp
// per row → threads-per-block = NR * 32). The thread-count match keeps the
// `row = local_block * NR + warp_id; if (warp_id >= NR) return;` guard from
// firing → no idle warps.
template <int NR>
static void launch_mr_decode(const NvFP4MoEQuantResult& w, const int32_t* expert_indices, const half* x,
                             half* y, int rows, int K, int x_stride, int top_k, cudaStream_t stream) {
    int blocks_per_expert = (rows + NR - 1) / NR;
    int total_blocks = top_k * blocks_per_expert;
    pdl::launch(gemv_nvfp4_moe_decode_mr_kernel<NR>, dim3(total_blocks), dim3(NR * 32), size_t(0), stream,
                reinterpret_cast<const uint8_t*>(w.packed_data),
                reinterpret_cast<const uint8_t*>(w.micro_scales), w.tensor_scales, expert_indices, x, y,
                rows, K, w.expert_stride_packed, w.expert_stride_ms, x_stride, blocks_per_expert);
}

void gemv_nvfp4_moe_decode(const NvFP4MoEQuantResult& w, const int32_t* expert_indices, const half* x,
                           half* y, int rows, int K, int x_stride, int top_k, cudaStream_t stream) {
    if (use_moe_multirow(K)) {
        switch (resolve_mr_nr()) {
            case 4:  launch_mr_decode<4>(w, expert_indices, x, y, rows, K, x_stride, top_k, stream); break;
            case 16: launch_mr_decode<16>(w, expert_indices, x, y, rows, K, x_stride, top_k, stream); break;
            case 32: launch_mr_decode<32>(w, expert_indices, x, y, rows, K, x_stride, top_k, stream); break;
            default: launch_mr_decode<8>(w, expert_indices, x, y, rows, K, x_stride, top_k, stream); break;
        }
    } else {
        int total_blocks = top_k * rows;
        pdl::launch(gemv_nvfp4_moe_decode_kernel, dim3(total_blocks), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(w.packed_data),
                    reinterpret_cast<const uint8_t*>(w.micro_scales), w.tensor_scales, expert_indices, x, y,
                    rows, K, w.expert_stride_packed, w.expert_stride_ms, x_stride, rows);
    }
}

template <int NR>
static void launch_mr_gate_up(const NvFP4MoEQuantResult& gate, const NvFP4MoEQuantResult& up,
                              const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                              int rows, int K, int top_k, cudaStream_t stream) {
    int blocks_per_expert = (rows + NR - 1) / NR;
    dim3 grid(top_k * blocks_per_expert, 2);
    pdl::launch(gemv_nvfp4_moe_gate_up_mr_kernel<NR>, grid, dim3(NR * 32), size_t(0), stream,
                reinterpret_cast<const uint8_t*>(gate.packed_data),
                reinterpret_cast<const uint8_t*>(gate.micro_scales), gate.tensor_scales,
                reinterpret_cast<const uint8_t*>(up.packed_data),
                reinterpret_cast<const uint8_t*>(up.micro_scales), up.tensor_scales, expert_indices, x,
                y_gate, y_up, rows, K, gate.expert_stride_packed, gate.expert_stride_ms,
                blocks_per_expert);
}

void gemv_nvfp4_moe_gate_up_fused(const NvFP4MoEQuantResult& gate, const NvFP4MoEQuantResult& up,
                                  const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                  int rows, int K, int top_k, cudaStream_t stream) {
    if (use_moe_multirow(K)) {
        switch (resolve_mr_nr()) {
            case 4:  launch_mr_gate_up<4>(gate, up, expert_indices, x, y_gate, y_up, rows, K, top_k, stream); break;
            case 16: launch_mr_gate_up<16>(gate, up, expert_indices, x, y_gate, y_up, rows, K, top_k, stream); break;
            case 32: launch_mr_gate_up<32>(gate, up, expert_indices, x, y_gate, y_up, rows, K, top_k, stream); break;
            default: launch_mr_gate_up<8>(gate, up, expert_indices, x, y_gate, y_up, rows, K, top_k, stream); break;
        }
    } else {
        dim3 grid(top_k * rows, 2);
        pdl::launch(gemv_nvfp4_moe_gate_up_fused_kernel, grid, dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(gate.packed_data),
                    reinterpret_cast<const uint8_t*>(gate.micro_scales), gate.tensor_scales,
                    reinterpret_cast<const uint8_t*>(up.packed_data),
                    reinterpret_cast<const uint8_t*>(up.micro_scales), up.tensor_scales, expert_indices, x,
                    y_gate, y_up, rows, K, gate.expert_stride_packed, gate.expert_stride_ms, rows);
    }
}

// (The fused SwiGLU MoE down-projection kernels were removed: the decode path
// now precomputes silu(gate)*up once via apply_expert_activation + a plain
// gemv_nvfp4_moe_decode — see executor_forward_moe_batch.cu run_moe_decode_fast.)

}  // namespace imp

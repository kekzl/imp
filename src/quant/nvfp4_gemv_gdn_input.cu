// M=1 GDN input projections in one launch: in_proj+gate (NVFP4) plus alpha+beta (FP16, the
// loader dequantizes them) share one grid, replacing four launches per GDN layer at batch=1
// on native-NVFP4 hybrids (the F16 gdn_input_packed path only covers F16/BF16 checkpoints).
// Per-row math and reduction order match the kernel each segment replaces, so every output
// is bit-identical to the four-launch path.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "quant/nvfp4_quant.h"
#include "core/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "core/pdl_device.cuh"

namespace imp {

namespace {

struct GdnInputArgs {
    const uint8_t* packed_in;
    const uint8_t* ms_in;
    float ts_in;
    half* y_in;
    int in_rows;
    const uint8_t* packed_gate;
    const uint8_t* ms_gate;
    float ts_gate;
    half* y_gate;
    int gate_rows;
    const half* w_alpha;
    half* y_alpha;
    const half* w_beta;
    half* y_beta;
    int ab_rows;
    const half* x;
    int K;
};

// Same loads and accumulation order as gemv_fp16_kernel (compute/gemm_gemv_dtype.cu):
// 16 halves per lane per iteration, half2 products summed in fp32. K % 16 == 0.
__device__ __forceinline__ float warp_dot_fp16(const half* __restrict__ row, const half* __restrict__ x,
                                               int K, int lane) {
    const int K_vec16 = K / 16;
    const float4* row_v = reinterpret_cast<const float4*>(row);
    const float4* x_v = reinterpret_cast<const float4*>(x);
    float sum = 0.0f;
    for (int i = lane; i < K_vec16; i += 32) {
        float4 a0 = row_v[2 * i];
        float4 a1 = row_v[2 * i + 1];
        float4 x0 = x_v[2 * i];
        float4 x1 = x_v[2 * i + 1];
        const half2* a_h2_0 = reinterpret_cast<const half2*>(&a0);
        const half2* x_h2_0 = reinterpret_cast<const half2*>(&x0);
        const half2* a_h2_1 = reinterpret_cast<const half2*>(&a1);
        const half2* x_h2_1 = reinterpret_cast<const half2*>(&x1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_0[j], x_h2_0[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_1[j], x_h2_1[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }
    return sum;
}

constexpr int kGdnInputNR = 8;  // warp-per-row rows per block (in_proj, alpha, beta)
constexpr int kGdnGateRowsPerBlock = kMRThreads / kKparThreads;  // 2: K-par rows per block

// Block ranges: [0,in_blocks) in_proj rows warp-per-row (multirow kernel's order), then
// gate rows at kKparThreads/row with reduce_kpar's order (gate takes gemv_nvfp4_kpar_kernel
// in the executor, not multirow, so this stays bit-identical too), then alpha/beta rows
// warp-per-row.
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_gdn_input_kernel(GdnInputArgs a) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = a.K / 2;
    const int n_mb = a.K / kMicroBlockSize;
    const int in_blocks = (a.in_rows + kGdnInputNR - 1) / kGdnInputNR;
    const int gate_blocks = (a.gate_rows + kGdnGateRowsPerBlock - 1) / kGdnGateRowsPerBlock;

    if (blockIdx.x >= in_blocks && blockIdx.x < in_blocks + gate_blocks) {
        // Gate: two rows per block, kKparThreads (128) threads each, the
        // K-par kernel's loop stride and cross-warp reduction order.
        __shared__ float warp_sums[kGdnGateRowsPerBlock][kKparWarps];
        const int half_id = threadIdx.x / kKparThreads;
        const int tid = threadIdx.x % kKparThreads;
        const int row = (blockIdx.x - in_blocks) * kGdnGateRowsPerBlock + half_id;
        const bool live = row < a.gate_rows;
        pdl_wait();
        float acc = 0.0f;
        if (live)
            acc = gemv_nvfp4_row(a.packed_gate + (int64_t)row * K_half, a.ms_gate + (int64_t)row * n_mb,
                                 a.ts_gate, a.x, n_mb, tid);
        pdl_trigger();
        acc = warp_reduce(acc);
        if ((tid & 31) == 0)
            warp_sums[half_id][tid / 32] = acc;
        __syncthreads();
        if (live && tid == 0) {
            float total = warp_sums[half_id][0];
#pragma unroll
            for (int w = 1; w < kKparWarps; w++)
                total += warp_sums[half_id][w];
            a.y_gate[row] = __float2half(total);
        }
        return;
    }

    // Warp-per-row segments: row is 0-based inside its own segment.
    const bool in_seg = blockIdx.x < in_blocks;
    const int row = (in_seg ? blockIdx.x : blockIdx.x - in_blocks - gate_blocks) * kGdnInputNR + warp_id;
    if (in_seg ? row >= a.in_rows : row >= 2 * a.ab_rows)
        return;
    pdl_wait();
    float acc;
    half* out;
    int local_row;
    if (in_seg) {
        local_row = row;
        out = a.y_in;
        acc = warp_k_loop(a.packed_in + (int64_t)row * K_half, a.ms_in + (int64_t)row * n_mb, a.ts_in, n_mb,
                          lane, [&] __device__(const uint8_t* pb, int off) {
                              return dot_micro_block(pb, a.x, off);
                          });
    } else {
        const bool is_alpha = row < a.ab_rows;
        local_row = is_alpha ? row : row - a.ab_rows;
        const half* w = is_alpha ? a.w_alpha : a.w_beta;
        out = is_alpha ? a.y_alpha : a.y_beta;
        acc = warp_dot_fp16(w + (int64_t)local_row * a.K, a.x, a.K, lane);
    }
    pdl_trigger();
    acc = warp_reduce(acc);
    if (lane == 0)
        out[local_row] = __float2half(acc);
}

}  // namespace

bool gemv_nvfp4_gdn_input_fused(const NvFP4QuantResult& w_in, const NvFP4QuantResult& w_gate,
                                const half* w_alpha, const half* w_beta, int ab_rows, const half* x,
                                half* y_in, half* y_gate, half* y_alpha, half* y_beta, int K,
                                cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    if (K % 16 != 0 || n_mb > 512 || w_in.K != K || w_gate.K != K || ab_rows <= 0)
        return false;
    GdnInputArgs a;
    a.packed_in = reinterpret_cast<const uint8_t*>(w_in.packed_data);
    a.ms_in = reinterpret_cast<const uint8_t*>(w_in.micro_scales);
    a.ts_in = w_in.tensor_scale;
    a.y_in = y_in;
    a.in_rows = static_cast<int>(w_in.N);
    a.packed_gate = reinterpret_cast<const uint8_t*>(w_gate.packed_data);
    a.ms_gate = reinterpret_cast<const uint8_t*>(w_gate.micro_scales);
    a.ts_gate = w_gate.tensor_scale;
    a.y_gate = y_gate;
    a.gate_rows = static_cast<int>(w_gate.N);
    a.w_alpha = w_alpha;
    a.y_alpha = y_alpha;
    a.w_beta = w_beta;
    a.y_beta = y_beta;
    a.ab_rows = ab_rows;
    a.x = x;
    a.K = K;
    const int blocks = (a.in_rows + kGdnInputNR - 1) / kGdnInputNR +
                       (a.gate_rows + kGdnGateRowsPerBlock - 1) / kGdnGateRowsPerBlock +
                       (2 * ab_rows + kGdnInputNR - 1) / kGdnInputNR;
    pdl::launch(gemv_nvfp4_gdn_input_kernel, dim3(blocks), dim3(kMRThreads), size_t(0), stream, a);
    return true;
}

void nvfp4_gdn_input_pdl_register() {
    pdl::enable_kernel(gemv_nvfp4_gdn_input_kernel);
    cudaFuncSetAttribute(gemv_nvfp4_gdn_input_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
}

}  // namespace imp

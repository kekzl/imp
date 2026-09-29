// Qwen4Exp PLE kernels. See ple.h for the math.

#include "compute/ple.h"

#include <cuda_fp16.h>

#include "core/logging.h"

namespace imp {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxKernel = 8;
constexpr int kConvRowsPerBlock = 64;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ float block_sum_(float v) {
    __shared__ float red[kThreads / 32];
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float r = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            r += __shfl_xor_sync(0xffffffffu, r, off);
        if (threadIdx.x == 0)
            red[0] = r;
    }
    __syncthreads();
    return red[0];
}

// One block per (row, stream): the dot needs the whole row before q's row is overwritten.
__global__ void ple_gate_value_kernel(const half* __restrict__ key, half* __restrict__ q_gv,
                                      const half* __restrict__ value, int hc, int d, float inv_sqrt_d) {
    const int row = blockIdx.x / hc;
    const int s = blockIdx.x % hc;
    const size_t base = (static_cast<size_t>(row) * hc + s) * d;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        dot += __half2float(key[base + j]) * __half2float(q_gv[base + j]);
    float g = block_sum_(dot) * inv_sqrt_d;
    g = copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g);
    const float gate = sigmoidf_(g);
    const half* vrow = value + static_cast<size_t>(row) * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        q_gv[base + j] = __float2half(gate * __half2float(vrow[j]));
}

__device__ __forceinline__ size_t seq_state_off_(int64_t slot_stride, const int* slots, int s) {
    return static_cast<size_t>((slots ? static_cast<int64_t>(slots[s]) : 0) * slot_stride);
}

// Grid (channels / kThreads, ceil(n / kConvRowsPerBlock)); each thread owns one channel.
// Row t belongs to sequence s = t / rows_per_seq; its past rows start at seq_state_off_(s).
__global__ void ple_conv_add_kernel(const half* __restrict__ gv, const half* __restrict__ gvn,
                                    const half* __restrict__ w, const half* __restrict__ conv_state,
                                    int64_t slot_stride, const int* __restrict__ slots, int rows_per_seq,
                                    half* __restrict__ hidden, int channels, int n, int kernel, int dilation,
                                    int state_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels)
        return;
    float wk[kMaxKernel];
    for (int k = 0; k < kernel; k++)
        wk[k] = __half2float(w[static_cast<size_t>(c) * kernel + k]);
    const int t0 = blockIdx.y * kConvRowsPerBlock;
    const int t1 = min(n, t0 + kConvRowsPerBlock);
    for (int t = t0; t < t1; t++) {
        const int s = t / rows_per_seq;
        const int row0 = s * rows_per_seq;
        const half* st = conv_state + seq_state_off_(slot_stride, slots, s);
        float acc = 0.0f;
        for (int k = 0; k < kernel; k++) {
            const int tp = t - row0 - (kernel - 1 - k) * dilation;
            const half x = (tp >= 0) ? gvn[static_cast<size_t>(row0 + tp) * channels + c]
                                     : st[static_cast<size_t>(state_len + tp) * channels + c];
            acc += wk[k] * __half2float(x);
        }
        const size_t i = static_cast<size_t>(t) * channels + c;
        const float out = __half2float(gv[i]) + acc * sigmoidf_(acc);
        hidden[i] = __float2half(__half2float(hidden[i]) + out);
    }
}

// Grid (channels / kThreads, n_seq). One thread per (channel, sequence) reads its state_len
// new values before writing: no cross-thread hazard. d_rows (device, nullable): rows committed
// per sequence (verify chunk: real rows, or 1 for the row-0 snapshot), capped at rows_per_seq.
__global__ void ple_conv_shift_kernel(const half* __restrict__ gvn, const half* src_state, half* dst_state,
                                      int64_t slot_stride, const int* __restrict__ slots, int rows_per_seq,
                                      const int* __restrict__ d_rows, int channels, int state_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels)
        return;
    const int s = blockIdx.y;
    const int row0 = s * rows_per_seq;
    const int rows = d_rows ? min(rows_per_seq, max(0, __ldg(d_rows))) : rows_per_seq;
    const size_t off = seq_state_off_(slot_stride, slots, s);
    const half* st = src_state + off;
    half* out = dst_state + off;
    half nv[kMaxKernel * kMaxKernel];
    for (int r = 0; r < state_len; r++) {
        const int tp = rows - state_len + r;
        nv[r] = (tp >= 0) ? gvn[static_cast<size_t>(row0 + tp) * channels + c]
                          : st[static_cast<size_t>(state_len + tp) * channels + c];
    }
    for (int r = 0; r < state_len; r++)
        out[static_cast<size_t>(r) * channels + c] = nv[r];
}

}  // namespace

void ple_gate_value(const Tensor& key, Tensor& q_gv, const Tensor& value, int hc, int d,
                    cudaStream_t stream) {
    const int rows = static_cast<int>(q_gv.shape[0]);
    if (rows == 0)
        return;
    ple_gate_value_kernel<<<rows * hc, kThreads, 0, stream>>>(static_cast<const half*>(key.data),
                                                              static_cast<half*>(q_gv.data),
                                                              static_cast<const half*>(value.data), hc, d,
                                                              rsqrtf(static_cast<float>(d)));
    IMP_CUDA_CHECK_LAUNCH();
}

void ple_conv_add(const Tensor& gv, const Tensor& gvn, const Tensor& w, void* conv_state, int64_t slot_stride,
                  const int* slots, int n_seq, Tensor& hidden, int channels, int kernel, int dilation,
                  cudaStream_t stream, void* snap_state, const int* d_snap_n, const int* d_real_n) {
    const int n = static_cast<int>(gv.shape[0]);
    const int state_len = (kernel - 1) * dilation;
    if (n == 0 || n_seq <= 0)
        return;
    if (kernel > kMaxKernel || state_len > kMaxKernel * kMaxKernel || n % n_seq != 0 ||
        conv_state == nullptr) {
        IMP_LOG_ERROR("ple_conv_add: kernel %d dilation %d rows %d sequences %d state %p not supported",
                      kernel, dilation, n, n_seq, conv_state);
        return;
    }
    const int rows_per_seq = n / n_seq;
    const dim3 grid((channels + kThreads - 1) / kThreads, (n + kConvRowsPerBlock - 1) / kConvRowsPerBlock);
    ple_conv_add_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const half*>(gv.data),
                                                       static_cast<const half*>(gvn.data),
                                                       static_cast<const half*>(w.data),
                                                       static_cast<const half*>(conv_state), slot_stride,
                                                       slots, rows_per_seq, static_cast<half*>(hidden.data),
                                                       channels, n, kernel, dilation, state_len);
    IMP_CUDA_CHECK_LAUNCH();
    const dim3 sgrid(grid.x, n_seq);
    const half* gvn_h = static_cast<const half*>(gvn.data);
    half* st = static_cast<half*>(conv_state);
    if (snap_state != nullptr && d_snap_n != nullptr && n_seq == 1) {  // before the commit overwrites st
        ple_conv_shift_kernel<<<sgrid, kThreads, 0, stream>>>(gvn_h, st, static_cast<half*>(snap_state), 0,
                                                              nullptr, rows_per_seq, d_snap_n, channels,
                                                              state_len);
        IMP_CUDA_CHECK_LAUNCH();
    }
    ple_conv_shift_kernel<<<sgrid, kThreads, 0, stream>>>(gvn_h, st, st, slot_stride, slots, rows_per_seq,
                                                          n_seq == 1 ? d_real_n : nullptr, channels,
                                                          state_len);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp

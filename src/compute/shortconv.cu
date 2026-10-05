#include "compute/shortconv.h"

#include "core/cuda_errors.h"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kMaxKernel = 8;

__device__ __forceinline__ float* window_of(void* state_or_pool, const int* seq_slots,
                                            int64_t slot_stride_bytes, int s) {
    char* base = static_cast<char*>(state_or_pool);
    if (seq_slots != nullptr)
        base += static_cast<int64_t>(seq_slots[s]) * slot_stride_bytes;
    return reinterpret_cast<float*>(base);
}

// v of token j of sequence s at channel c; j < 0 reads the window (window[L + j]).
__device__ __forceinline__ float v_at(const half* bcx, const float* win, int s, int n_tok, int j, int c,
                                      int hidden, int kernel_size) {
    if (j < 0)
        return win[static_cast<int64_t>(c) * kernel_size + kernel_size + j];
    const half* row = bcx + (static_cast<int64_t>(s) * n_tok + j) * 3 * hidden;
    return __half2float(row[c]) * __half2float(row[2 * hidden + c]);
}

// Grid (ceil(H/256), n_tok, n_seq): one output element each.
__global__ void shortconv_out_kernel(void* state_or_pool, const int* __restrict__ seq_slots,
                                     int64_t slot_stride_bytes, const half* __restrict__ bcx,
                                     const half* __restrict__ weight, half* __restrict__ y, int n_tok,
                                     int hidden, int kernel_size) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden)
        return;
    const int t = blockIdx.y, s = blockIdx.z;
    const float* win = window_of(state_or_pool, seq_slots, slot_stride_bytes, s);
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k)
        acc = fmaf(__half2float(weight[static_cast<int64_t>(c) * kernel_size + k]),
                   v_at(bcx, win, s, n_tok, t - (kernel_size - 1) + k, c, hidden, kernel_size), acc);
    const int64_t row = static_cast<int64_t>(s) * n_tok + t;
    y[row * hidden + c] = __float2half(__half2float(bcx[row * 3 * hidden + hidden + c]) * acc);
}

// Grid (ceil(H/256), n_seq): window <- last L values of v (old window entries when n_tok < L).
__global__ void shortconv_state_kernel(void* state_or_pool, const int* __restrict__ seq_slots,
                                       int64_t slot_stride_bytes, const half* __restrict__ bcx, int n_tok,
                                       int hidden, int kernel_size) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden)
        return;
    const int s = blockIdx.y;
    float* win = window_of(state_or_pool, seq_slots, slot_stride_bytes, s);
    // In place, ascending: slot i reads the old window at L + j = n_tok + i > i, not yet overwritten.
    for (int i = 0; i < kernel_size; ++i)
        win[static_cast<int64_t>(c) * kernel_size + i] = v_at(bcx, win, s, n_tok, n_tok - kernel_size + i, c,
                                                              hidden, kernel_size);
}

}  // namespace

void shortconv_forward(void* state_or_pool, const int* seq_slots, int64_t slot_stride_bytes, const half* bcx,
                       const half* weight, half* y, int n_seq, int n_tok, int hidden, int kernel_size,
                       cudaStream_t stream) {
    if (n_seq <= 0 || n_tok <= 0)
        return;
    if (kernel_size < 1 || kernel_size > kMaxKernel) {
        IMP_LOG_ERROR("shortconv: kernel size %d outside 1..%d", kernel_size, kMaxKernel);
        return;
    }
    constexpr int kThreads = 256;
    const int cblocks = (hidden + kThreads - 1) / kThreads;
    shortconv_out_kernel<<<dim3(cblocks, n_tok, n_seq), kThreads, 0, stream>>>(state_or_pool, seq_slots,
                                                                               slot_stride_bytes, bcx, weight,
                                                                               y, n_tok, hidden, kernel_size);
    IMP_CUDA_CHECK_LAUNCH();
    shortconv_state_kernel<<<dim3(cblocks, n_seq), kThreads, 0, stream>>>(state_or_pool, seq_slots,
                                                                          slot_stride_bytes, bcx, n_tok,
                                                                          hidden, kernel_size);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp

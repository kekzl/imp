// KDA token-serial scan (fla fused_recurrent_kda semantics), state columns in registers.
// Grid (n_heads, n_seq, kVSplit); 2 threads per value column, each owning 64 key rows.

#include "compute/kda_common.cuh"
#include "core/logging.h"

namespace imp {

using namespace kda_detail;

namespace {

constexpr int kRecSplit = 2;                     // threads per value column
constexpr int kRecRows = kHD / kRecSplit;        // key rows per thread
constexpr int kRecThreads = kVCols * kRecSplit;  // 64

template <typename StateT, bool kSoftplus>
__global__ void __launch_bounds__(kRecThreads) kda_recurrent_kernel(KdaScanArgs a) {
    const int h = static_cast<int>(blockIdx.x);
    const int seq = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int part = tid % kRecSplit;  // adjacent lanes share a column
    const int d = (static_cast<int>(blockIdx.z) * kVCols) + (tid / kRecSplit);
    const int s_base = part * kRecRows;

    __shared__ float s_q[kHD];
    __shared__ float s_k[kHD];
    __shared__ float s_decay[kHD];
    __shared__ float s_red[2][kRecThreads / 32];

    const KdaSeq sq = kda_resolve_seq(a, seq);
    StateT* state = kda_state_ptr<StateT>(a, seq, h);
    const int HK = a.n_heads * kHD;
    const float a_exp = -a.neg_a[h];
    const float lower_bound = a.lower_bound;
    const float scale = rsqrtf(static_cast<float>(kHD));

    float H[kRecRows];
#pragma unroll
    for (int s = 0; s < kRecRows; ++s)
        H[s] = kda_load(state + (static_cast<size_t>(s_base + s) * kHD) + d);

    for (int t = 0; t < sq.n_rows; ++t) {
        const size_t row = sq.row0 + static_cast<size_t>(t);
        const float* conv = a.conv_f32 + (row * static_cast<size_t>(a.conv_channels));
        const float* q_src = conv + (static_cast<size_t>(h) * kHD);
        const float* k_src = conv + HK + (static_cast<size_t>(h) * kHD);
        const half* f_src = a.f + (row * static_cast<size_t>(HK)) + (static_cast<size_t>(h) * kHD);
        const float* dtb = a.dt_bias + (static_cast<size_t>(h) * kHD);

        float qq = 0.0f, kk = 0.0f;
        for (int i = tid; i < kHD; i += kRecThreads) {
            const float qv = q_src[i];
            const float kv = k_src[i];
            s_q[i] = qv;
            s_k[i] = kv;
            qq += qv * qv;
            kk += kv * kv;
            s_decay[i] = __expf(kda_log_decay(__half2float(f_src[i]), a_exp, dtb[i], lower_bound, kSoftplus));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            qq += __shfl_xor_sync(0xffffffffu, qq, off);
            kk += __shfl_xor_sync(0xffffffffu, kk, off);
        }
        if ((tid & 31) == 0) {
            s_red[0][tid >> 5] = qq;
            s_red[1][tid >> 5] = kk;
        }
        __syncthreads();
        float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kRecThreads / 32; ++w) {
            q_sum += s_red[0][w];
            k_sum += s_red[1][w];
        }
        const float q_inv = rsqrtf(q_sum + 1e-6f) * scale;
        const float k_inv = rsqrtf(k_sum + 1e-6f);
        const float beta = kda_sigmoid(__half2float(a.beta[(row * static_cast<size_t>(a.n_heads)) + h]));
        const float v_d = conv[(2 * static_cast<size_t>(HK)) + (static_cast<size_t>(h) * kHD) + d];

        // Decay rows, then kv = k^T S (normalisation folded into k_inv).
        float kv = 0.0f;
#pragma unroll
        for (int s = 0; s < kRecRows; ++s) {
            H[s] *= s_decay[s_base + s];
            kv += H[s] * s_k[s_base + s];
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        const float delta = (v_d - (kv * k_inv)) * beta * k_inv;
        float o = 0.0f;
#pragma unroll
        for (int s = 0; s < kRecRows; ++s) {
            H[s] += s_k[s_base + s] * delta;
            o += H[s] * s_q[s_base + s];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        if (part == 0)
            a.y[(row * static_cast<size_t>(HK)) + (static_cast<size_t>(h) * kHD) + d] = __float2half(o *
                                                                                                     q_inv);
        if (t + 1 == sq.real_n) {
#pragma unroll
            for (int s = 0; s < kRecRows; ++s)
                kda_store(state + (static_cast<size_t>(s_base + s) * kHD) + d, H[s]);
        }
        __syncthreads();  // s_q/s_k/s_decay/s_red are rewritten by the next token
    }
}

}  // namespace

void kda_scan_recurrent(const KdaScanArgs& a, cudaStream_t stream) {
    kda_check_args(a, "kda_scan_recurrent");
    if (a.n_seq <= 0 || (!a.seq_row_offsets && a.n_tokens <= 0))
        return;
    const dim3 grid(static_cast<unsigned>(a.n_heads), static_cast<unsigned>(a.n_seq), kVSplit);
    // Gate kind as a template argument: a runtime branch in the token loop gave the BF16 kernel an 8 B
    // stack frame (168 registers either way).
    if (a.state_bf16 && a.softplus_gate) {
        kda_recurrent_kernel<__nv_bfloat16, true><<<grid, kRecThreads, 0, stream>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (a.state_bf16) {
        kda_recurrent_kernel<__nv_bfloat16, false><<<grid, kRecThreads, 0, stream>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (a.softplus_gate) {
        kda_recurrent_kernel<float, true><<<grid, kRecThreads, 0, stream>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        kda_recurrent_kernel<float, false><<<grid, kRecThreads, 0, stream>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace imp

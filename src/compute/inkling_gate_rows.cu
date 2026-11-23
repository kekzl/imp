// Inkling2-Loop message gates for calls of many rows (inkling_ops.h): the two projections of a gate are
// fp16 tensor-core GEMMs over all rows. The block forms (gate_down_k, gate_mix_k) took 157 ms of an 850 ms
// prefill of 3111 tokens.

#include "compute/inkling_ops.h"

#include "compute/gemm.h"
#include "compute/inkling_detail.cuh"
#include "core/logging.h"
#include "core/tensor.h"

namespace imp::inkling {

namespace {

using namespace detail;

// One block per (row, gate): x = fp16([h * unit-RMS factor, msg * unit-RMS factor]), [2H] per row.
__global__ void gate_prep_rows_k(GateNormArgs a, half* x, int T, int H) {
    __shared__ float red[kThreads / 32];
    const int t = blockIdx.x / a.n, g = blockIdx.x % a.n;
    const float* h = a.h[g] + (size_t)t * H;
    const float* msg = a.msg[g] + (size_t)t * H;
    half* o = x + ((size_t)g * T + t) * 2 * H;
    float sh = 0.0f, sm = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        sh += h[i] * h[i];
        sm += msg[i] * msg[i];
    }
    const float rh = rsqrtf(block_sum(sh, red) / H + a.eps);
    const float rm = rsqrtf(block_sum(sm, red) / H + a.eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        o[i] = __float2half(h[i] * rh);
        o[H + i] = __float2half(msg[i] * rm);
    }
}

// o = (1 - tanh(forget)) * h + tanh(gain) * msg.
__global__ void gate_out_rows_k(const float* h, const float* msg, const half* forget, const half* gain,
                                float* o, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        o[i] = (1.0f - tanhf(__half2float(forget[i]))) * h[i] + tanhf(__half2float(gain[i])) * msg[i];
}

// y [T, N] = x [T, K] . w [N, K]^T through gemm(): its algo pins keep a row's bits independent of T, so
// a prefill resumed at a checkpoint computes the rows a fresh run does.
void lin(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s) {
    const int64_t as[2] = {T, K}, bs[2] = {N, K}, cs[2] = {T, N};
    const Tensor A(const_cast<half*>(x), QType::F16, 2, as, true);
    const Tensor B(const_cast<half*>(w), QType::F16, 2, bs, true);
    Tensor C(y, QType::F16, 2, cs, true);
    gemm(A, B, C, 1.0f, 0.0f, s);
}

}  // namespace

bool gate_norm_rows(const GateNormArgs& a, const GateOut& o, const GateWork& w, int T, int H,
                    cudaStream_t s) {
    if (!w.x || !w.g || !w.forget || !w.gain || a.n < 1 || a.n > 3 || a.rank < 1)
        return false;
    gate_prep_rows_k<<<T * a.n, kThreads, 0, s>>>(a, w.x, T, H);
    IMP_CUDA_CHECK_LAUNCH();
    const size_t n = static_cast<size_t>(T) * H;
    for (int g = 0; g < a.n; ++g) {
        const half* x = w.x + static_cast<size_t>(g) * T * 2 * H;
        half* gg = w.g + static_cast<size_t>(g) * T * a.rank;
        half* forget = w.forget + static_cast<size_t>(g) * n;
        half* gain = w.gain + static_cast<size_t>(g) * n;
        lin(x, T, 2 * H, a.down[g], a.rank, gg, s);
        lin(gg, T, a.rank, a.forget[g], H, forget, s);
        lin(gg, T, a.rank, a.gain[g], H, gain, s);
        gate_out_rows_k<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(
            a.h[g], a.msg[g], forget, gain, o.o[g], n);
        IMP_CUDA_CHECK_LAUNCH();
    }
    gate_rms(a, o, T, H, s);
    return true;
}

}  // namespace imp::inkling

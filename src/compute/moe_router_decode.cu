// Decode MoE router: GEMV + sigmoid top-k in one launch (moe_router_decode.h).
#include "compute/moe_router_decode.h"
#include "compute/gemm_internal_device.cuh"
#include "core/logging.h"
#include "core/pdl_device.cuh"
#include "core/pdl_launch.cuh"

#include <cfloat>

namespace imp {

namespace {

constexpr int kRowsPerCta = 4;  // one warp per router row
constexpr int kThreads = kRowsPerCta * 32;
constexpr int kMaxSlots = 16;  // experts per lane in the select warp, ne <= 512
constexpr int kMaxTopK = 16;

// Top-k over the sigmoid scores by warp 0 of the last CTA; same selection, weights and renorm order as
// topk_gating_kernel + gating_write_topk, ties to the lowest expert index. Each lane sorts its SLOTS
// experts once (odd-even network, registers), then every round compares only the 32 lane heads; the
// winner shifts its list. A per-round scan of dynamically cleared slots ran from local memory (4.4 us).
struct Cand {
    float s;  // selection score (sigmoid + bias)
    float p;  // sigmoid, the weight
    int i;    // expert
};

__device__ __forceinline__ bool better(const Cand& a, const Cand& b) {
    return a.s > b.s || (a.s == b.s && a.i < b.i);
}

template <int SLOTS>
__device__ void router_select_warp(const float* logits, int ne, int top_k, bool normalize,
                                   const half* __restrict__ score_bias, int32_t* ids, float* weights) {
    __shared__ float s_tv[kMaxTopK];
    __shared__ int s_ti[kMaxTopK];
    const int lane = threadIdx.x;
    Cand c[SLOTS];
#pragma unroll
    for (int j = 0; j < SLOTS; j++) {
        const int e = lane + (32 * j);
        c[j] = {-FLT_MAX, 0.0f, 0x7fffffff};
        if (e < ne) {
            const float pj = 1.0f / (1.0f + expf(-__ldcg(&logits[e])));
            c[j] = {score_bias ? pj + __half2float(score_bias[e]) : pj, pj, e};
        }
    }
#pragma unroll
    for (int pass = 0; pass < SLOTS; pass++)
#pragma unroll
        for (int j = pass & 1; j + 1 < SLOTS; j += 2)
            if (better(c[j + 1], c[j])) {
                const Cand t = c[j];
                c[j] = c[j + 1];
                c[j + 1] = t;
            }
#pragma unroll 1
    for (int k = 0; k < top_k; k++) {
        float v = c[0].s;
        int ix = c[0].i;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
            const int oi = __shfl_xor_sync(0xFFFFFFFF, ix, off);
            if (ov > v || (ov == v && oi < ix)) {
                v = ov;
                ix = oi;
            }
        }
        const float pv = __shfl_sync(0xFFFFFFFF, c[0].p, ix & 31);
        if (lane == 0) {
            s_tv[k] = score_bias ? pv : v;
            s_ti[k] = ix;
        }
        if ((ix & 31) == lane) {
#pragma unroll
            for (int j = 0; j + 1 < SLOTS; j++)
                c[j] = c[j + 1];
            c[SLOTS - 1] = {-FLT_MAX, 0.0f, 0x7fffffff};
        }
    }
    __syncwarp();
    if (lane != 0)
        return;
    float multiplier = 1.0f;
    if (normalize) {
        float norm = 0.0f;
        for (int k = 0; k < top_k; k++)
            norm += s_tv[k];
        multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    }
    for (int k = 0; k < top_k; k++) {
        ids[k] = s_ti[k];
        weights[k] = s_tv[k] * multiplier;
    }
}

// ITERS = K / 64 half2 per lane, compile-time: a runtime bound left 64 predicated registers that the
// compiler rematerialised as a second weight load after the PDL wait.
template <int ITERS, int SLOTS>
__global__ void __launch_bounds__(kThreads) moe_router_decode_kernel(
    const half* __restrict__ W, const half* __restrict__ x, float* logits, int ne, int top_k, bool normalize,
    const half* __restrict__ score_bias, int32_t* ids, float* weights, unsigned int* ticket) {
    constexpr int K = ITERS * 64;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = min((blockIdx.x * kRowsPerCta) + warp, ne - 1);  // tail warps recompute the last row
    const bool owner = (blockIdx.x * kRowsPerCta) + warp < ne;

    // Router weights are immutable: issue every load before waiting on the producer of x.
    half2 w[ITERS];
    const half2* W2 = reinterpret_cast<const half2*>(W + (static_cast<size_t>(row) * K));
#pragma unroll
    for (int t = 0; t < ITERS; t++)
        w[t] = __ldg(&W2[lane + (32 * t)]);
    pdl_wait();
    pdl_trigger();
    // Element order and expression of gemv_gate_fp32_kernel: the logits are bit-identical.
    const half2* x2 = reinterpret_cast<const half2*>(x);
    float sum = 0.0f;
#pragma unroll
    for (int t = 0; t < ITERS; t++) {
        const half2 v = x2[lane + (32 * t)];
        sum += __half2float(w[t].x) * __half2float(v.x);
        sum += __half2float(w[t].y) * __half2float(v.y);
    }
    sum = warp_reduce_sum(sum);
    if (owner && lane == 0)
        logits[row] = sum;
    // One acq_rel fence per CTA around the ticket (cumulative over the barrier), not __threadfence()
    // (fence.sc) in every writer.
    __syncthreads();
    __shared__ bool s_last;
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const unsigned t = atomicAdd(ticket, 1U);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        s_last = t == gridDim.x - 1;
        if (s_last)
            *ticket = 0;  // re-arm: the next launch is stream-ordered after this grid
    }
    __syncthreads();
    if (!s_last || warp != 0)
        return;
    router_select_warp<SLOTS>(logits, ne, top_k, normalize, score_bias, ids, weights);
}

template <int ITERS, int SLOTS>
void launch_router(const MoeRouterDecodeArgs& a, cudaStream_t stream) {
    const dim3 grid((a.ne + kRowsPerCta - 1) / kRowsPerCta);
    pdl::enable_kernel(moe_router_decode_kernel<ITERS, SLOTS>);
    pdl::launch(moe_router_decode_kernel<ITERS, SLOTS>, grid, dim3(kThreads), size_t(0), stream, a.w, a.x,
                a.logits, a.ne, a.top_k, a.normalize, a.score_bias, a.ids, a.weights, a.ticket);
    IMP_CUDA_CHECK_LAUNCH();
}

template <int ITERS>
void launch_router(const MoeRouterDecodeArgs& a, cudaStream_t stream) {
    if (a.ne <= 256)
        launch_router<ITERS, 8>(a, stream);
    else
        launch_router<ITERS, 16>(a, stream);
}

}  // namespace

bool moe_router_decode(const MoeRouterDecodeArgs& a, cudaStream_t stream) {
    if (a.w == nullptr || a.x == nullptr || a.logits == nullptr || a.ids == nullptr || a.weights == nullptr ||
        a.ticket == nullptr)
        return false;
    if (a.ne <= 0 || a.ne > 32 * kMaxSlots || a.top_k < 1 || a.top_k > kMaxTopK || a.top_k > a.ne)
        return false;
    switch (a.K) {
        case 1024:
            launch_router<16>(a, stream);
            return true;
        case 2048:
            launch_router<32>(a, stream);
            return true;
        case 3072:
            launch_router<48>(a, stream);
            return true;
        case 4096:
            launch_router<64>(a, stream);
            return true;
        default:
            return false;
    }
}

}  // namespace imp

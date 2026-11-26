// Inkling2-Loop building blocks (inkling_ops.h). Reference-grade kernels: one block per token or
// per (token, head), no tensor cores.

#include "compute/inkling_ops.h"

#include "compute/inkling_detail.cuh"
#include "compute/inkling_fused.cuh"
#include "core/logging.h"

#include <cfloat>

namespace imp::inkling {

namespace {

using namespace detail;

__device__ inline float bf16_bits(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ inline float e4m3(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float mag = e == 0 ? scalbnf(static_cast<float>(m), -9)
                             : scalbnf(static_cast<float>(8 + m), e - 10);
    return (b & 0x80) ? -mag : mag;
}

inline unsigned grid_for(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

__global__ void bf16_to_f16_k(const uint16_t* in, half* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(bf16_bits(in[i]));
}
__global__ void bf16_to_f32_k(const uint16_t* in, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = bf16_bits(in[i]);
}
__global__ void f32_to_f16_k(const float* in, half* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(in[i]);
}
__global__ void f16_to_f32_k(const half* in, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}
__global__ void add_f32_k(const float* a, const float* b, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = b ? a[i] + b[i] : a[i];
}

__global__ void embed_rows_k(const half* table, const int32_t* ids, float* out, int H) {
    const half* row = table + (size_t)ids[blockIdx.x] * H;
    float* o = out + (size_t)blockIdx.x * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        o[i] = __half2float(row[i]);
}

__global__ void rms_norm_k(const float* x, const float* w, float* of, half* oh, int H, float eps) {
    __shared__ float sh[kThreads / 32];
    const float* xr = x + (size_t)blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xr[i] * r * (w ? w[i] : 1.0f);
        if (of)
            of[(size_t)blockIdx.x * H + i] = v;
        if (oh)
            oh[(size_t)blockIdx.x * H + i] = __float2half(v);
    }
}

constexpr int kRouteMaxH = 2304, kRouteWarps = 4;

// Router logits: one warp per expert, kRouteWarps experts per block. The row and each warp's weight row
// go through shared memory (8 halves / 4 floats per load: 18 loads per lane instead of 144); a lane
// then sums its strided elements in the order it always did, so the logits are bit-equal.
__global__ void route_scores_k(const half* x, const float* w, float* logits, int H, int n_experts,
                               int blocks_per_tok) {
    __shared__ float xs[kRouteMaxH];
    __shared__ float ws[kRouteWarps][kRouteMaxH];
    const int t = blockIdx.x / blocks_per_tok, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = (blockIdx.x % blocks_per_tok) * kRouteWarps + warp;
    const bool live = e < n_experts;
    const half* xr = x + (size_t)t * H;
    const float* wr = w + (size_t)(live ? e : 0) * H;
    const bool tiled = H <= kRouteMaxH && (H & 7) == 0;
    if (tiled) {
        for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8) {
            const uint4 p = *reinterpret_cast<const uint4*>(xr + i);
            for (int k = 0; k < 8; ++k)
                xs[i + k] = __half2float(reinterpret_cast<const half*>(&p)[k]);
        }
        for (int i = lane * 4; live && i < H; i += 128)
            *reinterpret_cast<float4*>(&ws[warp][i]) = *reinterpret_cast<const float4*>(wr + i);
    }
    __syncthreads();
    float acc = 0.0f;
    if (live && tiled)
        for (int i = lane; i < H; i += 32)
            acc += xs[i] * ws[warp][i];
    else if (live)
        for (int i = lane; i < H; i += 32)
            acc += __half2float(xr[i]) * wr[i];
    acc = warp_sum(acc);
    if (live && lane == 0)
        logits[(size_t)t * n_experts + e] = acc;
}

// route_scores_k for hidden 2304: a lane's 18 weight loads and the thread's row loads are issued before
// the first store. Same sums in the same order: bit-equal (FixedShapeKernelsAreBitEqual).
constexpr int kRouteFixH = 2304;
__global__ void route_scores_fixed_k(const half* x, const float* w, float* logits, int n_experts,
                                     int blocks_per_tok) {
    constexpr int H = kRouteFixH, kXl = (H / 8 + kRouteWarps * 32 - 1) / (kRouteWarps * 32), kWl = H / 128;
    __shared__ float xs[H];
    __shared__ float ws[kRouteWarps][H];
    const int t = blockIdx.x / blocks_per_tok, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = (blockIdx.x % blocks_per_tok) * kRouteWarps + warp;
    const bool live = e < n_experts;
    const half* xr = x + (size_t)t * H;
    const float* wr = w + (size_t)(live ? e : 0) * H;
    uint4 xp[kXl];
    float4 wp[kWl];
    for (int j = 0; j < kXl; ++j)
        xp[j] = *reinterpret_cast<const uint4*>(xr +
                                                min((int)threadIdx.x * 8 + j * kRouteWarps * 256, H - 8));
    for (int j = 0; j < kWl; ++j)
        wp[j] = *reinterpret_cast<const float4*>(wr + (size_t)(lane * 4 + j * 128));
    for (int j = 0; j < kXl; ++j) {
        const int i = threadIdx.x * 8 + j * kRouteWarps * 256;
        if (i >= H)
            continue;
        for (int k = 0; k < 8; ++k)
            xs[i + k] = __half2float(reinterpret_cast<const half*>(&xp[j])[k]);
    }
    for (int j = 0; live && j < kWl; ++j)
        *reinterpret_cast<float4*>(&ws[warp][lane * 4 + j * 128]) = wp[j];
    __syncthreads();
    float acc = 0.0f;
    if (live)
        for (int i = lane; i < H; i += 32)
            acc += xs[i] * ws[warp][i];
    acc = warp_sum(acc);
    if (live && lane == 0)
        logits[(size_t)t * n_experts + e] = acc;
}

// One block per token, one thread per expert (n_experts <= 128).
__global__ void route_pick_k(const float* logits, const float* bias, int32_t* idx, float* weights,
                             int n_experts, int top_k, float route_scale) {
    __shared__ float sel[128];
    __shared__ float sh[4];
    const int e = threadIdx.x;
    float score = 0.0f;
    if (e < n_experts) {
        score = 1.0f / (1.0f + expf(-logits[(size_t)blockIdx.x * n_experts + e]));
        sel[e] = score + bias[e];
    }
    __syncthreads();
    int rank = top_k;
    if (e < n_experts) {
        rank = 0;
        for (int j = 0; j < n_experts; ++j)
            rank += (sel[j] > sel[e]) || (sel[j] == sel[e] && j < e);
    }
    const float sum = block_sum(rank < top_k ? score : 0.0f, sh);
    if (rank < top_k) {
        idx[(size_t)blockIdx.x * top_k + rank] = e;
        weights[(size_t)blockIdx.x * top_k + rank] = score / (sum + 1e-20f) * route_scale;
    }
}

__global__ void expert_act_k(const half* gu, half* out, size_t n, float gate_cap, float up_cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float g = __half2float(gu[2 * i]), u = __half2float(gu[2 * i + 1]);
    out[i] = __float2half(gate_cap * tanhf(g / gate_cap) / (1.0f + expf(-g)) * up_cap * tanhf(u / up_cap));
}

constexpr int kFp8Rows = 8;  // rows per block, one warp each

__global__ void gemv_mxfp8_slots_k(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x,
                                   half* y, int rows, int K, int x_group, int blocks_per_slot) {
    const int slot = blockIdx.x / blocks_per_slot;
    const int row = (blockIdx.x % blocks_per_slot) * kFp8Rows + (threadIdx.x >> 5);
    if (row >= rows)
        return;
    const int lane = threadIdx.x & 31;
    const size_t e = static_cast<size_t>(idx[slot]);
    const uint8_t* wr = w + (e * rows + row) * K;
    const uint8_t* sr = sc + (e * rows + row) * (K / 32);
    const half* xr = x + (size_t)(slot / x_group) * K;
    float acc = 0.0f;
    for (int b = 0; b < K / 32; ++b) {
        const int k = b * 32 + lane;
        acc += e4m3(wr[k]) * scalbnf(1.0f, static_cast<int>(sr[b]) - 127) * __half2float(xr[k]);
    }
    acc = warp_sum(acc);
    if (lane == 0)
        y[(size_t)slot * rows + row] = __float2half(acc);
}

__global__ void softcap_f32_k(const half* x, float* out, size_t n, float cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = cap * tanhf(__half2float(x[i]) / cap);
}

}  // namespace

void bf16_to_f16(const uint16_t* in, half* out, size_t n, cudaStream_t s) {
    bf16_to_f16_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void bf16_to_f32(const uint16_t* in, float* out, size_t n, cudaStream_t s) {
    bf16_to_f32_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void f32_to_f16(const float* in, half* out, size_t n, cudaStream_t s) {
    f32_to_f16_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void f16_to_f32(const half* in, float* out, size_t n, cudaStream_t s) {
    f16_to_f32_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void add_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t s) {
    add_f32_k<<<grid_for(n), kThreads, 0, s>>>(a, b, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void embed_rows(const half* table, const int32_t* ids, float* out, int T, int H, cudaStream_t s) {
    embed_rows_k<<<T, kThreads, 0, s>>>(table, ids, out, H);
    IMP_CUDA_CHECK_LAUNCH();
}
void rms_norm(const float* x, const float* w, float* out_f, half* out_h, int T, int H, float eps,
              cudaStream_t s) {
    rms_norm_k<<<T, kThreads, 0, s>>>(x, w, out_f, out_h, H, eps);
    IMP_CUDA_CHECK_LAUNCH();
}
void route_topk(const half* x, const float* w, const float* bias, float* logits, int32_t* idx, float* weights,
                int T, int H, int n_experts, int top_k, float route_scale, cudaStream_t s) {
    if (n_experts > 128 || top_k > 16)
        throw_runtime_error("inkling router: %d experts / top %d out of range", n_experts, top_k);
    const int bpt = (n_experts + kRouteWarps - 1) / kRouteWarps;
    if (H == kRouteFixH && detail::fixed_shape_kernels()) {
        route_scores_fixed_k<<<T * bpt, kRouteWarps * 32, 0, s>>>(x, w, logits, n_experts, bpt);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        route_scores_k<<<T * bpt, kRouteWarps * 32, 0, s>>>(x, w, logits, H, n_experts, bpt);
        IMP_CUDA_CHECK_LAUNCH();
    }
    route_pick_k<<<T, 128, 0, s>>>(logits, bias, idx, weights, n_experts, top_k, route_scale);
    IMP_CUDA_CHECK_LAUNCH();
}
void expert_act(const half* gu, half* out, int slots, int I, float gate_cap, float up_cap, cudaStream_t s) {
    const size_t n = (size_t)slots * I;
    expert_act_k<<<grid_for(n), kThreads, 0, s>>>(gu, out, n, gate_cap, up_cap);
    IMP_CUDA_CHECK_LAUNCH();
}
void gemv_mxfp8_slots(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x, half* y,
                      int slots, int rows, int K, int x_group, cudaStream_t s) {
    const int bps = (rows + kFp8Rows - 1) / kFp8Rows;
    gemv_mxfp8_slots_k<<<slots * bps, kFp8Rows * 32, 0, s>>>(w, sc, idx, x, y, rows, K, x_group, bps);
    IMP_CUDA_CHECK_LAUNCH();
}
void softcap_f32(const half* x, float* out, size_t n, float cap, cudaStream_t s) {
    softcap_f32_k<<<grid_for(n), kThreads, 0, s>>>(x, out, n, cap);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling

// Inkling2-Loop MoE for calls of many rows (inkling_ops.h): the (row, expert) slots sorted by expert, so
// a grouped GEMM reads every routed expert once per call instead of once per slot.

#include "compute/inkling_ops.h"

#include "core/logging.h"

namespace imp::inkling {

namespace {

constexpr int kSortThreads = 1024;
constexpr int kSortMaxSlots = 4096;
constexpr int kSortMaxExperts = 512;

// One block: counting sort of the slots by expert, stable (a slot's rank among its expert's slots is
// the number of earlier slots with that expert). perm[slot] = sorted row, off [ne + 1] = prefix sums.
__global__ void moe_sort_k(const int32_t* idx, int slots, int ne, int32_t* perm, int32_t* off) {
    __shared__ int sidx[kSortMaxSlots];
    __shared__ int cnt[kSortMaxExperts + 1];
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < slots; i += nt)
        sidx[i] = idx[i];
    for (int e = tid; e <= ne; e += nt)
        cnt[e] = 0;
    __syncthreads();
    for (int i = tid; i < slots; i += nt)
        atomicAdd(&cnt[sidx[i]], 1);
    __syncthreads();
    if (tid == 0) {
        int a = 0;
        for (int e = 0; e < ne; ++e) {
            const int c = cnt[e];
            cnt[e] = a;
            a += c;
        }
        cnt[ne] = a;
    }
    __syncthreads();
    for (int e = tid; e <= ne; e += nt)
        off[e] = cnt[e];
    for (int i = tid; i < slots; i += nt) {
        const int e = sidx[i];
        int r = 0;
        for (int j = 0; j < i; ++j)
            r += sidx[j] == e ? 1 : 0;
        perm[i] = cnt[e] + r;
    }
}

// 16-byte pieces: to_sorted: dst[perm[slot]] = src[slot / group]; else dst[slot] = src[perm[slot]].
__global__ void moe_rows_copy_k(const half* src, const int32_t* perm, half* dst, size_t n, int per, int C,
                                int group, bool to_sorted) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const size_t slot = id / per, c = (id % per) * 8;
    const size_t p = static_cast<size_t>(perm[slot]);
    const size_t from = to_sorted ? slot / group : p, to = to_sorted ? p : slot;
    *reinterpret_cast<uint4*>(dst + to * C + c) = *reinterpret_cast<const uint4*>(src + from * C + c);
}

// One block per expert: the largest exponent byte of its n.
__global__ void mxfp8_bias_k(const uint8_t* sc, size_t n, int32_t* bias, float* ts) {
    __shared__ int red[8];
    const uint8_t* p = sc + (size_t)blockIdx.x * n;
    int m = 0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        m = max(m, static_cast<int>(p[i]));
    for (int o = 16; o > 0; o >>= 1)
        m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 8; ++i)
            m = max(m, red[i]);
        bias[blockIdx.x] = m - 6;
        ts[blockIdx.x] = scalbnf(1.0f, m - 6 - 127);
    }
}

}  // namespace

void mxfp8_expert_bias(const uint8_t* sc, int n_experts, size_t bytes_per_expert, int32_t* bias, float* ts,
                       cudaStream_t s) {
    mxfp8_bias_k<<<n_experts, 256, 0, s>>>(sc, bytes_per_expert, bias, ts);
    IMP_CUDA_CHECK_LAUNCH();
}

bool moe_sort(const int32_t* idx, int slots, int n_experts, int32_t* perm, int32_t* offsets, cudaStream_t s) {
    if (slots > kSortMaxSlots || n_experts > kSortMaxExperts)
        return false;
    moe_sort_k<<<1, kSortThreads, 0, s>>>(idx, slots, n_experts, perm, offsets);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void moe_rows_copy(const half* src, const int32_t* perm, half* dst, int slots, int C, int group,
                   bool to_sorted, cudaStream_t s) {
    const int per = C / 8;
    const size_t n = static_cast<size_t>(slots) * per;
    moe_rows_copy_k<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(src, perm, dst, n, per, C, group,
                                                                           to_sorted);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling

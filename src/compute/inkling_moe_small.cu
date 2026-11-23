// Inkling2-Loop expert projections for calls of 2..8 rows (inkling_ops.h): the slots grouped by expert, one
// tensor-core tile of up to 8 rows per chosen expert. The slot GEMVs compute every slot in scalar code:
// the experts took 4.46 ms of a 9.50 ms four-row call and 5.97 ms of a 12.23 ms eight-row call, against
// 1.57 ms of a one-row call.

#include "compute/inkling_ops.h"

#include "compute/gemm_mxfp8_wdq_dec.cuh"
#include "compute/gemm_nvfp4_wdq_dec.cuh"
#include "core/logging.h"

namespace imp::inkling {

namespace {

using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the decoders and the MMA helpers

constexpr int kSmBN = 64;        // weight rows per block
constexpr int kSmThreads = 128;  // 4 warps of 16 weight rows
constexpr int kSmRow = 128;      // bytes of a 64-k stage row
constexpr int kSmA = 16 * kSmRow;
constexpr int kSmAhead = 4;  // stages between a load and its decode

// One block of up to kMoeSmallSlots threads. Group g = the g-th distinct expert in slot order: gexp[g],
// gcnt[g] slots, their numbers at gslot[g * 8 ..] in slot order; gcnt = 0 for the groups not in use.
__global__ void moe_group_k(const int32_t* idx, int slots, int32_t* gexp, int32_t* gcnt, int32_t* gslot) {
    __shared__ int se[kMoeSmallSlots];
    __shared__ int first[kMoeSmallSlots];
    const int i = threadIdx.x;
    if (i < slots)
        se[i] = idx[i];
    __syncthreads();
    int rank = 0, total = 0, head = i;
    if (i < slots) {
        for (int j = slots - 1; j >= 0; --j) {
            const bool same = se[j] == se[i];
            total += same ? 1 : 0;
            rank += (same && j < i) ? 1 : 0;
            head = same ? j : head;
        }
        first[i] = rank == 0 ? 1 : 0;
    }
    __syncthreads();
    int groups = 0, gid = 0;
    for (int j = 0; j < slots; ++j) {
        groups += first[j];
        gid += j < head ? first[j] : 0;
    }
    if (i >= groups)
        gcnt[i] = 0;
    if (i >= slots || rank >= 8)
        return;
    gslot[(static_cast<size_t>(gid) * 8) + static_cast<size_t>(rank)] = i;
    if (rank == 0) {
        gexp[gid] = se[i];
        gcnt[gid] = min(total, 8);
    }
}

struct MoeSmall {
    const half* x;
    half* y;
    const int32_t *gexp, *gcnt, *gslot;
    const uint8_t* w;
    size_t w_stride;
    const uint8_t* sc;
    size_t sc_stride;
    const float* ts;
    const int32_t* bias;
    int N, K, x_group;
};

__device__ __forceinline__ int sm_chunk(int r, int c) { return c ^ (r & 7); }

// One block per (group, 64 weight rows). Per 64-k stage every thread decodes half a weight row into
// shared memory and loads one 16-byte piece of the group's input rows; then the four warps run the MMAs
// of their 16 weight rows. The next stage's loads are issued before the current one is decoded.
template <bool FP8>
__global__ void __launch_bounds__(kSmThreads) moe_small_k(MoeSmall g) {
    __shared__ __align__(128) uint8_t sm[kSmA + kSmBN * kSmRow];
    const int grp = blockIdx.y;
    const int cnt = g.gcnt[grp];
    if (cnt <= 0)
        return;
    const size_t e = static_cast<size_t>(g.gexp[grp]);
    const int n0 = static_cast<int>(blockIdx.x) * kSmBN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K = g.K, KT = K / 64;
    // Decode role: weight row drow, 32-k half hb.
    const int drow = tid >> 1, hb = tid & 1;
    const size_t n = static_cast<size_t>(n0) + static_cast<size_t>(drow);
    const size_t Kz = static_cast<size_t>(K), hz = static_cast<size_t>(hb);
    const uint8_t* wp = g.w + (e * g.w_stride) + (n * (FP8 ? Kz : Kz / 2)) + (hz * (FP8 ? 32 : 16));
    const uint8_t* sp = g.sc + (e * g.sc_stride) + (n * (FP8 ? Kz / 32 : Kz / 16)) + (hz * (FP8 ? 1 : 2));
    const int bias = FP8 ? g.bias[e] : 0;
    // Input role: row arow of the group, 16-byte piece ac of the stage.
    const int arow = tid >> 3, ac = tid & 7;
    const bool a_on = arow < cnt;
    const size_t gbase = static_cast<size_t>(grp) * 8;
    const half* ap = g.x;
    if (a_on)
        ap += (static_cast<size_t>(g.gslot[gbase + static_cast<size_t>(arow)] / g.x_group) * Kz) +
              (static_cast<size_t>(ac) * 8);
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
    uint8_t* wrow = sm + kSmA + (static_cast<size_t>(drow) * kSmRow);
    uint4* aslot = reinterpret_cast<uint4*>(sm + (static_cast<size_t>(arow) * kSmRow) +
                                            (static_cast<size_t>(sm_chunk(arow, ac)) * 16));
    const int lr = lane & 7, lm = lane >> 3, wn = warp * 16;

    // Loads run kSmAhead stages ahead of the stage being decoded: one stage ahead left every stage
    // waiting for its DRAM lines (555 GB/s on 87 MB of w13 experts).
    uint4 q0[kSmAhead], q1[kSmAhead], av[kSmAhead];
    uint32_t es[kSmAhead];
    const auto load = [&](int kt, uint4& a0, uint4& a1, uint32_t& sx, uint4& ax) {
        a0 = a1 = ax = make_uint4(0, 0, 0, 0);
        sx = 0;
        if (kt >= KT)
            return;
        const size_t k1 = static_cast<size_t>(kt);
        a0 = __ldg(reinterpret_cast<const uint4*>(wp + (k1 * (FP8 ? 64 : 32))));
        if (FP8) {
            a1 = __ldg(reinterpret_cast<const uint4*>(wp + (k1 * 64) + 16));
            sx = __ldg(sp + (k1 * 2));
        } else {
            sx = __ldg(reinterpret_cast<const uint16_t*>(sp + (k1 * 4)));
        }
        if (a_on)
            ax = __ldg(reinterpret_cast<const uint4*>(ap + (k1 * 64)));
    };
#pragma unroll
    for (int d = 0; d < kSmAhead; ++d)
        load(d, q0[d], q1[d], es[d], av[d]);
    float acc[2][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        const uint4 c0 = q0[0], c1 = q1[0], ca = av[0];
        const uint32_t ce = es[0];
#pragma unroll
        for (int d = 0; d + 1 < kSmAhead; ++d)
            q0[d] = q0[d + 1], q1[d] = q1[d + 1], es[d] = es[d + 1], av[d] = av[d + 1];
        load(kt + kSmAhead, q0[kSmAhead - 1], q1[kSmAhead - 1], es[kSmAhead - 1], av[kSmAhead - 1]);
        if (FP8) {
            const __half2 s2 = mx8_scale(static_cast<int>(ce) - bias);
            const uint32_t w8[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint4 o;
                mx8_word(w8[static_cast<size_t>(j) * 2], s2, o.x, o.y);
                mx8_word(w8[(static_cast<size_t>(j) * 2) + 1], s2, o.z, o.w);
                *reinterpret_cast<uint4*>(wrow +
                                          (static_cast<size_t>(sm_chunk(drow, (4 * hb) + j)) * 16)) = o;
            }
        } else {
            const __half2 s_lo = nv4_scale(ce & 0xFFu);
            const __half2 s_hi = nv4_scale(ce >> 8);
            const uint32_t w4[4] = {c0.x, c0.y, c0.z, c0.w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<uint4*>(wrow + (static_cast<size_t>(sm_chunk(drow, (4 * hb) + j)) *
                                                  16)) = nv4_chunk(w4[j], j < 2 ? s_lo : s_hi);
        }
        *aslot = ca;
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint32_t af[4], bf[4];
            const int ra = lr + ((lm & 1) * 8);
            ldsm_x4(af, sa + (ra * kSmRow) + (sm_chunk(ra, (c * 2) + (lm >> 1)) * 16));
            const int rb = wn + ((lm >> 1) * 8) + lr;
            ldsm_x4(bf, sa + kSmA + (rb * kSmRow) + (sm_chunk(rb, (c * 2) + (lm & 1)) * 16));
            mma32(acc[0], af, bf[0], bf[1]);
            mma32(acc[1], af, bf[2], bf[3]);
        }
        __syncthreads();
    }
    // Accumulator element (row T1 + 8 h, column wn + 8 nf + 2 T0 + {0, 1}): rows from 8 on are padding.
    const int T0 = lane & 3, T1 = lane >> 2;
    if (T1 >= cnt)
        return;
    const float alpha = g.ts[e] * (FP8 ? 1.0f : 256.0f);
    half* yr = g.y +
               (static_cast<size_t>(g.gslot[gbase + static_cast<size_t>(T1)]) * static_cast<size_t>(g.N)) +
               static_cast<size_t>(n0 + wn + (T0 * 2));
#pragma unroll
    for (int nf = 0; nf < 2; ++nf)
        *reinterpret_cast<__half2*>(
            yr + (static_cast<size_t>(nf) * 8)) = __floats2half2_rn(acc[nf][0] * alpha, acc[nf][1] * alpha);
}

}  // namespace

bool moe_group(const int32_t* idx, int slots, int32_t* groups, cudaStream_t s) {
    if (slots < 1 || slots > kMoeSmallSlots)
        return false;
    moe_group_k<<<1, kMoeSmallSlots, 0, s>>>(idx, slots, groups, groups + kMoeSmallSlots,
                                             groups + (static_cast<size_t>(kMoeSmallSlots) * 2));
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool moe_small(const MoeSmallArgs& a, cudaStream_t s) {
    const size_t wrow = a.fp8 ? static_cast<size_t>(a.K) : static_cast<size_t>(a.K) / 2;
    if (a.slots < 1 || a.slots > kMoeSmallSlots || a.N % kSmBN != 0 || a.K % 64 != 0 || a.x_group < 1 ||
        !a.x || !a.y || !a.groups || !a.w || !a.sc || !a.ts || (a.fp8 && !a.bias) || wrow % 16 != 0 ||
        reinterpret_cast<uintptr_t>(a.w) % 16 != 0 || a.w_stride % 16 != 0 ||
        reinterpret_cast<uintptr_t>(a.x) % 16 != 0 || reinterpret_cast<uintptr_t>(a.sc) % 2 != 0 ||
        a.sc_stride % 2 != 0)
        return false;
    const MoeSmall g{a.x,
                     a.y,
                     a.groups,
                     a.groups + kMoeSmallSlots,
                     a.groups + (static_cast<size_t>(kMoeSmallSlots) * 2),
                     static_cast<const uint8_t*>(a.w),
                     a.w_stride,
                     static_cast<const uint8_t*>(a.sc),
                     a.sc_stride,
                     a.ts,
                     a.bias,
                     a.N,
                     a.K,
                     a.x_group};
    const dim3 grid(a.N / kSmBN, a.slots);
    if (a.fp8)
        moe_small_k<true><<<grid, kSmThreads, 0, s>>>(g);
    else
        moe_small_k<false><<<grid, kSmThreads, 0, s>>>(g);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp::inkling

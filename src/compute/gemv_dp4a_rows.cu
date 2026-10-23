// Multi-row dp4a GEMV (1..4 activation rows) on GGUF blocks, generic over DequantTraits: the weight is read
// once per call; each row's dot product re-decodes the block (dp4a_block), cheap next to the DRAM read.

#include "compute/gemv_dp4a_rows.h"

#include "compute/gemv_dp4a_iq4_traits.cuh"
#include "compute/gemv_dp4a_traits.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kRowsMax = 4;

// A type whose block decode is worth sharing across the rows: load once, then one dot per row. Others
// call dp4a_block per row. Q4_K: same arithmetic order as q4k_dp4a_sub (bit-identical per row).
template <typename QT>
struct RowsOnce {
    static constexpr bool kOn = false;
};

template <>
struct RowsOnce<Q4_K_Traits> {
    static constexpr bool kOn = true;
    struct Block {
        uint32_t nib[8];
        float dsc, dmn;
    };
    static __device__ __forceinline__ Block load(const uint8_t* bp, int sub) {
        const uint4 hdr = *reinterpret_cast<const uint4*>(bp);
        const half2 dd = *reinterpret_cast<const half2*>(&hdr.x);
        uint8_t sc, mn;
        get_scale_min_k4_reg(hdr.y, hdr.z, hdr.w, sub, sc, mn);
        const uint8_t* qs = bp + 16 + (static_cast<ptrdiff_t>(sub / 2) * 32);
        const uint4 lo = *reinterpret_cast<const uint4*>(qs);
        const uint4 hi = *reinterpret_cast<const uint4*>(qs + 16);
        const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const uint32_t sh = (sub & 1) * 4;
        Block b;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            b.nib[j] = (w[j] >> sh) & 0x0F0F0F0Fu;
        b.dsc = __half2float(__low2half(dd)) * static_cast<float>(sc);
        b.dmn = __half2float(__high2half(dd)) * static_cast<float>(mn);
        return b;
    }
    static __device__ __forceinline__ float dot(const Block& b, const int* xi, float dq) {
        int32_t sumi = 0;
        int q8_sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = __dp4a(static_cast<int>(b.nib[j]), xi[j], sumi);
            q8_sum = __dp4a(xi[j], 0x01010101, q8_sum);
        }
        return dq * ((b.dsc * static_cast<float>(sumi)) - (b.dmn * static_cast<float>(q8_sum)));
    }
};

// kNw weight rows per block (4 at K >= 8192, else 1): each thread loads its activation slices once and reuses
// them per weight row (down_proj K=17408 at 3 rows: 48.2 -> 36.8 us; K=5120 is faster at 1).

// Q8_0: 32 s8 + fp16 d, d * dq * sumi as in DequantTraits<Q8_0>.
template <>
struct RowsOnce<Q8_0_Traits> {
    static constexpr bool kOn = true;
    struct Block {
        int w[8];
        float d;
    };
    static __device__ __forceinline__ Block load(const uint8_t* bp, int /*sub*/) {
        Block b;
        half dh;
        memcpy(&dh, bp, sizeof(half));
        b.d = __half2float(dh);
        memcpy(b.w, bp + 2, 32);
        return b;
    }
    static __device__ __forceinline__ float dot(const Block& b, const int* xi, float dq) {
        int32_t sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            sumi = __dp4a(b.w[j], xi[j], sumi);
        return b.d * dq * static_cast<float>(sumi);
    }
};

// Q5_K: nibbles and the fifth bit decoded once; same order as q5k_dp4a_sub.
template <>
struct RowsOnce<Q5_K_Traits> {
    static constexpr bool kOn = true;
    struct Block {
        uint32_t lo[8], hb[8];
        float dsc, dmn;
    };
    static __device__ __forceinline__ Block load(const uint8_t* bp, int sub) {
        const uint4 hdr = *reinterpret_cast<const uint4*>(bp);
        const half2 dd = *reinterpret_cast<const half2*>(&hdr.x);
        uint8_t sc, mn;
        get_scale_min_k4_reg(hdr.y, hdr.z, hdr.w, sub, sc, mn);
        const uint8_t* qs = bp + 48 + (static_cast<ptrdiff_t>(sub / 2) * 32);
        const uint4 q0 = *reinterpret_cast<const uint4*>(qs);
        const uint4 q1 = *reinterpret_cast<const uint4*>(qs + 16);
        const uint4 h0 = *reinterpret_cast<const uint4*>(bp + 16);
        const uint4 h1 = *reinterpret_cast<const uint4*>(bp + 32);
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint32_t sh = (sub & 1) * 4;
        Block b;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            b.lo[j] = (qw[j] >> sh) & 0x0F0F0F0Fu;
            b.hb[j] = ((hw[j] >> sub) & 0x01010101u) << 4;
        }
        b.dsc = __half2float(__low2half(dd)) * static_cast<float>(sc);
        b.dmn = __half2float(__high2half(dd)) * static_cast<float>(mn);
        return b;
    }
    static __device__ __forceinline__ float dot(const Block& b, const int* xi, float dq) {
        int32_t sumi = 0, sumi_h = 0;
        int q8_sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = __dp4a(static_cast<int>(b.lo[j]), xi[j], sumi);
            q8_sum = __dp4a(xi[j], 0x01010101, q8_sum);
            sumi_h = __dp4a(static_cast<int>(b.hb[j]), xi[j], sumi_h);
        }
        return dq * ((b.dsc * static_cast<float>(sumi + sumi_h)) - (b.dmn * static_cast<float>(q8_sum)));
    }
};

// IQ4_XS: codebook lookups once; d * (ls - 32) * dq * sumi as in DequantTraits<IQ4_XS>.
template <>
struct RowsOnce<IQ4_XS_Traits> {
    static constexpr bool kOn = true;
    struct Block {
        int lo[4], hi[4];
        float sc;
    };
    static __device__ __forceinline__ Block load(const uint8_t* bp, int sub) {
        const uint2 hdr = __ldg(reinterpret_cast<const uint2*>(bp));
        const uint2 qa = __ldg(reinterpret_cast<const uint2*>(bp + 8 + (static_cast<ptrdiff_t>(sub) * 16)));
        const uint2 qb = __ldg(reinterpret_cast<const uint2*>(bp + 16 + (static_cast<ptrdiff_t>(sub) * 16)));
        const uint32_t w[4] = {qa.x, qa.y, qb.x, qb.y};
        Block b;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = iq4nl_lut8(w[j]);
            b.lo[j] = v.x;
            b.hi[j] = v.y;
        }
        const int ls = static_cast<int>(((hdr.y >> (4 * sub)) & 0xFu) |
                                        (((hdr.x >> (16 + (2 * sub))) & 3u) << 4));
        b.sc = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xFFFFu))) *
               static_cast<float>(ls - 32);
        return b;
    }
    static __device__ __forceinline__ float dot(const Block& b, const int* xi, float dq) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = __dp4a(b.lo[j], xi[j], sumi);
            sumi = __dp4a(b.hi[j], xi[j + 4], sumi);
        }
        return b.sc * dq * static_cast<float>(sumi);
    }
};

template <typename QT, bool ADD, int kNw>
__global__ void __launch_bounds__(128) gemv_dp4a_rows_kernel(const uint8_t* __restrict__ W,
                                                             const block_q8_1* __restrict__ q8_1,
                                                             const float* __restrict__ d8, half* y, int N,
                                                             int K, int rows) {
    static_assert(!QT::kNeedsQ8Sum, "multi-row GEMV passes no q8 sum");
    constexpr int kWarps = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row0 = static_cast<int>(blockIdx.x) * kNw;
    const int total_q8 = K / 32;
    const size_t row_bytes = static_cast<size_t>(K / QT::kBlockElems) * QT::kBlockBytes;

    float sum[kNw][kRowsMax] = {};
    for (int b = (warp_id * 32) + lane; b < total_q8; b += kWarps * 32) {
        int xi[kRowsMax][8];
        float dq[kRowsMax];
#pragma unroll
        for (int r = 0; r < kRowsMax; ++r) {
            if (r < rows) {
                const size_t a = (static_cast<size_t>(r) * total_q8) + b;
                load_q8_1_qs(xi[r], q8_1 + a);
                dq[r] = d8[a];
            }
        }
        const size_t boff = static_cast<size_t>(b / QT::kQ8PerWeight) * QT::kBlockBytes;
        const int sub = b % QT::kQ8PerWeight;
#pragma unroll
        for (int w = 0; w < kNw; ++w) {
            if (row0 + w >= N)
                break;
            const uint8_t* bp = W + (static_cast<size_t>(row0 + w) * row_bytes) + boff;
            if constexpr (RowsOnce<QT>::kOn) {
                const auto blk = RowsOnce<QT>::load(bp, sub);
#pragma unroll
                for (int r = 0; r < kRowsMax; ++r)
                    if (r < rows)
                        sum[w][r] += RowsOnce<QT>::dot(blk, xi[r], dq[r]);
            } else {
#pragma unroll
                for (int r = 0; r < kRowsMax; ++r)
                    if (r < rows)
                        sum[w][r] += QT::dp4a_block(bp, sub, xi[r], dq[r], 0.0f);
            }
        }
    }
    __shared__ float partial[kWarps][kNw][kRowsMax];
#pragma unroll
    for (int w = 0; w < kNw; ++w) {
#pragma unroll
        for (int r = 0; r < kRowsMax; ++r) {
            float s = sum[w][r];
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xFFFFFFFFu, s, off);
            if (lane == 0)
                partial[warp_id][w][r] = s;
        }
    }
    __syncthreads();
    const int w = static_cast<int>(threadIdx.x) / kRowsMax;
    const int r = static_cast<int>(threadIdx.x) % kRowsMax;
    if (w < kNw && r < rows && row0 + w < N) {
        float total = partial[0][w][r] + partial[1][w][r] + partial[2][w][r] + partial[3][w][r];
        half* o = y + (static_cast<size_t>(r) * N) + row0 + w;
        if constexpr (ADD)
            total += __half2float(*o);
        *o = __float2half(total);
    }
}

template <typename QT, bool ADD>
void launch_nw(const uint8_t* w, const block_q8_1* q8_1, const float* d8, half* y, int N, int K, int rows,
               cudaStream_t stream) {
    const bool wide = K >= 8192;
    auto* k = wide ? gemv_dp4a_rows_kernel<QT, ADD, 4> : gemv_dp4a_rows_kernel<QT, ADD, 1>;
    k<<<wide ? (N + 3) / 4 : N, 128, 0, stream>>>(w, q8_1, d8, y, N, K, rows);
    IMP_CUDA_CHECK_LAUNCH();
}

template <typename QT>
bool launch_rows(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int N, int K, int rows,
                 bool add, cudaStream_t stream) {
    if (K % QT::kBlockElems != 0)
        return false;
    const auto* w = static_cast<const uint8_t*>(W);
    if (add)
        launch_nw<QT, true>(w, q8_1, d8, y, N, K, rows, stream);
    else
        launch_nw<QT, false>(w, q8_1, d8, y, N, K, rows, stream);
    return true;
}

}  // namespace

bool gemv_dp4a_rows(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int N,
                    int K, int rows, bool add, cudaStream_t stream) {
    if (rows < 1 || rows > kRowsMax || N < 1)
        return false;
    switch (qtype) {
        case QType::Q8_0:
            return launch_rows<Q8_0_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q4_0:
            return launch_rows<Q4_0_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q2_K:
            return launch_rows<Q2_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q3_K:
            return launch_rows<Q3_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q4_K:
            return launch_rows<Q4_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q5_K:
            return launch_rows<Q5_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q6_K:
            return launch_rows<Q6_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::IQ4_NL:
            return launch_rows<IQ4_NL_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::IQ4_XS:
            return launch_rows<IQ4_XS_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        default:
            return false;
    }
}

}  // namespace imp

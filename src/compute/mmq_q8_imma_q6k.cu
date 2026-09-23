// Q6_K RAW-read IMMA prefill kernel (sm_120a). Split out of mmq_q8_imma.cu
// (recompile-blast-radius gate); shared tile constants/cp.async/Q6 staging strides/
// q6k_smem_bytes() in mmq_q8_imma_internal.cuh; kernel launched from mmq_q8_imma.cu dispatch,
// which also keeps the activation quantizer.

#include "compute/mmq_q8_imma_internal.cuh"

#include <cstring>

namespace imp {

namespace {

// Q6_K RAW-read kernel (per-16 scales, symmetric, no beta/rowsum term).
// 210-B super-blocks are only 2-aligned (no cp.async): B is read in place through registers.
// Per-16 scale vs k32 MMA: HALF-MMA SPLIT, issue m16n8k32 twice as (b0,0) and (0,b1);
// zeroed operand yields the 16-wide partial sum with its own alpha=d*sc16 (same int-op
// count as a k16 MMA pair).
// Quad mapping (dequant_gpu.cu Q6_K header): K-step ks covers sub-block pair j=(2ks)%8,
// j+1 -> group g=j>>2, quads (j%4, j%4+1); pair shares ql bytes [g*64..+63] (quad&1
// selects 32-B half, quad>=2 the nibble) and qh bytes [g*32..+31] (shift quad*2).

// Activation tile + per-row activation scales of one K-step (cp.async); B is staged by the
// register path below.
template <int BM>
__device__ __forceinline__ void load_kstep_q6k_a(int tid, const int8_t* __restrict__ A,
                                                 const __half* __restrict__ Asc, int8_t (*sA)[kRow],
                                                 __half (*sAsc)[2], int base_m, int M, int K, int subs,
                                                 int k_base) {
#pragma unroll
    for (int i = tid; i < BM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col],
                       A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
}

// Register staging of the GGUF 210-B blocks (2-aligned, so no cp.async): per (row, part) task,
// part 0-3 = 16-B ql chunks, 4-5 = 16-B qh chunks, 6 = the 4 per-16 scales + d. Two halves of the
// tasks are in flight at a time (one per kb step of the MMA loop): all four would hold 24 registers
// across it and spill BM=128 (255 regs + 40 B stack).
constexpr int kQ6Gguf = 210;
constexpr int kUnpParts = 7;
constexpr int kUnpTasks = (kBN * kUnpParts + kThreads - 1) / kThreads;
constexpr int kUnpHalf = kUnpTasks / 2;
static_assert(kUnpTasks == 2 * kUnpHalf, "unpacked B tasks split into two halves");

struct Q6kUnpackedB {
    uint32_t w[kUnpHalf][5];
    uint32_t shifts;  // byte shift (0 or 16) of task i at bits [8i, 8i+8)
};

// Offsets are multiples of 4, so the shift is the 210-B block's 4-B misalignment (0 or 2 bytes).
template <int T0>
__device__ __forceinline__ void fetch_b_q6k_unpacked(Q6kUnpackedB& rb, int tid, const uint8_t* __restrict__ W,
                                                     int base_n, int sblk_count, int ks, int base_n_rows) {
    const int sblk = ks >> 2;
    const int j = (2 * ks) & 7;
    const int g = j >> 2;
    rb.shifts = 0;
#pragma unroll
    for (int i = 0; i < kUnpHalf; ++i) {
        const int t = tid + (T0 + i) * kThreads;
        const int row = t / kUnpParts;
        const int part = t % kUnpParts;
        const bool valid = t < kBN * kUnpParts && ((base_n_rows < 0) || (row < base_n_rows));
        const uint8_t* blk = W + (static_cast<size_t>(base_n + row) * sblk_count + sblk) * kQ6Gguf;
        const int off = part < 4   ? g * 64 + part * 16
                        : part < 6 ? 128 + g * 32 + (part - 4) * 16
                                   : 192 + 2 * j;
        const uintptr_t addr = reinterpret_cast<uintptr_t>(blk) + off;
        rb.shifts |= (static_cast<uint32_t>(addr & 3u) * 8u) << (8 * i);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(addr & ~static_cast<uintptr_t>(3));
        const int nw = part < 6 ? 5 : 2;
#pragma unroll
        for (int k = 0; k < 5; ++k)
            rb.w[i][k] = (valid && k < nw) ? __ldg(wp + k) : 0u;
        if (part == 6 && valid)  // d: 2-B load, an aligned word would read past the last block
            rb.w[i][2] = __ldg(reinterpret_cast<const unsigned short*>(blk + 208));
    }
}

template <int T0>
__device__ __forceinline__ void commit_b_q6k_unpacked(const Q6kUnpackedB& rb, int tid, uint8_t (*sQl)[kQlRow],
                                                      uint8_t (*sQh)[kQhRow], uint8_t (*sScd)[8]) {
#pragma unroll
    for (int i = 0; i < kUnpHalf; ++i) {
        const int t = tid + (T0 + i) * kThreads;
        if (t >= kBN * kUnpParts)
            break;
        const int row = t / kUnpParts;
        const int part = t % kUnpParts;
        const uint32_t s = (rb.shifts >> (8 * i)) & 0xFFu;
        if (part < 6) {
            uint4 v;
            v.x = __funnelshift_r(rb.w[i][0], rb.w[i][1], s);
            v.y = __funnelshift_r(rb.w[i][1], rb.w[i][2], s);
            v.z = __funnelshift_r(rb.w[i][2], rb.w[i][3], s);
            v.w = __funnelshift_r(rb.w[i][3], rb.w[i][4], s);
            uint8_t* dst = part < 4 ? &sQl[row][part * 16] : &sQh[row][(part - 4) * 16];
            *reinterpret_cast<uint4*>(dst) = v;
        } else {
            *reinterpret_cast<uint32_t*>(&sScd[row][0]) = __funnelshift_r(rb.w[i][0], rb.w[i][1], s);
            *reinterpret_cast<uint32_t*>(&sScd[row][4]) = rb.w[i][2];  // d + 2 zero pad bytes
        }
    }
}

}  // namespace

template <int BM, bool BETA1>
__global__ void __launch_bounds__(kThreads) mmq_imma_q6k_raw_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const uint8_t* __restrict__ Wq6k,
    __half* __restrict__ out, int M, int N, int K, const int32_t* __restrict__ expert_offsets,
    size_t w_stride_blocks) {
    constexpr int kWM = (BM == 128) ? 4 : 2;
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;
    constexpr int kTileN = kBN / kWN;
    constexpr int kMF = kTileM / 16;
    constexpr int kNF = kTileN / 8;

    int rows = M;
    size_t row_off = 0;
    const int e = blockIdx.z;
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        const int32_t o1 = __ldg(&expert_offsets[e + 1]);
        row_off = static_cast<size_t>(o0);
        rows = o1 - o0;
    }
    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= rows || rows == 0) return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);

    const int subs = K / 32;
    const int sblk_count = K / 256;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const uint8_t* W = Wq6k + static_cast<size_t>(e) * w_stride_blocks * kQ6Gguf;
    __half* C = out + row_off * N;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;

    // dynamic smem (BM=128 + Q6 staging exceeds the 48-KB static limit);
    // offsets are COMPILE-TIME constants — no runtime pointer arrays (the
    // 3-stage experiment's local-memory spill trap).
    extern __shared__ uint8_t smem6[];
    constexpr size_t kAOff = 0;
    constexpr size_t kQlOff = kAOff + static_cast<size_t>(kStages) * BM * kRow;
    constexpr size_t kQhOff = kQlOff + static_cast<size_t>(kStages) * kBN * kQlRow;
    constexpr size_t kScdOff = kQhOff + static_cast<size_t>(kStages) * kBN * kQhRow;
    constexpr size_t kAscOff = kScdOff + static_cast<size_t>(kStages) * kBN * 8;
    typedef int8_t ARow[kRow];
    typedef uint8_t QlRow[kQlRow];
    typedef uint8_t QhRow[kQhRow];
    typedef uint8_t ScdRow[8];
    typedef __half AscRow[2];
    auto sA = [&](int st) { return reinterpret_cast<ARow*>(smem6 + kAOff + static_cast<size_t>(st) * BM * kRow); };
    auto sQl = [&](int st) { return reinterpret_cast<QlRow*>(smem6 + kQlOff + static_cast<size_t>(st) * kBN * kQlRow); };
    auto sQh = [&](int st) { return reinterpret_cast<QhRow*>(smem6 + kQhOff + static_cast<size_t>(st) * kBN * kQhRow); };
    auto sScd = [&](int st) { return reinterpret_cast<ScdRow*>(smem6 + kScdOff + static_cast<size_t>(st) * kBN * 8); };
    auto sAsc = [&](int st) { return reinterpret_cast<AscRow*>(smem6 + kAscOff + static_cast<size_t>(st) * BM * 4); };

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j2 = 0; j2 < kNF; ++j2)
            acc[i][j2][0] = acc[i][j2][1] = acc[i][j2][2] = acc[i][j2][3] = 0.0f;

    const int ksteps = K / kBK;
    load_kstep_q6k_a<BM>(tid, A, Asc, sA(0), sAsc(0), base_m, rows, K, subs, 0);
    cp_async_commit();
    Q6kUnpackedB rb;
    fetch_b_q6k_unpacked<0>(rb, tid, W, base_n, sblk_count, 0, n_rem);
    commit_b_q6k_unpacked<0>(rb, tid, sQl(0), sQh(0), sScd(0));
    fetch_b_q6k_unpacked<kUnpHalf>(rb, tid, W, base_n, sblk_count, 0, n_rem);
    commit_b_q6k_unpacked<kUnpHalf>(rb, tid, sQl(0), sQh(0), sScd(0));

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep_q6k_a<BM>(tid, A, Asc, sA(nstage), sAsc(nstage), base_m, rows, K, subs,
                                 (ks + 1) * kBK);
            fetch_b_q6k_unpacked<0>(rb, tid, W, base_n, sblk_count, ks + 1, n_rem);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const int j = (2 * ks) & 7;
        const int quad_base = j & 3;  // quads (quad_base, quad_base+1)

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kc = kb * 32;
            const int quad = quad_base + kb;
            const int ql_half = (quad & 1) * 32;
            const uint32_t nib_shift = (quad >= 2) ? 4u : 0u;
            const uint32_t qh_shift = static_cast<uint32_t>(quad * 2);
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = kc + cl * 4;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA(stage)[arow_lo][acol]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA(stage)[arow_hi][acol]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA(stage)[arow_lo][acol + 16]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA(stage)[arow_hi][acol + 16]);
                const float da_lo = __half2float(sAsc(stage)[arow_lo][kb]);
                const float da_hi = __half2float(sAsc(stage)[arow_hi][kb]);

#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    const int bcol = warp_n * kTileN + nf * 8 + rl;
                    const uint32_t ql0 =
                        *reinterpret_cast<const uint32_t*>(&sQl(stage)[bcol][ql_half + cl * 4]);
                    const uint32_t ql1 = *reinterpret_cast<const uint32_t*>(
                        &sQl(stage)[bcol][ql_half + cl * 4 + 16]);
                    const uint32_t qh0 =
                        *reinterpret_cast<const uint32_t*>(&sQh(stage)[bcol][cl * 4]);
                    const uint32_t qh1 =
                        *reinterpret_cast<const uint32_t*>(&sQh(stage)[bcol][cl * 4 + 16]);
                    const uint32_t b0 = __vsub4(((ql0 >> nib_shift) & 0x0F0F0F0Fu) |
                                                    (((qh0 >> qh_shift) & 0x03030303u) << 4),
                                                0x20202020u);
                    const uint32_t b1 = __vsub4(((ql1 >> nib_shift) & 0x0F0F0F0Fu) |
                                                    (((qh1 >> qh_shift) & 0x03030303u) << 4),
                                                0x20202020u);

                    // HALF-MMA SPLIT: per-16 scales need the two 16-wide
                    // partial sums separately — zero the other B register.
                    int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                    int32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
#if __CUDA_ARCH__ >= 800
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0), "r"(0), "r"(0),
                          "r"(0), "r"(0));
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(q0), "=r"(q1), "=r"(q2), "=r"(q3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1), "r"(0), "r"(0),
                          "r"(0), "r"(0));
#endif
                    const int ncol_lo = warp_n * kTileN + nf * 8 + cl * 2;
                    // per-16 α from the staged scales: bytes [2*kb], [2*kb+1]; d at [4]
                    const int8_t* sc_lo = reinterpret_cast<const int8_t*>(&sScd(stage)[ncol_lo][0]);
                    const int8_t* sc_hi =
                        reinterpret_cast<const int8_t*>(&sScd(stage)[ncol_lo + 1][0]);
                    __half d_lo_h, d_hi_h;
                    memcpy(&d_lo_h, &sScd(stage)[ncol_lo][4], 2);
                    memcpy(&d_hi_h, &sScd(stage)[ncol_lo + 1][4], 2);
                    const float dlo = __half2float(d_lo_h);
                    const float dhi = __half2float(d_hi_h);
                    const float al1 = dlo * static_cast<float>(sc_lo[2 * kb]);
                    const float al2 = dlo * static_cast<float>(sc_lo[2 * kb + 1]);
                    const float ah1 = dhi * static_cast<float>(sc_hi[2 * kb]);
                    const float ah2 = dhi * static_cast<float>(sc_hi[2 * kb + 1]);
                    acc[mf][nf][0] += da_lo * fmaf(al1, static_cast<float>(p0),
                                                   al2 * static_cast<float>(q0));
                    acc[mf][nf][1] += da_lo * fmaf(ah1, static_cast<float>(p1),
                                                   ah2 * static_cast<float>(q1));
                    acc[mf][nf][2] += da_hi * fmaf(al1, static_cast<float>(p2),
                                                   al2 * static_cast<float>(q2));
                    acc[mf][nf][3] += da_hi * fmaf(ah1, static_cast<float>(p3),
                                                   ah2 * static_cast<float>(q3));
                }
            }
            // nstage was last read in iteration ks-1, whose trailing barrier orders these stores after it.
            if (ks + 1 < ksteps) {
                const int ns = (ks + 1) & 1;
                if (kb == 0) {
                    commit_b_q6k_unpacked<0>(rb, tid, sQl(ns), sQh(ns), sScd(ns));
                    fetch_b_q6k_unpacked<kUnpHalf>(rb, tid, W, base_n, sblk_count, ks + 1, n_rem);
                } else {
                    commit_b_q6k_unpacked<kUnpHalf>(rb, tid, sQl(ns), sQh(ns), sScd(ns));
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
            if (col + 1 >= N) continue;  // N-tail: OOB columns are never stored
            if (row_lo < rows) {
                __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_lo) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][0], acc[mf][nf][1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
            if (row_hi < rows) {
                __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_hi) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][2], acc[mf][nf][3]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

// Explicit instantiations launched by the dispatch in mmq_q8_imma.cu.
template __global__ void mmq_imma_q6k_raw_kernel<32, false>(const int8_t*, const __half*, const uint8_t*,
                                                            __half*, int, int, int, const int32_t*, size_t);
template __global__ void mmq_imma_q6k_raw_kernel<128, true>(const int8_t*, const __half*, const uint8_t*,
                                                            __half*, int, int, int, const int32_t*, size_t);
template __global__ void mmq_imma_q6k_raw_kernel<128, false>(const int8_t*, const __half*, const uint8_t*,
                                                             __half*, int, int, int, const int32_t*, size_t);

}  // namespace imp

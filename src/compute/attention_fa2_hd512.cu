// FA2 prefill for head_dim 512 (Gemma-4 global layers, ROADMAP row 123): causal, fp16 Q/K/V/O,
// fp32 scores and output accumulator. A full 16x512 fp32 output tile is 256 registers per lane, so
// the D axis is split: 8 warps = 4 row groups x 2 D halves. Both warps of a row group compute the
// partial scores over their D half, swap them through shared memory (double-buffered, one named
// barrier per KV tile), run the identical online softmax and accumulate P V over their D half.
// Q fragments stay in registers (64), O in registers (128). KV tiles of 16 rows, 2-stage cp.async.

#include "compute/attention_fa2_hd512.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

constexpr int kHD = 512;
constexpr int kHalf = 256;  // D columns per warp
constexpr int kBq = 64;     // query rows per CTA
constexpr int kBkv = 16;    // KV rows per tile
constexpr int kThreads = 256;
constexpr int kStages = 2;
constexpr int kRowH = kHD + 8;  // K/V smem row stride in halves (1040 B: 8 ldmatrix rows on 32 banks)
constexpr int kTileH = kBkv * kRowH;
constexpr int kStageH = 2 * kTileH;  // K then V
constexpr int kSlotF = 32 * 8;       // one warp's 16x16 fp32 partial scores, lane-major
constexpr int kSmemBytes = (kStages * kStageH * 2) + (4 * 2 * 2 * kSlotF * 4);

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldsm_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

struct Args {
    const half* q;
    const half* k;
    const half* v;
    half* o;
    int n, kv_len, nh, nkv, q_offset;
    float scale, softcap;
};

// One KV tile (K then V, 16 rows each) into a stage: 16 B chunks over the 256 threads, rows past
// kv_len zero-filled.
__device__ __forceinline__ void load_kv(uint32_t s_stage, const Args& a, int kvh, int kv0) {
    constexpr int kChunks = kHD / 8;  // per row
    for (int i = static_cast<int>(threadIdx.x); i < 2 * kBkv * kChunks; i += kThreads) {
        const bool is_v = i >= kBkv * kChunks;
        const int j = is_v ? i - (kBkv * kChunks) : i;
        const int r = j / kChunks;
        const int c = j % kChunks;
        const int kv = kv0 + r;
        const bool ok = kv < a.kv_len;
        const half* src = (is_v ? a.v : a.k) + ((static_cast<size_t>(ok ? kv : 0) * a.nkv + kvh) * kHD) +
                          (static_cast<size_t>(c) * 8);
        cp_async16(s_stage + 2 * ((is_v ? kTileH : 0) + (r * kRowH) + (c * 8)), src, ok ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;");
}

struct Lane {
    int rg, dh, g, t, lr, lm, lane, h, kvh, row_a;
    size_t ld;  // Q / O row stride in halves (nh * 512)
};

// Q fragments of this warp's 16 rows x 256 D: 16 k16 steps x 4 regs (rows past n read 0).
__device__ __forceinline__ void load_q(uint32_t (&qf)[16][4], const Args& a, const Lane& L) {
#pragma unroll
    for (int ks = 0; ks < 16; ++ks)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = L.row_a + ((i & 1) * 8);
            const int c = (L.dh * kHalf) + (ks * 16) + (2 * L.t) + ((i >> 1) * 8);
            const size_t off = (static_cast<size_t>(r) * L.ld) + (static_cast<size_t>(L.h) * kHD) + c;
            qf[ks][i] = r < a.n ? *reinterpret_cast<const uint32_t*>(a.q + off) : 0u;
        }
}

// Partial scores over this warp's D half (16 rows x 16 kv), swapped with the partner warp.
__device__ __forceinline__ void tile_scores(float (&s)[2][4], const uint32_t (&qf)[16][4], uint32_t sk,
                                            float* slots, int j, const Lane& L) {
#pragma unroll
    for (int nf = 0; nf < 2; ++nf)
        s[nf][0] = s[nf][1] = s[nf][2] = s[nf][3] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        uint32_t b[4];  // {nf0 b0, nf0 b1, nf1 b0, nf1 b1}
        const int kv = L.lr + ((L.lm >> 1) * 8);
        const int d = (L.dh * kHalf) + (ks * 16) + ((L.lm & 1) * 8);
        ldsm_x4(b, sk + 2 * ((kv * kRowH) + d));
        mma16816(s[0], qf[ks], b[0], b[1]);
        mma16816(s[1], qf[ks], b[2], b[3]);
    }
    const int buf = j & 1;
    const size_t mine_off = (static_cast<size_t>(((L.rg * 2) + L.dh) * 2 + buf) * kSlotF) +
                            (static_cast<size_t>(L.lane) * 8);
    const size_t other_off = (static_cast<size_t>(((L.rg * 2) + (L.dh ^ 1)) * 2 + buf) * kSlotF) +
                             (static_cast<size_t>(L.lane) * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        slots[mine_off + i] = s[i >> 2][i & 3];
    asm volatile("bar.sync %0, 64;" ::"r"(1 + L.rg));
#pragma unroll
    for (int i = 0; i < 8; ++i)
        s[i >> 2][i & 3] += slots[other_off + i];
}

// Scale, softcap, causal / length mask, online softmax; s becomes P, o is rescaled.
__device__ __forceinline__ void softmax_step(float (&s)[2][4], float (&o)[32][4], float (&m_r)[2],
                                             float (&l_r)[2], const Args& a, int j, const Lane& L) {
    const float sl2 = a.scale * 1.4426950408889634f;  // exp2 domain
    float mx[2] = {m_r[0], m_r[1]};
#pragma unroll
    for (int nf = 0; nf < 2; ++nf)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = L.row_a + ((i >> 1) * 8);
            const int kv = (j * kBkv) + (nf * 8) + (2 * L.t) + (i & 1);
            float x = s[nf][i];
            if (a.softcap > 0.0f)
                x = a.softcap * tanhf((x * a.scale) / a.softcap) / a.scale;
            x *= sl2;
            if (kv > row + a.q_offset || kv >= a.kv_len)
                x = -INFINITY;
            s[nf][i] = x;
            mx[i >> 1] = fmaxf(mx[i >> 1], x);
        }
    float corr[2], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        corr[r] = mx[r] == -INFINITY ? 1.0f : exp2f(m_r[r] - mx[r]);
    }
#pragma unroll
    for (int nf = 0; nf < 2; ++nf)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = mx[i >> 1] == -INFINITY ? 0.0f : exp2f(s[nf][i] - mx[i >> 1]);
            s[nf][i] = p;
            rs[i >> 1] += p;
        }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
        rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
        l_r[r] = (l_r[r] * corr[r]) + rs[r];
        m_r[r] = mx[r];
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i][0] *= corr[0];
        o[i][1] *= corr[0];
        o[i][2] *= corr[1];
        o[i][3] *= corr[1];
    }
}

// O[:, D half] += P V: P (16 x 16 kv) as the A fragment (the two score n-frags are its k halves);
// V rows are kv (k), columns d (n), ldmatrix.trans gives the col operand.
__device__ __forceinline__ void tile_pv(float (&o)[32][4], const float (&s)[2][4], uint32_t sv,
                                        const Lane& L) {
    const uint32_t pa[4] = {pack_h2(s[0][0], s[0][1]), pack_h2(s[0][2], s[0][3]), pack_h2(s[1][0], s[1][1]),
                            pack_h2(s[1][2], s[1][3])};
#pragma unroll
    for (int dp = 0; dp < 32; dp += 2) {  // two n8 frags (16 D columns) per step
        uint32_t b[4];                    // {d0-7 b0, d0-7 b1, d8-15 b0, d8-15 b1}
        const int kv = L.lr + ((L.lm & 1) * 8);
        const int d = (L.dh * kHalf) + (dp * 8) + ((L.lm >> 1) * 8);
        ldsm_x4_t(b, sv + 2 * ((kv * kRowH) + d));
        mma16816(o[dp], pa, b[0], b[1]);
        mma16816(o[dp + 1], pa, b[2], b[3]);
    }
}

// O / l, fp16 stores of this warp's 16 rows x 256 D.
__device__ __forceinline__ void store_o(const float (&o)[32][4], const float (&l_r)[2], const Args& a,
                                        const Lane& L) {
    const float inv[2] = {l_r[0] > 0.0f ? 1.0f / l_r[0] : 0.0f, l_r[1] > 0.0f ? 1.0f / l_r[1] : 0.0f};
#pragma unroll
    for (int nf = 0; nf < 32; ++nf)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = L.row_a + (r * 8);
            if (row >= a.n)
                continue;
            const int col = (L.dh * kHalf) + (nf * 8) + (2 * L.t);
            const size_t off = (static_cast<size_t>(row) * L.ld) + (static_cast<size_t>(L.h) * kHD) + col;
            *reinterpret_cast<__half2*>(a.o + off) = __floats2half2_rn(o[nf][r ? 2 : 0] * inv[r],
                                                                       o[nf][r ? 3 : 1] * inv[r]);
        }
}

__global__ void __launch_bounds__(kThreads, 1) fa2_hd512_kernel(Args a) {
    extern __shared__ __align__(128) uint8_t smem[];
    const uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    float* slots = reinterpret_cast<float*>(smem + (static_cast<size_t>(kStages) * kStageH * 2));
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    Lane L{};
    L.lane = static_cast<int>(threadIdx.x) & 31;
    L.rg = warp & 3;   // row group: rows rg*16 .. +16
    L.dh = warp >> 2;  // D half
    L.g = L.lane >> 2;
    L.t = L.lane & 3;
    L.lr = L.lane & 7;
    L.lm = L.lane >> 3;
    L.h = static_cast<int>(blockIdx.y);
    L.kvh = L.h / (a.nh / a.nkv);
    const int q0 = static_cast<int>(blockIdx.x) * kBq;
    L.row_a = q0 + (L.rg * 16) + L.g;  // this lane's rows: row_a, row_a + 8
    L.ld = static_cast<size_t>(a.nh) * kHD;

    uint32_t qf[16][4];
    load_q(qf, a, L);
    float o[32][4] = {};
    float m_r[2] = {-INFINITY, -INFINITY};
    float l_r[2] = {0.0f, 0.0f};
    const int last_q = min(q0 + kBq, a.n) - 1 + a.q_offset;  // causal bound of this CTA
    const int n_tiles = min((last_q / kBkv) + 1, (a.kv_len + kBkv - 1) / kBkv);

    load_kv(s0, a, L.kvh, 0);
    for (int j = 0; j < n_tiles; ++j) {
        if (j + 1 < n_tiles)
            load_kv(s0 + ((j + 1) % kStages) * kStageH * 2, a, L.kvh, (j + 1) * kBkv);
        else
            asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        const uint32_t sk = s0 + (j % kStages) * kStageH * 2;
        float s[2][4];
        tile_scores(s, qf, sk, slots, j, L);
        softmax_step(s, o, m_r, l_r, a, j, L);
        tile_pv(o, s, sk + (kTileH * 2), L);
        __syncthreads();  // every warp left this stage before it is refilled
    }
    store_o(o, l_r, a, L);
}

}  // namespace

bool fa2_hd512_prefill(const half* q, const half* k, const half* v, half* o, int n, int kv_len, int nh,
                       int nkv, float scale, float softcap, int q_offset, cudaStream_t stream) {
    if (n <= 0 || kv_len <= 0 || nh <= 0 || nkv <= 0 || nh % nkv != 0 || q == nullptr || k == nullptr ||
        v == nullptr || o == nullptr)
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(fa2_hd512_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const Args a{q, k, v, o, n, kv_len, nh, nkv, q_offset, scale, softcap};
    fa2_hd512_kernel<<<dim3((n + kBq - 1) / kBq, nh), kThreads, kSmemBytes, stream>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp

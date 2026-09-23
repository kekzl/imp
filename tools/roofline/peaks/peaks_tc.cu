// Tensor-core peaks per format: dense mma.sync throughput, kChains independent accumulators
// per warp, best of several warps-per-SM shapes. Operand values are fixed register noise.
#include "peaks_common.cuh"

#include <cstdint>

namespace peaks {
namespace {

enum Fmt {
    NVFP4,     // kind::mxf4nvf4 scale_vec::4X ue4m3, m16n8k64
    MXFP4,     // kind::mxf4 scale_vec::2X ue8m0, m16n8k64
    MXFP8,     // kind::mxf8f6f4 scale_vec::1X ue8m0 e4m3, m16n8k32
    FP8_F32,   // kind::f8f6f4 e4m3, f32 acc, m16n8k32
    FP8_F16,   // e4m3, f16 acc, m16n8k32
    BF16_F32,  // m16n8k16
    FP16_F32,  // m16n8k16
    FP16_F16,  // m16n8k16
    TF32_F32,  // m16n8k8
    INT8_S32,  // m16n8k32
};

constexpr int kChains = 4;

template <Fmt F>
__device__ __forceinline__ void mma(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    const uint32_t sf = 0x7f7f7f7fu;  // ue8m0 1.0; ue4m3 0x38 not needed for throughput
    constexpr uint16_t z = 0;
    if constexpr (F == NVFP4) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
            "e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
            "{%10}, {%11,%12};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x38383838u), "h"(z),
              "h"(z));
    } else if constexpr (F == MXFP4) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1."
            "f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
            "{%10}, {%11,%12};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(z), "h"(z));
    } else if constexpr (F == MXFP8) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
            "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
            "{%10}, {%11,%12};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "h"(z), "h"(z));
    } else if constexpr (F == FP8_F32) {
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == FP8_F16) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
            "{%0,%1};\n"
            : "+r"(d[0]), "+r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == BF16_F32) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == FP16_F32) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == FP16_F16) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
            "{%0,%1};\n"
            : "+r"(d[0]), "+r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == TF32_F32) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == INT8_S32) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

template <Fmt F>
__global__ void tc_kernel(int iters, uint32_t* sink) {
    const uint32_t t = threadIdx.x;
    // Small magnitudes (sign/exponent bits low) keep float accumulators finite.
    const uint32_t a[4] = {(t * 37u + 1u) & 0x03030303u, (t * 41u + 2u) & 0x03030303u,
                           (t * 43u + 3u) & 0x03030303u, (t * 47u + 4u) & 0x03030303u};
    const uint32_t b[2] = {(t * 53u + 5u) & 0x03030303u, (t * 59u + 6u) & 0x03030303u};
    uint32_t d[kChains][4] = {};
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c)
            mma<F>(d[c], a, b);
    }
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < kChains; ++c)
        s ^= d[c][0] ^ d[c][1] ^ d[c][2] ^ d[c][3];
    if (s == 0x9e3779b9u)
        sink[0] = s;
}

struct FmtInfo {
    Fmt fmt;
    const char* name;
    int k;  // MMA K; ops per mma = 2 * 16 * 8 * k
};

constexpr FmtInfo kFmts[] = {
    {NVFP4, "nvfp4 mxf4nvf4 ue4m3 4X f32", 64},
    {MXFP4, "mxfp4 mxf4 ue8m0 2X f32", 64},
    {MXFP8, "mxfp8 mxf8f6f4 ue8m0 1X f32", 32},
    {FP8_F32, "fp8 e4m3 f32acc", 32},
    {FP8_F16, "fp8 e4m3 f16acc", 32},
    {BF16_F32, "bf16 f32acc", 16},
    {FP16_F32, "fp16 f32acc", 16},
    {FP16_F16, "fp16 f16acc", 16},
    {TF32_F32, "tf32 f32acc", 8},
    {INT8_S32, "int8 s32acc", 32},
};

template <Fmt F>
void run_one(const FmtInfo& info, uint32_t* sink) {
    constexpr int kIters = 1 << 12;
    double best_med = 0.0;
    std::vector<double> best;
    std::string note;
    for (int warps : {4, 8, 16}) {
        for (int bps : {1, 2}) {
            const int grid = bps * sm_count();
            auto ms = time_launches([&] { tc_kernel<F><<<grid, 32 * warps>>>(kIters, sink); }, 5, 5,
                                    best.empty() ? 1.5 : 0.3);
            const double ops = double(grid) * warps * kChains * kIters * 2.0 * 16 * 8 * info.k;
            std::vector<double> s = ms;
            std::sort(s.begin(), s.end());
            if (ops / s[s.size() / 2] > best_med) {
                best_med = ops / s[s.size() / 2];
                best.clear();
                for (double t : ms)
                    best.push_back(t / ops);  // ms per op, rescaled below
                note = "[" + std::to_string(bps) + "x" + std::to_string(warps) + "w/SM, " +
                       std::to_string(kChains) + " chains]";
            }
        }
    }
    // record() computes work / (t * 1e-3); with t = ms per op this is ops/s; scale to T.
    record("tc", info.name, "TOPS", 1e-12, best, note);
}

template <int I>
void run_all(uint32_t* sink) {
    if constexpr (I < static_cast<int>(sizeof(kFmts) / sizeof(kFmts[0]))) {
        run_one<kFmts[I].fmt>(kFmts[I], sink);
        run_all<I + 1>(sink);
    }
}

}  // namespace

void run_tc() {
    uint32_t* sink = nullptr;
    PK_CHECK(cudaMalloc(&sink, 64));
    run_all<0>(sink);
    PK_CHECK(cudaFree(sink));
}

}  // namespace peaks

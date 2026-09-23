// SIMT peaks: FFMA, HFMA2 (fp16x2, bf16x2), SFU ex2/rsqrt/tanh. kChains independent chains
// per thread, 1024 threads per SM x 2 blocks. Rates count 2 flops per FMA lane, 1 op per SFU lane.
#include "peaks_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace peaks {
namespace {

constexpr int kChains = 8;
constexpr int kIters = 1 << 13;

enum Op { FFMA, HFMA2, BFMA2, EX2, RSQRT, TANH };

template <Op O>
__global__ void simt_kernel(float seed, uint32_t* sink) {
    float f[kChains];
    uint32_t h[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
        f[c] = seed + 1e-3f * (threadIdx.x + c);
        h[c] = 0x3c003c00u + c;
    }
    const float m = 0.999f, a = 1e-4f;
    const uint32_t hm = 0x3bff3bffu, ha = 0x00010001u;
#pragma unroll 4
    for (int i = 0; i < kIters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            if constexpr (O == FFMA) {
                f[c] = fmaf(f[c], m, a);
            } else if constexpr (O == HFMA2) {
                asm volatile("fma.rn.f16x2 %0, %0, %1, %2;\n" : "+r"(h[c]) : "r"(hm), "r"(ha));
            } else if constexpr (O == BFMA2) {
                asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;\n" : "+r"(h[c]) : "r"(0x3f7f3f7fu), "r"(ha));
            } else if constexpr (O == EX2) {
                asm volatile("ex2.approx.ftz.f32 %0, %0;\n" : "+f"(f[c]));
            } else if constexpr (O == RSQRT) {
                asm volatile("rsqrt.approx.ftz.f32 %0, %0;\n" : "+f"(f[c]));
            } else if constexpr (O == TANH) {
                asm volatile("tanh.approx.f32 %0, %0;\n" : "+f"(f[c]));
            }
        }
    }
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < kChains; ++c)
        s ^= __float_as_uint(f[c]) ^ h[c];
    if (s == 0x9e3779b9u)
        sink[0] = s;
}

template <Op O>
void run_one(const char* name, double ops_per_lane_op, const char* unit, uint32_t* sink) {
    const int grid = 2 * sm_count(), threads = 1024;
    auto ms = time_launches([&] { simt_kernel<O><<<grid, threads>>>(0.5f, sink); }, 5);
    const double work = double(grid) * threads * kChains * kIters * ops_per_lane_op * 1e-12;
    record("simt", name, unit, work, ms, "[2x1024/SM, 8 chains]");
}

}  // namespace

void run_simt() {
    uint32_t* sink = nullptr;
    PK_CHECK(cudaMalloc(&sink, 64));
    run_one<FFMA>("ffma fp32", 2.0, "TFLOPS", sink);
    run_one<HFMA2>("hfma2 fp16x2", 4.0, "TFLOPS", sink);
    run_one<BFMA2>("hfma2 bf16x2", 4.0, "TFLOPS", sink);
    run_one<EX2>("sfu ex2.approx", 1.0, "Tops", sink);
    run_one<RSQRT>("sfu rsqrt.approx", 1.0, "Tops", sink);
    run_one<TANH>("sfu tanh.approx", 1.0, "Tops", sink);
    PK_CHECK(cudaFree(sink));
}

}  // namespace peaks

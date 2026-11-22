// Hot/cold timing of the M=1 FP16 GEMV (skill kernel-tuning, section 3): the same launch on the
// weight it just read (L2) and rotating over more copies than the 96 MB L2 holds (VRAM).
// GPU-ONLY bench, disabled by default: test-compute --gtest_also_run_disabled_tests
// --gtest_filter='*GemvFp16HotCold*'

#include <gtest/gtest.h>

#include "compute/gemm.h"
#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

__device__ __forceinline__ float acc16(float sum, float4 a0, float4 a1, float4 x0, float4 x1) {
    const half2* a_h2_0 = reinterpret_cast<const half2*>(&a0);
    const half2* x_h2_0 = reinterpret_cast<const half2*>(&x0);
    const half2* a_h2_1 = reinterpret_cast<const half2*>(&a1);
    const half2* x_h2_1 = reinterpret_cast<const half2*>(&x1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        half2 prod = __hmul2(a_h2_0[j], x_h2_0[j]);
        sum += __half2float(prod.x) + __half2float(prod.y);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        half2 prod = __hmul2(a_h2_1[j], x_h2_1[j]);
        sum += __half2float(prod.x) + __half2float(prod.y);
    }
    return sum;
}

// U lane iterations issue their weight loads (HX: and their x loads) before the first use.
// U = 1, HX = false is the shipped loop.
// P: 0 plain launch, 1 PDL with the wait first (the shipped order), 2 PDL with a trigger at entry and
// the first U iterations of weight loads before the wait, 3 PDL with the trigger and the wait first.
template <int U, bool HX, int P = 0>
__global__ void gemv_fp16_hoist_k(const half* __restrict__ A, const half* __restrict__ x,
                                  half* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * warps_per_block) + warp_id;
    if (P >= 2)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (row >= M)
        return;
    if (P == 1 || P == 3)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    bool waited = P != 2;
    const half* A_row = A + ((int64_t)row * K);
    float sum = 0.0f;
    const int K_vec16 = K / 16;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < K_vec16; i += 32 * U) {
        float4 a[2 * U];
        float4 xx[2 * U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = i + 32 * u;
            if (j < K_vec16) {
                a[2 * u] = A_row_v[2 * j];
                a[2 * u + 1] = A_row_v[2 * j + 1];
                if (HX) {
                    xx[2 * u] = x_v[2 * j];
                    xx[2 * u + 1] = x_v[2 * j + 1];
                }
            }
        }
        if (!waited) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
            waited = true;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = i + 32 * u;
            if (j < K_vec16) {
                if (HX)
                    sum = acc16(sum, a[2 * u], a[2 * u + 1], xx[2 * u], xx[2 * u + 1]);
                else
                    sum = acc16(sum, a[2 * u], a[2 * u + 1], x_v[2 * j], x_v[2 * j + 1]);
            }
        }
    }
    if (!waited)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    // Tails in the shipped order: one 8-half vector step, then scalars.
    int base16 = K_vec16 * 16;
    if (base16 + 8 <= K) {
        const int K_vec8_rem = (K - base16) / 8;
        const float4* A_rem = reinterpret_cast<const float4*>(A_row + base16);
        const float4* x_rem = reinterpret_cast<const float4*>(x + base16);
        for (int i = lane; i < K_vec8_rem; i += 32) {
            float4 a_raw = A_rem[i];
            float4 x_raw = x_rem[i];
            const half2* a_h2 = reinterpret_cast<const half2*>(&a_raw);
            const half2* x_h2 = reinterpret_cast<const half2*>(&x_raw);
            for (int j = 0; j < 4; ++j) {
                half2 prod = __hmul2(a_h2[j], x_h2[j]);
                sum += __half2float(prod.x) + __half2float(prod.y);
            }
        }
        base16 = base16 + K_vec8_rem * 8;
    }
    for (int i = base16 + lane; i < K; i += 32)
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (lane == 0)
        y[row] = __float2half(sum);
}

using Launch = void (*)(const half*, const half*, half*, int, int);

template <int U, bool HX, int T = 256>
void launch_v(const half* A, const half* x, half* y, int M, int K) {
    gemv_fp16_hoist_k<U, HX><<<(M + T / 32 - 1) / (T / 32), T>>>(A, x, y, M, K);
}

template <int U, int P>
void launch_pdl(const half* A, const half* x, half* y, int M, int K) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((M + 7) / 8);
    cfg.blockDim = dim3(256);
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    (void)cudaLaunchKernelEx(&cfg, gemv_fp16_hoist_k<U, false, P>, A, x, y, M, K);
}

void launch_shipped(const half* A, const half* x, half* y, int M, int K) {
    const int64_t as[2] = {M, K}, xs[1] = {K}, ys[1] = {M};
    const imp::Tensor At(const_cast<half*>(A), imp::QType::F16, 2, as, true);
    const imp::Tensor xt(const_cast<half*>(x), imp::QType::F16, 1, xs, true);
    imp::Tensor yt(y, imp::QType::F16, 1, ys, true);
    imp::gemv(At, xt, yt, nullptr);
}

}  // namespace

TEST(GemvFp16HotCold, DISABLED_Bench) {
    struct Shape {
        int m, k;
        const char* tag;
    };
    // K a multiple of 8 keeps every row 16 B aligned (the float4 loads need it); 2312 covers the tails.
    const Shape shapes[] = {{4096, 2304, "inkling-proj"},
                            {2304, 3072, "inkling-o"},
                            {5120, 5120, "qwen-5120"},
                            {96, 2312, "tails"}};
    struct Variant {
        const char* tag;
        Launch fn;
    };
    const Variant vars[] = {{"shipped", launch_shipped},    {"U1", launch_v<1, false>},
                            {"U8", launch_v<8, false>},     {"U4+x", launch_v<4, true>},
                            {"pdl-wait", launch_pdl<1, 1>}, {"pdl-trig", launch_pdl<1, 3>},
                            {"pdl-pre2", launch_pdl<2, 2>}, {"pdl-pre4", launch_pdl<4, 2>},
                            {"pdl-pre8", launch_pdl<8, 2>}};
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (const auto& sh : shapes) {
        const size_t n = static_cast<size_t>(sh.m) * sh.k;
        const size_t bytes = n * sizeof(half);
        const int copies = static_cast<int>((size_t(400) << 20) / bytes) + 2;
        std::vector<half> wh(n), xh(sh.k);
        for (auto& h : wh)
            h = __float2half(dist(rng));
        for (auto& h : xh)
            h = __float2half(dist(rng));
        half *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_w, bytes * copies), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, sh.k * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, sh.m * sizeof(half)), cudaSuccess);
        for (int c = 0; c < copies; ++c)
            ASSERT_EQ(cudaMemcpy(d_w + n * c, wh.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_x, xh.data(), sh.k * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

        std::vector<half> ref(sh.m), got(sh.m);
        launch_shipped(d_w, d_x, d_y, sh.m, sh.k);
        ASSERT_EQ(cudaMemcpy(ref.data(), d_y, sh.m * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
        for (const auto& v : vars) {
            v.fn(d_w, d_x, d_y, sh.m, sh.k);
            ASSERT_EQ(cudaMemcpy(got.data(), d_y, sh.m * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(std::memcmp(ref.data(), got.data(), sh.m * sizeof(half)), 0) << sh.tag << " " << v.tag;
        }

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        // Warm clocks > 1 s on the ring.
        for (int i = 0; i < 60000; ++i)
            launch_v<1, false>(d_w + n * (i % copies), d_x, d_y, sh.m, sh.k);
        cudaDeviceSynchronize();
        constexpr int kIters = 400;
        constexpr int kRounds = 7;
        constexpr int kVars = static_cast<int>(sizeof(vars) / sizeof(vars[0]));
        std::vector<double> hot(kVars * kRounds), cold(kVars * kRounds);
        for (int round = 0; round < kRounds; ++round) {
            for (int vi = 0; vi < kVars; ++vi) {
                const auto& v = vars[vi];
                float ms_hot = 0, ms_cold = 0;
                cudaEventRecord(t0);
                for (int i = 0; i < kIters; ++i)
                    v.fn(d_w, d_x, d_y, sh.m, sh.k);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                cudaEventElapsedTime(&ms_hot, t0, t1);
                cudaEventRecord(t0);
                for (int i = 0; i < kIters; ++i)
                    v.fn(d_w + n * (i % copies), d_x, d_y, sh.m, sh.k);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                cudaEventElapsedTime(&ms_cold, t0, t1);
                const double us_h = ms_hot * 1000.0 / kIters, us_c = ms_cold * 1000.0 / kIters;
                hot[vi * kRounds + round] = us_h, cold[vi * kRounds + round] = us_c;
            }
        }
        for (int vi = 0; vi < kVars; ++vi) {
            std::sort(hot.begin() + vi * kRounds, hot.begin() + (vi + 1) * kRounds);
            std::sort(cold.begin() + vi * kRounds, cold.begin() + (vi + 1) * kRounds);
            const double* h = &hot[vi * kRounds];
            const double* c = &cold[vi * kRounds];
            printf(
                "%-13s %5dx%-5d %-9s hot min/med/max %6.2f %6.2f %6.2f us  cold %6.2f %6.2f %6.2f us (%5.0f "
                "GB/s)\n",
                sh.tag, sh.m, sh.k, vars[vi].tag, h[0], h[kRounds / 2], h[kRounds - 1], c[0], c[kRounds / 2],
                c[kRounds - 1], bytes / c[kRounds / 2] / 1e3);
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
}

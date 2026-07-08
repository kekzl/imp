// Standalone CUDA Tile C++ FlashAttention-2 prefill prototype (GOAL Phase 2 proto).
// Non-causal, single head, fp16 in / fp32 acc. Verifies the cuTile FA2 op-mapping
// runs correctly on sm_120a vs a CPU softmax-attention reference. NOT integrated
// into imp — pure viability/correctness prototype (zero degeneration risk).
//
// Run via the WSL-driver dev recipe (see docs/archive/tile-fa2-dispatch-shelved.md). Build:
//   nvcc -std=c++23 --enable-tile -arch=sm_120a -o tile_fa2_probe tile_fa2_probe.cu
#include "cuda_tile.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int S = 128;  // seq len (q and kv)
constexpr int D = 64;   // head dim
constexpr int TM = 64;  // query tile
constexpr int TN = 64;  // kv tile

// One block per query row-tile (grid.x = S/TM). Loops over KV tiles internally.
__tile_global__ void fa2(__half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ V,
                         __half* __restrict__ O, float scale) {
    auto qView = ct::partition_view{ct::tensor_span{Q, ct::extents{128_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto kView = ct::partition_view{ct::tensor_span{K, ct::extents{128_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto vView = ct::partition_view{ct::tensor_span{V, ct::extents{128_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};
    auto oView = ct::partition_view{ct::tensor_span{O, ct::extents{128_ic, 64_ic}}, ct::shape{64_ic, 64_ic}};

    int qb = ct::bid().x;
    auto q = qView.load_masked(qb, 0);  // [TM, D] fp16

    auto acc = ct::full<ct::tile<float, ct::shape<TM, D>>>(0.0f);
    auto m = ct::full<ct::tile<float, ct::shape<TM, 1>>>(-1e30f);
    auto l = ct::full<ct::tile<float, ct::shape<TM, 1>>>(0.0f);

    for (auto j : ct::irange(0, S / TN)) {
        auto k = kView.load_masked(j, 0);       // [TN, D]
        auto kt = ct::transpose(k);             // [D, TN]
        auto qk0 = ct::full<ct::tile<float, ct::shape<TM, TN>>>(0.0f);
        auto qk = ct::mma(q, kt, qk0) * scale;  // [TM, TN] fp32, scaled

        // causal mask: drop keys whose global col > query global row
        auto idx = ct::iota<ct::tile<int, ct::shape<TM, TN>>>();  // flat row-major 0..TM*TN-1
        auto grow = idx / TN + (qb * TM);                  // global query row per element
        auto gcol = idx % TN + (j * TN);                   // global key col per element
        auto neg = ct::full<ct::tile<float, ct::shape<TM, TN>>>(-1e30f);
        qk = ct::select(gcol > grow, neg, qk);

        auto rmax = ct::reduce_max(qk, 1_ic);              // [TM,1]
        auto mij = ct::select(m > rmax, m, rmax);          // elementwise max → [TM,1]
        auto alpha = ct::exp(m - mij);                     // [TM,1]
        acc = acc * alpha;                                 // auto-broadcast [TM,D]*[TM,1]
        auto p = ct::exp(qk - mij);                        // auto-broadcast [TM,TN]-[TM,1]
        l = l * alpha + ct::sum(p, 1_ic);
        m = mij;

        auto v = vView.load_masked(j, 0);                  // [TN, D]
        auto ph = ct::tile<__half, ct::shape<TM, TN>>(p);  // cast P → fp16 for mma
        acc = ct::mma(ph, v, acc);                         // [TM, D]
    }

    auto out = acc / l;                                    // auto-broadcast [TM,D]/[TM,1]
    auto outh = ct::tile<__half, ct::shape<TM, D>>(out);
    oView.store_masked(outh, qb, 0);
}

int main() {
    std::vector<float> Qf(S * D), Kf(S * D), Vf(S * D), Of(S * D, 0);
    for (int i = 0; i < S * D; i++) {
        Qf[i] = 0.02f * ((i * 7 + 3) % 13 - 6);
        Kf[i] = 0.02f * ((i * 11 + 5) % 13 - 6);
        Vf[i] = 0.02f * ((i * 13 + 7) % 13 - 6);
    }
    float scale = 1.0f / sqrtf((float)D);
    // CPU reference (non-causal softmax attention)
    std::vector<float> ref(S * D, 0);
    for (int q = 0; q < S; q++) {
        std::vector<float> s(S);
        float mx = -1e30f;
        for (int kk = 0; kk < S; kk++) {
            float d = 0;
            for (int x = 0; x < D; x++) d += Qf[q * D + x] * Kf[kk * D + x];
            s[kk] = (kk > q) ? -1e30f : d * scale;  // causal
            mx = fmaxf(mx, s[kk]);
        }
        float sum = 0;
        for (int kk = 0; kk < S; kk++) { s[kk] = expf(s[kk] - mx); sum += s[kk]; }
        for (int x = 0; x < D; x++) {
            float o = 0;
            for (int kk = 0; kk < S; kk++) o += (s[kk] / sum) * Vf[kk * D + x];
            ref[q * D + x] = o;
        }
    }

    __half *dQ, *dK, *dV, *dO;
    cudaMallocManaged(&dQ, S * D * 2);
    cudaMallocManaged(&dK, S * D * 2);
    cudaMallocManaged(&dV, S * D * 2);
    cudaMallocManaged(&dO, S * D * 2);
    for (int i = 0; i < S * D; i++) { dQ[i] = __float2half(Qf[i]); dK[i] = __float2half(Kf[i]); dV[i] = __float2half(Vf[i]); dO[i] = __float2half(-999.f); }

    fa2<<<dim3(S / TM, 1, 1), 1>>>(dQ, dK, dV, dO, scale);
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    printf("launch=%s sync=%s\n", cudaGetErrorString(le), cudaGetErrorString(se));

    double maxerr = 0;
    for (int i = 0; i < S * D; i++) {
        float got = __half2float(dO[i]);
        maxerr = fmax(maxerr, fabs(got - ref[i]) / fmax(1.f, fabs(ref[i])));
    }
    printf("max_rel_err = %.5f  c[0]=%.4f ref[0]=%.4f\n", maxerr, __half2float(dO[0]), ref[0]);
    printf(maxerr < 0.05 ? "TILE FA2 PREFILL CORRECT\n" : "TILE FA2 WRONG\n");
    return maxerr < 0.05 ? 0 : 1;
}

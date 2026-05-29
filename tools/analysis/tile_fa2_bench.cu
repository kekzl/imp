// Tile FA2 perf microbench (GOAL pivotal Q: is cuTile codegen competitive on sm_120?).
// Causal fp16 prefill, S=2048 D=128, single head replicated over grid.y to fill the GPU.
// cudaEvent timing → effective attention TFLOPS vs the 838 TFLOPS FP16 TC roofline.
// Standalone (no imp code). Correctness sanity vs CPU causal oracle for head 0.
#include "cuda_tile.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int S = 2048, D = 128, TM = 64, TN = 64;

__tile_global__ void fa2(__half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ V,
                         __half* __restrict__ O, float scale) {
    auto qView = ct::partition_view{ct::tensor_span{Q, ct::extents{2048_ic, 128_ic}}, ct::shape{64_ic, 128_ic}};
    auto kView = ct::partition_view{ct::tensor_span{K, ct::extents{2048_ic, 128_ic}}, ct::shape{64_ic, 128_ic}};
    auto vView = ct::partition_view{ct::tensor_span{V, ct::extents{2048_ic, 128_ic}}, ct::shape{64_ic, 128_ic}};
    auto oView = ct::partition_view{ct::tensor_span{O, ct::extents{2048_ic, 128_ic}}, ct::shape{64_ic, 128_ic}};

    int qb = ct::bid().x;
    auto q = qView.load_masked(qb, 0);
    auto acc = ct::full<ct::tile<float, ct::shape<TM, D>>>(0.0f);
    auto m = ct::full<ct::tile<float, ct::shape<TM, 1>>>(-1e30f);
    auto l = ct::full<ct::tile<float, ct::shape<TM, 1>>>(0.0f);

    for (auto j : ct::irange(0, S / TN)) {  // full KV range; causal mask zeroes future keys
        auto k = kView.load_masked(j, 0);
        auto kt = ct::transpose(k);
        auto qk = ct::mma(q, kt, ct::full<ct::tile<float, ct::shape<TM, TN>>>(0.0f)) * scale;
        auto idx = ct::iota<ct::tile<int, ct::shape<TM, TN>>>();
        auto grow = idx / TN + (qb * TM);
        auto gcol = idx % TN + (j * TN);
        auto neg = ct::full<ct::tile<float, ct::shape<TM, TN>>>(-1e30f);
        qk = ct::select(gcol > grow, neg, qk);
        auto rmax = ct::reduce_max(qk, 1_ic);
        auto mij = ct::select(m > rmax, m, rmax);
        auto alpha = ct::exp(m - mij);
        acc = acc * alpha;
        auto p = ct::exp(qk - mij);
        l = l * alpha + ct::sum(p, 1_ic);
        m = mij;
        auto v = vView.load_masked(j, 0);
        auto ph = ct::tile<__half, ct::shape<TM, TN>>(p);
        acc = ct::mma(ph, v, acc);
    }
    auto out = acc / l;
    oView.store_masked(ct::tile<__half, ct::shape<TM, D>>(out), qb, 0);
}

int main() {
    size_t n = (size_t)S * D;
    std::vector<float> Qf(n), Kf(n), Vf(n);
    for (size_t i = 0; i < n; i++) { Qf[i] = 0.02f * ((i * 7 + 3) % 13 - 6); Kf[i] = 0.02f * ((i * 11 + 5) % 13 - 6); Vf[i] = 0.02f * ((i * 13 + 7) % 13 - 6); }
    // host fp16 staging
    std::vector<__half> Qh(n), Kh(n), Vh(n), Oh(n);
    for (size_t i = 0; i < n; i++) { Qh[i] = __float2half(Qf[i]); Kh[i] = __float2half(Kf[i]); Vh[i] = __float2half(Vf[i]); }
    __half *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, n * 2); cudaMalloc(&dK, n * 2); cudaMalloc(&dV, n * 2); cudaMalloc(&dO, n * 2);
    cudaMemcpy(dQ, Qh.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kh.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vh.data(), n * 2, cudaMemcpyHostToDevice);
    float scale = 1.0f / sqrtf((float)D);

    // warmup + correctness sanity (head 0)
    fa2<<<dim3(S / TM, 1, 1), 1>>>(dQ, dK, dV, dO, scale);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("RUN ERROR\n"); return 2; }
    cudaMemcpy(Oh.data(), dO, n * 2, cudaMemcpyDeviceToHost);
    // check a few rows vs CPU causal ref
    double maxerr = 0;
    for (int q = 0; q < S; q += 257) {
        std::vector<float> s(S); float mx = -1e30f;
        for (int kk = 0; kk <= q; kk++) { float d = 0; for (int x = 0; x < D; x++) d += Qf[q*D+x]*Kf[kk*D+x]; s[kk] = d*scale; mx = fmaxf(mx, s[kk]); }
        float sm = 0; for (int kk = 0; kk <= q; kk++) { s[kk] = expf(s[kk]-mx); sm += s[kk]; }
        for (int x = 0; x < D; x++) { float o = 0; for (int kk = 0; kk <= q; kk++) o += s[kk]/sm*Vf[kk*D+x]; maxerr = fmax(maxerr, fabs(o - __half2float(Oh[q*D+x]))/fmax(1.f,fabs(o))); }
    }
    printf("correctness (sampled): max_rel_err=%.4f %s\n", maxerr, maxerr < 0.05 ? "OK" : "FAIL");

    // timed: replicate over grid.y to fill GPU
    const int HEADS = 32, REPS = 50;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < REPS; r++) fa2<<<dim3(S / TM, HEADS, 1), 1>>>(dQ, dK, dV, dO, scale);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= REPS;
    // causal attention FLOPs per head ≈ 2*S^2*D (QK + PV, ×2 macs, ×0.5 causal)
    double flop = 2.0 * (double)S * S * D * HEADS;
    printf("S=%d D=%d heads=%d : %.3f ms/iter, %.1f eff-TFLOPS (FP16 roofline 838)\n", S, D, HEADS, ms, flop / (ms * 1e-3) / 1e12);
    return 0;
}

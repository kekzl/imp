// CUDA Tile C++ viability probe for sm_120a (GOAL §5 investigation-first / §11).
// Minimal tiled matmul C[M,N]=A[M,K]*B[K,N] via cuda::tiles. Verifies correctness
// vs CPU and lets us inspect SASS (mma.sync/HMMA vs tcgen05). API mirrors the
// official "Develop High-Performance GPU Kernels in C++ with NVIDIA CUDA Tile" blog.
#include "cuda_tile.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace ct = cuda::tiles;
using namespace ct::literals;

// M=8, K=8, N=16; 4x4 output tiles → grid (M/4=2, N/4=4).
__tile_global__ void mm(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c) {
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{8_ic, 8_ic}}, ct::shape{4_ic, 8_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{8_ic, 16_ic}}, ct::shape{8_ic, 4_ic}};
    auto cView = ct::partition_view{ct::tensor_span{c, ct::extents{8_ic, 16_ic}}, ct::shape{4_ic, 4_ic}};
    auto accTile = ct::full<ct::tile<float, ct::shape<4, 4>>>(0);
    auto [xBlock, yBlock, dummy] = ct::bid();
    auto aTile = aView.load_masked(xBlock, 0);
    auto bTile = bView.load_masked(0, yBlock);
    accTile = ct::mma(aTile, bTile, accTile);
    cView.store_masked(accTile, xBlock, yBlock);
}

int main() {
    const int M = 8, K = 8, N = 16;
    float *a, *b, *c;
    cudaMallocManaged(&a, M * K * sizeof(float));
    cudaMallocManaged(&b, K * N * sizeof(float));
    cudaMallocManaged(&c, M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a[i] = (float)((i * 7 + 1) % 5) - 2.0f;
    for (int i = 0; i < K * N; i++) b[i] = (float)((i * 3 + 2) % 5) - 2.0f;
    for (int i = 0; i < M * N; i++) c[i] = -999.0f;

    mm<<<dim3(2, 4), 1>>>(a, b, c);
    cudaError_t le = cudaGetLastError();
    printf("launch err: %s\n", cudaGetErrorString(le));
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync err:   %s\n", cudaGetErrorString(e));
    printf("c[0..3] = %.1f %.1f %.1f %.1f\n", c[0], c[1], c[2], c[3]);
    if (e != cudaSuccess) { printf("LAUNCH/RUN ERROR\n"); return 2; }

    double max_err = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)a[m * K + k] * b[k * N + n];
            max_err = fmax(max_err, fabs(ref - c[m * N + n]));
        }
    printf("max_abs_err = %.6g  (c[0]=%.1f ref-check)\n", max_err, c[0]);
    printf(max_err < 1e-3 ? "TILE MATMUL CORRECT on sm_120a\n" : "TILE MATMUL WRONG\n");
    return max_err < 1e-3 ? 0 : 1;
}

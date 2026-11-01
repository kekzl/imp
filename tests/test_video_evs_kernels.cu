// EVS device half against a host reference: per-token dissimilarity (1 - cosine to the same token
// of the previous frame pair, 255 for pair 0) and the row gather that compacts the kept tokens.

#include "vision/video_evs.h"
#include "vision/video_evs_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace imp {
namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::vector<half> random_rows(int rows, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<half> v(static_cast<size_t>(rows) * dim);
    for (auto& x : v)
        x = __float2half(u(rng));
    return v;
}

std::vector<float> host_dissim(const std::vector<half>& emb, int groups, int tpg, int dim) {
    std::vector<float> out(static_cast<size_t>(groups) * tpg);
    for (int j = 0; j < groups * tpg; ++j) {
        if (j < tpg) {
            out[j] = kEvsFirstGroupScore;
            continue;
        }
        double xy = 0, xx = 0, yy = 0;
        for (int k = 0; k < dim; ++k) {
            const double a = __half2float(emb[(static_cast<size_t>(j) * dim) + k]);
            const double b = __half2float(emb[(static_cast<size_t>(j - tpg) * dim) + k]);
            xy += a * b;
            xx += a * a;
            yy += b * b;
        }
        out[j] = static_cast<float>(1.0 -
                                    (xy / (std::max(std::sqrt(xx), 1e-8) * std::max(std::sqrt(yy), 1e-8))));
    }
    return out;
}

void run_dissim_case(int groups, int tpg, int dim) {
    auto emb = random_rows(groups * tpg, dim, 1234u + static_cast<uint32_t>(dim));
    // Pair 2 token 1 repeats pair 1 token 1: cosine 1, dissimilarity 0.
    if (groups > 2)
        std::copy_n(emb.begin() + (static_cast<size_t>(tpg + 1) * dim), dim,
                    emb.begin() + (static_cast<size_t>((2 * tpg) + 1) * dim));
    const int n = groups * tpg;
    half* d_emb = nullptr;
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_emb, emb.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_emb, emb.data(), emb.size() * sizeof(half), cudaMemcpyHostToDevice);
    launch_evs_dissimilarity(d_emb, groups, tpg, dim, d_out, nullptr);
    std::vector<float> got(static_cast<size_t>(n));
    ASSERT_EQ(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_emb);
    cudaFree(d_out);
    const auto want = host_dissim(emb, groups, tpg, dim);
    for (int j = 0; j < n; ++j)
        EXPECT_NEAR(got[j], want[j], 1e-4f) << "row " << j << " dim " << dim;
    if (groups > 2)
        EXPECT_NEAR(got[(2 * tpg) + 1], 0.0f, 1e-5f);
}

TEST(VideoEvsKernels, DissimilarityMatchesHostCosine) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    run_dissim_case(4, 6, 2560);  // half2 path, Qwen3-VL-4B width
    run_dissim_case(3, 5, 33);    // odd width: scalar path
}

TEST(VideoEvsKernels, GatherCompactsKeptRowsInOrder) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    for (const int dim : {64, 30}) {  // uint4 path and the scalar tail path
        const int rows = 12;
        const auto in = random_rows(rows, dim, 77u);
        const std::vector<int32_t> keep = {0, 1, 2, 3, 5, 8, 11};
        half *d_in = nullptr, *d_out = nullptr;
        int32_t* d_idx = nullptr;
        ASSERT_EQ(cudaMalloc(&d_in, in.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, keep.size() * dim * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_idx, keep.size() * sizeof(int32_t)), cudaSuccess);
        cudaMemcpy(d_in, in.data(), in.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx, keep.data(), keep.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        launch_gather_rows(d_in, d_idx, static_cast<int>(keep.size()), dim, d_out, nullptr);
        std::vector<half> got(keep.size() * dim);
        ASSERT_EQ(cudaMemcpy(got.data(), d_out, got.size() * sizeof(half), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_idx);
        for (size_t r = 0; r < keep.size(); ++r)
            for (int k = 0; k < dim; ++k)
                EXPECT_EQ(__half_as_ushort(got[(r * dim) + k]),
                          __half_as_ushort(in[(static_cast<size_t>(keep[r]) * dim) + k]))
                    << "row " << r << " col " << k;
    }
}

}  // namespace
}  // namespace imp

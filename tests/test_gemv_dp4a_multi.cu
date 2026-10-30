// gemv_dp4a_multi (mixed-type multi-weight decode GEMV) against the single-weight dp4a GEMVs on the same
// q8_1 activation: the UD GGUF GDN input shape (in_proj 10240 Q5_K, gate 6144 IQ4_XS, alpha / beta 48 Q8_0,
// K 5120) and a Q4_K / Q6_K pair. Large rows run the same kpar kernel (bit-equal); small rows may take the
// row-parallel kernel (other reduction order), so the gate is a tolerance plus a bit-equal count.

#include "compute/gemm.h"
#include "compute/gemv_dp4a_multi.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

namespace {

int block_bytes(QType q) {
    switch (q) {
        case QType::Q8_0:
            return 34;
        case QType::Q4_K:
            return 144;
        case QType::Q5_K:
            return 176;
        case QType::Q6_K:
            return 210;
        case QType::IQ4_XS:
            return 136;
        default:
            return 0;
    }
}
int block_elems(QType q) { return q == QType::Q8_0 ? 32 : 256; }

// Random blocks with a small finite fp16 scale (and min) at the format's scale offsets.
std::vector<uint8_t> random_blocks(QType q, int rows, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    const int bb = block_bytes(q);
    const size_t n = static_cast<size_t>(rows) * (K / block_elems(q));
    std::vector<uint8_t> w(n * bb);
    for (auto& x : w)
        x = static_cast<uint8_t>(rng());
    std::uniform_real_distribution<float> dd(1e-5f, 4e-5f);  // keeps |y| far below the fp16 range
    for (size_t b = 0; b < n; ++b) {
        uint8_t* p = w.data() + (b * bb);
        const half d = __float2half(dd(rng));
        const half m = __float2half(dd(rng));
        if (q == QType::Q6_K) {
            std::memcpy(p + 208, &d, 2);
        } else {
            std::memcpy(p, &d, 2);
            if (q == QType::Q4_K || q == QType::Q5_K)
                std::memcpy(p + 2, &m, 2);
        }
    }
    return w;
}

void single_gemv(QType q, const void* w, const block_q8_1* q8, const float* d8, half* y, int rows, int K,
                 cudaStream_t s) {
    switch (q) {
        case QType::Q8_0:
            gemv_q8_0_q8_1(w, q8, d8, y, rows, K, s);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1(w, q8, d8, y, rows, K, s);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1(w, q8, d8, y, rows, K, s);
            break;
        case QType::Q6_K:
            gemv_q6k_q8_1(w, q8, d8, y, rows, K, s);
            break;
        case QType::IQ4_XS:
            gemv_iq4_xs_q8_1(w, q8, d8, y, rows, K, s);
            break;
        default:
            break;
    }
}

struct Case {
    std::vector<QType> types;
    std::vector<int> rows;
    int K;
};

class GemvDp4aMultiTest : public ::testing::TestWithParam<int> {};

const Case kCases[] = {
    {{QType::Q5_K, QType::IQ4_XS, QType::Q8_0, QType::Q8_0}, {10240, 6144, 48, 48}, 5120},
    {{QType::Q4_K, QType::Q6_K}, {2048, 1024}, 4096},
    {{QType::Q8_0, QType::Q4_K, QType::Q5_K}, {12288, 1024, 1024}, 5120},
};

}  // namespace

TEST_P(GemvDp4aMultiTest, MatchesSingleWeightGemvs) {
    const Case& c = kCases[GetParam()];
    const int K = c.K;
    const int n = static_cast<int>(c.types.size());
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
    std::vector<half> hx(K);
    for (auto& v : hx)
        v = __float2half(ux(rng));
    half* x = nullptr;
    block_q8_1* q8 = nullptr;
    float* d8 = nullptr;
    cudaMalloc(&x, K * sizeof(half));
    cudaMalloc(&q8, (K / 32) * sizeof(block_q8_1));
    cudaMalloc(&d8, (K / 32) * sizeof(float));
    cudaMemcpy(x, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice);
    quantize_fp16_to_q8_1(x, q8, d8, K, s);

    std::vector<void*> w(n);
    std::vector<half*> y_ref(n), y_own(n);
    std::vector<Dp4aGemvSeg> segs(n);
    for (int i = 0; i < n; ++i) {
        const auto host = random_blocks(c.types[i], c.rows[i], K, 100 + i);
        cudaMalloc(&w[i], host.size());
        cudaMemcpy(w[i], host.data(), host.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&y_ref[i], c.rows[i] * sizeof(half));
        cudaMalloc(&y_own[i], c.rows[i] * sizeof(half));
        single_gemv(c.types[i], w[i], q8, d8, y_ref[i], c.rows[i], K, s);
        segs[i] = {w[i], c.types[i], y_own[i], c.rows[i]};
    }
    ASSERT_TRUE(gemv_dp4a_multi(segs.data(), n, q8, d8, K, s));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    for (int i = 0; i < n; ++i) {
        std::vector<half> r(c.rows[i]), o(c.rows[i]);
        cudaMemcpy(r.data(), y_ref[i], r.size() * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(o.data(), y_own[i], o.size() * sizeof(half), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_d = 0.0;
        size_t exact = 0;
        for (size_t j = 0; j < r.size(); ++j) {
            const double a = __half2float(r[j]), b = __half2float(o[j]);
            max_ref = std::fmax(max_ref, std::fabs(a));
            max_d = std::fmax(max_d, std::fabs(a - b));
            exact += std::memcmp(&r[j], &o[j], sizeof(half)) == 0 ? 1 : 0;
        }
        std::fprintf(
            stderr,
            "[dp4a multi] case %d seg %d %s rows %d: %zu / %zu bit-equal, max |d| %.3g (max |y| %.3g)\n",
            GetParam(), i, qtype_name(c.types[i]), c.rows[i], exact, r.size(), max_d, max_ref);
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_d, 2e-3 * max_ref + 1e-3);
        cudaFree(w[i]);
        cudaFree(y_ref[i]);
        cudaFree(y_own[i]);
    }
    cudaFree(x);
    cudaFree(q8);
    cudaFree(d8);
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Cases, GemvDp4aMultiTest, ::testing::Values(0, 1, 2));

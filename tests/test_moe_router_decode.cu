// moe_router_decode against the path it replaces (gemv_gate_fp32 + moe_topk_gating): logits, expert ids
// and weights bit-identical over repeated launches (ticket re-arm), with and without selection bias;
// refusals.
#include "compute/gemm.h"
#include "compute/moe_router_decode.h"
#include "compute/moe_routing.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { cudaMalloc(&p, n); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

template <typename T>
std::vector<T> to_host(const void* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

void run_case(int ne, int K, int top_k, bool bias, bool normalize, int launches) {
    std::mt19937 rng(ne * 131 + K + top_k + (bias ? 7 : 0));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<half> w(static_cast<size_t>(ne) * K), b(ne);
    for (auto& v : w)
        v = __float2half(nd(rng) * 0.05f);
    for (auto& v : b)
        v = __float2half(nd(rng) * 0.1f);
    Dev dw(w.size() * sizeof(half)), dx(K * sizeof(half)), db(ne * sizeof(half));
    Dev ref_logits(ne * sizeof(float)), new_logits(ne * sizeof(float));
    Dev ids(top_k * sizeof(int32_t)), wts(top_k * sizeof(float)), ticket(sizeof(unsigned int));
    cudaMemcpy(dw.p, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(db.p, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(ticket.p, 0, sizeof(unsigned int));
    for (int l = 0; l < launches; ++l) {
        std::vector<half> x(K);
        for (auto& v : x)
            v = __float2half(nd(rng));
        cudaMemcpy(dx.p, x.data(), K * sizeof(half), cudaMemcpyHostToDevice);
        const half* bias_p = bias ? static_cast<const half*>(db.p) : nullptr;

        gemv_gate_fp32(static_cast<const half*>(dw.p), static_cast<const half*>(dx.p),
                       static_cast<float*>(ref_logits.p), ne, K);
        int64_t shape[2] = {1, ne};
        Tensor t(ref_logits.p, QType::F32, 2, shape, true);
        MoeRoutingResult r;
        moe_topk_gating(t, top_k, r, nullptr, /*use_sigmoid=*/true, normalize, bias_p);

        MoeRouterDecodeArgs a{static_cast<const half*>(dw.p),
                              static_cast<const half*>(dx.p),
                              static_cast<float*>(new_logits.p),
                              ne,
                              K,
                              top_k,
                              normalize,
                              bias_p,
                              static_cast<int32_t*>(ids.p),
                              static_cast<float*>(wts.p),
                              static_cast<unsigned int*>(ticket.p)};
        ASSERT_TRUE(moe_router_decode(a, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto lr = to_host<float>(ref_logits.p, ne), ln = to_host<float>(new_logits.p, ne);
        EXPECT_EQ(std::memcmp(lr.data(), ln.data(), ne * sizeof(float)), 0) << "logits, launch " << l;
        const auto ir = to_host<int32_t>(r.expert_indices.data, top_k), in = to_host<int32_t>(ids.p, top_k);
        const auto wr = to_host<float>(r.expert_weights.data, top_k), wn = to_host<float>(wts.p, top_k);
        for (int k = 0; k < top_k; ++k) {
            EXPECT_EQ(ir[k], in[k]) << "slot " << k << " launch " << l;
            EXPECT_EQ(std::memcmp(&wr[k], &wn[k], sizeof(float)), 0) << "weight " << k << " launch " << l;
        }
        EXPECT_EQ(to_host<unsigned int>(ticket.p, 1)[0], 0U) << "ticket not re-armed";
        cudaFree(r.expert_indices.data), cudaFree(r.expert_weights.data);
        cudaFree(r.sorted_token_ids.data), cudaFree(r.expert_offsets.data);
    }
}

TEST(MoeRouterDecode, FlintShapeMatchesGemvPlusTopk) { run_case(256, 2048, 6, false, true, 8); }
TEST(MoeRouterDecode, SelectionBiasMatches) { run_case(256, 2048, 8, true, true, 4); }
TEST(MoeRouterDecode, WideRouterUnnormalisedMatches) { run_case(512, 4096, 10, false, false, 3); }
TEST(MoeRouterDecode, OddRowCountMatches) { run_case(130, 1024, 4, true, true, 3); }

TEST(MoeRouterDecode, RefusesShapesItCannotServe) {
    Dev buf(1 << 20);
    auto* h = static_cast<const half*>(buf.p);
    auto* f = static_cast<float*>(buf.p);
    auto* i = static_cast<int32_t*>(buf.p);
    auto* u = static_cast<unsigned int*>(buf.p);
    EXPECT_FALSE(moe_router_decode({h, h, f, 513, 2048, 6, true, nullptr, i, f, u}, nullptr));  // ne > 512
    EXPECT_FALSE(
        moe_router_decode({h, h, f, 256, 2560, 6, true, nullptr, i, f, u}, nullptr));  // K not served
    EXPECT_FALSE(moe_router_decode({h, h, f, 256, 8192, 6, true, nullptr, i, f, u}, nullptr));   // K > 4096
    EXPECT_FALSE(moe_router_decode({h, h, f, 256, 2048, 17, true, nullptr, i, f, u}, nullptr));  // top_k > 16
    EXPECT_FALSE(
        moe_router_decode({h, h, f, 256, 2048, 6, true, nullptr, i, f, nullptr}, nullptr));  // no ticket
}

}  // namespace
}  // namespace imp

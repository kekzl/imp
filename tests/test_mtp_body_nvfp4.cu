// NVFP4 copy of the dense MTP body (mtp_body_nvfp4_allocate): the copy is the direct quantize of each
// weight, the plan mirror prices it, and one-row steps and multi-row feeds read it. Synthetic head with
// K % 128 == 0 (the small-M GEMM tile). GPU required.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"
#include "core/tensor.h"
#include "model/mtp_head.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

namespace {

constexpr int kH = 128;
constexpr int kVocab = 256;
constexpr int kNh = 4;
constexpr int kNkv = 2;
constexpr int kHd = 32;
constexpr int kDff = 256;
constexpr int kSeq = 64;
constexpr int kPairs = 9;

imp::Tensor upload(const std::vector<__half>& h, std::initializer_list<int64_t> shape) {
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(__half)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<int64_t> s(shape);
    return imp::Tensor(d, imp::QType::F16, static_cast<int>(s.size()), s.data(), /*on_device=*/true);
}

std::vector<__half> random_h(std::mt19937& rng, size_t n, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<__half> v(n);
    for (auto& x : v)
        x = __float2half(dist(rng));
    return v;
}

struct Head {
    imp::MtpHead head;
    imp::Tensor tok_emb;
    void* d_hidden = nullptr;
    std::vector<int32_t> tokens;

    void build(unsigned seed) {
        std::mt19937 rng(seed);
        const float w = 0.25f;
        auto vec = [&](int n) { return upload(random_h(rng, n, w), {n}); };
        auto mat = [&](int r, int c) { return upload(random_h(rng, size_t(r) * c, w), {r, c}); };
        head.pre_fc_norm_embedding = vec(kH);
        head.pre_fc_norm_hidden = vec(kH);
        head.fc = mat(kH, 2 * kH);
        head.input_layernorm = vec(kH);
        head.post_attention_layernorm = vec(kH);
        head.q_proj = mat(2 * kNh * kHd, kH);
        head.k_proj = mat(kNkv * kHd, kH);
        head.v_proj = mat(kNkv * kHd, kH);
        head.o_proj = mat(kH, kNh * kHd);
        head.q_norm = vec(kHd);
        head.k_norm = vec(kHd);
        head.shared_expert_gate_proj = mat(kDff, kH);
        head.shared_expert_up_proj = mat(kDff, kH);
        head.shared_expert_down_proj = mat(kH, kDff);
        head.final_norm = vec(kH);
        head.attn_output_gate = true;
        head.attn_rope = true;
        head.loaded = true;
        tok_emb = mat(kVocab, kH);
        auto rows = random_h(rng, size_t{kPairs} * kH, 1.0f);
        EXPECT_EQ(cudaMalloc(&d_hidden, rows.size() * sizeof(__half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_hidden, rows.data(), rows.size() * sizeof(__half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        std::uniform_int_distribution<int> tok(0, kVocab - 1);
        tokens.resize(kPairs);
        for (auto& t : tokens)
            t = tok(rng);
    }
    [[nodiscard]] std::vector<const imp::Tensor*> body() const {
        return {&head.fc,
                &head.q_proj,
                &head.k_proj,
                &head.v_proj,
                &head.o_proj,
                &head.shared_expert_gate_proj,
                &head.shared_expert_up_proj,
                &head.shared_expert_down_proj};
    }
};

void allocate(imp::MtpDraftWorkspace& ws) {
    ASSERT_TRUE(imp::mtp_workspace_allocate({.ws = &ws,
                                             .hidden_dim = kH,
                                             .vocab_size = kVocab,
                                             .n_experts = 0,
                                             .top_k = 0,
                                             .expert_d_ff = 0,
                                             .shared_d_ff = kDff,
                                             .num_heads = kNh,
                                             .num_kv_heads = kNkv,
                                             .head_dim = kHd,
                                             .max_seq_len = kSeq}));
    ws.rope_theta = 10000.0f;
    ws.rope_dim = 8;
    ws.mrope_sec0 = 2;
    ws.mrope_sec1 = 2;
    ws.mrope_sec2 = 0;
    ws.rms_norm_eps = 1e-6f;
    ws.arch_norm_offset = 0.0f;
}

std::vector<uint8_t> bytes(const void* d, size_t n) {
    std::vector<uint8_t> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

float rel_rms(const void* a, const void* b, size_t n) {
    std::vector<__half> ha(n), hb(n);
    EXPECT_EQ(cudaMemcpy(ha.data(), a, n * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(hb.data(), b, n * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = __half2float(ha[i]) - __half2float(hb[i]);
        diff2 += d * d;
        ref2 += static_cast<double>(__half2float(ha[i])) * __half2float(ha[i]);
    }
    return ref2 == 0.0 ? 1.0e9f : static_cast<float>(std::sqrt(diff2 / ref2));
}

class MtpBodyNvfp4Test : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

TEST_F(MtpBodyNvfp4Test, CopyIsTheDirectQuantizeOfEachWeightAndThePlanPricesIt) {
    Head s;
    s.build(11);
    imp::MtpDraftWorkspace ws;
    allocate(ws);
    const size_t before = ws.device_bytes;
    ASSERT_GT(imp::mtp_body_nvfp4_bytes(s.head), 0u);
    ASSERT_TRUE(imp::mtp_body_nvfp4_allocate(s.head, ws, nullptr));

    imp::MtpWorkspaceShape d;
    d.hidden_dim = kH;
    d.vocab_size = kVocab;
    d.shared_d_ff = kDff;
    d.num_heads = kNh;
    d.num_kv_heads = kNkv;
    d.head_dim = kHd;
    d.max_seq_len = kSeq;
    auto priced = d;
    priced.body_nvfp4_bytes = imp::mtp_body_nvfp4_bytes(s.head);
    priced.body_nvfp4_ws_bytes = imp::mtp_body_nvfp4_ws_bytes(s.head);
    EXPECT_EQ(ws.device_bytes - before, imp::mtp_workspace_bytes(priced) - imp::mtp_workspace_bytes(d));

    const auto body = s.body();
    for (int i = 0; i < 8; i++) {
        imp::NvFP4Owned ref;
        imp::quantize_fp16_to_nvfp4(*body[i], ref, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const size_t n = static_cast<size_t>(ref.N);
        const size_t k = static_cast<size_t>(ref.K);
        ASSERT_NE(ws.nv_p[i], nullptr) << i;
        EXPECT_EQ(ws.nv_ts[i], ref.tensor_scale) << i;
        EXPECT_EQ(bytes(ws.nv_p[i], n * (k / 2)), bytes(ref.packed_data, n * (k / 2))) << "weight " << i;
        EXPECT_EQ(bytes(ws.nv_m[i], n * (k / 16)), bytes(ref.micro_scales, n * (k / 16))) << "weight " << i;
    }
    EXPECT_FALSE(imp::mtp_body_nvfp4_allocate(s.head, ws, nullptr)) << "a second copy is refused";
    imp::mtp_workspace_free(ws);
    EXPECT_EQ(ws.device_bytes, 0u);
    EXPECT_EQ(ws.nv_p[0], nullptr);
    EXPECT_EQ(ws.d_nv_body, nullptr);
    EXPECT_EQ(ws.d_nv_ws, nullptr) << "the small-M scratch is released with the copy";

    auto moe = s.head;
    const int64_t shape[2] = {8, kH};
    moe.router = imp::Tensor(nullptr, imp::QType::F16, 2, shape, false);
    EXPECT_EQ(imp::mtp_body_nvfp4_bytes(moe), 0u) << "an MoE head keeps its FP16 body";
}

// The copy is read: a one-row step and a multi-row feed land near the FP16 result, not on it.
TEST_F(MtpBodyNvfp4Test, OneRowStepsAndMultiRowFeedsReadTheCopy) {
    Head s;
    s.build(12);
    for (const bool batched : {false, true}) {
        imp::MtpDraftWorkspace ws_f16, ws_nv;
        allocate(ws_f16);
        allocate(ws_nv);
        ASSERT_TRUE(imp::mtp_body_nvfp4_allocate(s.head, ws_nv, nullptr));
        for (imp::MtpDraftWorkspace* ws : {&ws_f16, &ws_nv}) {
            if (batched) {
                ASSERT_TRUE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, kPairs, s.head, s.tok_emb, *ws,
                                                kH, nullptr));
            } else {
                for (int j = 0; j < kPairs; ++j) {
                    const void* h_j = static_cast<const char*>(s.d_hidden) + size_t(j) * kH * sizeof(__half);
                    ASSERT_TRUE(imp::mtp_draft_step({.prev_token_id = s.tokens[j],
                                                     .d_h_prev = h_j,
                                                     .mtp = &s.head,
                                                     .main_tok_emb = &s.tok_emb,
                                                     .main_lm_head = &s.tok_emb,
                                                     .ws = ws,
                                                     .hidden_dim = kH,
                                                     .vocab_size = kVocab,
                                                     .out_token_id = nullptr,
                                                     .stream = nullptr}));
                }
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(ws->mtp_pos, kPairs);
        }
        const size_t kv = size_t{kPairs} * kNkv * kHd;
        const float dk = rel_rms(ws_f16.d_k_cache, ws_nv.d_k_cache, kv);
        const float dh = rel_rms(ws_f16.d_h_final, ws_nv.d_h_final, kH);
        EXPECT_GT(dk, 1e-4f) << "batched " << batched << ": the FP16 weights were read";
        EXPECT_LT(dk, 0.2f) << "batched " << batched;
        EXPECT_GT(dh, 1e-4f) << "batched " << batched;
        EXPECT_LT(dh, 0.35f) << "batched " << batched;
        imp::mtp_workspace_free(ws_f16);
        imp::mtp_workspace_free(ws_nv);
    }
}

}  // namespace

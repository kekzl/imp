// =============================================================================
// test_mtp_forward.cpp — Phase 2.2 MoE-block integration test
// =============================================================================
//
// Exercises mtp_draft_step() end-to-end against the real Qwen3.6-NVFP4 model:
//   1. Load main model + MTP head (BF16 → FP16, 1.57 GiB)
//   2. Allocate MTP workspace with full MoE config (256/top-8/512/512)
//   3. Run mtp_draft_step with a synthetic d_h_prev + token id
//   4. Assert it does NOT crash and out_token_id is in [0, vocab_size)
//
// GTEST_SKIPs if the model is absent so CI on bare hosts still passes.
//
// Spec: docs/superpowers/specs/2026-05-14-mtp-wiring-design.md
// =============================================================================

#include "model/model.h"
#include "model/safetensors_loader.h"
#include "runtime/engine.h"
#include "compute/mtp_forward.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Container bind-mount path (-v $(PWD)/models:/models), matching the rest of
// the SafeTensors suites; skips cleanly when absent (R6/#581 — was a host path
// that never existed in the container, so this test skipped there always).
constexpr const char kQwen36ModelDir[] =
    "/models/Qwen3.6-35B-A3B-NVFP4";

bool model_available() {
    return fs::exists(std::string(kQwen36ModelDir) + "/model_mtp.safetensors");
}

}  // namespace

TEST(MtpForwardTest, DraftStepProducesValidToken) {
    if (!model_available()) {
        GTEST_SKIP() << "Qwen3.6-NVFP4 with MTP not present at " << kQwen36ModelDir;
    }

    // Load model + upload weights with MTP enabled.
    auto model = imp::load_safetensors(kQwen36ModelDir, /*load_mtp_head=*/true);
    ASSERT_NE(model, nullptr);
    ASSERT_TRUE(model->mtp_.has_value());

    // upload_weights_gpu automatically uploads the MTP sidecar when
    // model->mtp_->loaded is set (the safetensors loader sets it after
    // parsing the sidecar tensors). See weight_upload.cu:2027 for the gate.
    ASSERT_TRUE(model->upload_weights_gpu(imp::QType::F16, nullptr, 1ULL << 30));
    ASSERT_TRUE(model->mtp_->loaded);

    // Build a synthetic d_h_prev (random FP16). The MTP forward should still
    // produce a valid token id even with non-realistic hidden state — we just
    // want to confirm the MoE block runs without crashing or producing OOB
    // tokens.
    const int hidden_dim   = model->config_.d_model;
    const int vocab_size   = model->config_.vocab_size;
    const int n_experts    = model->config_.n_experts;
    const int top_k        = model->config_.n_experts_active;
    const int expert_d_ff  = model->config_.expert_d_ff;
    const int shared_d_ff  = model->config_.expert_shared_d_ff;

    ASSERT_EQ(n_experts, 256);
    ASSERT_EQ(top_k, 8);
    ASSERT_EQ(expert_d_ff, 512);
    ASSERT_EQ(shared_d_ff, 512);

    // Attention dims: derive from MTP head's q_proj / v_proj shapes (different
    // from the main model because Qwen3.6 MTP uses attn_output_gate=True).
    ASSERT_NE(model->mtp_->q_proj.data, nullptr);
    ASSERT_NE(model->mtp_->v_proj.data, nullptr);
    const int q_out          = static_cast<int>(model->mtp_->q_proj.shape[0]);
    const int v_out          = static_cast<int>(model->mtp_->v_proj.shape[0]);
    const int mtp_head_dim   = model->config_.head_dim;
    const int mtp_num_heads  = q_out / (2 * mtp_head_dim);
    const int mtp_num_kv     = v_out / mtp_head_dim;
    EXPECT_EQ(mtp_head_dim, 256);
    EXPECT_EQ(mtp_num_heads, 16);
    EXPECT_EQ(mtp_num_kv, 2);

    imp::MtpDraftWorkspace ws{};
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws, hidden_dim, vocab_size,
                                             n_experts, top_k, expert_d_ff, shared_d_ff,
                                             mtp_num_heads, mtp_num_kv, mtp_head_dim));

    // Build a host-side random FP16 hidden state, upload.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<uint16_t> h_state(hidden_dim);
    auto float_to_fp16 = [](float v) -> uint16_t {
        // Quick-and-dirty: use cuda's __float2half via a device pass. For
        // simplicity, encode FP16 by hand for values near zero. We only need
        // values in a sane range, exact bits don't matter.
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        uint16_t s = (bits >> 31) & 1;
        int e = ((bits >> 23) & 0xFF) - 127;
        uint32_t m = bits & 0x7FFFFF;
        if (e > 15)  return (s << 15) | 0x7C00;
        if (e < -14) return (s << 15);
        return (s << 15) | ((e + 15) << 10) | (m >> 13);
    };
    for (int i = 0; i < hidden_dim; ++i) {
        h_state[i] = float_to_fp16(dist(rng));
    }
    void* d_h_prev = nullptr;
    ASSERT_EQ(cudaMalloc(&d_h_prev, hidden_dim * sizeof(uint16_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_h_prev, h_state.data(), hidden_dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), cudaSuccess);

    int out_token_id = -1;
    bool ok = imp::mtp_draft_step(
        /*prev_token_id=*/123,            // arbitrary valid token
        d_h_prev,
        *model->mtp_,
        model->tok_emb_,
        model->out_proj_,
        ws,
        hidden_dim, vocab_size,
        &out_token_id,
        /*stream=*/nullptr);

    EXPECT_TRUE(ok) << "mtp_draft_step returned false";
    EXPECT_GE(out_token_id, 0);
    EXPECT_LT(out_token_id, vocab_size);

    cudaFree(d_h_prev);
    imp::mtp_workspace_free(ws);
}

// mtp_draft_step input guard: attention weights present but no V projection buffer
// (num_kv_heads == 0, d_v_proj == nullptr) must return false before any CUDA call.
// Fake non-null pointers, never dereferenced on the guarded path; no GPU needed.
#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "model/mtp_head.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace imp {
namespace {

uint8_t g_byte[64] = {};

Tensor fake(int64_t rows, int64_t cols) {
    int64_t s[2] = {rows, cols};
    return Tensor(g_byte, QType::F16, 2, s, /*on_device=*/true);
}

TEST(MtpAttnGuard, ZeroKvHeadsWithAttentionWeightsReturnsFalse) {
    constexpr int kHidden = 8, kVocab = 16, kHeads = 2, kHeadDim = 4;
    MtpHead mtp;
    mtp.loaded = true;
    mtp.pre_fc_norm_embedding = fake(1, kHidden);
    mtp.pre_fc_norm_hidden = fake(1, kHidden);
    mtp.fc = fake(kHidden, 2 * kHidden);
    mtp.input_layernorm = fake(1, kHidden);
    mtp.q_proj = fake(kHeads * kHeadDim, kHidden);
    mtp.k_proj = fake(1, kHidden);  // v_out < head_dim: engine derives num_kv_heads = 0
    mtp.v_proj = fake(1, kHidden);
    mtp.o_proj = fake(kHidden, kHeads * kHeadDim);

    MtpDraftWorkspace ws;
    void* p = g_byte;
    ws.d_emb_norm = ws.d_h_norm = ws.d_fc_in = ws.d_fc_out = ws.d_h_final = ws.d_logits = p;
    ws.d_input_norm = ws.d_q_full = ws.d_q_attn = ws.d_attn_out = ws.d_attn_residual = p;
    ws.num_heads = kHeads;
    ws.head_dim = kHeadDim;
    ws.num_kv_heads = 0;  // mtp_workspace_allocate skips d_k_proj / d_v_proj
    ASSERT_EQ(ws.d_v_proj, nullptr);

    const Tensor emb = fake(kVocab, kHidden);
    const Tensor lm_head = fake(kVocab, kHidden);
    const int32_t* d_prev = reinterpret_cast<const int32_t*>(g_byte);
    int out = -1;
    const LogLevel saved = log_get_level();
    log_set_level(LogLevel::WARN);  // both log assertions need ERROR lines emitted
    testing::internal::CaptureStderr();
    const bool ok = mtp_draft_step(/*prev_token_id=*/0, g_byte, mtp, emb, lm_head, ws, kHidden, kVocab, &out,
                                   /*stream=*/nullptr, nullptr, 0, nullptr, d_prev);
    const std::string log = testing::internal::GetCapturedStderr();
    log_set_level(saved);
    EXPECT_FALSE(ok);
    EXPECT_EQ(out, -1);
    EXPECT_NE(log.find("attention weights without a V projection buffer"), std::string::npos) << log;
    EXPECT_EQ(log.find("CUDA"), std::string::npos) << log;  // guard fires before any launch
    ws = MtpDraftWorkspace{};  // fake pointers must not reach any destructor/free path
}

}  // namespace
}  // namespace imp

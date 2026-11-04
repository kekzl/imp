#pragma once
// Device bytes of the MTP draft workspace, CUDA-free so the capacity plan (CPU lane) prices it
// before KV is sized. Mirrors mtp_workspace_allocate + mtp_carry_allocate + the post-norm feed
// scratch byte for byte; MtpDraftWorkspace::device_bytes is the allocator's own tally to check it.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace imp {

// Max top-W width the MTP draft step can emit per position (tree-ceiling
// measurement, Stage 0). The draft argmax is top-0.
constexpr int kMtpMaxTopW = 8;

// Max device-side chain length (per-chain stride of
// MtpDraftWorkspace::d_chain_tokens). Longer speculative.mtp_k values fall
// back to the host chain loop.
constexpr int kMtpMaxChainK = 16;

// Blocks in pass 1 of the two-pass serving top-W kernel.
constexpr int kMtpTopWBlocks = 64;

// Rows per batched prefill-feed pass (mtp_feed_batch). Bounds the batch
// scratch: ~61 MiB at Qwen3.8-27B dims (H=5120, d_ff=17408, 24 heads).
constexpr int kMtpFeedRows = 256;

// ws.d_scan_part: [num_heads, kMtpScanMaxSplits, head_dim + 2] FP32, kMtpScanPartOffset bytes into
// the d_mtp_position allocation.
constexpr int kMtpScanMaxSplits = 32;
constexpr size_t kMtpScanPartOffset = 256;
inline size_t mtp_scan_part_bytes(int num_heads, int head_dim) {
    return sizeof(float) * num_heads * kMtpScanMaxSplits * (static_cast<size_t>(head_dim) + 2);
}

// sizeof(block_q8_1) (compute/gemm.h, static_assert in mtp_forward.cu).
constexpr size_t kMtpQ8_1BlockBytes = 48;

// Every input mtp_workspace_allocate sizes from, plus the two scratches enable adds after it.
struct MtpWorkspaceShape {
    int hidden_dim = 0;
    int vocab_size = 0;
    int n_experts = 0;
    int top_k = 0;
    int expert_d_ff = 0;
    int shared_d_ff = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 0;  // head KV rows per slot
    int n_kv_slots = 1;
    int hc_count = 0;
    int hc_lowrank = 0;
    int carry_rows = 0;    // speculative.mtp_carry rows per KV slot (mtp_carry_allocate)
    int prenorm_rows = 0;  // diagnostics.mtp_prenorm_h feed scratch rows
};

// Device footprint of one cudaMalloc: above 1 MiB it takes whole 2 MiB pages. Qwen3.8-27B head: 344.5 MiB
// requested, 360 MiB measured (imp_vram_own_bytes boot - init_complete); with this rounding 359.5 MiB.
constexpr size_t kMtpAllocGranule = size_t{2} << 20;
inline size_t mtp_alloc_footprint(size_t bytes) {
    return bytes > (size_t{1} << 20) ? (bytes + kMtpAllocGranule - 1) / kMtpAllocGranule * kMtpAllocGranule
                                     : bytes;
}

namespace mtp_ws_detail {
constexpr size_t kH2 = 2;  // sizeof(__half)

// Sums buffers one allocation at a time, each at its device footprint.
struct Tally {
    size_t bytes = 0;
    void add(size_t b) { bytes += mtp_alloc_footprint(b); }
};

inline bool has_attn(const MtpWorkspaceShape& s) { return s.num_heads > 0 && s.head_dim > 0; }
inline bool has_kv(const MtpWorkspaceShape& s) {
    return has_attn(s) && s.num_kv_heads > 0 && s.max_seq_len > 0;
}
inline bool has_moe(const MtpWorkspaceShape& s) {
    return s.n_experts > 0 && s.top_k > 0 && s.expert_d_ff > 0;
}
inline size_t q_elems(const MtpWorkspaceShape& s) {
    return has_attn(s) ? static_cast<size_t>(s.num_heads) * s.head_dim : 0;
}
inline size_t kv_elems(const MtpWorkspaceShape& s) {
    return has_attn(s) ? static_cast<size_t>(s.num_kv_heads) * s.head_dim : 0;
}

// Phase 2.1 buffers, always allocated.
inline void base_bytes(const MtpWorkspaceShape& s, Tally& t) {
    const size_t H = static_cast<size_t>(s.hidden_dim);
    for (int i = 0; i < 5; ++i)  // emb_norm, h_norm, fc_out, h_final, h_final_snap
        t.add(H * kH2);
    t.add(2 * H * kH2);  // fc_in
    t.add(static_cast<size_t>(s.vocab_size) * kH2);
    t.add(static_cast<size_t>(s.vocab_size) * sizeof(float));
    t.add(((H + 31) / 32) * kMtpQ8_1BlockBytes);
    t.add(((H + 31) / 32) * sizeof(float));
    t.add(static_cast<size_t>(kMtpMaxTopW) * sizeof(int));
    t.add(static_cast<size_t>(kMtpTopWBlocks) * kMtpMaxTopW * sizeof(float));
    t.add(static_cast<size_t>(kMtpTopWBlocks) * kMtpMaxTopW * sizeof(int));
    t.add(static_cast<size_t>(kMtpMaxTopW) * sizeof(float));
    t.add(sizeof(int32_t) * kMtpMaxTopW * kMtpMaxChainK);
    t.add(sizeof(int));      // d_argmax
    t.add(sizeof(int32_t));  // d_tok
}

// Attention row buffers, scan partials and the head KV slots.
inline void attn_bytes(const MtpWorkspaceShape& s, Tally& t) {
    if (!has_attn(s))
        return;
    const size_t H = static_cast<size_t>(s.hidden_dim);
    t.add(H * kH2);               // input_norm
    t.add(2 * q_elems(s) * kH2);  // q_full (incl. gate half)
    t.add(q_elems(s) * kH2);      // q_attn
    if (s.num_kv_heads > 0) {
        t.add(kv_elems(s) * kH2);  // k_proj
        t.add(kv_elems(s) * kH2);  // v_proj
    }
    t.add(q_elems(s) * kH2);  // attn_out
    t.add(H * kH2);           // attn_residual
    t.add(kMtpScanPartOffset + mtp_scan_part_bytes(s.num_heads, s.head_dim));
    if (has_kv(s)) {
        const size_t slot_kv = static_cast<size_t>(s.max_seq_len) * kv_elems(s) * kH2;
        t.add(slot_kv * static_cast<size_t>(std::max(1, s.n_kv_slots)));  // K, all slots
        t.add(slot_kv * static_cast<size_t>(std::max(1, s.n_kv_slots)));  // V
    }
}

// MoE block, MLP in/out, hyper-connection stream, shared expert / dense MLP.
inline void mlp_bytes(const MtpWorkspaceShape& s, Tally& t) {
    const size_t H = static_cast<size_t>(s.hidden_dim);
    const size_t dff = static_cast<size_t>(std::max(0, s.shared_d_ff));
    if (has_moe(s)) {
        const size_t es = s.hc_count > 0 ? static_cast<size_t>(s.top_k) : 1;
        t.add(es * 2 * static_cast<size_t>(s.expert_d_ff) * kH2);  // expert gate|up
        t.add(es * static_cast<size_t>(s.expert_d_ff) * kH2);      // expert act
        t.add(static_cast<size_t>(s.top_k) * H * kH2);             // expert outputs
    }
    if (has_moe(s) || dff > 0) {
        t.add(H * kH2);  // post_norm
        t.add(H * kH2);  // moe_out
    }
    if (s.hc_count > 0 && s.hc_lowrank > 0) {
        const size_t hw = static_cast<size_t>(s.hc_count) * H * kH2;
        t.add(hw);
        t.add(hw);
        t.add(hw);
        t.add(static_cast<size_t>(s.hc_lowrank) * kH2);
        t.add(static_cast<size_t>(s.hc_count) * kH2);
    }
    if (dff > 0) {
        t.add(dff * kH2);  // shared gate
        t.add(dff * kH2);  // shared up
        t.add(dff * kH2);  // shared act
        t.add(H * kH2);    // shared out
    }
}

// Batched prefill feed (dense-MLP heads with attention + KV only) and the mtp_carry rows it enables.
inline void feed_bytes(const MtpWorkspaceShape& s, Tally& t) {
    const size_t dff = static_cast<size_t>(std::max(0, s.shared_d_ff));
    if (has_moe(s) || dff == 0 || !has_kv(s))
        return;
    const size_t H = static_cast<size_t>(s.hidden_dim);
    const size_t n = kMtpFeedRows;
    t.add(4 * n * sizeof(int32_t));
    const size_t rows[] = {H,
                           H,
                           2 * H,
                           H,
                           H,  // emb, h_norm, fc_in, fc_out, norm
                           2 * q_elems(s),
                           q_elems(s),
                           kv_elems(s),
                           kv_elems(s),
                           q_elems(s),  // q_full .. attn_out
                           H,
                           dff,
                           dff,
                           dff};  // res, gate, up, act
    for (size_t r : rows)
        t.add(n * r * kH2);
    if (s.carry_rows > 0 && s.carry_rows <= kMtpFeedRows && s.hc_count == 0)
        t.add(static_cast<size_t>(std::max(1, s.n_kv_slots)) * static_cast<size_t>(s.carry_rows) * H * kH2);
}
}  // namespace mtp_ws_detail

// Device bytes the head takes, at cudaMalloc footprint. Excludes the MoE routing pool (1 token,
// < 64 KiB) and pinned host.
inline size_t mtp_workspace_bytes(const MtpWorkspaceShape& s) {
    using namespace mtp_ws_detail;
    if (s.hidden_dim <= 0 || s.vocab_size <= 0)
        return 0;
    Tally t;
    base_bytes(s, t);
    attn_bytes(s, t);
    mlp_bytes(s, t);
    feed_bytes(s, t);
    if (s.prenorm_rows > 0 && s.hc_count == 0)
        t.add(static_cast<size_t>(s.prenorm_rows) * static_cast<size_t>(s.hidden_dim) * kH2);
    return t.bytes;
}

}  // namespace imp

#include "exec/gdn_ragged.h"

#include "compute/gdn.h"
#include "compute/ssm.h"
#include "exec/inference_state.h"
#include "memory/ssm_state.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace imp {

GdnRaggedRows gdn_ragged_rows(const InferenceState& state) {
    GdnRaggedRows g;
    g.n_seq = state.n_sequences;
    g.n_riders = state.n_riders;
    g.h_seq_offsets = state.h_seq_offsets;
    g.h_ssm_slots = state.h_ssm_slots;
    g.d_ssm_slots = state.ssm_seq_slots;
    if (g.n_riders < 0 || g.n_riders > g.n_seq || !g.h_seq_offsets || !g.h_ssm_slots)
        throw std::runtime_error(
            "gdn_ragged_rows: ragged state without host offsets/slots or bad rider count");
    if (g.n_riders > 0 && (g.h_seq_offsets[g.n_seq] - g.rider_row0() != g.n_riders || !g.d_ssm_slots))
        throw std::runtime_error("gdn_ragged_rows: a rider does not own exactly one row");
    return g;
}

void gdn_ragged_conv(const GdnRaggedRows& g, SSMState& st, int ssm_idx, const Tensor& xBC, const Tensor& w,
                     const Tensor& b, float* conv_f32, int conv_kernel, cudaStream_t stream) {
    const int channels = static_cast<int>(xBC.shape[1]);
    const size_t es = dtype_size(xBC.qtype);
    // Riders on the decode conv need F16 input (the decode kernel's type).
    const bool riders_batched = g.n_riders > 0 && xBC.qtype == QType::F16;
    const int n_loop = riders_batched ? g.n_prompt() : g.n_seq;
    for (int s = 0; s < n_loop; ++s) {
        const int rb = g.h_seq_offsets[s];
        const int ns = g.h_seq_offsets[s + 1] - rb;
        if (ns <= 0)
            continue;
        int64_t seq_shape[2] = {static_cast<int64_t>(ns), static_cast<int64_t>(channels)};
        Tensor xs(static_cast<char*>(xBC.data) + (static_cast<size_t>(rb) * channels * es), xBC.qtype, 2,
                  seq_shape, true);
        ssm_conv1d_prefill_f32_silu(st.conv_state(g.h_ssm_slots[s], ssm_idx), xs, w, b,
                                    conv_f32 + (static_cast<size_t>(rb) * channels), conv_kernel, stream);
    }
    if (!riders_batched)
        return;
    const size_t r0 = static_cast<size_t>(g.rider_row0());
    ssm_conv1d_decode_f32_silu_batched(st.conv_state(0, ssm_idx), g.d_ssm_slots + g.n_prompt(),
                                       static_cast<int64_t>(st.slot_stride_bytes() / sizeof(float)),
                                       static_cast<const half*>(xBC.data) + (r0 * channels), w, b,
                                       conv_f32 + (r0 * channels), g.n_riders, channels, conv_kernel, stream);
}

bool gdn_ragged_scan_split(const GdnRaggedRows& g, SSMState& st, int ssm_idx, const GdnRaggedScanIo& io,
                           cudaStream_t stream) {
    if (io.chunkpar_ws == nullptr || io.head_dim != 128 || io.state_size != 128)
        return false;
    int n_small = 0, n_big = 0;
    for (int s = 0; s < g.n_prompt(); ++s) {
        const int rows = g.h_seq_offsets[s + 1] - g.h_seq_offsets[s];
        if (rows >= 128)
            n_big++;
        else if (rows > 0)
            n_small++;
    }
    // Many short prompts are faster on the one batched launch (comment in run_gdn).
    if (n_big < 1 || n_small > 4)
        return false;

    const bool bf16 = st.h_dtype() == QType::BF16;
    const int C = io.conv_channels, H = io.n_heads, D = io.head_dim, S = io.state_size, G = io.n_groups;
    const int gl = io.grouped_layout;
    const size_t inner = static_cast<size_t>(H) * D;
    for (int s = 0; s < g.n_prompt(); ++s) {
        const int r0 = g.h_seq_offsets[s];
        const int rows = g.h_seq_offsets[s + 1] - r0;
        if (rows <= 0)
            continue;
        const float* conv_s = io.conv_f32 + (static_cast<size_t>(r0) * C);
        const half* alpha_s = io.alpha + (static_cast<size_t>(r0) * H);
        const half* beta_s = io.beta + (static_cast<size_t>(r0) * H);
        half* y_s = io.y + (static_cast<size_t>(r0) * inner);
        void* h_s = st.h_state(g.h_ssm_slots[s], ssm_idx);
        if (rows >= 128 && bf16)
            gdn_scan_chunkpar_bf16(conv_s, C, alpha_s, beta_s, io.A_log, io.dt_bias,
                                   static_cast<__nv_bfloat16*>(h_s), y_s, rows, H, D, S, G, stream, gl,
                                   io.chunkpar_ws, io.chunkpar_bytes, io.chunkpar_strip);
        else if (rows >= 128)
            gdn_scan_chunkpar_f32(conv_s, C, alpha_s, beta_s, io.A_log, io.dt_bias, static_cast<float*>(h_s),
                                  y_s, rows, H, D, S, G, stream, gl, io.chunkpar_ws, io.chunkpar_bytes,
                                  io.chunkpar_strip);
        else if (bf16)
            gdn_scan_fused_bf16(conv_s, C, alpha_s, beta_s, io.A_log, io.dt_bias,
                                static_cast<__nv_bfloat16*>(h_s), y_s, rows, H, D, S, G, stream, gl, nullptr);
        else
            gdn_scan_fused_f32(conv_s, C, alpha_s, beta_s, io.A_log, io.dt_bias, static_cast<float*>(h_s),
                               y_s, rows, H, D, S, G, stream, gl, nullptr);
    }
    if (g.n_riders == 0)
        return true;
    // Riders: the separate decode step's launch (one token per sequence, slot table offset past
    // the prompt members), so a mixed step decodes them with the same kernel.
    const size_t r0 = static_cast<size_t>(g.rider_row0());
    const int* slots = g.d_ssm_slots + g.n_prompt();
    if (bf16)
        gdn_scan_fused_bf16_batched(io.conv_f32 + (r0 * C), C, io.alpha + (r0 * H), io.beta + (r0 * H),
                                    io.A_log, io.dt_bias, static_cast<__nv_bfloat16*>(st.h_state(0, ssm_idx)),
                                    slots,
                                    static_cast<int64_t>(st.slot_stride_bytes() / sizeof(__nv_bfloat16)),
                                    io.y + (r0 * inner), g.n_riders, 1, H, D, S, G, stream, gl);
    else
        gdn_scan_fused_f32_batched(io.conv_f32 + (r0 * C), C, io.alpha + (r0 * H), io.beta + (r0 * H),
                                   io.A_log, io.dt_bias, static_cast<float*>(st.h_state(0, ssm_idx)), slots,
                                   static_cast<int64_t>(st.slot_stride_bytes() / sizeof(float)),
                                   io.y + (r0 * inner), g.n_riders, 1, H, D, S, G, stream, gl);
    return true;
}

}  // namespace imp

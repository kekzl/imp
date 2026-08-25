#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_debug.h"
#include "exec/gemm_context.h"
#include "compute/layernorm.h"
#include "compute/activation.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "core/logging.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace imp {

// Helper: FP32 ssm_out + FP16 residual → FP16 h. Preserves FP32 GEMM-accum
// precision through the residual add so downstream RMSNorm sees bit-accurate
// inputs (fixes Qwen 3.6 sign flips). Used when the gdn.fp32_out config flag (was IMP_GDN_FP32_OUT
// env) is set.
__global__ void fp32_plus_fp16_to_fp16(const float* __restrict__ a_fp32, const __half* __restrict__ b_fp16,
                                       __half* __restrict__ out_fp16, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = a_fp32[idx] + __half2float(b_fp16[idx]);
        out_fp16[idx] = __float2half(v);
    }
}

// Map global layer index to SSM/GDN state index.
static inline int get_ssm_layer(const std::vector<int>& ssm_layer_map, int layer) {
    return ssm_layer_map[layer];
}

// ---------------------------------------------------------------------------
// SSM (Mamba2) sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ssm(int layer, const InferenceState& state, cudaStream_t stream) {
    // Configure shared workspace for SSM phase
    configure_ssm_workspace(ws_.shared_max_tokens());

    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    int n = state.n_tokens;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = cfg.ssm_conv_channels();
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = inner / n_heads;

    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    // Producer fusion: quantize into the small-M scratch inside the norm
    // kernel when ssm_in will take that route (batched decode, CUTLASS_NVFP4
    // tier); falls back to plain rmsnorm. The n==1 packed-input path is
    // unaffected (producer gate requires n >= 2).
    rmsnorm_for_smallm_(h, ly.attn_norm, no, ly.ssm_in_id, n, eps, stream, norm_w_off_);

    // GemmContext for all weight GEMM dispatches in this function.
    //
    // cur_spec_verify_ is load-bearing here and was missing until 2026-08-18:
    // without it ctx.spec_verify_small_m stays false, the M<=4 batched-GEMV
    // branch (#998/#1055) is unreachable, and every GDN projection in a
    // speculative verify chunk takes the CUTLASS prefill path instead. FFN and
    // attention already threaded it; this path did not, and on a GDN hybrid it
    // is 48 of 64 layers, so the small-M verify optimisation never reached the
    // layers that dominate the architecture. Measured at 148 CUTLASS launches
    // and 4.26 ms per verify.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq, cur_spec_verify_);

    // 2. ssm_in projection: [n, d_model] @ ssm_in^T -> [n, ssm_in_dim]
    //    ssm_in_dim = inner(z) + conv_channels(xBC) + n_heads(dt)
    Tensor proj = view_tokens(ssm_proj_buf_, n);
    gemm_via_handle_(ly.ssm_in_id, no, proj, ctx);

    // 3. Split projection output [n, total_dim] into z, xBC, dt by column slices.
    //    proj layout: each row has [z(inner) | xBC(conv_channels) | dt(n_heads)].
    size_t es = dtype_size(compute_dtype_);
    int total_dim = inner + conv_channels + n_heads;

    Tensor z_buf, xBC_in, dt_buf;
    bool views_into_proj = (n == 1);

    if (views_into_proj) {
        // Decode (n=1): create views directly into proj — no copies needed.
        // Conv1d output is redirected to ssm_xBC_buf_ to preserve z/dt views.
        int64_t z_shape[2] = {1, static_cast<int64_t>(inner)};
        z_buf = Tensor(proj.data, compute_dtype_, 2, z_shape, true);

        char* xbc_ptr = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        int64_t xbc_shape[2] = {1, static_cast<int64_t>(conv_channels)};
        xBC_in = Tensor(xbc_ptr, compute_dtype_, 2, xbc_shape, true);

        char* dt_ptr = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        int64_t dt_shape2[2] = {1, static_cast<int64_t>(n_heads)};
        dt_buf = Tensor(dt_ptr, compute_dtype_, 2, dt_shape2, true);
    } else {
        // Prefill (n>1): strided column extraction via cudaMemcpy2DAsync.
        size_t src_pitch = static_cast<size_t>(total_dim) * es;

        z_buf = view_tokens(ssm_z_buf_, n);
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(z_buf.data, static_cast<size_t>(inner) * es, proj.data,
                                             src_pitch, static_cast<size_t>(inner) * es, n,
                                             cudaMemcpyDeviceToDevice, stream));

        xBC_in = view_tokens(ssm_xBC_buf_, n);
        char* xBC_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(xBC_in.data, static_cast<size_t>(conv_channels) * es, xBC_src,
                                             src_pitch, static_cast<size_t>(conv_channels) * es, n,
                                             cudaMemcpyDeviceToDevice, stream));

        dt_buf = view_tokens(ssm_dt_buf_, n);
        char* dt_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(dt_buf.data, static_cast<size_t>(n_heads) * es, dt_src,
                                             src_pitch, static_cast<size_t>(n_heads) * es, n,
                                             cudaMemcpyDeviceToDevice, stream));
    }

    // 4. Conv1d on xBC
    //    For decode with views_into_proj: output to ssm_xBC_buf_ (preserves z/dt in proj).
    //    For prefill: output to ssm_proj_buf_ (proj data already copied out).
    int64_t conv_out_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    void* conv_out_ptr = views_into_proj ? ssm_xBC_buf_.data : ssm_proj_buf_.data;
    Tensor xBC_out(conv_out_ptr, compute_dtype_, 2, conv_out_shape, true);

    int ssm_idx = get_ssm_layer(ssm_layer_map_, layer);
    void* conv_st = (state.ssm_state && ssm_idx >= 0) ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                                                      : nullptr;

    if (conv_st) {
        if (state.is_prefill) {
            // Speculative verify: also write the conv window as of d_snap_n
            // rows into the snapshot slab, so a fully rejected draft can adopt
            // the state after the chunk's first row instead of re-forwarding.
            // Same wiring the GDN path has had since #847 — without it the slab
            // stays uninitialised and engine_spec_ngram.cpp's matched == 0 fast
            // path commits garbage.
            void* conv_snap = (state.spec_snap_slab && state.ssm_state && ssm_idx >= 0)
                                  ? state.ssm_state->conv_state_in(state.spec_snap_slab, ssm_idx)
                                  : nullptr;
            const void* conv_prev = (conv_snap && state.spec_prev_slab)
                                        ? state.ssm_state->conv_state_in(state.spec_prev_slab, ssm_idx)
                                        : nullptr;
            ssm_conv1d_prefill(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b, xBC_out, conv_kernel,
                               stream, state.d_chunk_len, conv_prev ? conv_snap : nullptr,
                               conv_prev ? state.d_snap_n : nullptr, conv_prev);
        } else {
            ssm_conv1d_decode(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b, xBC_out, conv_kernel,
                              stream);
        }
    }

    // 5. SiLU on full conv output (x, B, and C together).
    //    Mamba2 applies SiLU to the ENTIRE conv1d output, not just x.
    //    This matches causal_conv1d_fn(..., activation="silu").
    silu_inplace(xBC_out, stream);

    // 6-7. Split conv output into x/B/C per token, run SSM scan.
    int BC_size = n_groups * ssize;
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0) ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
                                                   : nullptr;
    void* h_snap = (state.spec_snap_slab && state.ssm_state && ssm_idx >= 0)
                       ? state.ssm_state->h_state_in(state.spec_snap_slab, ssm_idx)
                       : nullptr;

    if (h_st) {
        // xBC_out layout: [n, conv_channels] where each row = [x(inner) | B(BC_size) | C(BC_size)]
        // We need contiguous [n, inner], [n, BC_size], [n, BC_size] for the fused scan.
        // Extract x, B, C from interleaved xBC_out into contiguous y_buf (x), and
        // reuse ssm_xBC_buf_ for B and C. However, ssm_y_buf_ is the output so
        // we need separate buffers. Instead, use cudaMemcpy2DAsync to de-interleave.
        //
        // Actually, the stride within xBC_out is conv_channels per row, while
        // ssm_scan_kernel expects stride = inner_size per row for x.
        // For n=1 decode this is just pointer arithmetic (no copy needed).
        // For n>1 prefill, we must de-interleave.

        QType h_dtype = (state.ssm_state) ? state.ssm_state->h_dtype() : QType::F32;

        if (n == 1) {
            // Decode: single token, just pass pointers directly into xBC_out row
            char* row = static_cast<char*>(xBC_out.data);
            int64_t x_shape[1] = {static_cast<int64_t>(inner)};
            Tensor x_t(row, compute_dtype_, 1, x_shape, true);

            int64_t bc_shape[1] = {static_cast<int64_t>(BC_size)};
            Tensor B_t(row + static_cast<size_t>(inner) * es, compute_dtype_, 1, bc_shape, true);
            Tensor C_t(row + static_cast<size_t>(inner + BC_size) * es, compute_dtype_, 1, bc_shape, true);

            int64_t dt_shape[1] = {static_cast<int64_t>(n_heads)};
            Tensor dt_t(dt_buf.data, compute_dtype_, 1, dt_shape, true);

            int64_t y_shape[1] = {static_cast<int64_t>(inner)};
            Tensor y_t(y_buf.data, compute_dtype_, 1, y_shape, true);

            ssm_scan_decode(x_t, B_t, C_t, dt_t, ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st, y_t,
                            static_cast<const half*>(z_buf.data), n_heads, head_dim_ssm, ssize, n_groups,
                            h_dtype, stream);
        } else {
            // Prefill: de-interleave x, B, C from xBC_out [n, conv_channels]
            // into contiguous buffers, then single fused kernel launch.
            // Reuse ssm_y_buf_ tail for temporary B/C storage.
            // ssm_y_buf_ is [max_tokens, inner] — we need [n, BC_size] for B and C.
            // B/C total = n * BC_size * 2 * es. inner >= BC_size for typical configs,
            // so y_buf has enough space. Alternatively, use ssm_xBC_buf_ (already [n, conv_channels]).
            //
            // Strategy: extract x into y_buf (will be overwritten by scan output after),
            // extract B into xBC_in (reusable since conv1d is done),
            // extract C into second half of xBC_in.

            // x: extract [n, inner] from xBC_out with src_pitch=conv_channels*es
            char* x_contig = static_cast<char*>(y_buf.data);  // temp, overwritten by scan
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(x_contig, static_cast<size_t>(inner) * es, xBC_out.data,
                                                 static_cast<size_t>(conv_channels) * es,
                                                 static_cast<size_t>(inner) * es, n, cudaMemcpyDeviceToDevice,
                                                 stream));

            // B: extract [n, BC_size] from offset inner in xBC_out
            char* B_contig = static_cast<char*>(xBC_in.data);  // conv1d done, safe to reuse
            char* B_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner) * es;
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(B_contig, static_cast<size_t>(BC_size) * es, B_src,
                                                 static_cast<size_t>(conv_channels) * es,
                                                 static_cast<size_t>(BC_size) * es, n,
                                                 cudaMemcpyDeviceToDevice, stream));

            // C: extract [n, BC_size] from offset inner+BC_size in xBC_out
            char* C_contig = B_contig + static_cast<size_t>(n) * BC_size * es;
            char* C_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner + BC_size) * es;
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(C_contig, static_cast<size_t>(BC_size) * es, C_src,
                                                 static_cast<size_t>(conv_channels) * es,
                                                 static_cast<size_t>(BC_size) * es, n,
                                                 cudaMemcpyDeviceToDevice, stream));

            // Build tensors for the fused scan
            int64_t x_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(inner)};
            Tensor x_all(x_contig, compute_dtype_, 2, x_shape, true);

            int64_t bc_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(BC_size)};
            Tensor B_all(B_contig, compute_dtype_, 2, bc_shape, true);
            Tensor C_all(C_contig, compute_dtype_, 2, bc_shape, true);

            int64_t dt_shape_all[2] = {static_cast<int64_t>(n), static_cast<int64_t>(n_heads)};
            Tensor dt_all(dt_buf.data, compute_dtype_, 2, dt_shape_all, true);

            // Output goes into y_buf (overwrites x_contig which was temporary)
            Tensor y_all(y_buf.data, compute_dtype_, 2, x_shape, true);

            ssm_scan_prefill(x_all, B_all, C_all, dt_all, ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st, y_all,
                             static_cast<const half*>(z_buf.data), n, n_heads, head_dim_ssm, ssize, n_groups,
                             h_dtype, stream, state.d_chunk_len, h_snap, h_snap ? state.d_snap_n : nullptr);
        }
    }

    // 8. Gating: y = y * SiLU(z) — fused into ssm_scan kernel above.

    // 9. Group RMSNorm on y  [AFTER gating, per llama.cpp reference]
    group_rmsnorm(y_buf, ly.ssm_norm_w, y_buf, n_groups, eps, stream);

    // 10. ssm_out projection: [n, inner] @ ssm_out^T -> [n, d_model]
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_via_handle_(ly.ssm_out_id, y_buf, out_buf, ctx);

    // 11. Residual add: hidden = output + residual
    elementwise_add(out_buf, r, stream);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) layer forward pass
// Same pipeline as Mamba2 SSM but with delta rule scan and separate gating.
// Pipeline: ssm_in(attn_qkv) → conv1d → SiLU → split(x/B/C) → delta_rule_scan
//           → gate(SiLU) → group_norm → ssm_out → residual
// ---------------------------------------------------------------------------

void GraphExecutor::run_gdn(int layer, const InferenceState& state, cudaStream_t stream) {
    configure_ssm_workspace(ws_.shared_max_tokens());

    // Padded verify chunk (#847): state.d_chunk_len carries the real chunk
    // length on device. The conv1d and delta-rule scan kernels below commit
    // conv tail + h_state at the real last row; pad rows only produce their
    // (causally discarded) y values. Same contract as run_ssm.

    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    int n = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = cfg.ssm_conv_channels();
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = (n_heads > 0) ? inner / n_heads : 0;

    auto ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq, cur_spec_verify_);

    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    // Producer fusion: quantize into the small-M scratch inside the norm
    // kernel when ssm_in will take that route (batched decode, CUTLASS_NVFP4
    // tier); falls back to plain rmsnorm. The n==1 packed-input path is
    // unaffected (producer gate requires n >= 2).
    rmsnorm_for_smallm_(h, ly.attn_norm, no, ly.ssm_in_id, n, eps, stream, norm_w_off_);

    // M=1 decode 4-way input fusion: when the load-time pack succeeded, run a
    // single GEMV against [conv_channels+inner+2*n_heads, d_model] and slice
    // the output for proj / gate_out / alpha / beta. Saves 3 gemv launches per
    // layer per decode step. Prefill (n>1) keeps the 4-call path unchanged so
    // we don't have to deinterleave the GEMM output.
    const bool fused_input = (n == 1) && (ly.gdn_input_packed.data != nullptr) &&
                             (gdn_fused_proj_buf_.data != nullptr);
    int packed_conv_channels = fused_input ? ly.gdn_packed_conv_channels : 0;
    int packed_inner = fused_input ? ly.gdn_packed_inner : 0;
    int packed_n_heads = fused_input ? ly.gdn_packed_n_heads : 0;

    // 2. ssm_in (attn_qkv) projection → [n, conv_channels]
    //    GDN: no z-split, no dt — the full projection goes to conv1d.
    int64_t proj_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    Tensor proj;
    if (fused_input) {
        // One GEMV produces [proj | gate | alpha | beta] contiguously in
        // gdn_fused_proj_buf_; the views below take offset slices.
        int total_out = packed_conv_channels + packed_inner + 2 * packed_n_heads;
        int64_t fused_shape[2] = {1, total_out};
        Tensor fused_out(gdn_fused_proj_buf_.data, compute_dtype_, 2, fused_shape, true);
        gemm_via_handle_(ly.gdn_input_packed_id, no, fused_out, ctx);
        proj = Tensor(gdn_fused_proj_buf_.data, compute_dtype_, 2, proj_shape, true);
    } else {
        // Original path: ssm_proj_buf_ is [max_tokens, ssm_in_dim] but we only
        // need [n, conv_channels].
        proj = Tensor(ssm_proj_buf_.data, compute_dtype_, 2, proj_shape, true);
        gemm_via_handle_(ly.ssm_in_id, no, proj, ctx);
    }
    dump_tensor_npy("gdn_ssm_in_out", proj, stream, layer, cur_decode_step_);

    // 3. Conv1d on full projection output [n, conv_channels]
    int64_t conv_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    Tensor xBC_in(proj.data, compute_dtype_, 2, conv_shape, true);
    Tensor xBC_out(ssm_xBC_buf_.data, compute_dtype_, 2, conv_shape, true);

    int ssm_idx = get_ssm_layer(ssm_layer_map_, layer);
    void* conv_st = (state.ssm_state && ssm_idx >= 0) ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                                                      : nullptr;

    // conv_f32 destination for FP32 pipeline (conv+SiLU output)
    float* conv_f32 = static_cast<float*>(ssm_proj_buf_.data);

    if (conv_st) {
        if (state.is_prefill) {
            // Fused: conv1d + SiLU + FP32 output in one kernel (saves 2 launches).
            // Copy FP16 input to xBC_out first to avoid aliasing (conv_f32 = ssm_proj_buf_ = xBC_in).
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                                static_cast<size_t>(n) * conv_channels * dtype_size(compute_dtype_),
                                cudaMemcpyDeviceToDevice, stream));
            // Speculative verify: also write the conv window as of d_snap_n
            // rows into the snapshot slab, so a partial acceptance landing on
            // that row can adopt the state instead of re-forwarding to it.
            void* conv_snap = (state.spec_snap_slab && state.ssm_state && ssm_idx >= 0)
                                  ? state.ssm_state->conv_state_in(state.spec_snap_slab, ssm_idx)
                                  : nullptr;
            const void* conv_prev = (conv_snap && state.spec_prev_slab)
                                        ? state.ssm_state->conv_state_in(state.spec_prev_slab, ssm_idx)
                                        : nullptr;
            ssm_conv1d_prefill_f32_silu(conv_st, xBC_out, ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
                                        conv_kernel, stream, state.d_chunk_len,
                                        conv_prev ? conv_snap : nullptr, conv_prev ? state.d_snap_n : nullptr,
                                        conv_prev);
        } else {
            // Decode: FP32 fused conv+SiLU (matching llama.cpp precision).
            // Copy FP16 input to xBC_out first to avoid aliasing: conv_f32
            // writes FP32 back into ssm_proj_buf_ which overlaps xBC_in.
            // n rows, not one: at batched decode the step carries one row per
            // sequence, and copying a single row left the other sequences
            // reading stale buffer contents (measured as coherent output for
            // sequence 0 and garbage for the rest). n == 1 on the
            // single-sequence path, so this is the same copy it always was.
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                                static_cast<size_t>(n) * conv_channels * dtype_size(compute_dtype_),
                                cudaMemcpyDeviceToDevice, stream));
            // Batched decode: ssm_n_seq sequences, one token each, each with
            // its own conv state. Falls through to the single-sequence call
            // when the step carries one sequence.
            if (state.ssm_seq_slots && state.ssm_n_seq > 1) {
                ssm_conv1d_decode_f32_silu_batched(
                    state.ssm_state->conv_state(0, ssm_idx), state.ssm_seq_slots,
                    static_cast<int64_t>(state.ssm_state->per_seq_bytes() / sizeof(float)),
                    static_cast<const half*>(xBC_out.data), ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
                    state.ssm_n_seq, conv_channels, conv_kernel, stream);
            } else {
                ssm_conv1d_decode_f32_silu(conv_st, xBC_out, ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
                                           conv_kernel, stream);
            }
        }
    } else {
        // Fallback: copy input to output + SiLU + FP32 conversion
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                            static_cast<size_t>(n) * conv_channels * dtype_size(compute_dtype_),
                            cudaMemcpyDeviceToDevice, stream));
        silu_inplace(xBC_out, stream);
        int64_t total = static_cast<int64_t>(n) * conv_channels;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(xBC_out.data), conv_f32,
                                                            total);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Per-element dump: post-conv1d post-SiLU FP32 scan input.
    // Compare to llama's `conv_output_silu-{layer}` tensor.
    {
        int64_t cf_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
        Tensor conv_f32_t(conv_f32, QType::F32, 2, cf_shape, true);
        dump_tensor_npy("gdn_conv_f32", conv_f32_t, stream, layer, cur_decode_step_);
    }

    // 4b. V-head reorder (tiled → grouped) for asymmetric-head GDN models.
    // The GGUF converter may store V in tiled layout (h0_g0, h1_g1, ... then
    // h0_g0_r1, ...) while the scan kernel reads V[h*HD+d] assuming grouped
    // layout. Qwen 3.6 uses 16K/32V asymmetric heads and triggers this
    // mismatch. Gated by the gdn.vhead_reorder config flag (was IMP_GDN_VHEAD_REORDER env) to avoid
    // regressing symmetric
    // models or models whose converters already emit grouped layout.
    const bool vhead_reorder = runtime_config().gdn.vhead_reorder;
    if (vhead_reorder && n_groups != n_heads) {
        const int inner_v = n_heads * head_dim_ssm;  // V channels = 32 * 128 = 4096 for Qwen 3.6
        const int BC_size_vh = n_groups * ssize;     // Q/K per-group size = 16 * 128 = 2048
        // Scratch buffers at tail of ssm_proj_buf_ (FP32, max_tokens * ssm_in_dim floats).
        // ssm_proj_buf_ already contains conv_f32 (n * conv_channels floats) at the head;
        // reserve two scratch regions past that.
        float* v_scratch_tiled = conv_f32 + static_cast<size_t>(n) * conv_channels;
        float* v_scratch_grouped = v_scratch_tiled + static_cast<size_t>(n) * inner_v;

        // Gather V slice from strided conv_f32 into contiguous [n, n_heads, head_dim_ssm].
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(v_scratch_tiled, static_cast<size_t>(inner_v) * sizeof(float),
                                             conv_f32 + 2 * BC_size_vh,
                                             static_cast<size_t>(conv_channels) * sizeof(float),
                                             static_cast<size_t>(inner_v) * sizeof(float), n,
                                             cudaMemcpyDeviceToDevice, stream));
        // Tiled → grouped reorder.
        vhead_tiled_to_grouped_f32(v_scratch_tiled, v_scratch_grouped, n, n_heads, head_dim_ssm, n_groups,
                                   stream);
        // Scatter reordered V back into conv_f32 V slice.
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(conv_f32 + 2 * BC_size_vh,
                                             static_cast<size_t>(conv_channels) * sizeof(float),
                                             v_scratch_grouped, static_cast<size_t>(inner_v) * sizeof(float),
                                             static_cast<size_t>(inner_v) * sizeof(float), n,
                                             cudaMemcpyDeviceToDevice, stream));
    }

    // 5. Run delta rule scan
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0) ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
                                                   : nullptr;
    void* h_snap = (state.spec_snap_slab && state.ssm_state && ssm_idx >= 0)
                       ? state.ssm_state->h_state_in(state.spec_snap_slab, ssm_idx)
                       : nullptr;

    // Gate projection — computed before scan, used after in RMSNormGated.
    // Fused-input path: the gate slice is already in gdn_fused_proj_buf_ from
    // the single up-front GEMV, no dispatch needed.
    int64_t gate_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(inner)};
    Tensor gate_out;
    if (fused_input) {
        size_t es_loc = dtype_size(compute_dtype_);
        char* gate_ptr =
            static_cast<char*>(gdn_fused_proj_buf_.data) + static_cast<size_t>(packed_conv_channels) * es_loc;
        gate_out = Tensor(gate_ptr, compute_dtype_, 2, gate_shape, true);
    } else {
        gate_out = Tensor(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);
        // ssm_in (above) quantized the same normed input into the activation
        // scratch and no GEMM dispatches in between — mark it shared (the
        // dispatch verifies against its scratch tag before skipping).
        GemmContext gate_ctx = ctx;
        if (n > 1 && prefill_routes_cutlass_nvfp4_(ly.ssm_in_id, n) &&
            prefill_routes_cutlass_nvfp4_(ly.gdn_gate_id, n))
            gate_ctx = ctx.with_act_quant_hint(no.data, n, static_cast<int>(no.shape[1]));
        gemm_via_handle_(ly.gdn_gate_id, no, gate_out, gate_ctx);
    }

    const bool use_fp32_scan = runtime_config().gdn.fp32_scan;

    if (h_st) {
        size_t es = dtype_size(compute_dtype_);

        // Compute alpha/beta projections from norm_out (input to this layer)
        Tensor alpha_proj_out, beta_proj_out;
        {
            int64_t ab_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(n_heads)};

            if (fused_input) {
                // Step 2: alpha and beta slices are at the tail of the fused
                // GEMV output. No dispatch, no copies.
                char* alpha_ptr = static_cast<char*>(gdn_fused_proj_buf_.data) +
                                  static_cast<size_t>(packed_conv_channels + packed_inner) * es;
                char* beta_ptr =
                    alpha_ptr + static_cast<size_t>(packed_n_heads) * es;
                alpha_proj_out = Tensor(alpha_ptr, compute_dtype_, 2, ab_shape, true);
                beta_proj_out = Tensor(beta_ptr, compute_dtype_, 2, ab_shape, true);
            } else if (n == 1 && ly.gdn_alpha_beta_packed.data) {
                // Step 1: alpha+beta-only fusion. One GEMV against [d_model,
                // 2*n_heads] produces [α | β] contiguous in ssm_dt_buf_.
                int64_t ab2_shape[2] = {1, static_cast<int64_t>(2 * n_heads)};
                Tensor ab_packed_out(ssm_dt_buf_.data, compute_dtype_, 2, ab2_shape, true);
                alpha_proj_out = Tensor(ssm_dt_buf_.data, compute_dtype_, 2, ab_shape, true);
                beta_proj_out = Tensor(static_cast<char*>(ssm_dt_buf_.data) + n_heads * es, compute_dtype_,
                                       2, ab_shape, true);
                gemm_via_handle_(ly.gdn_alpha_beta_packed_id, no, ab_packed_out, ctx);
            } else {
                // 4-call fallback: ssm_dt_buf_ for alpha, +offset for beta.
                alpha_proj_out = Tensor(ssm_dt_buf_.data, compute_dtype_, 2, ab_shape, true);
                char* beta_ptr = static_cast<char*>(ssm_dt_buf_.data) +
                                 ((static_cast<size_t>(n) * n_heads * es + 255) & ~size_t(255));
                beta_proj_out = Tensor(beta_ptr, compute_dtype_, 2, ab_shape, true);
                gemm_via_handle_(ly.gdn_alpha_id, no, alpha_proj_out, ctx);
                gemm_via_handle_(ly.gdn_beta_id, no, beta_proj_out, ctx);
            }
            // Per-element dump: pre-softplus alpha and pre-sigmoid beta projections.
            // Compare to llama's `alpha-{layer}` and `beta-{layer}`.
            dump_tensor_npy("gdn_alpha", alpha_proj_out, stream, layer, cur_decode_step_);
            dump_tensor_npy("gdn_beta", beta_proj_out, stream, layer, cur_decode_step_);
        }

        // 5b. Multi-token delta rule scan.
        // Default: fused kernel with register-cached state (n×32 fewer launches,
        // 125× less state memory traffic).
        // gdn.ref_kernel config flag (was IMP_GDN_REF env): reference kernel with shared-memory state, no
        // fusion.
        //   Use when debugging new architectures — if outputs differ it isolates
        //   bugs to the fused-kernel optimizations.
        // gdn.fp32_scan config flag (was IMP_GDN_FP32_SCAN env): keep scan output in FP32 through
        // RMSNorm+Gate.
        //   FP16 subnormal truncation (~6e-5) breaks RMS for near-zero heads on
        //   models with sparse scan activations (Qwen 3.6 L1 head 0).
        const bool use_ref = runtime_config().gdn.ref_kernel;
        const bool use_chunkwise = runtime_config().gdn.chunkwise_scan && !use_ref;
        if (use_fp32_scan) {
            // Layout in conv_f32 tail:
            //   [n*conv_channels)                : conv_f32 (done)
            //   [+n*inner)                       : y_fp32 scan output
            //   [+n*inner)                       : y_fp32_postnorm (rmsnorm_gated_silu output)
            // The post-norm FP32 buffer feeds directly into the ssm_out GEMM when
            // FP32_OUT is also active — bypasses the FP16 y_buf that would drop
            // ~3-4 bits of per-element precision (root cause of Qwen 3.6 L0 drift).
            float* y_fp32 = conv_f32 + static_cast<size_t>(n) * conv_channels;
            float* y_fp32_postnorm = y_fp32 + static_cast<size_t>(n) * n_heads * head_dim_ssm;
            const int gl = cfg.gdn_grouped_head_layout ? 1 : 0;
            if (use_chunkwise) {
                gdn_scan_chunkwise_fp32out(conv_f32, conv_channels,
                                           static_cast<const half*>(alpha_proj_out.data),
                                           static_cast<const half*>(beta_proj_out.data),
                                           static_cast<const float*>(ly.ssm_a.data),
                                           static_cast<const float*>(ly.ssm_dt_b.data),
                                           static_cast<float*>(h_st), y_fp32, n, n_heads, head_dim_ssm, ssize,
                                           n_groups, stream, /*chunk_size=*/64, gl, state.d_chunk_len,
                                           static_cast<float*>(h_snap), h_snap ? state.d_snap_n : nullptr);
            } else {
                gdn_scan_fused_fp32out(conv_f32, conv_channels, static_cast<const half*>(alpha_proj_out.data),
                                       static_cast<const half*>(beta_proj_out.data),
                                       static_cast<const float*>(ly.ssm_a.data),
                                       static_cast<const float*>(ly.ssm_dt_b.data), static_cast<float*>(h_st),
                                       y_fp32, n, n_heads, head_dim_ssm, ssize, n_groups, stream, gl,
                                       state.d_chunk_len, static_cast<float*>(h_snap),
                                       h_snap ? state.d_snap_n : nullptr);
            }
            // FP32-in, FP32-out RMSNorm+Gate+SiLU: preserves precision for the
            // ssm_out matmul. The FP32→FP16 copy below is REQUIRED, not optional
            // — the !use_fp32_out path of ssm_out reads y_buf as FP16 input. The
            // older code gated this on debug_forward_enabled() which left y_buf
            // uninitialized in production runs (= ssm_out fed zeros) and only
            // appeared coherent under the diagnostics.debug_forward config flag (was IMP_DEBUG_FORWARD env).
            gdn_rmsnorm_gated_silu_fp32inout(y_fp32_postnorm, y_fp32, static_cast<const half*>(gate_out.data),
                                             static_cast<const half*>(ly.ssm_norm_w.data), eps, n, n_heads,
                                             head_dim_ssm, stream);
            {
                int64_t total = static_cast<int64_t>(n) * n_heads * head_dim_ssm;
                int threads_ = 256;
                int blocks_ = static_cast<int>((total + threads_ - 1) / threads_);
                fp32_to_fp16_kernel<<<blocks_, threads_, 0, stream>>>(y_fp32_postnorm,
                                                                      static_cast<__half*>(y_buf.data),
                                                                      total);
                IMP_CUDA_CHECK_LAUNCH();
            }
        } else if (use_ref) {
            const int gl = cfg.gdn_grouped_head_layout ? 1 : 0;
            gdn_scan_reference_f32(conv_f32, conv_channels, static_cast<const half*>(alpha_proj_out.data),
                                   static_cast<const half*>(beta_proj_out.data),
                                   static_cast<const float*>(ly.ssm_a.data),
                                   static_cast<const float*>(ly.ssm_dt_b.data), static_cast<float*>(h_st),
                                   static_cast<half*>(y_buf.data), n, n_heads, head_dim_ssm, ssize, n_groups,
                                   stream, gl, state.d_chunk_len);
        } else if (state.ssm_seq_slots && state.ssm_n_seq > 1) {
            // Batched decode over independent sequences: n rows are n
            // sequences, one token each. The chunkwise path is a prefill
            // optimisation (it splits ONE sequence's timeline), so it does not
            // apply here — at one token per sequence there is nothing to chunk.
            const int gl = cfg.gdn_grouped_head_layout ? 1 : 0;
            gdn_scan_fused_f32_batched(
                conv_f32, conv_channels, static_cast<const half*>(alpha_proj_out.data),
                static_cast<const half*>(beta_proj_out.data), static_cast<const float*>(ly.ssm_a.data),
                static_cast<const float*>(ly.ssm_dt_b.data),
                static_cast<float*>(state.ssm_state->h_state(0, ssm_idx)), state.ssm_seq_slots,
                static_cast<int64_t>(state.ssm_state->per_seq_bytes() / sizeof(float)),
                static_cast<half*>(y_buf.data), state.ssm_n_seq, /*n_tokens=*/1, n_heads, head_dim_ssm,
                ssize, n_groups, stream, gl, /*d_real_n=*/nullptr);
        } else {
            const int gl = cfg.gdn_grouped_head_layout ? 1 : 0;
            if (use_chunkwise) {
                gdn_scan_chunkwise_f32(conv_f32, conv_channels,
                                       static_cast<const half*>(alpha_proj_out.data),
                                       static_cast<const half*>(beta_proj_out.data),
                                       static_cast<const float*>(ly.ssm_a.data),
                                       static_cast<const float*>(ly.ssm_dt_b.data), static_cast<float*>(h_st),
                                       static_cast<half*>(y_buf.data), n, n_heads, head_dim_ssm, ssize,
                                       n_groups, stream, /*chunk_size=*/64, gl, state.d_chunk_len);
            } else {
                gdn_scan_fused_f32(conv_f32, conv_channels, static_cast<const half*>(alpha_proj_out.data),
                                   static_cast<const half*>(beta_proj_out.data),
                                   static_cast<const float*>(ly.ssm_a.data),
                                   static_cast<const float*>(ly.ssm_dt_b.data), static_cast<float*>(h_st),
                                   static_cast<half*>(y_buf.data), n, n_heads, head_dim_ssm, ssize, n_groups,
                                   stream, gl, state.d_chunk_len);
            }
        }
    }

    debug_tensor_stats("gdn_after_scan_y", y_buf, stream);
    debug_tensor_stats("gdn_gate_out", gate_out, stream);

    // Per-element dump: raw scan output (FP16), pre-rmsnorm_gated_silu.
    // Only populated in the default path — FP32_SCAN writes to y_fp32 and only
    // copies back to y_buf when the diagnostics.debug_forward config flag (was IMP_DEBUG_FORWARD env)
    // is set. Compare to llama's
    // `attn_output-{layer}` from common-eval-callback.
    if (!use_fp32_scan) {
        dump_tensor_npy("gdn_y_post_scan", y_buf, stream, layer, cur_decode_step_);
    }

    // 6. Fused RMSNormGated + SiLU: y = rmsnorm(y) * silu(gate)
    // Single kernel launch for all tokens × heads (replaces n×32×2 launches).
    // When the gdn.fp32_scan config flag (was IMP_GDN_FP32_SCAN env) is active, this was already done
    // inline with the
    // scan above (FP32-input variant). Skip to avoid double-applying.
    // [gdn] norm_eps_override > 0 can override eps (diagnostic).
    float norm_eps = eps;
    if (runtime_config().gdn.norm_eps_override > 0.0f) {
        norm_eps = runtime_config().gdn.norm_eps_override;
    }
    if (!use_fp32_scan) {
        gdn_rmsnorm_gated_silu(static_cast<half*>(y_buf.data), static_cast<const half*>(gate_out.data),
                               static_cast<const half*>(ly.ssm_norm_w.data), norm_eps, n, n_heads,
                               head_dim_ssm, stream);
    }

    // Per-element dump: post-rmsnorm-gated scan output (FP16), pre-ssm_out GEMM.
    // Compare to llama's final_output tensor before build_lora_mm(ssm_out, ...).
    dump_tensor_npy("gdn_y_post_norm", y_buf, stream, layer, cur_decode_step_);

    // 7. ssm_out projection: [n, inner] → [n, d_model]
    // gdn.fp32_out config flag (was IMP_GDN_FP32_OUT env): compute the ssm_out GEMM with FP32 output
    // and add the
    // residual in FP32 before downcast. FP16 accumulation here drifts ~5% per
    // element vs llama.cpp; that small drift amplifies to sign flips in
    // downstream near-zero projections and breaks Qwen 3.6.
    const bool use_fp32_out = runtime_config().gdn.fp32_out;
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    if (use_fp32_out) {
        // Allocate FP32 scratch via cudaMallocAsync (small: n * d_model * 4 bytes)
        size_t fp32_bytes = static_cast<size_t>(n) * cfg.d_model * sizeof(float);
        void* fp32_out = nullptr;
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&fp32_out, fp32_bytes, stream));
        int64_t out_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(cfg.d_model)};
        Tensor fp32_out_t(fp32_out, QType::F32, 2, out_shape, true);

        // NOTE: when FP32_SCAN is also active, the post-norm-gated tensor lives
        // in FP32 memory, but we can't pass it directly to gemm_dispatch because
        // cuBLAS FP32-input × FP16-cached-weight path silently produces zeros on
        // sm_120 with CUBLAS_COMPUTE_32F (tested 2026-04-21). Stay on FP16 y_buf
        // input; precision benefit from FP32 scan is limited to the rmsnorm stage.
        gemm_via_handle_(ly.ssm_out_id, y_buf, fp32_out_t, ctx);

        // FP16 residual → FP32 + add + FP16 writeback to h in one kernel.
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp32_plus_fp16_to_fp16<<<blocks, threads, 0, stream>>>(static_cast<const float*>(fp32_out),
                                                               static_cast<const __half*>(r.data),
                                                               static_cast<__half*>(h.data), total);
        IMP_CUDA_CHECK_LAUNCH();
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(fp32_out, stream));
    } else {
        gemm_via_handle_(ly.ssm_out_id, y_buf, out_buf, ctx);
        // Per-element dump: linear_attn_out post-ssm_out GEMM, pre-residual.
        // Compare to llama's `linear_attn_out-{layer}` from eval-callback.
        dump_tensor_npy("gdn_linear_attn_out", out_buf, stream, layer, cur_decode_step_);
        elementwise_add(out_buf, r, stream);
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }
}

}  // namespace imp

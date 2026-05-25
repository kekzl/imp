#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/executor_debug.h"
#include "exec/gemm_context.h"
#include "runtime/config.h"
#include <cstdio>
#include <stdexcept>
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "memory/kv_cache_manager.h"
#include "exec/executor_kernels.h"
#include "memory/gdn_state.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/hadamard.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace imp {

// ---------------------------------------------------------------------------
// Quant type dispatch helpers
// ---------------------------------------------------------------------------
// Shared helpers: executor_helpers.h (get_kv_layer, vram_alloc, etc.)
// Layer methods: executor_attention.cu, executor_ffn.cu, executor_ssm_gdn.cu

// LM head dp4a GEMV dispatch: y = W @ x (FP32 output for logits).
static void dispatch_gemv_fp32(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, float* y,
                               int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_q6k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_0:
            gemv_q4_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q2_K:
            gemv_q2_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q3_K:
            gemv_q3_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        default:
            gemv_q8_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
    }
}

// Gemma 4: per-layer output scale. Multiplies all elements of `data` by the
// scalar half stored at `scale_ptr` (device memory). Used at end of each layer
// to keep the residual stream bounded.
__global__ __launch_bounds__(256) void scale_fp16_by_fp16ptr_kernel(half* __restrict__ data,
                                                                    const half* __restrict__ scale_ptr,
                                                                    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx < n2) {
        half2 s2 = __half2half2(*scale_ptr);
        half2* d2 = reinterpret_cast<half2*>(data);
        d2[idx] = __hmul2(d2[idx], s2);
    }
}

__global__ __launch_bounds__(256) void scale_fp32_by_fp16ptr_kernel(float* __restrict__ data,
                                                                    const half* __restrict__ scale_ptr,
                                                                    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = __half2float(*scale_ptr);
        data[idx] *= s;
    }
}

// write_kv_cache: moved to executor_kv_write.cu

// ---------------------------------------------------------------------------
// Forward pass diagnostics (IMP_DEBUG_FORWARD=1)
// ---------------------------------------------------------------------------

// debug_forward_enabled, debug_tensor_stats: shared via executor_debug.h

// Print top-k logits with token IDs
void debug_top_logits(const Tensor& logits, cudaStream_t stream, int topk = 10) {
    if (!debug_forward_enabled())
        return;
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    int nrows = (logits.ndim >= 2) ? static_cast<int>(logits.shape[0]) : 1;
    // Dump the LAST row (the one that actually gets sampled from for the next token).
    int row = nrows - 1;
    int64_t row_stride = (logits.ndim >= 2 && logits.stride[0] > 0) ? logits.stride[0] : vocab;
    const size_t elem_sz = (logits.qtype == QType::F32) ? sizeof(float) : sizeof(half);
    const char* src = static_cast<const char*>(logits.data) + (int64_t)row * row_stride * elem_sz;
    std::vector<float> host(vocab);

    if (logits.qtype == QType::F32) {
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(host.data(), src, vocab * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else if (logits.qtype == QType::F16) {
        std::vector<half> tmp(vocab);
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(tmp.data(), src, vocab * sizeof(half), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        for (int i = 0; i < vocab; i++)
            host[i] = __half2float(tmp[i]);
    }
    cudaStreamSynchronize(stream);
    // Also print min/max/L2 over the dumped row for cross-impl comparison.
    float mn = host[0], mx = host[0];
    double ss = 0.0;
    for (int i = 0; i < vocab; i++) {
        mn = std::min(mn, host[i]);
        mx = std::max(mx, host[i]);
        ss += (double)host[i] * host[i];
    }
    fprintf(stderr, "[DEBUG_FWD] logits row=%d/%d  min=%+.4f max=%+.4f L2=%.4f\n", row, nrows, mn, mx,
            std::sqrt(ss));

    // Find top-k by partial sort
    std::vector<std::pair<float, int>> scored(vocab);
    for (int i = 0; i < vocab; i++)
        scored[i] = {host[i], i};
    std::partial_sort(scored.begin(), scored.begin() + std::min(topk, vocab), scored.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    fprintf(stderr, "[DEBUG_FWD] Top-%d logits:\n", topk);
    for (int i = 0; i < std::min(topk, vocab); i++) {
        fprintf(stderr, "  [%2d] token_id=%6d  logit=%+.6f\n", i, scored[i].second, scored[i].first);
    }
}

// run_attention: moved to executor_attention.cu

// run_ffn: moved to executor_ffn.cu

// run_ssm, run_gdn: moved to executor_ssm_gdn.cu

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

void GraphExecutor::forward_logits(const InferenceState& state, Tensor& logits_out, cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("GraphExecutor::forward_logits called before init()");
        return;
    }

    const auto& cfg = model_->config();
    int n = state.n_tokens;
    if (n <= 0) {
        IMP_LOG_ERROR("n_tokens must be positive, got %d", n);
        return;
    }
    if (n > max_tokens_) {
        // Throwing here lets the BatchingEngine's try/catch cancel the
        // request with HTTP 500 instead of silently returning an
        // uninitialized logits tensor that the caller then reshapes
        // (→ `terminate: reshape: numel mismatch` on the worker thread,
        // which used to kill the entire imp-server container).
        char msg[128];
        snprintf(msg, sizeof(msg), "GraphExecutor::forward_logits: n_tokens (%d) exceeds max_tokens (%d)", n,
                 max_tokens_);
        IMP_LOG_ERROR("%s", msg);
        throw std::invalid_argument(msg);
    }

    // Store for use by run_ffn (which doesn't receive the InferenceState).
    cur_n_tokens_ = n;
    // Decode step counter for debug dump tagging. Shared with GDN path via
    // debug_decode_step() so run_gdn can tag its dumps with the same step.
    int& s_decode_step = debug_decode_step();
    if (n == 1)
        s_decode_step++;
    const int decode_step = (n == 1) ? s_decode_step : 0;
    cur_decode_step_ = decode_step;
    cur_force_fp16_ = state.force_fp16_gemm;
    cur_per_row_lm_ = state.per_row_lm_head;

    // Clear any stale CUDA error state before starting the forward pass.
    {
        cudaError_t e_ = cudaGetLastError();
        if (e_ != cudaSuccess)
            IMP_LOG_WARN("Cleared stale error before forward: %s", cudaGetErrorString(e_));
    }

    // ---- Optional per-component profiling (IMP_PROFILE=1) ----
    // Profiling disables CUDA graph capture (they are incompatible).
    // Use IMP_PROFILE=1 for diagnostic runs only.
    const bool do_profile = runtime_config().diagnostics.profile;
    static int profile_step_ = 0;
    static float acc_total = 0, acc_attn = 0, acc_ffn = 0, acc_lm = 0;
    bool profiling = do_profile;
    int profile_idx = profiling ? profile_step_++ : 0;
    // Skip first 2 decode steps (warmup / graph capture attempt)
    bool profile_active = profiling && (profile_idx >= 2);

    // RAII guard: ensures cudaEventDestroy is called even on early return.
    struct ProfileEvents {
        cudaEvent_t ev_start = nullptr, ev_emb = nullptr, ev_lm = nullptr;
        std::vector<cudaEvent_t> ev_attn, ev_ffn;
        bool active = false;
        ~ProfileEvents() {
            if (!active)
                return;
            if (ev_start)
                cudaEventDestroy(ev_start);
            if (ev_emb)
                cudaEventDestroy(ev_emb);
            if (ev_lm)
                cudaEventDestroy(ev_lm);
            for (auto e : ev_attn)
                if (e)
                    cudaEventDestroy(e);
            for (auto e : ev_ffn)
                if (e)
                    cudaEventDestroy(e);
        }
    } prof;
    // Alias references for minimal churn in the rest of the function.
    auto& ev_start = prof.ev_start;
    auto& ev_emb = prof.ev_emb;
    auto& ev_lm = prof.ev_lm;
    auto& ev_attn = prof.ev_attn;
    auto& ev_ffn = prof.ev_ffn;
    if (profile_active) {
        prof.active = true;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_emb);
        cudaEventCreate(&ev_lm);
        ev_attn.resize(cfg.n_layers);
        ev_ffn.resize(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; i++) {
            cudaEventCreate(&ev_attn[i]);
            cudaEventCreate(&ev_ffn[i]);
        }
        cudaEventRecord(ev_start, stream);
    }

    // All member tensors are [max_tokens_, cols]. view_tokens creates [n, cols]
    // views on the fly without modifying the members.

    // ---- Step 1: Embedding lookup ----
    //    For Q8_0/Q6_K embeddings, dequantizes only the needed rows on the fly.
    if (debug_forward_enabled()) {
        std::vector<int32_t> h_ids(n);
        IMP_CUDA_CHECK_LOG(
            cudaMemcpy(h_ids.data(), state.token_ids, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[DEBUG_FWD] [step=%d] input_tokens (%d):", decode_step, n);
        for (int i = 0; i < n; i++)
            fprintf(stderr, " %d", h_ids[i]);
        fprintf(stderr, "\n");
        // Dump positions
        std::vector<int> h_pos(n);
        IMP_CUDA_CHECK_LOG(
            cudaMemcpy(h_pos.data(), state.positions, n * sizeof(int), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[DEBUG_FWD] [step=%d] positions (%d):", decode_step, n);
        for (int i = 0; i < std::min(n, 30); i++)
            fprintf(stderr, " %d", h_pos[i]);
        fprintf(stderr, "\n");
    }
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup(model_->token_embedding(), state.token_ids, n, h, model_->tok_emb_.qtype, stream);

    // Gemma: scale embeddings by sqrt(d_model)
    if (cfg.embed_scale > 0.0f && h.qtype == QType::F16) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(h.data),
                                                          __float2half(cfg.embed_scale), total);
    }

    // Replace vision token positions with vision embeddings (multimodal)
    if (state.vision_embeddings && state.vision_token_id >= 0 && state.n_vision_tokens > 0) {
        // Declared in vision/vision_encoder.cu
        extern void launch_replace_vision_embeddings(half * hidden, const int32_t* token_ids,
                                                     const half* vision_emb, int vision_token_id,
                                                     int n_tokens, int d_model, int n_vision_tokens,
                                                     cudaStream_t stream);
        launch_replace_vision_embeddings(static_cast<half*>(h.data), state.token_ids, state.vision_embeddings,
                                         state.vision_token_id, n, cfg.d_model, state.n_vision_tokens,
                                         stream);
    }

    debug_tensor_stats("after_embedding", h, stream);
    debug_tensor_stats_all("after_embedding_all", view_tokens(h, n), stream);

    // Initialize FP32 residual accumulator from FP16 embedding (post-norm models only).
    if (fp32_accum_buf_) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(h.data), static_cast<float*>(view_tokens(fp32_hidden_, n).data), total);
    }

    // Dump FP32 accumulator for decode debugging
    if (fp32_accum_buf_ && n == 1 && debug_forward_enabled()) {
        float tmp[4];
        cudaMemcpyAsync(tmp, view_tokens(fp32_hidden_, n).data, 4 * sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
        fprintf(stderr, "[DEBUG_FWD] [step=%d] fp32_accum_init: [%.4f %.4f %.4f %.4f]\n", decode_step, tmp[0],
                tmp[1], tmp[2], tmp[3]);
    }
    // Binary dump: write the full FP16 hidden state to file
    if (!runtime_config().diagnostics.dump_hidden_dir.empty()) {
        std::vector<half> h_buf(n * cfg.d_model);
        cudaMemcpy(h_buf.data(), h.data, h_buf.size() * sizeof(half), cudaMemcpyDeviceToHost);
        char fname[256];
        snprintf(fname, sizeof(fname), "/tmp/imp_embed_step%d.bin", decode_step);
        FILE* f = fopen(fname, "wb");
        if (f) {
            fwrite(h_buf.data(), sizeof(half), h_buf.size(), f);
            fclose(f);
        }
        fprintf(stderr, "[DUMP_BIN] Wrote %s (%zu halfs)\n", fname, h_buf.size());
    }

    if (profile_active)
        cudaEventRecord(ev_emb, stream);

    // ---- Step 2: Transformer/Hybrid layers ----
    int max_layer = (state.exit_layer > 0) ? std::min(state.exit_layer, cfg.n_layers) : cfg.n_layers;
    // [diagnostics] exit_layer = N runs only N layers (-1 = full forward).
    {
        const int s_exit = runtime_config().diagnostics.exit_layer;
        if (s_exit > 0)
            max_layer = std::min(max_layer, s_exit);
    }
    const int skip_start = state.skip_layer_start;
    const int skip_end = state.skip_layer_end;
    for (int i = 0; i < max_layer; ++i) {
        // Layer skipping: skip layers in [skip_start, skip_end)
        if (skip_start >= 0 && skip_end > skip_start && i >= skip_start && i < skip_end)
            continue;
        // Layer offloading: ensure weights are on GPU, prefetch next layer
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) {
                offload_mgr_->prefetch_layer(i + 1);
            }
        }

        // Layer-diff dump: Snapshot A — pre-attention layer input.
        dump_tensor_npy("A_pre_attn", view_tokens(h, n), stream, i, decode_step);

        // Attention, GDN, or SSM (mutually exclusive per layer).
        // GDN check first: GDN layers have ssm_in (from attn_qkv) but use delta rule.
        if (layer_has_gdn(i)) {
            run_gdn(i, state, stream);
        } else if (layer_has_ssm(i)) {
            run_ssm(i, state, stream);
        } else if (layer_has_attention(i)) {
            run_attention(i, state, stream);
        } else if (layer_has_ssm(i)) {
            run_ssm(i, state, stream);
        }
        // Layer-diff dump: Snapshot B — post-attention residual-added state (input to FFN).
        dump_tensor_npy("B_post_attn", view_tokens(h, n), stream, i, decode_step);
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[step=%d] after_layer%02d_%s", decode_step, i,
                     layer_has_gdn(i)         ? "gdn"
                     : layer_has_attention(i) ? "attn"
                                              : "ssm");
            debug_tensor_stats_all(buf, view_tokens(h, n), stream);
            const bool dump_this_layer = debug_forward_enabled();
            if (dump_this_layer) {
                char rbuf[64];
                snprintf(rbuf, sizeof(rbuf), "[step=%d] attn_out-%d", decode_step, i);
                debug_tensor_rows(rbuf, view_tokens(h, n), stream);
            }
        }
        if (profile_active)
            cudaEventRecord(ev_attn[i], stream);

        // FFN: MoE, dense, or none (attention-only layers may have no FFN)
        const bool skip_moe = runtime_config().moe.skip;
        if (skip_moe) {
            // Debug: skip all FFN/MoE to isolate attention bugs
        } else if (layer_has_moe(i)) {
            run_moe_ffn(i, stream);
        } else if (layer_has_dense_ffn(i)) {
            run_ffn(i, stream);
        }
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[step=%d] after_layer%02d_%s", decode_step, i,
                     layer_has_moe(i) ? "moe" : (layer_has_dense_ffn(i) ? "ffn" : "no_ffn"));
            debug_tensor_stats_all(buf, view_tokens(h, n), stream);
            const bool dump_this_layer = debug_forward_enabled();
            if (dump_this_layer) {
                char rbuf[64];
                snprintf(rbuf, sizeof(rbuf), "[step=%d] moe_out-%d", decode_step, i);
                debug_tensor_rows(rbuf, view_tokens(h, n), stream);
            }
        }

        // Gemma 4: per-layer output scale (a scalar weight). llama.cpp's
        // gemma4-iswa.cpp:215-218 multiplies the layer output by this BEFORE the
        // next layer reads it. Without it, the residual stream grows unbounded.
        {
            const auto& ly = model_->layer(i);
            if (ly.layer_out_scale.data != nullptr && ly.layer_out_scale.on_device && h.qtype == QType::F16) {
                int64_t total = static_cast<int64_t>(n) * cfg.d_model;
                int threads = 256;
                int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
                scale_fp16_by_fp16ptr_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<half*>(h.data), static_cast<const half*>(ly.layer_out_scale.data), total);
                // Also scale the FP32 residual accumulator so next layer's
                // attention sees the correctly-scaled residual stream.
                // Without this the FP32 accum grows unbounded (layer_out_scale
                // ~0.1-0.2 compensates residual growth per llama's gemma4-iswa).
                if (fp32_accum_buf_ && cfg.arch == ModelArch::GEMMA4) {
                    int blocks_f32 = static_cast<int>((total + threads - 1) / threads);
                    scale_fp32_by_fp16ptr_kernel<<<blocks_f32, threads, 0, stream>>>(
                        static_cast<float*>(view_tokens(fp32_hidden_, n).data),
                        static_cast<const half*>(ly.layer_out_scale.data), total);
                }
                if (debug_forward_enabled()) {
                    float sval = 0.0f;
                    half h_scale;
                    cudaMemcpyAsync(&h_scale, ly.layer_out_scale.data, sizeof(half), cudaMemcpyDeviceToHost,
                                    stream);
                    cudaStreamSynchronize(stream);
                    sval = __half2float(h_scale);
                    if (i == 0 || i == 29)
                        fprintf(stderr, "[DEBUG_FWD] L%d_out_scale = %.6f\n", i, sval);
                    // Dump FP16 hidden after scale for all layers (decode only)
                    if (n == 1 && debug_forward_enabled()) {
                        half h_tmp[8];
                        cudaMemcpyAsync(h_tmp, view_tokens(h, n).data, 8 * sizeof(half),
                                        cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                        fprintf(stderr, "[DUMP] step=%d L%02d h=[%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n",
                                decode_step, i, __half2float(h_tmp[0]), __half2float(h_tmp[1]),
                                __half2float(h_tmp[2]), __half2float(h_tmp[3]), __half2float(h_tmp[4]),
                                __half2float(h_tmp[5]), __half2float(h_tmp[6]), __half2float(h_tmp[7]));
                    }
                    // Binary dump: full hidden state for selected layers.
                    // [diagnostics] dump_hidden_dir = "<path>"  → layers 0/5/15/29.
                    // [diagnostics] dump_hidden_dir = "all"     → every layer.
                    const std::string& dh = runtime_config().diagnostics.dump_hidden_dir;
                    if (!dh.empty()) {
                        const bool dump_all = (dh == "all");
                        bool sel = dump_all || (i == 0 || i == 5 || i == 15 || i == 29);
                        if (sel) {
                            std::vector<half> h_buf(n * cfg.d_model);
                            cudaMemcpy(h_buf.data(), view_tokens(h, n).data, h_buf.size() * sizeof(half),
                                       cudaMemcpyDeviceToHost);
                            char fname[256];
                            snprintf(fname, sizeof(fname), "/tmp/imp_L%02d_step%d.bin", i, decode_step);
                            FILE* f = fopen(fname, "wb");
                            if (f) {
                                fwrite(h_buf.data(), sizeof(half), h_buf.size(), f);
                                fclose(f);
                            }
                        }
                    }
                }
            }
            if (debug_forward_enabled()) {
                char buf[64];
                snprintf(buf, sizeof(buf), "[step=%d] L%02d_after_out_scale", decode_step, i);
                debug_tensor_stats_all(buf, view_tokens(h, n), stream);
                {
                    char rbuf[64];
                    snprintf(rbuf, sizeof(rbuf), "[step=%d] l_out-%d", decode_step, i);
                    debug_tensor_rows(rbuf, view_tokens(h, n), stream);
                }
            }
        }

        // Gemma-4 FP32 residual sync: run_moe_ffn now updates fp32_hidden_ itself
        // via the rmsnorm_fp32_accum_to_fp16_kernel path when post_ffn_norm is
        // present and fp32_accum_buf_ is active. The forced FP16→FP32 sync here
        // would clobber the FP32 precision and cause ~1-2% drift per layer.
        // Only sync when the MoE path did NOT go through the FP32 accum kernel
        // (e.g. layer has no post_ffn_norm or residual was fused into decode path).
        if (fp32_accum_buf_ && cfg.arch == ModelArch::GEMMA4 &&
            runtime_config().moe.force_fp16_sync) {
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            int64_t total = static_cast<int64_t>(n) * cfg.d_model;
            int threads = 256;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(h.data),
                                                                static_cast<float*>(fp32_h.data), total);
        }

        // Layer-diff dump: Snapshot C — end-of-layer state (input to next layer).
        // Captures the final hidden state after attention, FFN/MoE, residuals,
        // layer_out_scale (Gemma-4), and any FP16→FP32 sync.
        dump_tensor_npy("C_post_layer", view_tokens(h, n), stream, i, decode_step);
        // Also dump the FP32 shadow so we can diff FP32-truth vs FP16-view vs llama.cpp.
        if (fp32_accum_buf_) {
            dump_tensor_npy("C_fp32_shadow", view_tokens(fp32_hidden_, n), stream, i, decode_step);
        }

        if (i == max_layer - 1) {
            debug_tensor_stats("after_last_layer", h, stream);
        }
        if (profile_active)
            cudaEventRecord(ev_ffn[i], stream);

        // Release offloaded layer (restore host pointers)
        if (offload_mgr_) {
            offload_mgr_->release_layer(i);
        }
    }

    // BitDecoding Phase 3: advance the residual ring state once per decode
    // step. Has to happen INSIDE the captured graph (otherwise replays would
    // reuse the captured-time write_idx). Must come AFTER the layer loop —
    // each layer reads the same write_idx for its KV write/attention so the
    // bump applies starting next step. Decode-only (n==1 or n==n_sequences).
    if (!state.is_prefill && state.kv_manager != nullptr &&
        state.kv_manager->residual_enabled() && state.kv_seq_id >= 0) {
        int slot = state.kv_manager->residual_slot_of(state.kv_seq_id);
        if (slot >= 0) {
            advance_residual_state_kernel<<<1, 1, 0, stream>>>(
                state.kv_manager->d_residual_widx_ptr(),
                state.kv_manager->d_residual_fc_ptr(),
                slot,
                state.kv_manager->residual_n_tokens());
        }
    }

    // Final FP32→FP16 conversion for the tokens that need LM head projection.
    // run_attention/run_ffn already keep hidden_ in sync with fp32_hidden_,
    // but this ensures the final state is clean (no stale data from earlier layers).
    if (debug_forward_enabled() && fp32_accum_buf_) {
        // Compare FP32 accum vs FP16 hidden for last token before final conversion
        Tensor h_view = view_tokens(hidden_, n);
        Tensor fp32_view = view_tokens(fp32_hidden_, n);
        debug_tensor_stats("pre_final_conv_fp16", h_view, stream);
        debug_tensor_stats("pre_final_conv_fp32", fp32_view, stream);
    }
    if (fp32_accum_buf_) {
        fp32_to_fp16_rowscale_kernel<<<n, 256, 256 * sizeof(float), stream>>>(
            static_cast<const float*>(view_tokens(fp32_hidden_, n).data), static_cast<half*>(h.data), n,
            cfg.d_model);
    }

    // ---- Step 3+4: Final RMSNorm + LM head projection ----
    // Only project the tokens that actually need sampling:
    //   Prefill: last token only (all others just populate KV cache)
    //   Decode:  all tokens (one per sequence)
    //
    // For raw Q6_K/Q8_0 output projection with single token (n=1 or prefill last):
    // use fused RMSNorm→Q8_1 + dp4a GEMV with FP32 output. Saves ~2.45x VRAM
    // bandwidth vs cuBLAS FP16 path (reads quantized weights directly).
    const auto out_qtype = model_->out_proj_.qtype;
    const bool use_dp4a_lm = qscratch_.q8_1_buf && compute_dtype_ == QType::F16 && is_dp4a_qtype(out_qtype) &&
                             !runtime_config().gemm.no_dp4a_lm;

    // GemmContext for LM head GEMM dispatches.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq);

    // Registry handle for LM head — replaces wcache_ probe per call (Task 3.5).
    const WeightHandle* lm_h = (model_->out_proj_id != kInvalidTensorID)
                                   ? &registry_.handle(model_->out_proj_id)
                                   : nullptr;
    const StorageTier lm_tier = lm_h ? lm_h->primary_tier : StorageTier::Undefined;
    // Phase-3 secondary NVFP4 decode cache (Q8_0/Q6_K/Q5_K source LM head).
    // c8763ad refactor lost this fallback; restore by also probing wcache_.nvfp4.
    auto lm_nvfp4_it = wcache_.nvfp4.find(model_->output_proj().data);
    const bool lm_nvfp4_secondary = (lm_nvfp4_it != wcache_.nvfp4.end());
    const bool lm_has_fp8 = (wcache_.fp8.count(model_->output_proj().data) != 0);
    const bool lm_is_nvfp4 = !lm_has_fp8 && ((lm_tier == StorageTier::NVFP4) || lm_nvfp4_secondary);

    // L2 streaming hint for the LM head projection (QW3 from review/phase5_synthesis.md §2.1):
    // output_proj is huge (vocab_size × d_model — ~780 MiB for Qwen3-8B Q8_0) and
    // touched exactly once per forward, so it pollutes L2 if cached normally. The
    // streaming policy marks the read as evict-on-touch so the cache stays available
    // for KV-cache and other reuse-heavy data the next decode step will need.
    // num_bytes is clamped to cudaDevAttrMaxAccessPolicyWindowSize (128 MiB on 5090)
    // inside set_l2_streaming; the first 128 MiB of the weight matters most because
    // mid-decode the vocab logits row exits L2 before it can be re-read anyway.
    {
        const Tensor& w = model_->output_proj();
        if (w.data && w.nbytes() > 0 && !w.dropped_source)
            set_l2_streaming(stream, w.data, w.nbytes());
    }

    if (state.is_prefill && !state.all_logits) {
        Tensor h_last = view_tokens(hidden_, n).slice(n - 1, n);
        Tensor lg = view_tokens(logits_, 1);

        if (lm_tier == StorageTier::MXFP4 && lm_h->payload.mxfp4.linear_scales) {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            int hbs = lm_h->payload.mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no_last.data),
                                        static_cast<half*>(no_last.data), 1, cfg.d_model, hbs, stream);
            CutlassMxFP4Weight mxfp4_lm_w{};
            mxfp4_lm_w.data = lm_h->payload.mxfp4.weight;
            mxfp4_lm_w.scale_factors = lm_h->payload.mxfp4.scales;
            mxfp4_lm_w.linear_scales = lm_h->payload.mxfp4.linear_scales;
            mxfp4_lm_w.hadamard_bs = lm_h->payload.mxfp4.hadamard_bs;
            gemv_mxfp4_kpar_fp32(mxfp4_lm_w, static_cast<const half*>(no_last.data),
                                 static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (lm_has_fp8) {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            auto fp8_it = wcache_.fp8.find(model_->output_proj().data);
            int64_t wshape[2] = {static_cast<int64_t>(cfg.vocab_size), static_cast<int64_t>(cfg.d_model)};
            Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
            gemm_cublaslt(no_last, fp8_w, lg, 1.0f, 0.0f, nullptr, fp8_it->second.d_scale, stream);
        } else if (lm_is_nvfp4) {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            NvFP4QuantResult nvfp4_lm_r;
            if (lm_nvfp4_secondary) {
                nvfp4_lm_r = lm_nvfp4_it->second;
            } else {
                nvfp4_lm_r.packed_data = lm_h->payload.nvfp4.data;
                nvfp4_lm_r.micro_scales = lm_h->payload.nvfp4.block_scales;
                nvfp4_lm_r.tensor_scale = (lm_h->payload.nvfp4.tensor_scale != nullptr)
                                              ? *lm_h->payload.nvfp4.tensor_scale
                                              : 1.0f;
                nvfp4_lm_r.N = cfg.vocab_size;
                nvfp4_lm_r.K = cfg.d_model;
            }
            gemv_nvfp4_kpar_fp32(nvfp4_lm_r, static_cast<const half*>(no_last.data),
                                 static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (use_dp4a_lm) {
            if (debug_forward_enabled()) {
                Tensor no_last = view_tokens(norm_out_, 1);
                debug_tensor_stats("before_final_rmsnorm", h_last, stream);
                debug_tensor_stats("W_output_norm", model_->output_norm(), stream);
                rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_last, stream);
                debug_tensor_rows("after_final_rmsnorm_row", no_last, stream);
                debug_tensor_rows("h_last_row", h_last, stream);
            }
            // DEBUG experiment (b): bypass fused rmsnorm_quantize_q8_1 + dp4a GEMV.
            // Dequant output_proj to FP16 temp buffer, cuBLAS FP16 GEMM into FP32 logits.
            // If this gives llama-matching top logit (~+2.07) then the dp4a path is buggy;
            // if it still gives +8.83 then the bug is in hidden state or output_norm.
            if (runtime_config().generation.lm_dequant_fp16) {
                Tensor no_last = view_tokens(norm_out_, 1);
                rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
                int N = cfg.vocab_size;
                int K = cfg.d_model;
                void* w_fp16_dev = nullptr;
                size_t fp16_bytes = static_cast<size_t>(N) * K * sizeof(half);
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(&w_fp16_dev, fp16_bytes, stream));
                dequant_gpu(model_->output_proj().data, w_fp16_dev, out_qtype, N, K, stream);
                int64_t w_shape[2] = {N, K};
                Tensor w_fp16(w_fp16_dev, QType::F16, 2, w_shape, true);
                gemm(no_last, w_fp16, lg, 1.0f, 0.0f, stream);
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(w_fp16_dev, stream));
                fprintf(stderr, "[DEBUG_FWD] LM head via dequant->FP16->cuBLAS path\n");
            } else {
                auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
                rmsnorm_quantize_q8_1(static_cast<const half*>(h_last.data),
                                      static_cast<const half*>(model_->output_norm().data), q8,
                                      qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream,
                                      norm_w_off_);
                dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                                   static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else {
            Tensor no_last = view_tokens(norm_out_, 1);
            debug_tensor_stats("before_final_rmsnorm", h_last, stream);
            debug_tensor_stats("W_output_norm", model_->output_norm(), stream);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            gemm_via_handle_(model_->out_proj_id, no_last, lg, ctx);
        }
        logits_out = lg;
        debug_top_logits(lg, stream);
    } else {
        Tensor h_final = view_tokens(hidden_, n);
        Tensor lg = view_tokens(logits_, n);

        if (n == 1 && lm_tier == StorageTier::MXFP4 && lm_h->payload.mxfp4.linear_scales) {
            Tensor no_final = view_tokens(norm_out_, 1);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            int hbs = lm_h->payload.mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no_final.data),
                                        static_cast<half*>(no_final.data), 1, cfg.d_model, hbs, stream);
            CutlassMxFP4Weight mxfp4_lm_w{};
            mxfp4_lm_w.data = lm_h->payload.mxfp4.weight;
            mxfp4_lm_w.scale_factors = lm_h->payload.mxfp4.scales;
            mxfp4_lm_w.linear_scales = lm_h->payload.mxfp4.linear_scales;
            mxfp4_lm_w.hadamard_bs = lm_h->payload.mxfp4.hadamard_bs;
            gemv_mxfp4_kpar_fp32(mxfp4_lm_w, static_cast<const half*>(no_final.data),
                                 static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (n == 1 && lm_has_fp8) {
            Tensor no_final = view_tokens(norm_out_, 1);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            auto fp8_it = wcache_.fp8.find(model_->output_proj().data);
            int64_t wshape[2] = {static_cast<int64_t>(cfg.vocab_size), static_cast<int64_t>(cfg.d_model)};
            Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
            gemm_cublaslt(no_final, fp8_w, lg, 1.0f, 0.0f, nullptr, fp8_it->second.d_scale, stream);
        } else if (n == 1 && lm_is_nvfp4) {
            Tensor no_final = view_tokens(norm_out_, 1);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            NvFP4QuantResult nvfp4_lm_r;
            if (lm_nvfp4_secondary) {
                nvfp4_lm_r = lm_nvfp4_it->second;
            } else {
                nvfp4_lm_r.packed_data = lm_h->payload.nvfp4.data;
                nvfp4_lm_r.micro_scales = lm_h->payload.nvfp4.block_scales;
                nvfp4_lm_r.tensor_scale = (lm_h->payload.nvfp4.tensor_scale != nullptr)
                                              ? *lm_h->payload.nvfp4.tensor_scale
                                              : 1.0f;
                nvfp4_lm_r.N = cfg.vocab_size;
                nvfp4_lm_r.K = cfg.d_model;
            }
            gemv_nvfp4_kpar_fp32(nvfp4_lm_r, static_cast<const half*>(no_final.data),
                                 static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (n == 1 && use_dp4a_lm) {
            if (debug_forward_enabled()) {
                Tensor no_final = view_tokens(norm_out_, 1);
                rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            }
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            rmsnorm_quantize_q8_1(static_cast<const half*>(h_final.data),
                                  static_cast<const half*>(model_->output_norm().data), q8, qscratch_.d8_buf,
                                  nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                               static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (n > 1 && lm_is_nvfp4) {
            // Per-row NVFP4 GEMV LM head for batched decode.
            // NVFP4 GEMV is M=1 only — loop over rows.
            NvFP4QuantResult nvfp4_lm_r;
            if (lm_nvfp4_secondary) {
                nvfp4_lm_r = lm_nvfp4_it->second;
            } else {
                nvfp4_lm_r.packed_data = lm_h->payload.nvfp4.data;
                nvfp4_lm_r.micro_scales = lm_h->payload.nvfp4.block_scales;
                nvfp4_lm_r.tensor_scale = (lm_h->payload.nvfp4.tensor_scale != nullptr)
                                              ? *lm_h->payload.nvfp4.tensor_scale
                                              : 1.0f;
                nvfp4_lm_r.N = cfg.vocab_size;
                nvfp4_lm_r.K = cfg.d_model;
            }
            Tensor no_row = view_tokens(norm_out_, 1);
            for (int row = 0; row < n; ++row) {
                Tensor h_row = h_final.slice(row, row + 1);
                Tensor lg_row = lg.slice(row, row + 1);
                rmsnorm(h_row, model_->output_norm(), no_row, cfg.rms_norm_eps, stream, norm_w_off_);
                gemv_nvfp4_kpar_fp32(nvfp4_lm_r, static_cast<const half*>(no_row.data),
                                     static_cast<float*>(lg_row.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else if (use_dp4a_lm && n > 1) {
            // Per-row Q8_1 GEMV LM head for batched decode.
            // Quantized weights (Q8_0/Q6_K) can't be passed to cuBLAS directly.
            // Check if FP16 cache has the output projection — use cuBLAS GEMM if so.
            if (lm_tier == StorageTier::FP16) {
                Tensor no_final = view_tokens(norm_out_, n);
                rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
                gemm_via_handle_(model_->out_proj_id, no_final, lg, ctx);
                goto lm_head_done;
            }
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            for (int row = 0; row < n; ++row) {
                Tensor h_row = h_final.slice(row, row + 1);
                Tensor lg_row = lg.slice(row, row + 1);
                int64_t lg_flat[1] = {static_cast<int64_t>(cfg.vocab_size)};
                Tensor lg_1d = lg_row.reshape(1, lg_flat);

                rmsnorm_quantize_q8_1(static_cast<const half*>(h_row.data),
                                      static_cast<const half*>(model_->output_norm().data), q8,
                                      qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream,
                                      norm_w_off_);
                dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                                   static_cast<float*>(lg_1d.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else {
            Tensor no_final = view_tokens(norm_out_, n);
            debug_tensor_stats("before_final_rmsnorm", h_final, stream);
            debug_tensor_stats("W_output_norm", model_->output_norm(), stream);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);

            // For n>1 decode with quantized output weights, use FP8 GEMM or FP16 cache.
            // Raw gemm() can't handle Q8_0/Q6_K weights with cuBLAS.
            if (lm_tier == StorageTier::FP8 && qscratch_.fp8_act != nullptr &&
                qscratch_.d_act_scale != nullptr) {
                int64_t wshape[2] = {lm_h->shape[0], lm_h->shape[1]};
                Tensor fp8_lm_w(lm_h->payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
                Tensor fp8_no(qscratch_.fp8_act, QType::FP8_E4M3, no_final.ndim, no_final.shape, true);
                quantize_fp16_to_fp8_e4m3(no_final, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax,
                                          qscratch_.fp8_max_grid);
                gemm_cublaslt(fp8_no, fp8_lm_w, lg, 1.0f, 0.0f, qscratch_.d_act_scale,
                              lm_h->payload.fp8.d_scale, stream);
            } else {
                gemm_via_handle_(model_->out_proj_id, no_final, lg, ctx);
            }
        }
    lm_head_done:
        logits_out = lg;
        debug_top_logits(lg, stream);
    }

    // ---- Final logit soft-capping (Gemma-2/3/4, cap=30) ----
    const bool skip_softcap = runtime_config().generation.no_logit_softcap;
    if (cfg.final_logit_softcap > 0.0f && !skip_softcap) {
        int64_t n_logits = static_cast<int64_t>(logits_out.shape[0]) * cfg.vocab_size;
        int threads = 256;
        int blocks = static_cast<int>((n_logits + threads - 1) / threads);
        logit_softcap_fp32_kernel<<<blocks, threads, 0, stream>>>(static_cast<float*>(logits_out.data),
                                                                  cfg.final_logit_softcap,
                                                                  1.0f / cfg.final_logit_softcap, n_logits);
    }

    // ---- Profile summary ----
    if (profile_active) {
        cudaEventRecord(ev_lm, stream);
        cudaStreamSynchronize(stream);

        float t_emb = 0, t_lm = 0;
        float t_attn_total = 0, t_ffn_total = 0;
        cudaEventElapsedTime(&t_emb, ev_start, ev_emb);

        cudaEvent_t prev = ev_emb;
        for (int i = 0; i < cfg.n_layers; i++) {
            float t_attn = 0, t_ffn = 0;
            cudaEventElapsedTime(&t_attn, prev, ev_attn[i]);
            cudaEventElapsedTime(&t_ffn, ev_attn[i], ev_ffn[i]);
            t_attn_total += t_attn;
            t_ffn_total += t_ffn;
            prev = ev_ffn[i];
        }
        cudaEventElapsedTime(&t_lm, prev, ev_lm);

        float t_total = 0;
        cudaEventElapsedTime(&t_total, ev_start, ev_lm);
        acc_total += t_total;
        acc_attn += t_attn_total;
        acc_ffn += t_ffn_total;
        acc_lm += t_lm;

        int steps_profiled = profile_idx - 1;  // subtract warmup steps
        // Print every 32 steps
        if ((profile_idx & 31) == 0) {
            IMP_LOG_INFO(
                "PROFILE avg over %d steps: total=%.2fms  attn=%.2fms (%.0f%%)  "
                "ffn/moe=%.2fms (%.0f%%)  lm_head=%.2fms (%.0f%%)  "
                "(per-layer: attn=%.3fms  ffn=%.3fms)",
                steps_profiled, acc_total / steps_profiled, acc_attn / steps_profiled,
                100.0f * acc_attn / acc_total, acc_ffn / steps_profiled, 100.0f * acc_ffn / acc_total,
                acc_lm / steps_profiled, 100.0f * acc_lm / acc_total,
                acc_attn / steps_profiled / cfg.n_layers, acc_ffn / steps_profiled / cfg.n_layers);
        }

        // Cleanup handled by ProfileEvents RAII destructor.
    }
}

}  // namespace imp

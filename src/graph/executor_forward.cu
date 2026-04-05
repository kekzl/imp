#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
#include "graph/executor_debug.h"
#include "graph/gemm_context.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/attention_cutlass_fmha.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "memory/gdn_state.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/hadamard.h"
#include "compute/gemm_cublaslt_nvfp4.h"
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
static void dispatch_gemv_fp32(GGMLQuantType qtype,
                                const void* W, const block_q8_1* q8_1, const float* d8,
                                float* y, int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case GGMLQuantType::Q6_K: gemv_q6k_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q4_0: gemv_q4_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q4_K: gemv_q4_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q5_K: gemv_q5_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q2_K: gemv_q2_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q3_K: gemv_q3_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
        default:                  gemv_q8_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream); break;
    }
}

// write_kv_cache: moved to executor_kv_write.cu

// ---------------------------------------------------------------------------
// Forward pass diagnostics (IMP_DEBUG_FORWARD=1)
// ---------------------------------------------------------------------------

// debug_forward_enabled, debug_tensor_stats: shared via executor_debug.h

// Print top-k logits with token IDs
void debug_top_logits(const Tensor& logits, cudaStream_t stream, int topk = 10) {
    if (!debug_forward_enabled()) return;
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    std::vector<float> host(vocab);

    if (logits.dtype == DType::FP32) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(host.data(), logits.data, vocab * sizeof(float),
                         cudaMemcpyDeviceToHost, stream));
    } else if (logits.dtype == DType::FP16) {
        std::vector<half> tmp(vocab);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(tmp.data(), logits.data, vocab * sizeof(half),
                         cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        for (int i = 0; i < vocab; i++) host[i] = __half2float(tmp[i]);
    }
    cudaStreamSynchronize(stream);

    // Find top-k by partial sort
    std::vector<std::pair<float, int>> scored(vocab);
    for (int i = 0; i < vocab; i++) scored[i] = {host[i], i};
    std::partial_sort(scored.begin(), scored.begin() + std::min(topk, vocab),
                      scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    fprintf(stderr, "[DEBUG_FWD] Top-%d logits:\n", topk);
    for (int i = 0; i < std::min(topk, vocab); i++) {
        fprintf(stderr, "  [%2d] token_id=%6d  logit=%+.6f\n",
                i, scored[i].second, scored[i].first);
    }
}

// run_attention: moved to executor_attention.cu


// run_ffn: moved to executor_ffn.cu

// run_ssm, run_gdn: moved to executor_ssm_gdn.cu

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

void GraphExecutor::forward_logits(const InferenceState& state,
                                   Tensor& logits_out,
                                   cudaStream_t stream) {
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
        IMP_LOG_ERROR("n_tokens (%d) exceeds max_tokens (%d)", n, max_tokens_);
        return;
    }

    // Store for use by run_ffn (which doesn't receive the InferenceState).
    cur_n_tokens_ = n;
    cur_force_fp16_ = state.force_fp16_gemm;
    cur_per_row_lm_ = state.per_row_lm_head;

    // Clear any stale CUDA error state before starting the forward pass.
    { cudaError_t e_ = cudaGetLastError();
      if (e_ != cudaSuccess) IMP_LOG_WARN("Cleared stale error before forward: %s", cudaGetErrorString(e_)); }

    // ---- Optional per-component profiling (IMP_PROFILE=1) ----
    // Profiling disables CUDA graph capture (they are incompatible).
    // Use IMP_PROFILE=1 for diagnostic runs only.
    static const bool do_profile = (std::getenv("IMP_PROFILE") != nullptr);
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
            if (!active) return;
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_emb) cudaEventDestroy(ev_emb);
            if (ev_lm) cudaEventDestroy(ev_lm);
            for (auto e : ev_attn) if (e) cudaEventDestroy(e);
            for (auto e : ev_ffn) if (e) cudaEventDestroy(e);
        }
    } prof;
    // Alias references for minimal churn in the rest of the function.
    auto& ev_start = prof.ev_start;
    auto& ev_emb   = prof.ev_emb;
    auto& ev_lm    = prof.ev_lm;
    auto& ev_attn  = prof.ev_attn;
    auto& ev_ffn   = prof.ev_ffn;
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
        IMP_CUDA_CHECK_LOG(cudaMemcpy(h_ids.data(), state.token_ids, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[DEBUG_FWD] input_tokens (%d):", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", h_ids[i]);
        fprintf(stderr, "\n");
    }
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup(model_->token_embedding(), state.token_ids, n, h,
                     model_->tok_emb_qtype_, stream);

    // Gemma: scale embeddings by sqrt(d_model)
    if (cfg.embed_scale > 0.0f && h.dtype == DType::FP16) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(h.data), __float2half(cfg.embed_scale), total);
    }

    // Replace vision token positions with vision embeddings (multimodal)
    if (state.vision_embeddings && state.vision_token_id >= 0 && state.n_vision_tokens > 0) {
        // Declared in vision/vision_encoder.cu
        extern void launch_replace_vision_embeddings(
            half* hidden, const int32_t* token_ids, const half* vision_emb,
            int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
            cudaStream_t stream);
        launch_replace_vision_embeddings(
            static_cast<half*>(h.data), state.token_ids, state.vision_embeddings,
            state.vision_token_id, n, cfg.d_model, state.n_vision_tokens, stream);
    }

    debug_tensor_stats("after_embedding", h, stream);

    // Initialize FP32 residual accumulator from FP16 embedding (post-norm models only).
    if (fp32_accum_buf_) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(h.data),
            static_cast<float*>(view_tokens(fp32_hidden_, n).data), total);
    }

    if (profile_active) cudaEventRecord(ev_emb, stream);

    // ---- Step 2: Transformer/Hybrid layers ----
    int max_layer = (state.exit_layer > 0)
                    ? std::min(state.exit_layer, cfg.n_layers)
                    : cfg.n_layers;
    const int skip_start = state.skip_layer_start;
    const int skip_end   = state.skip_layer_end;
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
        if (i <= 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_layer%d_%s", i,
                     layer_has_gdn(i) ? "gdn" :
                     layer_has_attention(i) ? "attn" : "ssm");
            debug_tensor_stats(buf, h, stream);
        }
        if (profile_active) cudaEventRecord(ev_attn[i], stream);


        // FFN: MoE, dense, or none (attention-only layers may have no FFN)
        if (layer_has_moe(i)) {
            run_moe_ffn(i, stream);
        } else if (layer_has_dense_ffn(i)) {
            run_ffn(i, stream);
        }
        if (i <= 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_layer%d_%s", i,
                     layer_has_moe(i) ? "moe" : (layer_has_dense_ffn(i) ? "ffn" : "no_ffn"));
            debug_tensor_stats(buf, h, stream);
        }
        if (i == max_layer - 1) {
            debug_tensor_stats("after_last_layer", h, stream);
        }
        if (profile_active) cudaEventRecord(ev_ffn[i], stream);

        // Release offloaded layer (restore host pointers)
        if (offload_mgr_) {
            offload_mgr_->release_layer(i);
        }
    }

    // Final FP32→FP16 conversion for the tokens that need LM head projection.
    // run_attention/run_ffn already keep hidden_ in sync with fp32_hidden_,
    // but this ensures the final state is clean (no stale data from earlier layers).
    if (fp32_accum_buf_) {
        fp32_to_fp16_rowscale_kernel<<<n, 256, 256 * sizeof(float), stream>>>(
            static_cast<const float*>(view_tokens(fp32_hidden_, n).data),
            static_cast<half*>(h.data), n, cfg.d_model);
    }

    // ---- Step 3+4: Final RMSNorm + LM head projection ----
    // Only project the tokens that actually need sampling:
    //   Prefill: last token only (all others just populate KV cache)
    //   Decode:  all tokens (one per sequence)
    //
    // For raw Q6_K/Q8_0 output projection with single token (n=1 or prefill last):
    // use fused RMSNorm→Q8_1 + dp4a GEMV with FP32 output. Saves ~2.45x VRAM
    // bandwidth vs cuBLAS FP16 path (reads quantized weights directly).
    const auto out_qtype = model_->out_proj_qtype_;
    const bool use_dp4a_lm = qscratch_.q8_1_buf && compute_dtype_ == DType::FP16 &&
        is_dp4a_qtype(out_qtype);

    // GemmContext for LM head GEMM dispatches.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, cur_force_fp16_);

    if (state.is_prefill && !state.all_logits) {
        Tensor h_last = view_tokens(hidden_, n).slice(n - 1, n);
        Tensor lg = view_tokens(logits_, 1);

        auto mxfp4_lm_pf = wcache_.cutlass_mxfp4.find(model_->output_proj().data);
        auto nvfp4_lm_pf = wcache_.nvfp4.find(model_->output_proj().data);
        if (mxfp4_lm_pf != wcache_.cutlass_mxfp4.end() && mxfp4_lm_pf->second.linear_scales) {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            int hbs = mxfp4_lm_pf->second.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no_last.data),
                                        static_cast<half*>(no_last.data), 1, cfg.d_model, hbs, stream);
            gemv_mxfp4_kpar_fp32(mxfp4_lm_pf->second,
                                  static_cast<const half*>(no_last.data),
                                  static_cast<float*>(lg.data),
                                  cfg.vocab_size, cfg.d_model, stream);
        } else if (nvfp4_lm_pf != wcache_.nvfp4.end()) {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            gemv_nvfp4_kpar_fp32(nvfp4_lm_pf->second,
                                  static_cast<const half*>(no_last.data),
                                  static_cast<float*>(lg.data),
                                  cfg.vocab_size, cfg.d_model, stream);
        } else if (use_dp4a_lm) {
            if (debug_forward_enabled()) {
                Tensor no_last = view_tokens(norm_out_, 1);
                rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            }
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_last.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                               static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            // LM head fallback: use gemm_dispatch for consistent MXFP4/FP16 handling
            gemm_dispatch(no_last, model_->output_proj(), model_->out_proj_qtype_, lg, ctx);
        }
        logits_out = lg;
        debug_top_logits(lg, stream);
    } else {
        Tensor h_final = view_tokens(hidden_, n);
        Tensor lg = view_tokens(logits_, n);

        auto mxfp4_lm = wcache_.cutlass_mxfp4.find(model_->output_proj().data);
        auto nvfp4_lm = wcache_.nvfp4.find(model_->output_proj().data);
        if (n == 1 && mxfp4_lm != wcache_.cutlass_mxfp4.end() && mxfp4_lm->second.linear_scales) {
            Tensor no_final = view_tokens(norm_out_, 1);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            int hbs = mxfp4_lm->second.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no_final.data),
                                        static_cast<half*>(no_final.data), 1, cfg.d_model, hbs, stream);
            gemv_mxfp4_kpar_fp32(mxfp4_lm->second,
                                  static_cast<const half*>(no_final.data),
                                  static_cast<float*>(lg.data),
                                  cfg.vocab_size, cfg.d_model, stream);
        } else if (n == 1 && nvfp4_lm != wcache_.nvfp4.end()) {
            Tensor no_final = view_tokens(norm_out_, 1);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            gemv_nvfp4_kpar_fp32(nvfp4_lm->second,
                                  static_cast<const half*>(no_final.data),
                                  static_cast<float*>(lg.data),
                                  cfg.vocab_size, cfg.d_model, stream);
        } else if (n == 1 && use_dp4a_lm) {
            if (debug_forward_enabled()) {
                Tensor no_final = view_tokens(norm_out_, 1);
                rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            }
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_final.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                               static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else if (n > 1 && nvfp4_lm != wcache_.nvfp4.end()) {
            // Per-row NVFP4 GEMV LM head for batched decode.
            // NVFP4 GEMV is M=1 only — loop over rows.
            Tensor no_row = view_tokens(norm_out_, 1);
            for (int row = 0; row < n; ++row) {
                Tensor h_row = h_final.slice(row, row + 1);
                Tensor lg_row = lg.slice(row, row + 1);
                rmsnorm(h_row, model_->output_norm(), no_row, cfg.rms_norm_eps, stream, norm_w_off_);
                gemv_nvfp4_kpar_fp32(nvfp4_lm->second,
                                      static_cast<const half*>(no_row.data),
                                      static_cast<float*>(lg_row.data),
                                      cfg.vocab_size, cfg.d_model, stream);
            }
        } else if (use_dp4a_lm && n > 1) {
            // Per-row Q8_1 GEMV LM head for batched decode.
            // Quantized weights (Q8_0/Q6_K) can't be passed to cuBLAS directly.
            // Check if FP16 cache has the output projection — use cuBLAS GEMM if so.
            if (wcache_.fp16.count(model_->output_proj().data)) {
                Tensor no_final = view_tokens(norm_out_, n);
                rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
                gemm_dispatch(no_final, model_->output_proj(), model_->out_proj_qtype_, lg, ctx);
                goto lm_head_done;
            }
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            for (int row = 0; row < n; ++row) {
                Tensor h_row = h_final.slice(row, row + 1);
                Tensor lg_row = lg.slice(row, row + 1);
                int64_t lg_flat[1] = {static_cast<int64_t>(cfg.vocab_size)};
                Tensor lg_1d = lg_row.reshape(1, lg_flat);

                rmsnorm_quantize_q8_1(
                    static_cast<const half*>(h_row.data),
                    static_cast<const half*>(model_->output_norm().data),
                    q8, qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
                dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                                   static_cast<float*>(lg_1d.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else {
            Tensor no_final = view_tokens(norm_out_, n);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);

            // For n>1 decode with quantized output weights, use FP8 GEMM or FP16 cache.
            // Raw gemm() can't handle Q8_0/Q6_K weights with cuBLAS.
            auto fp8_lm = wcache_.fp8.find(model_->output_proj().data);
            if (fp8_lm != wcache_.fp8.end() && qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
                Tensor fp8_no(qscratch_.fp8_act, DType::FP8_E4M3, no_final.ndim, no_final.shape, true);
                quantize_fp16_to_fp8_e4m3(no_final, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                gemm_cublaslt(fp8_no, fp8_lm->second.weight, lg, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_lm->second.d_scale, stream);
            } else {
                // Fallback: gemm_dispatch handles FP16 cache, MXFP4, quantized, etc.
                gemm_dispatch(no_final, model_->output_proj(), model_->out_proj_qtype_, lg, ctx);
            }
        }
    lm_head_done:
        logits_out = lg;
        debug_top_logits(lg, stream);
    }

    // ---- Final logit soft-capping (Gemma-2/3) ----
    if (cfg.final_logit_softcap > 0.0f) {
        int64_t n_logits = static_cast<int64_t>(logits_out.shape[0]) * cfg.vocab_size;
        int threads = 256;
        int blocks = static_cast<int>((n_logits + threads - 1) / threads);
        logit_softcap_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(logits_out.data),
            cfg.final_logit_softcap, 1.0f / cfg.final_logit_softcap, n_logits);
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
            IMP_LOG_INFO("PROFILE avg over %d steps: total=%.2fms  attn=%.2fms (%.0f%%)  "
                         "ffn/moe=%.2fms (%.0f%%)  lm_head=%.2fms (%.0f%%)  "
                         "(per-layer: attn=%.3fms  ffn=%.3fms)",
                         steps_profiled,
                         acc_total / steps_profiled,
                         acc_attn / steps_profiled,
                         100.0f * acc_attn / acc_total,
                         acc_ffn / steps_profiled,
                         100.0f * acc_ffn / acc_total,
                         acc_lm / steps_profiled,
                         100.0f * acc_lm / acc_total,
                         acc_attn / steps_profiled / cfg.n_layers,
                         acc_ffn / steps_profiled / cfg.n_layers);
        }

        // Cleanup handled by ProfileEvents RAII destructor.
    }
}


} // namespace imp

#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
#include "graph/executor_gemv_helpers.h"
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
// Quant type dispatch helpers (file-local)
// ---------------------------------------------------------------------------

// is_dp4a_qtype() and dispatch_dp4a_gemv() are defined in executor_kernels.h

// Fused QKV GEMV dispatch by quant type (all share identical signatures).
static void dispatch_gemv_qkv_fused(GGMLQuantType qtype,
                                     const void* W_q, const void* W_k, const void* W_v,
                                     const block_q8_1* q8_1, const float* d8,
                                     half* y_q, half* y_k, half* y_v,
                                     int q_rows, int k_rows, int v_rows, int K,
                                     cudaStream_t stream) {
    switch (qtype) {
        case GGMLQuantType::Q6_K: gemv_qkv_fused_q6k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        case GGMLQuantType::Q4_0: gemv_qkv_fused_q4_0_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        case GGMLQuantType::Q4_K: gemv_qkv_fused_q4_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        case GGMLQuantType::Q5_K: gemv_qkv_fused_q5_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        case GGMLQuantType::Q2_K: gemv_qkv_fused_q2_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        case GGMLQuantType::Q3_K: gemv_qkv_fused_q3_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
        default:                  gemv_qkv_fused_q8_0_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K, stream); break;
    }
}

// dispatch_gemv_residual: from executor_gemv_helpers.h
// get_kv_layer: from executor_helpers.h

// Set L2 persistence hint for KV cache data on the given stream.
// Tells the GPU to prioritize keeping this address range in L2 cache.
// Resets automatically when the stream attribute is overwritten next layer.
static void set_l2_persist_kv(cudaStream_t stream, const void* kv_ptr, size_t kv_bytes) {
    if (!kv_ptr || kv_bytes == 0 || !stream) return;
    // Clamp to device's persisting L2 cache limit
    static size_t max_persist = 0;
    if (max_persist == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist == 0) return;  // L2 persistence not supported
    }
    // Use proportional hitRatio when KV exceeds L2 capacity, so the hardware
    // probabilistically persists a representative subset across the full range
    // (not just the first max_persist bytes).
    float ratio = (kv_bytes <= max_persist) ? 1.0f
                : static_cast<float>(max_persist) / static_cast<float>(kv_bytes);
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(kv_ptr);
    attr.accessPolicyWindow.num_bytes = kv_bytes;
    attr.accessPolicyWindow.hitRatio = ratio;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

// Alias for the shared clear_l2_policy helper (back-compat name for call sites).
static void clear_l2_persist(cudaStream_t stream) { clear_l2_policy(stream); }

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state,
                                  cudaStream_t stream) {
    // Configure shared workspace for attention phase
    configure_attn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n   = state.n_tokens;
    int nh  = cfg.n_heads;
    // Per-layer head_dim / n_kv_heads (Gemma 4 dual geometry + Nemotron-H hybrid)
    int nkv = (!cfg.n_kv_heads_per_layer.empty() && layer < (int)cfg.n_kv_heads_per_layer.size() &&
               cfg.n_kv_heads_per_layer[layer] > 0)
              ? cfg.n_kv_heads_per_layer[layer] : cfg.n_kv_heads;
    int hd  = (!cfg.head_dim_per_layer.empty() && layer < (int)cfg.head_dim_per_layer.size() &&
               cfg.head_dim_per_layer[layer] > 0)
              ? cfg.head_dim_per_layer[layer]
              : (cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh));
    // Gemma 4: derive per-layer n_heads and n_kv_heads from actual tensor shapes.
    // Layer 0 (SWA) wq=[4096,2816] wk=[2048,2816] → 16 Q × hd=256, 8 KV × hd=256
    // Layer 5 (Global) wq=[8192,2816] wk=[1024,2816] → 16 Q × hd=512, 2 KV × hd=512
    // Authoritative source = the loaded tensor shapes; per-layer config can lag.
    if (cfg.arch == ModelArch::GEMMA4 && hd > 0 && ly.wq.data != nullptr) {
        int wq_out = static_cast<int>(ly.wq.shape[0]);
        if (wq_out > 0 && (wq_out % hd) == 0) {
            int nh_layer = wq_out / hd;
            if (nh_layer > 0 && nh_layer != nh) nh = nh_layer;
        }
        if (ly.wk.data != nullptr) {
            int wk_out = static_cast<int>(ly.wk.shape[0]);
            if (wk_out > 0 && (wk_out % hd) == 0) {
                int nkv_layer = wk_out / hd;
                if (nkv_layer > 0 && nkv_layer != nkv) nkv = nkv_layer;
            }
        }
    }
    float eps = cfg.rms_norm_eps;


    // Sized views for this call (never mutates member tensors).
    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // Qwen3.5 attention: Q projection has ×2 output (Q + output_gate fused).
    // Detect by checking if wq output dim > n_heads * head_dim.
    int q_out_dim = static_cast<int>(ly.wq.shape[0]);
    bool has_attn_output_gate = (q_out_dim > nh * hd);
    int q_actual_dim = nh * hd;  // actual Q dimension (without gate)

    Tensor qv = view_tokens(q_,        n);
    Tensor kk = view_tokens(k_,        n);
    Tensor vv = view_tokens(v_,        n);
    Tensor ao = view_tokens(attn_out_, n);
    Tensor po = view_tokens(proj_out_, n);

    const bool per_layer_shapes =
        (!cfg.head_dim_per_layer.empty() || !cfg.n_kv_heads_per_layer.empty());

    // Per-layer shape narrowing: Q/K/V/ao workspace tensors are allocated with
    // max shapes (for worst-case layer). For layers with smaller head_dim (Gemma 4
    // SWA), narrow the view so cuBLAS gemm writes with the correct leading dim.
    if (per_layer_shapes) {
        auto narrow_cols = [](Tensor& t, int64_t new_cols) {
            if (t.shape[1] != new_cols) {
                t.shape[1] = new_cols;
                t.compute_strides();
            }
        };
        narrow_cols(qv, static_cast<int64_t>(nh)  * hd);
        narrow_cols(kk, static_cast<int64_t>(nkv) * hd);
        narrow_cols(vv, static_cast<int64_t>(nkv) * hd);
        narrow_cols(ao, static_cast<int64_t>(nh)  * hd);
    }

    // For Qwen3.5 attention output gate: allocate larger Q buffer AFTER all
    // standard attention buffers to avoid overlap (q_/k_/v_/attn_out_/proj_out_
    // all share the same shared_workspace_ memory).
    Tensor qv_full;
    if (has_attn_output_gate) {
        auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
        size_t es_a = dtype_size(compute_dtype_);
        // Place after proj_out_ (last standard buffer)
        char* after_proj = static_cast<char*>(po.data) +
                           align256(static_cast<size_t>(n) * cfg.d_model * es_a);
        int64_t qfull_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(q_out_dim)};
        qv_full = Tensor(after_proj, compute_dtype_, 2, qfull_shape, true);
    }

    // Per-step diagnostics for n>1 decode debugging (layer 0 only)
    bool debug_attn_steps = (layer == 0 && n > 1 && debug_forward_enabled());
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step0_h_input", h, stream);
    }

    // 1. Save residual for later add-back.
    //    Optimization: for decode (n=1) with dp4a, fuse residual into GEMV.
    //    For prefill (n>1) with FP16 cache, use cuBLAS beta=1 to fuse residual
    //    into the wo projection GEMM — no separate residual save/add/copy needed.
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    // True sandwich norm: post_attn_norm applied inside run_attention AND separate
    // ffn_norm for run_ffn (Gemma-3 pattern). When ffn_norm is absent (Qwen3.5),
    // post_attn_norm serves as FFN input norm in run_ffn — NOT a sandwich norm.
    // Without this check, post_attn_norm is applied TWICE (here + run_ffn fallback).
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr && ly.ffn_norm.data != nullptr);
    // FP32 residual accumulator (Gemma-3 dense + Gemma-4 MoE post-norm architecture).
    // Kernel semantics: fp32_h += rmsnorm(po) * w. llama's build_norm(attn) + residual
    // is mathematically identical for both Gemma-3 and Gemma-4 (normalize-then-add).
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
    if (debug_forward_enabled() && layer <= 1) {
        fprintf(stderr, "[DEBUG_FWD] L%d attn: has_post_attn_norm=%d using_fp32_accum=%d "
                "post_attn_norm=%p ffn_norm=%p\n",
                layer, (int)has_post_attn_norm, (int)using_fp32_accum,
                ly.post_attn_norm.data, ly.ffn_norm.data);
    }
    bool will_fuse_o_nvfp4 = (!has_post_attn_norm && n == 1 && h.dtype == DType::FP16 &&
                               wcache_.nvfp4.count(ly.wo.data));
    bool will_fuse_o_residual = (!has_post_attn_norm && !will_fuse_o_nvfp4 &&
                                  n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                                  h.dtype == DType::FP16 && is_dp4a_qtype(ly.wo_qtype));
    bool will_fuse_o_beta1 = (!has_post_attn_norm && !will_fuse_o_residual && !will_fuse_o_nvfp4 &&
                               n > 1 &&
                               (wcache_.fp16.count(ly.wo.data) || wcache_.fp8.count(ly.wo.data)));
    // Dequant beta=1 path: when force_fp16_gemm bypasses FP8, dequant weights on-the-fly
    bool will_fuse_o_dequant_beta1 = (!has_post_attn_norm && !will_fuse_o_residual &&
                                      !will_fuse_o_nvfp4 && !will_fuse_o_beta1 &&
                                      n > 1 && qscratch_.dequant != nullptr &&
                                      dequant_gpu_supported(ly.wo_qtype));
    if (!will_fuse_o_residual && !will_fuse_o_beta1 && !will_fuse_o_dequant_beta1 &&
        !will_fuse_o_nvfp4 && !using_fp32_accum) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream));
    }

    // For Qwen3.5: Q projection writes to larger buffer (includes gate), then split
    Tensor q_target = has_attn_output_gate ? qv_full : qv;

    // GemmContext for all weight GEMM dispatches in this function.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, cur_force_fp16_);

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    For decode (n=1) with matching quant types: fused RMSNorm→Q8_1→QKV GEMV.
    //    This skips the intermediate norm_out FP16 buffer entirely.
    //    Otherwise falls back to separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        // MXFP4 decode path: native MXFP4 GEMV with UE8M0 scales
        auto mxfp4_wq = wcache_.cutlass_mxfp4.find(ly.wq.data);
        auto mxfp4_wk = wcache_.cutlass_mxfp4.find(ly.wk.data);
        auto mxfp4_wv = wcache_.cutlass_mxfp4.find(ly.wv.data);
        bool mxfp4_qkv = (!has_attn_output_gate && n == 1 &&
                          mxfp4_wq != wcache_.cutlass_mxfp4.end() && mxfp4_wq->second.linear_scales &&
                          mxfp4_wk != wcache_.cutlass_mxfp4.end() && mxfp4_wk->second.linear_scales &&
                          mxfp4_wv != wcache_.cutlass_mxfp4.end() && mxfp4_wv->second.linear_scales);
        // NVFP4 decode path: uses FP16 input (no Q8_1 quantization needed)
        auto nvfp4_wq = wcache_.nvfp4.find(ly.wq.data);
        auto nvfp4_wk = wcache_.nvfp4.find(ly.wk.data);
        auto nvfp4_wv = wcache_.nvfp4.find(ly.wv.data);
        bool nvfp4_qkv = (!has_attn_output_gate && n == 1 && nvfp4_wq != wcache_.nvfp4.end() &&
                          nvfp4_wk != wcache_.nvfp4.end() && nvfp4_wv != wcache_.nvfp4.end());
        // Gemma-4: disable fused QKV when FP32 accum is active — the fused kernel
        // reads FP16 h instead of fp32_hidden_, losing precision through 128-expert routing.
        bool fused_qkv = (!has_attn_output_gate && n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                          no.dtype == DType::FP16 &&
                          ly.wq_qtype == ly.wk_qtype && ly.wk_qtype == ly.wv_qtype &&
                          is_dp4a_qtype(ly.wq_qtype) &&
                          !(using_fp32_accum && cfg.arch == ModelArch::GEMMA4));
        if (mxfp4_qkv) {
            // MXFP4 fused QKV: RMSNorm, optional Hadamard, then MXFP4 GEMV
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            int K = static_cast<int>(ly.wq.shape[1]);
            int hbs = mxfp4_wq->second.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no.data),
                                        static_cast<half*>(no.data), 1, K, hbs, stream);
            gemv_mxfp4_qkv_fused(mxfp4_wq->second, mxfp4_wk->second, mxfp4_wv->second,
                                  static_cast<const half*>(no.data),
                                  static_cast<half*>(qv.data),
                                  static_cast<half*>(kk.data),
                                  static_cast<half*>(vv.data),
                                  q_rows, k_rows, v_rows, K, stream);
        } else if (nvfp4_qkv) {
            // NVFP4 fused QKV: RMSNorm to FP16, then NVFP4 GEMV (no Q8_1 needed)
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(nvfp4_wq->second.N);
            int k_rows = static_cast<int>(nvfp4_wk->second.N);
            int v_rows = static_cast<int>(nvfp4_wv->second.N);
            int K = static_cast<int>(nvfp4_wq->second.K);
            gemv_nvfp4_qkv_fused(nvfp4_wq->second, nvfp4_wk->second, nvfp4_wv->second,
                                  static_cast<const half*>(no.data),
                                  static_cast<half*>(qv.data),
                                  static_cast<half*>(kk.data),
                                  static_cast<half*>(vv.data),
                                  q_rows, k_rows, v_rows, K, stream);
        } else if (fused_qkv) {
            // Fused: RMSNorm + Q8_1 quantization in one kernel (no norm_out write)
            int K = static_cast<int>(ly.wq.shape[1]);
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                    static_cast<const half*>(ly.attn_norm.data),
                                    q8, qscratch_.d8_buf, nullptr /*skip norm_out*/,
                                    K, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            dispatch_gemv_qkv_fused(ly.wq_qtype,
                                     ly.wq.data, ly.wk.data, ly.wv.data,
                                     q8, qscratch_.d8_buf,
                                     static_cast<half*>(qv.data),
                                     static_cast<half*>(kk.data),
                                     static_cast<half*>(vv.data),
                                     q_rows, k_rows, v_rows, K, stream);
        } else {
            // Separate RMSNorm + dispatch.
            // Gemma-4 FP32 accum path: read FP32 residual directly to avoid the
            // FP16 round-trip that drops ~1-2% precision per layer and causes
            // the last-token hidden state to drift by sign-flip at L29.
            if (using_fp32_accum && cfg.arch == ModelArch::GEMMA4) {
                Tensor fp32_h = view_tokens(fp32_hidden_, n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.attn_norm, no, eps, stream, norm_w_off_);
            } else {
                rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            }

            // FP8 prefill path: quantize norm_out→FP8 once, 3 separate FP8 GEMMs
            auto fp8_wq = wcache_.fp8.find(ly.wq.data);
            auto fp8_wk = wcache_.fp8.find(ly.wk.data);
            auto fp8_wv = wcache_.fp8.find(ly.wv.data);
            if (wcache_.use_fp8 && n > 1 && !state.force_fp16_gemm &&
                fp8_wq != wcache_.fp8.end() &&
                fp8_wk != wcache_.fp8.end() && fp8_wv != wcache_.fp8.end() &&
                qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
                Tensor fp8_no(qscratch_.fp8_act, DType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                gemm_cublaslt(fp8_no, fp8_wq->second.weight, q_target, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_wq->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wk->second.weight, kk, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_wk->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wv->second.weight, vv, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_wv->second.d_scale, stream);
            } else {
                // Try fused K+V path: single strided batched GEMM for both projections
                auto fused_kv_it = wcache_.fused_kv.find(layer);
                // Gemma 4 per-layer shapes break strided-batched K+V layout assumptions.
                if (n > 1 && fused_kv_it != wcache_.fused_kv.end() && !per_layer_shapes) {
                    // Q: still separate (different output dim with GQA)
                    gemm_dispatch(no, ly.wq, ly.wq_qtype, q_target, ctx);
                    // K+V: one batched cuBLAS call
                    gemm_kv_batched(no, fused_kv_it->second, kk, vv, stream);
                } else {
                    gemm_dispatch(no, ly.wq, ly.wq_qtype, q_target, ctx);
                    gemm_dispatch(no, ly.wk, ly.wk_qtype, kk, ctx);
                    if (ly.wv.data != nullptr) {
                        gemm_dispatch(no, ly.wv, ly.wv_qtype, vv, ctx);
                    }
                    // else: K=V sharing path — vv populated below from kk.
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2) — fused 3-way for 1 launch
        add_bias_3way(qv, ly.q_bias, kk, ly.k_bias, vv, ly.v_bias, stream);
    }

    // Gemma 4: K=V sharing for global attention layers (wv == null).
    // These layers have no V projection — V is aliased from K. Copy K→V here
    // so all downstream code (QK-norm, V-norm, KV-write, attention) sees a
    // valid V tensor.
    if (cfg.arch == ModelArch::GEMMA4 && ly.wv.data == nullptr &&
        kk.data != nullptr && vv.data != nullptr) {
        size_t kv_bytes = static_cast<size_t>(n) * nkv * hd * dtype_size(kk.dtype);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(vv.data, kk.data, kv_bytes,
                        cudaMemcpyDeviceToDevice, stream));
    }

    // V-normalization (Gemma 4): per-head RMSNorm with NO learned weight.
    // Matches llama.cpp's `Vcur = ggml_rms_norm(Vcur, eps)` (gemma4-iswa.cpp:82).
    // Required for both K=V-shared global layers and standard SWA layers.
    if (cfg.arch == ModelArch::GEMMA4 && v_norm_ones_buf_ != nullptr) {
        int64_t vflat_shape[4] = {static_cast<int64_t>(n) * nkv, hd, 0, 0};
        Tensor v_flat(vv.data, vv.dtype, 2, vflat_shape, true);
        int64_t ones_shape[4] = {hd, 0, 0, 0};
        Tensor ones_w(v_norm_ones_buf_, DType::FP16, 1, ones_shape, true);
        rmsnorm(v_flat, ones_w, v_flat, eps, stream, 0.0f);
    }

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step1_after_qkv_q", qv, stream);
        debug_tensor_stats("L0_step1_after_qkv_k", kk, stream);
    }

    // Per-layer RoPE theta and sliding window (Gemma-3: alternating local/global layers)
    float layer_rope_theta = cfg.rope_theta;
    float layer_rope_freq_scale = cfg.rope_freq_scale;
    int layer_sliding_window = cfg.sliding_window;
    if (cfg.arch == ModelArch::GEMMA4 && !cfg.swa_layers.empty()) {
        // Gemma 4: per-layer SWA pattern stored in cfg.swa_layers (1=SWA, 0=global).
        bool is_swa = (layer < (int)cfg.swa_layers.size() && cfg.swa_layers[layer]);
        if (is_swa) {
            layer_rope_theta = (cfg.rope_theta_swa > 0.0f) ? cfg.rope_theta_swa : cfg.rope_local_theta;
            layer_rope_freq_scale = 1.0f;
            // layer_sliding_window stays at cfg.sliding_window (1024)
        } else {
            // Global layer: full attention, model rope_theta, with freq scaling
            layer_sliding_window = 0;
        }
    } else if (cfg.sliding_window_pattern > 0) {
        bool is_global = (layer % cfg.sliding_window_pattern) == (cfg.sliding_window_pattern - 1);
        if (is_global) {
            // Global layer: full attention, model-level rope_theta, with freq scaling
            layer_sliding_window = 0;
        } else {
            // Local layer: sliding window, local rope_theta, no freq scaling
            if (cfg.rope_local_theta > 0.0f)
                layer_rope_theta = cfg.rope_local_theta;
            layer_rope_freq_scale = 1.0f;  // no scaling for local layers
        }
    }

    // Select LongRoPE frequency table based on context length (nullptr if not longrope)
    const float* longrope_freqs = nullptr;
    if (longrope_short_freqs_) {
        longrope_freqs = (state.max_context_len <= longrope_orig_max_pos_)
                         ? longrope_short_freqs_ : longrope_long_freqs_;
    }
    // Gemma 4: global attention layers use a separate `rope_freqs` tensor
    // (loaded as a top-level tensor, fanned out to every global layer's slot).
    if (cfg.arch == ModelArch::GEMMA4 && ly.rope_freqs.data != nullptr &&
        ly.rope_freqs.on_device) {
        longrope_freqs = static_cast<const float*>(ly.rope_freqs.data);
    }

    // Qwen3.5 attention output gate: Q projection is INTERLEAVED per head:
    //   [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...]
    // NOT [Q_all, Gate_all]. De-interleave Q and gate.
    Tensor attn_gate_buf;
    if (has_attn_output_gate) {
        size_t es_q = dtype_size(compute_dtype_);
        int64_t gate_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(q_actual_dim)};
        attn_gate_buf = Tensor(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);

        // De-interleave: src has stride 2*hd per head, dst has stride hd per head
        // For each token t, for each head h:
        //   Q[t, h*hd : (h+1)*hd] = src[t, h*2*hd : h*2*hd + hd]
        //   Gate[t, h*hd : (h+1)*hd] = src[t, h*2*hd + hd : (h+1)*2*hd]
        for (int h_idx = 0; h_idx < nh; h_idx++) {
            // Q: copy hd elements per head per token
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(
                static_cast<char*>(qv.data) + h_idx * hd * es_q,          // dst: Q head h
                static_cast<size_t>(q_actual_dim) * es_q,                  // dst pitch (full Q row)
                static_cast<char*>(q_target.data) + h_idx * 2 * hd * es_q, // src: interleaved Q_h
                static_cast<size_t>(q_out_dim) * es_q,                     // src pitch (full QG row)
                static_cast<size_t>(hd) * es_q,                            // width (one head)
                n,                                                          // height (n tokens)
                cudaMemcpyDeviceToDevice, stream));
            // Gate: copy hd elements per head per token
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(
                static_cast<char*>(attn_gate_buf.data) + h_idx * hd * es_q,
                static_cast<size_t>(q_actual_dim) * es_q,
                static_cast<char*>(q_target.data) + (h_idx * 2 + 1) * hd * es_q,
                static_cast<size_t>(q_out_dim) * es_q,
                static_cast<size_t>(hd) * es_q,
                n,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    // 4+5+6. QK-norm + RoPE: fused into single kernel for decode (n=1)
    //    For prefill or models without QK-norm, use separate kernels.
    //    For decode with FP16 cache: fuse K-RoPE into KV write (saves 1 launch).
    bool rope_k_deferred = false;  // true when K-RoPE will be fused into KV write
    {
        bool has_qk_norm = (ly.attn_q_norm.data != nullptr && ly.attn_k_norm.data != nullptr);
        // Determine if we can fuse K-RoPE into KV cache write
        bool can_fuse_rope_kv = (!state.is_prefill && n == 1 &&
                                  qv.dtype == DType::FP16 &&
                                  state.kv_cache &&
                                  state.kv_cache->dtype() == DType::FP16);
        // Per-layer rope_dim. Gemma 4: SWA layers (hd=256) rotate full head_dim,
        // but Global layers (hd=512) use partial_rotary_factor=0.25 per HF config
        // → only first 128 of 512 dims are rotated. Matches llama.cpp rope_freqs
        // tensor length (loaded as ly.rope_freqs for global layers).
        int fused_rope_dim = cfg.rope_dim;
        if (cfg.arch == ModelArch::GEMMA4) {
            bool is_swa_l = (!cfg.swa_layers.empty() && layer < (int)cfg.swa_layers.size() &&
                             cfg.swa_layers[layer]);
            // SWA: hd=256, rope_dim=256 (full). Global: hd=512, rope_dim=256 (half).
            // llama: "full_attention layer only use half of the RoPE dimensions"
            // SWA: rope_dim = hd = 256 (full rotation)
            // Global: rope_dim = hd/2 = 256 (half rotation, matching llama's n_rot_full/2)
            fused_rope_dim = is_swa_l ? hd : (hd / 2);
        } else if (fused_rope_dim > hd || fused_rope_dim <= 0) {
            fused_rope_dim = hd;
        }
        static bool no_qknorm_fused = getenv("IMP_NO_QKNORM_FUSED") != nullptr;
        if (has_qk_norm && n == 1 && qv.dtype == DType::FP16 && !no_qknorm_fused) {
            // Fused: QK-norm + RoPE in one kernel launch (saves 2 launches)
            qknorm_rope_fused(static_cast<half*>(qv.data),
                               static_cast<half*>(kk.data),
                               static_cast<const half*>(ly.attn_q_norm.data),
                               static_cast<const half*>(ly.attn_k_norm.data),
                               nh, nkv, hd, eps,
                               state.positions,  // device pointer
                               layer_rope_theta, layer_rope_freq_scale,
                               fused_rope_dim, cfg.rope_neox, stream, norm_w_off_,
                               cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                               cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr,
                               longrope_freqs);
        } else if (can_fuse_rope_kv && !has_qk_norm) {
            // Fused path: Q-only RoPE here, K-RoPE deferred to KV write
            const int effective_rope_dim = fused_rope_dim;
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            rope_q_only_fp16_kernel<<<dim3(1, nh), pairs, 0, stream>>>(
                static_cast<half*>(qv.data), state.positions,
                nh, hd, layer_rope_theta, inv_scaling, pairs, cfg.rope_neox,
                longrope_freqs);
            rope_k_deferred = true;
        } else {
            // Separate path: QK-norm (if present) + RoPE on both Q and K
            if (ly.attn_q_norm.data != nullptr) {
                int64_t q_flat[2] = {static_cast<int64_t>(n) * nh, static_cast<int64_t>(hd)};
                Tensor q_flat_view = qv.reshape(2, q_flat);
                rmsnorm(q_flat_view, ly.attn_q_norm, q_flat_view, eps, stream, norm_w_off_);
            }
            if (ly.attn_k_norm.data != nullptr) {
                int64_t k_flat[2] = {static_cast<int64_t>(n) * nkv, static_cast<int64_t>(hd)};
                Tensor k_flat_view = kk.reshape(2, k_flat);
                rmsnorm(k_flat_view, ly.attn_k_norm, k_flat_view, eps, stream, norm_w_off_);
            }
            int64_t q4r[4] = {1, n, nh,  hd};
            int64_t k4r[4] = {1, n, nkv, hd};
            Tensor q4r_t = qv.reshape(4, q4r);
            Tensor k4r_t = kk.reshape(4, k4r);
            // Per-layer rope_dim. Gemma 4: SWA full rotation, Global partial 1/4.
            int layer_rope_dim = cfg.rope_dim;
            if (cfg.arch == ModelArch::GEMMA4) {
                bool is_swa_l = (!cfg.swa_layers.empty() && layer < (int)cfg.swa_layers.size() &&
                                 cfg.swa_layers[layer]);
                // SWA: hd=256, rope_dim=256 (full). Global: hd=512, rope_dim=256 (half).
                layer_rope_dim = is_swa_l ? hd : (hd / 2);
            } else if (layer_rope_dim > hd || layer_rope_dim <= 0) {
                layer_rope_dim = hd;  // safety clamp
            }
            rope_forward(q4r_t, k4r_t, state.positions, hd, layer_rope_theta, layer_rope_freq_scale,
                         layer_rope_dim, cfg.rope_neox,
                         cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                         cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, stream,
                         longrope_freqs);
        }
    }


    if (debug_attn_steps) {
        debug_tensor_stats("L0_step2_after_rope_q", qv, stream);
        debug_tensor_stats("L0_step2_after_rope_k", kk, stream);
    }

    // 7. Attention scale.
    //   Standard archs: 1/sqrt(head_dim).
    //   Gemma 4: 1.0 (confirmed by llama.cpp print_info: f_attn_scale = 1.0.
    //                 Q-norm and K-norm absorb the per-element scaling).
    float scale = (cfg.arch == ModelArch::GEMMA4)
                  ? 1.0f
                  : (1.0f / std::sqrt(static_cast<float>(hd)));

    if (state.is_prefill) {
        bool sliding_active = (layer_sliding_window > 0 && n > layer_sliding_window);

        // cuBLAS QK^T materialization: faster than flash attention for short prefills
        // (pp<=512). Benchmarked: pp128 cuBLAS 3270 vs FMHA 2918 (+12%), pp512 ~equal.
        // Falls back to flash attention for long sequences, sliding window, or when
        // the S-matrix buffer wasn't allocated (VRAM-constrained).
        // Set IMP_NO_CUBLAS_ATTN=1 to force flash attention (for benchmarking).
        // Gemma 4: flash attention kernels don't support head_dim=512, so we MUST
        // use cuBLAS attention for all layers (it handles arbitrary head_dim).
        static bool no_cublas_attn = getenv("IMP_NO_CUBLAS_ATTN");
        bool force_cublas_attn = per_layer_shapes;  // Gemma 4 dual head_dim
        if ((force_cublas_attn || !no_cublas_attn) && attn_scores_buf_ && n <= static_cast<int>(attn_scores_.shape[1]) &&
            n <= 1024 && !sliding_active) {
            int64_t s_shape[3] = {static_cast<int64_t>(nh),
                                  static_cast<int64_t>(n),
                                  static_cast<int64_t>(n)};
            Tensor s_view(attn_scores_buf_, DType::FP16, 3, s_shape, true);

            attention_cublas_prefill(qv, kk, vv, ao, s_view,
                                     nh, nkv, hd, scale, /*causal=*/true,
                                     cfg.attn_logit_softcap, stream);
        } else {
            // Flash attention: tiled O(n) memory, handles softcap + sliding window.
            // Dispatch chain: CUTLASS FMHA → Blackwell WMMA → Hopper WMMA → scalar.
            int64_t q4s[4]  = {1, n, nh,  hd};
            int64_t kv4s[4] = {1, n, nkv, hd};
            int64_t o4s[4]  = {1, n, nh,  hd};

            Tensor q4  = qv.reshape(4, q4s);
            Tensor k4  = kk.reshape(4, kv4s);
            Tensor v4  = vv.reshape(4, kv4s);
            Tensor o4  = ao.reshape(4, o4s);

            attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true,
                                       layer_sliding_window, cfg.attn_logit_softcap, stream);
        }

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream);
    } else {
        // Decode: write new token's K/V to cache first
        if (rope_k_deferred) {
            // Fused: apply RoPE to K during KV cache write (saves 1 kernel launch)
            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            KVCache* cache = state.kv_cache;
            const int kv_block_size_d = cache->block_size();
            int row_elems    = nkv * hd;
            int block_stride = kv_block_size_d * row_elems;
            int threads = std::min(row_elems, 256);
            // Per-layer rope_dim (same as prefill rope path above)
            int effective_rope_dim;
            if (cfg.arch == ModelArch::GEMMA4) {
                bool is_swa_l = (!cfg.swa_layers.empty() && layer < (int)cfg.swa_layers.size() &&
                                 cfg.swa_layers[layer]);
                // SWA: hd=256, rope_dim=256 (full). Global: hd=512, rope_dim=128 (quarter).
                // Uses hd/4 because rope_freqs are pre-computed with base_freq baked in.
                // The kernel reads longrope_inv_freqs directly (no base_freq multiply).
                effective_rope_dim = is_swa_l ? hd : (hd / 4);
            } else {
                effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
                if (effective_rope_dim > hd) effective_rope_dim = hd;
            }
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            Tensor kv_view = view_tokens(k_, n);
            Tensor vv_view = view_tokens(v_, n);
            dim3 fused_grid(n, 2);
            write_kv_cache_rope_fused_kernel<<<fused_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv_view.data),
                static_cast<const half*>(vv_view.data),
                state.positions, state.block_tables,
                static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)),
                block_stride, row_elems, kv_block_size_d, n,
                state.max_blocks_per_seq, state.n_sequences,
                nkv, hd, layer_rope_theta, inv_scaling, pairs, cfg.rope_neox,
                longrope_freqs);
        } else {
            write_kv_cache(layer, state, stream);
        }

        // DEBUG: force cuBLAS attention for decode to isolate paged attention bugs.
        // When enabled, uses the same materialized QK^T path as prefill.
        static bool force_cublas_decode = (getenv("IMP_FORCE_CUBLAS_DECODE") != nullptr);
        if (force_cublas_decode && n == 1 && attn_scores_buf_) {
            // Reconstruct K/V from cache for this position
            KVCache* cache_dbg = state.kv_cache;
            int kv_layer_dbg = get_kv_layer(kv_layer_map_, layer);
            int ctx_len = 0;
            cudaMemcpy(&ctx_len, state.context_lens, sizeof(int), cudaMemcpyDeviceToHost);
            // Allocate temp K/V for all context tokens
            int kv_elems = ctx_len * nkv * hd;
            half *k_flat = nullptr, *v_flat = nullptr;
            cudaMalloc(&k_flat, kv_elems * sizeof(half));
            cudaMalloc(&v_flat, kv_elems * sizeof(half));
            // Copy from paged KV cache to contiguous buffer
            int kv_bs = cache_dbg->block_size();
            int32_t h_block_table[1024];
            int n_blocks = (ctx_len + kv_bs - 1) / kv_bs;
            cudaMemcpy(h_block_table, state.block_tables, n_blocks * sizeof(int32_t), cudaMemcpyDeviceToHost);
            for (int b = 0; b < n_blocks; b++) {
                int block_id = h_block_table[b];
                int toks_in_block = std::min(kv_bs, ctx_len - b * kv_bs);
                size_t row_bytes = nkv * hd * sizeof(half);
                half* k_src = static_cast<half*>(cache_dbg->k_ptr(kv_layer_dbg, block_id));
                half* v_src = static_cast<half*>(cache_dbg->v_ptr(kv_layer_dbg, block_id));
                cudaMemcpy(k_flat + b * kv_bs * nkv * hd, k_src, toks_in_block * row_bytes, cudaMemcpyDeviceToDevice);
                cudaMemcpy(v_flat + b * kv_bs * nkv * hd, v_src, toks_in_block * row_bytes, cudaMemcpyDeviceToDevice);
            }
            // Reshape for cuBLAS attention: Q[1,nh,hd], K[ctx_len,nkv,hd], V[ctx_len,nkv,hd]
            int64_t k_shape[2] = {ctx_len, nkv * hd};
            int64_t v_shape[2] = {ctx_len, nkv * hd};
            Tensor k_cont(k_flat, DType::FP16, 2, k_shape, true);
            Tensor v_cont(v_flat, DType::FP16, 2, v_shape, true);
            // Use n=1 cuBLAS attention with causal=false (all context visible)
            int64_t s_shape[3] = {(int64_t)nh, 1, (int64_t)ctx_len};
            half* s_buf = nullptr;
            cudaMalloc(&s_buf, nh * ctx_len * sizeof(half));
            Tensor s_view(s_buf, DType::FP16, 3, s_shape, true);
            attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view,
                                     nh, nkv, hd, scale, /*causal=*/false,
                                     cfg.attn_logit_softcap, stream);
            cudaFree(k_flat);
            cudaFree(v_flat);
            cudaFree(s_buf);
            goto after_attention;
        }

        // Paged attention: Q shape depends on batch size
        int n_seq = state.n_sequences;
        // For decode, n_tokens == n_sequences (one token per seq)
        int64_t qd[4] = {n_seq, 1, nh, hd};
        int64_t od[4] = {n_seq, 1, nh, hd};
        Tensor q4 = qv.reshape(4, qd);
        Tensor o4 = ao.reshape(4, od);

        KVCache* cache = state.kv_cache;
        const int kv_bs = cache->block_size();
        int total_blk  = cache->total_blocks();
        DType cache_dtype = cache->dtype();
        int64_t cs[4]  = {static_cast<int64_t>(total_blk),
                          static_cast<int64_t>(kv_bs),
                          static_cast<int64_t>(nkv),
                          static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = get_kv_layer(kv_layer_map_, layer);
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        // L2 persistence hint: keep this layer's KV cache in L2 during attention.
        // RTX 5090 has 96 MB L2 — enough for ~3K tokens of KV at FP8.
        set_l2_persist_kv(stream, k_c.data, k_c.nbytes() + v_c.nbytes());

        if (cache_dtype == DType::TURBOQUANT_LITE) {
            // TurboQuant Lite paged attention: QJL sketch-only K + INT4 V (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_turboquant_lite(q4, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(qjl_proj_.matrix),
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale, qjl_proj_.sketch_dim,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq);
        } else if (cache_dtype == DType::TURBOQUANT) {
            // TurboQuant paged attention: PolarQuant K + QJL correction + INT4 V (Split-K enabled)
            // K_mscales: non-null if MXFP4 path (FP4 E2M1 + UE8M0), null for uniform INT4
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            const uint8_t* k_mscales = cache->use_mxfp4()
                ? static_cast<const uint8_t*>(cache->k_mscale_ptr(kv_layer, 0))
                : nullptr;
            paged_attention_decode_turboquant(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(qjl_proj_.matrix),
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale, qjl_proj_.sketch_dim,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq,
                                        k_mscales);
        } else if (cache_dtype == DType::INT4) {
            // INT4 paged attention with per-head scales and INT4 unpack (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int4(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq);
        } else if (cache_dtype == DType::INT8) {
            // INT8 dp4a paged attention with per-head scales (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int8(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq);
        } else if (cache_dtype == DType::FP8_E4M3) {
            // FP8 paged attention with on-the-fly dequant (Split-K enabled)
            float kv_scale = (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size()))
                             ? kv_scales_[kv_layer] : 1.0f;
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_fp8(q4, k_c, v_c, o4,
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale, kv_scale,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq);
        } else {
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode(q4, k_c, v_c, o4,
                                    state.block_tables, state.context_lens,
                                    kv_bs, scale, state.max_context_len,
                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                    state.max_blocks_per_seq);
        }

        // Clear L2 persistence hint (weights loaded next need L2 space)
        clear_l2_persist(stream);
    }

    after_attention:

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step3_after_paged_attn", ao, stream);
        debug_tensor_stats("L0_step3_h_before_oproj", h, stream);
    }

    // Qwen3.5 attention output gate: ao[i] *= sigmoid(gate[i])
    if (has_attn_output_gate) {
        sigmoid_mul(ao, attn_gate_buf, ao, stream);
    }

    // 8+9. O projection + residual connection.
    //    For decode (n=1) with dp4a: fuse residual add into GEMV, write directly
    //    to hidden buffer. When will_fuse_o_residual is set, we skipped the
    //    initial h→r memcpy and use h.data itself as the residual source.
    //    This is safe because h.data is only READ (never written) between the
    //    start of run_attention and this point.
    if (will_fuse_o_nvfp4) {
        // NVFP4 Wo + residual: attn_out (FP16) @ wo_nvfp4^T + residual → hidden
        auto& wo_nvfp4 = wcache_.nvfp4.at(ly.wo.data);
        int M_o = static_cast<int>(wo_nvfp4.N);
        int K_o = static_cast<int>(wo_nvfp4.K);
        gemv_nvfp4_residual(wo_nvfp4,
                             static_cast<const half*>(ao.data),
                             static_cast<half*>(h.data),
                             static_cast<const half*>(h.data),
                             M_o, K_o, stream);
    } else if (will_fuse_o_residual) {
        int K_o = static_cast<int>(ly.wo.shape[1]);
        int M_o = static_cast<int>(ly.wo.shape[0]);
        // Separate quant + K-parallel GEMV: higher warp occupancy than inline_quant.
        // quantize_fp16_to_q8_1 is a lightweight kernel (~2 us for d_model=3072).
        // The K-parallel GEMV achieves 48 warps/SM vs inline_quant's ~8 warps/SM.
        const half* attn_fp16 = static_cast<const half*>(ao.data);
        const half* residual_ptr = static_cast<const half*>(h.data);
        quantize_fp16_to_q8_1(attn_fp16, static_cast<block_q8_1*>(qscratch_.q8_1_buf),
                               qscratch_.d8_buf, K_o, stream);
        dispatch_gemv_residual(ly.wo_qtype, ly.wo.data,
                               static_cast<block_q8_1*>(qscratch_.q8_1_buf),
                               qscratch_.d8_buf, static_cast<half*>(h.data), residual_ptr,
                               M_o, K_o, stream);
    } else if (will_fuse_o_beta1 && !cur_force_fp16_ &&
               wcache_.fp8.count(ly.wo.data) &&
               qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
        // FP8 beta=1: hidden = fp8(attn_out) @ fp8(wo)^T + hidden
        auto& e = wcache_.fp8.at(ly.wo.data);
        Tensor fp8_ao(qscratch_.fp8_act, DType::FP8_E4M3, ao.ndim, ao.shape, true);
        quantize_fp16_to_fp8_e4m3(ao, fp8_ao, qscratch_.d_act_scale, stream,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
        gemm_cublaslt(fp8_ao, e.weight, h, 1.0f, 1.0f, qscratch_.d_act_scale, e.d_scale, stream);
    } else if (will_fuse_o_beta1 && wcache_.fp16.count(ly.wo.data)) {
        // Fused: hidden = attn_out @ wo^T + hidden (cuBLAS beta=1).
        // Safe: hidden is only READ (never written) between attn_norm and here.
        const Tensor& wo_fp16 = wcache_.fp16.at(ly.wo.data);
        // TODO: migrate to gemm_dispatch with beta=1.0
        gemm(ao, wo_fp16, h, 1.0f, 1.0f, stream);
    } else if ((will_fuse_o_beta1 || will_fuse_o_dequant_beta1) &&
               qscratch_.dequant != nullptr && dequant_gpu_supported(ly.wo_qtype) &&
               !per_layer_shapes) {  // Gemma 4: workspace stride mismatch with narrow ao
        // Dequant beta=1: dequant weights on-the-fly, then FP16 GEMM + residual
        int rows = static_cast<int>(ly.wo.shape[0]);
        int cols = static_cast<int>(ly.wo.shape[1]);
        dequant_gpu(ly.wo.data, qscratch_.dequant, ly.wo_qtype, rows, cols, stream);
        Tensor w_fp16(qscratch_.dequant, DType::FP16, ly.wo.ndim, ly.wo.shape, true);
        // TODO: migrate to gemm_dispatch with beta=1.0
        gemm(ao, w_fp16, h, 1.0f, 1.0f, stream);
    } else {
        // Fallback: separate O-projection + optional post-norm + residual add
        gemm_dispatch(ao, ly.wo, ly.wo_qtype, po, ctx);
        if (debug_attn_steps) {
            debug_tensor_stats_all("L0_ao_pre_wo",  view_tokens(ao, n), stream);
            debug_tensor_stats_all("L0_po_after_wo", view_tokens(po, n), stream);
            debug_tensor_rows    ("po_wo-0",        view_tokens(po, n), stream);
            debug_tensor_rows    ("ao_pre_wo-0",    view_tokens(ao, n), stream);
            // dump wo weight shape info
            fprintf(stderr, "[DEBUG_FWD] wo_shape: ndim=%d shape=[%ld,%ld] qtype=%d\n",
                    ly.wo.ndim, (long)ly.wo.shape[0], (long)ly.wo.shape[1], (int)ly.wo_qtype);
        }
        if (has_post_attn_norm && using_fp32_accum) {
            // Sandwich norm with FP32 accumulator (Gemma-3):
            // FP32 residual += attn_out, then post_attn_norm → FP16 hidden.
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            float eps = model_->config().rms_norm_eps;
            // Add attn output to FP32 accumulator, apply post_attn_norm, write FP16
            // 256 threads: d_model_v = d_model/8 (e.g. 480 for Gemma-3 3840),
            // so 2 iterations/thread. 512 wastes half the threads on idle lanes.
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                static_cast<const half*>(po.data),
                static_cast<const half*>(ly.post_attn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                model_->config().d_model, eps, norm_w_off_);
        } else if (has_post_attn_norm && model_->config().arch == ModelArch::GEMMA4) {
            // Gemma 4 sandwich norm: h = r + post_attn_norm(po).
            // Normalize attention output first, THEN add residual (HF reference order).
            rmsnorm(po, ly.post_attn_norm, po,
                    model_->config().rms_norm_eps, stream, norm_w_off_);
            elementwise_add_store(po, r, h, stream);
        } else if (has_post_attn_norm) {
            // Sandwich norm without FP32 accumulator: h = rmsnorm(po + r)
            // Fused: 3 ops (add_store + rmsnorm + memcpy) → 1 kernel
            add_rmsnorm_inplace(po, r, h, ly.post_attn_norm,
                                model_->config().rms_norm_eps, stream, norm_w_off_);
        } else {
            // Standard pre-norm: h = attn_out + residual
            elementwise_add_store(po, r, h, stream);
        }
    }
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step4_after_oproj_residual", h, stream);
        debug_tensor_rows("step4_h-0", view_tokens(h, n), stream);
        debug_tensor_stats_all("L0_step4_post_attn_all", view_tokens(h, n), stream);
    }

}

} // namespace imp

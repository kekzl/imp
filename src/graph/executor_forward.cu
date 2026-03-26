#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#ifdef IMP_USE_CUTLASS
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/attention_cutlass_fmha.h"
#endif
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

// Returns true if the quant type supports dp4a (Q8_1-input) GEMV kernels.
static inline bool is_dp4a_qtype(GGMLQuantType qt) {
    return qt == GGMLQuantType::Q6_K || qt == GGMLQuantType::Q8_0 ||
           qt == GGMLQuantType::Q4_0 || qt == GGMLQuantType::Q4_K ||
           qt == GGMLQuantType::Q5_K || qt == GGMLQuantType::Q2_K ||
           qt == GGMLQuantType::Q3_K;
}

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

// Residual-fused GEMV dispatch by quant type: y[i] = dot(W[i], x) + residual[i].
static void dispatch_gemv_residual(GGMLQuantType qtype,
                                    const void* W, const block_q8_1* q8_1, const float* d8,
                                    half* y, const half* residual,
                                    int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case GGMLQuantType::Q6_K: gemv_q6k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        case GGMLQuantType::Q4_0: gemv_q4_0_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        case GGMLQuantType::Q4_K: gemv_q4_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        case GGMLQuantType::Q5_K: gemv_q5_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        case GGMLQuantType::Q2_K: gemv_q2_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        case GGMLQuantType::Q3_K: gemv_q3_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
        default:                  gemv_q8_0_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream); break;
    }
}

// Plain dp4a GEMV dispatch: y = W @ x (FP16 output).
static void dispatch_gemv_q8_1(GGMLQuantType qtype,
                                const void* W, const block_q8_1* q8_1, const float* d8,
                                half* y, int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case GGMLQuantType::Q6_K: gemv_q6k_q8_1(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q4_0: gemv_q4_0_q8_1(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q4_K: gemv_q4_k_q8_1(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q5_K: gemv_q5_k_q8_1(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q2_K: gemv_q2_k_q8_1(W, q8_1, d8, y, M, K, stream); break;
        case GGMLQuantType::Q3_K: gemv_q3_k_q8_1(W, q8_1, d8, y, M, K, stream); break;
        default:                  gemv_q8_0_q8_1(W, q8_1, d8, y, M, K, stream); break;
    }
}

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

// Map global layer index to KV cache layer index (-1 if not an attention layer).
static inline int get_kv_layer(const std::vector<int>& kv_layer_map, int layer) {
    return kv_layer_map.empty() ? layer : kv_layer_map[layer];
}

// Map global layer index to SSM/GDN state index.
static inline int get_ssm_layer(const std::vector<int>& ssm_layer_map, int layer) {
    return ssm_layer_map[layer];
}

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state,
                                   cudaStream_t stream) {
    if (!state.kv_cache || !state.block_tables) return;

    // Map global layer index to KV cache layer index
    int kv_layer = get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0) return;  // not an attention layer

    KVCache* cache = state.kv_cache;
    int n        = state.n_tokens;
    int nkv      = cache->n_kv_heads();
    int hd       = cache->head_dim();
    const int kv_block_size = cache->block_size();
    int row_elems    = nkv * hd;
    int block_stride = kv_block_size * row_elems;

    int threads = std::min(row_elems, 256);
    int nblocks = n;   // one CUDA block per token

    bool use_fp8 = (cache->dtype() == DType::FP8_E4M3);
    bool use_int8 = (cache->dtype() == DType::INT8);
    bool use_int4 = (cache->dtype() == DType::INT4);
    bool use_turboquant = (cache->dtype() == DType::TURBOQUANT);
    bool use_turboquant_lite = (cache->dtype() == DType::TURBOQUANT_LITE);

    if (use_turboquant_lite) {
        // TurboQuant Lite: QJL sketch-only K + INT4 V
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;
        int scale_block_stride_tql = kv_block_size * nkv;
        int sketch_dim = qjl_proj_.sketch_dim;
        int sketch_block_stride = kv_block_size * nkv * (sketch_dim / 8);
        dim3 grid_tql(n, 2);
        write_kv_cache_turboquant_lite_kernel<<<grid_tql, 256, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
            static_cast<const uint8_t*>(qjl_proj_.matrix),
            int4_block_stride, scale_block_stride_tql, sketch_block_stride,
            nkv, hd, sketch_dim,
            kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    } else if (use_turboquant) {
        // TurboQuant KV cache write: PolarQuant INT4 directions + QJL sketch for K, INT4 for V
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;
        int scale_block_stride_tq = kv_block_size * nkv;
        int sketch_dim = qjl_proj_.sketch_dim;
        int sketch_block_stride = kv_block_size * nkv * (sketch_dim / 8);
        dim3 grid_tq(n, 2);
        write_kv_cache_turboquant_kernel<<<grid_tq, 256, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
            static_cast<const uint8_t*>(qjl_proj_.matrix),
            int4_block_stride, scale_block_stride_tq, sketch_block_stride,
            nkv, hd, sketch_dim,
            kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    } else if (use_int4) {
        // INT4 quantized KV cache write — 2 values packed per byte, per-head scales
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;  // bytes (half the INT8 stride)
        int scale_block_stride = kv_block_size * nkv;
        dim3 grid_int4(n, 2);
        write_kv_cache_int4_kernel<<<grid_int4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
            int4_block_stride, scale_block_stride, nkv, hd,
            kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    } else if (use_int8) {
        // INT8 quantized KV cache write path with per-head scales.
        // No per-layer calibration needed — scales are computed per-head at write time.
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);

        int scale_block_stride = kv_block_size * nkv;  // half elems per scale block
        dim3 grid_int8(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_int8_kernel<<<grid_int8, 256, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<int8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<int8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
            block_stride, scale_block_stride, nkv, hd,
            kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    } else if (use_fp8) {
        // FP8 E4M3 quantized KV cache write path with online calibration.
        // On first write to each KV layer, calibrate scale from K/V data.
        float inv_scale;
        if (!kv_calibrated_.empty() && kv_layer < static_cast<int>(kv_calibrated_.size()) &&
            !kv_calibrated_[kv_layer]) {
            // Calibrate from current K/V data: scale = absmax / 448.0
            Tensor kv_cal = view_tokens(k_, n);
            Tensor vv_cal = view_tokens(v_, n);
            float k_scale = calibrate_fp8_scale(kv_cal, stream);
            float v_scale = calibrate_fp8_scale(vv_cal, stream);
            float scale = std::max(k_scale, v_scale);
            if (scale < 1e-12f) scale = 1.0f;  // safety for all-zero
            kv_scales_[kv_layer] = scale;
            kv_calibrated_[kv_layer] = true;
            inv_scale = 1.0f / scale;
        } else if (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size())) {
            inv_scale = 1.0f / kv_scales_[kv_layer];
        } else {
            inv_scale = 1.0f;
        }

        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        // Fused K+V FP8 write: single kernel launch with blockIdx.y
        dim3 fp8_grid(n, 2);
        write_kv_cache_fp8_fused_kernel<<<fp8_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    } else {
        // Standard FP16 KV cache write path — fused K+V in single launch
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        dim3 fused_grid(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_fused_kernel<<<fused_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<half*>(cache->k_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_ptr(kv_layer, 0)),
            block_stride, row_elems, kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    }
}

// ---------------------------------------------------------------------------
// Forward pass diagnostics (IMP_DEBUG_FORWARD=1)
// ---------------------------------------------------------------------------

bool debug_forward_enabled() {
    static const bool enabled = (std::getenv("IMP_DEBUG_FORWARD") != nullptr);
    return enabled;
}

// Print min/max/mean/L2norm of a GPU tensor (first row only for multi-row tensors).
// Syncs the stream — only call when IMP_DEBUG_FORWARD is active.
void debug_tensor_stats(const char* name, const Tensor& t, cudaStream_t stream,
                                int row = 0, int max_rows = 1) {
    if (!debug_forward_enabled()) return;
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = std::min(max_rows, static_cast<int>(t.shape[0]) - row);
    int n = cols * nrows;
    std::vector<float> host(n);

    if (t.dtype == DType::FP16) {
        std::vector<half> tmp(n);
        cudaMemcpyAsync(tmp.data(), static_cast<const half*>(t.data) + (int64_t)row * cols,
                         n * sizeof(half), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < n; i++) host[i] = __half2float(tmp[i]);
    } else if (t.dtype == DType::FP32) {
        cudaMemcpyAsync(host.data(), static_cast<const float*>(t.data) + (int64_t)row * cols,
                         n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        fprintf(stderr, "[DEBUG_FWD] %s: unsupported dtype %d\n", name, (int)t.dtype);
        return;
    }

    float vmin = host[0], vmax = host[0], vsum = 0, vl2 = 0;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n; i++) {
        float v = host[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
        vl2 += v * v;
    }
    float mean = vsum / std::max(n - nan_count - inf_count, 1);
    float l2 = std::sqrt(vl2);
    fprintf(stderr, "[DEBUG_FWD] %-30s  min=%+.6e  max=%+.6e  mean=%+.6e  L2=%.6e",
            name, vmin, vmax, mean, l2);
    if (nan_count > 0) fprintf(stderr, "  NaN=%d", nan_count);
    if (inf_count > 0) fprintf(stderr, "  Inf=%d", inf_count);
    fprintf(stderr, "\n");
}

// Print top-k logits with token IDs
void debug_top_logits(const Tensor& logits, cudaStream_t stream, int topk = 10) {
    if (!debug_forward_enabled()) return;
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    std::vector<float> host(vocab);

    if (logits.dtype == DType::FP32) {
        cudaMemcpyAsync(host.data(), logits.data, vocab * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    } else if (logits.dtype == DType::FP16) {
        std::vector<half> tmp(vocab);
        cudaMemcpyAsync(tmp.data(), logits.data, vocab * sizeof(half),
                         cudaMemcpyDeviceToHost, stream);
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
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
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
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
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
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // For Qwen3.5: Q projection writes to larger buffer (includes gate), then split
    Tensor q_target = has_attn_output_gate ? qv_full : qv;

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    For decode (n=1) with matching quant types: fused RMSNorm→Q8_1→QKV GEMV.
    //    This skips the intermediate norm_out FP16 buffer entirely.
    //    Otherwise falls back to separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        // NVFP4 decode path: uses FP16 input (no Q8_1 quantization needed)
        auto nvfp4_wq = wcache_.nvfp4.find(ly.wq.data);
        auto nvfp4_wk = wcache_.nvfp4.find(ly.wk.data);
        auto nvfp4_wv = wcache_.nvfp4.find(ly.wv.data);
        bool nvfp4_qkv = (!has_attn_output_gate && n == 1 && nvfp4_wq != wcache_.nvfp4.end() &&
                          nvfp4_wk != wcache_.nvfp4.end() && nvfp4_wv != wcache_.nvfp4.end());
        bool fused_qkv = (!has_attn_output_gate && n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                          no.dtype == DType::FP16 &&
                          ly.wq_qtype == ly.wk_qtype && ly.wk_qtype == ly.wv_qtype &&
                          is_dp4a_qtype(ly.wq_qtype));
        if (nvfp4_qkv) {
            // NVFP4 fused QKV: RMSNorm to FP16, then NVFP4 GEMV (no Q8_1 needed)
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            int K = static_cast<int>(ly.wq.shape[1]);
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
            // Separate RMSNorm + dispatch
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

            // FP8 prefill path: quantize norm_out→FP8 once, 3 separate FP8 GEMMs
            auto fp8_wq = wcache_.fp8.find(ly.wq.data);
            auto fp8_wk = wcache_.fp8.find(ly.wk.data);
            auto fp8_wv = wcache_.fp8.find(ly.wv.data);
            if (n > 1 && !state.force_fp16_gemm &&
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
                if (n > 1 && fused_kv_it != wcache_.fused_kv.end()) {
                    // Q: still separate (different output dim with GQA)
                    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, q_target,
                                  qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4,
                                  (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4,
                                  qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4,
                                  qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                    // K+V: one batched cuBLAS call
                    gemm_kv_batched(no, fused_kv_it->second, kk, vv, stream);
                } else {
                    const auto* nv4p = (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4;
                    const auto* ct4p = (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4;
                    const auto* mx4p = (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4;
                    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, q_target, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  nv4p, ct4p, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                    gemm_dispatch(no, ly.wk, ly.wk_scales, ly.wk_qtype, kk, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  nv4p, ct4p, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                    gemm_dispatch(no, ly.wv, ly.wv_scales, ly.wv_qtype, vv, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  nv4p, ct4p, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2)
        add_bias(qv, ly.q_bias, stream);
        add_bias(kk, ly.k_bias, stream);
        add_bias(vv, ly.v_bias, stream);
    }
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step1_after_qkv_q", qv, stream);
        debug_tensor_stats("L0_step1_after_qkv_k", kk, stream);
    }

    // Per-layer RoPE theta and sliding window (Gemma-3: alternating local/global layers)
    float layer_rope_theta = cfg.rope_theta;
    float layer_rope_freq_scale = cfg.rope_freq_scale;
    int layer_sliding_window = cfg.sliding_window;
    if (cfg.sliding_window_pattern > 0) {
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
            cudaMemcpy2DAsync(
                static_cast<char*>(qv.data) + h_idx * hd * es_q,          // dst: Q head h
                static_cast<size_t>(q_actual_dim) * es_q,                  // dst pitch (full Q row)
                static_cast<char*>(q_target.data) + h_idx * 2 * hd * es_q, // src: interleaved Q_h
                static_cast<size_t>(q_out_dim) * es_q,                     // src pitch (full QG row)
                static_cast<size_t>(hd) * es_q,                            // width (one head)
                n,                                                          // height (n tokens)
                cudaMemcpyDeviceToDevice, stream);
            // Gate: copy hd elements per head per token
            cudaMemcpy2DAsync(
                static_cast<char*>(attn_gate_buf.data) + h_idx * hd * es_q,
                static_cast<size_t>(q_actual_dim) * es_q,
                static_cast<char*>(q_target.data) + (h_idx * 2 + 1) * hd * es_q,
                static_cast<size_t>(q_out_dim) * es_q,
                static_cast<size_t>(hd) * es_q,
                n,
                cudaMemcpyDeviceToDevice, stream);
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
        if (has_qk_norm && n == 1 && qv.dtype == DType::FP16) {
            // Fused: QK-norm + RoPE in one kernel launch (saves 2 launches)
            qknorm_rope_fused(static_cast<half*>(qv.data),
                               static_cast<half*>(kk.data),
                               static_cast<const half*>(ly.attn_q_norm.data),
                               static_cast<const half*>(ly.attn_k_norm.data),
                               nh, nkv, hd, eps,
                               state.positions,  // device pointer
                               layer_rope_theta, layer_rope_freq_scale,
                               cfg.rope_dim, cfg.rope_neox, stream, norm_w_off_,
                               cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                               cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr,
                               longrope_freqs);
        } else if (can_fuse_rope_kv && !has_qk_norm) {
            // Fused path: Q-only RoPE here, K-RoPE deferred to KV write
            const int effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
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
            rope_forward(q4r_t, k4r_t, state.positions, hd, layer_rope_theta, layer_rope_freq_scale,
                         cfg.rope_dim, cfg.rope_neox,
                         cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                         cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, stream,
                         longrope_freqs);
        }
    }


    if (debug_attn_steps) {
        debug_tensor_stats("L0_step2_after_rope_q", qv, stream);
        debug_tensor_stats("L0_step2_after_rope_k", kk, stream);
    }

    // 7. Attention
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    if (state.is_prefill) {
        bool sliding_active = (layer_sliding_window > 0 && n > layer_sliding_window);

        // cuBLAS QK^T materialization: faster than flash attention for short prefills
        // (pp<=512). Benchmarked: pp128 cuBLAS 3270 vs FMHA 2918 (+12%), pp512 ~equal.
        // Falls back to flash attention for long sequences, sliding window, or when
        // the S-matrix buffer wasn't allocated (VRAM-constrained).
        // Set IMP_NO_CUBLAS_ATTN=1 to force flash attention (for benchmarking).
        static bool no_cublas_attn = getenv("IMP_NO_CUBLAS_ATTN");
        if (!no_cublas_attn && attn_scores_buf_ && n <= static_cast<int>(attn_scores_.shape[1]) &&
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
            const int effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
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
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_turboquant(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
                                        static_cast<const uint8_t*>(qjl_proj_.matrix),
                                        state.block_tables, state.context_lens,
                                        kv_bs, scale, qjl_proj_.sketch_dim,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream,
                                        state.max_blocks_per_seq);
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
    }


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
        int M_o = static_cast<int>(ly.wo.shape[0]);
        int K_o = static_cast<int>(ly.wo.shape[1]);
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
        gemm(ao, wo_fp16, h, 1.0f, 1.0f, stream);
    } else if ((will_fuse_o_beta1 || will_fuse_o_dequant_beta1) &&
               qscratch_.dequant != nullptr && dequant_gpu_supported(ly.wo_qtype)) {
        // Dequant beta=1: dequant weights on-the-fly, then FP16 GEMM + residual
        int rows = static_cast<int>(ly.wo.shape[0]);
        int cols = static_cast<int>(ly.wo.shape[1]);
        dequant_gpu(ly.wo.data, qscratch_.dequant, ly.wo_qtype, rows, cols, stream);
        Tensor w_fp16(qscratch_.dequant, DType::FP16, ly.wo.ndim, ly.wo.shape, true);
        gemm(ao, w_fp16, h, 1.0f, 1.0f, stream);
    } else {
        // Fallback: separate O-projection + optional post-norm + residual add
        gemm_dispatch(ao, ly.wo, ly.wo_scales, ly.wo_qtype, po, qscratch_.dequant, stream,
                      static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                      (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                      qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                      (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4,
                      (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4,
                      qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                      (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4,
                      qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
        if (has_post_attn_norm && using_fp32_accum) {
            // Sandwich norm with FP32 accumulator (Gemma-3):
            // FP32 residual += attn_out, then post_attn_norm → FP16 hidden.
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            float eps = model_->config().rms_norm_eps;
            // Add attn output to FP32 accumulator, apply post_attn_norm, write FP16
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 512, 0, stream>>>(
                static_cast<const half*>(po.data),
                static_cast<const half*>(ly.post_attn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                model_->config().d_model, eps, norm_w_off_);
        } else if (has_post_attn_norm) {
            // Sandwich norm without FP32 accumulator: add + norm → h
            elementwise_add_store(po, r, h, stream);
            Tensor no = view_tokens(norm_out_, n);
            rmsnorm(h, ly.post_attn_norm, no, model_->config().rms_norm_eps, stream, norm_w_off_);
            cudaMemcpyAsync(h.data, no.data, h.nbytes(),
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            // Standard pre-norm: h = attn_out + residual
            elementwise_add_store(po, r, h, stream);
        }
    }
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step4_after_oproj_residual", h, stream);
    }

}

// ---------------------------------------------------------------------------
// FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for dense FFN phase
    configure_ffn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    // cur_n_tokens_ is set by forward_logits before the layer loop.
    int n   = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;

    Tensor h  = view_tokens(hidden_,     n);
    Tensor r  = view_tokens(residual_,   n);
    Tensor no = view_tokens(norm_out_,   n);
    Tensor go = view_tokens(gate_out_,   n);
    Tensor uo = view_tokens(up_out_,     n);
    Tensor so = view_tokens(swiglu_out_, n);
    Tensor fo = view_tokens(ffn_out_,    n);

    // 1. Save residual (skip if fused down-proj+residual will handle it).
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    // Qwen3.5: uses post_attn_norm instead of ffn_norm (ffn_norm is null)
    const Tensor& ffn_norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm :
                                (ly.post_attn_norm.data != nullptr) ? ly.post_attn_norm : ly.attn_norm;
    const bool has_post_ffn_norm = (ly.post_ffn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_ffn_norm);
    bool will_fuse_down_nvfp4 = (!has_post_ffn_norm && n == 1 && h.dtype == DType::FP16 &&
                                  wcache_.nvfp4.count(ly.w_down.data));
    bool will_fuse_down_residual = (!has_post_ffn_norm && !will_fuse_down_nvfp4 &&
                                     n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                                     h.dtype == DType::FP16 && is_dp4a_qtype(ly.w_down_qtype));
    bool will_fuse_down_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                  !will_fuse_down_nvfp4 && n > 1 &&
                                  (wcache_.fp16.count(ly.w_down.data) || wcache_.fp8.count(ly.w_down.data)));
    bool will_fuse_down_dequant_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                          !will_fuse_down_nvfp4 &&
                                          !will_fuse_down_beta1 && n > 1 &&
                                          qscratch_.dequant != nullptr &&
                                          dequant_gpu_supported(ly.w_down_qtype));
    if (!will_fuse_down_residual && !will_fuse_down_beta1 &&
        !will_fuse_down_dequant_beta1 && !will_fuse_down_nvfp4 && !using_fp32_accum) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. Gate and Up projections
    //    For decode (n=1): fuse RMSNorm→Q8_1→GEMV to avoid redundant quantization.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        int d = static_cast<int>(h.shape[1]);
        // NVFP4 gate+up decode path
        auto nvfp4_wg = wcache_.nvfp4.find(ly.w_gate.data);
        auto nvfp4_wu = wcache_.nvfp4.find(ly.w_up.data);
        bool nvfp4_ffn = (n == 1 && nvfp4_wg != wcache_.nvfp4.end() &&
                          nvfp4_wu != wcache_.nvfp4.end());
        bool fused_ffn_norm = (n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                               h.dtype == DType::FP16 && is_dp4a_qtype(ly.w_gate_qtype));
        if (nvfp4_ffn) {
            // NVFP4 gate+up: RMSNorm to FP16, then NVFP4 fused GEMV
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            gemv_nvfp4_gate_up_fused(nvfp4_wg->second, nvfp4_wu->second,
                                      static_cast<const half*>(no.data),
                                      static_cast<half*>(go.data),
                                      static_cast<half*>(uo.data),
                                      ffn_rows, d, stream);
        } else if (fused_ffn_norm) {
            // Fused RMSNorm + Q8_1: quantize once, use for both gate and up
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                    static_cast<const half*>(ffn_norm_w.data),
                                    q8, qscratch_.d8_buf, static_cast<half*>(no.data),
                                    d, eps, stream, norm_w_off_);
            // Fused gate+up GEMV: single kernel launch for both projections
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            gemv_gate_up_fused(ly.w_gate.data, ly.w_up.data, q8, qscratch_.d8_buf,
                                static_cast<half*>(go.data),
                                static_cast<half*>(uo.data),
                                ffn_rows, d, ly.w_gate_qtype, stream);
        } else {
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);

            // FP8 prefill path: quantize norm_out→FP8 once, 2 separate FP8 GEMMs
            auto fp8_wg = wcache_.fp8.find(ly.w_gate.data);
            auto fp8_wu = wcache_.fp8.find(ly.w_up.data);
            if (n > 1 && !cur_force_fp16_ &&
                fp8_wg != wcache_.fp8.end() && fp8_wu != wcache_.fp8.end() &&
                qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
                Tensor fp8_no(qscratch_.fp8_act, DType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                gemm_cublaslt(fp8_no, fp8_wg->second.weight, go, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_wg->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wu->second.weight, uo, 1.0f, 0.0f,
                              qscratch_.d_act_scale, fp8_wu->second.d_scale, stream);
            } else {
                auto fused_gu_it = wcache_.fused_gate_up.find(layer);
                if (n > 1 && fused_gu_it != wcache_.fused_gate_up.end()) {
                    // Batched gate+up: single cuBLAS call for both projections
                    gemm_pair_batched(no, fused_gu_it->second, go, uo, stream);
                } else {
                    const auto* nv4p = (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4;
                    const auto* ct4p = (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4;
                    const auto* mx4p = (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4;
                    gemm_dispatch(no, ly.w_gate, ly.w_gate_scales, ly.w_gate_qtype, go, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  nv4p, ct4p, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                    gemm_dispatch(no, ly.w_up,   ly.w_up_scales,   ly.w_up_qtype,   uo, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  nv4p, ct4p, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                }
            }
        }
    }

    // 4+5+6. Gated activation + Down projection + residual add.
    //    For decode (n=1) with dp4a: fuse activation→Q8_1→GEMV+residual.
    //    SwiGLU case: swiglu_quantize_q8_1 fuses activation + Q8_1 in one kernel,
    //    eliminating the intermediate FP16 buffer write and one kernel launch.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        bool fused_down_residual = (!has_post_ffn_norm &&
                                     n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                                     so.dtype == DType::FP16 && is_dp4a_qtype(ly.w_down_qtype));
        if (will_fuse_down_nvfp4) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            auto& wd_nvfp4 = wcache_.nvfp4.at(ly.w_down.data);
            int n_mb_d = K_d / 16;
            if (n_mb_d <= 512) {
                // Small K: fused SwiGLU/GeGLU + NVFP4 GEMV + residual (MR path)
                if (cfg.ffn_activation != FFNActivation::GEGLU) {
                    gemv_nvfp4_swiglu_residual(wd_nvfp4,
                                                static_cast<const half*>(go.data),
                                                static_cast<const half*>(uo.data),
                                                static_cast<half*>(h.data),
                                                static_cast<const half*>(h.data),
                                                M_d, K_d, stream);
                } else {
                    gemv_nvfp4_geglu_residual(wd_nvfp4,
                                               static_cast<const half*>(go.data),
                                               static_cast<const half*>(uo.data),
                                               static_cast<half*>(h.data),
                                               static_cast<const half*>(h.data),
                                               M_d, K_d, stream);
                }
            } else {
                // Large K (e.g. 12288): split activation + GEMV.
                // The fused kernel's silu/gelu compute dominates at large K,
                // while the separate vectorized activation kernel is ~1 μs,
                // then GEMV uses prmt dequant (no smem LUT, no bank conflicts).
                if (cfg.ffn_activation != FFNActivation::GEGLU) {
                    swiglu(go, uo, so, stream);
                } else {
                    geglu(go, uo, so, stream);
                }
                gemv_nvfp4_residual(wd_nvfp4,
                                     static_cast<const half*>(so.data),
                                     static_cast<half*>(h.data),
                                     static_cast<const half*>(h.data),
                                     M_d, K_d, stream);
            }
        } else if (fused_down_residual) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            // Fuse activation + Q8_1 quantization into a single kernel when possible.
            // This saves 1 kernel launch per layer (activation + quantize → single kernel).
            // NOTE: tried fusing act+quant+GEMV into one kernel but it regresses ~22%
            // because the 2-pass SwiGLU recomputation doubles gate/up L2 reads and the
            // kpar GEMV is already memory-bound on weight reads (same issue as O-proj
            // inline quant at line 674). Separate quant + kpar achieves higher occupancy.
            if (cfg.ffn_activation != FFNActivation::GEGLU) {
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            } else {
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            }
            // Use h.data as residual source (memcpy was skipped)
            const half* residual_ptr = static_cast<const half*>(h.data);
            dispatch_gemv_residual(ly.w_down_qtype, ly.w_down.data, q8, qscratch_.d8_buf,
                                   static_cast<half*>(h.data), residual_ptr,
                                   M_d, K_d, stream);
        } else if (has_post_ffn_norm && using_fp32_accum && n == 1 &&
                   wcache_.nvfp4.count(ly.w_down.data) && h.dtype == DType::FP16) {
            // NVFP4 post-norm FP32 accum decode: activation → NVFP4 GEMV → post-norm.
            // ~40% less weight traffic than dp4a Q8_0 path.
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            auto& wd_nvfp4 = wcache_.nvfp4.at(ly.w_down.data);
            if (cfg.ffn_activation != FFNActivation::GEGLU)
                swiglu(go, uo, so, stream);
            else
                geglu(go, uo, so, stream);
            gemv_nvfp4_kpar(wd_nvfp4, static_cast<const half*>(so.data),
                             static_cast<half*>(fo.data), M_d, K_d, stream);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 512, 0, stream>>>(
                static_cast<const half*>(fo.data),
                static_cast<const half*>(ly.post_ffn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                cfg.d_model, eps, norm_w_off_);
        } else if (has_post_ffn_norm && using_fp32_accum && n == 1 &&
                   q8 != nullptr && qscratch_.d8_buf != nullptr &&
                   is_dp4a_qtype(ly.w_down_qtype)) {
            // Post-norm FP32 accum decode: fused activation→Q8_1 + GEMV + fused post-norm.
            // Saves 3 kernel launches per layer vs the fallback path.
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            if (cfg.ffn_activation != FFNActivation::GEGLU)
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            else
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                    static_cast<const half*>(uo.data),
                                    q8, qscratch_.d8_buf, K_d, stream);
            half* fo_ptr = static_cast<half*>(fo.data);
            dispatch_gemv_q8_1(ly.w_down_qtype, ly.w_down.data, q8, qscratch_.d8_buf,
                               fo_ptr, M_d, K_d, stream);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 512, 0, stream>>>(
                static_cast<const half*>(fo.data),
                static_cast<const half*>(ly.post_ffn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                cfg.d_model, eps, norm_w_off_);
        } else {
            // Non-dp4a paths: activation must produce FP16 intermediate in so.
            switch (cfg.ffn_activation) {
                case FFNActivation::GEGLU:  geglu(go, uo, so, stream);  break;
                default:                    swiglu(go, uo, so, stream);  break;
            }
            if (will_fuse_down_beta1 && !cur_force_fp16_ &&
                wcache_.fp8.count(ly.w_down.data) &&
                qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
                // FP8 beta=1: hidden = fp8(swiglu_out) @ fp8(w_down)^T + hidden
                auto& e = wcache_.fp8.at(ly.w_down.data);
                Tensor fp8_so(qscratch_.fp8_act, DType::FP8_E4M3, so.ndim, so.shape, true);
                quantize_fp16_to_fp8_e4m3(so, fp8_so, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                gemm_cublaslt(fp8_so, e.weight, h, 1.0f, 1.0f, qscratch_.d_act_scale, e.d_scale, stream);
            } else if (will_fuse_down_beta1 && wcache_.fp16.count(ly.w_down.data)) {
                // Fused: hidden = swiglu_out @ w_down^T + hidden (cuBLAS beta=1).
                const Tensor& wd_fp16 = wcache_.fp16.at(ly.w_down.data);
                gemm(so, wd_fp16, h, 1.0f, 1.0f, stream);
            } else if ((will_fuse_down_beta1 || will_fuse_down_dequant_beta1) &&
                       qscratch_.dequant != nullptr && dequant_gpu_supported(ly.w_down_qtype)) {
                // Dequant into scratch, then beta=1.0 GEMM directly into hidden (which holds residual)
                int rows = static_cast<int>(ly.w_down.shape[0]);
                int cols = static_cast<int>(ly.w_down.shape[1]);
                dequant_gpu(ly.w_down.data, qscratch_.dequant, ly.w_down_qtype, rows, cols, stream);
                Tensor w_fp16(qscratch_.dequant, DType::FP16, ly.w_down.ndim, ly.w_down.shape, true);
                gemm(so, w_fp16, h, 1.0f, 1.0f, stream);
            } else {
                gemm_dispatch(so, ly.w_down, ly.w_down_scales, ly.w_down_qtype, fo, qscratch_.dequant, stream,
                              static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                              (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                              qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                              (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4,
                              (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4,
                              qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                              (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4,
                              qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                if (has_post_ffn_norm && using_fp32_accum) {
                    // Post-FFN norm → FP32 accumulation (no D2D copy needed)
                    Tensor fp32_h = view_tokens(fp32_hidden_, n);
                    rmsnorm_fp32_accum_to_fp16_kernel<<<n, 512, 0, stream>>>(
                        static_cast<const half*>(fo.data),
                        static_cast<const half*>(ly.post_ffn_norm.data),
                        static_cast<float*>(fp32_h.data),
                        static_cast<half*>(h.data),
                        cfg.d_model, eps, norm_w_off_);
                } else if (has_post_ffn_norm) {
                    // Post-FFN norm → residual add (norm directly to h, no copies)
                    rmsnorm(fo, ly.post_ffn_norm, h, eps, stream, norm_w_off_);
                    elementwise_add(h, r, stream);
                } else {
                    // No post-norm: h = fo + residual (fused add-store, no copy)
                    elementwise_add_store(fo, r, h, stream);
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// SSM (Mamba2) sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ssm(int layer, const InferenceState& state,
                            cudaStream_t stream) {
    // Configure shared workspace for SSM phase
    configure_ssm_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n = state.n_tokens;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = inner + 2 * n_groups * ssize;
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = inner / n_heads;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

    // 2. ssm_in projection: [n, d_model] @ ssm_in^T -> [n, ssm_in_dim]
    //    ssm_in_dim = inner(z) + conv_channels(xBC) + n_heads(dt)
    Tensor proj = view_tokens(ssm_proj_buf_, n);
    const auto* nvfp4_ssm_ptr = (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4;
    const auto* ct4_ssm_ptr = (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4;
    const auto* mx4p = (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4;
    gemm_dispatch(no, ly.ssm_in, Tensor(), ly.ssm_in_qtype, proj, qscratch_.dequant, stream,
                  static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                  nvfp4_ssm_ptr, ct4_ssm_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

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
        cudaMemcpy2DAsync(z_buf.data, static_cast<size_t>(inner) * es,
                          proj.data, src_pitch,
                          static_cast<size_t>(inner) * es, n,
                          cudaMemcpyDeviceToDevice, stream);

        xBC_in = view_tokens(ssm_xBC_buf_, n);
        char* xBC_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        cudaMemcpy2DAsync(xBC_in.data, static_cast<size_t>(conv_channels) * es,
                          xBC_src, src_pitch,
                          static_cast<size_t>(conv_channels) * es, n,
                          cudaMemcpyDeviceToDevice, stream);

        dt_buf = view_tokens(ssm_dt_buf_, n);
        char* dt_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        cudaMemcpy2DAsync(dt_buf.data, static_cast<size_t>(n_heads) * es,
                          dt_src, src_pitch,
                          static_cast<size_t>(n_heads) * es, n,
                          cudaMemcpyDeviceToDevice, stream);
    }

    // 4. Conv1d on xBC
    //    For decode with views_into_proj: output to ssm_xBC_buf_ (preserves z/dt in proj).
    //    For prefill: output to ssm_proj_buf_ (proj data already copied out).
    int64_t conv_out_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    void* conv_out_ptr = views_into_proj ? ssm_xBC_buf_.data : ssm_proj_buf_.data;
    Tensor xBC_out(conv_out_ptr, compute_dtype_, 2, conv_out_shape, true);

    int ssm_idx = get_ssm_layer(ssm_layer_map_, layer);
    void* conv_st = (state.ssm_state && ssm_idx >= 0)
                    ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                    : nullptr;

    if (conv_st) {
        if (state.is_prefill) {
            ssm_conv1d_prefill(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                               xBC_out, conv_kernel, stream);
        } else {
            ssm_conv1d_decode(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                              xBC_out, conv_kernel, stream);
        }
    }

    // 5. SiLU on full conv output (x, B, and C together).
    //    Mamba2 applies SiLU to the ENTIRE conv1d output, not just x.
    //    This matches causal_conv1d_fn(..., activation="silu").
    silu_inplace(xBC_out, stream);

    // 6-7. Split conv output into x/B/C per token, run SSM scan.
    int BC_size = n_groups * ssize;
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0)
                 ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
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

        DType h_dtype = (state.ssm_state) ? state.ssm_state->h_dtype() : DType::FP32;

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

            ssm_scan_decode(x_t, B_t, C_t, dt_t,
                            ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st,
                            y_t, static_cast<const half*>(z_buf.data),
                            n_heads, head_dim_ssm, ssize, n_groups, h_dtype, stream);
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
            cudaMemcpy2DAsync(x_contig, static_cast<size_t>(inner) * es,
                              xBC_out.data, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(inner) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

            // B: extract [n, BC_size] from offset inner in xBC_out
            char* B_contig = static_cast<char*>(xBC_in.data);  // conv1d done, safe to reuse
            char* B_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner) * es;
            cudaMemcpy2DAsync(B_contig, static_cast<size_t>(BC_size) * es,
                              B_src, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(BC_size) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

            // C: extract [n, BC_size] from offset inner+BC_size in xBC_out
            char* C_contig = B_contig + static_cast<size_t>(n) * BC_size * es;
            char* C_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner + BC_size) * es;
            cudaMemcpy2DAsync(C_contig, static_cast<size_t>(BC_size) * es,
                              C_src, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(BC_size) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

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

            ssm_scan_prefill(x_all, B_all, C_all, dt_all,
                             ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st,
                             y_all, static_cast<const half*>(z_buf.data),
                             n, n_heads, head_dim_ssm, ssize, n_groups, h_dtype, stream);
        }
    }

    // 8. Gating: y = y * SiLU(z) — fused into ssm_scan kernel above.

    // 9. Group RMSNorm on y  [AFTER gating, per llama.cpp reference]
    group_rmsnorm(y_buf, ly.ssm_norm_w, y_buf, n_groups, eps, stream);

    // 10. ssm_out projection: [n, inner] @ ssm_out^T -> [n, d_model]
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_dispatch(y_buf, ly.ssm_out, Tensor(), ly.ssm_out_qtype, out_buf, qscratch_.dequant, stream,
                  static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                  nvfp4_ssm_ptr, ct4_ssm_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

    // 11. Residual add: hidden = output + residual
    elementwise_add(out_buf, r, stream);
    cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);

}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) layer forward pass
// Same pipeline as Mamba2 SSM but with delta rule scan and separate gating.
// Pipeline: ssm_in(attn_qkv) → conv1d → SiLU → split(x/B/C) → delta_rule_scan
//           → gate(SiLU) → group_norm → ssm_out → residual
// ---------------------------------------------------------------------------

void GraphExecutor::run_gdn(int layer, const InferenceState& state,
                            cudaStream_t stream) {
    configure_ssm_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = inner + 2 * n_groups * ssize;
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = (n_heads > 0) ? inner / n_heads : 0;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

    // 2. ssm_in (attn_qkv) projection → [n, conv_channels]
    //    GDN: no z-split, no dt — the full projection goes to conv1d.
    //    ssm_proj_buf_ is [max_tokens, ssm_in_dim] but we only need [n, conv_channels].
    int64_t proj_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    Tensor proj(ssm_proj_buf_.data, compute_dtype_, 2, proj_shape, true);
    const auto* nvfp4_ptr = (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4;
    const auto* ct4_ptr = (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4;
    const auto* mx4p = (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4;
    gemm_dispatch(no, ly.ssm_in, Tensor(), ly.ssm_in_qtype, proj, qscratch_.dequant, stream,
                  static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                  nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

    // 3. Conv1d on full projection output [n, conv_channels]
    int64_t conv_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    Tensor xBC_in(proj.data, compute_dtype_, 2, conv_shape, true);
    Tensor xBC_out(ssm_xBC_buf_.data, compute_dtype_, 2, conv_shape, true);

    int ssm_idx = get_ssm_layer(ssm_layer_map_, layer);
    void* conv_st = (state.ssm_state && ssm_idx >= 0)
                    ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                    : nullptr;

    // conv_f32 destination for FP32 pipeline (conv+SiLU output)
    float* conv_f32 = static_cast<float*>(ssm_proj_buf_.data);

    if (conv_st) {
        if (state.is_prefill) {
            // Fused: conv1d + SiLU + FP32 output in one kernel (saves 2 launches).
            // Copy FP16 input to xBC_out first to avoid aliasing (conv_f32 = ssm_proj_buf_ = xBC_in).
            cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                            static_cast<size_t>(n) * conv_channels * dtype_size(compute_dtype_),
                            cudaMemcpyDeviceToDevice, stream);
            ssm_conv1d_prefill_f32_silu(conv_st, xBC_out, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                                         conv_f32, conv_kernel, stream);
        } else {
            // Decode: FP32 fused conv+SiLU (matching llama.cpp precision).
            // Copy FP16 input to xBC_out first to avoid aliasing: conv_f32
            // writes FP32 back into ssm_proj_buf_ which overlaps xBC_in.
            cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                            static_cast<size_t>(conv_channels) * dtype_size(compute_dtype_),
                            cudaMemcpyDeviceToDevice, stream);
            ssm_conv1d_decode_f32_silu(conv_st, xBC_out, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                                       conv_f32, conv_kernel, stream);
        }
    } else {
        // Fallback: copy input to output + SiLU + FP32 conversion
        cudaMemcpyAsync(xBC_out.data, xBC_in.data,
                        static_cast<size_t>(n) * conv_channels * dtype_size(compute_dtype_),
                        cudaMemcpyDeviceToDevice, stream);
        silu_inplace(xBC_out, stream);
        int64_t total = static_cast<int64_t>(n) * conv_channels;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(xBC_out.data), conv_f32, total);
    }

    // 5. Split conv output into x/B/C and run delta rule scan
    //    x[inner]: value vectors (V in delta rule terminology)
    //    B[n_groups*ssize]: key vectors (K in delta rule)
    //    C[n_groups*ssize]: query vectors (Q in delta rule)
    int BC_size = n_groups * ssize;
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0)
                 ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
                 : nullptr;

    // GDN layers lack ssm_d (D skip connection). Use ssm_a as dummy (same shape [n_heads]).
    // The ssm_scan kernel reads D[h] for skip connections — when using A_log values as D,
    // the skip contribution is non-zero but small (A_log are typically negative).
    // TODO: allocate a proper zero-filled D tensor for GDN layers.
    const Tensor& ssm_d_ref = (ly.ssm_d.data != nullptr) ? ly.ssm_d : ly.ssm_a;

    // Gate projection — computed before scan, used after in RMSNormGated
    int64_t gate_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(inner)};
    Tensor gate_out(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);
    gemm_dispatch(no, ly.gdn_gate, Tensor(), ly.gdn_gate_qtype, gate_out,
                  qscratch_.dequant, stream,
                  static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr,
                  qscratch_.fp8_act, qscratch_.d_act_scale, qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                  nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf,
                  qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

    if (h_st) {
        size_t es = dtype_size(compute_dtype_);

        // Compute alpha/beta projections from norm_out (input to this layer)
        Tensor alpha_proj_out, beta_proj_out;
        {
            int64_t ab_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(n_heads)};
            // Use ssm_dt_buf_ for alpha (it's [max_tokens, n_heads] — perfect fit)
            // Beta can go right after in the same buffer (plenty of room)
            alpha_proj_out = Tensor(ssm_dt_buf_.data, compute_dtype_, 2, ab_shape, true);
            char* beta_ptr = static_cast<char*>(ssm_dt_buf_.data) +
                             ((static_cast<size_t>(n) * n_heads * es + 255) & ~size_t(255));
            beta_proj_out = Tensor(beta_ptr, compute_dtype_, 2, ab_shape, true);

            // Alpha/beta weights are already dequantized to FP16 during weight upload.
            // Use gemm() (not gemm_dispatch with qtype) to avoid re-interpreting FP16 as Q8_0.
            gemm(no, ly.gdn_alpha, alpha_proj_out, 1.0f, 0.0f, stream);
            gemm(no, ly.gdn_beta, beta_proj_out, 1.0f, 0.0f, stream);
        }

        // 5b. Fused multi-token delta rule scan.
        // Single kernel launch processes ALL tokens with register-cached state.
        // Eliminates n×32 kernel launches and 125x state memory traffic.
        gdn_scan_fused_f32(conv_f32, conv_channels,
                            static_cast<const half*>(alpha_proj_out.data),
                            static_cast<const half*>(beta_proj_out.data),
                            static_cast<const float*>(ly.ssm_a.data),
                            static_cast<const float*>(ly.ssm_dt_b.data),
                            static_cast<float*>(h_st),
                            static_cast<half*>(y_buf.data),
                            n, n_heads, head_dim_ssm, ssize, n_groups, stream);
    }

    debug_tensor_stats("gdn_after_scan_y", y_buf, stream);
    debug_tensor_stats("gdn_gate_out", gate_out, stream);

    // 6. Fused RMSNormGated + SiLU: y = rmsnorm(y) * silu(gate)
    // Single kernel launch for all tokens × heads (replaces n×32×2 launches).
    gdn_rmsnorm_gated_silu(static_cast<half*>(y_buf.data),
                            static_cast<const half*>(gate_out.data),
                            static_cast<const half*>(ly.ssm_norm_w.data),
                            eps, n, n_heads, head_dim_ssm, stream);


    // 7. ssm_out projection: [n, inner] → [n, d_model]
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_dispatch(y_buf, ly.ssm_out, Tensor(), ly.ssm_out_qtype, out_buf, qscratch_.dequant, stream,
                  static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, &wcache_.fp16,
                  (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                  qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                  nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

    // 8. Residual add
    elementwise_add(out_buf, r, stream);
    cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);

}

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
        cudaMemcpy(h_ids.data(), state.token_ids, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
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

    if (state.is_prefill && !state.all_logits) {
        Tensor h_last = view_tokens(hidden_, n).slice(n - 1, n);
        Tensor lg = view_tokens(logits_, 1);

        auto nvfp4_lm_pf = wcache_.nvfp4.find(model_->output_proj().data);
        if (nvfp4_lm_pf != wcache_.nvfp4.end()) {
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
            gemm(no_last, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        }
        logits_out = lg;
        debug_top_logits(lg, stream);
    } else {
        Tensor h_final = view_tokens(hidden_, n);
        Tensor lg = view_tokens(logits_, n);

        auto nvfp4_lm = wcache_.nvfp4.find(model_->output_proj().data);
        if (n == 1 && nvfp4_lm != wcache_.nvfp4.end()) {
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
                gemm(no_final, wcache_.fp16.at(model_->output_proj().data), lg, 1.0f, 0.0f, stream);
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
            } else if (wcache_.fp16.count(model_->output_proj().data)) {
                gemm(no_final, wcache_.fp16.at(model_->output_proj().data), lg, 1.0f, 0.0f, stream);
            } else {
                gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
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

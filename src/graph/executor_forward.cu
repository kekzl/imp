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
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace imp {

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state,
                                   cudaStream_t stream) {
    if (!state.kv_cache || !state.block_tables) return;

    // Map global layer index to KV cache layer index
    int kv_layer = layer;
    if (!kv_layer_map_.empty()) {
        kv_layer = kv_layer_map_[layer];
        if (kv_layer < 0) return;  // not an attention layer
    }

    KVCache* cache = state.kv_cache;
    int n        = state.n_tokens;
    int nkv      = cache->n_kv_heads();
    int hd       = cache->head_dim();
    int row_elems    = nkv * hd;
    int block_stride = kKVBlockSize * row_elems;

    int threads = std::min(row_elems, 256);
    int nblocks = n;   // one CUDA block per token

    bool use_fp8 = (cache->dtype() == DType::FP8_E4M3);
    bool use_int8 = (cache->dtype() == DType::INT8);

    if (use_int8) {
        // INT8 quantized KV cache write path with per-head scales.
        // No per-layer calibration needed — scales are computed per-head at write time.
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);

        int scale_block_stride = kKVBlockSize * nkv;  // half elems per scale block
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
            kKVBlockSize, n,
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

        // K view: [n_tokens, nkv * hd]
        Tensor kv = view_tokens(k_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif

        // V view
        Tensor vv = view_tokens(v_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif
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
            block_stride, row_elems, kKVBlockSize, n,
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
    Tensor qv = view_tokens(q_,        n);
    Tensor kk = view_tokens(k_,        n);
    Tensor vv = view_tokens(v_,        n);
    Tensor ao = view_tokens(attn_out_, n);
    Tensor po = view_tokens(proj_out_, n);

    // 1. Save residual for later add-back.
    //    Optimization: for decode (n=1) with dp4a, fuse residual into GEMV.
    //    For prefill (n>1) with FP16 cache, use cuBLAS beta=1 to fuse residual
    //    into the wo projection GEMM — no separate residual save/add/copy needed.
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
    bool will_fuse_o_nvfp4 = (!has_post_attn_norm && n == 1 && h.dtype == DType::FP16 &&
                               nvfp4_cache_.count(ly.wo.data));
    bool will_fuse_o_residual = (!has_post_attn_norm && !will_fuse_o_nvfp4 &&
                                  n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                                  h.dtype == DType::FP16 &&
                                  (ly.wo_qtype == GGMLQuantType::Q6_K || ly.wo_qtype == GGMLQuantType::Q8_0 ||
                                   ly.wo_qtype == GGMLQuantType::Q4_0 || ly.wo_qtype == GGMLQuantType::Q4_K ||
                                   ly.wo_qtype == GGMLQuantType::Q5_K ||
                                   ly.wo_qtype == GGMLQuantType::Q2_K || ly.wo_qtype == GGMLQuantType::Q3_K));
    bool will_fuse_o_beta1 = (!has_post_attn_norm && !will_fuse_o_residual && !will_fuse_o_nvfp4 &&
                               n > 1 &&
                               (fp16_cache_.count(ly.wo.data) || fp8_cache_.count(ly.wo.data)));
    if (!will_fuse_o_residual && !will_fuse_o_beta1 && !will_fuse_o_nvfp4 && !using_fp32_accum) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    For decode (n=1) with matching quant types: fused RMSNorm→Q8_1→QKV GEMV.
    //    This skips the intermediate norm_out FP16 buffer entirely.
    //    Otherwise falls back to separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        // NVFP4 decode path: uses FP16 input (no Q8_1 quantization needed)
        auto nvfp4_wq = nvfp4_cache_.find(ly.wq.data);
        auto nvfp4_wk = nvfp4_cache_.find(ly.wk.data);
        auto nvfp4_wv = nvfp4_cache_.find(ly.wv.data);
        bool nvfp4_qkv = (n == 1 && nvfp4_wq != nvfp4_cache_.end() &&
                          nvfp4_wk != nvfp4_cache_.end() && nvfp4_wv != nvfp4_cache_.end());
        bool fused_qkv = (n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                          no.dtype == DType::FP16 &&
                          ly.wq_qtype == ly.wk_qtype && ly.wk_qtype == ly.wv_qtype &&
                          (ly.wq_qtype == GGMLQuantType::Q6_K ||
                           ly.wq_qtype == GGMLQuantType::Q8_0 ||
                           ly.wq_qtype == GGMLQuantType::Q4_0 ||
                           ly.wq_qtype == GGMLQuantType::Q4_K ||
                           ly.wq_qtype == GGMLQuantType::Q5_K ||
                           ly.wq_qtype == GGMLQuantType::Q2_K ||
                           ly.wq_qtype == GGMLQuantType::Q3_K));
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
                                    q8, d8_buf_, nullptr /*skip norm_out*/,
                                    K, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            if (ly.wq_qtype == GGMLQuantType::Q6_K) {
                gemv_qkv_fused_q6k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                          q8, d8_buf_,
                                          static_cast<half*>(qv.data),
                                          static_cast<half*>(kk.data),
                                          static_cast<half*>(vv.data),
                                          q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q4_0) {
                gemv_qkv_fused_q4_0_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q4_K) {
                gemv_qkv_fused_q4_k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q5_K) {
                gemv_qkv_fused_q5_k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q2_K) {
                gemv_qkv_fused_q2_k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q3_K) {
                gemv_qkv_fused_q3_k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else {
                gemv_qkv_fused_q8_0_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            }
        } else {
            // Separate RMSNorm + dispatch
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

            // FP8 prefill path: quantize norm_out→FP8 once, 3 separate FP8 GEMMs
            auto fp8_wq = fp8_cache_.find(ly.wq.data);
            auto fp8_wk = fp8_cache_.find(ly.wk.data);
            auto fp8_wv = fp8_cache_.find(ly.wv.data);
            if (n > 1 && fp8_wq != fp8_cache_.end() &&
                fp8_wk != fp8_cache_.end() && fp8_wv != fp8_cache_.end() &&
                fp8_act_buf_ != nullptr && d_act_scale_ != nullptr) {
                Tensor fp8_no(fp8_act_buf_, DType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, d_act_scale_, stream,
                                          d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_);
                gemm_cublaslt(fp8_no, fp8_wq->second.weight, qv, 1.0f, 0.0f,
                              d_act_scale_, fp8_wq->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wk->second.weight, kk, 1.0f, 0.0f,
                              d_act_scale_, fp8_wk->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wv->second.weight, vv, 1.0f, 0.0f,
                              d_act_scale_, fp8_wv->second.d_scale, stream);
            } else {
                // Try fused K+V path: single strided batched GEMM for both projections
                auto fused_kv_it = fused_kv_cache_.find(layer);
                if (n > 1 && fused_kv_it != fused_kv_cache_.end()) {
                    // Q: still separate (different output dim with GQA)
                    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv,
                                  dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_,
                                  cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_,
                                  cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_,
                                  mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                    // K+V: one batched cuBLAS call
                    gemm_kv_batched(no, fused_kv_it->second, kk, vv, stream);
                } else {
                    const auto* nv4p = nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_;
                    const auto* ct4p = cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_;
                    const auto* mx4p = cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_;
                    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nv4p, ct4p, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                    gemm_dispatch(no, ly.wk, ly.wk_scales, ly.wk_qtype, kk, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nv4p, ct4p, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                    gemm_dispatch(no, ly.wv, ly.wv_scales, ly.wv_qtype, vv, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nv4p, ct4p, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2)
        add_bias(qv, ly.q_bias, stream);
        add_bias(kk, ly.k_bias, stream);
        add_bias(vv, ly.v_bias, stream);
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
            !sliding_active) {
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
            int kv_layer = layer;
            if (!kv_layer_map_.empty()) kv_layer = kv_layer_map_[layer];
            KVCache* cache = state.kv_cache;
            int row_elems    = nkv * hd;
            int block_stride = kKVBlockSize * row_elems;
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
                block_stride, row_elems, kKVBlockSize, n,
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
        int total_blk  = cache->total_blocks();
        DType cache_dtype = cache->dtype();
        int64_t cs[4]  = {static_cast<int64_t>(total_blk),
                          static_cast<int64_t>(kKVBlockSize),
                          static_cast<int64_t>(nkv),
                          static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = layer;
        if (!kv_layer_map_.empty()) {
            kv_layer = kv_layer_map_[layer];
        }
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        if (cache_dtype == DType::INT8) {
            // INT8 dp4a paged attention with per-head scales (Split-K enabled)
            paged_attention_set_splitk_scratch(splitk_scratch_, splitk_scratch_size_);
            paged_attention_decode_int8(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        state.block_tables, state.context_lens,
                                        kKVBlockSize, scale,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream);
        } else if (cache_dtype == DType::FP8_E4M3) {
            // FP8 paged attention with on-the-fly dequant (Split-K enabled)
            float kv_scale = (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size()))
                             ? kv_scales_[kv_layer] : 1.0f;
            paged_attention_set_splitk_scratch(splitk_scratch_, splitk_scratch_size_);
            paged_attention_decode_fp8(q4, k_c, v_c, o4,
                                        state.block_tables, state.context_lens,
                                        kKVBlockSize, scale, kv_scale,
                                        state.max_context_len, layer_sliding_window,
                                        cfg.attn_logit_softcap, stream);
        } else {
            paged_attention_set_splitk_scratch(splitk_scratch_, splitk_scratch_size_);
            paged_attention_decode(q4, k_c, v_c, o4,
                                    state.block_tables, state.context_lens,
                                    kKVBlockSize, scale, state.max_context_len,
                                    layer_sliding_window, cfg.attn_logit_softcap, stream);
        }
    }


    // 8+9. O projection + residual connection.
    //    For decode (n=1) with dp4a: fuse residual add into GEMV, write directly
    //    to hidden buffer. When will_fuse_o_residual is set, we skipped the
    //    initial h→r memcpy and use h.data itself as the residual source.
    //    This is safe because h.data is only READ (never written) between the
    //    start of run_attention and this point.
    if (will_fuse_o_nvfp4) {
        // NVFP4 Wo + residual: attn_out (FP16) @ wo_nvfp4^T + residual → hidden
        auto& wo_nvfp4 = nvfp4_cache_.at(ly.wo.data);
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
        quantize_fp16_to_q8_1(attn_fp16, static_cast<block_q8_1*>(q8_1_buf_),
                               d8_buf_, K_o, stream);
        if (ly.wo_qtype == GGMLQuantType::Q6_K) {
            gemv_q6k_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                    d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                    M_o, K_o, stream);
        } else if (ly.wo_qtype == GGMLQuantType::Q4_0) {
            gemv_q4_0_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        } else if (ly.wo_qtype == GGMLQuantType::Q4_K) {
            gemv_q4_k_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        } else if (ly.wo_qtype == GGMLQuantType::Q5_K) {
            gemv_q5_k_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        } else if (ly.wo_qtype == GGMLQuantType::Q2_K) {
            gemv_q2_k_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        } else if (ly.wo_qtype == GGMLQuantType::Q3_K) {
            gemv_q3_k_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        } else {
            gemv_q8_0_q8_1_residual(ly.wo.data, static_cast<block_q8_1*>(q8_1_buf_),
                                      d8_buf_, static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        }
    } else if (will_fuse_o_beta1 && fp8_cache_.count(ly.wo.data) &&
               fp8_act_buf_ != nullptr && d_act_scale_ != nullptr) {
        // FP8 beta=1: hidden = fp8(attn_out) @ fp8(wo)^T + hidden
        auto& e = fp8_cache_.at(ly.wo.data);
        Tensor fp8_ao(fp8_act_buf_, DType::FP8_E4M3, ao.ndim, ao.shape, true);
        quantize_fp16_to_fp8_e4m3(ao, fp8_ao, d_act_scale_, stream,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_);
        gemm_cublaslt(fp8_ao, e.weight, h, 1.0f, 1.0f, d_act_scale_, e.d_scale, stream);
    } else if (will_fuse_o_beta1) {
        // Fused: hidden = attn_out @ wo^T + hidden (cuBLAS beta=1).
        // Safe: hidden is only READ (never written) between attn_norm and here.
        const Tensor& wo_fp16 = fp16_cache_.at(ly.wo.data);
        gemm(ao, wo_fp16, h, 1.0f, 1.0f, stream);
    } else {
        // Fallback: separate O-projection + optional post-norm + residual add
        gemm_dispatch(ao, ly.wo, ly.wo_scales, ly.wo_qtype, po, dequant_scratch_, stream,
                      static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_,
                      use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                      d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                      nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_,
                      cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_,
                      cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                      cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_,
                      mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
        if (has_post_attn_norm && using_fp32_accum) {
            // Fused: RMSNorm + FP32 accum add + FP32→FP16 in one kernel.
            // Saves 2 kernel launches + 2 DRAM round-trips per layer.
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                static_cast<const half*>(po.data),
                static_cast<const half*>(ly.post_attn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                cfg.d_model, eps, norm_w_off_);
        } else if (has_post_attn_norm) {
            // Post-attn norm → residual add (norm directly to h, no copies)
            rmsnorm(po, ly.post_attn_norm, h, eps, stream, norm_w_off_);
            elementwise_add(h, r, stream);
        } else {
            // No post-norm: h = po + residual (fused add-store, no copy)
            elementwise_add_store(po, r, h, stream);
        }
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
    const Tensor& ffn_norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;
    const bool has_post_ffn_norm = (ly.post_ffn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_ffn_norm);
    bool will_fuse_down_nvfp4 = (!has_post_ffn_norm && n == 1 && h.dtype == DType::FP16 &&
                                  nvfp4_cache_.count(ly.w_down.data));
    bool will_fuse_down_residual = (!has_post_ffn_norm && !will_fuse_down_nvfp4 &&
                                     n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                                     h.dtype == DType::FP16 &&
                                     (ly.w_down_qtype == GGMLQuantType::Q6_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q8_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q5_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q2_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q3_K));
    bool will_fuse_down_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                  !will_fuse_down_nvfp4 && n > 1 &&
                                  (fp16_cache_.count(ly.w_down.data) || fp8_cache_.count(ly.w_down.data)));
    bool will_fuse_down_dequant_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                          !will_fuse_down_nvfp4 &&
                                          !will_fuse_down_beta1 && n > 1 &&
                                          dequant_scratch_ != nullptr &&
                                          dequant_gpu_supported(ly.w_down_qtype));
    if (!will_fuse_down_residual && !will_fuse_down_beta1 &&
        !will_fuse_down_dequant_beta1 && !will_fuse_down_nvfp4 && !using_fp32_accum) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. Gate and Up projections
    //    For decode (n=1): fuse RMSNorm→Q8_1→GEMV to avoid redundant quantization.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        int d = static_cast<int>(h.shape[1]);
        // NVFP4 gate+up decode path
        auto nvfp4_wg = nvfp4_cache_.find(ly.w_gate.data);
        auto nvfp4_wu = nvfp4_cache_.find(ly.w_up.data);
        bool nvfp4_ffn = (n == 1 && nvfp4_wg != nvfp4_cache_.end() &&
                          nvfp4_wu != nvfp4_cache_.end());
        bool fused_ffn_norm = (n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                               h.dtype == DType::FP16 &&
                               (ly.w_gate_qtype == GGMLQuantType::Q6_K ||
                                ly.w_gate_qtype == GGMLQuantType::Q8_0 ||
                                ly.w_gate_qtype == GGMLQuantType::Q4_0 ||
                                ly.w_gate_qtype == GGMLQuantType::Q4_K ||
                                ly.w_gate_qtype == GGMLQuantType::Q5_K ||
                                ly.w_gate_qtype == GGMLQuantType::Q2_K ||
                                ly.w_gate_qtype == GGMLQuantType::Q3_K));
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
                                    q8, d8_buf_, static_cast<half*>(no.data),
                                    d, eps, stream, norm_w_off_);
            // Fused gate+up GEMV: single kernel launch for both projections
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            gemv_gate_up_fused(ly.w_gate.data, ly.w_up.data, q8, d8_buf_,
                                static_cast<half*>(go.data),
                                static_cast<half*>(uo.data),
                                ffn_rows, d, ly.w_gate_qtype, stream);
        } else {
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);

            // FP8 prefill path: quantize norm_out→FP8 once, 2 separate FP8 GEMMs
            auto fp8_wg = fp8_cache_.find(ly.w_gate.data);
            auto fp8_wu = fp8_cache_.find(ly.w_up.data);
            if (n > 1 && fp8_wg != fp8_cache_.end() && fp8_wu != fp8_cache_.end() &&
                fp8_act_buf_ != nullptr && d_act_scale_ != nullptr) {
                Tensor fp8_no(fp8_act_buf_, DType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, d_act_scale_, stream,
                                          d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_);
                gemm_cublaslt(fp8_no, fp8_wg->second.weight, go, 1.0f, 0.0f,
                              d_act_scale_, fp8_wg->second.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_wu->second.weight, uo, 1.0f, 0.0f,
                              d_act_scale_, fp8_wu->second.d_scale, stream);
            } else {
                auto fused_gu_it = fused_gate_up_cache_.find(layer);
                if (n > 1 && fused_gu_it != fused_gate_up_cache_.end()) {
                    // Batched gate+up: single cuBLAS call for both projections
                    gemm_pair_batched(no, fused_gu_it->second, go, uo, stream);
                } else {
                    const auto* nv4p = nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_;
                    const auto* ct4p = cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_;
                    const auto* mx4p = cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_;
                    gemm_dispatch(no, ly.w_gate, ly.w_gate_scales, ly.w_gate_qtype, go, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nv4p, ct4p, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                    gemm_dispatch(no, ly.w_up,   ly.w_up_scales,   ly.w_up_qtype,   uo, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                                  nv4p, ct4p, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                }
            }
        }
    }

    // 4+5+6. Gated activation + Down projection + residual add.
    //    For decode (n=1) with dp4a: fuse activation→Q8_1→GEMV+residual.
    //    SwiGLU case: swiglu_quantize_q8_1 fuses activation + Q8_1 in one kernel,
    //    eliminating the intermediate FP16 buffer write and one kernel launch.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        bool fused_down_residual = (!has_post_ffn_norm &&
                                     n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                                     so.dtype == DType::FP16 &&
                                     (ly.w_down_qtype == GGMLQuantType::Q6_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q8_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q5_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q2_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q3_K));
        if (will_fuse_down_nvfp4) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            auto& wd_nvfp4 = nvfp4_cache_.at(ly.w_down.data);
            if (cfg.ffn_activation != FFNActivation::GEGLU) {
                // Fused SwiGLU + NVFP4 GEMV + residual (saves 1 kernel launch)
                gemv_nvfp4_swiglu_residual(wd_nvfp4,
                                            static_cast<const half*>(go.data),
                                            static_cast<const half*>(uo.data),
                                            static_cast<half*>(h.data),
                                            static_cast<const half*>(h.data),
                                            M_d, K_d, stream);
            } else {
                // Fused GeGLU + NVFP4 GEMV + residual (saves 1 kernel launch)
                gemv_nvfp4_geglu_residual(wd_nvfp4,
                                           static_cast<const half*>(go.data),
                                           static_cast<const half*>(uo.data),
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
                                     q8, d8_buf_, K_d, stream);
            } else {
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, d8_buf_, K_d, stream);
            }
            // Use h.data as residual source (memcpy was skipped)
            const half* residual_ptr = static_cast<const half*>(h.data);
            if (ly.w_down_qtype == GGMLQuantType::Q6_K) {
                gemv_q6k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                        static_cast<half*>(h.data), residual_ptr,
                                        M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q4_0) {
                gemv_q4_0_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q4_K) {
                gemv_q4_k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q5_K) {
                gemv_q5_k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q2_K) {
                gemv_q2_k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q3_K) {
                gemv_q3_k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else {
                gemv_q8_0_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            }
        } else if (has_post_ffn_norm && using_fp32_accum && n == 1 &&
                   q8 != nullptr && d8_buf_ != nullptr &&
                   (ly.w_down_qtype == GGMLQuantType::Q6_K ||
                    ly.w_down_qtype == GGMLQuantType::Q8_0 ||
                    ly.w_down_qtype == GGMLQuantType::Q4_0 ||
                    ly.w_down_qtype == GGMLQuantType::Q4_K ||
                    ly.w_down_qtype == GGMLQuantType::Q5_K ||
                    ly.w_down_qtype == GGMLQuantType::Q2_K ||
                    ly.w_down_qtype == GGMLQuantType::Q3_K)) {
            // Post-norm FP32 accum decode: fused activation→Q8_1 + GEMV + fused post-norm.
            // Saves 3 kernel launches per layer vs the fallback path.
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            if (cfg.ffn_activation != FFNActivation::GEGLU)
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, d8_buf_, K_d, stream);
            else
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                    static_cast<const half*>(uo.data),
                                    q8, d8_buf_, K_d, stream);
            half* fo_ptr = static_cast<half*>(fo.data);
            if (ly.w_down_qtype == GGMLQuantType::Q6_K)
                gemv_q6k_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else if (ly.w_down_qtype == GGMLQuantType::Q8_0)
                gemv_q8_0_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else if (ly.w_down_qtype == GGMLQuantType::Q4_0)
                gemv_q4_0_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else if (ly.w_down_qtype == GGMLQuantType::Q4_K)
                gemv_q4_k_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else if (ly.w_down_qtype == GGMLQuantType::Q5_K)
                gemv_q5_k_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else if (ly.w_down_qtype == GGMLQuantType::Q2_K)
                gemv_q2_k_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            else
                gemv_q3_k_q8_1(ly.w_down.data, q8, d8_buf_, fo_ptr, M_d, K_d, stream);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
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
            if (will_fuse_down_beta1 && fp8_cache_.count(ly.w_down.data) &&
                fp8_act_buf_ != nullptr && d_act_scale_ != nullptr) {
                // FP8 beta=1: hidden = fp8(swiglu_out) @ fp8(w_down)^T + hidden
                auto& e = fp8_cache_.at(ly.w_down.data);
                Tensor fp8_so(fp8_act_buf_, DType::FP8_E4M3, so.ndim, so.shape, true);
                quantize_fp16_to_fp8_e4m3(so, fp8_so, d_act_scale_, stream,
                                          d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_);
                gemm_cublaslt(fp8_so, e.weight, h, 1.0f, 1.0f, d_act_scale_, e.d_scale, stream);
            } else if (will_fuse_down_beta1) {
                // Fused: hidden = swiglu_out @ w_down^T + hidden (cuBLAS beta=1).
                const Tensor& wd_fp16 = fp16_cache_.at(ly.w_down.data);
                gemm(so, wd_fp16, h, 1.0f, 1.0f, stream);
            } else if (will_fuse_down_dequant_beta1) {
                // Dequant into scratch, then beta=1.0 GEMM directly into hidden (which holds residual)
                int rows = static_cast<int>(ly.w_down.shape[0]);
                int cols = static_cast<int>(ly.w_down.shape[1]);
                dequant_gpu(ly.w_down.data, dequant_scratch_, ly.w_down_qtype, rows, cols, stream);
                Tensor w_fp16(dequant_scratch_, DType::FP16, ly.w_down.ndim, ly.w_down.shape, true);
                gemm(so, w_fp16, h, 1.0f, 1.0f, stream);
            } else {
                gemm_dispatch(so, ly.w_down, ly.w_down_scales, ly.w_down_qtype, fo, dequant_scratch_, stream,
                              static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_,
                              use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                              d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                              nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_,
                              cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_,
                              cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                              cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_,
                              mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                if (has_post_ffn_norm && using_fp32_accum) {
                    // Post-FFN norm → FP32 accumulation (no D2D copy needed)
                    Tensor fp32_h = view_tokens(fp32_hidden_, n);
                    rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
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
// MoE FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    int n       = cur_n_tokens_;
    int d       = cfg.d_model;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    int eff     = max_expert_eff_;
    float eps   = cfg.rms_norm_eps;
    size_t es   = dtype_size(compute_dtype_);
    int expanded = n * top_k;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    bool residual_fused = false;  // set true if decode fast path fuses residual add

    // 1. Save residual (skip if decode fast path will handle it —
    //    h.data is never written before the final weighted_sum_residual).
    const Tensor& norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;

    // Pre-check decode fast path (same logic as will_decode_fast below)
    GGMLQuantType up_qtype_pre = ly.expert_up_qtype;
    bool will_skip_residual_copy = (n == 1 &&
        ly.expert_up_packed.data != nullptr && moe_dequant_buf_ != nullptr &&
        compute_dtype_ == DType::FP16 &&
        ly.expert_up_packed.on_device &&
        (up_qtype_pre == GGMLQuantType::Q6_K || up_qtype_pre == GGMLQuantType::Q8_0 ||
         up_qtype_pre == GGMLQuantType::Q4_0 || up_qtype_pre == GGMLQuantType::Q4_K ||
         up_qtype_pre == GGMLQuantType::Q5_K || up_qtype_pre == GGMLQuantType::Q2_K ||
         up_qtype_pre == GGMLQuantType::Q3_K) &&
        ly.w_up_shared.data == nullptr);  // must not have shared expert for full residual fusion

    if (!will_skip_residual_copy) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }
    bool moe_fused_norm_q8 = (n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                               h.dtype == DType::FP16);
    if (moe_fused_norm_q8) {
        // Fused: RMSNorm + Q8_1 (also writes FP16 norm_out for gate logits)
        rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                static_cast<const half*>(norm_w.data),
                                static_cast<block_q8_1*>(q8_1_buf_), d8_buf_,
                                static_cast<half*>(no.data),
                                d, eps, stream, norm_w_off_);
    } else {
        rmsnorm(h, norm_w, no, eps, stream, norm_w_off_);
    }

    // 3. Gate logits + top-k routing
    //    For n=1 decode with FP16 weights and pre-allocated buffers: use fused
    //    kernel that computes gate GEMV + softmax/sigmoid + top-k in one launch.
    //    Otherwise: separate gate GEMV + topk gating kernels.
    const void* router_bias_ptr = ly.moe_router_bias.data;
    bool use_sigmoid = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;

    GGMLQuantType up_qtype = ly.expert_up_qtype;
    bool will_decode_fast = (n == 1 &&
                             ly.expert_up_packed.data != nullptr && moe_dequant_buf_ != nullptr &&
                             compute_dtype_ == DType::FP16 &&
                             ly.expert_up_packed.on_device &&
                             (up_qtype == GGMLQuantType::Q6_K || up_qtype == GGMLQuantType::Q8_0 ||
                              up_qtype == GGMLQuantType::Q4_0 || up_qtype == GGMLQuantType::Q4_K ||
                              up_qtype == GGMLQuantType::Q5_K || up_qtype == GGMLQuantType::Q2_K ||
                              up_qtype == GGMLQuantType::Q3_K));

    MoeRoutingResult routing;

    // Fused gate GEMV + topk is only beneficial when n_experts fits in the
    // number of warps (8). For high expert counts (e.g., 128 in Qwen3-Coder),
    // the separate gemv_gate_fp32 (128 parallel blocks) is much faster than
    // serializing 128/8=16 experts per warp in a single block.
    constexpr int kMaxFusedExperts = 8;
    if (ne <= kMaxFusedExperts &&
        n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16 &&
        moe_routing_buffers_.pool && will_decode_fast) {
        // Fused: gate GEMV + softmax/sigmoid + top-k in one kernel (1 launch)
        moe_gate_topk_fused(static_cast<const half*>(ly.moe_gate.data),
                            static_cast<const half*>(no.data),
                            ne, d, top_k,
                            moe_routing_buffers_, routing, stream,
                            use_sigmoid, norm_weights, router_bias_ptr);
    } else {
        // Separate: gate GEMV → intermediate logits → topk gating
        Tensor gate_logits_f32 = slice_rows(moe_gate_logits_, n);

        if (n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16) {
            gemv_gate_fp32(static_cast<const half*>(ly.moe_gate.data),
                           static_cast<const half*>(no.data),
                           static_cast<float*>(gate_logits_f32.data),
                           ne, d, stream);
        } else {
            int64_t gl_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(ne)};
            Tensor gate_logits_tmp(moe_gathered_.data, compute_dtype_, 2, gl_shape, true);
            gemm(no, ly.moe_gate, gate_logits_tmp, 1.0f, 0.0f, stream);

            int64_t numel = static_cast<int64_t>(n) * ne;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == DType::FP16) {
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const half*>(gate_logits_tmp.data),
                    static_cast<float*>(gate_logits_f32.data),
                    numel);
            } else {
                cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                                static_cast<size_t>(numel) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }

        if (moe_routing_buffers_.pool) {
            moe_topk_gating(gate_logits_f32, top_k, moe_routing_buffers_, routing, stream, use_sigmoid, norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
        }
    }

    // 4b. Expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data),
            cfg.expert_weights_scale, n_weights);
    }

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    bool use_packed_dequant = (ly.expert_up_packed.data != nullptr &&
                               moe_dequant_buf_ != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    bool non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                              (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (use_packed_dequant) {
        int64_t ref_eff = non_gated_experts
            ? ly.expert_up_packed.shape[1]
            : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != eff || down_eff != eff) {
            IMP_LOG_ERROR("CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
                         "down_packed.shape[2]=%ld. Using packed tensor shapes instead.",
                         eff, (long)ref_eff, (long)down_eff);
            eff = static_cast<int>(ref_eff);
        }
    }

    // =========================================================================
    // DECODE FAST PATH: n=1, device-resident packed experts, Q6_K or Q8_0.
    // Skips gather/scatter and D2H sync. All top_k experts dispatched in a
    // single kernel launch per projection. CUDA-graph capturable.
    // =========================================================================
    // decode_fast_path == will_decode_fast (computed earlier before routing).
    // will_decode_fast already checks packed data + dequant buf + FP16 + on_device + Q6K/Q8_0.
    bool decode_fast_path = will_decode_fast;

    if (decode_fast_path) {
        // Device pointers from routing result (no D2H copy needed)
        const int32_t* expert_indices = static_cast<const int32_t*>(routing.expert_indices.data);
        const float* expert_weights   = static_cast<const float*>(routing.expert_weights.data);

        // Compute expert stride (bytes between experts in packed tensor)
        auto expert_stride = [](const Tensor& packed, GGMLQuantType qtype) -> size_t {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            return static_cast<size_t>(rows) * ggml_quant_row_bytes(qtype, cols);
        };

        half* norm_ptr = static_cast<half*>(no.data);
        half* gate_buf = static_cast<half*>(moe_expert_gate_.data);   // [top_k, eff]
        half* up_buf   = static_cast<half*>(moe_expert_up_.data);     // [top_k, eff]
        half* act_buf  = static_cast<half*>(moe_expert_swiglu_.data); // [top_k, eff]
        half* down_buf = static_cast<half*>(moe_expert_down_.data);   // [top_k, d]

        // --- NVFP4 MoE path: takes FP16 input directly, no Q8_1 needed ---
        auto nvfp4_up_it = nvfp4_moe_cache_.find(ly.expert_up_packed.data);
        auto nvfp4_down_it = nvfp4_moe_cache_.find(ly.expert_down_packed.data);
        bool use_nvfp4_moe = (nvfp4_up_it != nvfp4_moe_cache_.end() &&
                              nvfp4_down_it != nvfp4_moe_cache_.end());
        if (use_nvfp4_moe && !non_gated_experts) {
            auto nvfp4_gate_it = nvfp4_moe_cache_.find(ly.expert_gate_packed.data);
            use_nvfp4_moe = (nvfp4_gate_it != nvfp4_moe_cache_.end());
        }

        if (use_nvfp4_moe) {
            // Gate+Up projection: NVFP4 MoE GEMV with FP16 input (norm_ptr)
            if (!non_gated_experts) {
                gemv_nvfp4_moe_gate_up_fused(
                    nvfp4_moe_cache_.at(ly.expert_gate_packed.data),
                    nvfp4_moe_cache_.at(ly.expert_up_packed.data),
                    expert_indices, norm_ptr,
                    gate_buf, up_buf, eff, d, top_k, stream);
            } else {
                gemv_nvfp4_moe_decode(
                    nvfp4_moe_cache_.at(ly.expert_up_packed.data),
                    expert_indices, norm_ptr, up_buf,
                    eff, d, /*x_stride=*/0, top_k, stream);
            }

            // Down projection (fused SwiGLU+GEMV for gated, separate for non-gated)
            if (!non_gated_experts) {
                // Fused: swiglu(gate,up) computed inline during down GEMV
                gemv_nvfp4_moe_swiglu_decode(
                    nvfp4_moe_cache_.at(ly.expert_down_packed.data),
                    expert_indices, gate_buf, up_buf, down_buf,
                    d, eff, /*x_stride=*/eff, top_k, stream);
            } else {
                int64_t act_shape[2] = {static_cast<int64_t>(top_k),
                                         static_cast<int64_t>(eff)};
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
                gemv_nvfp4_moe_decode(
                    nvfp4_moe_cache_.at(ly.expert_down_packed.data),
                    expert_indices, up_buf, down_buf,
                    d, eff, /*x_stride=*/eff, top_k, stream);
            }

            // Weighted sum + residual
            {
                bool has_shared_expert = (ly.w_up_shared.data != nullptr);
                const void* res_ptr = has_shared_expert ? nullptr :
                    (will_skip_residual_copy ? h.data : r.data);
                moe_weighted_sum_residual(down_buf, expert_weights, res_ptr,
                                          h.data, d, top_k, stream);
                if (!has_shared_expert) residual_fused = true;
            }

            goto moe_after_experts;
        }

        // Use dp4a MMVQ path when Q8_1 buffers are available
        bool use_dp4a = (q8_1_buf_ != nullptr && d8_buf_ != nullptr);

        if (use_dp4a) {
            // Q8_1 may already be computed by the fused norm+quant above.
            // If not (e.g., prefill or non-FP16), quantize norm_out now.
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            if (!moe_fused_norm_q8) {
                quantize_fp16_to_q8_1(norm_ptr, q8, d8_buf_, d, stream);
            }

            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            // 5'+6'. Fused gate+up projection (single kernel launch)
            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q4_K) {
                    gemv_q4_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q5_K) {
                    gemv_q5_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q4_0) {
                    gemv_q4_0_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q2_K) {
                    gemv_q2_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q3_K) {
                    gemv_q3_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                }
            } else {
                // Non-gated: up projection only
                auto moe_gemv_dp4a = (up_qtype == GGMLQuantType::Q6_K)
                    ? gemv_q6k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q4_0)
                    ? gemv_q4_0_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q4_K)
                    ? gemv_q4_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q5_K)
                    ? gemv_q5_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q2_K)
                    ? gemv_q2_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q3_K)
                    ? gemv_q3_k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
                moe_gemv_dp4a(ly.expert_up_packed.data, expert_indices,
                              q8, d8_buf_, up_buf,
                              eff, d, up_stride_bytes,
                              /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
            }
        } else {
            // Fallback: FP16 dequant path
            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                }
            } else {
                auto moe_gemv = (up_qtype == GGMLQuantType::Q6_K)
                    ? gemv_q6k_moe_decode : gemv_q8_0_moe_decode;
                moe_gemv(ly.expert_up_packed.data, expert_indices,
                         norm_ptr, up_buf,
                         eff, d, up_stride_bytes, /*x_stride=*/0, top_k, stream);
            }
        }

        // 7'+8'. Activation + down projection
        //
        // When dp4a is active and experts are gated (SwiGLU), fuse the activation
        // and Q8_1 quantization into a single kernel, eliminating the intermediate
        // FP16 act_buf write+read.
        if (use_dp4a) {
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            int eff_q8_blocks = eff / 32;

            if (!non_gated_experts) {
                // Fused SwiGLU → Q8_1 (1 kernel instead of 2)
                swiglu_quantize_q8_1(gate_buf, up_buf, q8, d8_buf_,
                                      top_k * eff, stream);
            } else {
                // Non-gated (relu²): fused relu² + Q8_1 quantization (1 kernel)
                relu_sqr_quantize_q8_1(up_buf, q8, d8_buf_, top_k * eff, stream);
            }

            // Down projection with dp4a GEMV
            auto moe_gemv_dp4a_down = (up_qtype == GGMLQuantType::Q6_K)
                ? gemv_q6k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q4_0)
                ? gemv_q4_0_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q4_K)
                ? gemv_q4_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q5_K)
                ? gemv_q5_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q2_K)
                ? gemv_q2_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q3_K)
                ? gemv_q3_k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            moe_gemv_dp4a_down(ly.expert_down_packed.data, expert_indices,
                          q8, d8_buf_, down_buf,
                          d, eff, down_stride,
                          /*q8_1_stride=*/eff_q8_blocks, /*d8_stride=*/eff_q8_blocks,
                          top_k, stream);
        } else {
            // Non-dp4a: separate activation + FP16 down GEMV
            int64_t act_shape[2] = {static_cast<int64_t>(top_k),
                                     static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                // relu² in-place on up_buf, then use up_buf directly for down projection
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor gate_t(gate_buf, compute_dtype_, 2, act_shape, true);
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                Tensor act_t(act_buf, compute_dtype_, 2, act_shape, true);
                swiglu(gate_t, up_t, act_t, stream);
            }
            auto moe_gemv = (up_qtype == GGMLQuantType::Q6_K)
                ? gemv_q6k_moe_decode : gemv_q8_0_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            half* down_input = non_gated_experts ? up_buf : act_buf;
            moe_gemv(ly.expert_down_packed.data, expert_indices,
                     down_input, down_buf,
                     d, eff, down_stride, /*x_stride=*/eff, top_k, stream);
        }

        // 9'. Fused weighted sum + FP16 output (+ residual if no shared expert)
        {
            bool has_shared_expert = (ly.w_up_shared.data != nullptr);
            // Use h.data as residual source when memcpy was skipped
            const void* res_ptr = has_shared_expert ? nullptr :
                (will_skip_residual_copy ? h.data : r.data);
            moe_weighted_sum_residual(down_buf, expert_weights, res_ptr,
                                      h.data, d, top_k, stream);
            if (!has_shared_expert) residual_fused = true;
        }

        goto moe_after_experts;
    }

    // =========================================================================
    // GENERAL PATH: prefill or host-offloaded or non-Q6K/Q8_0 experts
    // =========================================================================

    // =========================================================================
    // FUSED Q6_K PREFILL PATH: reads Q6_K weights directly, eliminates the
    // intermediate FP16/FP8 dequant buffer. Two variants:
    //   TC (tensor core): WMMA 16×16×16, preferred for large batches
    //   Scalar: disabled (FP16 batch path always wins for small batches)
    // =========================================================================
    {
    bool can_fused_q6k = (ne > 16 &&
                          ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == GGMLQuantType::Q6_K &&
                          ly.expert_down_qtype == GGMLQuantType::Q6_K &&
                          compute_dtype_ == DType::FP16);
    if (can_fused_q6k && !non_gated_experts)
        can_fused_q6k = (ly.expert_gate_packed.data &&
                         ly.expert_gate_packed.on_device &&
                         ly.expert_gate_qtype == GGMLQuantType::Q6_K);

    bool use_tc = can_fused_q6k && (expanded > ne * 24);
    bool use_scalar = can_fused_q6k && !use_tc && (expanded <= ne * 12);

    if (use_tc || use_scalar) {
        if (layer == 0) IMP_LOG_INFO("MoE prefill: fused Q6_K %s path (n=%d, expanded=%d)",
                                      use_tc ? "TC" : "scalar", n, expanded);
        const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
        const int32_t* d_sorted  = static_cast<const int32_t*>(routing.sorted_token_ids.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        auto expert_stride_fn = [](const Tensor& packed, GGMLQuantType qtype) -> size_t {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            return static_cast<size_t>(rows) * ggml_quant_row_bytes(qtype, cols);
        };

        if (use_tc) {
            // TC path: gather-free via sorted_token_ids indirection.
            // Gate and up read from original hidden state (no.data), down reads
            // from SwiGLU output (already in expanded layout, no indirection).

            // Gate projection (gated models only)
            if (!non_gated_experts)
                gemm_q6k_fused_moe_prefill_tc(
                    ly.expert_gate_packed.data,
                    no.data, expert_gate_base, d_offsets,
                    eff, d,
                    expert_stride_fn(ly.expert_gate_packed, ly.expert_gate_qtype),
                    ne, stream, d_sorted);

            // Up projection
            gemm_q6k_fused_moe_prefill_tc(
                ly.expert_up_packed.data,
                no.data, expert_up_base, d_offsets,
                eff, d,
                expert_stride_fn(ly.expert_up_packed, up_qtype),
                ne, stream, d_sorted);

        } else {
            // Scalar path: needs gathered buffer
            {
                int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                          static_cast<int64_t>(d)};
                Tensor gathered(moe_gathered_.data, compute_dtype_, 2, gath_shape, true);
                moe_gather(no, routing, gathered, stream);
            }
            char* gathered_base = static_cast<char*>(moe_gathered_.data);

            if (!non_gated_experts)
                gemm_q6k_fused_moe_prefill(
                    ly.expert_gate_packed.data,
                    gathered_base, expert_gate_base, d_offsets,
                    eff, d,
                    expert_stride_fn(ly.expert_gate_packed, ly.expert_gate_qtype),
                    ne, stream);

            gemm_q6k_fused_moe_prefill(
                ly.expert_up_packed.data,
                gathered_base, expert_up_base, d_offsets,
                eff, d,
                expert_stride_fn(ly.expert_up_packed, up_qtype),
                ne, stream);
        }

        // Activation (FP16)
        {
            int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                swiglu(g, u, a, stream);
            }
        }

        // Down projection (reads from expanded-layout SwiGLU output, no indirection)
        char* fused_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        if (use_tc) {
            gemm_q6k_fused_moe_prefill_tc(
                ly.expert_down_packed.data,
                fused_down_act, expert_down_base, d_offsets,
                d, eff,
                expert_stride_fn(ly.expert_down_packed, ly.expert_down_qtype),
                ne, stream);
        } else {
            gemm_q6k_fused_moe_prefill(
                ly.expert_down_packed.data,
                fused_down_act, expert_down_base, d_offsets,
                d, eff,
                expert_stride_fn(ly.expert_down_packed, ly.expert_down_qtype),
                ne, stream);
        }

        // Falls through to scatter (step 7)
    } else {
    // =========================================================================
    // FP16 BATCH or FP8 BATCH PREFILL PATH
    // Pre-check: FP16 batch + device-grouped GEMM is preferred (no D2H sync,
    // simpler pipeline). FP8 batch is only used as fallback when FP16 batch
    // isn't available.
    // =========================================================================

    // Gather: reorder tokens by expert assignment (required for batch/legacy paths)
    {
        int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                  static_cast<int64_t>(d)};
        Tensor gathered(moe_gathered_.data, compute_dtype_, 2, gath_shape, true);
        moe_gather(no, routing, gathered, stream);
    }

    {
    // FP16 batch check: can we dequant all experts to FP16 and use device-grouped GEMM?
    size_t fp16_per_expert = static_cast<size_t>(std::max(
        ly.expert_up_packed.shape[1] * ly.expert_up_packed.shape[2],
        ly.expert_down_packed.shape[1] * ly.expert_down_packed.shape[2])) * sizeof(half);
    bool can_fp16_batch_nosync = (
        moe_batch_dequant_buf_ != nullptr &&
        moe_batch_dequant_buf_size_ >= static_cast<size_t>(ne) * fp16_per_expert &&
        d_moe_weight_ptrs_ && d_moe_weight_ptrs_count_ >= ne &&
        ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
        ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
        dequant_gpu_supported(up_qtype) &&
        dequant_gpu_supported(ly.expert_down_qtype));
    if (can_fp16_batch_nosync && !non_gated_experts)
        can_fp16_batch_nosync = (ly.expert_gate_packed.data &&
                                  ly.expert_gate_packed.on_device &&
                                  dequant_gpu_supported(ly.expert_gate_qtype));

    // FP8 batch check: fallback when FP16 batch isn't available
    size_t up_fp8_sz   = static_cast<size_t>(ne) * ly.expert_up_packed.shape[1]
                       * ly.expert_up_packed.shape[2];
    size_t down_fp8_sz = static_cast<size_t>(ne) * ly.expert_down_packed.shape[1]
                       * ly.expert_down_packed.shape[2];
    size_t max_act_cols = std::max(static_cast<size_t>(ly.expert_up_packed.shape[2]),
                                   static_cast<size_t>(ly.expert_down_packed.shape[2]));
    size_t fp8_buf_needed = std::max(up_fp8_sz, down_fp8_sz)
                          + static_cast<size_t>(expanded) * max_act_cols;
    bool can_fp8_batch = (!can_fp16_batch_nosync &&
                          moe_batch_dequant_buf_ != nullptr &&
                          moe_batch_dequant_buf_size_ >= fp8_buf_needed &&
                          ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == GGMLQuantType::Q6_K &&
                          ly.expert_down_qtype == GGMLQuantType::Q6_K &&
                          compute_dtype_ == DType::FP16 &&
                          !fp16_cache_.count(ly.expert_up_packed.data));
    if (can_fp8_batch && !non_gated_experts)
        can_fp8_batch = (ly.expert_gate_packed.data &&
                         ly.expert_gate_packed.on_device &&
                         ly.expert_gate_qtype == GGMLQuantType::Q6_K);

    if (can_fp16_batch_nosync) {
        // =================================================================
        // FP16 BATCH DEQUANT + cublasGemmGroupedBatchedEx
        // Dequants all experts Q6_K→FP16 into batch buffer, then runs
        // a single cublasGemmGroupedBatchedEx per projection. One D2H
        // sync per layer for offsets (unavoidable for grouped GEMM API).
        // =================================================================
        if (layer == 0) IMP_LOG_INFO("MoE prefill: FP16 batch + grouped GEMM path (n=%d, expanded=%d)",
                                      n, expanded);

        // One D2H sync per layer for expert offsets
        std::vector<int32_t> h_offsets(ne + 1);
        cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                        static_cast<size_t>(ne + 1) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        char* buf = static_cast<char*>(moe_batch_dequant_buf_);
        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        auto batch_dequant_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                       const char* a_base, char* c_base,
                                       int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

            dequant_gpu(static_cast<const uint8_t*>(packed.data), buf, qtype,
                        ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            gemm_moe_batched(a_base, c_base,
                             h_offsets.data(), b_ptrs.data(),
                             K_dim, N_dim, DType::FP16, ne, stream,
                             d_moe_work_ptrs_);
        };

        // Gate projection
        if (!non_gated_experts)
            batch_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                                gathered_base, expert_gate_base, d, eff);

        // Up projection
        batch_dequant_gemm(ly.expert_up_packed, up_qtype,
                            gathered_base, expert_up_base, d, eff);

        // Activation
        {
            int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                swiglu(g, u, a, stream);
            }
        }

        // Down projection
        char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        batch_dequant_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                            down_act, expert_down_base, eff, d);

        // Falls through to scatter (step 7)

    } else if (can_fp8_batch) {
        if (layer == 0) IMP_LOG_INFO("MoE prefill: FP8 batch path (n=%d, expanded=%d, buf=%.1f MiB, need=%.1f MiB)",
                                      n, expanded, moe_batch_dequant_buf_size_ / (1024.0*1024.0), fp8_buf_needed / (1024.0*1024.0));
        // Expert offsets: device-grouped path uses d_offsets directly on GPU.
        // Host offsets + sync are deferred to the legacy fallback path only.
        char* buf = static_cast<char*>(moe_batch_dequant_buf_);

        // FP8 batched GEMM lambda: dequant Q6_K→FP8, quantize FP16 acts→FP8, cuBLAS FP8 GEMM→FP16
        auto chunked_fp8_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                     const char* a_base_fp16, char* c_base_fp16,
                                     int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t weight_fp8_bytes = static_cast<size_t>(ne) * rows * cols;  // 1 byte per FP8 element

            // Buffer layout in moe_batch_dequant_buf_:
            //   [0 .. weight_fp8_bytes)                     = FP8 weights for all experts
            //   [weight_fp8_bytes .. weight_fp8_bytes + act) = FP8 activations
            uint8_t* fp8_weights = reinterpret_cast<uint8_t*>(buf);
            uint8_t* fp8_acts = fp8_weights + weight_fp8_bytes;

            // 1. Dequant all experts Q6_K → FP8 E4M3
            dequant_gpu_fp8(packed.data, fp8_weights, qtype,
                            ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            // 2. Per-expert FP8 scaling: calibrate absmax per expert, quantize with
            //    per-expert scale. Falls back to scale=1.0 if scale buffer unavailable.
            const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
            if (d_moe_fp8_scales_) {
                // Calibrate per-expert: writes scales to d_moe_fp8_scales_
                calibrate_fp8_scales_per_expert(a_base_fp16, K_dim, d_offsets, ne,
                                                 d_moe_fp8_scales_, stream);
                // Quantize with per-expert scale
                quantize_fp16_to_fp8_e4m3_per_expert(a_base_fp16, fp8_acts,
                                                      K_dim, d_offsets, ne,
                                                      d_moe_fp8_scales_, stream);
                // Note: no D2H sync here — device-grouped path uses device-side
                // scales directly. Host scales are only needed by the fallback path.
            } else {
                // Fallback: uniform scale=1.0
                quantize_fp16_to_fp8_e4m3_scaled(a_base_fp16, fp8_acts,
                                                  expanded * K_dim, 1.0f, stream);
            }

            // 3. Build per-expert FP8 weight pointers and dispatch GEMM via
            //    cublasGemmGroupedBatchedEx (single call for all experts).
            size_t expert_fp8_sz = static_cast<size_t>(rows) * cols;
            std::vector<int32_t> h_offsets(ne + 1);
            cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                            static_cast<size_t>(ne + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            std::vector<float> h_act_scales(ne, 1.0f);
            if (d_moe_fp8_scales_) {
                cudaMemcpyAsync(h_act_scales.data(), d_moe_fp8_scales_,
                                static_cast<size_t>(ne) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
            }
            cudaStreamSynchronize(stream);
            std::vector<const void*> weight_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                weight_ptrs[e] = fp8_weights + static_cast<size_t>(e) * expert_fp8_sz;

            gemm_moe_batched(fp8_acts, c_base_fp16,
                             h_offsets.data(), weight_ptrs.data(),
                             K_dim, N_dim, DType::FP8_E4M3, ne, stream,
                             d_moe_work_ptrs_, /*output_dtype=*/DType::FP16,
                             d_moe_fp8_scales_ ? h_act_scales.data() : nullptr);
        };

        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        // Gate projection (gated models only)
        if (!non_gated_experts)
            chunked_fp8_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                             gathered_base, expert_gate_base, d, eff);

        // Up projection
        chunked_fp8_gemm(ly.expert_up_packed, up_qtype,
                         gathered_base, expert_up_base, d, eff);

        // Activation (FP16 — reuse existing kernels)
        {
            int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                // relu² in-place on up buffer (no memcpy needed)
                Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                swiglu(g, u, a, stream);
            }
        }

        // Down projection: up buffer for non-gated (relu² in-place), swiglu for gated
        char* fp8_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        chunked_fp8_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                         fp8_down_act, expert_down_base, eff, d);

        // Falls through to existing scatter (step 7)

    } else {
    // =========================================================================
    // LEGACY FALLBACK: D2H sync + per-expert or batched GEMM
    // =========================================================================
    if (layer == 0) IMP_LOG_INFO("MoE prefill: legacy FP16 fallback path (n=%d, expanded=%d)",
                                  n, expanded);
    {
    std::vector<int32_t> h_offsets(ne + 1);
    cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                    static_cast<size_t>(ne + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Helper: dequant one expert's weight from packed tensor into dequant scratch slot 0.
    // Returns a Tensor view into the scratch buffer with shape [rows, cols], FP16.
    // Uses slot 0 always -- safe because all ops are on the same stream, so the previous
    // GEMM reading from slot 0 completes before the next dequant writes to it.
    auto dequant_expert = [&](const Tensor& packed, GGMLQuantType qtype,
                              int expert_idx) -> Tensor {
        int64_t rows = packed.shape[1];
        int64_t cols = packed.shape[2];
        size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
        size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
        size_t total_raw = static_cast<size_t>(packed.shape[0]) * expert_raw;
        size_t offset = static_cast<size_t>(expert_idx) * expert_raw;

        // Bounds check: verify offset + expert_raw <= total allocated
        if (offset + expert_raw > total_raw) {
            IMP_LOG_ERROR("dequant_expert: OOB! expert %d offset=%zu + raw=%zu > total=%zu "
                    "(packed shape [%ld,%ld,%ld] qtype=%u)",
                    expert_idx, offset, expert_raw, total_raw,
                    (long)packed.shape[0], (long)packed.shape[1], (long)packed.shape[2],
                    (unsigned)qtype);
            return Tensor();
        }

        // Check dequant buffer is large enough
        size_t dequant_needed = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
        if (dequant_needed > moe_dequant_buf_size_) {
            IMP_LOG_ERROR("dequant_expert: dequant buffer too small! "
                    "need=%zu have=%zu (rows=%ld cols=%ld)",
                    dequant_needed, moe_dequant_buf_size_, (long)rows, (long)cols);
            return Tensor();
        }

        const char* src;
        if (!packed.on_device) {
            // Expert weights offloaded to host — try LRU cache first, then staging buffer.
            const char* host_ptr = static_cast<const char*>(packed.data) + offset;
            if (expert_cache_.n_slots_ > 0) {
                ExpertCacheKey ck{packed.data, expert_idx};
                void* cached = expert_cache_.get_or_load(ck, host_ptr, expert_raw, stream);
                src = static_cast<const char*>(cached);
            } else if (moe_raw_staging_buf_) {
                cudaMemcpyAsync(moe_raw_staging_buf_, host_ptr, expert_raw,
                                cudaMemcpyHostToDevice, stream);
                src = static_cast<const char*>(moe_raw_staging_buf_);
            } else {
                IMP_LOG_ERROR("dequant_expert: no staging buffer for host expert %d", expert_idx);
                return Tensor();
            }
        } else {
            src = static_cast<const char*>(packed.data) + offset;
        }

        char* dst = static_cast<char*>(moe_dequant_buf_);  // always slot 0

        dequant_gpu(src, dst, qtype, static_cast<int>(rows), static_cast<int>(cols), stream);

        int64_t shape[2] = {rows, cols};
        return Tensor(dst, DType::FP16, 2, shape, true);
    };

    // Helper: try fused quantized GEMV for count=1 decode (dequant+dot in one kernel),
    // else fall back to dequant_expert + cuBLAS gemm.
    // For host-resident experts: H2D to staging buffer, then fused GEMV on staging —
    // eliminates separate dequant_gpu + cuBLAS gemm overhead.
    auto expert_gemm = [&](const Tensor& a, Tensor& c,
                            const Tensor& packed, GGMLQuantType qtype,
                            const std::vector<Tensor>& fallback, int eidx) {
        if (a.shape[0] == 1 && use_packed_dequant &&
            compute_dtype_ == DType::FP16 &&
            (qtype == GGMLQuantType::Q6_K || qtype == GGMLQuantType::Q8_0)) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t rb = ggml_quant_row_bytes(qtype, cols);
            const void* w = nullptr;

            if (packed.on_device) {
                // On-device: point directly into packed tensor
                w = static_cast<const char*>(packed.data) +
                    (size_t)eidx * (size_t)rows * rb;
            } else {
                // Host-resident: try LRU cache, then staging buffer.
                size_t expert_raw = (size_t)rows * rb;
                size_t offset = (size_t)eidx * expert_raw;
                const char* host_ptr = static_cast<const char*>(packed.data) + offset;
                if (expert_cache_.n_slots_ > 0) {
                    ExpertCacheKey ck{packed.data, eidx};
                    w = expert_cache_.get_or_load(ck, host_ptr, expert_raw, stream);
                } else if (moe_raw_staging_buf_ && expert_raw <= moe_raw_staging_size_) {
                    cudaMemcpyAsync(moe_raw_staging_buf_, host_ptr, expert_raw,
                                    cudaMemcpyHostToDevice, stream);
                    w = moe_raw_staging_buf_;
                }
            }

            if (w) {
                auto fn = (qtype == GGMLQuantType::Q6_K) ? gemv_q6k : gemv_q8_0;
                fn(w, static_cast<const half*>(a.data), static_cast<half*>(c.data),
                   static_cast<int>(rows), static_cast<int>(cols), stream);
                return;
            }
        }
        // Fallback: separate dequant + cuBLAS GEMM
        {
            Tensor b = use_packed_dequant
                ? dequant_expert(packed, qtype, eidx)
                : fallback[eidx];
            if (!b.data) return;  // dequant_expert failed (OOB or buffer too small)
            gemm(a, b, c, 1.0f, 0.0f, stream);
        }
    };

        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        // Helper: get FP16 expert weight pointer from pre-dequant cache or unpacked weights.
        auto get_fp16_expert_ptr = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        int eidx) -> const void* {
            if (packed.data && fp16_cache_.count(packed.data)) {
                const Tensor& cached = fp16_cache_.at(packed.data);
                int64_t rows = packed.shape[1];
                int64_t cols = packed.shape[2];
                size_t expert_offset = static_cast<size_t>(eidx) * rows * cols * sizeof(half);
                return static_cast<const char*>(cached.data) + expert_offset;
            }
            if (!fallback.empty() && static_cast<size_t>(eidx) < fallback.size() &&
                fallback[eidx].data && fallback[eidx].dtype == DType::FP16 &&
                fallback[eidx].on_device) {
                return fallback[eidx].data;
            }
            return nullptr;
        };

        // Helper: batch dequant all experts + single grouped GEMM.
        // Dequants all experts to FP16, then runs a single batched GEMM.
        // CUTLASS 2.x GemmGrouped provides lower launch overhead than cuBLAS.
        auto chunked_dequant_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        const char* a_base, char* c_base,
                                        int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);
            size_t expert_raw_sz = static_cast<size_t>(rows)
                                   * ggml_quant_row_bytes(qtype, cols);

            if (!moe_batch_dequant_buf_ || expert_fp16_sz == 0) {
                // No buffer — serial fallback
                for (int e = 0; e < ne; ++e) {
                    int start = h_offsets[e];
                    int count = h_offsets[e + 1] - start;
                    if (count == 0) continue;
                    int64_t count64 = static_cast<int64_t>(count);
                    int64_t a_shape[2] = {count64, static_cast<int64_t>(K_dim)};
                    Tensor a_view(const_cast<void*>(static_cast<const void*>(
                                  a_base + static_cast<size_t>(start) * K_dim * es)),
                                  compute_dtype_, 2, a_shape, true);
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(N_dim)};
                    Tensor c_view(c_base + static_cast<size_t>(start) * N_dim * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, packed, qtype, fallback, e);
                }
                return;
            }

            const uint8_t* raw_base = static_cast<const uint8_t*>(packed.data);
            char* buf = static_cast<char*>(moe_batch_dequant_buf_);

            // Dequant all experts in one batch, then single GEMM.
            // With pp=512 and top_k=8, nearly all 128 experts are active, so
            // dequanting all at once is optimal (one big bandwidth-saturating kernel).
            dequant_gpu(raw_base, buf, qtype,
                        ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            // Use cublasGemmGroupedBatchedEx — single call for all experts.
            // We already have h_offsets from D2H sync, so no need for
            // gemm_moe_device_grouped (which does its own D2H sync + 128
            // individual cublasLtMatmul calls).
            gemm_moe_batched(a_base, c_base,
                             h_offsets.data(), b_ptrs.data(),
                             K_dim, N_dim, DType::FP16, ne, stream,
                             d_moe_work_ptrs_);
        };

        // Determine which path to use:
        // 1. Pre-cached FP16 path: all experts in fp16_cache_ (fastest, no dequant)
        // 2. Dequant-then-batch path: packed experts on device + batch buffer available
        // 3. Serial path: fallback (one expert at a time)
        // Note: fused Q6K dp4a path is handled above (before the D2H sync).

        bool has_precached_up = (ly.expert_up_packed.data && fp16_cache_.count(ly.expert_up_packed.data));
        bool can_dequant_batch = (moe_batch_dequant_buf_ != nullptr &&
                                   ly.expert_up_packed.data != nullptr &&
                                   ly.expert_up_packed.on_device &&
                                   dequant_gpu_supported(ly.expert_up_qtype));

        if (has_precached_up) {
            // Pre-cached FP16 path — all expert packs in fp16_cache_
            // ===== PRE-CACHED FP16 BATCHED GEMM PATH =====
            std::vector<const void*> gate_w_ptrs(ne, nullptr);
            std::vector<const void*> up_w_ptrs(ne, nullptr);
            std::vector<const void*> down_w_ptrs(ne, nullptr);

            for (int e = 0; e < ne; e++) {
                up_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_up_packed, ly.expert_up_qtype,
                                                     ly.expert_w_up, e);
                if (!non_gated_experts)
                    gate_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_gate_packed, ly.expert_gate_qtype,
                                                           ly.expert_w_gate, e);
                down_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_down_packed, ly.expert_down_qtype,
                                                       ly.expert_w_down, e);
            }

            if (!non_gated_experts)
                gemm_moe_batched(gathered_base, expert_gate_base,
                                  h_offsets.data(), gate_w_ptrs.data(),
                                  d, eff, DType::FP16, ne, stream, d_moe_work_ptrs_);
            gemm_moe_batched(gathered_base, expert_up_base,
                              h_offsets.data(), up_w_ptrs.data(),
                              d, eff, DType::FP16, ne, stream, d_moe_work_ptrs_);

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            {
                char* batch_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                gemm_moe_batched(batch_down_act, expert_down_base,
                                  h_offsets.data(), down_w_ptrs.data(),
                                  eff, d, DType::FP16, ne, stream, d_moe_work_ptrs_);
            }

        } else if (can_dequant_batch) {
            // ===== BATCH DEQUANT + GROUPED GEMM =====
            // Dequant all experts to FP16, then single grouped GEMM via CUTLASS.

            if (!non_gated_experts)
                chunked_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                                     ly.expert_w_gate, gathered_base, expert_gate_base, d, eff);
            chunked_dequant_gemm(ly.expert_up_packed, ly.expert_up_qtype,
                                 ly.expert_w_up, gathered_base, expert_up_base, d, eff);

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            {
                char* dequant_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                chunked_dequant_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                                     ly.expert_w_down, dequant_down_act, expert_down_base, eff, d);
            }

        } else {
            // ===== SERIAL PATH (fallback) =====
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, a_shape, true);

                if (!non_gated_experts) {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_gate_packed,
                                ly.expert_gate_qtype, ly.expert_w_gate, e);
                }

                {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_up_packed,
                                ly.expert_up_qtype, ly.expert_w_up, e);
                }
            }

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            // Down projection activation source: up buffer for non-gated (relu² in-place),
            // swiglu buffer for gated.
            char* down_act_base = non_gated_experts ? expert_up_base : expert_swiglu_base;
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor a_view(down_act_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, a_shape, true);
                int64_t c_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor c_view(expert_down_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, c_shape, true);
                expert_gemm(a_view, c_view, ly.expert_down_packed,
                            ly.expert_down_qtype, ly.expert_w_down, e);
            }
        }
    }
    } // legacy inner scope
    } // else of can_fp16/fp8_batch/legacy
    } // FP8/FP16 prefill scope
    } // else branch of can_fused_q6k + fused Q6_K scope

    // 7+8. Scatter expert outputs back to token positions.
    //      Fused path: token-centric scatter + FP16 convert (+ residual if no shared expert).
    //      Fallback: atomicAdd scatter + FP32->FP16 convert.
    {
        bool has_shared_expert = (ly.w_up_shared.data != nullptr);
        if (routing.token_to_expanded && compute_dtype_ == DType::FP16) {
            // Fused token-centric scatter: no atomics, no FP32 intermediate buffer.
            // If no shared expert, also fuse residual add.
            const void* res_ptr = (!has_shared_expert && !residual_fused) ? r.data : nullptr;
            moe_scatter_fused_residual(
                moe_expert_down_.data, routing.token_to_expanded,
                static_cast<const float*>(routing.expert_weights.data),
                res_ptr, h.data,
                n, d, top_k, stream);
            if (!has_shared_expert) residual_fused = true;
        } else {
            // Fallback: atomicAdd scatter into FP32 buffer, then convert
            int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded),
                                            static_cast<int64_t>(d)};
            Tensor expert_down_view(moe_expert_down_.data, compute_dtype_,
                                    2, expert_out_shape, true);
            Tensor scatter_out = slice_rows(moe_scatter_out_, n);
            cudaMemsetAsync(scatter_out.data, 0,
                            static_cast<size_t>(n) * d * sizeof(float), stream);
            moe_scatter(expert_down_view, routing, scatter_out, stream);

            int64_t numel = static_cast<int64_t>(n) * d;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == DType::FP16) {
                fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const float*>(moe_scatter_out_.data),
                    static_cast<half*>(h.data),
                    numel);
            } else {
                cudaMemcpyAsync(h.data, moe_scatter_out_.data,
                                static_cast<size_t>(numel) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

moe_after_experts:
    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    if (ly.w_up_shared.data != nullptr) {
        int eff_shared = static_cast<int>(ly.w_up_shared.shape[0]);
        bool shared_gated = (ly.w_gate_shared.data != nullptr);

        // Reuse moe_expert_gate_, moe_expert_up_, moe_expert_swiglu_ as scratch.
        int64_t sh_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(eff_shared)};
        Tensor sh_up(moe_expert_up_.data, compute_dtype_, 2, sh_shape, true);
        Tensor sh_swiglu(moe_expert_swiglu_.data, compute_dtype_, 2, sh_shape, true);

        // Down projection output: [n, d_model]. Reuse moe_expert_down_.
        int64_t sh_down_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
        Tensor sh_down(moe_expert_down_.data, compute_dtype_, 2, sh_down_shape, true);

        // Up projection (dp4a MMVQ for decode)
        {
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            const auto* nvfp4_ptr = nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_;
            const auto* ct4_ptr = cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_;
            const auto* mx4p = cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_;
            gemm_dispatch(no, ly.w_up_shared, Tensor(), ly.w_up_shared_qtype,
                          sh_up, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                          use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                          d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                          nvfp4_ptr, ct4_ptr, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);

            if (shared_gated) {
                // Gated: gate + SwiGLU
                Tensor sh_gate(moe_expert_gate_.data, compute_dtype_, 2, sh_shape, true);
                gemm_dispatch(no, ly.w_gate_shared, Tensor(), ly.w_gate_shared_qtype,
                              sh_gate, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                              use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                              d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                              nvfp4_ptr, ct4_ptr, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
                swiglu(sh_gate, sh_up, sh_swiglu, stream);
            } else {
                // Non-gated: relu^2(up) in-place [Nemotron-H uses squared ReLU]
                relu_sqr_inplace(sh_up, stream);
            }

            // Down projection (reads from sh_up for non-gated since relu² was in-place)
            Tensor& sh_act = shared_gated ? sh_swiglu : sh_up;
            gemm_dispatch(sh_act, ly.w_down_shared, Tensor(), ly.w_down_shared_qtype,
                          sh_down, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_,
                          use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                          d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                          nvfp4_ptr, ct4_ptr, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);
        }

        // Add shared expert output to hidden (which already has routed expert output)
        elementwise_add(h, sh_down, stream);
    }

    // 9. Residual connection: hidden += residual
    //    Skipped when decode fast path already fused residual into weighted_sum.
    if (!residual_fused) {
        elementwise_add(h, r, stream);
    }

    // 10. Free routing result tensors only if allocated by moe_topk_gating.
    //     When using pre-allocated buffers, memory belongs to moe_routing_buffers_.
    if (routing.owns_memory) {
        cudaFree(routing.expert_indices.data);
        cudaFree(routing.expert_weights.data);
        cudaFree(routing.sorted_token_ids.data);
        cudaFree(routing.expert_offsets.data);
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
    const auto* nvfp4_ssm_ptr = nvfp4_cache_.empty() ? nullptr : &nvfp4_cache_;
    const auto* ct4_ssm_ptr = cutlass_nvfp4_cache_.empty() ? nullptr : &cutlass_nvfp4_cache_;
    const auto* mx4p = cutlass_mxfp4_cache_.empty() ? nullptr : &cutlass_mxfp4_cache_;
    gemm_dispatch(no, ly.ssm_in, Tensor(), ly.ssm_in_qtype, proj, dequant_scratch_, stream,
                  static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_,
                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                  nvfp4_ssm_ptr, ct4_ssm_ptr, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);

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

    int ssm_idx = ssm_layer_map_[layer];
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
    gemm_dispatch(y_buf, ly.ssm_out, Tensor(), ly.ssm_out_qtype, out_buf, dequant_scratch_, stream,
                  static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_,
                  use_fp8_cache_ ? &fp8_cache_ : nullptr, fp8_act_buf_, d_act_scale_,
                  d_fp8_block_maxes_, d_fp8_absmax_, fp8_max_grid_,
                  nvfp4_ssm_ptr, ct4_ssm_ptr, cutlass_act_data_, cutlass_act_sf_, cutlass_workspace_, cutlass_workspace_size_,
                                  mx4p, mxfp4_act_sf_, mxfp4_workspace_, mxfp4_workspace_size_);

    // 11. Residual add: hidden = output + residual
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

    // Clear any stale CUDA error state before starting the forward pass.
    { cudaError_t e_ = cudaGetLastError();
      if (e_ != cudaSuccess) IMP_LOG_DEBUG("Cleared stale error before forward: %s", cudaGetErrorString(e_)); }

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

    cudaEvent_t ev_start, ev_emb, ev_lm;
    std::vector<cudaEvent_t> ev_attn, ev_ffn;
    if (profile_active) {
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
    for (int i = 0; i < cfg.n_layers; ++i) {
        // Layer offloading: ensure weights are on GPU, prefetch next layer
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) {
                offload_mgr_->prefetch_layer(i + 1);
            }
        }

        // Attention or SSM (mutually exclusive per layer)
        if (layer_has_attention(i)) {
            run_attention(i, state, stream);
        } else if (layer_has_ssm(i)) {
            run_ssm(i, state, stream);
        }
        if (i <= 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_layer%d_%s", i,
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
        if (i == cfg.n_layers - 1) {
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
    const bool use_dp4a_lm = q8_1_buf_ && compute_dtype_ == DType::FP16 &&
        (out_qtype == GGMLQuantType::Q6_K || out_qtype == GGMLQuantType::Q8_0 ||
         out_qtype == GGMLQuantType::Q4_0 || out_qtype == GGMLQuantType::Q4_K ||
         out_qtype == GGMLQuantType::Q5_K || out_qtype == GGMLQuantType::Q2_K ||
         out_qtype == GGMLQuantType::Q3_K);

    if (state.is_prefill) {
        Tensor h_last = view_tokens(hidden_, n).slice(n - 1, n);
        Tensor lg = view_tokens(logits_, 1);

        auto nvfp4_lm_pf = nvfp4_cache_.find(model_->output_proj().data);
        if (nvfp4_lm_pf != nvfp4_cache_.end()) {
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
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_last.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            if (out_qtype == GGMLQuantType::Q6_K)
                gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                   static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_0)
                gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_K)
                gemv_q4_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q5_K)
                gemv_q5_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q2_K)
                gemv_q2_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q3_K)
                gemv_q3_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else
                gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
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

        auto nvfp4_lm = nvfp4_cache_.find(model_->output_proj().data);
        if (n == 1 && nvfp4_lm != nvfp4_cache_.end()) {
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
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_final.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            if (out_qtype == GGMLQuantType::Q6_K)
                gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                   static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_0)
                gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_K)
                gemv_q4_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q5_K)
                gemv_q5_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q2_K)
                gemv_q2_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q3_K)
                gemv_q3_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else
                gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else {
            Tensor no_final = view_tokens(norm_out_, n);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        }
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

        // Cleanup events
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_emb);
        cudaEventDestroy(ev_lm);
        for (int i = 0; i < cfg.n_layers; i++) {
            cudaEventDestroy(ev_attn[i]);
            cudaEventDestroy(ev_ffn[i]);
        }
    }
}


} // namespace imp

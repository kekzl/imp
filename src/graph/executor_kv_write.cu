// KV cache write dispatch — extracted from executor_forward.cu (RF-004).
// Handles all KV cache dtype paths: TurboQuant, INT4, INT8, FP8, FP16.

#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>

namespace imp {

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream) {
    if (!state.kv_cache || !state.block_tables)
        return;

    // Map global layer index to KV cache layer index
    int kv_layer = get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0)
        return;  // not an attention layer

    KVCache* cache = state.kv_cache;
    int n = state.n_tokens;
    const auto& cfg = model_->config();
    // Per-layer shape support (Gemma 4 dual attention geometry)
    int nkv, hd;
    if (!cfg.n_kv_heads_per_layer.empty() && layer < (int)cfg.n_kv_heads_per_layer.size() &&
        cfg.n_kv_heads_per_layer[layer] > 0) {
        nkv = cfg.n_kv_heads_per_layer[layer];
    } else {
        nkv = cache->n_kv_heads();
    }
    if (!cfg.head_dim_per_layer.empty() && layer < (int)cfg.head_dim_per_layer.size() &&
        cfg.head_dim_per_layer[layer] > 0) {
        hd = cfg.head_dim_per_layer[layer];
    } else {
        hd = cache->head_dim();
    }
    const int kv_block_size = cache->block_size();
    int row_elems = nkv * hd;
    int block_stride = kv_block_size * row_elems;

    int threads = std::min(row_elems, 256);

    bool use_fp8 = (cache->qtype() == QType::FP8_E4M3);
    bool use_int8 = (cache->qtype() == QType::INT8);
    bool use_int4 = (cache->qtype() == QType::INT4);
    bool use_turboquant = (cache->qtype() == QType::TURBOQUANT);
    bool use_turboquant_lite = (cache->qtype() == QType::TURBOQUANT_LITE);

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
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
            static_cast<const uint8_t*>(qjl_proj_.matrix), int4_block_stride, scale_block_stride_tql,
            sketch_block_stride, nkv, hd, sketch_dim, kv_block_size, n, state.max_blocks_per_seq,
            state.n_sequences);
    } else if (use_turboquant) {
        // TurboQuant KV cache write: PolarQuant directions + QJL sketch for K, INT4 for V
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;
        int scale_block_stride_tq = kv_block_size * nkv;
        int sketch_dim = qjl_proj_.sketch_dim;
        int sketch_block_stride = kv_block_size * nkv * (sketch_dim / 8);
        dim3 grid_tq(n, 2);

        if (cache->use_mxfp4()) {
            // MXFP4 path: FP4 E2M1 directions + UE8M0 per-32-element micro-scales
            int n_groups = hd / 32;
            int mscale_block_stride = kv_block_size * nkv * n_groups;
            write_kv_cache_turboquant_mxfp4_kernel<<<grid_tq, 256, 0, stream>>>(
                static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
                state.block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
                static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
                static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
                static_cast<uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
                static_cast<uint8_t*>(cache->k_mscale_ptr(kv_layer, 0)),
                static_cast<const uint8_t*>(qjl_proj_.matrix), int4_block_stride, scale_block_stride_tq,
                sketch_block_stride, mscale_block_stride, nkv, hd, sketch_dim, kv_block_size, n,
                state.max_blocks_per_seq, state.n_sequences);
        } else {
            // INT4 uniform path
            write_kv_cache_turboquant_kernel<<<grid_tq, 256, 0, stream>>>(
                static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
                state.block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
                static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
                static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)),
                static_cast<uint8_t*>(cache->k_sketch_ptr(kv_layer, 0)),
                static_cast<const uint8_t*>(qjl_proj_.matrix), int4_block_stride, scale_block_stride_tq,
                sketch_block_stride, nkv, hd, sketch_dim, kv_block_size, n, state.max_blocks_per_seq,
                state.n_sequences);
        }
    } else if (use_int4) {
        // INT4 quantized KV cache write — 2 values packed per byte, per-head scales
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;  // bytes (half the INT8 stride)
        int scale_block_stride = kv_block_size * nkv;
        dim3 grid_int4(n, 2);
        write_kv_cache_int4_kernel<<<grid_int4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), int4_block_stride, scale_block_stride, nkv,
            hd, kv_block_size, n, state.max_blocks_per_seq, state.n_sequences);
    } else if (use_int8) {
        // INT8 quantized KV cache write path with per-head scales.
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);

        int scale_block_stride = kv_block_size * nkv;
        dim3 grid_int8(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_int8_kernel<<<grid_int8, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<int8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<int8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), block_stride, scale_block_stride, nkv, hd,
            kv_block_size, n, state.max_blocks_per_seq, state.n_sequences);
    } else if (use_fp8) {
        // FP8 E4M3 quantized KV cache write path with online calibration.
        //
        // Calibration strategy: high-water-mark per layer. The first prefill
        // for a given kv_calibrated_ slot sets the initial scale; subsequent
        // prefills (after Engine::warmup() resets the calibrated_ flag) only
        // promote the scale if their absmax exceeds the stored value. The
        // scale is never reduced, which avoids the warmup-pollution failure
        // mode where synthetic BOS tokens produced a too-small scale and
        // real generation overflowed FP8_MAX (was: Llama-3.2-3B with
        // --kv-fp8 → " France, and, 2008, 201, …"; now: " The capital of
        // Italy is Rome…").
        float inv_scale;
        if (!kv_calibrated_.empty() && kv_layer < static_cast<int>(kv_calibrated_.size()) &&
            !kv_calibrated_[kv_layer]) {
            // Narrow the calibration view to the per-layer K/V shape. The k_/v_
            // workspaces are sized for max_nkv * max_head_dim across all layers
            // (Gemma-4 dual head_dim 256 SWA / 512 global; uniform on Llama / Qwen).
            // Without narrowing, calibrate_fp8_scale would absmax-reduce over
            // uninitialized memory beyond the live data region for layers with
            // smaller head_dim, producing a scale derived from junk and
            // permanently locking the FP8 dynamic range to the wrong value
            // (was the root cause of the Gemma-4 force-FP16 carve-out at
            // engine.cpp:567).
            Tensor kv_cal = view_tokens(k_, n);
            Tensor vv_cal = view_tokens(v_, n);
            const int64_t live_cols = static_cast<int64_t>(nkv) * hd;
            if (kv_cal.shape[1] != live_cols) {
                kv_cal.shape[1] = live_cols;
                kv_cal.compute_strides();
            }
            if (vv_cal.shape[1] != live_cols) {
                vv_cal.shape[1] = live_cols;
                vv_cal.compute_strides();
            }
            float k_scale = calibrate_fp8_scale(kv_cal, stream);
            float v_scale = calibrate_fp8_scale(vv_cal, stream);
            float new_scale = std::max(k_scale, v_scale);
            if (new_scale < 1e-12f)
                new_scale = 1.0f;
            // Promote only — the high-water mark is the union of every
            // prefill we've seen, so values fit FP8 dynamic range.
            kv_scales_[kv_layer] = std::max(kv_scales_[kv_layer], new_scale);
            kv_calibrated_[kv_layer] = true;
            inv_scale = 1.0f / kv_scales_[kv_layer];
        } else if (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size())) {
            inv_scale = 1.0f / kv_scales_[kv_layer];
        } else {
            inv_scale = 1.0f;
        }

        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        dim3 fp8_grid(n, 2);
        write_kv_cache_fp8_fused_kernel<<<fp8_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)), inv_scale, block_stride, row_elems,
            kv_block_size, n, state.max_blocks_per_seq, state.n_sequences);
    } else {
        // Standard FP16 KV cache write path — fused K+V in single launch
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        dim3 fused_grid(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_fused_kernel<<<fused_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size, n,
            state.max_blocks_per_seq, state.n_sequences);
    }
}

}  // namespace imp

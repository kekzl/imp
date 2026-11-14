// KV cache write dispatch — extracted from executor_forward.cu (RF-004).
// Handles all KV cache dtype paths: TurboQuant, INT4, INT8, FP8, FP16.

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_kernels.cuh"
#include "exec/executor_helpers.h"
#include "quant/fp8_quant.h"
#include "core/kv_dtype.h"
#include "core/kv_fp8_scale.h"
#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "core/pdl_launch.cuh"

namespace imp {

// Row-range view into a shared K/V workspace, starting at row_begin. Per-layer shapes: run_attention
// narrowed K/V to nkv*hd columns, so rows are packed at that width (Gemma-4 ragged prefill read
// sequence 2+ rows at the 4096-column workspace width).
static Tensor kv_rows_view(Tensor t, const ModelConfig& cfg, int row_begin, int row_elems) {
    const bool per_layer_shapes = !cfg.head_dim_per_layer.empty() || !cfg.n_kv_heads_per_layer.empty();
    if (per_layer_shapes && t.shape[1] != row_elems) {
        t.shape[1] = row_elems;
        t.compute_strides();
    }
    if (row_begin > 0)
        t.data = static_cast<char*>(t.data) +
                 (static_cast<size_t>(row_begin) * t.shape[1] * dtype_size(t.qtype));
    return t;
}

// Per-layer K/V geometry (Gemma 4 dual attention geometry), else the cache's.
static void kv_layer_geometry(const ModelConfig& cfg, const KVCache& cache, int layer, int& nkv, int& hd) {
    if (!cfg.n_kv_heads_per_layer.empty() && layer < (int)cfg.n_kv_heads_per_layer.size() &&
        cfg.n_kv_heads_per_layer[layer] > 0) {
        nkv = cfg.n_kv_heads_per_layer[layer];
    } else {
        nkv = cache.n_kv_heads();
    }
    if (!cfg.head_dim_per_layer.empty() && layer < (int)cfg.head_dim_per_layer.size() &&
        cfg.head_dim_per_layer[layer] > 0) {
        hd = cfg.head_dim_per_layer[layer];
    } else {
        hd = cache.head_dim();
    }
}

// kv_cache.fp8_scale=calibrate: FP16 exponent histogram of one calibration K or V view (one-time D2H).
static void add_fp8_kv_hist(const Tensor& t, uint32_t* hist, cudaStream_t stream) {
    std::vector<uint16_t> h(static_cast<size_t>(t.numel()));
    if (h.empty())
        return;
    IMP_CUDA_CHECK_VOID(
        cudaMemcpyAsync(h.data(), t.data, h.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_VOID(cudaStreamSynchronize(stream));
    for (uint16_t b : h)
        ++hist[fp8_kv_hist_bin(b)];
}

Fp8KvScalePair GraphExecutor::fp8_kv_scales_(int kv_layer, int row_begin, int n, int row_elems,
                                             cudaStream_t stream) {
    // FP8 E4M3 KV scale per layer: the first prefill after Engine::warmup() resets calibrated_
    // sets it (max with the 1.0 floor), every later prefill reuses it (cached FP8 blocks were
    // written with it). The reset avoids warmup pollution (synthetic BOS tokens give a too-small scale).
    if (!kv_calibrated_.empty() && kv_layer < static_cast<int>(kv_calibrated_.size()) &&
        !kv_calibrated_[kv_layer]) {
        // Narrows the calibration view to the per-layer K/V shape. k_/v_
        // workspaces are sized for max_nkv*max_head_dim across all layers (Gemma-4
        // dual head_dim 256/512); without narrowing, calibrate_fp8_scale would
        // absmax-reduce over uninitialized memory past the live region for
        // smaller-head_dim layers, locking the FP8 range to junk (was the root cause of the Gemma-4
        // force-FP16 carve-out, engine.cpp:516).
        const auto& cfg = model_->config();
        Tensor kv_cal = kv_rows_view(view_tokens(k_, n), cfg, row_begin, row_elems);
        Tensor vv_cal = kv_rows_view(view_tokens(v_, n), cfg, row_begin, row_elems);
        const int64_t live_cols = row_elems;
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
        // kv_cache.fp8_scale=calibrate: the engine turns these into static scales (an all-zero
        // tensor reads as 448: calibrate_fp8_scale returns 1.0 for it).
        const size_t cal_k = 2 * static_cast<size_t>(kv_layer);
        if (kv_cal_collect_ && cal_k + 1 < kv_cal_amax_.size()) {
            kv_cal_amax_[cal_k] = k_scale * kFp8KvE4M3Max;
            kv_cal_amax_[cal_k + 1] = v_scale * kFp8KvE4M3Max;
            if ((cal_k + 2) * kFp8KvHistBins <= kv_cal_hist_.size()) {
                add_fp8_kv_hist(kv_cal, &kv_cal_hist_[cal_k * kFp8KvHistBins], stream);
                add_fp8_kv_hist(vv_cal, &kv_cal_hist_[(cal_k + 1) * kFp8KvHistBins], stream);
            }
        }
        float new_scale = std::max(k_scale, v_scale);
        if (new_scale < 1e-12f)
            new_scale = 1.0f;
        // Promote only — the high-water mark is the union of every
        // prefill we've seen, so values fit FP8 dynamic range. Legacy: one shared scale (k == v).
        const float shared = std::max(kv_scales_[kv_layer].k, new_scale);
        kv_scales_[kv_layer] = {shared, shared};
        kv_calibrated_[kv_layer] = true;
        return kv_scales_[kv_layer];
    }
    if (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size()))
        return kv_scales_[kv_layer];
    return {};
}

void GraphExecutor::set_kv_layer_map(std::vector<int> map) {
    kv_layer_map_ = std::move(map);
    int n_kv = 0;
    for (int idx : kv_layer_map_)
        n_kv = std::max(n_kv, idx + 1);  // -1 = not an attention layer
    kv_scales_.assign(n_kv, Fp8KvScalePair{});
    kv_calibrated_.assign(n_kv, false);
    kv_scales_static_ = false;
}

void GraphExecutor::reset_kv_calibration() {
    if (kv_scales_static_)
        return;  // kv_cache.fp8_scale checkpoint / calibrate: fixed for the process
    std::fill(kv_scales_.begin(), kv_scales_.end(), Fp8KvScalePair{});
    std::fill(kv_calibrated_.begin(), kv_calibrated_.end(), false);
    ++kv_calib_gen_;
}

bool GraphExecutor::kv_scales_calibrated() const {
    return std::find(kv_calibrated_.begin(), kv_calibrated_.end(), false) == kv_calibrated_.end();
}

void GraphExecutor::set_static_kv_scales(const std::vector<Fp8KvScalePair>& scales) {
    IMP_CHECK(scales.size() == kv_scales_.size(), "set_static_kv_scales: %zu scales for %zu KV layers",
              scales.size(), kv_scales_.size());
    kv_scales_ = scales;
    kv_calibrated_.assign(kv_scales_.size(), true);
    kv_scales_static_ = true;
    kv_cal_collect_ = false;
    ++kv_calib_gen_;  // prefill graphs keyed on the generation never replay an older scale
}

void GraphExecutor::begin_kv_scale_calibration() {
    // Scale 0 drops the 1.0 floor: the std::max above keeps the calibration prefill's own amax / 448.
    kv_scales_static_ = false;
    kv_cal_collect_ = true;
    std::fill(kv_scales_.begin(), kv_scales_.end(), Fp8KvScalePair{0.0f, 0.0f});
    std::fill(kv_calibrated_.begin(), kv_calibrated_.end(), false);
    kv_cal_amax_.assign(2 * kv_scales_.size(), 0.0f);
    kv_cal_hist_.assign(2 * kv_scales_.size() * kFp8KvHistBins, 0u);
    ++kv_calib_gen_;
}

void GraphExecutor::round_prefill_kv_to_cache(int layer, const InferenceState& state, Tensor& seq_k,
                                              Tensor& seq_v, int row_begin, int n_rows, cudaStream_t stream) {
    if (state.kv_cache == nullptr || n_rows <= 0 || state.kv_cache->qtype() != QType::FP8_E4M3)
        return;
    const int kv_layer = get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0 || model_->config().layer_kv_shared(layer))
        return;  // no write of its own (see write_kv_cache)
    int nkv = 0;
    int hd = 0;
    kv_layer_geometry(model_->config(), *state.kv_cache, layer, nkv, hd);
    const int row_elems = nkv * hd;
    IMP_CHECK(seq_k.shape[1] >= row_elems && seq_v.shape[1] >= row_elems,
              "round_prefill_kv_to_cache: K/V rows narrower than nkv*hd=%d", row_elems);
    // The scales paged_kv_gather_fp8_to_fp16 reads (gather_past_kv_fp16).
    const Fp8KvScalePair s = fp8_kv_scales_(kv_layer, row_begin, n_rows, row_elems, stream);
    round_kv_fp8_inplace_kernel<<<dim3(n_rows, 2), std::min(row_elems, 256), 0, stream>>>(
        static_cast<half*>(seq_k.data), static_cast<half*>(seq_v.data), seq_k.shape[1], seq_v.shape[1],
        row_elems, n_rows, 1.0f / s.k, s.k, 1.0f / s.v, s.v);
    IMP_CUDA_CHECK_LAUNCH();
}

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream,
                                   int row_begin, int n_rows, const int* bt_flat,
                                   const int* bt_swa_flat, const int* positions_override) {
    if (!state.kv_cache || !state.block_tables)
        return;

    // Map global layer index to KV cache layer index
    int kv_layer = get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0)
        return;  // not an attention layer
    if (model_->config().layer_kv_shared(layer))
        return;  // Gemma-4 KV-shared layer: its slot is the source layer's, written there

    KVCache* cache = state.kv_cache;
    const bool row_range = (bt_flat != nullptr);
    int n = (n_rows >= 0) ? n_rows : state.n_tokens;
    const int* positions = positions_override ? positions_override : state.positions;
    // Row-range mode indexes the flat per-seq table (single-sequence kernel
    // semantics); the default mode keeps the historical 2D/flat selection.
    const int wr_max_blocks = row_range ? 0 : state.max_blocks_per_seq;
    const int wr_n_seq = row_range ? 1 : state.n_sequences;
    const auto& cfg = model_->config();
    // SWA-aware KV sizing (kv_cache.swa_sizing): windowed layers write into
    // the SWA block group through the parallel table (-1 holes below the
    // window are skipped by the kernels' block_id<0 guard).
    const bool layer_swa = layer_swa_window(cfg, model_->profile(), layer) > 0;
    const int* block_tables = row_range ? bt_flat : state.block_tables;
    if (row_range) {
        if (bt_swa_flat != nullptr && layer_swa)
            block_tables = bt_swa_flat;
    } else if (state.block_tables_swa != nullptr && layer_swa) {
        block_tables = state.block_tables_swa;
    }
    int nkv = 0;
    int hd = 0;
    kv_layer_geometry(cfg, *cache, layer, nkv, hd);
    const int kv_block_size = cache->block_size();
    int row_elems = nkv * hd;
    int block_stride = kv_block_size * row_elems;
    auto view_rows = [&](const Tensor& buf, int rows) -> Tensor {
        return kv_rows_view(view_tokens(buf, rows), cfg, row_begin, row_elems);
    };

    int threads = std::min(row_elems, 256);

    // Quantized write strides from kv_dtype_info(): packed data in bytes, scales in entries.
    const QType kvq = cache->qtype();
    const int packed_block_stride = static_cast<int>(kv_block_data_bytes(kvq, kv_block_size, nkv, hd));
    const int scale_entry_stride = static_cast<int>(kv_block_scale_entries(kvq, kv_block_size, nkv, hd));

    bool use_fp8 = (cache->qtype() == QType::FP8_E4M3);
    bool use_int8 = (cache->qtype() == QType::INT8);
    bool use_int4 = (cache->qtype() == QType::INT4);
    bool use_nvfp4 = (cache->qtype() == QType::NVFP4);
    bool use_mxfp4_kv = (cache->qtype() == QType::MXFP4_KV);
    if (use_nvfp4) {
        // NVFP4 quantized KV cache write — 2 FP4 values packed per byte, UE4M3 scale per group of 16
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        dim3 grid_nvfp4(n, 2);
        pdl::enable_kernel(write_kv_cache_nvfp4_kernel);
        pdl::launch(write_kv_cache_nvfp4_kernel, grid_nvfp4, dim3(256), size_t(0), stream,
                    static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
                    block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
                    static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
                    static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                    static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), packed_block_stride,
                    scale_entry_stride, nkv, hd, kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_mxfp4_kv) {
        // MXFP4-KV quantized KV cache write — identical layout to NVFP4 but UE8M0 scales
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        dim3 grid_mxfp4(n, 2);
        write_kv_cache_mxfp4_kv_kernel<<<grid_mxfp4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions, block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), packed_block_stride, scale_entry_stride,
            nkv, hd, kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_int4) {
        // INT4 quantized KV cache write — 2 values packed per byte, per-head scales
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);

        dim3 grid_int4(n, 2);
        write_kv_cache_int4_kernel<<<grid_int4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions, block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), packed_block_stride, scale_entry_stride, nkv,
            hd, kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_int8) {
        // INT8 quantized KV cache write path with per-head scales.
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);

        dim3 grid_int8(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_int8_kernel<<<grid_int8, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions, block_tables,
            static_cast<int8_t*>(cache->k_ptr(kv_layer, 0)), static_cast<int8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), block_stride, scale_entry_stride, nkv, hd,
            kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_fp8) {
        const Fp8KvScalePair s = fp8_kv_scales_(kv_layer, row_begin, n, row_elems, stream);
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        dim3 fp8_grid(n, 2);
        pdl::enable_kernel(write_kv_cache_fp8_fused_kernel);
        pdl::launch(write_kv_cache_fp8_fused_kernel, fp8_grid, dim3(threads), size_t(0), stream,
                    static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
                    block_tables, static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                    static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)), 1.0f / s.k, 1.0f / s.v,
                    block_stride, row_elems, kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Standard FP16 KV cache write, fused K+V in one launch. MLA: V (v_) is
        // over-allocated to head_dim (hd) per head by mla_assemble_kv (real
        // v_head_dim values first, tail zeroed), so V already shares K's hd-wide
        // layout here; no asymmetric V path needed. row_elems == nkv*hd for both K and V.
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        dim3 fused_grid(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_fused_kernel<<<fused_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size, n,
            wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Sparse decode attention metadata is maintained by ONE batched all-layer
    // launch at the end of the forward, not the per-layer inline launch this
    // replaced (cost the multi-stream serving prefill ~12% wall).
}

}  // namespace imp

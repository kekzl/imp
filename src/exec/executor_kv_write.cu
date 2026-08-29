// KV cache write dispatch — extracted from executor_forward.cu (RF-004).
// Handles all KV cache dtype paths: TurboQuant, INT4, INT8, FP8, FP16.

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

namespace imp {

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream,
                                   int row_begin, int n_rows, const int* bt_flat,
                                   const int* bt_swa_flat, const int* positions_override) {
    if (!state.kv_cache || !state.block_tables)
        return;

    // Map global layer index to KV cache layer index
    int kv_layer = get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0)
        return;  // not an attention layer

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
    // Row-range view into the shared K/V workspaces: same [rows, cols] layout,
    // starting at row_begin instead of 0.
    auto view_rows = [&](const Tensor& buf, int rows) -> Tensor {
        Tensor t = view_tokens(buf, rows);
        if (row_begin > 0) {
            t.data = static_cast<char*>(t.data) +
                     static_cast<size_t>(row_begin) * t.shape[1] * dtype_size(t.qtype);
        }
        return t;
    };
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
    bool use_nvfp4 = (cache->qtype() == QType::NVFP4);
    bool use_mxfp4_kv = (cache->qtype() == QType::MXFP4_KV);
    if (use_nvfp4) {
        // NVFP4 quantized KV cache write — 2 FP4 values packed per byte, UE4M3 scale per group of 16
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        int nvfp4_block_stride = kv_block_size * nkv * hd / 2;            // bytes
        int nvfp4_scale_block_stride = kv_block_size * nkv * (hd / 16);   // bytes (UE4M3)
        dim3 grid_nvfp4(n, 2);
        write_kv_cache_nvfp4_kernel<<<grid_nvfp4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), nvfp4_block_stride,
            nvfp4_scale_block_stride, nkv, hd, kv_block_size, n, wr_max_blocks,
            wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_mxfp4_kv) {
        // MXFP4-KV quantized KV cache write — identical layout to NVFP4 but UE8M0 scales
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        int mxfp4_block_stride = kv_block_size * nkv * hd / 2;           // bytes (same as NVFP4)
        int mxfp4_scale_block_stride = kv_block_size * nkv * (hd / 16);  // bytes (UE8M0)
        dim3 grid_mxfp4(n, 2);
        write_kv_cache_mxfp4_kv_kernel<<<grid_mxfp4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), mxfp4_block_stride,
            mxfp4_scale_block_stride, nkv, hd, kv_block_size, n, wr_max_blocks,
            wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_int4) {
        // INT4 quantized KV cache write — 2 values packed per byte, per-head scales
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        int int4_block_stride = kv_block_size * nkv * hd / 2;  // bytes (half the INT8 stride)
        int scale_block_stride = kv_block_size * nkv;
        dim3 grid_int4(n, 2);
        write_kv_cache_int4_kernel<<<grid_int4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), int4_block_stride, scale_block_stride, nkv,
            hd, kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (use_int8) {
        // INT8 quantized KV cache write path with per-head scales.
        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);

        int scale_block_stride = kv_block_size * nkv;
        dim3 grid_int8(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_int8_kernel<<<grid_int8, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<int8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<int8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<half*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_scale_ptr(kv_layer, 0)), block_stride, scale_block_stride, nkv, hd,
            kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
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
            Tensor kv_cal = view_rows(k_, n);
            Tensor vv_cal = view_rows(v_, n);
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

        Tensor kv = view_rows(k_, n);
        Tensor vv = view_rows(v_, n);
        dim3 fp8_grid(n, 2);
        write_kv_cache_fp8_fused_kernel<<<fp8_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), positions,
            block_tables, static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)), inv_scale, block_stride, row_elems,
            kv_block_size, n, wr_max_blocks, wr_n_seq);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Standard FP16 KV cache write path — fused K+V in single launch.
        //
        // MLA note: the V workspace (v_) is over-allocated to head_dim (hd) per
        // head by mla_assemble_kv (real v_head_dim values first, tail zeroed), so
        // V already shares K's hd-wide layout here. The fused write therefore
        // stores hd-wide K and hd-wide (padded) V uniformly — no asymmetric V
        // path is needed. Decode reads only v_head_dim per V head; the zero tail
        // is harmless. row_elems == nkv * hd for both K and V.
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
    // launch at the end of the forward (run_forward) for every forward shape
    // - the per-layer inline launch that used to sit here cost the
    // multi-stream serving prefill ~12% wall (2026-08-29).

    // ─── Phase 3c: BitDecoding residual write-through (decode only) ────────
    //
    // Append each seq's just-computed FP16 K/V (one token per seq,
    // n_sequences ≥ 1, n_tokens == n_sequences) to its residual ring slot.
    // The paged write above already cached the same data in its native dtype;
    // the residual is a lookaside copy that lets the TC attention kernel skip
    // dequant on the freshest tokens. Eviction-free: when the ring fills, the
    // slot at write_idx is overwritten and the older copy stays in paged.
    // Skipped on prefill (warm-up writes only), and non-NVFP4 caches (residual
    // is gated to NVFP4 by KVCacheManager::enable_residual_buffer).
    //
    // Two seq-id sources, mirroring the attention dispatcher:
    //   - state.h_residual_seq_ids: host array of length n_sequences (multi-seq)
    //   - state.kv_seq_id: single int (legacy single-seq, used when h_… is null)
    if (!row_range && !state.is_prefill && use_nvfp4 && state.kv_manager != nullptr &&
        state.kv_manager->residual_enabled() && n == state.n_sequences) {
        const int res_n_kv = cache->n_kv_heads();
        const int res_hd = cache->head_dim();
        // Per-layer geometry must match the residual's allocated stride
        // (uniform-head_dim guard from enable_residual_buffer); skip on
        // mismatch (e.g. Gemma-4 dual-head_dim layers).
        if (res_n_kv == nkv && res_hd == hd) {
            const int slot_elems = nkv * hd;
            Tensor kv_src = view_tokens(k_, n);
            Tensor vv_src = view_tokens(v_, n);
            const half* src_k_base = static_cast<const half*>(kv_src.data);
            const half* src_v_base = static_cast<const half*>(vv_src.data);
            constexpr int kThreads = 256;
            const int blocks_y = (slot_elems + kThreads - 1) / kThreads;

            if (state.n_sequences == 1) {
                // Single-seq fast path: resolve destination on the host and
                // pass scalar pointers — avoids per-step device-pointer-array
                // upload. Two cudaMemcpyAsync had been the bottleneck here
                // (-3× decode regression at 4K ctx); a single kernel launch
                // is several × cheaper.
                int seq_id;
                if (state.h_residual_seq_ids != nullptr) {
                    seq_id = state.h_residual_seq_ids[0];
                } else if (state.kv_seq_id >= 0) {
                    seq_id = state.kv_seq_id;
                } else {
                    seq_id = -1;
                }
                if (seq_id >= 0) {
                    int slot = state.kv_manager->residual_slot_of(seq_id);
                    void* base_k = state.kv_manager->residual_k_ptr(seq_id, kv_layer);
                    void* base_v = state.kv_manager->residual_v_ptr(seq_id, kv_layer);
                    if (slot >= 0 && base_k != nullptr && base_v != nullptr) {
                        // Graph-safe: kernel reads write_idx from device at execution
                        // time. Per-step advance happens once at end of forward_logits
                        // (a tiny advance_residual_state_kernel), inside the captured
                        // graph — replays update ring state correctly.
                        dim3 grid_single(2, blocks_y);
                        residual_kv_write_indirect_kernel<<<grid_single, kThreads, 0, stream>>>(
                            src_k_base, src_v_base,
                            static_cast<half*>(base_k), static_cast<half*>(base_v),
                            state.kv_manager->d_residual_widx_ptr(), slot, slot_elems);
                        IMP_CUDA_CHECK_LAUNCH();
                        // No host advance_residual: ring state lives on device, advanced
                        // once per step by advance_residual_state_kernel in forward_logits.
                    }
                }
            } else if (state.d_residual_seq_slots != nullptr) {
                // Multi-seq, graph-safe (#1708). Everything the destination
                // depends on is resolved on the DEVICE at execution time: the
                // layer base plus the per-seq stride, the residual slot from
                // the engine's per-step upload, and the ring index from the
                // manager's device array.
                //
                // What this replaces built a device array of destination
                // pointers per call and per layer, with `cudaMallocAsync` and a
                // matching `cudaFreeAsync` around the launch - inside the
                // captured region. A replay wrote through the captured address
                // after it had been freed and reused ("an illegal memory
                // access"), with the ring index frozen at capture time on top.
                // It was written when `n_sequences > 1` was rare; the comment
                // said so.
                void* k_layer_base = state.kv_manager->residual_k_layer_base(kv_layer);
                void* v_layer_base = state.kv_manager->residual_v_layer_base(kv_layer);
                const int* d_widx = state.kv_manager->d_residual_widx_ptr();
                if (k_layer_base != nullptr && v_layer_base != nullptr && d_widx != nullptr) {
                    const int64_t seq_stride_elems = static_cast<int64_t>(
                        state.kv_manager->residual_seq_stride_bytes() / sizeof(half));
                    dim3 grid_multi(n, 2);
                    residual_kv_write_multi_indirect_kernel<<<grid_multi, kThreads, 0, stream>>>(
                        src_k_base, src_v_base, static_cast<half*>(k_layer_base),
                        static_cast<half*>(v_layer_base), seq_stride_elems, state.d_residual_seq_slots,
                        d_widx, slot_elems);
                    IMP_CUDA_CHECK_LAUNCH();
                }
                // No host advance_residual: the ring is advanced on device once
                // per step by advance_residual_state_multi_kernel, inside the
                // captured graph. The host call this replaces ran at capture
                // time only, so replays left the ring where the capture found
                // it.
            }
        }
    }
}

}  // namespace imp

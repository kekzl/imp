// KV cache write dispatch — extracted from executor_forward.cu (RF-004).
// Handles all KV cache dtype paths: TurboQuant, INT4, INT8, FP8, FP16.

#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
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
    bool use_nvfp4 = (cache->qtype() == QType::NVFP4);
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
    } else if (use_nvfp4) {
        // NVFP4 quantized KV cache write — 2 FP4 values packed per byte, UE4M3 scale per group of 16
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int nvfp4_block_stride = kv_block_size * nkv * hd / 2;            // bytes
        int nvfp4_scale_block_stride = kv_block_size * nkv * (hd / 16);   // bytes (UE4M3)
        dim3 grid_nvfp4(n, 2);
        write_kv_cache_nvfp4_kernel<<<grid_nvfp4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), nvfp4_block_stride,
            nvfp4_scale_block_stride, nkv, hd, kv_block_size, n, state.max_blocks_per_seq,
            state.n_sequences);
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
    static const bool skip_residual_write = []() {
        const char* e = std::getenv("IMP_BITDECODING_SKIP_WRITE");
        return e && e[0] == '1';
    }();
    if (!skip_residual_write && !state.is_prefill && use_nvfp4 && state.kv_manager != nullptr &&
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
                    void* dst_k = state.kv_manager->residual_k_ptr(seq_id, kv_layer);
                    void* dst_v = state.kv_manager->residual_v_ptr(seq_id, kv_layer);
                    if (dst_k != nullptr && dst_v != nullptr) {
                        auto rs = state.kv_manager->residual_state(seq_id);
                        half* k_dst = static_cast<half*>(dst_k) + rs.write_idx * slot_elems;
                        half* v_dst = static_cast<half*>(dst_v) + rs.write_idx * slot_elems;
                        // Diagnostic env to skip the kernel launch but still advance ring state.
                        static const bool no_kernel_launch = []() {
                            const char* e = std::getenv("IMP_BITDECODING_NO_LAUNCH");
                            return e && e[0] == '1';
                        }();
                        if (!no_kernel_launch) {
                            dim3 grid_single(2, blocks_y);
                            residual_kv_write_single_kernel<<<grid_single, kThreads, 0, stream>>>(
                                src_k_base, src_v_base, k_dst, v_dst, slot_elems);
                        }
                        state.kv_manager->advance_residual(seq_id);
                    }
                }
            } else if (state.h_residual_seq_ids != nullptr) {
                // Multi-seq: build device pointer arrays per layer (host-side,
                // upload via the per-step residual_meta buffer is left to the
                // engine to pre-upload — we fall back to per-call upload here
                // for simplicity; n_sequences > 1 is rare so this is OK).
                std::vector<half*> k_ptrs(n, nullptr), v_ptrs(n, nullptr);
                for (int s = 0; s < n; s++) {
                    int seq_id = state.h_residual_seq_ids[s];
                    if (seq_id < 0) continue;
                    void* dst_k = state.kv_manager->residual_k_ptr(seq_id, kv_layer);
                    void* dst_v = state.kv_manager->residual_v_ptr(seq_id, kv_layer);
                    if (dst_k == nullptr || dst_v == nullptr) continue;
                    auto rs = state.kv_manager->residual_state(seq_id);
                    k_ptrs[s] = static_cast<half*>(dst_k) + rs.write_idx * slot_elems;
                    v_ptrs[s] = static_cast<half*>(dst_v) + rs.write_idx * slot_elems;
                }
                half** d_k_ptrs = nullptr;
                half** d_v_ptrs = nullptr;
                cudaMallocAsync(&d_k_ptrs, n * sizeof(half*), stream);
                cudaMallocAsync(&d_v_ptrs, n * sizeof(half*), stream);
                cudaMemcpyAsync(d_k_ptrs, k_ptrs.data(), n * sizeof(half*),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_v_ptrs, v_ptrs.data(), n * sizeof(half*),
                                cudaMemcpyHostToDevice, stream);
                dim3 grid_multi(n, 2);
                residual_kv_write_multi_kernel<<<grid_multi, kThreads, 0, stream>>>(
                    src_k_base, src_v_base, d_k_ptrs, d_v_ptrs, slot_elems);
                cudaFreeAsync(d_k_ptrs, stream);
                cudaFreeAsync(d_v_ptrs, stream);
                for (int s = 0; s < n; s++) {
                    int seq_id = state.h_residual_seq_ids[s];
                    if (seq_id >= 0) state.kv_manager->advance_residual(seq_id);
                }
            }
        }
    }
}

}  // namespace imp

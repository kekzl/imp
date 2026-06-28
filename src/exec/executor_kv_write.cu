// KV cache write dispatch — extracted from executor_forward.cu (RF-004).
// Handles all KV cache dtype paths: TurboQuant, INT4, INT8, FP8, FP16.

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_kernels_internal.cuh"
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

// MLA V write: source V is compact [n_tokens, nkv * v_head_dim],
// destination slots are hd-wide (over-allocated, kv_block_stride = kv_bs * nkv * hd).
// Writes v_row_elems = nkv * v_head_dim elements per slot at the slot base address.
// The remaining hd - v_head_dim elements per slot are left uninitialised (never read).
__global__ __launch_bounds__(256)
void write_kv_cache_mla_v_kernel(const half* __restrict__ v_in,
                                  const int* __restrict__ positions,
                                  const int* __restrict__ block_tables,
                                  half* __restrict__ v_cache_base,
                                  int kv_block_stride,   // kv_bs * nkv * hd  (pool stride in elements)
                                  int slot_stride,       // nkv * hd            (slot stride in pool, elements)
                                  int v_row_elems,       // nkv * v_head_dim   (src row width / copy width)
                                  int block_size,
                                  int n_tokens,
                                  int max_blocks_per_seq,
                                  int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx,
                                    max_blocks_per_seq, n_sequences, slot_in_block);
    if (block_id < 0)
        return;

    // dst is at the slot base: write v_row_elems elements (compact source)
    // into a slot_stride-wide slot (over-allocated with hd per head).
    half* dst = v_cache_base + static_cast<int64_t>(block_id) * kv_block_stride +
                static_cast<int64_t>(slot_in_block) * slot_stride;
    const half* src = v_in + static_cast<int64_t>(token_idx) * v_row_elems;

    // Scalar copy (v_row_elems = nkv * vhd, typically a multiple of 8)
    for (int i = threadIdx.x; i < v_row_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

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
    bool use_mxfp4_kv = (cache->qtype() == QType::MXFP4_KV);
    if (use_nvfp4) {
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
    } else if (use_mxfp4_kv) {
        // MXFP4-KV quantized KV cache write — identical layout to NVFP4 but UE8M0 scales
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        int mxfp4_block_stride = kv_block_size * nkv * hd / 2;           // bytes (same as NVFP4)
        int mxfp4_scale_block_stride = kv_block_size * nkv * (hd / 16);  // bytes (UE8M0)
        dim3 grid_mxfp4(n, 2);
        write_kv_cache_mxfp4_kv_kernel<<<grid_mxfp4, 256, 0, stream>>>(
            static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
            state.block_tables, static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
            static_cast<uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), mxfp4_block_stride,
            mxfp4_scale_block_stride, nkv, hd, kv_block_size, n, state.max_blocks_per_seq,
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
        // Standard FP16 KV cache write path.
        // MLA: V has v_head_dim < head_dim. Write K with full hd, V with vhd
        // (over-allocation: V slots are hd-sized but only vhd elements are valid).
        // Non-MLA (vhd == hd): fused single launch as before.
        const int vhd = (cfg.is_mla() && cfg.v_head_dim > 0 && cfg.v_head_dim != hd)
                            ? cfg.v_head_dim : hd;
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        if (vhd != hd) {
            // MLA asymmetric write: K uses hd, V uses vhd
            // K write: standard — uses row_elems = nkv * hd
            dim3 k_grid(n);
            write_kv_cache_kernel<<<k_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv.data), state.positions, state.block_tables,
                static_cast<half*>(cache->k_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size, n,
                state.max_blocks_per_seq, state.n_sequences);
            // V write: asymmetric — src is compact [n, nkv*vhd], dst slot stride = nkv*hd
            int v_row_elems = nkv * vhd;
            int v_threads = std::min(v_row_elems, 256);
            dim3 v_grid(n);
            write_kv_cache_mla_v_kernel<<<v_grid, v_threads, 0, stream>>>(
                static_cast<const half*>(vv.data), state.positions, state.block_tables,
                static_cast<half*>(cache->v_ptr(kv_layer, 0)),
                block_stride,   // pool block stride = kv_bs * nkv * hd (elements)
                row_elems,      // slot stride = nkv * hd (elements)
                v_row_elems,    // src row width = nkv * vhd (elements to copy)
                kv_block_size, n, state.max_blocks_per_seq, state.n_sequences);
        } else {
            // Non-MLA: fused K+V in single launch
            dim3 fused_grid(n, 2);  // blockIdx.y: 0=K, 1=V
            write_kv_cache_fused_kernel<<<fused_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv.data), static_cast<const half*>(vv.data), state.positions,
                state.block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size, n,
                state.max_blocks_per_seq, state.n_sequences);
        }
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
    if (!state.is_prefill && use_nvfp4 && state.kv_manager != nullptr &&
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
                        // No host advance_residual: ring state lives on device, advanced
                        // once per step by advance_residual_state_kernel in forward_logits.
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

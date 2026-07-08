// Legacy MoE prefill fallback: D2H sync + serial/batch dequant + cuBLAS.
// Extracted from executor_forward_moe.cu for maintainability.

#include "exec/executor.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "exec/executor_debug.h"
#include "runtime/config.h"
#include <atomic>
#include "compute/embedding.h"
#include "compute/gemv_ggml_compat.h"
#include "compute/ggml_mmvq.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

namespace imp {

void GraphExecutor::run_moe_legacy_fallback_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int&    n        = ctx.n;
    int&    d        = ctx.d;
    int&    ne       = ctx.ne;
    int&    eff      = ctx.eff;
    int&    expanded = ctx.expanded;
    size_t& es       = ctx.es;
    bool&   non_gated_experts  = ctx.non_gated_experts;
    bool&   use_packed_dequant = ctx.use_packed_dequant;
    MoeRoutingResult& routing  = ctx.routing;

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: legacy FP16 fallback path (n=%d, expanded=%d)", n,
                     expanded);
    {
        moe_host_args_capture_guard(stream);
        std::vector<int32_t> h_offsets(ne + 1);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                           static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Helper: dequant one expert's weight from packed tensor into dequant scratch slot 0.
        // Returns a Tensor view into the scratch buffer with shape [rows, cols], FP16.
        // Uses slot 0 always -- safe because all ops are on the same stream, so the previous
        // GEMM reading from slot 0 completes before the next dequant writes to it.
        auto dequant_expert = [&](const Tensor& packed, QType qtype,
                                  int expert_idx, ExpertProj proj) -> Tensor {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t row_bytes = qtype_row_bytes(qtype, cols);
            size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
            size_t total_raw = static_cast<size_t>(packed.shape[0]) * expert_raw;
            size_t offset = static_cast<size_t>(expert_idx) * expert_raw;

            // Bounds check: verify offset + expert_raw <= total allocated
            if (offset + expert_raw > total_raw) {
                IMP_LOG_ERROR(
                    "dequant_expert: OOB! expert %d offset=%zu + raw=%zu > total=%zu "
                    "(packed shape [%ld,%ld,%ld] qtype=%u)",
                    expert_idx, offset, expert_raw, total_raw, (long)packed.shape[0],
                    (long)packed.shape[1], (long)packed.shape[2], std::to_underlying(qtype));
                return Tensor();
            }

            // Check dequant buffer is large enough
            size_t dequant_needed = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            if (dequant_needed > moe_.dequant_buf_size) {
                IMP_LOG_ERROR(
                    "dequant_expert: dequant buffer too small! "
                    "need=%zu have=%zu (rows=%ld cols=%ld)",
                    dequant_needed, moe_.dequant_buf_size, (long)rows, (long)cols);
                return Tensor();
            }

            const char* src;
            if (!packed.on_device) {
                // Expert weights offloaded to host — try LRU cache first, then staging
                // buffer.
                const char* host_ptr = static_cast<const char*>(packed.data) + offset;
                if (expert_cache_.n_slots_ > 0) {
                    ExpertCacheKey ck{packed.data, expert_idx};
                    void* cached = expert_cache_.get_or_load(layer, proj, ck, host_ptr,
                                                             expert_raw, stream);
                    src = static_cast<const char*>(cached);
                } else if (moe_.raw_staging_buf) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.raw_staging_buf, host_ptr,
                                                       expert_raw, cudaMemcpyHostToDevice,
                                                       stream));
                    src = static_cast<const char*>(moe_.raw_staging_buf);
                } else {
                    IMP_LOG_ERROR("dequant_expert: no staging buffer for host expert %d",
                                  expert_idx);
                    return Tensor();
                }
            } else {
                src = static_cast<const char*>(packed.data) + offset;
            }

            char* dst = static_cast<char*>(moe_.dequant_buf);  // always slot 0

            dequant_gpu(src, dst, qtype, static_cast<int>(rows), static_cast<int>(cols),
                        stream);

            int64_t shape[2] = {rows, cols};
            return Tensor(dst, QType::F16, 2, shape, true);
        };

        // Helper: try fused quantized GEMV for count=1 decode (dequant+dot in one kernel),
        // else fall back to dequant_expert + cuBLAS gemm.
        // For host-resident experts: H2D to staging buffer, then fused GEMV on staging —
        // eliminates separate dequant_gpu + cuBLAS gemm overhead.
        auto expert_gemm = [&](const Tensor& a, Tensor& c, const Tensor& packed, QType qtype,
                               const std::vector<Tensor>& fallback,
                               const std::vector<TensorID>& fallback_ids, int eidx,
                               ExpertProj proj) {
            // NVFP4 MoE batch cache path (Nemotron-H non-gated, and any
            // NVFP4 MoE model when batch_dequant_buf is too small to fire
            // the NVFP4→FP16 batch path). After cache_moe_native_nvfp4
            // builds the contiguous buffer and frees per-expert allocs,
            // `fallback[eidx].data` is nullptr and dequant_expert can't
            // dispatch NVFP4. Slice the cached MoE result instead.
            if (qtype == QType::NVFP4) {
                auto it = wcache_.nvfp4_moe.find(packed.data);
                if (it != wcache_.nvfp4_moe.end()) {
                    const auto& moe_cache = it->second;
                    size_t pkd_off = static_cast<size_t>(eidx) *
                                     moe_cache.expert_stride_packed;
                    size_t ms_off = static_cast<size_t>(eidx) *
                                    moe_cache.expert_stride_ms;
                    // tensor_scale per expert: device array, sync read.
                    // For prefill this fires once per active expert per
                    // layer (~128*3*23 = ~9k syncs for 200-token prompt).
                    // Optimization: pre-cache to host at promote time
                    // (left as follow-up; correctness first).
                    float ts_h = 1.0f;
                    if (moe_cache.tensor_scales) {
                        cudaMemcpyAsync(&ts_h,
                                        moe_cache.tensor_scales + eidx,
                                        sizeof(float),
                                        cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);
                    }
                    NvFP4QuantResult nw;
                    nw.packed_data = static_cast<char*>(moe_cache.packed_data) +
                                     pkd_off;
                    nw.micro_scales = static_cast<char*>(moe_cache.micro_scales) +
                                      ms_off;
                    nw.tensor_scale = ts_h;
                    nw.N = static_cast<int>(moe_cache.N);
                    nw.K = static_cast<int>(moe_cache.K);
                    int M = static_cast<int>(a.shape[0]);
                    if (M == 1) {
                        gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                                        static_cast<half*>(c.data),
                                        static_cast<int>(nw.N),
                                        static_cast<int>(nw.K), stream);
                    } else {
                        int64_t a_shape[2] = {a.shape[0],
                                              static_cast<int64_t>(nw.K)};
                        int64_t c_shape[2] = {a.shape[0],
                                              static_cast<int64_t>(nw.N)};
                        Tensor a_t(
                            const_cast<void*>(static_cast<const void*>(a.data)),
                            QType::F16, 2, a_shape, true);
                        Tensor c_t(c.data, QType::F16, 2, c_shape, true);
                        gemm_nvfp4(nw, a_t, c_t, stream);
                    }
                    return;
                }
            }

            // NVFP4 prequant path: native NVFP4 GEMV (any batch size)
            const bool has_nvfp4_id = (!fallback_ids.empty() &&
                                       static_cast<size_t>(eidx) < fallback_ids.size() &&
                                       fallback_ids[eidx] != kInvalidTensorID &&
                                       registry_.handle(fallback_ids[eidx]).primary_tier ==
                                           StorageTier::NVFP4);
            if (has_nvfp4_id) {
                const auto& wh = registry_.handle(fallback_ids[eidx]);
                NvFP4QuantResult nw;
                nw.packed_data = wh.payload.nvfp4.data;
                nw.micro_scales = wh.payload.nvfp4.block_scales;
                // tensor_scale: payload.nvfp4.tensor_scale is a HOST float pointer
                // (borrowed from wcache_.nvfp4 map entry — stable address). Read directly.
                nw.tensor_scale = (wh.payload.nvfp4.tensor_scale != nullptr)
                                      ? *wh.payload.nvfp4.tensor_scale
                                      : 1.0f;
                nw.N = wh.shape[0];
                // wh.shape[1] stores the PACKED column count (K/2 for FP4 packed format).
                // NvFP4QuantResult.K must be the logical K = packed_cols * 2.
                nw.K = wh.shape[1] * 2;
                int N_dim = static_cast<int>(nw.N);
                int K_dim = static_cast<int>(nw.K);
                int M = static_cast<int>(a.shape[0]);

                if (M == 1) {
                    // Single-token decode: direct NVFP4 GEMV (verified coherent).
                    gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                                    static_cast<half*>(c.data), N_dim, K_dim, stream);
                } else {
                    // Multi-token (legacy MoE prefill): the per-row gemv_nvfp4_kpar
                    // loop produces wrong output on Gemma-4 NVFP4 experts even though
                    // it works for Mistral dense decode at the same kernel/dimensions
                    // (see commit message + memory/llm_compressor_phase2_item2…). The
                    // dense-path mirror — gemm_nvfp4 (NVFP4 → FP16 dequant + cuBLAS
                    // gemm) — is correct on Gemma-4 and is what Mistral dense prefill
                    // already uses, so route the multi-token expert prefill through
                    // it. Bisected via IMP_EXPERT_NVFP4_DEQUANT_MR=1 on 2026-04-27:
                    // M=1 on gemv_kpar + M>1 on gemm_nvfp4 → "The capital of France
                    // is Paris."; M>1 on gemv_kpar → token-stuck loop.
                    int64_t a_shape[2] = {static_cast<int64_t>(M),
                                          static_cast<int64_t>(K_dim)};
                    int64_t c_shape[2] = {static_cast<int64_t>(M),
                                          static_cast<int64_t>(N_dim)};
                    Tensor a_t(const_cast<void*>(static_cast<const void*>(a.data)),
                               QType::F16, 2, a_shape, true);
                    Tensor c_t(c.data, QType::F16, 2, c_shape, true);
                    gemm_nvfp4(nw, a_t, c_t, stream);
                }
                return;
            }
            if (a.shape[0] == 1 && use_packed_dequant && compute_dtype_ == QType::F16 &&
                (qtype == QType::Q6_K || qtype == QType::Q8_0)) {
                int64_t rows = packed.shape[1];
                int64_t cols = packed.shape[2];
                size_t rb = qtype_row_bytes(qtype, cols);
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
                        w = expert_cache_.get_or_load(layer, proj, ck, host_ptr,
                                                       expert_raw, stream);
                    } else if (moe_.raw_staging_buf && expert_raw <= moe_.raw_staging_size) {
                        cudaMemcpyAsync(moe_.raw_staging_buf, host_ptr, expert_raw,
                                        cudaMemcpyHostToDevice, stream);
                        w = moe_.raw_staging_buf;
                    }
                }

                if (w) {
                    auto fn = (qtype == QType::Q6_K) ? gemv_q6k : gemv_q8_0;
                    fn(w, static_cast<const half*>(a.data), static_cast<half*>(c.data),
                       static_cast<int>(rows), static_cast<int>(cols), stream);
                    return;
                }
            }
            // Fallback: separate dequant + cuBLAS GEMM
            {
                Tensor b = use_packed_dequant ? dequant_expert(packed, qtype, eidx, proj)
                                              : fallback[eidx];
                if (!b.data)
                    return;  // dequant_expert failed (OOB or buffer too small)

                // SafeTensors NVFP4 prequant: per-expert weights got promoted to
                // qtype=NVFP4 + scales/tensor_scale sidecars at engine init
                // (executor_pre_dequant.cu Phase 0). The legacy fallback below
                // expects an FP16 weight; calling cuBLAS gemm with qtype=NVFP4
                // would crash with "unsupported dtype 71". Route through the
                // native NVFP4 path — same logic as the WeightHandle-driven
                // has_nvfp4_id branch above.
                if (b.qtype == QType::NVFP4 && b.scales != nullptr) {
                    NvFP4QuantResult nw;
                    nw.packed_data = b.data;
                    nw.micro_scales = b.scales;
                    nw.tensor_scale = b.tensor_scale;
                    nw.N = static_cast<int>(b.shape[0]);
                    nw.K = static_cast<int>(b.shape[1]) * 2;  // packed → logical
                    if (a.shape[0] == 1) {
                        gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                                        static_cast<half*>(c.data), static_cast<int>(nw.N),
                                        static_cast<int>(nw.K), stream);
                    } else {
                        int64_t a_shape[2] = {a.shape[0], static_cast<int64_t>(nw.K)};
                        int64_t c_shape[2] = {a.shape[0], static_cast<int64_t>(nw.N)};
                        Tensor a_t(const_cast<void*>(static_cast<const void*>(a.data)),
                                   QType::F16, 2, a_shape, true);
                        Tensor c_t(c.data, QType::F16, 2, c_shape, true);
                        gemm_nvfp4(nw, a_t, c_t, stream);
                    }
                    return;
                }

                gemm(a, b, c, 1.0f, 0.0f, stream);
            }
        };

        char* gathered_base = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

        // Helper: get FP16 expert weight pointer from pre-dequant cache or unpacked weights.
        // fp16_cache is the borrowed Tensor* for the packed tensor's FP16 cache entry.
        auto get_fp16_expert_ptr = [&](const Tensor& packed, QType /*qtype*/,
                                       const std::vector<Tensor>& fallback,
                                       const Tensor* fp16_cache, int eidx) -> const void* {
            if (fp16_cache != nullptr) {
                int64_t rows = packed.shape[1];
                int64_t cols = packed.shape[2];
                size_t expert_offset = static_cast<size_t>(eidx) * rows * cols * sizeof(half);
                return static_cast<const char*>(fp16_cache->data) + expert_offset;
            }
            if (!fallback.empty() && static_cast<size_t>(eidx) < fallback.size() &&
                fallback[eidx].data && fallback[eidx].qtype == QType::F16 &&
                fallback[eidx].on_device) {
                return fallback[eidx].data;
            }
            return nullptr;
        };

        // Helper: batch dequant all experts + single grouped GEMM.
        // Dequants all experts to FP16, then runs a single batched GEMM.
        // CUTLASS 2.x GemmGrouped provides lower launch overhead than cuBLAS.
        auto chunked_dequant_gemm = [&](const Tensor& packed, QType qtype,
                                        const std::vector<Tensor>& fallback,
                                        const std::vector<TensorID>& fallback_ids,
                                        const char* a_base, char* c_base, int K_dim,
                                        int N_dim, ExpertProj proj) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);
            size_t expert_raw_sz = static_cast<size_t>(rows) * qtype_row_bytes(qtype, cols);

            if (!moe_.batch_dequant_buf || expert_fp16_sz == 0) {
                // No buffer — serial fallback
                for (int e = 0; e < ne; ++e) {
                    int start = h_offsets[e];
                    int count = h_offsets[e + 1] - start;
                    if (count == 0)
                        continue;
                    int64_t count64 = static_cast<int64_t>(count);
                    int64_t a_shape[2] = {count64, static_cast<int64_t>(K_dim)};
                    Tensor a_view(const_cast<void*>(static_cast<const void*>(
                                      a_base + static_cast<size_t>(start) * K_dim * es)),
                                  compute_dtype_, 2, a_shape, true);
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(N_dim)};
                    Tensor c_view(c_base + static_cast<size_t>(start) * N_dim * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, packed, qtype, fallback, fallback_ids, e,
                                proj);
                }
                return;
            }

            const uint8_t* raw_base = static_cast<const uint8_t*>(packed.data);
            char* buf = static_cast<char*>(moe_.batch_dequant_buf);

            // Dequant all experts in one batch, then single GEMM.
            // With pp=512 and top_k=8, nearly all 128 experts are active, so
            // dequanting all at once is optimal (one big bandwidth-saturating kernel).
            dequant_gpu(raw_base, buf, qtype, ne * static_cast<int>(rows),
                        static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            // Use cublasGemmGroupedBatchedEx — single call for all experts.
            // We already have h_offsets from D2H sync, so no need for
            // gemm_moe_device_grouped (which does its own D2H sync + 128
            // individual cublasLtMatmul calls).
            gemm_moe_batched(a_base, c_base, h_offsets.data(), b_ptrs.data(), K_dim, N_dim,
                             QType::F16, ne, stream, moe_.d_work_ptrs);
        };

        // Determine which path to use:
        // 1. Pre-cached FP16 path: all experts in fp16_packed_*_cache (fastest, no dequant)
        // 2. Dequant-then-batch path: packed experts on device + batch buffer available
        // 3. Serial path: fallback (one expert at a time)
        // Note: fused Q6K dp4a path is handled above (before the D2H sync).

        bool has_precached_up = (ly.fp16_packed_up_cache != nullptr);
        bool can_dequant_batch = (moe_.batch_dequant_buf != nullptr &&
                                  ly.expert_up_packed.data != nullptr &&
                                  ly.expert_up_packed.on_device &&
                                  dequant_gpu_supported(ly.expert_up_packed.qtype));

        if (has_precached_up) {
            // Pre-cached FP16 path — all expert packs in fp16_packed_*_cache
            // ===== PRE-CACHED FP16 BATCHED GEMM PATH =====
            std::vector<const void*> gate_w_ptrs(ne, nullptr);
            std::vector<const void*> up_w_ptrs(ne, nullptr);
            std::vector<const void*> down_w_ptrs(ne, nullptr);

            for (int e = 0; e < ne; e++) {
                up_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_up_packed,
                                                   ly.expert_up_packed.qtype, ly.expert_w_up,
                                                   ly.fp16_packed_up_cache, e);
                if (!non_gated_experts)
                    gate_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_gate_packed,
                                                         ly.expert_gate_packed.qtype,
                                                         ly.expert_w_gate,
                                                         ly.fp16_packed_gate_cache, e);
                down_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_down_packed,
                                                     ly.expert_down_packed.qtype,
                                                     ly.expert_w_down,
                                                     ly.fp16_packed_down_cache, e);
            }

            if (!non_gated_experts)
                gemm_moe_batched(gathered_base, expert_gate_base, h_offsets.data(),
                                 gate_w_ptrs.data(), d, eff, QType::F16, ne, stream,
                                 moe_.d_work_ptrs);
            gemm_moe_batched(gathered_base, expert_up_base, h_offsets.data(),
                             up_w_ptrs.data(), d, eff, QType::F16, ne, stream,
                             moe_.d_work_ptrs);

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                    compute_dtype_, cfg.ffn_activation, stream);

            {
                char* batch_down_act = non_gated_experts ? expert_up_base
                                                         : expert_swiglu_base;
                gemm_moe_batched(batch_down_act, expert_down_base, h_offsets.data(),
                                 down_w_ptrs.data(), eff, d, QType::F16, ne, stream,
                                 moe_.d_work_ptrs);
            }

        } else if (can_dequant_batch) {
            // ===== BATCH DEQUANT + GROUPED GEMM =====
            // Dequant all experts to FP16, then single grouped GEMM via CUTLASS.

            if (!non_gated_experts)
                chunked_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_packed.qtype,
                                     ly.expert_w_gate, ly.expert_gate_ids, gathered_base,
                                     expert_gate_base, d, eff, ExpertProj::Gate);
            chunked_dequant_gemm(ly.expert_up_packed, ly.expert_up_packed.qtype,
                                 ly.expert_w_up, ly.expert_up_ids, gathered_base,
                                 expert_up_base, d, eff, ExpertProj::Up);

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                    compute_dtype_, cfg.ffn_activation, stream);

            {
                char* dequant_down_act = non_gated_experts ? expert_up_base
                                                           : expert_swiglu_base;
                chunked_dequant_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype,
                                     ly.expert_w_down, ly.expert_down_ids, dequant_down_act,
                                     expert_down_base, eff, d, ExpertProj::Down);
            }

        } else {
            // ===== SERIAL PATH (fallback) =====
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0)
                    continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, a_shape, true);

                if (!non_gated_experts) {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_gate_packed,
                                ly.expert_gate_packed.qtype, ly.expert_w_gate,
                                ly.expert_gate_ids, e, ExpertProj::Gate);
                }

                {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_up_packed,
                                ly.expert_up_packed.qtype, ly.expert_w_up, ly.expert_up_ids,
                                e, ExpertProj::Up);
                }
            }

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                    compute_dtype_, cfg.ffn_activation, stream);

            // Down projection activation source: up buffer for non-gated (relu² in-place),
            // swiglu buffer for gated.
            char* down_act_base = non_gated_experts ? expert_up_base : expert_swiglu_base;
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0)
                    continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor a_view(down_act_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, a_shape, true);
                int64_t c_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor c_view(expert_down_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, c_shape, true);
                expert_gemm(a_view, c_view, ly.expert_down_packed,
                            ly.expert_down_packed.qtype, ly.expert_w_down, ly.expert_down_ids,
                            e, ExpertProj::Down);
            }
        }
    }
}

}  // namespace imp

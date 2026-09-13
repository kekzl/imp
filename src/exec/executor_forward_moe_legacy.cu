// Legacy MoE prefill fallback: D2H sync + serial/batch dequant + cuBLAS.
// Extracted from executor_forward_moe.cu for maintainability.

#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/nvfp4_expert_offload.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "exec/executor_debug.h"
#include <atomic>
#include "compute/embedding.h"
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
#include "core/pdl.h"
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

        // Use the LRU cache only when this dispatch's working set fits the layer's
        // slot pool: one dispatch touches kExpertProjCount cells per ACTIVE
        // expert, and above slots_per_layer the cache retains nothing (every
        // access misses and evicts), doing strictly more work than single-slot
        // staging for the same H2D bytes. Decode stays under the pool; prefill
        // (essentially every expert active) is what overflows it. Harness:
        // tools/analysis/expert_cache_offload_sweep.sh.
        int n_active_experts = 0;
        for (int e = 0; e < ne; ++e)
            if (h_offsets[e + 1] > h_offsets[e])
                ++n_active_experts;
        const int dispatch_cells = n_active_experts * kExpertProjCount;
        // Asymmetric with the decode-fast path: there, the pool holding the whole
        // working set is a CORRECTNESS requirement (all 3*top_k experts staged
        // before the kernels run). Here each expert is consumed immediately after
        // staging, so eviction is harmless; this is a performance heuristic only.
        // Tested: a pool at ~79% of the working set still retains ~0% hit rate and
        // is slower than bypassing the cache, so the full-fit threshold stays exact.
        const bool use_expert_cache = expert_cache_.n_slots_ > 0 &&
                                      dispatch_cells <= expert_cache_.slots_per_layer_;

        // Host-resident NVFP4 experts, n>1: stage the WHOLE layer once instead of
        // two H2D per expert per projection (prefill activates essentially every
        // expert, so this moves the same bytes in ~6 transfers instead of ~768).
        // Decode is excluded: it wants only top_k experts, and staging all of them
        // would be strictly more traffic than the slot cache it already uses. Reuses a CUTLASS-staged layer
        // rather than transferring twice.
        if (!ctx.staged_done && n > 1)
            ctx.staged_done = stage_nvfp4_layer_(layer, stream, ctx.staged);
        const StagedProj* staged = ctx.staged;
        const bool layer_staged = ctx.staged_done;

        // Dequants one expert's weight from packed tensor into dequant scratch
        // slot 0, returning a [rows,cols] FP16 view. Slot 0 is always safe: all
        // ops share one stream, so the previous GEMM reading it completes before the next dequant writes it.
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
                // Expert weights offloaded to host - try LRU cache first, then staging
                // buffer.
                const char* host_ptr = static_cast<const char*>(packed.data) + offset;
                if (use_expert_cache) {
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

        // Tries a fused quantized GEMV for count=1 decode (dequant+dot in one
        // kernel), else falls back to dequant_expert+cuBLAS. Host-resident
        // experts: H2D to a staging buffer, then fused GEMV on staging, eliminating a separate
        // dequant_gpu+cuBLAS round trip.
        auto expert_gemm = [&](const Tensor& a, Tensor& c, const Tensor& packed, QType qtype,
                               const std::vector<Tensor>& fallback,
                               const std::vector<TensorID>& fallback_ids, int eidx,
                               ExpertProj proj) {
            // NVFP4 MoE batch cache path (Nemotron-H non-gated, or any NVFP4 MoE model
            // when batch_dequant_buf is too small): after cache_moe_native_nvfp4
            // builds the contiguous buffer and frees per-expert allocs,
            // fallback[eidx].data is nullptr, so dequant_expert can't dispatch NVFP4; slice the cached MoE
            // result instead.
            if (qtype == QType::NVFP4) {
                auto it = wcache_.nvfp4_moe.find(packed.data);
                if (it != wcache_.nvfp4_moe.end()) {
                    const auto& moe_cache = it->second;
                    size_t pkd_off = static_cast<size_t>(eidx) *
                                     moe_cache.expert_stride_packed;
                    size_t ms_off = static_cast<size_t>(eidx) *
                                    moe_cache.expert_stride_ms;
                    // tensor_scale per expert is a device array read with a sync; for prefill
                    // this fires once per active expert per layer (~9k syncs for a 200-token
                    // prompt on a 128-expert model). Pre-caching to host is a follow-up (correctness first).
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

            // Host-resident NVFP4 expert: stages it into the LRU cache's slot pool
            // first, then runs the ordinary NVFP4 GEMM against the slot. Without this,
            // the branch below hands a HOST pointer straight to gemm_nvfp4, which
            // dequantizes on device: illegal access on the full expert matrix (#1370's
            // GGUF path avoids this via dequant_expert's staging buffer; per-expert
            // NVFP4 tensors do not). Staged through the LRU pool when this dispatch
            // fits it, through the single staging buffer otherwise (same working-set
            // rule as the GGUF path); correctness is unaffected either way, only
            // whether the copies evict something. Condition is on the EXPERT tensor,
            // not `qtype`: that parameter stays empty for host-resident NVFP4 layers
            // (Phase 3 only stamps it for device-resident ones); testing it instead is
            // the mismatch that made #1384 and #1403 miss what they meant to catch.
            if (static_cast<size_t>(eidx) < fallback.size() &&
                fallback[eidx].qtype == QType::NVFP4) {
                const Tensor& w = fallback[eidx];
                if (w.data && !w.on_device && w.scales && w.ndim == 2) {
                    const auto layout = nvfp4_slot_layout(w.shape[0], w.shape[1] * 2);

                    // Whole-layer staging already put this expert on the
                    // device - read it there and skip the per-expert copies
                    // entirely.
                    const StagedProj& sp = staged[std::to_underlying(proj)];
                    if (layer_staged && sp.covers(eidx)) {
                        NvFP4QuantResult nw;
                        nw.packed_data = const_cast<char*>(sp.packed) +
                                         static_cast<size_t>(eidx) * sp.packed_stride;
                        nw.micro_scales = const_cast<char*>(sp.ms) +
                                          static_cast<size_t>(eidx) * sp.ms_stride;
                        nw.tensor_scale = w.tensor_scale;
                        nw.N = static_cast<int>(w.shape[0]);
                        nw.K = static_cast<int>(w.shape[1] * 2);
                        const int M = static_cast<int>(a.shape[0]);
                        if (M == 1) {
                            gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                                            static_cast<half*>(c.data), nw.N, nw.K, stream);
                        } else {
                            int64_t a_shape[2] = {static_cast<int64_t>(M),
                                                  static_cast<int64_t>(nw.K)};
                            int64_t c_shape[2] = {static_cast<int64_t>(M),
                                                  static_cast<int64_t>(nw.N)};
                            Tensor a_t(const_cast<void*>(static_cast<const void*>(a.data)),
                                       QType::F16, 2, a_shape, true);
                            Tensor c_t(c.data, QType::F16, 2, c_shape, true);
                            gemm_nvfp4(nw, a_t, c_t, stream);
                        }
                        return;
                    }

                    void* staged_ptr = nullptr;
                    if (layout.slot_bytes() > 0) {
                        if (use_expert_cache && nvfp4_host_pool_ready_for_staging(expert_cache_) &&
                            layout.slot_bytes() <= expert_cache_.slot_size_) {
                            ExpertCacheKey ck{fallback[0].data, eidx};
                            staged_ptr = expert_cache_.get_or_load_nvfp4(
                                layer, proj, ck, w.data, layout.packed_bytes, w.scales,
                                layout.ms_bytes, layout.packed_off(), w.tensor_scale, stream);
                        } else if (moe_.raw_staging_buf &&
                                   moe_.raw_staging_size >= layout.slot_bytes()) {
                            // Same slot layout, one buffer: packed at 0, the
                            // micro-scales behind it, so the NvFP4QuantResult
                            // below is built the same way for both routes.
                            char* st = static_cast<char*>(moe_.raw_staging_buf);
                            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(st, w.data, layout.packed_bytes,
                                                               cudaMemcpyHostToDevice, stream));
                            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(st + layout.packed_off(), w.scales,
                                                               layout.ms_bytes,
                                                               cudaMemcpyHostToDevice, stream));
                            staged_ptr = st;
                        }
                    }
                    if (staged_ptr) {
                        NvFP4QuantResult nw;
                        nw.packed_data = staged_ptr;
                        nw.micro_scales = static_cast<char*>(staged_ptr) + layout.packed_off();
                        nw.tensor_scale = w.tensor_scale;
                        nw.N = static_cast<int>(w.shape[0]);
                        nw.K = static_cast<int>(w.shape[1] * 2);
                        const int M = static_cast<int>(a.shape[0]);
                        if (M == 1) {
                            gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                                            static_cast<half*>(c.data), nw.N, nw.K, stream);
                        } else {
                            // M>1 routes through gemm_nvfp4 for the same reason the resident path
                            // does: the per-row gemv loop is wrong on multi-token expert prefill (bisected
                            // 2026-04-27, see below).
                            int64_t a_shape[2] = {static_cast<int64_t>(M),
                                                  static_cast<int64_t>(nw.K)};
                            int64_t c_shape[2] = {static_cast<int64_t>(M),
                                                  static_cast<int64_t>(nw.N)};
                            Tensor a_t(const_cast<void*>(static_cast<const void*>(a.data)),
                                       QType::F16, 2, a_shape, true);
                            Tensor c_t(c.data, QType::F16, 2, c_shape, true);
                            gemm_nvfp4(nw, a_t, c_t, stream);
                        }
                        return;
                    }
                    // IMP_CHECK, not IMP_LOG_FATAL: continuing would hand a HOST pointer to a
                    // device kernel (state corruption, not a failable request); IMP_LOG_FATAL only logs.
                    IMP_CHECK(false,
                              "MoE legacy fallback: host-resident NVFP4 expert %d on layer %d cannot be "
                              "staged by either route (needs %zu B; cache slot %zu B, staging buffer "
                              "%zu B). Continuing would hand a host pointer to a device kernel.",
                              eidx, layer, layout.slot_bytes(), expert_cache_.slot_size_,
                              moe_.raw_staging_size);
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
                // (borrowed from wcache_.nvfp4 map entry - stable address). Read directly.
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
                    // Multi-token (legacy MoE prefill): the per-row gemv_nvfp4_kpar loop
                    // produces wrong output on Gemma-4 NVFP4 experts (works for Mistral dense
                    // decode at the same kernel/dims). gemm_nvfp4 (dequant+cuBLAS), the dense-
                    // path mirror, is correct on Gemma-4 and already serves Mistral dense
                    // prefill, so multi-token expert prefill routes through it too. Bisected:
                    // M=1 gemv_kpar + M>1 gemm_nvfp4 -> correct; M>1 on gemv_kpar -> token-stuck loop.
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
                    if (use_expert_cache) {
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

                // SafeTensors NVFP4 prequant experts were promoted to qtype=NVFP4 +
                // scales/tensor_scale at engine init (Phase 0); the legacy fallback here
                // expects an FP16 weight, and cuBLAS gemm with qtype=NVFP4 crashes
                // ("unsupported dtype 71"). Route through the native NVFP4 path instead.
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
                // No buffer - serial fallback
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

            // Uses cublasGemmGroupedBatchedEx, one call for all experts: h_offsets is
            // already available from the D2H sync above, so no need for
            // gemm_moe_device_grouped (which does its own D2H sync + per-expert cublasLtMatmul calls).
            gemm_moe_batched(a_base, c_base, h_offsets.data(), b_ptrs.data(), K_dim, N_dim,
                             QType::F16, ne, stream, moe_.d_work_ptrs);
        };

        // Path selection: 1 pre-cached FP16 (all experts in fp16_packed_*_cache,
        // fastest, no dequant); 2 dequant-then-batch (packed experts on device +
        // batch buffer); 3 serial fallback (one expert at a time). Fused Q6K dp4a is handled above, before
        // the D2H sync.

        bool has_precached_up = (ly.fp16_packed_up_cache != nullptr);
        bool can_dequant_batch = (moe_.batch_dequant_buf != nullptr &&
                                  ly.expert_up_packed.data != nullptr &&
                                  ly.expert_up_packed.on_device &&
                                  dequant_gpu_supported(ly.expert_up_packed.qtype));

        if (has_precached_up) {
            // Pre-cached FP16 path - all expert packs in fp16_packed_*_cache
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

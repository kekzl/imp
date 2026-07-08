// executor_gemm_dispatch.cu — GEMM dispatch entry for the executor.
#include <utility>
//
// Extracted from executor_kernels.cu (D3, structural-debt audit): the
// WeightHandle-keyed GEMM dispatch (GraphExecutor::gemm_via_handle_) and its
// uncached dequant->cuBLAS safety-net fallback. Pulled out of the kernel
// grab-bag so the dispatch path lives on its own.
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "exec/gemm_kernel_registry.h"
#include "exec/executor.h"
#include "compute/weight_dispatch.h"
#include "core/tensor_kind.h"
#include "core/logging.h"
#include "runtime/config.h"
#include "compute/gemm.h"
#include "compute/gemm_q4k.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/ggml_mmvq.h"
#include "compute/mmq_q8_imma.h"
#include "exec/gemm_kernel_q4k_hmma.h"
#include "compute/hadamard.h"
#include "runtime/pdl.h"
#include "compute/ptx92_utils.cuh"
#include "compute/warp_reduce.cuh"  // kWarpSize

namespace imp {

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Uncached fallback: safety net for weights without a WeightHandle
// (kInvalidTensorID, budget-exhausted) and for M=1 beta!=0 residual-add.
// ---------------------------------------------------------------------------
static void gemm_dispatch_uncached_fallback(const Tensor& input, const Tensor& weight,
                                            Tensor& output, const GemmContext& ctx) {
    const auto* wc = ctx.wcache;
    const auto* qs = ctx.qscratch;
    if (!wc || !qs)
        return;

    const QType qtype = weight.qtype;

    if (ctx.beta != 0.0f) {
        auto it = wc->fp16.find(weight.data);
        if (it != wc->fp16.end()) {
            gemm(input, it->second, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (qs->dequant != nullptr && dequant_gpu_supported(qtype) && !weight.dropped_source) {
            int rows = static_cast<int>(weight.shape[0]);
            int cols = static_cast<int>(weight.shape[1]);
            dequant_gpu(weight.data, qs->dequant, qtype, rows, cols, ctx.stream);
            Tensor w_fp16(qs->dequant, QType::F16, weight.ndim, weight.shape, true);
            gemm(input, w_fp16, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (qtype == QType::F16 || qtype == QType::BF16) {
            gemm(input, weight, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        IMP_LOG_ERROR(
            "gemm_dispatch_uncached_fallback: beta=%.3f but no FP16 path for qtype=%d",
            ctx.beta, std::to_underlying(qtype));
        return;
    }

    const int M = static_cast<int>(input.shape[0]);

    // Generic dequant catch-all (M>1 prefill for uncached weights)
    if (M > 1 && input.qtype == QType::F16 && !weight.dropped_source) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.beta = ctx.beta;
        args.weight_payload = &weight;
        args.dequant_scratch = qs->dequant;
        GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }

    if (weight.dropped_source) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN(
                "gemm_dispatch_uncached_fallback: dropped weight at final fallback! "
                "M=%d qtype=%d kind=%s — overlay coverage gap.",
                M, std::to_underlying(qtype), tensor_kind_name(weight.kind));
        }
        return;
    }

    // Block-quant types (Q4_K, Q5_K, Q8_0, etc.): dequant to FP16 then cuBLAS.
    if (dequant_gpu_supported(qtype) && qs->dequant != nullptr) {
        int rows = static_cast<int>(weight.shape[0]);
        int cols = static_cast<int>(weight.shape[1]);
        dequant_gpu(weight.data, qs->dequant, qtype, rows, cols, ctx.stream);
        Tensor w_fp16(qs->dequant, QType::F16, weight.ndim, weight.shape, true);
        gemm(input, w_fp16, output, 1.0f, 0.0f, ctx.stream);
        return;
    }

    gemm(input, weight, output, 1.0f, 0.0f, ctx.stream);
}

// ---------------------------------------------------------------------------
// prefill_routes_cutlass_nvfp4_ — conservative mirror of gemm_via_handle_'s
// M>1 routing: true only when the dispatch is guaranteed to reach the
// CUTLASS NVFP4 prefill block below (which quantizes the input into the
// shared activation scratch). Every earlier-return route in gemm_via_handle_
// must answer false here.
// ---------------------------------------------------------------------------
bool GraphExecutor::prefill_routes_cutlass_nvfp4_(TensorID id) const {
    if (id == kInvalidTensorID)
        return false;
    const auto& h = registry_.handle(id);
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4)
        return false;
    StorageTier prefill =
        (h.prefill_tier != StorageTier::Undefined) ? h.prefill_tier : h.primary_tier;
    // FP16 / FP8 prefill-cache hits intercept before the CUTLASS block.
    if (prefill == StorageTier::FP16 && wcache_.fp16.count(h.source_data))
        return false;
    if (prefill == StorageTier::FP8 && wcache_.fp8.count(h.source_data))
        return false;
    // GGUF source with NVFP4 decode overlay: prefill dequants the original
    // quant — never reaches the CUTLASS block.
    if (h.source_data != nullptr && dequant_gpu_supported(h.source_qtype))
        return false;
    // The CUTLASS block itself requires the activation scratch.
    return qscratch_.cutlass_act_data != nullptr && qscratch_.cutlass_act_sf != nullptr;
}

// ---------------------------------------------------------------------------
// gemm_via_handle_ — WeightHandle dispatch for all registered weights.
// M>1 routes through weight_dispatch. M=1 beta=0 routes through gemv_dispatch
// or tier-specific handlers. M=1 beta!=0 routes through weight_dispatch
// (cuBLAS GEMM with beta). Undefined tier (budget-exhausted) reconstructs
// the weight Tensor from the handle and uses the uncached dequant fallback.
// ---------------------------------------------------------------------------
void GraphExecutor::gemm_via_handle_(TensorID id, const Tensor& input,
                                     Tensor& output, const GemmContext& ctx) {
    const auto& h = registry_.handle(id);

    if (h.primary_tier == StorageTier::Undefined) {
        Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        gemm_dispatch_uncached_fallback(input, weight, output, ctx);
        return;
    }

    int M = static_cast<int>(input.shape[0]);

    // ---- Decode (M=1, beta=0): use decode_tier for GEMV dispatch ----
    if (M == 1 && ctx.beta == 0.0f) {
        StorageTier decode = h.decode_tier;
        if (decode == StorageTier::Undefined)
            decode = h.primary_tier;

        switch (decode) {
            case StorageTier::NVFP4: {
                auto it = wcache_.nvfp4.find(h.source_data);
                if (it != wcache_.nvfp4.end()) {
                    gemv_nvfp4_kpar(it->second,
                                    reinterpret_cast<const half*>(input.data),
                                    reinterpret_cast<half*>(output.data),
                                    static_cast<int>(it->second.N),
                                    static_cast<int>(it->second.K), ctx.stream);
                    return;
                }
                break;
            }
            case StorageTier::FP8: {
                auto fp8_it = wcache_.fp8.find(h.source_data);
                if (fp8_it != wcache_.fp8.end()) {
                    int64_t wshape[2] = {h.shape[0],
                        (h.primary_tier == StorageTier::CUTLASS_NVFP4) ? h.shape[1] * 2 : h.shape[1]};
                    Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
                    gemv_fp8(fp8_w, input, output, fp8_it->second.host_scale, ctx.stream);
                    return;
                }
                break;
            }
            case StorageTier::CUTLASS_NVFP4: {
                // Native NVFP4: use source data for GEMV (micro_scales in source_scales)
                if (h.source_data && h.source_scales) {
                    NvFP4QuantResult nv;
                    nv.packed_data = const_cast<void*>(h.source_data);
                    nv.micro_scales = h.source_scales;
                    nv.tensor_scale = h.source_tensor_scale;
                    nv.N = h.shape[0];
                    nv.K = h.shape[1] * 2;
                    gemv_nvfp4_kpar(nv,
                                    reinterpret_cast<const half*>(input.data),
                                    reinterpret_cast<half*>(output.data),
                                    static_cast<int>(nv.N),
                                    static_cast<int>(nv.K), ctx.stream);
                    return;
                }
                break;
            }
            case StorageTier::MXFP4:
                imp::gemv_dispatch(h, input, output, ctx.stream);
                return;
            case StorageTier::FP16: {
                if (h.source_qtype != QType::NONE && h.source_qtype != QType::F16 &&
                    h.source_qtype != QType::BF16 && h.source_data != nullptr &&
                    output.qtype != QType::F32 &&
                    input.stride[0] == h.shape[1]) {
                    const auto* qs = ctx.qscratch;
                    if (qs) {
                        Tensor src(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
                        GemmKernelArgs args{};
                        args.input = &input;
                        args.output = &output;
                        args.stream = ctx.stream;
                        args.weight_payload = &src;
                        args.q8_1_buf = qs->q8_1_buf;
                        args.d8_buf = qs->d8_buf;
                        args.dequant_scratch = qs->dequant;
                        args.force_mmvq = ctx.force_mmvq;
                        args.no_mmvq = ctx.gemm_no_mmvq;
                        args.no_mmvq_q8_0 = ctx.gemm_no_mmvq_q8_0;
                        args.no_dp4a_gemv = ctx.gemm_no_dp4a_gemv;
                        GemmStrategy strat{StorageTier::FP16, h.source_qtype, true};
                        if (GemmKernelRegistry::instance().dispatch(strat, args) ==
                            GemmDispatchResult::Ok)
                            return;
                    }
                }
                imp::gemv_dispatch(h, input, output, ctx.stream);
                return;
            }
            default:
                imp::gemv_dispatch(h, input, output, ctx.stream);
                return;
        }
    }

    // ---- Prefill (M>1): use prefill_tier for GEMM dispatch ----
    {
        StorageTier prefill = h.prefill_tier;
        if (prefill == StorageTier::Undefined)
            prefill = h.primary_tier;

        // Q4_K HMMA GEMM: in-SMEM dequant + FP16 HMMA m16n8k16 tile kernel.
        // Config-gated (gemm.q4k_hmma_enabled, default false). Bypasses
        // dequant-to-FP16 + cuBLAS by decoding Q4_K nibbles directly in SMEM.
        if (ctx.q4k_hmma_enabled && h.source_qtype == QType::Q4_K &&
            prefill == StorageTier::FP16 && ctx.beta == 0.0f && M >= 32) {
            int N = static_cast<int>(h.shape[0]);
            int K = static_cast<int>(h.shape[1]);
            if (try_q4k_hmma_dispatch(input.data, h.source_data, output.data,
                                      M, N, K, ctx.stream))
                return;
        }

        // dp4a dense: compute directly from Q4_K/Q5_K blocks (0.55 B/elem)
        // instead of the FP16 cache (2.0 B/elem). Weight-stationary with
        // TILE_M=16 → re-reads weight ceil(M/16) times. Only wins at small
        // M where the GEMM is memory-bound (M ≤ 64). At M=512, cuBLAS FP16
        // with tensor cores + single weight read is faster.
        // sm_120 caps smem at 99 KiB — K up to ~4400 fits.
        if (prefill == StorageTier::FP16 && ctx.beta == 0.0f && M <= 64) {
            const auto* qs = ctx.qscratch;
            if (qs && qs->q8_1_prefill_buf && qs->d8_prefill_buf) {
                int N = static_cast<int>(h.shape[0]);
                int K = static_cast<int>(h.shape[1]);
                constexpr size_t kSmemLimit = 101376;
                size_t smem_needed = static_cast<size_t>(16) * (K / 32) * 36;
                size_t needed_q8 = static_cast<size_t>(M) * ((K + 31) / 32) * sizeof(block_q8_1);
                size_t needed_d8 = static_cast<size_t>(M) * ((K + 31) / 32) * sizeof(float);
                if (smem_needed <= kSmemLimit &&
                    needed_q8 <= qs->q8_1_prefill_bytes && needed_d8 <= qs->d8_prefill_bytes) {
                    if (h.source_qtype == QType::Q4_K) {
                        gemm_q4k_dp4a_dense(h.source_data,
                                            reinterpret_cast<const half*>(input.data),
                                            reinterpret_cast<half*>(output.data),
                                            qs->q8_1_prefill_buf, qs->d8_prefill_buf,
                                            M, N, K, ctx.stream);
                        return;
                    }
                    if (h.source_qtype == QType::Q5_K) {
                        gemm_q5k_dp4a_dense(h.source_data,
                                            reinterpret_cast<const half*>(input.data),
                                            reinterpret_cast<half*>(output.data),
                                            qs->q8_1_prefill_buf, qs->d8_prefill_buf,
                                            M, N, K, ctx.stream);
                        return;
                    }
                }
            }
        }

        if (prefill == StorageTier::FP16) {
            auto fp16_it = wcache_.fp16.find(h.source_data);
            if (fp16_it != wcache_.fp16.end()) {
                gemm(input, fp16_it->second, output, 1.0f, ctx.beta, ctx.stream);
                return;
            }
        }
        if (prefill == StorageTier::FP8) {
            auto fp8_it = wcache_.fp8.find(h.source_data);
            if (fp8_it != wcache_.fp8.end()) {
                int64_t wshape[2] = {h.shape[0],
                    (h.primary_tier == StorageTier::CUTLASS_NVFP4) ? h.shape[1] * 2 : h.shape[1]};
                Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
                gemm_cublaslt(input, fp8_w, output, 1.0f, ctx.beta,
                              nullptr, fp8_it->second.d_scale, ctx.stream);
                return;
            }
        }
    }

    // ---- GGUF source with an NVFP4 *decode* overlay: dequant for prefill ----
    // A weight whose primary tier is NVFP4/CUTLASS_NVFP4 but whose source is a
    // dequantable GGUF quant (Q8_0/Q6_K/Q5_K) carries the NVFP4 cache as a
    // DECODE-ONLY overlay (mode 1 additive). For prefill (M>1) we must dequant
    // the ORIGINAL Q*_K source to FP16 — running prefill on the 4-bit NVFP4
    // overlay corrupts the prompt context and degenerates output. This path is
    // reached only when no FP16/FP8 prefill cache exists (the checks above
    // return first when one does), i.e. on sm_120 where FP8 prefill is disabled
    // (PR #428). Decode (M=1, handled earlier) still uses the fast NVFP4 cache.
    // Native NVFP4 SafeTensors models are excluded: their source_qtype is
    // NVFP4/F16, which dequant_gpu_supported() rejects → they use the CUTLASS
    // path below as before.
    if (M > 1 && h.source_data != nullptr && dequant_gpu_supported(h.source_qtype) &&
        (h.primary_tier == StorageTier::CUTLASS_NVFP4 || h.primary_tier == StorageTier::NVFP4)) {
        // Q8_0 INT8 IMMA fast path (gemm.q8_imma_enabled, default off): fused
        // dequant on the int8 tensor cores instead of the materialize-to-FP16
        // → cuBLAS round-trip (the dominant Q8_0 prefill tax, see
        // docs/audit/prefill_gap_2026_06_07.md §4.1). Covers beta=0 and the
        // beta=1 residual-add form; declines (shape / capture-guard) fall
        // through to the dequant fallback below.
        // M >= 2 (was >= 64): below 64 the dequant tax dominates even harder —
        // an M=9 spec-decode verify chunk re-dequantized the ENTIRE model every
        // step (56% of GPU time, issue #667). The IMMA kernel zero-fills M-tail
        // rows, and the MoE path already runs it at per-expert M≈32.
        const bool imma_eligible = input.qtype == QType::F16 && output.qtype == QType::F16 &&
                                   M >= 2 && input.stride[0] == h.shape[1] &&
                                   output.stride[0] == h.shape[0];
        if (ctx.q8_imma_enabled && h.source_qtype == QType::Q8_0 && imma_eligible) {
            if (mmq_q8_imma_gemm(h.source_data, reinterpret_cast<const __half*>(input.data),
                                 reinterpret_cast<__half*>(output.data), M,
                                 static_cast<int>(h.shape[0]), static_cast<int>(h.shape[1]),
                                 ctx.stream, ctx.beta))
                return;
        }
        if (ctx.q4k_imma_prefill && h.source_qtype == QType::Q4_K && imma_eligible) {
            if (mmq_q4k_imma_gemm(h.source_data, reinterpret_cast<const __half*>(input.data),
                                  reinterpret_cast<__half*>(output.data), M,
                                  static_cast<int>(h.shape[0]), static_cast<int>(h.shape[1]),
                                  ctx.stream, ctx.beta))
                return;
        }
        // NOTE: dense Q6_K is deliberately NOT routed through IMMA — measured
        // 2026-06-07 on Qwen3-14B-Q6_K: 4.5k vs 6.6k pp512 for the
        // dequant→cuBLAS-fp16acc path. The half-MMA split halves the int8
        // rate, and on large dense shapes full-rate f16-acc HMMA wins; the
        // fusion saving only dominates in the MoE regime (64% dequant tax),
        // where Q6_K-IMMA ships (down_proj, see the MoE batch path).
        Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        gemm_dispatch_uncached_fallback(input, weight, output, ctx);
        return;
    }

    // ---- CUTLASS NVFP4 prefill via GemmKernelRegistry (qscratch buffers) ----
    if (M > 1 && h.primary_tier == StorageTier::CUTLASS_NVFP4) {
        const auto* qs = ctx.qscratch;
        if (qs && qs->cutlass_act_data && qs->cutlass_act_sf) {
            CutlassNvFP4Weight cw;
            cw.data = h.payload.cutlass_nvfp4.weight;
            cw.scale_factors = h.payload.cutlass_nvfp4.sf;
            cw.tensor_scale = h.payload.cutlass_nvfp4.global_scale
                                  ? *h.payload.cutlass_nvfp4.global_scale
                                  : 1.0f;
            cw.N = h.shape[0];
            cw.K = h.shape[1] * 2;
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.weight_payload = &cw;
            args.cutlass_act_data = qs->cutlass_act_data;
            args.cutlass_act_sf = qs->cutlass_act_sf;
            args.cutlass_workspace = qs->cutlass_workspace;
            args.cutlass_workspace_size = qs->cutlass_workspace_size;
            // Act-quant dedupe: a prior dispatch on this exact input already
            // quantized it into the activation scratch (QKV / gate-up share
            // one normed input — see with_act_quant_hint call sites).
            args.act_prequantized = (ctx.act_quant_hint_data != nullptr &&
                                     ctx.act_quant_hint_data == input.data &&
                                     ctx.act_quant_hint_m == M &&
                                     ctx.act_quant_hint_k == static_cast<int>(input.shape[1]));
            GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, false};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
        // FP8 fallback for CUTLASS_NVFP4 (legacy path)
        auto fp8_it = wcache_.fp8.find(h.source_data);
        if (fp8_it != wcache_.fp8.end()) {
            int64_t wshape[2] = {h.shape[0], h.shape[1] * 2};
            Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
            gemm_cublaslt(input, fp8_w, output, 1.0f, ctx.beta,
                          nullptr, fp8_it->second.d_scale, ctx.stream);
            return;
        }
    }
    imp::gemm_dispatch(nullptr, h, input, output, 1.0f, ctx.beta,
                       ws_.shared(), ws_.shared_size(), ctx.stream);
}

}  // namespace imp

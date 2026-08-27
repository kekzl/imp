// executor_gemm_dispatch.cu — GEMM dispatch entry for the executor.
#include "core/dispatch_policy.h"
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
#include "compute/activation.h"
#include "compute/layernorm.h"
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
bool GraphExecutor::prefill_routes_cutlass_nvfp4_(TensorID id, int M) const {
    if (id == kInvalidTensorID)
        return false;
    const auto& h = registry_.handle(id);
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4)
        return false;
    // #1055: small-M verify chunks divert to the batched NVFP4 GEMV overlay
    // (native branch in gemm_via_handle_) — no CUTLASS activation quant.
    if (cur_spec_verify_ && runtime_config().speculative.verify_nvfp4_gemm && M <= 4 &&
        h.source_data != nullptr && h.source_scales != nullptr)
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

    // Activation calibration ([calibration] enabled). Placed before the tier
    // switch so the statistic is the activation the weight consumes, not
    // whatever a particular tier's kernel happens to materialise.
    if (calib_)
        calib_->accumulate(cur_layer_, h.kind, input, ctx.stream);

    if (h.primary_tier == StorageTier::Undefined) {
        Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        // Preserve semantic identity + NVFP4 scale sidecars so the uncached
        // fallback can (a) diagnose which weight leaked and (b) route packed
        // NVFP4 to a real dequant instead of feeding raw bytes to cuBLAS.
        weight.kind = h.kind;
        weight.scales = h.source_scales;
        weight.tensor_scale = h.source_tensor_scale;
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
                // gemv_fp8 writes half — decline non-F16 outputs (falls back
                // to the source-tier GEMV below, matching the FP16 case).
                if (fp8_it != wcache_.fp8.end() && output.qtype == QType::F16) {
                    int64_t wshape[2] = {h.shape[0],
                        (h.primary_tier == StorageTier::CUTLASS_NVFP4) ? h.shape[1] * 2 : h.shape[1]};
                    Tensor fp8_w(fp8_it->second.weight.data, QType::FP8_E4M3, 2, wshape, true);
                    if (fp8_it->second.d_row_scales)
                        gemv_fp8_rowscale(fp8_w, input, output, fp8_it->second.d_row_scales,
                                          ctx.stream);
                    else
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

        // Spec-verify chunk (#998): read the NVFP4 decode overlay in one
        // weight pass per MR<=4 tile instead of dequantizing the source. On
        // GGUF K-quants (no direct small-M kernel, e.g. Q6_K) the per-chunk
        // dequant made a verify step cost ~7x a decode step (dequant_q6k =
        // 52% of the tg window at ctx 2048, tg −39% vs spec-off). Reading
        // the same weights as decode also aligns verify argmax with what
        // the decode path would emit. M cap = largest capture bucket (33).
        // Scoped to dequantable GGUF sources: native-ST NVFP4 weights already
        // read NVFP4 directly via the CUTLASS prefill block below (and the
        // prefill_routes_cutlass_nvfp4_ mirror above must stay in sync with
        // every earlier return here — dequantable sources answer false there).
        if (ctx.spec_verify_small_m && (ctx.beta == 0.0f || ctx.beta == 1.0f) && M <= 33 &&
            input.qtype == QType::F16 && output.qtype == QType::F16 &&
            h.source_data != nullptr && dequant_gpu_supported(h.source_qtype)) {
            auto it = ctx.wcache->nvfp4.find(h.source_data);
            if (it != ctx.wcache->nvfp4.end()) {
                // beta=1 (o/down residual add, #1055): accumulate variant —
                // previously these two fell through to the per-chunk dequant
                // path the overlay exists to avoid.
                auto& fn = (ctx.beta == 1.0f) ? gemm_nvfp4_batched_acc : gemm_nvfp4_batched;
                fn(it->second, reinterpret_cast<const half*>(input.data),
                   reinterpret_cast<half*>(output.data), static_cast<int>(h.shape[0]),
                   static_cast<int>(h.shape[1]), M, ctx.stream);
                return;
            }
        }
        // #1055: native ST-NVFP4 verify chunks. The CUTLASS prefill block
        // below serves them correctly but at ~51% of the weight-sweep
        // bandwidth for tiny M (measured 200 launches x 39-51 us per bucket-17
        // verify). The batched GEMV reads the weight once per MR=4 activation
        // tile, so it only wins in the single-tile regime — hard cap M <= 4
        // (bucket 3 + the 4-row edge); larger chunks stay on CUTLASS (5+
        // weight sweeps at M=17 would be ~3.5x worse). Same weight + linear
        // micro-scales the M=1 decode GEMV reads (source_data/source_scales),
        // but NOT the same kernel, and there is no argmax parity: decode takes
        // gemv_nvfp4_kpar (32-lane warp_k_loop K-partition) for shapes
        // 10240x5120 and 12288x5120 while the verify chunk takes
        // gemm_nvfp4_batched here, and the FFN shapes 17408x5120 / 5120x17408
        // never reach this file at decode at all, because the n==1-gated fused
        // NVFP4 kernels in executor_ffn.cu serve them there. Measured on
        // Qwen3.8-27B-NVFP4: a speculative arm does not reproduce the
        // non-speculative greedy output, see docs/LIMITATIONS.md.
        if (ctx.spec_verify_small_m && (ctx.beta == 0.0f || ctx.beta == 1.0f) && M <= 4 &&
            input.qtype == QType::F16 && output.qtype == QType::F16 &&
            h.primary_tier == StorageTier::CUTLASS_NVFP4 && h.source_data != nullptr &&
            h.source_scales != nullptr && !dequant_gpu_supported(h.source_qtype)) {
            NvFP4QuantResult nv;
            nv.packed_data = const_cast<void*>(h.source_data);
            nv.micro_scales = h.source_scales;
            nv.tensor_scale = h.source_tensor_scale;
            nv.N = h.shape[0];
            nv.K = h.shape[1] * 2;
            auto& fn = (ctx.beta == 1.0f) ? gemm_nvfp4_batched_acc : gemm_nvfp4_batched;
            fn(nv, reinterpret_cast<const half*>(input.data),
               reinterpret_cast<half*>(output.data), static_cast<int>(nv.N),
               static_cast<int>(nv.K), M, ctx.stream);
            return;
        }

        // Small-M NVFP4 GEMM (gemm.nvfp4_smallm, default ON since v2):
        // batched decode at n_seq <= 32 used to run these through the
        // CUTLASS 128x128 block-scaled tile — 40 CTAs on the N=5120 shapes,
        // 41.4 us for a 14 MB weight read (19% of the floor). impl 2 (the
        // default) is the native mxf4nvf4 producer/consumer pipeline on the
        // SAME plain weight bytes the M=1 decode GEMVs read — measured
        // +16.0% aggregate at 32 streams / +36.0% at 8 on Qwen3.8-27B-NVFP4
        // (gemm.h has the numbers); impl 1 keeps the refuted W4A16
        // dequant+HMMA kernel for A/B. Both read quantized activations
        // (same numerics family as the CUTLASS path). Spec-verify chunks
        // keep their documented paths (argmax parity, #1055).
        if (runtime_config().gemm.nvfp4_smallm && !ctx.spec_verify_small_m &&
            !overlap_prefill_active_ && M <= 32 &&
            (ctx.beta == 0.0f || ctx.beta == 1.0f) && input.qtype == QType::F16 &&
            output.qtype == QType::F16 && h.primary_tier == StorageTier::CUTLASS_NVFP4 &&
            h.source_data != nullptr && h.source_scales != nullptr &&
            !dequant_gpu_supported(h.source_qtype) && (h.shape[1] * 2) % 128 == 0) {
            const int N = static_cast<int>(h.shape[0]);
            const int K = static_cast<int>(h.shape[1] * 2);
            // impl 2 = the native mxf4nvf4 pipeline kernel (v2), impl 1 = the
            // W4A16 dequant+HMMA kernel; unaligned shapes fall back to v1.
            const bool v2 = runtime_config().gemm.nvfp4_smallm_impl == 2 && (K % 256) == 0 && (N % 64) == 0;
            const size_t need = v2 ? gemm_nvfp4_smallm_v2_workspace_bytes(N, K)
                                   : gemm_nvfp4_smallm_workspace_bytes(N);
            if (need > smallm_ws_bytes_) {
                cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(ctx.stream, &cap) != cudaSuccess)
                    cap = cudaStreamCaptureStatusActive;  // be conservative
                if (cap == cudaStreamCaptureStatusNone) {
                    if (smallm_ws_)
                        cudaFree(smallm_ws_);
                    if (cudaMalloc(&smallm_ws_, need) == cudaSuccess)
                        smallm_ws_bytes_ = need;
                    else {
                        smallm_ws_ = nullptr;
                        smallm_ws_bytes_ = 0;
                    }
                }
                // Capturing with a too-small workspace: fall through to
                // CUTLASS; the eager warmup pass sizes it for the graph run.
            }
            // A4: quantize the activation rows into the executor scratch
            // (plain layout, unit tensor scale) and read both sides packed.
            // The FP16 variant lost ~11% e2e to L2 eviction of its 327 KiB
            // x tile; packed x is ~92 KiB. Scratch sized for K_max on first
            // eager use, like the split-K workspace.
            const size_t xq_need = (size_t)32 * (K / 2) + (size_t)32 * (K / 16);
            ensure_smallm_xq_(xq_need, ctx.stream);
            if (smallm_ws_bytes_ >= need && smallm_xq_bytes_ >= xq_need) {
                uint8_t* xq_packed = static_cast<uint8_t*>(smallm_xq_);
                uint8_t* xq_scales = xq_packed + (size_t)32 * (K / 2);
                // Shared-activation skip: the call site marked this input as
                // already quantized by the PREVIOUS dispatch (act-quant hint,
                // same mechanism the CUTLASS prefill block uses), and the
                // scratch tag confirms the scratch still holds exactly that
                // quantize. Saves one quantize launch + a [M,K] FP16 read per
                // second member of a gate/up, q/k/v or GDN in/z pair.
                const bool tag_match = smallm_xq_src_ == input.data && smallm_xq_src_m_ == M &&
                                       smallm_xq_src_k_ == K;
                const bool hint_match = ctx.act_quant_hint_data != nullptr &&
                                        ctx.act_quant_hint_data == input.data &&
                                        ctx.act_quant_hint_m == M && ctx.act_quant_hint_k == K;
                // Producer fusion (fused rmsnorm/swiglu + quantize) accepts a
                // matching tag without a hint: the producer re-tags on the
                // very write that produced the FP16 buffer, so the pointer
                // cannot hold newer content than the scratch.
                const bool prequant = tag_match && (hint_match || smallm_xq_from_producer_);
                if (!prequant) {
                    quantize_fp16_to_nvfp4_into(input.data, M, K, xq_packed, xq_scales,
                                                /*tensor_scale=*/1.0f, ctx.stream);
                    smallm_xq_src_ = input.data;
                    smallm_xq_src_m_ = M;
                    smallm_xq_src_k_ = K;
                    smallm_xq_from_producer_ = false;
                }
                NvFP4QuantResult nv;
                nv.packed_data = const_cast<void*>(h.source_data);
                nv.micro_scales = h.source_scales;
                nv.tensor_scale = h.source_tensor_scale;
                nv.N = N;
                nv.K = K;
                NvFP4QuantResult xq;
                xq.packed_data = xq_packed;
                xq.micro_scales = xq_scales;
                xq.tensor_scale = 1.0f;
                xq.N = M;
                xq.K = K;
                const bool ok = v2 ? gemm_nvfp4_smallm_v2_a4(nv, xq, reinterpret_cast<half*>(output.data), M,
                                                             N, K, smallm_ws_, ctx.stream,
                                                             /*accumulate=*/ctx.beta == 1.0f)
                                   : gemm_nvfp4_smallm_a4(nv, xq, reinterpret_cast<half*>(output.data), M, N,
                                                          K, smallm_ws_, ctx.stream,
                                                          /*accumulate=*/ctx.beta == 1.0f);
                if (ok)
                    return;
            }
        }

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
            // FP16-resident or dequantable source with no cache entry — the
            // source IS the weight (e.g. one whose only cache is the
            // fp8_ssm_proj decode sidecar: F16 on native hybrids, Q8_0 on GGUF
            // hybrids). Route through the uncached fallback exactly as the
            // pre-sidecar Undefined tier did (block quants dequant→cuBLAS);
            // falling further through would hand a null payload to cuBLAS.
            if (h.source_data &&
                (h.source_qtype == QType::F16 || h.source_qtype == QType::BF16 ||
                 dequant_gpu_supported(h.source_qtype))) {
                Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
                weight.kind = h.kind;
                weight.scales = h.source_scales;
                weight.tensor_scale = h.source_tensor_scale;
                gemm_dispatch_uncached_fallback(input, weight, output, ctx);
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
        // docs/archive/prefill_gap_2026_06_07.md §4.1). Covers beta=0 and the
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

    // Native-NVFP4 prefill (M>1) safety net: dequant the packed weight to FP16
    // and run cuBLAS. Reached when the fast CUTLASS-NVFP4 path above declined
    // (e.g. kernel shape/workspace decline) and there is no FP16/FP8 prefill
    // companion — the source is native NVFP4, which dequant_gpu_supported()
    // rejects, so the GGUF-overlay dequant (above) was skipped too. Without
    // this, dispatch falls through to imp::gemm_dispatch → the generic gemm(),
    // which sees the raw packed NVFP4 payload (QType::INT8-typed bytes) and
    // hands cuBLAS an unsupported FP16×INT8 GEMM → CUBLAS_STATUS_NOT_SUPPORTED
    // (status 15) → silent repeated-token garbage + downstream IMA. This hit
    // the Qwen3.6-35B-A3B-NVFP4 shared-expert gate/up projections (the only
    // native-NVFP4 dense weights routed through gemm_via_handle_). Slower than
    // CUTLASS but correct; only fires on the decline.
    if (M > 1 && h.source_data && h.source_scales &&
        (h.primary_tier == StorageTier::CUTLASS_NVFP4 || h.primary_tier == StorageTier::NVFP4) &&
        (h.source_qtype == QType::NVFP4 || h.source_qtype == QType::INT8)) {
        NvFP4QuantResult nv;
        nv.packed_data = const_cast<void*>(h.source_data);
        nv.micro_scales = h.source_scales;
        nv.tensor_scale = h.source_tensor_scale;
        nv.N = h.shape[0];
        nv.K = h.shape[1] * 2;  // packed K/2 → logical K
        nv.owned = false;       // borrows resident weight storage
        gemm_nvfp4(nv, input, output, ctx.stream, ctx.beta);
        return;
    }

    imp::gemm_dispatch(nullptr, h, input, output, 1.0f, ctx.beta,
                       ws_.shared(), ws_.shared_size(), ctx.stream);
}

// ---------------------------------------------------------------------------
// Producer-side NVFP4 quantize fusion (batched decode).
//
// The small-M block above quantizes its FP16 input into smallm_xq_ once per
// consumer GROUP (the act-quant hint dedupes pair members). The producer
// fusion moves that quantize into the kernel that WRITES the FP16 buffer
// (fused rmsnorm / swiglu), killing the separate launch and the [M,K] FP16
// re-read. Gate mirrors the small-M route conditions; when any of them
// fails the caller runs the unfused kernels and the dispatch quantizes as
// before.
// ---------------------------------------------------------------------------
// Grow the small-M activation-quantize scratch to `xq_need` bytes. Never
// allocates while `stream` is capturing (the eager warmup pass sizes it for
// the graph run); a resize invalidates the shared-activation tag.
void GraphExecutor::ensure_smallm_xq_(size_t xq_need, cudaStream_t stream) {
    if (xq_need <= smallm_xq_bytes_)
        return;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
        cap = cudaStreamCaptureStatusActive;  // be conservative
    if (cap != cudaStreamCaptureStatusNone)
        return;
    if (smallm_xq_)
        cudaFree(smallm_xq_);
    if (cudaMalloc(&smallm_xq_, xq_need) == cudaSuccess)
        smallm_xq_bytes_ = xq_need;
    else {
        smallm_xq_ = nullptr;
        smallm_xq_bytes_ = 0;
    }
    smallm_xq_src_ = nullptr;  // fresh scratch holds nothing
    smallm_xq_from_producer_ = false;
}

uint8_t* GraphExecutor::smallm_producer_xq_(TensorID consumer_id, int M, int K, cudaStream_t stream,
                                            uint8_t** scales_out) {
    if (!runtime_config().gemm.nvfp4_smallm || cur_spec_verify_ || overlap_prefill_active_)
        return nullptr;
    if (M < 2 || M > 32 || K <= 0 || (K & 255) != 0)
        return nullptr;
    if (consumer_id == kInvalidTensorID)
        return nullptr;
    const auto& h = registry_.handle(consumer_id);
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4 || h.source_data == nullptr ||
        h.source_scales == nullptr || dequant_gpu_supported(h.source_qtype) ||
        static_cast<int>(h.shape[1] * 2) != K)
        return nullptr;
    const size_t xq_need = (size_t)32 * (K / 2) + (size_t)32 * (K / 16);
    ensure_smallm_xq_(xq_need, stream);
    if (smallm_xq_bytes_ < xq_need)
        return nullptr;
    *scales_out = static_cast<uint8_t*>(smallm_xq_) + (size_t)32 * (K / 2);
    return static_cast<uint8_t*>(smallm_xq_);
}

void GraphExecutor::smallm_producer_tag_(const void* out_data, int M, int K) {
    smallm_xq_src_ = out_data;
    smallm_xq_src_m_ = M;
    smallm_xq_src_k_ = K;
    smallm_xq_from_producer_ = true;
}

void GraphExecutor::rmsnorm_for_smallm_(const Tensor& h, const Tensor& w, Tensor& no,
                                        TensorID consumer_id, int n, float eps, cudaStream_t stream,
                                        float weight_offset) {
    const int K = static_cast<int>(h.shape[1]);
    uint8_t* xq_scales = nullptr;
    uint8_t* xq_packed = smallm_producer_xq_(consumer_id, n, K, stream, &xq_scales);
    if (xq_packed != nullptr &&
        rmsnorm_nvfp4(h, w, no, xq_packed, xq_scales, eps, stream, weight_offset)) {
        smallm_producer_tag_(no.data, n, K);
        return;
    }
    rmsnorm(h, w, no, eps, stream, weight_offset);
    // The unfused write may have replaced the content behind a still-matching
    // tag (same buffer, same shape, new values) — invalidate it.
    if (smallm_xq_src_ == no.data && smallm_xq_src_m_ == n && smallm_xq_src_k_ == K)
        smallm_xq_from_producer_ = false;
}

void GraphExecutor::swiglu_for_smallm_(const Tensor& go, const Tensor& uo, Tensor& so,
                                       TensorID consumer_id, int n, cudaStream_t stream) {
    const int K = static_cast<int>(so.shape[1]);
    uint8_t* xq_scales = nullptr;
    uint8_t* xq_packed = smallm_producer_xq_(consumer_id, n, K, stream, &xq_scales);
    if (xq_packed != nullptr && swiglu_quantize_nvfp4(go, uo, so, xq_packed, xq_scales, stream)) {
        smallm_producer_tag_(so.data, n, K);
        return;
    }
    swiglu(go, uo, so, stream);
    if (smallm_xq_src_ == so.data && smallm_xq_src_m_ == n && smallm_xq_src_k_ == K)
        smallm_xq_from_producer_ = false;
}

bool GraphExecutor::try_smallm_pair_dispatch_(TensorID id_a, TensorID id_b, const Tensor& input,
                                              Tensor& out_a, Tensor& out_b, const GemmContext& ctx) {
    // Mirror of the single-tensor smallm v2 eligibility in gemm_via_handle_
    // (see the block there for the rationale of each condition) applied to
    // BOTH weights, plus: same K, v2 only, stripes==1 shapes only, fresh
    // outputs only. Every decline is a plain `false` — the caller issues the
    // two single dispatches it would have issued anyway.
    if (!runtime_config().gemm.nvfp4_smallm || runtime_config().gemm.nvfp4_smallm_impl != 2 ||
        !runtime_config().gemm.nvfp4_smallm_pair || ctx.spec_verify_small_m ||
        overlap_prefill_active_ || ctx.beta != 0.0f || id_a == kInvalidTensorID ||
        id_b == kInvalidTensorID)
        return false;
    const int M = static_cast<int>(input.shape[0]);
    // M==1 stays on the fused decode GEMVs; M>32 is prefill.
    if (M < 2 || M > 32)
        return false;
    if (input.qtype != QType::F16 || out_a.qtype != QType::F16 || out_b.qtype != QType::F16)
        return false;
    const auto& ha = registry_.handle(id_a);
    const auto& hb = registry_.handle(id_b);
    auto eligible = [](const WeightHandle& h) {
        return h.primary_tier == StorageTier::CUTLASS_NVFP4 && h.source_data != nullptr &&
               h.source_scales != nullptr && !dequant_gpu_supported(h.source_qtype);
    };
    if (!eligible(ha) || !eligible(hb))
        return false;
    const int K = static_cast<int>(ha.shape[1] * 2);
    if (static_cast<int>(hb.shape[1] * 2) != K || (K % 256) != 0)
        return false;
    const int N1 = static_cast<int>(ha.shape[0]);
    const int N2 = static_cast<int>(hb.shape[0]);
    if ((N1 % 64) != 0 || (N2 % 64) != 0)
        return false;
    if (gemm_nvfp4_smallm_v2_stripes(N1, K) != 1 || gemm_nvfp4_smallm_v2_stripes(N2, K) != 1)
        return false;
    const size_t xq_need = (size_t)32 * (K / 2) + (size_t)32 * (K / 16);
    ensure_smallm_xq_(xq_need, ctx.stream);
    if (smallm_xq_bytes_ < xq_need)
        return false;
    // Same statistic the single path records: both weights consume `input`.
    if (calib_) {
        calib_->accumulate(cur_layer_, ha.kind, input, ctx.stream);
        calib_->accumulate(cur_layer_, hb.kind, input, ctx.stream);
    }
    uint8_t* xq_packed = static_cast<uint8_t*>(smallm_xq_);
    uint8_t* xq_scales = xq_packed + (size_t)32 * (K / 2);
    // Quantize dedupe — identical contract to the single-tensor block: a
    // matching scratch tag plus either the caller's act-quant hint or a
    // producer-side tag skips the re-quantize.
    const bool tag_match = smallm_xq_src_ == input.data && smallm_xq_src_m_ == M && smallm_xq_src_k_ == K;
    const bool hint_match = ctx.act_quant_hint_data != nullptr && ctx.act_quant_hint_data == input.data &&
                            ctx.act_quant_hint_m == M && ctx.act_quant_hint_k == K;
    if (!(tag_match && (hint_match || smallm_xq_from_producer_))) {
        quantize_fp16_to_nvfp4_into(input.data, M, K, xq_packed, xq_scales,
                                    /*tensor_scale=*/1.0f, ctx.stream);
        smallm_xq_src_ = input.data;
        smallm_xq_src_m_ = M;
        smallm_xq_src_k_ = K;
        smallm_xq_from_producer_ = false;
    }
    NvFP4QuantResult nva;
    nva.packed_data = const_cast<void*>(ha.source_data);
    nva.micro_scales = ha.source_scales;
    nva.tensor_scale = ha.source_tensor_scale;
    nva.N = N1;
    nva.K = K;
    NvFP4QuantResult nvb;
    nvb.packed_data = const_cast<void*>(hb.source_data);
    nvb.micro_scales = hb.source_scales;
    nvb.tensor_scale = hb.source_tensor_scale;
    nvb.N = N2;
    nvb.K = K;
    NvFP4QuantResult xq;
    xq.packed_data = xq_packed;
    xq.micro_scales = xq_scales;
    xq.tensor_scale = 1.0f;
    xq.N = M;
    xq.K = K;
    return gemm_nvfp4_smallm_v2_pair_a4(nva, nvb, xq, reinterpret_cast<half*>(out_a.data),
                                        reinterpret_cast<half*>(out_b.data), M, N1, N2, K, ctx.stream);
}

}  // namespace imp

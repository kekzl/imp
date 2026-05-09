#include "compute/weight_dispatch.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "quant/mxfp4_gemm.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>

namespace imp {

// ---------------------------------------------------------------------------
// gemm_dispatch — proxy entry point for prefill / multi-token GEMM.
//
// Phase-2 shim: reconstructs a per-tier descriptor from handle.payload and
// calls the existing low-level GEMM.  No consumers call this path yet (that
// is Phase 3); correctness is verified by the WeightDispatch* test suite.
//
// Activation-quantization notes
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// FP8 / CUTLASS_NVFP4 / MXFP4 require the activation to be pre-quantized
// before the weight GEMM.  In the full runtime this is done by the caller
// using its private scratch buffers (qscratch_).  In the dispatch proxy we
// carve the passed workspace into three sub-regions:
//
//   [0 .. M*K/2)            packed FP4 activation  (CUTLASS/MXFP4 only)
//   [M*K/2 .. M*K/2+sf)     SfAtom scale factors   (CUTLASS/MXFP4 only)
//   [aligned past sf ..)    cuBLAS/CUTLASS workspace
//
// For FP8, workspace bytes [0..M*K) hold the FP8 activation and bytes
// [M*K .. M*K+sizeof(float)) hold d_act_scale.  If workspace is too small
// for any of these arrangements, we fall back to gemm_nvfp4 / plain gemm.
// ---------------------------------------------------------------------------

void gemm_dispatch(cublasLtHandle_t, const WeightHandle& w, const Tensor& x, Tensor& y, float alpha,
                   float beta, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    switch (w.primary_tier) {
        // ---- FP16 -------------------------------------------------------
        case StorageTier::FP16: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            Tensor w_tensor(w.payload.fp16.data, QType::F16, 2, wshape, true);
            gemm(w_tensor, x, y, alpha, beta, stream);
            return;
        }

        // ---- FP8 E4M3 ---------------------------------------------------
        // Weight is pre-quantized FP8 E4M3 [N, K].
        // If the activation x is already FP8_E4M3, call gemm_cublaslt with
        // both FP8 operands and the weight's d_scale.
        // If x is FP16, use the FP16×FP8 mixed path (gemm_cublaslt_generic
        // handles mixed dtype via cuBLASLt algorithm selection).
        case StorageTier::FP8: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            Tensor w_tensor(w.payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
            // aScale: activation scale.  For FP8 x the caller has already
            // embedded the scale in the tensor metadata; pass nullptr and let
            // cuBLASLt use default scale=1.0.  For FP16 x no aScale is needed.
            gemm_cublaslt(x, w_tensor, y, alpha, beta,
                          /*aScale=*/nullptr, w.payload.fp8.d_scale, stream);
            return;
        }

        // ---- NVFP4 -------------------------------------------------------
        // Weight is NVFP4 (packed nibbles + FP8 E4M3 micro-scales + host
        // tensor_scale).  In the Phase-2 shim payload.nvfp4.tensor_scale is
        // nullptr (no device-side fp32 copy).  We reconstruct a temporary
        // NvFP4QuantResult using tensor_scale=1.0f as the fallback value and
        // call gemm_nvfp4 which internally dequants → FP16 GEMM for M>1.
        //
        // NOTE: if tensor_scale != 1.0 the result will be scaled incorrectly.
        // Phase 3 migration MUST ensure the handle carries a valid device ptr
        // (or store the host value in an extra field) before consumers call
        // this path.
        case StorageTier::NVFP4: {
            NvFP4QuantResult tmp;
            tmp.packed_data = w.payload.nvfp4.data;
            tmp.micro_scales = w.payload.nvfp4.block_scales;
            // tensor_scale: payload.nvfp4.tensor_scale is a HOST float pointer
            // borrowed from the wcache_.nvfp4 entry (stable address). Read it
            // directly — using cudaMemcpyDeviceToHost on a host pointer is
            // undefined and silently corrupts the scale.  This was the Phase-1
            // fix in executor_pre_dequant.cu / executor_ffn.cu / etc.; this
            // dispatch path was missed.
            tmp.tensor_scale = (w.payload.nvfp4.tensor_scale != nullptr) ? *w.payload.nvfp4.tensor_scale
                                                                         : 1.0f;
            tmp.N = w.shape[0];
            // shape[1] holds LOGICAL K — matches MXFP4 dispatch (line ~348)
            // and WeightRegistry::reserve(kind, t.shape[0], t.shape[1]) in
            // executor_pre_dequant.cu where t.shape[1] is logical K.
            tmp.K = w.shape[1];

            int M = static_cast<int>(x.shape[0]);
            // Diagnostic: IMP_NVFP4_FORCE_DEQUANT=1 routes the M=1 decode path
            // through gemm_nvfp4 (dequant→cuBLAS GEMV) instead of the native
            // gemv_nvfp4_kpar kernel. Used to bisect Mistral-Small-3.2-NVFP4
            // long-form repetition loops — if forcing dequant fixes coherence,
            // the bug is in gemv_nvfp4_kpar (numerical drift over many decode
            // steps). Mirrors the Gemma-4 MoE M>1 fallback pattern.
            static int force_dequant = -1;
            if (force_dequant < 0) {
                const char* env = std::getenv("IMP_NVFP4_FORCE_DEQUANT");
                force_dequant = (env && env[0] == '1') ? 1 : 0;
            }
            if (M == 1 && !force_dequant) {
                // GEMV path
                gemv_nvfp4_kpar(tmp, reinterpret_cast<const half*>(x.data), reinterpret_cast<half*>(y.data),
                                static_cast<int>(tmp.N), static_cast<int>(tmp.K), stream);
            } else {
                // Prefill OR forced-dequant decode: dequant + FP16 GEMM.
                gemm_nvfp4(tmp, x, y, stream);
            }
            return;
        }

        // ---- CUTLASS NVFP4 -----------------------------------------------
        // Weight is CutlassNvFP4Weight (SfAtom layout).
        // For M=1 decode: NVFP4 GEMV path (same as NVFP4 tier).
        // For M>1 prefill: quantize activation → NVFP4 CUTLASS format using
        // workspace, then call gemm_nvfp4_cutlass_sm120.  Falls back to
        // gemm_nvfp4 (dequant) if workspace is too small.
        case StorageTier::CUTLASS_NVFP4: {
            int M = static_cast<int>(x.shape[0]);
            int K = static_cast<int>(x.shape[1]);
            int N = static_cast<int>(w.shape[0]);

            // Reconstruct CutlassNvFP4Weight from handle payload (borrowed).
            CutlassNvFP4Weight cw;
            cw.data = w.payload.cutlass_nvfp4.weight;
            cw.scale_factors = w.payload.cutlass_nvfp4.sf;
            cw.tensor_scale = (w.payload.cutlass_nvfp4.global_scale != nullptr)
                                  ? *w.payload.cutlass_nvfp4.global_scale
                                  : 1.0f;
            cw.N = N;
            cw.K = K;
            cw.sf_bytes = cutlass_nvfp4_sf_size(N, K);

            if (M == 1) {
                // Decode via CUTLASS_NVFP4: the payload holds SfAtom (not per-16 FP8
                // micro_scales), so gemv_nvfp4_kpar cannot be used directly.
                // Use gemm_nvfp4_cutlass_sm120 with a M=1 quantized activation if
                // workspace is large enough; otherwise log an error.
                size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
                size_t act_sf_bytes = cutlass_nvfp4_sf_size(M, K);
                size_t ws_needed = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);
                size_t total_needed = act_data_bytes + act_sf_bytes + ws_needed;

                if (workspace != nullptr && workspace_bytes >= total_needed) {
                    uint8_t* act_data = reinterpret_cast<uint8_t*>(workspace);
                    void* act_sf = act_data + act_data_bytes;
                    void* ws_buf = reinterpret_cast<uint8_t*>(act_sf) + act_sf_bytes;
                    // SfAtom layout has padding bytes the kernel doesn't touch;
                    // workspace path can't assume the slice is pre-zeroed (unlike
                    // qscratch_.cutlass_act_sf which is zeroed at allocation).
                    cudaMemsetAsync(act_sf, 0, act_sf_bytes, stream);
                    quantize_fp16_to_nvfp4_cutlass(x.data, act_data, act_sf, M, K, stream);
                    bool ok = gemm_nvfp4_cutlass_sm120(act_data, act_sf, cw, y.data, M, N, K, ws_buf,
                                                       ws_needed, stream);
                    if (ok)
                        return;
                }
                // Fallback: dequant + cuBLAS (no NvFP4QuantResult with valid micro_scales).
                IMP_LOG_WARN(
                    "gemm_dispatch CUTLASS_NVFP4 M=1: workspace too small or CUTLASS failed, "
                    "falling back to gemm_cublaslt with FP16 weight approximation");
                // Can't dequant without NvFP4QuantResult micro_scales — log error.
                IMP_LOG_ERROR("gemm_dispatch CUTLASS_NVFP4 M=1: no valid fallback available");
                return;
            }

            // M>1 prefill: activate → NVFP4 cutlass, then GEMM.
            size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
            size_t act_sf_bytes = cutlass_nvfp4_sf_size(M, K);
            size_t ws_needed = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);
            size_t total_needed = act_data_bytes + act_sf_bytes + ws_needed;

            if (workspace != nullptr && workspace_bytes >= total_needed) {
                uint8_t* act_data = reinterpret_cast<uint8_t*>(workspace);
                void* act_sf = act_data + act_data_bytes;
                void* ws_buf = reinterpret_cast<uint8_t*>(act_sf) + act_sf_bytes;
                quantize_fp16_to_nvfp4_cutlass(x.data, act_data, act_sf, M, K, stream);
                bool ok = gemm_nvfp4_cutlass_sm120(act_data, act_sf, cw, y.data, M, N, K, ws_buf, ws_needed,
                                                   stream);
                if (ok)
                    return;
                // CUTLASS failed; fall through to NvFP4 fallback.
            }

            // Fallback: gemm_nvfp4 (internal dequant + cuBLAS GEMM).
            // Build NvFP4QuantResult — but CUTLASS_NVFP4 payload doesn't carry
            // the original NvFP4 micro_scales.  We can't safely dequant here.
            // Log and return.
            IMP_LOG_ERROR(
                "gemm_dispatch CUTLASS_NVFP4: workspace too small (need %zu, have %zu) "
                "and no dequant fallback (micro_scales unavailable in payload)",
                total_needed, workspace_bytes);
            return;
        }

        // ---- MXFP4 -------------------------------------------------------
        // Weight is CutlassMxFP4Weight (UE8M0 SfAtom scales + linear_scales).
        // For M=1 decode: gemv_mxfp4_kpar (uses linear_scales).
        // For M>1 prefill: quantize activation → MXFP4 cutlass, then GEMM.
        // Falls back to NVFP4 dequant path if workspace is insufficient.
        case StorageTier::MXFP4: {
            int M = static_cast<int>(x.shape[0]);
            int K = static_cast<int>(x.shape[1]);
            int N = static_cast<int>(w.shape[0]);

            // Reconstruct CutlassMxFP4Weight from handle payload (borrowed).
            CutlassMxFP4Weight mw;
            mw.data = w.payload.mxfp4.weight;
            mw.scale_factors = w.payload.mxfp4.scales;
            mw.linear_scales = w.payload.mxfp4.linear_scales;
            mw.tensor_scale = 1.0f;  // absorbed into UE8M0 scales
            mw.N = N;
            mw.K = K;
            mw.sf_bytes = cutlass_mxfp4_sf_size(N, K);
            mw.owns_data = false;
            mw.hadamard_bs = w.payload.mxfp4.hadamard_bs;

            if (M == 1) {
                // Decode: MXFP4 GEMV using linear_scales.
                if (mw.linear_scales != nullptr) {
                    gemv_mxfp4_kpar(mw, reinterpret_cast<const half*>(x.data),
                                    reinterpret_cast<half*>(y.data), N, K, stream);
                } else {
                    IMP_LOG_ERROR("gemm_dispatch MXFP4 M=1: linear_scales is null, cannot GEMV");
                }
                return;
            }

            // M>1 prefill: quantize activation → MXFP4, then CUTLASS GEMM.
            if (K % 32 != 0) {
                IMP_LOG_ERROR(
                    "gemm_dispatch MXFP4: K=%d is not a multiple of 32, "
                    "MXFP4 GEMM requires K%%32==0",
                    K);
                return;
            }

            size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
            size_t act_sf_bytes = cutlass_mxfp4_sf_size(M, K);
            size_t ws_needed = gemm_mxfp4_cutlass_sm120_workspace(M, N, K);
            size_t total_needed = act_data_bytes + act_sf_bytes + ws_needed;

            if (workspace != nullptr && workspace_bytes >= total_needed) {
                uint8_t* act_data = reinterpret_cast<uint8_t*>(workspace);
                void* act_sf = act_data + act_data_bytes;
                void* ws_buf = reinterpret_cast<uint8_t*>(act_sf) + act_sf_bytes;
                quantize_fp16_to_mxfp4_cutlass(x.data, act_data, act_sf, M, K, stream);
                bool ok = gemm_mxfp4_cutlass_sm120(act_data, act_sf, mw, y.data, M, N, K, ws_buf, ws_needed,
                                                   stream);
                if (ok)
                    return;
                // CUTLASS failed; fall through.
            }

            IMP_LOG_ERROR(
                "gemm_dispatch MXFP4: workspace too small (need %zu, have %zu) "
                "or CUTLASS GEMM failed",
                total_needed, workspace_bytes);
            return;
        }

        case StorageTier::FP32:
        case StorageTier::Undefined:
            IMP_LOG_FATAL("gemm_dispatch: handle in invalid tier %d", static_cast<int>(w.primary_tier));
            return;
    }
}

// ---------------------------------------------------------------------------
// gemv_dispatch — proxy entry point for decode (batch=1) GEMV.
// ---------------------------------------------------------------------------

void gemv_dispatch(const WeightHandle& w, const Tensor& x, Tensor& y, cudaStream_t stream) {
    switch (w.primary_tier) {
        // ---- FP16 -------------------------------------------------------
        case StorageTier::FP16: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            Tensor w_tensor(w.payload.fp16.data, QType::F16, 2, wshape, true);
            gemm(w_tensor, x, y, 1.0f, 0.0f, stream);
            return;
        }

        // ---- FP8 --------------------------------------------------------
        // Single-token (M=1): use gemv_fp8 (FP8 weight × FP16 activation).
        // For M>1: gemm_cublaslt with FP8 weight + FP16 activation.
        case StorageTier::FP8: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            float host_scale = 1.0f;
            if (w.payload.fp8.d_scale != nullptr) {
                cudaMemcpyAsync(&host_scale, w.payload.fp8.d_scale, sizeof(float), cudaMemcpyDeviceToHost,
                                stream);
                cudaStreamSynchronize(stream);
            }
            Tensor w_tensor(w.payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
            gemv_fp8(w_tensor, x, y, host_scale, stream);
            return;
        }

        // ---- NVFP4 -------------------------------------------------------
        // Decode GEMV: reconstruct NvFP4QuantResult and call kpar GEMV.
        case StorageTier::NVFP4: {
            NvFP4QuantResult tmp;
            tmp.packed_data = w.payload.nvfp4.data;
            tmp.micro_scales = w.payload.nvfp4.block_scales;
            // tensor_scale: HOST pointer borrowed from wcache_.nvfp4 — read
            // directly. cudaMemcpyDeviceToHost on a host pointer is undefined.
            tmp.tensor_scale = (w.payload.nvfp4.tensor_scale != nullptr) ? *w.payload.nvfp4.tensor_scale
                                                                         : 1.0f;
            tmp.N = w.shape[0];
            // Logical K — matches MXFP4 dispatch + WeightRegistry::reserve.
            tmp.K = w.shape[1];
            gemv_nvfp4_kpar(tmp, reinterpret_cast<const half*>(x.data), reinterpret_cast<half*>(y.data),
                            static_cast<int>(tmp.N), static_cast<int>(tmp.K), stream);
            return;
        }

        // ---- CUTLASS_NVFP4 -----------------------------------------------
        // Decode path: CUTLASS_NVFP4 payload does not carry NvFP4 micro_scales
        // in the phase-2 shim (only SfAtom layout is stored).  We cannot
        // directly call gemv_nvfp4_kpar (which needs per-16 FP8 micro_scales).
        //
        // Fallback: call gemv_fp8 using the FP4-packed data interpreted as
        // FP8_E4M3 (wrong dtype but same pointer width) — this is NOT
        // numerically correct and should not be used in production until
        // Phase 3 migrates this path to carry the correct NvFP4QuantResult.
        //
        // For now: log an error and return (stub behavior for Phase 2).
        case StorageTier::CUTLASS_NVFP4: {
            // CUTLASS_NVFP4 is a prefill tier (M>1).  Decode falls through
            // to NVFP4 GEMV in the consumer (executor_kernels.cu line 1951).
            // gemv_dispatch is only called for decode (M=1); in phase-2 the
            // consumer still uses the wcache_ NVFP4 entry directly.
            // Stub: log error and do nothing so tests can verify routing.
            IMP_LOG_ERROR(
                "gemv_dispatch CUTLASS_NVFP4: not directly callable for decode "
                "(no FP8 micro_scales in payload); consumer should use NVFP4 tier");
            return;
        }

        // ---- MXFP4 -------------------------------------------------------
        // Decode GEMV: reconstruct CutlassMxFP4Weight and call kpar GEMV.
        case StorageTier::MXFP4: {
            CutlassMxFP4Weight mw;
            mw.data = w.payload.mxfp4.weight;
            mw.scale_factors = w.payload.mxfp4.scales;
            mw.linear_scales = w.payload.mxfp4.linear_scales;
            mw.tensor_scale = 1.0f;
            mw.N = w.shape[0];
            mw.K = w.shape[1];
            mw.sf_bytes = cutlass_mxfp4_sf_size(static_cast<int>(w.shape[0]), static_cast<int>(w.shape[1]));
            mw.owns_data = false;
            mw.hadamard_bs = w.payload.mxfp4.hadamard_bs;

            if (mw.linear_scales == nullptr) {
                IMP_LOG_ERROR("gemv_dispatch MXFP4: linear_scales is null");
                return;
            }
            gemv_mxfp4_kpar(mw, reinterpret_cast<const half*>(x.data), reinterpret_cast<half*>(y.data),
                            static_cast<int>(w.shape[0]), static_cast<int>(w.shape[1]), stream);
            return;
        }

        default:
            IMP_LOG_FATAL("gemv_dispatch: handle in invalid tier %d", static_cast<int>(w.primary_tier));
            return;
    }
}

// ---------------------------------------------------------------------------
// gemm_grouped_dispatch — MoE grouped GEMM proxy.
//
// Supported tiers (Task 3.4):
//   FP16: Reconstruct Tensor views from handles, call gemm_moe_batched.
//         expert_counts[e] is the token count for expert e; offsets are
//         computed here as a prefix-sum.
//
// Unsupported tiers (FATAL — MoE models that reach these use specialised
// paths in executor_forward_moe.cu that call lower-level helpers directly):
//   FP8, NVFP4, CUTLASS_NVFP4, MXFP4
// ---------------------------------------------------------------------------
void gemm_grouped_dispatch(cublasLtHandle_t /*lt*/, std::span<const WeightHandle* const> experts,
                           const Tensor& x_flat, Tensor& y_flat, const int* expert_counts,
                           void* /*workspace*/, size_t /*workspace_bytes*/, cudaStream_t stream) {
    if (experts.empty())
        return;
    const int ne = static_cast<int>(experts.size());

    // Validate: all handles must have the same primary_tier.
    StorageTier tier = experts[0]->primary_tier;
    for (int e = 1; e < ne; ++e) {
        if (experts[e]->primary_tier != tier) {
            IMP_LOG_FATAL("gemm_grouped_dispatch: mixed tiers in expert set (%d vs %d)",
                          static_cast<int>(experts[0]->primary_tier),
                          static_cast<int>(experts[e]->primary_tier));
            return;
        }
    }

    switch (tier) {
        case StorageTier::FP16: {
            // Build prefix-sum offsets from expert_counts.
            std::vector<int32_t> offsets(ne + 1, 0);
            for (int e = 0; e < ne; ++e)
                offsets[e + 1] = offsets[e] + expert_counts[e];

            int K = static_cast<int>(x_flat.shape[1]);
            int N = static_cast<int>(y_flat.shape[1]);

            // Build per-expert weight pointer array.
            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e) {
                b_ptrs[e] = experts[e]->payload.fp16.data;
            }

            gemm_moe_batched(x_flat.data, y_flat.data, offsets.data(), b_ptrs.data(), K, N, QType::F16, ne,
                             stream,
                             /*d_work_ptrs=*/nullptr);
            return;
        }

        case StorageTier::FP8:
            IMP_LOG_FATAL(
                "gemm_grouped_dispatch: FP8 tier not implemented "
                "(executor_forward_moe uses specialised FP8 batch path)");
            return;

        case StorageTier::NVFP4:
            IMP_LOG_FATAL(
                "gemm_grouped_dispatch: NVFP4 tier not implemented "
                "(executor_forward_moe uses gemv_nvfp4_moe_* directly)");
            return;

        case StorageTier::CUTLASS_NVFP4:
            IMP_LOG_FATAL(
                "gemm_grouped_dispatch: CUTLASS_NVFP4 tier not implemented "
                "(executor_forward_moe uses gemm_grouped_cutlass_3x_nvfp4 directly)");
            return;

        case StorageTier::MXFP4:
            IMP_LOG_FATAL(
                "gemm_grouped_dispatch: MXFP4 tier not implemented "
                "(no MXFP4 MoE path in current runtime)");
            return;

        default:
            IMP_LOG_FATAL("gemm_grouped_dispatch: undefined tier %d", static_cast<int>(tier));
            return;
    }
}

}  // namespace imp

#include "compute/weight_dispatch.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "quant/mxfp4_gemm.h"
#include "core/logging.h"
#include "core/process_diag.h"

#include <cstdio>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <utility>

namespace imp {

// gemm_dispatch: proxy entry point for prefill/multi-token GEMM. Phase-2 shim
// reconstructing a per-tier descriptor from handle.payload; no consumers yet (Phase 3).
// FP8/CUTLASS_NVFP4/MXFP4 need the activation pre-quantized before the weight GEMM. The
// dispatch proxy carves the workspace into: [0..M*K/2) packed FP4 activation
// (CUTLASS/MXFP4); [M*K/2..+sf) SfAtom scales (CUTLASS/MXFP4); [aligned past sf..)
// cuBLAS/CUTLASS workspace. FP8: [0..M*K) FP8 activation, [M*K..+sizeof(float)) d_act_scale.
// Too-small workspace falls back to gemm_nvfp4 / plain gemm.

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

        // FP8 E4M3: weight is pre-quantized [N,K]. x already FP8_E4M3: gemm_cublaslt with both
        // FP8 operands + the weight's d_scale. x FP16: mixed FP16xFP8 path (gemm_cublaslt_generic
        // via cuBLASLt algorithm selection).
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

        // NVFP4: weight is packed nibbles + FP8 E4M3 micro-scales + host tensor_scale. Phase-2
        // shim: payload.nvfp4.tensor_scale is null (no device fp32 copy), so a temporary
        // NvFP4QuantResult is built with tensor_scale=1.0 fallback; gemm_nvfp4 dequants -> FP16
        // GEMM for M>1.
        // NOTE: wrong result if tensor_scale != 1.0. Phase 3 must carry a valid device ptr (or the
        // host value) before consumers use this path.
        case StorageTier::NVFP4: {
            NvFP4QuantResult tmp;
            tmp.packed_data = w.payload.nvfp4.data;
            tmp.micro_scales = w.payload.nvfp4.block_scales;
            // tensor_scale: payload.nvfp4.tensor_scale is a HOST float pointer (stable address from
            // wcache_.nvfp4). Read directly - cudaMemcpyDeviceToHost on a host pointer is undefined
            // and silently corrupts the scale. This was the Phase-1 fix in executor_pre_dequant.cu /
            // executor_ffn.cu / etc.; this dispatch path was missed.
            tmp.tensor_scale = (w.payload.nvfp4.tensor_scale != nullptr) ? *w.payload.nvfp4.tensor_scale
                                                                         : 1.0f;
            tmp.N = w.shape[0];
            // Logical K = the activation's K (x is [M,K], x.shape[1] IS logical K). Deriving K from
            // the weight handle is unsafe: shape[1] is PACKED (K/2) for prequant-loaded NVFP4 but
            // LOGICAL for handles built elsewhere - an inconsistent convention. The old `=w.shape[1]`
            // took the packed value, aborting the M>1 dequant->cuBLAS fallback with "B.shape[1]=<2K>
            // must equal weight K=<K>" whenever prefill reached this shim with a prequant weight.
            tmp.K = static_cast<int>(x.shape[1]);

            int M = static_cast<int>(x.shape[0]);
            // diagnostics.nvfp4_force_dequant routes the M=1 decode path through gemm_nvfp4
            // (dequant->cuBLAS GEMV) instead of the native gemv_nvfp4_kpar kernel. Used to bisect
            // Mistral-Small-3.2-NVFP4 long-form repetition loops: if forcing dequant fixes coherence,
            // the bug is numerical drift in gemv_nvfp4_kpar. Mirrors the Gemma-4 MoE M>1 fallback.
            const bool force_dequant = imp::process_diag_nvfp4_force_dequant();
            if (M == 1 && !force_dequant && beta == 0.0f) {
                // GEMV path (no beta — gemv_nvfp4_kpar overwrites output)
                gemv_nvfp4_kpar(tmp, reinterpret_cast<const half*>(x.data), reinterpret_cast<half*>(y.data),
                                static_cast<int>(tmp.N), static_cast<int>(tmp.K), stream);
            } else {
                // Prefill, forced-dequant, or beta!=0: dequant + FP16 GEMM.
                gemm_nvfp4(tmp, x, y, stream, beta);
            }
            return;
        }

        // CUTLASS NVFP4: weight is CutlassNvFP4Weight (SfAtom layout). M=1 decode: same NVFP4 GEMV
        // path. M>1 prefill: quantize activation to NVFP4 CUTLASS format via workspace, call
        // gemm_nvfp4_cutlass_sm120; falls back to gemm_nvfp4 (dequant) if workspace too small.
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
                // Decode via CUTLASS_NVFP4: payload holds SfAtom (not per-16 FP8 micro_scales), so
                // gemv_nvfp4_kpar cannot be used directly. Uses gemm_nvfp4_cutlass_sm120 with an M=1
                // quantized activation if workspace is large enough; else logs an error.
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
                IMP_LOG_ERROR(
                    "gemm_dispatch CUTLASS_NVFP4: GEMM failed (M=%d N=%d K=%d), "
                    "no dequant fallback (micro_scales unavailable in payload)",
                    M, N, K);
                return;
            }

            IMP_LOG_ERROR(
                "gemm_dispatch CUTLASS_NVFP4: workspace too small (need %zu, have %zu) "
                "and no dequant fallback (micro_scales unavailable in payload)",
                total_needed, workspace_bytes);
            return;
        }

        // MXFP4: weight is CutlassMxFP4Weight (UE8M0 SfAtom scales + linear_scales). M=1 decode:
        // gemv_mxfp4_kpar (linear_scales). M>1 prefill: quantize activation to MXFP4 cutlass, then
        // GEMM. Falls back to the NVFP4 dequant path if workspace is insufficient.
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
        case StorageTier::Undefined: {
            // Was IMP_LOG_FATAL + return, leaving output holding whatever it held: IMP_LOG_FATAL does
            // not abort (only IMP_CHECK does), so "FATAL" bought a log line and nothing else. Throws
            // instead (same reason as the CUTLASS_NVFP4 case, #1508): no tier accepted is an error,
            // not a degraded answer (SETTLED.md S-22); imp_api.cpp turns it into ImpError.
            char msg[128];
            snprintf(msg, sizeof(msg), "gemm_dispatch: handle in invalid tier %d",
                     std::to_underlying(w.primary_tier));
            throw std::runtime_error(msg);
        }
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

        // Decode CUTLASS_NVFP4: phase-2 shim payload carries only SfAtom layout, not per-16 FP8
        // micro_scales, so gemv_nvfp4_kpar cannot be called directly.
        // Fallback: gemv_fp8 on the FP4-packed data reinterpreted as FP8_E4M3 (wrong dtype, same
        // pointer width) - NOT numerically correct, not for production until Phase 3 carries the
        // real NvFP4QuantResult. For now: log an error and return (stub, Phase 2).
        case StorageTier::CUTLASS_NVFP4: {
            // Unreachable today, and must stay an error rather than a silent return: CUTLASS_NVFP4 is
            // a prefill tier (M>1); decode reaches the NVFP4 GEMV through the consumer, and
            // decode_tier is only ever assigned tier/FP8/NVFP4
            // (pre_dequant_phase4_tensor_registry.cu:90,96,98,100), so no caller routes here.
            // Before 2026-08-21 this branch logged and returned WITHOUT WRITING `y` (the shape #654
            // removed from attention_prefill_dispatch; SETTLED.md S-22: "no tier accepted" is an
            // error, not a degraded answer). An unreachable branch is exactly where a silent-wrong-
            // output path survives, since nothing exercises it to prove otherwise.
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "gemv_dispatch: CUTLASS_NVFP4 is not directly callable for decode "
                     "(no FP8 micro_scales in the payload); the consumer must use the "
                     "NVFP4 tier. shape=[%lld,%lld]",
                     static_cast<long long>(w.shape[0]), static_cast<long long>(w.shape[1]));
            throw std::runtime_error(msg);
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

        default: {
            // Same class as the branch above, worse structurally: this is `default:`, the branch
            // catching a tier nobody anticipated - the case where continuing with an unwritten output
            // is least affordable.
            char msg[128];
            snprintf(msg, sizeof(msg), "gemv_dispatch: handle in invalid tier %d",
                     std::to_underlying(w.primary_tier));
            throw std::runtime_error(msg);
        }
    }
}

}  // namespace imp

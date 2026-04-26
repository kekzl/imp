#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_gemv_helpers.h"
#include "graph/executor_helpers.h"
#include "graph/gemm_context.h"
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/activation.h"
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

// dispatch_gemv_residual: from executor_gemv_helpers.h

// ---------------------------------------------------------------------------
// FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for dense FFN phase
    configure_ffn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    // cur_n_tokens_ is set by forward_logits before the layer loop.
    int n   = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;

    Tensor h  = view_tokens(hidden_,     n);
    Tensor r  = view_tokens(residual_,   n);
    Tensor no = view_tokens(norm_out_,   n);
    Tensor go = view_tokens(gate_out_,   n);
    Tensor uo = view_tokens(up_out_,     n);
    Tensor so = view_tokens(swiglu_out_, n);
    Tensor fo = view_tokens(ffn_out_,    n);

    // L2 streaming hint: deprioritize FFN weight data in L2 to preserve cached
    // activations and KV data from prior layers.  The access policy window covers
    // the address span of the three weight tensors (gate, up, down).
    {
        const char* lo = static_cast<const char*>(ly.w_gate.data);
        const char* hi = lo;
        auto update = [&](const void* p, size_t sz) {
            if (!p) return;
            auto* cp = static_cast<const char*>(p);
            if (cp < lo) lo = cp;
            if (cp + sz > hi) hi = cp + sz;
        };
        update(ly.w_gate.data, ly.w_gate.nbytes());
        update(ly.w_up.data,   ly.w_up.nbytes());
        update(ly.w_down.data, ly.w_down.nbytes());
        if (lo < hi)
            set_l2_streaming(stream, lo, static_cast<size_t>(hi - lo));
    }

    // 1. Save residual (skip if fused down-proj+residual will handle it).
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    // Qwen3.5: uses post_attn_norm instead of ffn_norm (ffn_norm is null)
    const Tensor& ffn_norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm :
                                (ly.post_attn_norm.data != nullptr) ? ly.post_attn_norm : ly.attn_norm;
    const bool has_post_ffn_norm = (ly.post_ffn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_ffn_norm);

    // Fetch handles for w_down (used in multiple fuse-flag decisions below).
    const WeightHandle* hwd = (ly.w_down_id != kInvalidTensorID)
                               ? &registry_.handle(ly.w_down_id) : nullptr;
    const StorageTier wd_tier = hwd ? hwd->primary_tier : StorageTier::Undefined;

    bool will_fuse_down_mxfp4 = (!has_post_ffn_norm && n == 1 && h.dtype == DType::FP16 &&
                                  wd_tier == StorageTier::MXFP4 &&
                                  hwd->payload.mxfp4.linear_scales != nullptr);
    bool will_fuse_down_nvfp4 = (!has_post_ffn_norm && !will_fuse_down_mxfp4 &&
                                  n == 1 && h.dtype == DType::FP16 &&
                                  wd_tier == StorageTier::NVFP4);
    bool will_fuse_down_residual = (!has_post_ffn_norm && !will_fuse_down_nvfp4 &&
                                     n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                                     h.dtype == DType::FP16 && is_dp4a_qtype(ly.w_down_qtype));
    bool will_fuse_down_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                  !will_fuse_down_nvfp4 && n > 1 &&
                                  (wd_tier == StorageTier::FP16 || wd_tier == StorageTier::FP8));
    bool will_fuse_down_dequant_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                          !will_fuse_down_nvfp4 &&
                                          !will_fuse_down_beta1 && n > 1 &&
                                          qscratch_.dequant != nullptr &&
                                          dequant_gpu_supported(ly.w_down_qtype));
    if (!will_fuse_down_residual && !will_fuse_down_beta1 &&
        !will_fuse_down_dequant_beta1 && !will_fuse_down_nvfp4 && !will_fuse_down_mxfp4 &&
        !using_fp32_accum) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream));
    }

    // GemmContext for all weight GEMM dispatches in this function.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, cur_force_fp16_);

    // 3. Gate and Up projections
    //    For decode (n=1): fuse RMSNorm→Q8_1→GEMV to avoid redundant quantization.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        int d = static_cast<int>(h.shape[1]);
        // Fetch handles for w_gate and w_up
        const WeightHandle* hwg = (ly.w_gate_id != kInvalidTensorID)
                                   ? &registry_.handle(ly.w_gate_id) : nullptr;
        const WeightHandle* hwu = (ly.w_up_id != kInvalidTensorID)
                                   ? &registry_.handle(ly.w_up_id) : nullptr;

        // MXFP4 gate+up decode path
        bool mxfp4_ffn = (n == 1 &&
                          hwg && hwg->primary_tier == StorageTier::MXFP4 && hwg->payload.mxfp4.linear_scales &&
                          hwu && hwu->primary_tier == StorageTier::MXFP4 && hwu->payload.mxfp4.linear_scales);
        // NVFP4 gate+up decode path
        bool nvfp4_ffn = (n == 1 &&
                          hwg && hwg->primary_tier == StorageTier::NVFP4 &&
                          hwu && hwu->primary_tier == StorageTier::NVFP4);
        bool fused_ffn_norm = (n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                               h.dtype == DType::FP16 && is_dp4a_qtype(ly.w_gate_qtype));
        if (mxfp4_ffn) {
            // MXFP4 gate+up: RMSNorm, optional Hadamard, then MXFP4 fused GEMV
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            int d = static_cast<int>(h.shape[1]);
            int hbs = hwg->payload.mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no.data),
                                        static_cast<half*>(no.data), 1, d, hbs, stream);
            // Reconstruct CutlassMxFP4Weight structs from handle payloads.
            auto make_mxfp4 = [](const WeightHandle* hw) {
                CutlassMxFP4Weight mw{};
                mw.data          = hw->payload.mxfp4.weight;
                mw.scale_factors = hw->payload.mxfp4.scales;
                mw.linear_scales = hw->payload.mxfp4.linear_scales;
                mw.hadamard_bs   = hw->payload.mxfp4.hadamard_bs;
                return mw;
            };
            auto mw_g = make_mxfp4(hwg);
            auto mw_u = make_mxfp4(hwu);
            gemv_mxfp4_gate_up_fused(mw_g, mw_u,
                                      static_cast<const half*>(no.data),
                                      static_cast<half*>(go.data),
                                      static_cast<half*>(uo.data),
                                      ffn_rows, d, stream);
        } else if (nvfp4_ffn) {
            // NVFP4 gate+up: RMSNorm to FP16, then NVFP4 fused GEMV
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            // Reconstruct NvFP4QuantResult structs from handle payloads.
            auto make_nvfp4 = [](const WeightHandle* hw) {
                NvFP4QuantResult tmp;
                tmp.packed_data  = hw->payload.nvfp4.data;
                tmp.micro_scales = hw->payload.nvfp4.block_scales;
                // tensor_scale: host float pointer borrowed from wcache_.nvfp4 map.
                tmp.tensor_scale = (hw->payload.nvfp4.tensor_scale != nullptr)
                                   ? *hw->payload.nvfp4.tensor_scale : 1.0f;
                tmp.N = static_cast<int>(hw->shape[0]);
                tmp.K = static_cast<int>(hw->shape[1]) * 2;  // packed → logical K
                return tmp;
            };
            auto nv_g = make_nvfp4(hwg);
            auto nv_u = make_nvfp4(hwu);
            gemv_nvfp4_gate_up_fused(nv_g, nv_u,
                                      static_cast<const half*>(no.data),
                                      static_cast<half*>(go.data),
                                      static_cast<half*>(uo.data),
                                      ffn_rows, d, stream);
        } else if (fused_ffn_norm) {
            // Fused RMSNorm + Q8_1: quantize once, use for both gate and up
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                    static_cast<const half*>(ffn_norm_w.data),
                                    q8, qscratch_.d8_buf, static_cast<half*>(no.data),
                                    d, eps, stream, norm_w_off_);
            // Fused gate+up GEMV: single kernel launch for both projections
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            gemv_gate_up_fused(ly.w_gate.data, ly.w_up.data, q8, qscratch_.d8_buf,
                                static_cast<half*>(go.data),
                                static_cast<half*>(uo.data),
                                ffn_rows, d, ly.w_gate_qtype, stream);
        } else {
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);

            // FP8 prefill path: quantize norm_out→FP8 once, 2 separate FP8 GEMMs
            bool fp8_ffn = (n > 1 && !cur_force_fp16_ &&
                            hwg && hwg->primary_tier == StorageTier::FP8 &&
                            hwu && hwu->primary_tier == StorageTier::FP8 &&
                            qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr);
            if (fp8_ffn) {
                // Reconstruct FP8 weight tensors from handle payloads.
                auto make_fp8_tensor = [](const WeightHandle* hw) {
                    int64_t wshape[2] = {hw->shape[0], hw->shape[1]};
                    return Tensor(hw->payload.fp8.data, DType::FP8_E4M3, 2, wshape, true);
                };
                Tensor fp8_no(qscratch_.fp8_act, DType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                Tensor fp8_tg = make_fp8_tensor(hwg);
                Tensor fp8_tu = make_fp8_tensor(hwu);
                gemm_cublaslt(fp8_no, fp8_tg, go, 1.0f, 0.0f,
                              qscratch_.d_act_scale, hwg->payload.fp8.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_tu, uo, 1.0f, 0.0f,
                              qscratch_.d_act_scale, hwu->payload.fp8.d_scale, stream);
            } else {
                // Read fused gate+up via WeightRegistry handle — the wcache_
                // map is no longer the lookup mechanism (it remains the
                // storage owner; cleanup happens via wcache_.clear()).
                const Tensor* fused_gu = nullptr;
                Tensor fused_from_handle;
                if (ly.fused_gate_up_id != kInvalidTensorID) {
                    const auto& h = registry_.handle(ly.fused_gate_up_id);
                    if (h.payload.fp16.data) {
                        fused_from_handle = Tensor(h.payload.fp16.data, DType::FP16,
                                                   2, h.shape, true);
                        fused_gu = &fused_from_handle;
                    }
                }
                if (n > 1 && fused_gu) {
                    // Batched gate+up: single cuBLAS call for both projections
                    gemm_pair_batched(no, *fused_gu, go, uo, stream);
                } else {
                    gemm_dispatch(no, ly.w_gate, ly.w_gate_qtype, go, ctx);
                    gemm_dispatch(no, ly.w_up,   ly.w_up_qtype,   uo, ctx);
                }
            }
        }
    }

    // 4+5+6. Gated activation + Down projection + residual add.
    //    For decode (n=1) with dp4a: fuse activation→Q8_1→GEMV+residual.
    //    SwiGLU case: swiglu_quantize_q8_1 fuses activation + Q8_1 in one kernel,
    //    eliminating the intermediate FP16 buffer write and one kernel launch.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        bool fused_down_residual = (!has_post_ffn_norm &&
                                     n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                                     so.dtype == DType::FP16 && is_dp4a_qtype(ly.w_down_qtype));
        if (will_fuse_down_mxfp4) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            // Reconstruct CutlassMxFP4Weight from handle payload.
            CutlassMxFP4Weight wd_mxfp4{};
            wd_mxfp4.data          = hwd->payload.mxfp4.weight;
            wd_mxfp4.scale_factors = hwd->payload.mxfp4.scales;
            wd_mxfp4.linear_scales = hwd->payload.mxfp4.linear_scales;
            wd_mxfp4.hadamard_bs   = hwd->payload.mxfp4.hadamard_bs;
            // MXFP4 fused SwiGLU/GeGLU + GEMV + residual
            int hbs = wd_mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs)) {
                // Hadamard: can't fuse SwiGLU with GEMV — need rotation between them
                if (cfg.ffn_activation != FFNActivation::GEGLU) {
                    swiglu(go, uo, so, stream);
                } else {
                    geglu(go, uo, so, stream);
                }
                hadamard_transform_fp16(static_cast<const half*>(so.data),
                                        static_cast<half*>(so.data), 1, K_d, hbs, stream);
                gemv_mxfp4_residual(wd_mxfp4,
                                     static_cast<const half*>(so.data),
                                     static_cast<half*>(h.data),
                                     static_cast<const half*>(h.data),
                                     M_d, K_d, stream);
            } else if (cfg.ffn_activation != FFNActivation::GEGLU) {
                gemv_mxfp4_swiglu_residual(wd_mxfp4,
                                            static_cast<const half*>(go.data),
                                            static_cast<const half*>(uo.data),
                                            static_cast<half*>(h.data),
                                            static_cast<const half*>(h.data),
                                            M_d, K_d, stream);
            } else {
                gemv_mxfp4_geglu_residual(wd_mxfp4,
                                           static_cast<const half*>(go.data),
                                           static_cast<const half*>(uo.data),
                                           static_cast<half*>(h.data),
                                           static_cast<const half*>(h.data),
                                           M_d, K_d, stream);
            }
        } else if (will_fuse_down_nvfp4) {
            // Reconstruct NvFP4QuantResult from handle payload.
            NvFP4QuantResult wd_nvfp4;
            wd_nvfp4.packed_data  = hwd->payload.nvfp4.data;
            wd_nvfp4.micro_scales = hwd->payload.nvfp4.block_scales;
            wd_nvfp4.tensor_scale = (hwd->payload.nvfp4.tensor_scale != nullptr)
                                    ? *hwd->payload.nvfp4.tensor_scale : 1.0f;
            wd_nvfp4.N = static_cast<int>(hwd->shape[0]);
            wd_nvfp4.K = static_cast<int>(hwd->shape[1]) * 2;  // packed → logical K
            int K_d = wd_nvfp4.K;
            int M_d = wd_nvfp4.N;
            int n_mb_d = K_d / 16;
            if (n_mb_d <= 512) {
                // Small K: fused SwiGLU/GeGLU + NVFP4 GEMV + residual (MR path)
                if (cfg.ffn_activation != FFNActivation::GEGLU) {
                    gemv_nvfp4_swiglu_residual(wd_nvfp4,
                                                static_cast<const half*>(go.data),
                                                static_cast<const half*>(uo.data),
                                                static_cast<half*>(h.data),
                                                static_cast<const half*>(h.data),
                                                M_d, K_d, stream);
                } else {
                    gemv_nvfp4_geglu_residual(wd_nvfp4,
                                               static_cast<const half*>(go.data),
                                               static_cast<const half*>(uo.data),
                                               static_cast<half*>(h.data),
                                               static_cast<const half*>(h.data),
                                               M_d, K_d, stream);
                }
            } else {
                // Large K (e.g. 12288): split activation + GEMV.
                // The fused kernel's silu/gelu compute dominates at large K,
                // while the separate vectorized activation kernel is ~1 μs,
                // then GEMV uses prmt dequant (no smem LUT, no bank conflicts).
                if (cfg.ffn_activation != FFNActivation::GEGLU) {
                    swiglu(go, uo, so, stream);
                } else {
                    geglu(go, uo, so, stream);
                }
                gemv_nvfp4_residual(wd_nvfp4,
                                     static_cast<const half*>(so.data),
                                     static_cast<half*>(h.data),
                                     static_cast<const half*>(h.data),
                                     M_d, K_d, stream);
            }
        } else if (fused_down_residual) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            // Fuse activation + Q8_1 quantization into a single kernel when possible.
            // This saves 1 kernel launch per layer (activation + quantize → single kernel).
            // NOTE: tried fusing act+quant+GEMV into one kernel but it regresses ~22%
            // because the 2-pass SwiGLU recomputation doubles gate/up L2 reads and the
            // kpar GEMV is already memory-bound on weight reads (same issue as O-proj
            // inline quant at line 674). Separate quant + kpar achieves higher occupancy.
            if (cfg.ffn_activation != FFNActivation::GEGLU) {
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            } else {
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            }
            // Use h.data as residual source (memcpy was skipped)
            const half* residual_ptr = static_cast<const half*>(h.data);
            dispatch_gemv_residual(ly.w_down_qtype, ly.w_down.data, q8, qscratch_.d8_buf,
                                   static_cast<half*>(h.data), residual_ptr,
                                   M_d, K_d, stream);
        } else if (has_post_ffn_norm && using_fp32_accum && n == 1 &&
                   wd_tier == StorageTier::NVFP4 && h.dtype == DType::FP16) {
            // NVFP4 post-norm FP32 accum decode: activation → NVFP4 GEMV → post-norm.
            // ~40% less weight traffic than dp4a Q8_0 path.
            NvFP4QuantResult wd_nvfp4;
            wd_nvfp4.packed_data  = hwd->payload.nvfp4.data;
            wd_nvfp4.micro_scales = hwd->payload.nvfp4.block_scales;
            wd_nvfp4.tensor_scale = (hwd->payload.nvfp4.tensor_scale != nullptr)
                                    ? *hwd->payload.nvfp4.tensor_scale : 1.0f;
            wd_nvfp4.N = static_cast<int>(hwd->shape[0]);
            wd_nvfp4.K = static_cast<int>(hwd->shape[1]) * 2;  // packed → logical K
            int K_d = wd_nvfp4.K;
            int M_d = wd_nvfp4.N;
            if (cfg.ffn_activation != FFNActivation::GEGLU)
                swiglu(go, uo, so, stream);
            else
                geglu(go, uo, so, stream);
            gemv_nvfp4_kpar(wd_nvfp4, static_cast<const half*>(so.data),
                             static_cast<half*>(fo.data), M_d, K_d, stream);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                static_cast<const half*>(fo.data),
                static_cast<const half*>(ly.post_ffn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                cfg.d_model, eps, norm_w_off_);
        } else if (has_post_ffn_norm && using_fp32_accum && n == 1 &&
                   q8 != nullptr && qscratch_.d8_buf != nullptr &&
                   is_dp4a_qtype(ly.w_down_qtype)) {
            // Post-norm FP32 accum decode: fused activation→Q8_1 + GEMV + fused post-norm.
            // Saves 3 kernel launches per layer vs the fallback path.
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            if (cfg.ffn_activation != FFNActivation::GEGLU)
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, qscratch_.d8_buf, K_d, stream);
            else
                geglu_quantize_q8_1(static_cast<const half*>(go.data),
                                    static_cast<const half*>(uo.data),
                                    q8, qscratch_.d8_buf, K_d, stream);
            half* fo_ptr = static_cast<half*>(fo.data);
            dispatch_dp4a_gemv(ly.w_down_qtype, ly.w_down.data, q8, qscratch_.d8_buf,
                               fo_ptr, M_d, K_d, stream);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                static_cast<const half*>(fo.data),
                static_cast<const half*>(ly.post_ffn_norm.data),
                static_cast<float*>(fp32_h.data),
                static_cast<half*>(h.data),
                cfg.d_model, eps, norm_w_off_);
        } else {
            // Non-dp4a paths: activation must produce FP16 intermediate in so.
            switch (cfg.ffn_activation) {
                case FFNActivation::GEGLU:  geglu(go, uo, so, stream);  break;
                default:                    swiglu(go, uo, so, stream);  break;
            }
            if (will_fuse_down_beta1 && !cur_force_fp16_ &&
                wd_tier == StorageTier::FP8 &&
                qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
                // FP8 beta=1: hidden = fp8(swiglu_out) @ fp8(w_down)^T + hidden
                Tensor fp8_so(qscratch_.fp8_act, DType::FP8_E4M3, so.ndim, so.shape, true);
                quantize_fp16_to_fp8_e4m3(so, fp8_so, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
                int64_t wshape[2] = {hwd->shape[0], hwd->shape[1]};
                Tensor fp8_wd(hwd->payload.fp8.data, DType::FP8_E4M3, 2, wshape, true);
                gemm_cublaslt(fp8_so, fp8_wd, h, 1.0f, 1.0f, qscratch_.d_act_scale, hwd->payload.fp8.d_scale, stream);
            } else if (will_fuse_down_beta1 && wd_tier == StorageTier::FP16) {
                // Fused: hidden = swiglu_out @ w_down^T + hidden (cuBLAS beta=1).
                gemm_dispatch(so, ly.w_down, ly.w_down_qtype, h, ctx.with_beta(1.0f));
            } else if ((will_fuse_down_beta1 || will_fuse_down_dequant_beta1) &&
                       qscratch_.dequant != nullptr && dequant_gpu_supported(ly.w_down_qtype)) {
                // Dequant into scratch, then beta=1.0 GEMM directly into hidden (which holds residual)
                gemm_dispatch(so, ly.w_down, ly.w_down_qtype, h, ctx.with_beta(1.0f));
            } else {
                gemm_dispatch(so, ly.w_down, ly.w_down_qtype, fo, ctx);
                if (has_post_ffn_norm && using_fp32_accum) {
                    // Post-FFN norm → FP32 accumulation (no D2D copy needed)
                    Tensor fp32_h = view_tokens(fp32_hidden_, n);
                    rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                        static_cast<const half*>(fo.data),
                        static_cast<const half*>(ly.post_ffn_norm.data),
                        static_cast<float*>(fp32_h.data),
                        static_cast<half*>(h.data),
                        cfg.d_model, eps, norm_w_off_);
                } else if (has_post_ffn_norm) {
                    // Post-FFN norm + residual: h = rmsnorm(fo) + r
                    // Fused: 2 ops → 1 kernel
                    rmsnorm_add_residual(fo, ly.post_ffn_norm, r, h,
                                         eps, stream, norm_w_off_);
                } else {
                    // No post-norm: h = fo + residual (fused add-store, no copy)
                    elementwise_add_store(fo, r, h, stream);
                }
            }
        }
    }

    // Clear L2 streaming hint so subsequent layers start with default policy.
    clear_l2_policy(stream);
}

} // namespace imp

// MoE FFN forward pass — extracted from executor_forward.cu for maintainability.
// Contains: GraphExecutor::run_moe_ffn() — complete MoE pipeline:
//   routing → expert dispatch → scatter → residual add
//
// Dispatch paths:
//   1. NVFP4 decode fast (n=1, NVFP4 cached weights)
//   2. TC fused (tensor-core Q6K/Q4K GEMM, persistent work-queue)
//   3. Scalar fused (dp4a GEMV for small expert sizes)
//   4. Batch path (cuBLAS/CUTLASS for large batches)
//   5. Shared expert path (parallel dense FFN when present)

#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_debug.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "memory/gdn_state.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "compute/gemm_cublaslt_nvfp4.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>


namespace imp {

// Multiply each routing weight by the scale of its selected expert.
// Used by Gemma 4 to apply per-expert output scale before the routing sum.
__global__ void moe_apply_per_expert_scale_kernel(
    float* __restrict__ weights,            // [n_weights = n_tokens * top_k]
    const int32_t* __restrict__ indices,    // [n_weights]
    const __half* __restrict__ scales,      // [n_experts]
    int n_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_weights) return;
    int eid = indices[idx];
    float s = __half2float(scales[eid]);
    weights[idx] *= s;
}

// Replace +/-Inf with 0 in an FP16 tensor (in-place).
// Used to sanitize FP16 GEMM outputs that overflow (e.g. Gemma 4 shared MLP at deep layers).
__global__ void sanitize_fp16_kernel(__half* __restrict__ data, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __half v = data[idx];
    // __hisinf returns ±1 for ±inf, 0 otherwise. __hisnan returns non-zero for NaN.
    if (__hisinf(v) != 0 || __hisnan(v)) {
        data[idx] = __float2half(0.0f);
    }
}

static void sanitize_fp16(__half* data, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sanitize_fp16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

namespace {

// ---------------------------------------------------------------------------
// Decode-fast eligibility predicate.
// Returns true when n=1, packed expert weights are on-device in a supported
// quantization format, the dequant buffer exists, and compute dtype is FP16.
// ---------------------------------------------------------------------------
static bool can_decode_fast(int n, const Tensor& expert_up_packed,
                            GGMLQuantType up_qtype, void* dequant_buf,
                            DType compute_dtype) {
    return (n == 1 &&
            expert_up_packed.data != nullptr && dequant_buf != nullptr &&
            compute_dtype == DType::FP16 &&
            expert_up_packed.on_device &&
            (up_qtype == GGMLQuantType::Q6_K || up_qtype == GGMLQuantType::Q8_0 ||
             up_qtype == GGMLQuantType::Q4_0 || up_qtype == GGMLQuantType::Q4_K ||
             up_qtype == GGMLQuantType::Q5_K || up_qtype == GGMLQuantType::Q2_K ||
             up_qtype == GGMLQuantType::Q3_K));
}

// ---------------------------------------------------------------------------
// Expert activation dispatch: SwiGLU for gated experts, ReLU^2 for non-gated.
// Operates on the expanded-layout buffers (shape [rows, eff]).
// ---------------------------------------------------------------------------
static void apply_expert_activation(void* gate_data, void* up_data, void* swiglu_data,
                                    bool non_gated, int64_t rows, int64_t eff,
                                    DType compute_dtype, FFNActivation act_type,
                                    cudaStream_t stream) {
    int64_t act_shape[2] = {rows, eff};
    if (non_gated) {
        Tensor up_t(up_data, compute_dtype, 2, act_shape, true);
        relu_sqr_inplace(up_t, stream);
    } else {
        Tensor g(gate_data, compute_dtype, 2, act_shape, true);
        Tensor u(up_data, compute_dtype, 2, act_shape, true);
        Tensor a(swiglu_data, compute_dtype, 2, act_shape, true);
        if (act_type == FFNActivation::GEGLU) geglu(g, u, a, stream);
        else swiglu(g, u, a, stream);
    }
}

// ---------------------------------------------------------------------------
// Compute expert stride (bytes between experts in a packed tensor).
// ---------------------------------------------------------------------------
static size_t expert_stride(const Tensor& packed, GGMLQuantType qtype) {
    int64_t rows = packed.shape[1];
    int64_t cols = packed.shape[2];
    return static_cast<size_t>(rows) * ggml_quant_row_bytes(qtype, cols);
}

} // anonymous namespace

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    int n       = cur_n_tokens_;
    int d       = cfg.d_model;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    int eff     = max_expert_eff_;
    float eps   = cfg.rms_norm_eps;
    size_t es   = dtype_size(compute_dtype_);
    int expanded = n * top_k;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    bool residual_fused = false;  // set true if decode fast path fuses residual add

    // 1. Save residual (skip if decode fast path will handle it —
    //    h.data is never written before the final weighted_sum_residual).
    // Gemma 4: parallel branches — MoE experts use rmsnorm(h, pre_ffw_norm_2),
    // shared MLP uses rmsnorm(h, ffn_norm). Pick MoE-side norm here; the shared
    // branch recomputes its own norm later (reading from the saved residual).
    const Tensor& norm_w =
        (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr)
            ? ly.ffn_pre_norm_2
        : (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;

    // Pre-check: does NVFP4 MoE cache cover all expert tensors for this layer?
    // If so, the NVFP4 path doesn't need Q8_1 quantization (takes FP16 directly).
    bool nvfp4_covers_layer = false;
    if (n == 1 && compute_dtype_ == DType::FP16) {
        bool has_up = wcache_.nvfp4_moe.count(ly.expert_up_packed.data) > 0;
        bool has_down = wcache_.nvfp4_moe.count(ly.expert_down_packed.data) > 0;
        nvfp4_covers_layer = has_up && has_down;
        if (nvfp4_covers_layer && ly.expert_gate_packed.data != nullptr) {
            nvfp4_covers_layer = wcache_.nvfp4_moe.count(ly.expert_gate_packed.data) > 0;
        }
    }

    // Pre-check decode fast path (same logic as will_decode_fast below)
    bool will_skip_residual_copy = can_decode_fast(n, ly.expert_up_packed,
        ly.expert_up_qtype, moe_.dequant_buf, compute_dtype_) &&
        ly.w_up_shared.data == nullptr;  // must not have shared expert for full residual fusion

    if (!will_skip_residual_copy) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream));
    }
    // Fused RMSNorm + Q8_1: skip for NVFP4-covered layers (NVFP4 takes FP16 directly)
    bool moe_fused_norm_q8 = (n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                               h.dtype == DType::FP16 && !nvfp4_covers_layer);
    if (moe_fused_norm_q8) {
        // Fused: RMSNorm + Q8_1 (also writes FP16 norm_out for gate logits)
        rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                static_cast<const half*>(norm_w.data),
                                static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf,
                                static_cast<half*>(no.data),
                                d, eps, stream, norm_w_off_);
    } else {
        rmsnorm(h, norm_w, no, eps, stream, norm_w_off_);
    }

    // 3. Gate logits + top-k routing
    //    Gemma 4: router input = rms_norm(h) * ffn_gate_inp_scale / sqrt(d_model).
    //    Other archs: router input = no (ffn-normalized h).
    Tensor router_in = no;
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_gate_inp_scale.data != nullptr &&
        getenv("IMP_G4_NO_CUSTOM_ROUTER") == nullptr) {
        // Reuse moe_.scatter_out (FP32, max_tokens*d) as FP16 scratch:
        // FP16 needs half the bytes so we only use half of it.
        int64_t ri_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
        router_in = Tensor(moe_.scatter_out.data, compute_dtype_, 2, ri_shape, true);
        // Step 1: rmsnorm(h, ffn_gate_inp_scale, router_in) with offset=0
        //         (the scale tensor holds raw per-channel scales, no -1 trick)
        rmsnorm(h, ly.ffn_gate_inp_scale, router_in, eps, stream, 0.0f);
        // Step 2: multiply by 1/sqrt(d_model) scalar
        int64_t total_elems = static_cast<int64_t>(n) * d;
        int threads = 256;
        int blocks = static_cast<int>((total_elems / 2 + threads - 1) / threads);
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(router_in.data),
            __float2half(inv_sqrt_d),
            total_elems);
    }

    const void* router_bias_ptr = ly.moe_router_bias.data;
    bool use_sigmoid = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;

    GGMLQuantType up_qtype = ly.expert_up_qtype;
    bool will_decode_fast = can_decode_fast(n, ly.expert_up_packed, up_qtype,
                                            moe_.dequant_buf, compute_dtype_);
    // Gemma 4: decode fast path uses gemv_*_moe_gate_up_fused kernels that have
    // stride/layout issues with the split gate_up_exps tensor. Force slow path.
    if (cfg.arch == ModelArch::GEMMA4) {
        will_decode_fast = false;
    }

    MoeRoutingResult routing;

    // Fused gate GEMV + topk is only beneficial when n_experts fits in the
    // number of warps (8). For high expert counts (e.g., 128 in Qwen3-Coder),
    // the separate gemv_gate_fp32 (128 parallel blocks) is much faster than
    // serializing 128/8=16 experts per warp in a single block.
    constexpr int kMaxFusedExperts = 8;
    if (ne <= kMaxFusedExperts &&
        n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16 &&
        moe_.routing_buffers.pool && will_decode_fast) {
        // Fused: gate GEMV + softmax/sigmoid + top-k in one kernel (1 launch)
        moe_gate_topk_fused(static_cast<const half*>(ly.moe_gate.data),
                            static_cast<const half*>(router_in.data),
                            ne, d, top_k,
                            moe_.routing_buffers, routing, stream,
                            use_sigmoid, norm_weights, router_bias_ptr);
    } else {
        // Separate: gate GEMV → intermediate logits → topk gating
        Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, n);

        if (n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16) {
            gemv_gate_fp32(static_cast<const half*>(ly.moe_gate.data),
                           static_cast<const half*>(router_in.data),
                           static_cast<float*>(gate_logits_f32.data),
                           ne, d, stream);
        } else {
            int64_t gl_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(ne)};
            Tensor gate_logits_tmp(moe_.gathered.data, compute_dtype_, 2, gl_shape, true);
            gemm(router_in, ly.moe_gate, gate_logits_tmp, 1.0f, 0.0f, stream);

            int64_t numel = static_cast<int64_t>(n) * ne;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == DType::FP16) {
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const half*>(gate_logits_tmp.data),
                    static_cast<float*>(gate_logits_f32.data),
                    numel);
            } else {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                                static_cast<size_t>(numel) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
            }
        }

        if (moe_.routing_buffers.pool) {
            moe_topk_gating(gate_logits_f32, top_k, moe_.routing_buffers, routing, stream, use_sigmoid, norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
        }
    }

    // 4b. Expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data),
            cfg.expert_weights_scale, n_weights);
    }

    // 4c. Gemma 4: per-expert output scale. Multiply each token's routing weight
    // by the scale of its selected expert. Mathematically equivalent to scaling
    // each expert's down output then summing — saves a separate scatter pass.
    // (Matches llama.cpp ffn_moe_down_scaled = MUL(ffn_moe_down, repeat(scale)).)
    if (cfg.arch == ModelArch::GEMMA4 && ly.expert_down_scale.data != nullptr &&
        ly.expert_down_scale.on_device) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        moe_apply_per_expert_scale_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data),
            static_cast<const int32_t*>(routing.expert_indices.data),
            static_cast<const half*>(ly.expert_down_scale.data),
            static_cast<int>(n_weights));
    }

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    bool use_packed_dequant = (ly.expert_up_packed.data != nullptr &&
                               moe_.dequant_buf != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    bool non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                              (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (use_packed_dequant) {
        int64_t ref_eff = non_gated_experts
            ? ly.expert_up_packed.shape[1]
            : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != eff || down_eff != eff) {
            IMP_LOG_ERROR("CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
                         "down_packed.shape[2]=%ld. Using packed tensor shapes instead.",
                         eff, (long)ref_eff, (long)down_eff);
            eff = static_cast<int>(ref_eff);
        }
    }

    // =========================================================================
    // DECODE FAST PATH: n=1, device-resident packed experts, Q6_K or Q8_0.
    // Skips gather/scatter and D2H sync. All top_k experts dispatched in a
    // single kernel launch per projection. CUDA-graph capturable.
    // =========================================================================
    // decode_fast_path == will_decode_fast (computed earlier before routing).
    // will_decode_fast already checks packed data + dequant buf + FP16 + on_device + Q6K/Q8_0.
    bool decode_fast_path = will_decode_fast;

    if (decode_fast_path) {
        // Device pointers from routing result (no D2H copy needed)
        const int32_t* expert_indices = static_cast<const int32_t*>(routing.expert_indices.data);
        const float* expert_weights   = static_cast<const float*>(routing.expert_weights.data);

        half* norm_ptr = static_cast<half*>(no.data);
        half* gate_buf = static_cast<half*>(moe_.expert_gate.data);   // [top_k, eff]
        half* up_buf   = static_cast<half*>(moe_.expert_up.data);     // [top_k, eff]
        half* act_buf  = static_cast<half*>(moe_.expert_swiglu.data); // [top_k, eff]
        half* down_buf = static_cast<half*>(moe_.expert_down.data);   // [top_k, d]

        // --- NVFP4 MoE path: takes FP16 input directly, no Q8_1 needed ---
        auto nvfp4_up_it = wcache_.nvfp4_moe.find(ly.expert_up_packed.data);
        auto nvfp4_down_it = wcache_.nvfp4_moe.find(ly.expert_down_packed.data);
        bool use_nvfp4_moe = (nvfp4_up_it != wcache_.nvfp4_moe.end() &&
                              nvfp4_down_it != wcache_.nvfp4_moe.end());
        if (use_nvfp4_moe && !non_gated_experts) {
            auto nvfp4_gate_it = wcache_.nvfp4_moe.find(ly.expert_gate_packed.data);
            use_nvfp4_moe = (nvfp4_gate_it != wcache_.nvfp4_moe.end());
        }

        if (use_nvfp4_moe) {
            // Gate+Up projection: NVFP4 MoE GEMV with FP16 input (norm_ptr)
            if (!non_gated_experts) {
                gemv_nvfp4_moe_gate_up_fused(
                    wcache_.nvfp4_moe.at(ly.expert_gate_packed.data),
                    wcache_.nvfp4_moe.at(ly.expert_up_packed.data),
                    expert_indices, norm_ptr,
                    gate_buf, up_buf, eff, d, top_k, stream);
            } else {
                gemv_nvfp4_moe_decode(
                    wcache_.nvfp4_moe.at(ly.expert_up_packed.data),
                    expert_indices, norm_ptr, up_buf,
                    eff, d, /*x_stride=*/0, top_k, stream);
            }

            // Down projection (fused SwiGLU+GEMV for gated, separate for non-gated)
            if (!non_gated_experts) {
                // Fused: swiglu(gate,up) computed inline during down GEMV
                gemv_nvfp4_moe_swiglu_decode(
                    wcache_.nvfp4_moe.at(ly.expert_down_packed.data),
                    expert_indices, gate_buf, up_buf, down_buf,
                    d, eff, /*x_stride=*/eff, top_k, stream);
            } else {
                int64_t act_shape[2] = {static_cast<int64_t>(top_k),
                                         static_cast<int64_t>(eff)};
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
                gemv_nvfp4_moe_decode(
                    wcache_.nvfp4_moe.at(ly.expert_down_packed.data),
                    expert_indices, up_buf, down_buf,
                    d, eff, /*x_stride=*/eff, top_k, stream);
            }

            // Weighted sum + residual
            {
                bool has_shared_expert = (ly.w_up_shared.data != nullptr);
                const void* res_ptr = has_shared_expert ? nullptr :
                    (will_skip_residual_copy ? h.data : r.data);
                moe_weighted_sum_residual(down_buf, expert_weights, res_ptr,
                                          h.data, d, top_k, stream);
                if (!has_shared_expert) residual_fused = true;
            }

            goto moe_after_experts;
        }

        // Use dp4a MMVQ path when Q8_1 buffers are available
        bool use_dp4a = (qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr);

        if (use_dp4a) {
            // Q8_1 may already be computed by the fused norm+quant above.
            // If not (e.g., prefill or non-FP16), quantize norm_out now.
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            if (!moe_fused_norm_q8) {
                quantize_fp16_to_q8_1(norm_ptr, q8, qscratch_.d8_buf, d, stream);
            }

            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            // 5'+6'. Fused gate+up projection (single kernel launch)
            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q4_K) {
                    gemv_q4_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q5_K) {
                    gemv_q5_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q4_0) {
                    gemv_q4_0_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q2_K) {
                    gemv_q2_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q3_K) {
                    gemv_q3_k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                }
            } else {
                // Non-gated: up projection only
                auto moe_gemv_dp4a = (up_qtype == GGMLQuantType::Q6_K)
                    ? gemv_q6k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q4_0)
                    ? gemv_q4_0_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q4_K)
                    ? gemv_q4_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q5_K)
                    ? gemv_q5_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q2_K)
                    ? gemv_q2_k_q8_1_moe_decode
                    : (up_qtype == GGMLQuantType::Q3_K)
                    ? gemv_q3_k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
                moe_gemv_dp4a(ly.expert_up_packed.data, expert_indices,
                              q8, qscratch_.d8_buf, up_buf,
                              eff, d, up_stride_bytes,
                              /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
            }
        } else {
            // Fallback: FP16 dequant path
            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q8_0) {
                    gemv_q8_0_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                } else {
                    IMP_LOG_ERROR("MoE non-dp4a gate_up_fused: no kernel for qtype %d, skipping GEMV", (int)up_qtype);
                }
            } else {
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_moe_decode(ly.expert_up_packed.data, expert_indices,
                                        norm_ptr, up_buf,
                                        eff, d, up_stride_bytes, /*x_stride=*/0, top_k, stream);
                } else if (up_qtype == GGMLQuantType::Q8_0) {
                    gemv_q8_0_moe_decode(ly.expert_up_packed.data, expert_indices,
                                         norm_ptr, up_buf,
                                         eff, d, up_stride_bytes, /*x_stride=*/0, top_k, stream);
                } else {
                    IMP_LOG_ERROR("MoE non-dp4a up projection: no kernel for qtype %d, skipping GEMV", (int)up_qtype);
                }
            }
        }

        // 7'+8'. Activation + down projection
        //
        // When dp4a is active and experts are gated (SwiGLU), fuse the activation
        // and Q8_1 quantization into a single kernel, eliminating the intermediate
        // FP16 act_buf write+read.
        if (use_dp4a) {
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            int eff_q8_blocks = eff / 32;

            if (!non_gated_experts) {
                // Fused SwiGLU → Q8_1 (1 kernel instead of 2)
                swiglu_quantize_q8_1(gate_buf, up_buf, q8, qscratch_.d8_buf,
                                      top_k * eff, stream);
            } else {
                // Non-gated (relu²): fused relu² + Q8_1 quantization (1 kernel)
                relu_sqr_quantize_q8_1(up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
            }

            // Down projection with dp4a GEMV
            auto moe_gemv_dp4a_down = (up_qtype == GGMLQuantType::Q6_K)
                ? gemv_q6k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q4_0)
                ? gemv_q4_0_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q4_K)
                ? gemv_q4_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q5_K)
                ? gemv_q5_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q2_K)
                ? gemv_q2_k_q8_1_moe_decode
                : (up_qtype == GGMLQuantType::Q3_K)
                ? gemv_q3_k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            moe_gemv_dp4a_down(ly.expert_down_packed.data, expert_indices,
                          q8, qscratch_.d8_buf, down_buf,
                          d, eff, down_stride,
                          /*q8_1_stride=*/eff_q8_blocks, /*d8_stride=*/eff_q8_blocks,
                          top_k, stream);
        } else {
            // Non-dp4a: separate activation + FP16 down GEMV
            apply_expert_activation(gate_buf, up_buf, act_buf,
                                    non_gated_experts, top_k, eff,
                                    compute_dtype_, cfg.ffn_activation, stream);
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            half* down_input = non_gated_experts ? up_buf : act_buf;
            if (up_qtype == GGMLQuantType::Q6_K) {
                gemv_q6k_moe_decode(ly.expert_down_packed.data, expert_indices,
                                    down_input, down_buf,
                                    d, eff, down_stride, /*x_stride=*/eff, top_k, stream);
            } else if (up_qtype == GGMLQuantType::Q8_0) {
                gemv_q8_0_moe_decode(ly.expert_down_packed.data, expert_indices,
                                     down_input, down_buf,
                                     d, eff, down_stride, /*x_stride=*/eff, top_k, stream);
            } else {
                IMP_LOG_ERROR("MoE non-dp4a down projection: no kernel for qtype %d, skipping GEMV", (int)up_qtype);
            }
        }

        // 9'. Fused weighted sum + FP16 output (+ residual if no shared expert)
        {
            bool has_shared_expert = (ly.w_up_shared.data != nullptr);
            // Use h.data as residual source when memcpy was skipped
            const void* res_ptr = has_shared_expert ? nullptr :
                (will_skip_residual_copy ? h.data : r.data);
            moe_weighted_sum_residual(down_buf, expert_weights, res_ptr,
                                      h.data, d, top_k, stream);
            if (!has_shared_expert) residual_fused = true;
        }

        goto moe_after_experts;
    }

    // =========================================================================
    // GENERAL PATH: prefill or host-offloaded or non-Q6K/Q8_0 experts
    // =========================================================================

    // =========================================================================
    // FUSED Q6_K PREFILL PATH: reads Q6_K weights directly, eliminates the
    // intermediate FP16/FP8 dequant buffer. Two variants:
    //   TC (tensor core): WMMA 16×16×16, preferred for large batches
    //   Scalar: disabled (FP16 batch path always wins for small batches)
    // =========================================================================
    {
    bool can_fused_q6k = (ne > 16 &&
                          ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == GGMLQuantType::Q6_K &&
                          ly.expert_down_qtype == GGMLQuantType::Q6_K &&
                          compute_dtype_ == DType::FP16);
    if (can_fused_q6k && !non_gated_experts)
        can_fused_q6k = (ly.expert_gate_packed.data &&
                         ly.expert_gate_packed.on_device &&
                         ly.expert_gate_qtype == GGMLQuantType::Q6_K);

    bool use_tc = can_fused_q6k && (expanded > ne * 12);
    bool use_scalar = can_fused_q6k && !use_tc && (expanded <= ne * 12);

    if (use_tc || use_scalar) {
        if (layer == 0) IMP_LOG_INFO("MoE prefill: fused Q6_K %s path (n=%d, expanded=%d)",
                                      use_tc ? "TC" : "scalar", n, expanded);
        const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
        const int32_t* d_sorted  = static_cast<const int32_t*>(routing.sorted_token_ids.data);
        char* expert_gate_base  = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base    = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base= static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base  = static_cast<char*>(moe_.expert_down.data);

        if (use_tc) {
            // TC path: gather-free via sorted_token_ids indirection.
            // Gate and up read from original hidden state (no.data), down reads
            // from SwiGLU output (already in expanded layout, no indirection).

            // Gate projection (gated models only)
            if (!non_gated_experts)
                gemm_q6k_fused_moe_prefill_tc(
                    ly.expert_gate_packed.data,
                    no.data, expert_gate_base, d_offsets,
                    eff, d,
                    expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype),
                    ne, stream, d_sorted);

            // Up projection
            gemm_q6k_fused_moe_prefill_tc(
                ly.expert_up_packed.data,
                no.data, expert_up_base, d_offsets,
                eff, d,
                expert_stride(ly.expert_up_packed, up_qtype),
                ne, stream, d_sorted);

        } else {
            // Scalar path: needs gathered buffer
            {
                int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                          static_cast<int64_t>(d)};
                Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
                moe_gather(no, routing, gathered, stream);
            }
            char* gathered_base = static_cast<char*>(moe_.gathered.data);

            if (!non_gated_experts)
                gemm_q6k_fused_moe_prefill(
                    ly.expert_gate_packed.data,
                    gathered_base, expert_gate_base, d_offsets,
                    eff, d,
                    expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype),
                    ne, stream);

            gemm_q6k_fused_moe_prefill(
                ly.expert_up_packed.data,
                gathered_base, expert_up_base, d_offsets,
                eff, d,
                expert_stride(ly.expert_up_packed, up_qtype),
                ne, stream);
        }

        // Activation (FP16)
        apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                moe_.expert_swiglu.data, non_gated_experts,
                                expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

        // Down projection (reads from expanded-layout SwiGLU output, no indirection)
        char* fused_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        if (use_tc) {
            gemm_q6k_fused_moe_prefill_tc(
                ly.expert_down_packed.data,
                fused_down_act, expert_down_base, d_offsets,
                d, eff,
                expert_stride(ly.expert_down_packed, ly.expert_down_qtype),
                ne, stream);
        } else {
            gemm_q6k_fused_moe_prefill(
                ly.expert_down_packed.data,
                fused_down_act, expert_down_base, d_offsets,
                d, eff,
                expert_stride(ly.expert_down_packed, ly.expert_down_qtype),
                ne, stream);
        }

        // Falls through to scatter (step 7)
    } else {
    // =========================================================================
    // FP16 BATCH or FP8 BATCH PREFILL PATH
    // Pre-check: FP16 batch + device-grouped GEMM is preferred (no D2H sync,
    // simpler pipeline). FP8 batch is only used as fallback when FP16 batch
    // isn't available.
    // =========================================================================

    // Gather: reorder tokens by expert assignment (required for batch/legacy paths)
    {
        int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                  static_cast<int64_t>(d)};
        Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
        moe_gather(no, routing, gathered, stream);
    }

    {
    // FP16 batch check: can we dequant all experts to FP16 and use device-grouped GEMM?
    size_t fp16_per_expert = static_cast<size_t>(std::max(
        ly.expert_up_packed.shape[1] * ly.expert_up_packed.shape[2],
        ly.expert_down_packed.shape[1] * ly.expert_down_packed.shape[2])) * sizeof(half);
    bool can_fp16_batch_nosync = (
        moe_.batch_dequant_buf != nullptr &&
        moe_.batch_dequant_buf_size >= static_cast<size_t>(ne) * fp16_per_expert &&
        moe_.d_weight_ptrs && moe_.d_weight_ptrs_count >= ne &&
        ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
        ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
        dequant_gpu_supported(up_qtype) &&
        dequant_gpu_supported(ly.expert_down_qtype));
    if (can_fp16_batch_nosync && !non_gated_experts)
        can_fp16_batch_nosync = (ly.expert_gate_packed.data &&
                                  ly.expert_gate_packed.on_device &&
                                  dequant_gpu_supported(ly.expert_gate_qtype));

    // FP8 batch check: fallback when FP16 batch isn't available
    size_t up_fp8_sz   = static_cast<size_t>(ne) * ly.expert_up_packed.shape[1]
                       * ly.expert_up_packed.shape[2];
    size_t down_fp8_sz = static_cast<size_t>(ne) * ly.expert_down_packed.shape[1]
                       * ly.expert_down_packed.shape[2];
    size_t max_act_cols = std::max(static_cast<size_t>(ly.expert_up_packed.shape[2]),
                                   static_cast<size_t>(ly.expert_down_packed.shape[2]));
    size_t fp8_buf_needed = std::max(up_fp8_sz, down_fp8_sz)
                          + static_cast<size_t>(expanded) * max_act_cols;
    bool can_fp8_batch = (!can_fp16_batch_nosync &&
                          moe_.batch_dequant_buf != nullptr &&
                          moe_.batch_dequant_buf_size >= fp8_buf_needed &&
                          ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == GGMLQuantType::Q6_K &&
                          ly.expert_down_qtype == GGMLQuantType::Q6_K &&
                          compute_dtype_ == DType::FP16 &&
                          !wcache_.fp16.count(ly.expert_up_packed.data));
    if (can_fp8_batch && !non_gated_experts)
        can_fp8_batch = (ly.expert_gate_packed.data &&
                         ly.expert_gate_packed.on_device &&
                         ly.expert_gate_qtype == GGMLQuantType::Q6_K);

    if (can_fp16_batch_nosync) {
        // =================================================================
        // FP16 BATCH DEQUANT + cublasGemmGroupedBatchedEx
        // Dequants all experts Q6_K→FP16 into batch buffer, then runs
        // a single cublasGemmGroupedBatchedEx per projection. One D2H
        // sync per layer for offsets (unavoidable for grouped GEMM API).
        // =================================================================
        if (layer == 0) IMP_LOG_INFO("MoE prefill: FP16 batch + grouped GEMM path (n=%d, expanded=%d)",
                                      n, expanded);

        // One D2H sync per layer for expert offsets
        std::vector<int32_t> h_offsets(ne + 1);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                        static_cast<size_t>(ne + 1) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        char* buf = static_cast<char*>(moe_.batch_dequant_buf);
        char* gathered_base     = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base  = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base    = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base= static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base  = static_cast<char*>(moe_.expert_down.data);

        auto batch_dequant_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                       const char* a_base, char* c_base,
                                       int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

            dequant_gpu(static_cast<const uint8_t*>(packed.data), buf, qtype,
                        ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            gemm_moe_batched(a_base, c_base,
                             h_offsets.data(), b_ptrs.data(),
                             K_dim, N_dim, DType::FP16, ne, stream,
                             moe_.d_work_ptrs);
        };

        // Gate projection
        if (!non_gated_experts)
            batch_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                                gathered_base, expert_gate_base, d, eff);

        // Up projection
        batch_dequant_gemm(ly.expert_up_packed, up_qtype,
                            gathered_base, expert_up_base, d, eff);

        // Activation
        apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                moe_.expert_swiglu.data, non_gated_experts,
                                expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

        // Down projection
        char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        batch_dequant_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                            down_act, expert_down_base, eff, d);

        // Falls through to scatter (step 7)

    } else if (can_fp8_batch) {
        if (layer == 0) IMP_LOG_INFO("MoE prefill: FP8 batch path (n=%d, expanded=%d, buf=%.1f MiB, need=%.1f MiB)",
                                      n, expanded, moe_.batch_dequant_buf_size / (1024.0*1024.0), fp8_buf_needed / (1024.0*1024.0));
        // Expert offsets: device-grouped path uses d_offsets directly on GPU.
        // Host offsets + sync are deferred to the legacy fallback path only.
        char* buf = static_cast<char*>(moe_.batch_dequant_buf);

        // FP8 batched GEMM lambda: dequant Q6_K→FP8, quantize FP16 acts→FP8, cuBLAS FP8 GEMM→FP16
        auto chunked_fp8_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                     const char* a_base_fp16, char* c_base_fp16,
                                     int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t weight_fp8_bytes = static_cast<size_t>(ne) * rows * cols;  // 1 byte per FP8 element

            // Buffer layout in moe_.batch_dequant_buf:
            //   [0 .. weight_fp8_bytes)                     = FP8 weights for all experts
            //   [weight_fp8_bytes .. weight_fp8_bytes + act) = FP8 activations
            uint8_t* fp8_weights = reinterpret_cast<uint8_t*>(buf);
            uint8_t* fp8_acts = fp8_weights + weight_fp8_bytes;

            // 1. Dequant all experts Q6_K → FP8 E4M3
            dequant_gpu_fp8(packed.data, fp8_weights, qtype,
                            ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            // 2. Per-expert FP8 scaling: calibrate absmax per expert, quantize with
            //    per-expert scale. Falls back to scale=1.0 if scale buffer unavailable.
            const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
            if (moe_.d_fp8_scales) {
                // Calibrate per-expert: writes scales to moe_.d_fp8_scales
                calibrate_fp8_scales_per_expert(a_base_fp16, K_dim, d_offsets, ne,
                                                 moe_.d_fp8_scales, stream);
                // Quantize with per-expert scale
                quantize_fp16_to_fp8_e4m3_per_expert(a_base_fp16, fp8_acts,
                                                      K_dim, d_offsets, ne,
                                                      moe_.d_fp8_scales, stream);
                // Note: no D2H sync here — device-grouped path uses device-side
                // scales directly. Host scales are only needed by the fallback path.
            } else {
                // Fallback: uniform scale=1.0
                quantize_fp16_to_fp8_e4m3_scaled(a_base_fp16, fp8_acts,
                                                  expanded * K_dim, 1.0f, stream);
            }

            // 3. Build per-expert FP8 weight pointers and dispatch GEMM via
            //    cublasGemmGroupedBatchedEx (single call for all experts).
            size_t expert_fp8_sz = static_cast<size_t>(rows) * cols;
            std::vector<int32_t> h_offsets(ne + 1);
            cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                            static_cast<size_t>(ne + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            std::vector<float> h_act_scales(ne, 1.0f);
            if (moe_.d_fp8_scales) {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_act_scales.data(), moe_.d_fp8_scales,
                                static_cast<size_t>(ne) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
            }
            cudaStreamSynchronize(stream);
            std::vector<const void*> weight_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                weight_ptrs[e] = fp8_weights + static_cast<size_t>(e) * expert_fp8_sz;

            gemm_moe_batched(fp8_acts, c_base_fp16,
                             h_offsets.data(), weight_ptrs.data(),
                             K_dim, N_dim, DType::FP8_E4M3, ne, stream,
                             moe_.d_work_ptrs, /*output_dtype=*/DType::FP16,
                             moe_.d_fp8_scales ? h_act_scales.data() : nullptr);
        };

        char* gathered_base     = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base  = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base    = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base= static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base  = static_cast<char*>(moe_.expert_down.data);

        // Gate projection (gated models only)
        if (!non_gated_experts)
            chunked_fp8_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                             gathered_base, expert_gate_base, d, eff);

        // Up projection
        chunked_fp8_gemm(ly.expert_up_packed, up_qtype,
                         gathered_base, expert_up_base, d, eff);

        // Activation (FP16 — reuse existing kernels)
        apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                moe_.expert_swiglu.data, non_gated_experts,
                                expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

        // Down projection: up buffer for non-gated (relu² in-place), swiglu for gated
        char* fp8_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        chunked_fp8_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                         fp8_down_act, expert_down_base, eff, d);

        // Falls through to existing scatter (step 7)

    } else if (wcache_.nvfp4_moe.count(ly.expert_up_packed.data) &&
               wcache_.nvfp4_moe.count(ly.expert_down_packed.data) &&
               (non_gated_experts || wcache_.nvfp4_moe.count(ly.expert_gate_packed.data)) &&
               moe_.batch_dequant_buf != nullptr &&
               moe_.batch_dequant_buf_size >= static_cast<size_t>(ne) * fp16_per_expert &&
               moe_.d_weight_ptrs && moe_.d_weight_ptrs_count >= ne) {
        // =================================================================
        // NVFP4→FP16 BATCH DEQUANT + grouped GEMM (prefill fallback when Q6K freed)
        // Dequants NVFP4 expert weights to FP16, then same grouped GEMM as FP16 batch.
        // =================================================================
        if (layer == 0) IMP_LOG_INFO("MoE prefill: NVFP4→FP16 batch path (n=%d, expanded=%d)",
                                      n, expanded);

        std::vector<int32_t> h_offsets(ne + 1);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                        static_cast<size_t>(ne + 1) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        char* buf = static_cast<char*>(moe_.batch_dequant_buf);
        char* gathered_base     = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base  = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base    = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base= static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base  = static_cast<char*>(moe_.expert_down.data);

        auto nvfp4_batch_dequant_gemm = [&](const void* packed_key,
                                             const char* a_base, char* c_base,
                                             int K_dim, int N_dim) {
            const auto& nvfp4 = wcache_.nvfp4_moe.at(packed_key);
            int64_t rows = nvfp4.N;
            int64_t cols = nvfp4.K;
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

            // Dequant all experts NVFP4 → FP16
            dequantize_nvfp4_moe_to_fp16(nvfp4, buf, stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            gemm_moe_batched(a_base, c_base,
                             h_offsets.data(), b_ptrs.data(),
                             K_dim, N_dim, DType::FP16, ne, stream,
                             moe_.d_work_ptrs);
        };

        // Gate projection
        if (!non_gated_experts)
            nvfp4_batch_dequant_gemm(ly.expert_gate_packed.data,
                                      gathered_base, expert_gate_base, d, eff);

        // Up projection
        nvfp4_batch_dequant_gemm(ly.expert_up_packed.data,
                                  gathered_base, expert_up_base, d, eff);

        // Activation
        apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                moe_.expert_swiglu.data, non_gated_experts,
                                expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

        // Down projection
        char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        nvfp4_batch_dequant_gemm(ly.expert_down_packed.data,
                                  down_act, expert_down_base, eff, d);

        // Falls through to scatter (step 7)

    } else {
    // =========================================================================
    // LEGACY FALLBACK: D2H sync + per-expert or batched GEMM
    // =========================================================================
    if (layer == 0) IMP_LOG_INFO("MoE prefill: legacy FP16 fallback path (n=%d, expanded=%d)",
                                  n, expanded);
    {
    std::vector<int32_t> h_offsets(ne + 1);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                    static_cast<size_t>(ne + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Helper: dequant one expert's weight from packed tensor into dequant scratch slot 0.
    // Returns a Tensor view into the scratch buffer with shape [rows, cols], FP16.
    // Uses slot 0 always -- safe because all ops are on the same stream, so the previous
    // GEMM reading from slot 0 completes before the next dequant writes to it.
    auto dequant_expert = [&](const Tensor& packed, GGMLQuantType qtype,
                              int expert_idx) -> Tensor {
        int64_t rows = packed.shape[1];
        int64_t cols = packed.shape[2];
        size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
        size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
        size_t total_raw = static_cast<size_t>(packed.shape[0]) * expert_raw;
        size_t offset = static_cast<size_t>(expert_idx) * expert_raw;

        // Bounds check: verify offset + expert_raw <= total allocated
        if (offset + expert_raw > total_raw) {
            IMP_LOG_ERROR("dequant_expert: OOB! expert %d offset=%zu + raw=%zu > total=%zu "
                    "(packed shape [%ld,%ld,%ld] qtype=%u)",
                    expert_idx, offset, expert_raw, total_raw,
                    (long)packed.shape[0], (long)packed.shape[1], (long)packed.shape[2],
                    (unsigned)qtype);
            return Tensor();
        }

        // Check dequant buffer is large enough
        size_t dequant_needed = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
        if (dequant_needed > moe_.dequant_buf_size) {
            IMP_LOG_ERROR("dequant_expert: dequant buffer too small! "
                    "need=%zu have=%zu (rows=%ld cols=%ld)",
                    dequant_needed, moe_.dequant_buf_size, (long)rows, (long)cols);
            return Tensor();
        }

        const char* src;
        if (!packed.on_device) {
            // Expert weights offloaded to host — try LRU cache first, then staging buffer.
            const char* host_ptr = static_cast<const char*>(packed.data) + offset;
            if (expert_cache_.n_slots_ > 0) {
                ExpertCacheKey ck{packed.data, expert_idx};
                void* cached = expert_cache_.get_or_load(ck, host_ptr, expert_raw, stream);
                src = static_cast<const char*>(cached);
            } else if (moe_.raw_staging_buf) {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.raw_staging_buf, host_ptr, expert_raw,
                                cudaMemcpyHostToDevice, stream));
                src = static_cast<const char*>(moe_.raw_staging_buf);
            } else {
                IMP_LOG_ERROR("dequant_expert: no staging buffer for host expert %d", expert_idx);
                return Tensor();
            }
        } else {
            src = static_cast<const char*>(packed.data) + offset;
        }

        char* dst = static_cast<char*>(moe_.dequant_buf);  // always slot 0

        dequant_gpu(src, dst, qtype, static_cast<int>(rows), static_cast<int>(cols), stream);

        int64_t shape[2] = {rows, cols};
        return Tensor(dst, DType::FP16, 2, shape, true);
    };

    // Helper: try fused quantized GEMV for count=1 decode (dequant+dot in one kernel),
    // else fall back to dequant_expert + cuBLAS gemm.
    // For host-resident experts: H2D to staging buffer, then fused GEMV on staging —
    // eliminates separate dequant_gpu + cuBLAS gemm overhead.
    auto expert_gemm = [&](const Tensor& a, Tensor& c,
                            const Tensor& packed, GGMLQuantType qtype,
                            const std::vector<Tensor>& fallback, int eidx) {
        // NVFP4 prequant path: native NVFP4 GEMV (any batch size)
        if (!fallback.empty() &&
            static_cast<size_t>(eidx) < fallback.size() &&
            wcache_.nvfp4.count(fallback[eidx].data)) {
            const auto& nw = wcache_.nvfp4.at(fallback[eidx].data);
            int N_dim = static_cast<int>(nw.N);
            int K_dim = static_cast<int>(nw.K);
            int M = static_cast<int>(a.shape[0]);
            if (M == 1) {
                // Single-token decode: direct NVFP4 GEMV
                gemv_nvfp4_kpar(nw, static_cast<const half*>(a.data),
                               static_cast<half*>(c.data), N_dim, K_dim, stream);
            } else {
                // Multi-token: per-row GEMV (each row is one token)
                for (int r = 0; r < M; r++) {
                    const half* a_row = static_cast<const half*>(a.data) + r * K_dim;
                    half* c_row = static_cast<half*>(c.data) + r * N_dim;
                    gemv_nvfp4_kpar(nw, a_row, c_row, N_dim, K_dim, stream);
                }
            }
            return;
        }
        if (a.shape[0] == 1 && use_packed_dequant &&
            compute_dtype_ == DType::FP16 &&
            (qtype == GGMLQuantType::Q6_K || qtype == GGMLQuantType::Q8_0)) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t rb = ggml_quant_row_bytes(qtype, cols);
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
                    w = expert_cache_.get_or_load(ck, host_ptr, expert_raw, stream);
                } else if (moe_.raw_staging_buf && expert_raw <= moe_.raw_staging_size) {
                    cudaMemcpyAsync(moe_.raw_staging_buf, host_ptr, expert_raw,
                                    cudaMemcpyHostToDevice, stream);
                    w = moe_.raw_staging_buf;
                }
            }

            if (w) {
                auto fn = (qtype == GGMLQuantType::Q6_K) ? gemv_q6k : gemv_q8_0;
                fn(w, static_cast<const half*>(a.data), static_cast<half*>(c.data),
                   static_cast<int>(rows), static_cast<int>(cols), stream);
                return;
            }
        }
        // Fallback: separate dequant + cuBLAS GEMM
        {
            Tensor b = use_packed_dequant
                ? dequant_expert(packed, qtype, eidx)
                : fallback[eidx];
            if (!b.data) return;  // dequant_expert failed (OOB or buffer too small)
            gemm(a, b, c, 1.0f, 0.0f, stream);
        }
    };

        char* gathered_base     = static_cast<char*>(moe_.gathered.data);
        char* expert_gate_base  = static_cast<char*>(moe_.expert_gate.data);
        char* expert_up_base    = static_cast<char*>(moe_.expert_up.data);
        char* expert_swiglu_base= static_cast<char*>(moe_.expert_swiglu.data);
        char* expert_down_base  = static_cast<char*>(moe_.expert_down.data);

        // Helper: get FP16 expert weight pointer from pre-dequant cache or unpacked weights.
        auto get_fp16_expert_ptr = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        int eidx) -> const void* {
            if (packed.data && wcache_.fp16.count(packed.data)) {
                const Tensor& cached = wcache_.fp16.at(packed.data);
                int64_t rows = packed.shape[1];
                int64_t cols = packed.shape[2];
                size_t expert_offset = static_cast<size_t>(eidx) * rows * cols * sizeof(half);
                return static_cast<const char*>(cached.data) + expert_offset;
            }
            if (!fallback.empty() && static_cast<size_t>(eidx) < fallback.size() &&
                fallback[eidx].data && fallback[eidx].dtype == DType::FP16 &&
                fallback[eidx].on_device) {
                return fallback[eidx].data;
            }
            return nullptr;
        };

        // Helper: batch dequant all experts + single grouped GEMM.
        // Dequants all experts to FP16, then runs a single batched GEMM.
        // CUTLASS 2.x GemmGrouped provides lower launch overhead than cuBLAS.
        auto chunked_dequant_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        const char* a_base, char* c_base,
                                        int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);
            size_t expert_raw_sz = static_cast<size_t>(rows)
                                   * ggml_quant_row_bytes(qtype, cols);

            if (!moe_.batch_dequant_buf || expert_fp16_sz == 0) {
                // No buffer — serial fallback
                for (int e = 0; e < ne; ++e) {
                    int start = h_offsets[e];
                    int count = h_offsets[e + 1] - start;
                    if (count == 0) continue;
                    int64_t count64 = static_cast<int64_t>(count);
                    int64_t a_shape[2] = {count64, static_cast<int64_t>(K_dim)};
                    Tensor a_view(const_cast<void*>(static_cast<const void*>(
                                  a_base + static_cast<size_t>(start) * K_dim * es)),
                                  compute_dtype_, 2, a_shape, true);
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(N_dim)};
                    Tensor c_view(c_base + static_cast<size_t>(start) * N_dim * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, packed, qtype, fallback, e);
                }
                return;
            }

            const uint8_t* raw_base = static_cast<const uint8_t*>(packed.data);
            char* buf = static_cast<char*>(moe_.batch_dequant_buf);

            // Dequant all experts in one batch, then single GEMM.
            // With pp=512 and top_k=8, nearly all 128 experts are active, so
            // dequanting all at once is optimal (one big bandwidth-saturating kernel).
            dequant_gpu(raw_base, buf, qtype,
                        ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

            // Use cublasGemmGroupedBatchedEx — single call for all experts.
            // We already have h_offsets from D2H sync, so no need for
            // gemm_moe_device_grouped (which does its own D2H sync + 128
            // individual cublasLtMatmul calls).
            gemm_moe_batched(a_base, c_base,
                             h_offsets.data(), b_ptrs.data(),
                             K_dim, N_dim, DType::FP16, ne, stream,
                             moe_.d_work_ptrs);
        };

        // Determine which path to use:
        // 1. Pre-cached FP16 path: all experts in wcache_.fp16 (fastest, no dequant)
        // 2. Dequant-then-batch path: packed experts on device + batch buffer available
        // 3. Serial path: fallback (one expert at a time)
        // Note: fused Q6K dp4a path is handled above (before the D2H sync).

        bool has_precached_up = (ly.expert_up_packed.data && wcache_.fp16.count(ly.expert_up_packed.data));
        bool can_dequant_batch = (moe_.batch_dequant_buf != nullptr &&
                                   ly.expert_up_packed.data != nullptr &&
                                   ly.expert_up_packed.on_device &&
                                   dequant_gpu_supported(ly.expert_up_qtype));

        if (has_precached_up) {
            // Pre-cached FP16 path — all expert packs in wcache_.fp16
            // ===== PRE-CACHED FP16 BATCHED GEMM PATH =====
            std::vector<const void*> gate_w_ptrs(ne, nullptr);
            std::vector<const void*> up_w_ptrs(ne, nullptr);
            std::vector<const void*> down_w_ptrs(ne, nullptr);

            for (int e = 0; e < ne; e++) {
                up_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_up_packed, ly.expert_up_qtype,
                                                     ly.expert_w_up, e);
                if (!non_gated_experts)
                    gate_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_gate_packed, ly.expert_gate_qtype,
                                                           ly.expert_w_gate, e);
                down_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_down_packed, ly.expert_down_qtype,
                                                       ly.expert_w_down, e);
            }

            if (!non_gated_experts)
                gemm_moe_batched(gathered_base, expert_gate_base,
                                  h_offsets.data(), gate_w_ptrs.data(),
                                  d, eff, DType::FP16, ne, stream, moe_.d_work_ptrs);
            gemm_moe_batched(gathered_base, expert_up_base,
                              h_offsets.data(), up_w_ptrs.data(),
                              d, eff, DType::FP16, ne, stream, moe_.d_work_ptrs);

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts,
                                    expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

            {
                char* batch_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                gemm_moe_batched(batch_down_act, expert_down_base,
                                  h_offsets.data(), down_w_ptrs.data(),
                                  eff, d, DType::FP16, ne, stream, moe_.d_work_ptrs);
            }

        } else if (can_dequant_batch) {
            // ===== BATCH DEQUANT + GROUPED GEMM =====
            // Dequant all experts to FP16, then single grouped GEMM via CUTLASS.

            if (!non_gated_experts)
                chunked_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                                     ly.expert_w_gate, gathered_base, expert_gate_base, d, eff);
            chunked_dequant_gemm(ly.expert_up_packed, ly.expert_up_qtype,
                                 ly.expert_w_up, gathered_base, expert_up_base, d, eff);

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts,
                                    expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

            {
                char* dequant_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                chunked_dequant_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                                     ly.expert_w_down, dequant_down_act, expert_down_base, eff, d);
            }

        } else {
            // ===== SERIAL PATH (fallback) =====
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, a_shape, true);

                if (!non_gated_experts) {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_gate_packed,
                                ly.expert_gate_qtype, ly.expert_w_gate, e);
                }

                {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_up_packed,
                                ly.expert_up_qtype, ly.expert_w_up, e);
                }
            }

            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                    moe_.expert_swiglu.data, non_gated_experts,
                                    expanded, eff, compute_dtype_, cfg.ffn_activation, stream);

            // Down projection activation source: up buffer for non-gated (relu² in-place),
            // swiglu buffer for gated.
            char* down_act_base = non_gated_experts ? expert_up_base : expert_swiglu_base;
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor a_view(down_act_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, a_shape, true);
                int64_t c_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor c_view(expert_down_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, c_shape, true);
                expert_gemm(a_view, c_view, ly.expert_down_packed,
                            ly.expert_down_qtype, ly.expert_w_down, e);
            }
        }
    }
    } // legacy inner scope
    } // else of can_fp16/fp8_batch/legacy
    } // FP8/FP16 prefill scope
    } // else branch of can_fused_q6k + fused Q6_K scope

    // 7+8. Scatter expert outputs back to token positions.
    //      Fused path: token-centric scatter + FP16 convert (+ residual if no shared expert).
    //      Fallback: atomicAdd scatter + FP32->FP16 convert.
    {
        bool has_shared_expert = (ly.w_up_shared.data != nullptr);
        if (routing.token_to_expanded && compute_dtype_ == DType::FP16) {
            // Fused token-centric scatter: no atomics, no FP32 intermediate buffer.
            // If no shared expert, also fuse residual add.
            const void* res_ptr = (!has_shared_expert && !residual_fused) ? r.data : nullptr;
            moe_scatter_fused_residual(
                moe_.expert_down.data, routing.token_to_expanded,
                static_cast<const float*>(routing.expert_weights.data),
                res_ptr, h.data,
                n, d, top_k, stream);
            if (!has_shared_expert) residual_fused = true;
        } else {
            // Fallback: atomicAdd scatter into FP32 buffer, then convert
            int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded),
                                            static_cast<int64_t>(d)};
            Tensor expert_down_view(moe_.expert_down.data, compute_dtype_,
                                    2, expert_out_shape, true);
            Tensor scatter_out = slice_rows(moe_.scatter_out, n);
            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(scatter_out.data, 0,
                            static_cast<size_t>(n) * d * sizeof(float), stream));
            moe_scatter(expert_down_view, routing, scatter_out, stream);

            int64_t numel = static_cast<int64_t>(n) * d;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == DType::FP16) {
                fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const float*>(moe_.scatter_out.data),
                    static_cast<half*>(h.data),
                    numel);
            } else {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h.data, moe_.scatter_out.data,
                                static_cast<size_t>(numel) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

moe_after_experts:
    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    // Gemma 4: sanitize inf/NaN in MoE scatter output before post-norm.
    if (cfg.arch == ModelArch::GEMMA4) {
        sanitize_fp16(static_cast<__half*>(h.data),
                      static_cast<int64_t>(n) * d, stream);
    }

    // Gemma 4: apply post_ffw_norm_2 on the MoE branch output (h) BEFORE shared adds.
    // This matches the parallel-branch structure: rms_norm(moe_out, post_ffw_norm_2).
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_2.data != nullptr &&
        getenv("IMP_G4_NO_POST_FFW_2") == nullptr) {
        rmsnorm(h, ly.ffn_post_norm_2, h, eps, stream, norm_w_off_);
    }

    // Gemma 4: re-derive `no` for the shared MLP from the saved residual
    // (which still holds the original hidden state) using ffn_norm — the MoE
    // branch consumed `no` produced from pre_ffw_norm_2 above.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr &&
        ly.w_up_shared.data != nullptr && ly.ffn_norm.data != nullptr) {
        rmsnorm(r, ly.ffn_norm, no, eps, stream, norm_w_off_);
    }

    // Shared expert: enabled by default. Gemma 4: requires post_ffw_norm_1 to be uploaded.
    if (ly.w_up_shared.data != nullptr) {
        int eff_shared = static_cast<int>(ly.w_up_shared.shape[0]);
        bool shared_gated = (ly.w_gate_shared.data != nullptr);

        // Reuse moe_.expert_gate, moe_.expert_up, moe_.expert_swiglu as scratch.
        int64_t sh_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(eff_shared)};
        Tensor sh_up(moe_.expert_up.data, compute_dtype_, 2, sh_shape, true);
        Tensor sh_swiglu(moe_.expert_swiglu.data, compute_dtype_, 2, sh_shape, true);

        // Down projection output: [n, d_model]. Reuse moe_.expert_down.
        int64_t sh_down_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
        Tensor sh_down(moe_.expert_down.data, compute_dtype_, 2, sh_down_shape, true);

        // Up projection (dp4a MMVQ for decode)
        {
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            const auto* nvfp4_ptr = (wcache_.nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.nvfp4;
            const auto* ct4_ptr = (wcache_.cutlass_nvfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_nvfp4;
            const auto* mx4p = (wcache_.cutlass_mxfp4.empty() || cur_force_fp16_) ? nullptr : &wcache_.cutlass_mxfp4;
            gemm_dispatch(no, ly.w_up_shared, Tensor(), ly.w_up_shared_qtype,
                          sh_up, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                          (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                          nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);

            if (shared_gated) {
                // Gated: gate + SwiGLU
                Tensor sh_gate(moe_.expert_gate.data, compute_dtype_, 2, sh_shape, true);
                gemm_dispatch(no, ly.w_gate_shared, Tensor(), ly.w_gate_shared_qtype,
                              sh_gate, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                              (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                              qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                              nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
                if (cfg.ffn_activation == FFNActivation::GEGLU)
                    geglu(sh_gate, sh_up, sh_swiglu, stream);
                else
                    swiglu(sh_gate, sh_up, sh_swiglu, stream);
            } else {
                // Non-gated: relu^2(up) in-place [Nemotron-H uses squared ReLU]
                relu_sqr_inplace(sh_up, stream);
            }

            // Down projection (reads from sh_up for non-gated since relu² was in-place)
            Tensor& sh_act = shared_gated ? sh_swiglu : sh_up;
            gemm_dispatch(sh_act, ly.w_down_shared, Tensor(), ly.w_down_shared_qtype,
                          sh_down, qscratch_.dequant, stream, q8, qscratch_.d8_buf, &wcache_.fp16,
                          (wcache_.use_fp8 && !cur_force_fp16_) ? &wcache_.fp8 : nullptr, qscratch_.fp8_act, qscratch_.d_act_scale,
                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                          nvfp4_ptr, ct4_ptr, qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, qscratch_.cutlass_workspace, qscratch_.cutlass_workspace_size,
                                  mx4p, qscratch_.mxfp4_act_sf, qscratch_.mxfp4_workspace, qscratch_.mxfp4_workspace_size);
        }

        // Gemma 4: shared MLP can overflow FP16 at deep layers. Sanitize inf/NaN
        // to zero before the post-norm so rmsnorm doesn't produce all-zero output.
        if (cfg.arch == ModelArch::GEMMA4) {
            sanitize_fp16(static_cast<__half*>(sh_down.data),
                          static_cast<int64_t>(n) * d, stream);
        }
        // Gemma 4: apply post_ffw_norm_1 on shared MLP output (sh_down).
        if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_1.data != nullptr &&
            getenv("IMP_G4_NO_POST_FFW_1") == nullptr) {
            rmsnorm(sh_down, ly.ffn_post_norm_1, sh_down, eps, stream, norm_w_off_);
        }

        // Add shared expert output to hidden (which already has routed expert output)
        elementwise_add(h, sh_down, stream);
    }

    // Gemma 4: apply post_ffn_norm (combined post-norm) BEFORE residual add.
    if (cfg.arch == ModelArch::GEMMA4 && ly.post_ffn_norm.data != nullptr) {
        rmsnorm(h, ly.post_ffn_norm, h, eps, stream, norm_w_off_);
    }

    // 9. Residual connection: hidden += residual
    //    Skipped when decode fast path already fused residual into weighted_sum.
    if (!residual_fused) {
        elementwise_add(h, r, stream);
    }

    // 10. Free routing result tensors only if allocated by moe_topk_gating.
    //     When using pre-allocated buffers, memory belongs to moe_.routing_buffers.
    if (routing.owns_memory) {
        IMP_CUDA_CHECK_LOG(cudaFree(routing.expert_indices.data));
        IMP_CUDA_CHECK_LOG(cudaFree(routing.expert_weights.data));
        IMP_CUDA_CHECK_LOG(cudaFree(routing.sorted_token_ids.data));
        IMP_CUDA_CHECK_LOG(cudaFree(routing.expert_offsets.data));
    }
}

} // namespace imp

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
#include "graph/gemm_context.h"
#include "graph/executor_debug.h"
#include "runtime/config.h"
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
#include "graph/executor_forward_moe_cutlass3x.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
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

namespace imp {

// Multiply each routing weight by the scale of its selected expert.
// Used by Gemma 4 to apply per-expert output scale before the routing sum.
__global__ void moe_apply_per_expert_scale_kernel(
    float* __restrict__ weights,          // [n_weights = n_tokens * top_k]
    const int32_t* __restrict__ indices,  // [n_weights]
    const __half* __restrict__ scales,    // [n_experts]
    int n_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_weights)
        return;
    int eid = indices[idx];
    float s = __half2float(scales[eid]);
    weights[idx] *= s;
}

// Replace +/-Inf with 0 in an FP16 tensor (in-place).
// Used to sanitize FP16 GEMM outputs that overflow (e.g. Gemma 4 shared MLP at deep layers).
__global__ void sanitize_fp16_kernel(__half* __restrict__ data, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    __half v = data[idx];
    // __hisinf returns ±1 for ±inf, 0 otherwise. __hisnan returns non-zero for NaN.
    if (__hisinf(v) != 0 || __hisnan(v)) {
        data[idx] = __float2half(0.0f);
    }
}

static void sanitize_fp16(__half* data, int64_t n, cudaStream_t stream) {
    if (n <= 0)
        return;
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
static bool can_decode_fast(int n, const Tensor& expert_up_packed, QType up_qtype, void* dequant_buf,
                            QType compute_dtype) {
    return (n == 1 && expert_up_packed.data != nullptr && dequant_buf != nullptr &&
            compute_dtype == QType::F16 && expert_up_packed.on_device &&
            (up_qtype == QType::Q6_K || up_qtype == QType::Q8_0 || up_qtype == QType::Q4_0 ||
             up_qtype == QType::Q4_K || up_qtype == QType::Q5_K || up_qtype == QType::Q2_K ||
             up_qtype == QType::Q3_K || up_qtype == QType::Q5_1 || up_qtype == QType::NVFP4));
}

// ---------------------------------------------------------------------------
// Expert activation dispatch: SwiGLU for gated experts, ReLU^2 for non-gated.
// Operates on the expanded-layout buffers (shape [rows, eff]).
// ---------------------------------------------------------------------------
static void apply_expert_activation(void* gate_data, void* up_data, void* swiglu_data, bool non_gated,
                                    int64_t rows, int64_t eff, QType compute_dtype, FFNActivation act_type,
                                    cudaStream_t stream) {
    int64_t act_shape[2] = {rows, eff};
    if (non_gated) {
        Tensor up_t(up_data, compute_dtype, 2, act_shape, true);
        relu_sqr_inplace(up_t, stream);
    } else {
        Tensor g(gate_data, compute_dtype, 2, act_shape, true);
        Tensor u(up_data, compute_dtype, 2, act_shape, true);
        Tensor a(swiglu_data, compute_dtype, 2, act_shape, true);
        if (act_type == FFNActivation::GEGLU)
            geglu(g, u, a, stream);
        else
            swiglu(g, u, a, stream);
    }
}

// ---------------------------------------------------------------------------
// Compute expert stride (bytes between experts in a packed tensor).
// ---------------------------------------------------------------------------
static size_t expert_stride(const Tensor& packed, QType qtype) {
    int64_t rows = packed.shape[1];
    int64_t cols = packed.shape[2];
    return static_cast<size_t>(rows) * qtype_row_bytes(qtype, cols);
}

}  // anonymous namespace

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    // DIAGNOSTIC (Phase 2 Item 2 follow-up): zero MoE workspace buffers so any
    // legacy-serial-fallback uninit reads become deterministic zero reads.
    // Set IMP_MOE_ZERO_WORKSPACE=1 to enable. Cheap (~1 MiB total memset).
    if (RuntimeConfig::current().moe.zero_workspace) {
        cudaMemsetAsync(moe_.expert_gate.data, 0, moe_.expert_gate.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_up.data, 0, moe_.expert_up.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_swiglu.data, 0, moe_.expert_swiglu.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_down.data, 0, moe_.expert_down.nbytes(), stream);
        cudaMemsetAsync(moe_.gathered.data, 0, moe_.gathered.nbytes(), stream);
    }

    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    int n = cur_n_tokens_;
    int d = cfg.d_model;
    int ne = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff = max_expert_eff_;
    float eps = cfg.rms_norm_eps;
    size_t es = dtype_size(compute_dtype_);
    int expanded = n * top_k;

    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    bool residual_fused = false;  // set true if decode fast path fuses residual add

    // 1. Save residual (skip if decode fast path will handle it —
    //    h.data is never written before the final weighted_sum_residual).
    // Gemma 4: parallel branches — MoE experts use rmsnorm(h, pre_ffw_norm_2),
    // shared MLP uses rmsnorm(h, ffn_norm). Pick MoE-side norm here; the shared
    // branch recomputes its own norm later (reading from the saved residual).
    // Qwen3.5/3.6 GGUFs store FFN input norm as `post_attention_norm` (no
    // dedicated `ffn_norm`); match the fallback chain used in run_ffn. Without
    // this, MoE reuses the pre-attention norm and the residual stream explodes
    // (observed on Qwen3.6-35B-A3B GDN+MoE: logits L2=100k, garbage output).
    const Tensor& norm_w = (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr)
                               ? ly.ffn_pre_norm_2
                           : (ly.ffn_norm.data != nullptr)       ? ly.ffn_norm
                           : (ly.post_attn_norm.data != nullptr) ? ly.post_attn_norm
                                                                 : ly.attn_norm;

    // Pre-check: does NVFP4 MoE cache cover all expert tensors for this layer?
    // If so, the NVFP4 path doesn't need Q8_1 quantization (takes FP16 directly).
    bool nvfp4_covers_layer = false;
    if (n == 1 && compute_dtype_ == QType::F16) {
        bool has_up = (ly.nvfp4_moe_up_ptr != nullptr);
        bool has_down = (ly.nvfp4_moe_down_ptr != nullptr);
        nvfp4_covers_layer = has_up && has_down;
        if (nvfp4_covers_layer && ly.expert_gate_packed.data != nullptr) {
            nvfp4_covers_layer = (ly.nvfp4_moe_gate_ptr != nullptr);
        }
    }

    // Pre-check decode fast path (same logic as will_decode_fast below)
    bool will_skip_residual_copy = can_decode_fast(n, ly.expert_up_packed, ly.expert_up_packed.qtype,
                                                   moe_.dequant_buf, compute_dtype_) &&
                                   ly.w_up_shared.data ==
                                       nullptr;  // must not have shared expert for full residual fusion

    if (!will_skip_residual_copy) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }
    // Fused RMSNorm + Q8_1: skip for NVFP4-covered layers (NVFP4 takes FP16 directly)
    // Gemma-4 with FP32 accum: compute norm from FP32 residual, then quantize to Q8_1
    // separately. The fused kernel reads FP16 h which loses ~0.03% per element that
    // compounds catastrophically through the 128-expert top-8 MoE routing.
    bool gemma4_fp32_norm = (cfg.arch == ModelArch::GEMMA4 && fp32_accum_buf_ != nullptr);
    // When FP32 residual accumulator is active AND post_ffn_norm exists, defer the
    // residual add to rmsnorm_fp32_accum_to_fp16_kernel (which keeps fp32_hidden_ in
    // sync + applies overflow scaling). Without this, moe_weighted_sum_residual
    // adds residual in FP16 and the shadow goes stale — measured ~7% drift at L3
    // that compounds to 260% by L29 vs llama.cpp (docs/gemma4_layer_diff.md).
    bool moe_use_fp32_residual = (cfg.arch == ModelArch::GEMMA4 && fp32_accum_buf_ != nullptr &&
                                  ly.post_ffn_norm.data != nullptr);
    // Diagnostic: keep MoE down-projection output in FP32 (no FP16 truncation
    // at GEMM output) to isolate precision drift. Allocated below in the FP16
    // batch path; freed at moe_after_experts. Other prefill paths ignore this.
    const bool fp32_down_active = (cfg.arch == ModelArch::GEMMA4 &&
                                   RuntimeConfig::current().gemma4.fp32_expert_down);
    void* fp32_down_buf = nullptr;
    bool moe_fused_norm_q8 = (n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                              h.qtype == QType::F16 && !nvfp4_covers_layer && !gemma4_fp32_norm);
    if (moe_fused_norm_q8) {
        // Fused: RMSNorm + Q8_1 (also writes FP16 norm_out for gate logits)
        rmsnorm_quantize_q8_1(static_cast<const half*>(h.data), static_cast<const half*>(norm_w.data),
                              static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf,
                              static_cast<half*>(no.data), d, eps, stream, norm_w_off_);
    } else {
        // Gemma-4 FP32 accum: read FP32 residual directly to avoid FP16 round-trip.
        if (gemma4_fp32_norm) {
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            rmsnorm_fp32_to_fp16(fp32_h, norm_w, no, eps, stream, norm_w_off_);
            // Populate Q8_1 buffer from FP16 norm_out for dp4a decode path
            if (n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr) {
                quantize_fp16_to_q8_1(static_cast<const half*>(no.data),
                                      static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, d,
                                      stream);
            }
        } else {
            rmsnorm(h, norm_w, no, eps, stream, norm_w_off_);
        }
    }

    // 3. Gate logits + top-k routing
    Tensor router_in = no;
    bool fp32_gate_logits_ready = false;  // true when FP32 gate logits computed directly
    // Gemma-4 custom router: logits = gate_inp @ (rmsnorm_noweight(h) * (1/sqrt(d)) * gate_inp_scale)
    // This matches llama.cpp's gemma4-iswa.cpp:151-155.
    // The standard path (router_in = rmsnorm(h, ffn_pre_norm_2)) uses the WRONG norm weight
    // and produces ~2x too small router logits, causing wrong expert selection.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_gate_inp_scale.data != nullptr) {
        // Gemma-4 custom router: logits = gate_inp @ (rmsnorm_noweight(h) * scale * (1/sqrt(d)))
        // Keep router_in in FP32 to prevent precision loss that causes routing instability
        // at later layers (L29). The FP16 intermediate loses enough precision to change
        // expert selection in the 128-expert top-8 MoE.
        if (fp32_accum_buf_ != nullptr && n == 1) {
            // FP32 router path: rmsnorm(fp32_h, gate_inp_scale) * 1/sqrt(d) → FP32
            // Stays in FP32 all the way through gate GEMV (no FP16 truncation).
            float* fp32_router = static_cast<float*>(moe_.scatter_out.data);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            rmsnorm_fp32_to_fp32(fp32_h, ly.ffn_gate_inp_scale, fp32_router, n, d, eps, stream, 0.0f);
            {
                int64_t total = static_cast<int64_t>(n) * d;
                int thr = 256;
                int blk = static_cast<int>((total + thr - 1) / thr);
                scale_fp32_kernel<<<blk, thr, 0, stream>>>(fp32_router, inv_sqrt_d, total);
            }
            // Gate GEMV directly from FP32 router → FP32 logits (no FP16 truncation).
            // Writes to moe_.gate_logits — same buffer the topk gating reads from.
            {
                Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, n);
                gemv_gate_fp32_fp32input(static_cast<const half*>(ly.moe_gate.data), fp32_router,
                                         static_cast<float*>(gate_logits_f32.data), ne, d, stream);
            }
            fp32_gate_logits_ready = true;
        } else {
            // FP16 fallback (prefill or non-FP32-accum)
            int64_t ri_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
            router_in = Tensor(moe_.scatter_out.data, compute_dtype_, 2, ri_shape, true);
            if (fp32_accum_buf_ != nullptr) {
                Tensor fp32_h = view_tokens(fp32_hidden_, n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.ffn_gate_inp_scale, router_in, eps, stream, 0.0f);
            } else {
                rmsnorm(h, ly.ffn_gate_inp_scale, router_in, eps, stream, 0.0f);
            }
            int64_t total_elems = static_cast<int64_t>(n) * d;
            int threads_s = 256;
            int blocks_s = static_cast<int>((total_elems / 2 + threads_s - 1) / threads_s);
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            scale_fp16_kernel<<<blocks_s, threads_s, 0, stream>>>(static_cast<half*>(router_in.data),
                                                                  __float2half(inv_sqrt_d), total_elems);
        }
    }

    const void* router_bias_ptr = ly.moe_router_bias.data;
    // Gemma-4: moe_router_bias may hold ffn_down_exps.scale (per-expert output
    // multiplier) due to GGUF name collision — NOT a router bias. Don't use it.
    if (cfg.arch == ModelArch::GEMMA4 && router_bias_ptr != nullptr) {
        if (layer == 0)
            IMP_LOG_INFO("Gemma 4: ignoring moe_router_bias (likely ffn_down_exps.scale, not router bias)");
        router_bias_ptr = nullptr;
    }
    bool use_sigmoid = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;

    QType up_qtype = ly.expert_up_packed.qtype;
    bool will_decode_fast = can_decode_fast(n, ly.expert_up_packed, up_qtype, moe_.dequant_buf,
                                            compute_dtype_);
    // Gemma 4: dp4a decode fast path ENABLED by default. dp4a matches llama's
    // Q4_K×Q8_1 accumulation for MoE experts, preventing the routing drift that
    // occurs with FP16 dequant+cuBLAS. Set IMP_G4_NO_DECODE_FAST=1 to disable.
    if (cfg.arch == ModelArch::GEMMA4 && RuntimeConfig::current().gemma4.no_decode_fast) {
        will_decode_fast = false;
    }

    MoeRoutingResult routing;

    // Fused gate GEMV + topk is only beneficial when n_experts fits in the
    // number of warps (8). For high expert counts (e.g., 128 in Qwen3-Coder),
    // the separate gemv_gate_fp32 (128 parallel blocks) is much faster than
    // serializing 128/8=16 experts per warp in a single block.
    constexpr int kMaxFusedExperts = 8;
    if (fp32_gate_logits_ready) {
        // Gate logits already computed by FP32 router path above — skip to topk.
        Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, n);
        // Diagnostic: dump gate logits pre-topk when IMP_DUMP_LOGITS is set.
        // Compares imp's softmax-input against llama.cpp's ffn_moe_probs-N.
        if (const std::string& dl = RuntimeConfig::current().diagnostics.dump_logits_dir; !dl.empty()) {
            bool dump_all = (dl == "all");
            if (layer == 29 || dump_all) {
                int last_tok = n - 1;
                std::vector<float> h_logits(ne);
                cudaStreamSynchronize(stream);
                cudaMemcpy(h_logits.data(), static_cast<const float*>(gate_logits_f32.data) + last_tok * ne,
                           ne * sizeof(float), cudaMemcpyDeviceToHost);
                // Find top-8 by value
                std::vector<std::pair<float, int>> sorted;
                sorted.reserve(ne);
                for (int i = 0; i < ne; ++i)
                    sorted.emplace_back(h_logits[i], i);
                std::partial_sort(sorted.begin(), sorted.begin() + 8, sorted.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
                fprintf(stderr, "[LOGITS] L%02d tok=%d top8_by_value: ", layer, last_tok);
                for (int i = 0; i < 8; ++i)
                    fprintf(stderr, "[e=%d v=%.4f] ", sorted[i].second, sorted[i].first);
                fprintf(stderr, "\n");
            }
        }
        if (moe_.routing_buffers.pool) {
            moe_topk_gating(gate_logits_f32, top_k, moe_.routing_buffers, routing, stream, use_sigmoid,
                            norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights,
                            router_bias_ptr);
        }
    } else if (ne <= kMaxFusedExperts && n == 1 && compute_dtype_ == QType::F16 &&
               ly.moe_gate.qtype == QType::F16 && moe_.routing_buffers.pool && will_decode_fast) {
        // Fused: gate GEMV + softmax/sigmoid + top-k in one kernel (1 launch)
        moe_gate_topk_fused(static_cast<const half*>(ly.moe_gate.data),
                            static_cast<const half*>(router_in.data), ne, d, top_k, moe_.routing_buffers,
                            routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
    } else {
        // Separate: gate GEMV → intermediate logits → topk gating
        Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, n);

        if (n == 1 && compute_dtype_ == QType::F16 && ly.moe_gate.qtype == QType::F16) {
            gemv_gate_fp32(static_cast<const half*>(ly.moe_gate.data),
                           static_cast<const half*>(router_in.data),
                           static_cast<float*>(gate_logits_f32.data), ne, d, stream);
        } else {
            int64_t gl_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(ne)};
            Tensor gate_logits_tmp(moe_.gathered.data, compute_dtype_, 2, gl_shape, true);
            gemm(router_in, ly.moe_gate, gate_logits_tmp, 1.0f, 0.0f, stream);

            int64_t numel = static_cast<int64_t>(n) * ne;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == QType::F16) {
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(
                                                                        gate_logits_tmp.data),
                                                                    static_cast<float*>(gate_logits_f32.data),
                                                                    numel);
            } else {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                                                   static_cast<size_t>(numel) * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream));
            }
        }

        if (debug_forward_enabled() && layer == 0) {
            // Dump router logits for last token
            std::vector<float> rl(ne);
            int last_tok = n - 1;
            cudaMemcpy(rl.data(), static_cast<const float*>(gate_logits_f32.data) + last_tok * ne,
                       ne * sizeof(float), cudaMemcpyDeviceToHost);
            double rsum = 0;
            for (auto v : rl)
                rsum += v;
            double rss = 0;
            for (auto v : rl)
                rss += v * v;
            fprintf(stderr, "[DEBUG_FWD] L0_router_logits[%d]: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f\n",
                    last_tok, rsum, std::sqrt(rss), rl[0], rl[1], rl[2]);
        }
        // IMP_DUMP_LOGITS=all: dump top-8-by-value of gate logits per layer.
        if (const std::string& dl = RuntimeConfig::current().diagnostics.dump_logits_dir; !dl.empty()) {
            bool dump_all = (dl == "all");
            if (layer == 29 || dump_all) {
                int last_tok = n - 1;
                std::vector<float> h_logits(ne);
                cudaStreamSynchronize(stream);
                cudaMemcpy(h_logits.data(), static_cast<const float*>(gate_logits_f32.data) + last_tok * ne,
                           ne * sizeof(float), cudaMemcpyDeviceToHost);
                std::vector<std::pair<float, int>> sorted;
                sorted.reserve(ne);
                for (int i = 0; i < ne; ++i)
                    sorted.emplace_back(h_logits[i], i);
                std::partial_sort(sorted.begin(), sorted.begin() + 8, sorted.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
                fprintf(stderr, "[LOGITS] L%02d tok=%d top8_by_value: ", layer, last_tok);
                for (int i = 0; i < 8; ++i)
                    fprintf(stderr, "[e=%d v=%.4f] ", sorted[i].second, sorted[i].first);
                fprintf(stderr, "\n");
            }
        }
        if (moe_.routing_buffers.pool) {
            moe_topk_gating(gate_logits_f32, top_k, moe_.routing_buffers, routing, stream, use_sigmoid,
                            norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights,
                            router_bias_ptr);
        }
    }

    // Dump routing for last token. [diagnostics] dump_routing_dir = "<path>"
    // dumps layer 0 only; dump_routing_dir = "all" dumps every layer.
    if (const std::string& drv = RuntimeConfig::current().diagnostics.dump_routing_dir; !drv.empty()) {
        bool dump_all = (drv == "all");
        if (layer == 0 || dump_all) {
            int last_tok = n - 1;
            std::vector<int32_t> h_idx(top_k);
            std::vector<float> h_wts(top_k);
            cudaMemcpy(h_idx.data(),
                       static_cast<const int32_t*>(routing.expert_indices.data) + last_tok * top_k,
                       top_k * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_wts.data(),
                       static_cast<const float*>(routing.expert_weights.data) + last_tok * top_k,
                       top_k * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr,
                    "[ROUTE] L%02d tok=%d experts=[%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d] "
                    "weights=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    layer, last_tok, h_idx[0], h_idx[1], h_idx[2], h_idx[3], h_idx[4], h_idx[5], h_idx[6],
                    h_idx[7], h_wts[0], h_wts[1], h_wts[2], h_wts[3], h_wts[4], h_wts[5], h_wts[6], h_wts[7]);
        }
    }
    // 4b. Expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(static_cast<float*>(
                                                                  routing.expert_weights.data),
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
            static_cast<const half*>(ly.expert_down_scale.data), static_cast<int>(n_weights));
    }

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    bool use_packed_dequant = (ly.expert_up_packed.data != nullptr && moe_.dequant_buf != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    bool non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                              (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (use_packed_dequant) {
        int64_t ref_eff = non_gated_experts ? ly.expert_up_packed.shape[1] : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != eff || down_eff != eff) {
            IMP_LOG_ERROR(
                "CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
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
        const float* expert_weights = static_cast<const float*>(routing.expert_weights.data);

        half* norm_ptr = static_cast<half*>(no.data);
        half* gate_buf = static_cast<half*>(moe_.expert_gate.data);   // [top_k, eff]
        half* up_buf = static_cast<half*>(moe_.expert_up.data);       // [top_k, eff]
        half* act_buf = static_cast<half*>(moe_.expert_swiglu.data);  // [top_k, eff]
        half* down_buf = static_cast<half*>(moe_.expert_down.data);   // [top_k, d]

        // --- NVFP4 MoE path: takes FP16 input directly, no Q8_1 needed ---
        bool use_nvfp4_moe = (ly.nvfp4_moe_up_ptr != nullptr && ly.nvfp4_moe_down_ptr != nullptr);
        if (use_nvfp4_moe && !non_gated_experts) {
            use_nvfp4_moe = (ly.nvfp4_moe_gate_ptr != nullptr);
        }

        if (use_nvfp4_moe) {
            // Gate+Up projection: NVFP4 MoE GEMV with FP16 input (norm_ptr)
            if (!non_gated_experts) {
                gemv_nvfp4_moe_gate_up_fused(*ly.nvfp4_moe_gate_ptr, *ly.nvfp4_moe_up_ptr, expert_indices,
                                             norm_ptr, gate_buf, up_buf, eff, d, top_k, stream);
            } else {
                gemv_nvfp4_moe_decode(*ly.nvfp4_moe_up_ptr, expert_indices, norm_ptr, up_buf, eff, d,
                                      /*x_stride=*/0, top_k, stream);
            }

            // Down projection (fused SwiGLU+GEMV for gated, separate for non-gated)
            if (!non_gated_experts) {
                // Fused: swiglu(gate,up) computed inline during down GEMV
                gemv_nvfp4_moe_swiglu_decode(*ly.nvfp4_moe_down_ptr, expert_indices, gate_buf, up_buf,
                                             down_buf, d, eff, /*x_stride=*/eff, top_k, stream);
            } else {
                int64_t act_shape[2] = {static_cast<int64_t>(top_k), static_cast<int64_t>(eff)};
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
                gemv_nvfp4_moe_decode(*ly.nvfp4_moe_down_ptr, expert_indices, up_buf, down_buf, d, eff,
                                      /*x_stride=*/eff, top_k, stream);
            }

            // Weighted sum + residual
            {
                bool has_shared_expert = (ly.w_up_shared.data != nullptr);
                const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                                          ? nullptr
                                          : (will_skip_residual_copy ? h.data : r.data);
                moe_weighted_sum_residual(down_buf, expert_weights, res_ptr, h.data, d, top_k, stream);
                if (!has_shared_expert && !moe_use_fp32_residual)
                    residual_fused = true;
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
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
                if (up_qtype == QType::Q6_K) {
                    gemv_q6k_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                    expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                    eff, d, gate_stride, up_stride_bytes,
                                                    /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q4_K) {
                    gemv_q4_k_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q5_K) {
                    gemv_q5_k_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q4_0) {
                    gemv_q4_0_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q2_K) {
                    gemv_q2_k_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q3_K) {
                    gemv_q3_k_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_q8_1_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                     expert_indices, q8, qscratch_.d8_buf, gate_buf, up_buf,
                                                     eff, d, gate_stride, up_stride_bytes,
                                                     /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                }
            } else {
                // Non-gated: up projection only
                auto moe_gemv_dp4a = (up_qtype == QType::Q6_K)   ? gemv_q6k_q8_1_moe_decode
                                     : (up_qtype == QType::Q4_0) ? gemv_q4_0_q8_1_moe_decode
                                     : (up_qtype == QType::Q4_K) ? gemv_q4_k_q8_1_moe_decode
                                     : (up_qtype == QType::Q5_K) ? gemv_q5_k_q8_1_moe_decode
                                     : (up_qtype == QType::Q2_K) ? gemv_q2_k_q8_1_moe_decode
                                     : (up_qtype == QType::Q3_K) ? gemv_q3_k_q8_1_moe_decode
                                                                 : gemv_q8_0_q8_1_moe_decode;
                moe_gemv_dp4a(ly.expert_up_packed.data, expert_indices, q8, qscratch_.d8_buf, up_buf, eff, d,
                              up_stride_bytes,
                              /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
            }
        } else {
            // Fallback: FP16 dequant path
            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
                if (up_qtype == QType::Q6_K) {
                    gemv_q6k_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                               expert_indices, norm_ptr, gate_buf, up_buf, eff, d,
                                               gate_stride, up_stride_bytes,
                                               /*x_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q8_0) {
                    gemv_q8_0_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                                expert_indices, norm_ptr, gate_buf, up_buf, eff, d,
                                                gate_stride, up_stride_bytes,
                                                /*x_stride=*/0, top_k, stream);
                } else {
                    IMP_LOG_ERROR("MoE non-dp4a gate_up_fused: no kernel for qtype %d, skipping GEMV",
                                  (int)up_qtype);
                }
            } else {
                if (up_qtype == QType::Q6_K) {
                    gemv_q6k_moe_decode(ly.expert_up_packed.data, expert_indices, norm_ptr, up_buf, eff, d,
                                        up_stride_bytes, /*x_stride=*/0, top_k, stream);
                } else if (up_qtype == QType::Q8_0) {
                    gemv_q8_0_moe_decode(ly.expert_up_packed.data, expert_indices, norm_ptr, up_buf, eff, d,
                                         up_stride_bytes, /*x_stride=*/0, top_k, stream);
                } else {
                    IMP_LOG_ERROR("MoE non-dp4a up projection: no kernel for qtype %d, skipping GEMV",
                                  (int)up_qtype);
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
                // Fused activation → Q8_1 (1 kernel instead of 2)
                if (cfg.ffn_activation == FFNActivation::GEGLU) {
                    if (layer == 0 && RuntimeConfig::current().diagnostics.debug_forward)
                        fprintf(stderr, "[DEBUG_MoE] Using GEGLU activation for MoE experts\n");
                    geglu_quantize_q8_1(gate_buf, up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
                } else {
                    swiglu_quantize_q8_1(gate_buf, up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
                }
            } else {
                // Non-gated (relu²): fused relu² + Q8_1 quantization (1 kernel)
                relu_sqr_quantize_q8_1(up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
            }

            // Down projection with dp4a GEMV — use down_qtype, NOT up_qtype
            QType dqt = ly.expert_down_packed.qtype;
            auto moe_gemv_dp4a_down = (dqt == QType::Q6_K)   ? gemv_q6k_q8_1_moe_decode
                                      : (dqt == QType::Q4_0) ? gemv_q4_0_q8_1_moe_decode
                                      : (dqt == QType::Q4_K) ? gemv_q4_k_q8_1_moe_decode
                                      : (dqt == QType::Q5_K) ? gemv_q5_k_q8_1_moe_decode
                                      : (dqt == QType::Q2_K) ? gemv_q2_k_q8_1_moe_decode
                                      : (dqt == QType::Q3_K) ? gemv_q3_k_q8_1_moe_decode
                                      : (dqt == QType::Q5_1) ? gemv_q5_1_q8_1_moe_decode
                                                             : gemv_q8_0_q8_1_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype);
            moe_gemv_dp4a_down(ly.expert_down_packed.data, expert_indices, q8, qscratch_.d8_buf, down_buf, d,
                               eff, down_stride,
                               /*q8_1_stride=*/eff_q8_blocks, /*d8_stride=*/eff_q8_blocks, top_k, stream);
        } else {
            // Non-dp4a: separate activation + FP16 down GEMV
            apply_expert_activation(gate_buf, up_buf, act_buf, non_gated_experts, top_k, eff, compute_dtype_,
                                    cfg.ffn_activation, stream);
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype);
            half* down_input = non_gated_experts ? up_buf : act_buf;
            if (ly.expert_down_packed.qtype == QType::Q6_K) {
                gemv_q6k_moe_decode(ly.expert_down_packed.data, expert_indices, down_input, down_buf, d, eff,
                                    down_stride, /*x_stride=*/eff, top_k, stream);
            } else if (ly.expert_down_packed.qtype == QType::Q8_0) {
                gemv_q8_0_moe_decode(ly.expert_down_packed.data, expert_indices, down_input, down_buf, d, eff,
                                     down_stride, /*x_stride=*/eff, top_k, stream);
            } else {
                IMP_LOG_ERROR("MoE non-dp4a down projection: no kernel for qtype %d, skipping GEMV",
                              (int)ly.expert_down_packed.qtype);
            }
        }

        // 9'. Fused weighted sum + FP16 output (+ residual if no shared expert)
        {
            bool has_shared_expert = (ly.w_up_shared.data != nullptr);
            // Use h.data as residual source when memcpy was skipped
            const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                                      ? nullptr
                                      : (will_skip_residual_copy ? h.data : r.data);
            moe_weighted_sum_residual(down_buf, expert_weights, res_ptr, h.data, d, top_k, stream);
            if (!has_shared_expert && !moe_use_fp32_residual)
                residual_fused = true;
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
        bool can_fused_q6k = (ne > 16 && ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                              ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                              up_qtype == QType::Q6_K && ly.expert_down_packed.qtype == QType::Q6_K &&
                              compute_dtype_ == QType::F16);
        if (can_fused_q6k && !non_gated_experts)
            can_fused_q6k = (ly.expert_gate_packed.data && ly.expert_gate_packed.on_device &&
                             ly.expert_gate_packed.qtype == QType::Q6_K);

        bool use_tc = can_fused_q6k && (expanded > ne * 12);
        bool use_scalar = can_fused_q6k && !use_tc && (expanded <= ne * 12);

        if (use_tc || use_scalar) {
            if (layer == 0)
                IMP_LOG_INFO("MoE prefill: fused Q6_K %s path (n=%d, expanded=%d)", use_tc ? "TC" : "scalar",
                             n, expanded);
            const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
            const int32_t* d_sorted = static_cast<const int32_t*>(routing.sorted_token_ids.data);
            char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
            char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
            char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
            char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

            if (use_tc) {
                // TC path: gather-free via sorted_token_ids indirection.
                // Gate and up read from original hidden state (no.data), down reads
                // from SwiGLU output (already in expanded layout, no indirection).

                // Gate projection (gated models only)
                if (!non_gated_experts)
                    gemm_q6k_fused_moe_prefill_tc(ly.expert_gate_packed.data, no.data, expert_gate_base,
                                                  d_offsets, eff, d,
                                                  expert_stride(ly.expert_gate_packed,
                                                                ly.expert_gate_packed.qtype),
                                                  ne, stream, d_sorted);

                // Up projection
                gemm_q6k_fused_moe_prefill_tc(ly.expert_up_packed.data, no.data, expert_up_base, d_offsets,
                                              eff, d, expert_stride(ly.expert_up_packed, up_qtype), ne,
                                              stream, d_sorted);

            } else {
                // Scalar path: needs gathered buffer
                {
                    int64_t gath_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
                    Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
                    moe_gather(no, routing, gathered, stream);
                }
                char* gathered_base = static_cast<char*>(moe_.gathered.data);

                if (!non_gated_experts)
                    gemm_q6k_fused_moe_prefill(
                        ly.expert_gate_packed.data, gathered_base, expert_gate_base, d_offsets, eff, d,
                        expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype), ne, stream);

                gemm_q6k_fused_moe_prefill(ly.expert_up_packed.data, gathered_base, expert_up_base, d_offsets,
                                           eff, d, expert_stride(ly.expert_up_packed, up_qtype), ne, stream);
            }

            // Activation (FP16)
            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                                    non_gated_experts, expanded, eff, compute_dtype_, cfg.ffn_activation,
                                    stream);

            // Down projection (reads from expanded-layout SwiGLU output, no indirection)
            char* fused_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
            if (use_tc) {
                gemm_q6k_fused_moe_prefill_tc(
                    ly.expert_down_packed.data, fused_down_act, expert_down_base, d_offsets, d, eff,
                    expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype), ne, stream);
            } else {
                gemm_q6k_fused_moe_prefill(ly.expert_down_packed.data, fused_down_act, expert_down_base,
                                           d_offsets, d, eff,
                                           expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype),
                                           ne, stream);
            }

            // Falls through to scatter (step 7)
        } else if (RuntimeConfig::current().gemma4.ggml_prefill && cfg.arch == ModelArch::GEMMA4 &&
                   ly.expert_gate_packed.on_device && ly.expert_up_packed.on_device &&
                   ly.expert_down_packed.on_device) {
            // =========================================================================
            // GEMMA-4 ggml MMVQ PREFILL: process each token individually using ggml-
            // compatible Q4_K×Q8_1 dp4a kernels with FP32 norm output, matching llama's
            // full-precision path (no FP16 truncation at norm output).
            // =========================================================================
            if (layer == 0)
                IMP_LOG_INFO("MoE prefill: ggml MMVQ per-token path (n=%d, top_k=%d)", n, top_k);

            half* gate_buf = static_cast<half*>(moe_.expert_gate.data);
            half* up_buf = static_cast<half*>(moe_.expert_up.data);
            half* down_buf = static_cast<half*>(moe_.expert_down.data);

            size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
            size_t up_stride = expert_stride(ly.expert_up_packed, up_qtype);
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype);

            // Dedicated Q8_1 scratch buffer (persistent, sized for max of gate/up input d and down input eff)
            static void* s_q8_scratch = nullptr;
            static size_t s_q8_scratch_size = 0;
            size_t q8_needed = std::max(static_cast<size_t>((d + 31) / 32) * 36,
                                        static_cast<size_t>((eff + 31) / 32) * 36);
            if (!s_q8_scratch || s_q8_scratch_size < q8_needed) {
                if (s_q8_scratch)
                    cudaFree(s_q8_scratch);
                cudaMalloc(&s_q8_scratch, q8_needed);
                s_q8_scratch_size = q8_needed;
            }

            // FP32 norm output buffer (1 token × d floats, persistent)
            static float* s_norm_fp32 = nullptr;
            static int s_norm_fp32_d = 0;
            if (!s_norm_fp32 || s_norm_fp32_d < d) {
                if (s_norm_fp32)
                    cudaFree(s_norm_fp32);
                cudaMalloc(&s_norm_fp32, static_cast<size_t>(d) * sizeof(float));
                s_norm_fp32_d = d;
            }

            // Process each token
            for (int t = 0; t < n; t++) {
                const int32_t* tok_experts = static_cast<const int32_t*>(routing.expert_indices.data) +
                                             static_cast<int64_t>(t) * top_k;
                const float* tok_weights = static_cast<const float*>(routing.expert_weights.data) +
                                           static_cast<int64_t>(t) * top_k;

                // FP32 norm output (FP32 residual → FP32 norm → FP32 Q8_1 quant)
                const float* tok_norm_f32 = s_norm_fp32;
                if (fp32_accum_buf_ != nullptr) {
                    int64_t tok_shape[2] = {1, static_cast<int64_t>(d)};
                    Tensor fp32_tok(static_cast<float*>(fp32_hidden_.data) + static_cast<int64_t>(t) * d,
                                    QType::F32, 2, tok_shape, true);
                    rmsnorm_fp32_to_fp32(fp32_tok, norm_w, s_norm_fp32, 1, d, eps, stream, norm_w_off_);
                } else {
                    tok_norm_f32 = nullptr;
                }

                // D2H copy expert indices
                int32_t h_experts[32];
                cudaMemcpyAsync(h_experts, tok_experts, top_k * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                stream);
                cudaStreamSynchronize(stream);

                // Gate + Up projections: dispatch by actual quant type
                for (int k = 0; k < top_k; k++) {
                    int32_t eid = h_experts[k];
                    const uint8_t* gate_w = static_cast<const uint8_t*>(ly.expert_gate_packed.data) +
                                            static_cast<size_t>(eid) * gate_stride;
                    const uint8_t* up_w = static_cast<const uint8_t*>(ly.expert_up_packed.data) +
                                          static_cast<size_t>(eid) * up_stride;

                    auto do_mmvq = [&](const uint8_t* w, half* out, QType qt) {
                        if (tok_norm_f32) {
                            switch (qt) {
                                case QType::Q4_K:
                                    ggml_mmvq_q4k_f32(w, tok_norm_f32, out, 1, eff, d, s_q8_scratch,
                                                      s_q8_scratch_size, stream);
                                    break;
                                default:
                                    ggml_mmvq_q4k_f32(w, tok_norm_f32, out, 1, eff, d, s_q8_scratch,
                                                      s_q8_scratch_size, stream);
                                    break;
                            }
                        } else {
                            const half* tok_norm_fp16 = static_cast<const half*>(no.data) +
                                                        static_cast<int64_t>(t) * d;
                            switch (qt) {
                                case QType::Q4_K:
                                    ggml_mmvq_q4k(w, tok_norm_fp16, out, 1, eff, d, s_q8_scratch,
                                                  s_q8_scratch_size, stream);
                                    break;
                                case QType::Q8_0:
                                    ggml_mmvq_q8_0(w, tok_norm_fp16, out, 1, eff, d, s_q8_scratch,
                                                   s_q8_scratch_size, stream);
                                    break;
                                default:
                                    ggml_mmvq_q4k(w, tok_norm_fp16, out, 1, eff, d, s_q8_scratch,
                                                  s_q8_scratch_size, stream);
                                    break;
                            }
                        }
                    };
                    do_mmvq(gate_w, gate_buf + static_cast<int64_t>(k) * eff, ly.expert_gate_packed.qtype);
                    do_mmvq(up_w, up_buf + static_cast<int64_t>(k) * eff, up_qtype);
                }

                // Expert activation (GeGLU for Gemma-4)
                half* swiglu_buf = static_cast<half*>(moe_.expert_swiglu.data);
                apply_expert_activation(gate_buf, up_buf, swiglu_buf, false /*non_gated*/, top_k, eff,
                                        compute_dtype_, cfg.ffn_activation, stream);

                // Down projection: dispatch by actual quant type
                for (int k = 0; k < top_k; k++) {
                    int32_t eid = h_experts[k];
                    size_t expert_off = static_cast<size_t>(eid) * down_stride;
                    const uint8_t* w_down = static_cast<const uint8_t*>(ly.expert_down_packed.data) +
                                            expert_off;
                    half* act_k = swiglu_buf + static_cast<int64_t>(k) * eff;
                    half* out_k = down_buf + static_cast<int64_t>(k) * d;

                    switch (ly.expert_down_packed.qtype) {
                        case QType::Q4_K:
                            ggml_mmvq_q4k(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                          stream);
                            break;
                        case QType::Q5_K:
                            ggml_mmvq_q5k(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                          stream);
                            break;
                        case QType::Q8_0:
                            ggml_mmvq_q8_0(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                           stream);
                            break;
                        case QType::Q5_1:
                            ggml_mmvq_q5_1(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                           stream);
                            break;
                        default:
                            ggml_mmvq_q4k(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                          stream);
                            break;
                    }
                }

                // Weighted sum for this token → write to h[t]
                half* h_tok = static_cast<half*>(h.data) + static_cast<int64_t>(t) * d;
                bool has_shared_expert = (ly.w_up_shared.data != nullptr);
                const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                                          ? nullptr
                                          : static_cast<const void*>(static_cast<const half*>(r.data) +
                                                                     static_cast<int64_t>(t) * d);
                moe_weighted_sum_residual(down_buf, tok_weights, res_ptr, h_tok, d, top_k, stream);
            }
            if (ly.w_up_shared.data == nullptr && !moe_use_fp32_residual)
                residual_fused = true;
            goto moe_after_experts;

        } else {
            // =========================================================================
            // FP16 BATCH or FP8 BATCH PREFILL PATH
            // Pre-check: FP16 batch + device-grouped GEMM is preferred (no D2H sync,
            // simpler pipeline). FP8 batch is only used as fallback when FP16 batch
            // isn't available.
            // =========================================================================

            // Gather: reorder tokens by expert assignment (required for batch/legacy paths)
            {
                int64_t gath_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
                Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
                moe_gather(no, routing, gathered, stream);
            }

            {
                // FP16 batch check: can we dequant all experts to FP16 and use device-grouped GEMM?
                size_t fp16_per_expert = static_cast<size_t>(std::max(ly.expert_up_packed.shape[1] *
                                                                          ly.expert_up_packed.shape[2],
                                                                      ly.expert_down_packed.shape[1] *
                                                                          ly.expert_down_packed.shape[2])) *
                                         sizeof(half);
                bool can_fp16_batch_nosync = (moe_.batch_dequant_buf != nullptr &&
                                              moe_.batch_dequant_buf_size >=
                                                  static_cast<size_t>(ne) * fp16_per_expert &&
                                              moe_.d_weight_ptrs && moe_.d_weight_ptrs_count >= ne &&
                                              ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                                              ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                                              dequant_gpu_supported(up_qtype) &&
                                              dequant_gpu_supported(ly.expert_down_packed.qtype));
                if (can_fp16_batch_nosync && !non_gated_experts)
                    can_fp16_batch_nosync = (ly.expert_gate_packed.data && ly.expert_gate_packed.on_device &&
                                             dequant_gpu_supported(ly.expert_gate_packed.qtype));

                // FP8 batch check: fallback when FP16 batch isn't available
                size_t up_fp8_sz = static_cast<size_t>(ne) * ly.expert_up_packed.shape[1] *
                                   ly.expert_up_packed.shape[2];
                size_t down_fp8_sz = static_cast<size_t>(ne) * ly.expert_down_packed.shape[1] *
                                     ly.expert_down_packed.shape[2];
                size_t max_act_cols = std::max(static_cast<size_t>(ly.expert_up_packed.shape[2]),
                                               static_cast<size_t>(ly.expert_down_packed.shape[2]));
                size_t fp8_buf_needed = std::max(up_fp8_sz, down_fp8_sz) +
                                        static_cast<size_t>(expanded) * max_act_cols;
                bool can_fp8_batch = (!can_fp16_batch_nosync && moe_.batch_dequant_buf != nullptr &&
                                      moe_.batch_dequant_buf_size >= fp8_buf_needed &&
                                      ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                                      ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                                      up_qtype == QType::Q6_K && ly.expert_down_packed.qtype == QType::Q6_K &&
                                      compute_dtype_ == QType::F16 && ly.fp16_packed_up_cache == nullptr);
                if (can_fp8_batch && !non_gated_experts)
                    can_fp8_batch = (ly.expert_gate_packed.data && ly.expert_gate_packed.on_device &&
                                     ly.expert_gate_packed.qtype == QType::Q6_K);

                if (debug_forward_enabled() && layer == 0) {
                    int64_t gs[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
                    Tensor gv(moe_.gathered.data, compute_dtype_, 2, gs, true);
                    debug_tensor_stats("L0_moe_gathered", gv, stream);
                    debug_tensor_stats("L0_moe_norm_out_no", no, stream);
                }
                if (can_fp16_batch_nosync) {
                    // =================================================================
                    // FP16 BATCH DEQUANT + cublasGemmGroupedBatchedEx
                    // Dequants all experts Q6_K→FP16 into batch buffer, then runs
                    // a single cublasGemmGroupedBatchedEx per projection. One D2H
                    // sync per layer for offsets (unavoidable for grouped GEMM API).
                    // =================================================================
                    if (layer == 0)
                        IMP_LOG_INFO("MoE prefill: FP16 batch + grouped GEMM path (n=%d, expanded=%d)", n,
                                     expanded);

                    // One D2H sync per layer for expert offsets
                    std::vector<int32_t> h_offsets(ne + 1);
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                                       static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                                       cudaMemcpyDeviceToHost, stream));
                    cudaStreamSynchronize(stream);

                    char* buf = static_cast<char*>(moe_.batch_dequant_buf);
                    char* gathered_base = static_cast<char*>(moe_.gathered.data);
                    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
                    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
                    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
                    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

                    if (fp32_down_active) {
                        size_t fp32_bytes = static_cast<size_t>(expanded) * d * sizeof(float);
                        // Prefer the pre-allocated persistent scratch (avoids per-call
                        // cudaMallocAsync — see moe_workspace.h::fp32_down_buf).
                        if (moe_.fp32_down_buf && moe_.fp32_down_buf_size >= fp32_bytes) {
                            fp32_down_buf = moe_.fp32_down_buf;
                        } else {
                            IMP_CUDA_CHECK_LOG(cudaMallocAsync(&fp32_down_buf, fp32_bytes, stream));
                        }
                    }

                    auto batch_dequant_gemm = [&](const Tensor& packed, QType qtype, const char* a_base,
                                                  char* c_base, int K_dim, int N_dim,
                                                  QType out_dtype = QType::F16) {
                        int64_t rows = packed.shape[1];
                        int64_t cols = packed.shape[2];
                        size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

                        dequant_gpu(static_cast<const uint8_t*>(packed.data), buf, qtype,
                                    ne * static_cast<int>(rows), static_cast<int>(cols), stream);

                        std::vector<const void*> b_ptrs(ne);
                        for (int e = 0; e < ne; ++e)
                            b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

                        gemm_moe_batched(a_base, c_base, h_offsets.data(), b_ptrs.data(), K_dim, N_dim,
                                         QType::F16, ne, stream, moe_.d_work_ptrs, out_dtype);
                    };

                    // Gate projection
                    if (!non_gated_experts)
                        batch_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_packed.qtype, gathered_base,
                                           expert_gate_base, d, eff);
                    if (debug_forward_enabled() && layer == 0) {
                        int64_t sh[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                        Tensor gv(moe_.expert_gate.data, compute_dtype_, 2, sh, true);
                        debug_tensor_stats("L0_moe_gate_out", gv, stream);
                    }

                    // Up projection
                    batch_dequant_gemm(ly.expert_up_packed, up_qtype, gathered_base, expert_up_base, d, eff);
                    if (debug_forward_enabled() && layer == 0) {
                        int64_t sh[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                        Tensor uv(moe_.expert_up.data, compute_dtype_, 2, sh, true);
                        debug_tensor_stats("L0_moe_up_out", uv, stream);
                    }

                    // Activation
                    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                            moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                            compute_dtype_, cfg.ffn_activation, stream);
                    if (debug_forward_enabled() && layer == 0) {
                        int64_t sh[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                        Tensor sv(moe_.expert_swiglu.data, compute_dtype_, 2, sh, true);
                        debug_tensor_stats("L0_moe_swiglu_out", sv, stream);
                    }

                    // Down projection
                    char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                    char* down_target = fp32_down_active ? static_cast<char*>(fp32_down_buf)
                                                         : expert_down_base;
                    QType down_out_dtype = fp32_down_active ? QType::F32 : QType::F16;
                    batch_dequant_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype, down_act,
                                       down_target, eff, d, down_out_dtype);
                    if (debug_forward_enabled() && layer == 0 && !fp32_down_active) {
                        int64_t sh[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
                        Tensor dv(moe_.expert_down.data, compute_dtype_, 2, sh, true);
                        debug_tensor_stats("L0_moe_down_out", dv, stream);
                    }

                    // Falls through to scatter (step 7)

                } else if (can_fp8_batch) {
                    if (layer == 0)
                        IMP_LOG_INFO(
                            "MoE prefill: FP8 batch path (n=%d, expanded=%d, buf=%.1f MiB, need=%.1f MiB)", n,
                            expanded, moe_.batch_dequant_buf_size / (1024.0 * 1024.0),
                            fp8_buf_needed / (1024.0 * 1024.0));
                    // Expert offsets: device-grouped path uses d_offsets directly on GPU.
                    // Host offsets + sync are deferred to the legacy fallback path only.
                    char* buf = static_cast<char*>(moe_.batch_dequant_buf);

                    // FP8 batched GEMM lambda: dequant Q6_K→FP8, quantize FP16 acts→FP8, cuBLAS FP8 GEMM→FP16
                    auto chunked_fp8_gemm = [&](const Tensor& packed, QType qtype, const char* a_base_fp16,
                                                char* c_base_fp16, int K_dim, int N_dim) {
                        int64_t rows = packed.shape[1];
                        int64_t cols = packed.shape[2];
                        size_t weight_fp8_bytes = static_cast<size_t>(ne) * rows *
                                                  cols;  // 1 byte per FP8 element

                        // Buffer layout in moe_.batch_dequant_buf:
                        //   [0 .. weight_fp8_bytes)                     = FP8 weights for all experts
                        //   [weight_fp8_bytes .. weight_fp8_bytes + act) = FP8 activations
                        uint8_t* fp8_weights = reinterpret_cast<uint8_t*>(buf);
                        uint8_t* fp8_acts = fp8_weights + weight_fp8_bytes;

                        // 1. Dequant all experts Q6_K → FP8 E4M3
                        dequant_gpu_fp8(packed.data, fp8_weights, qtype, ne * static_cast<int>(rows),
                                        static_cast<int>(cols), stream);

                        // 2. Per-expert FP8 scaling: calibrate absmax per expert, quantize with
                        //    per-expert scale. Falls back to scale=1.0 if scale buffer unavailable.
                        const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
                        if (moe_.d_fp8_scales) {
                            // Calibrate per-expert: writes scales to moe_.d_fp8_scales
                            calibrate_fp8_scales_per_expert(a_base_fp16, K_dim, d_offsets, ne,
                                                            moe_.d_fp8_scales, stream);
                            // Quantize with per-expert scale
                            quantize_fp16_to_fp8_e4m3_per_expert(a_base_fp16, fp8_acts, K_dim, d_offsets, ne,
                                                                 moe_.d_fp8_scales, stream);
                            // Note: no D2H sync here — device-grouped path uses device-side
                            // scales directly. Host scales are only needed by the fallback path.
                        } else {
                            // Fallback: uniform scale=1.0
                            quantize_fp16_to_fp8_e4m3_scaled(a_base_fp16, fp8_acts, expanded * K_dim, 1.0f,
                                                             stream);
                        }

                        // 3. Build per-expert FP8 weight pointers and dispatch GEMM via
                        //    cublasGemmGroupedBatchedEx (single call for all experts).
                        size_t expert_fp8_sz = static_cast<size_t>(rows) * cols;
                        std::vector<int32_t> h_offsets(ne + 1);
                        cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                        static_cast<size_t>(ne + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                        stream);
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

                        gemm_moe_batched(fp8_acts, c_base_fp16, h_offsets.data(), weight_ptrs.data(), K_dim,
                                         N_dim, QType::FP8_E4M3, ne, stream, moe_.d_work_ptrs,
                                         /*output_dtype=*/QType::F16,
                                         moe_.d_fp8_scales ? h_act_scales.data() : nullptr);
                    };

                    char* gathered_base = static_cast<char*>(moe_.gathered.data);
                    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
                    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
                    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
                    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

                    // Gate projection (gated models only)
                    if (!non_gated_experts)
                        chunked_fp8_gemm(ly.expert_gate_packed, ly.expert_gate_packed.qtype, gathered_base,
                                         expert_gate_base, d, eff);

                    // Up projection
                    chunked_fp8_gemm(ly.expert_up_packed, up_qtype, gathered_base, expert_up_base, d, eff);

                    // Activation (FP16 — reuse existing kernels)
                    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                            moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                            compute_dtype_, cfg.ffn_activation, stream);

                    // Down projection: up buffer for non-gated (relu² in-place), swiglu for gated
                    char* fp8_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                    chunked_fp8_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype, fp8_down_act,
                                     expert_down_base, eff, d);

                    // Falls through to existing scatter (step 7)

                } else if (([&] {
                               // Predicate: CUTLASS 3.x NVFP4 grouped path.
                               // Measured on Qwen3-Coder-30B-A3B-FP4:
                               //   Prefill n=120: ~2750 tok/s (vs legacy ~77)   — 35× win
                               //   Decode n=1:    ~48 tok/s (vs legacy ~38)     — 25% win
                               // After shared-quantize gate+up (2026-04-20), 3.x beats legacy at all n.
                               // `IMP_NO_CUTLASS3X_MOE=1` forces legacy (for debugging).
                               // Reordered above the NVFP4→FP16 dequant batch path 2026-05-10:
                               // SafeTensors NVFP4 prequant models populate both predicates, but the
                               // dequant→cuBLAS path was 88% of pp512 wall time vs CUTLASS 3.x running
                               // the GEMM directly on FP4 tensor cores. covers_ids() now passes for
                               // MoE experts because cache_moe_native_nvfp4 also registers per-expert
                               // CUTLASS_NVFP4 wcache entries.
                               static const bool force_off = RuntimeConfig::current().moe.no_cutlass3x;
                               if (force_off)
                                   return false;
                               if (!cutlass_grouped_3x_nvfp4_available())
                                   return false;
                               if (!moe_.cutlass3x_packed || !moe_.cutlass3x_sf)
                                   return false;
                               auto covers_ids = [&](const std::vector<TensorID>& ids) {
                                   if (static_cast<int>(ids.size()) < ne)
                                       return false;
                                   for (int e = 0; e < ne; ++e) {
                                       if (ids[e] == kInvalidTensorID)
                                           return false;
                                       if (registry_.handle(ids[e]).primary_tier !=
                                           StorageTier::CUTLASS_NVFP4)
                                           return false;
                                   }
                                   return true;
                               };
                               if (!covers_ids(ly.expert_up_ids))
                                   return false;
                               if (!covers_ids(ly.expert_down_ids))
                                   return false;
                               if (!non_gated_experts && !covers_ids(ly.expert_gate_ids))
                                   return false;
                               return true;
                           })()) {
                    // =========================================================================
                    // CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM path (NVFP4 × NVFP4 → FP16).
                    // Gated by IMP_CUTLASS3X_MOE=1. Zero dequant overhead vs the nvfp4→FP16
                    // batch path; per-group alpha via CUTLASS fusion_args.alpha_ptr_array.
                    // =========================================================================
                    std::vector<int32_t> h_offsets(ne + 1);
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                                       static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                                       cudaMemcpyDeviceToHost, stream));
                    // Populate device-resident d_M_per in parallel with the D2H copy.
                    // Phase 1 of MoE-prefill-graphs lever: foundation for graph-safe
                    // dispatch (Phase 2+ migrates host M_per[] uses to this buffer).
                    if (moe_.d_M_per && moe_.d_M_per_count >= ne) {
                        imp::compute_M_per_from_offsets_device(
                            static_cast<const int32_t*>(routing.expert_offsets.data),
                            moe_.d_M_per, ne, stream);
                    }
                    cudaStreamSynchronize(stream);

                    std::vector<int> M_per(ne);
                    for (int e = 0; e < ne; ++e)
                        M_per[e] = h_offsets[e + 1] - h_offsets[e];

                    char* gathered_base = static_cast<char*>(moe_.gathered.data);
                    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
                    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
                    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
                    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

                    // ---------------------------------------------------------------------
                    // Optional smallM kernel branch — opt-in via IMP_NVFP4_SMALLM=1.
                    // Activates when max(M_per) <= IMP_NVFP4_SMALLM_THRESHOLD (default 64)
                    // AND all three NVFP4 native MoE pointers are populated for this layer
                    // (the native [n_experts, N, K/16] layout is what smallM consumes).
                    // Falls through to CUTLASS 3.x on any failure / unavailability.
                    // ---------------------------------------------------------------------
                    bool smallM_done = false;
                    {
                        const char* smallM_env = ::getenv("IMP_NVFP4_SMALLM");
                        const bool smallM_optin = smallM_env && atoi(smallM_env) != 0;
                        if (smallM_optin && imp::gemm_grouped_nvfp4_smallM_available()) {
                            const char* thr_env = ::getenv("IMP_NVFP4_SMALLM_THRESHOLD");
                            const int smallM_threshold =
                                thr_env ? std::clamp(atoi(thr_env), 0, 128) : 64;
                            int max_M = 0;
                            for (int e = 0; e < ne; ++e)
                                max_M = std::max(max_M, M_per[e]);
                            const bool native_up_ok = (ly.nvfp4_moe_up_ptr != nullptr);
                            const bool native_down_ok = (ly.nvfp4_moe_down_ptr != nullptr);
                            const bool native_gate_ok =
                                non_gated_experts || (ly.nvfp4_moe_gate_ptr != nullptr);
                            const bool gate_ne_ok =
                                non_gated_experts ||
                                (ly.nvfp4_moe_gate_ptr && ly.nvfp4_moe_gate_ptr->n_experts == ne);
                            const bool up_ne_ok =
                                native_up_ok && ly.nvfp4_moe_up_ptr->n_experts == ne;
                            const bool down_ne_ok =
                                native_down_ok && ly.nvfp4_moe_down_ptr->n_experts == ne;
                            const bool use_smallM = max_M > 0 && max_M <= smallM_threshold &&
                                                    native_up_ok && native_down_ok &&
                                                    native_gate_ok && up_ne_ok && down_ne_ok &&
                                                    gate_ne_ok;
                            if (use_smallM) {
                                if (layer == 0) {
                                    IMP_LOG_INFO(
                                        "MoE prefill: smallM kernel branch (n=%d, expanded=%d, "
                                        "max_M=%d, thr=%d)",
                                        n, expanded, max_M, smallM_threshold);
                                }

                                // Per-expert offsets into the activation scratch (native
                                // row-major: K/2 bytes per FP4 row, K/16 per UE4M3-SF row).
                                auto compute_offsets =
                                    [&](int K_in, std::vector<size_t>& packed_offs,
                                        std::vector<size_t>& sf_offs) {
                                        packed_offs.assign(ne + 1, 0);
                                        sf_offs.assign(ne + 1, 0);
                                        for (int e = 0; e < ne; ++e) {
                                            packed_offs[e + 1] =
                                                packed_offs[e] +
                                                static_cast<size_t>(M_per[e]) * K_in / 2;
                                            sf_offs[e + 1] =
                                                sf_offs[e] +
                                                static_cast<size_t>(M_per[e]) * K_in / 16;
                                        }
                                    };

                                char* act_packed_base =
                                    static_cast<char*>(moe_.cutlass3x_packed);
                                char* act_sf_base = static_cast<char*>(moe_.cutlass3x_sf);

                                std::vector<size_t> packed_offs_du, sf_offs_du;
                                compute_offsets(d, packed_offs_du, sf_offs_du);

                                bool ok = (packed_offs_du[ne] <= moe_.cutlass3x_packed_size) &&
                                          (sf_offs_du[ne] <= moe_.cutlass3x_sf_size);
                                if (!ok) {
                                    IMP_LOG_ERROR(
                                        "smallM gate/up scratch too small (need %zu/%zu, "
                                        "have %zu/%zu); falling back to CUTLASS 3.x",
                                        packed_offs_du[ne], sf_offs_du[ne],
                                        moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
                                }

                                std::vector<void*> act_packed_ptrs(ne), act_sf_ptrs(ne);
                                // d_act_tscales stays on device — no D2H sync.
                                float* d_act_tscales = nullptr;
                                if (ok) {
                                    for (int e = 0; e < ne; ++e) {
                                        act_packed_ptrs[e] = act_packed_base + packed_offs_du[e];
                                        act_sf_ptrs[e] = act_sf_base + sf_offs_du[e];
                                    }

                                    // Allocate a transient device buffer for per-expert
                                    // activation tensor_scales. Tiny — ne*4 bytes.
                                    IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                                        &d_act_tscales,
                                        static_cast<size_t>(ne) * sizeof(float), stream));
                                    // Quantize gathered FP16 activations native row-major,
                                    // and have the kernel emit per-expert tensor_scales.
                                    imp::quantize_fp16_to_nvfp4_moe_native_with_scales(
                                        reinterpret_cast<const __half*>(gathered_base),
                                        act_packed_ptrs.data(), act_sf_ptrs.data(),
                                        d_act_tscales,
                                        static_cast<const int*>(routing.expert_offsets.data),
                                        expanded, d, ne, stream);
                                    // No D2H sync — d_act_tscales stays on device.
                                }

                                // Build per-expert weight pointer arrays from the native
                                // NvFP4MoEQuantResult cache. alpha = act_ts * weight_ts,
                                // computed entirely on device via compute_moe_alpha_device.
                                //
                                // run_proj signature: takes d_act_ts (device float*) instead of
                                // the old host vector. Weight scales are H2D'd once per
                                // projection (~ne*4 bytes = 256–512 bytes) as a device buffer,
                                // then multiplied on device with no round-trip.
                                auto run_proj = [&](const NvFP4MoEQuantResult* W,
                                                    const std::vector<void*>& act_packed,
                                                    const std::vector<void*>& act_sf,
                                                    const float* d_act_ts,  // device ptr
                                                    char* c_base,
                                                    int K_in, int N_out) -> bool {
                                    // Upload weight tensor_scales H2D once per projection.
                                    // W->tensor_scales is already on device if non-null.
                                    float* d_alpha = nullptr;
                                    IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                                        &d_alpha, static_cast<size_t>(ne) * sizeof(float), stream));
                                    if (W && W->tensor_scales) {
                                        // Compute alpha = act_ts * weight_ts on device.
                                        imp::compute_moe_alpha_device(
                                            d_act_ts, W->tensor_scales, d_alpha, ne, stream);
                                    } else {
                                        // No weight tensor_scale: alpha = act_ts * 1.0
                                        // Copy act_ts into d_alpha directly.
                                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                                            d_alpha, d_act_ts,
                                            static_cast<size_t>(ne) * sizeof(float),
                                            cudaMemcpyDeviceToDevice, stream));
                                    }
                                    std::vector<int> active_M_local;
                                    std::vector<const void*> hA, hSFA, hB, hSFB;
                                    std::vector<void*> hD;
                                    // Note: no hAlpha — alpha stays on device.
                                    // We build a device-indexed view of d_alpha for active
                                    // experts. Since gemm_grouped_nvfp4_smallM accepts a
                                    // contiguous [n_experts] device array and uses blockIdx.x
                                    // as the expert index, we need d_alpha indexed by the
                                    // active-expert position. Build a compact device buffer.
                                    std::vector<float> h_alpha_compact;
                                    active_M_local.reserve(ne);
                                    hA.reserve(ne);
                                    hSFA.reserve(ne);
                                    hB.reserve(ne);
                                    hSFB.reserve(ne);
                                    hD.reserve(ne);
                                    h_alpha_compact.reserve(ne);
                                    // We need to read d_alpha to compact it for active experts
                                    // only when M_per[e]==0 experts are skipped. Read d_alpha
                                    // back if any expert is inactive; otherwise pass d_alpha as-is.
                                    // Optimization: if all experts are active, pass d_alpha directly.
                                    bool all_active = true;
                                    for (int e = 0; e < ne; ++e)
                                        if (M_per[e] == 0) { all_active = false; break; }

                                    for (int e = 0; e < ne; ++e) {
                                        if (M_per[e] == 0)
                                            continue;
                                        active_M_local.push_back(M_per[e]);
                                        hA.push_back(act_packed[e]);
                                        hSFA.push_back(act_sf[e]);
                                        hB.push_back(static_cast<char*>(W->packed_data) +
                                                     static_cast<size_t>(e) *
                                                         W->expert_stride_packed);
                                        hSFB.push_back(static_cast<char*>(W->micro_scales) +
                                                       static_cast<size_t>(e) *
                                                           W->expert_stride_ms);
                                        hD.push_back(c_base +
                                                     static_cast<size_t>(h_offsets[e]) * N_out *
                                                         sizeof(half));
                                    }
                                    const int na = static_cast<int>(active_M_local.size());
                                    if (na == 0) {
                                        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha, stream));
                                        return true;
                                    }

                                    // d_alpha_active: compact device buffer for the na active
                                    // experts. When all experts are active, d_alpha == d_alpha_active.
                                    float* d_alpha_active = d_alpha;
                                    float* d_alpha_compact_dev = nullptr;
                                    if (!all_active) {
                                        // Need to compact alpha for active experts.
                                        // Cheapest: D2H the small d_alpha buffer (ne floats),
                                        // compact, H2D compact array. Still eliminates the
                                        // D2H of activation scales (the expensive one).
                                        std::vector<float> h_alpha_full(ne);
                                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                                            h_alpha_full.data(), d_alpha,
                                            static_cast<size_t>(ne) * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream));
                                        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                                        for (int e = 0; e < ne; ++e)
                                            if (M_per[e] > 0)
                                                h_alpha_compact.push_back(h_alpha_full[e]);
                                        IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                                            &d_alpha_compact_dev,
                                            static_cast<size_t>(na) * sizeof(float), stream));
                                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                                            d_alpha_compact_dev, h_alpha_compact.data(),
                                            static_cast<size_t>(na) * sizeof(float),
                                            cudaMemcpyHostToDevice, stream));
                                        d_alpha_active = d_alpha_compact_dev;
                                    }

                                    bool ret = imp::gemm_grouped_nvfp4_smallM(
                                        na, active_M_local.data(), N_out, K_in, hA.data(),
                                        hSFA.data(), hB.data(), hSFB.data(), hD.data(),
                                        d_alpha_active, stream);
                                    if (d_alpha_compact_dev)
                                        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha_compact_dev, stream));
                                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_alpha, stream));
                                    return ret;
                                };

                                bool ok_gate = ok;
                                if (ok && !non_gated_experts) {
                                    ok_gate = run_proj(ly.nvfp4_moe_gate_ptr, act_packed_ptrs,
                                                       act_sf_ptrs, d_act_tscales,
                                                       expert_gate_base, d, eff);
                                }
                                bool ok_up = ok;
                                if (ok) {
                                    ok_up = run_proj(ly.nvfp4_moe_up_ptr, act_packed_ptrs,
                                                     act_sf_ptrs, d_act_tscales,
                                                     expert_up_base, d, eff);
                                }

                                // Free gate/up activation scales (down will use a fresh d_act_tscales_dn).
                                if (d_act_tscales) {
                                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_act_tscales, stream));
                                    d_act_tscales = nullptr;
                                }

                                if (ok && ok_gate && ok_up) {
                                    apply_expert_activation(
                                        moe_.expert_gate.data, moe_.expert_up.data,
                                        moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                        compute_dtype_, cfg.ffn_activation, stream);

                                    // Down projection: re-quantize post-activation buffer
                                    // (K_in = eff). Reuse staging via stream-ordered
                                    // overwrite.
                                    std::vector<size_t> packed_offs_dn, sf_offs_dn;
                                    compute_offsets(eff, packed_offs_dn, sf_offs_dn);
                                    bool down_ok =
                                        (packed_offs_dn[ne] <= moe_.cutlass3x_packed_size) &&
                                        (sf_offs_dn[ne] <= moe_.cutlass3x_sf_size);
                                    if (!down_ok) {
                                        IMP_LOG_ERROR(
                                            "smallM down scratch too small (need %zu/%zu, "
                                            "have %zu/%zu); falling back to CUTLASS 3.x",
                                            packed_offs_dn[ne], sf_offs_dn[ne],
                                            moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
                                    }
                                    if (down_ok) {
                                        for (int e = 0; e < ne; ++e) {
                                            act_packed_ptrs[e] =
                                                act_packed_base + packed_offs_dn[e];
                                            act_sf_ptrs[e] = act_sf_base + sf_offs_dn[e];
                                        }
                                        char* down_act = non_gated_experts ? expert_up_base
                                                                            : expert_swiglu_base;
                                        // Re-quantize post-SwiGLU activations; keep scales on device.
                                        float* d_act_tscales_dn = nullptr;
                                        IMP_CUDA_CHECK_LOG(cudaMallocAsync(
                                            &d_act_tscales_dn,
                                            static_cast<size_t>(ne) * sizeof(float), stream));
                                        imp::quantize_fp16_to_nvfp4_moe_native_with_scales(
                                            reinterpret_cast<const __half*>(down_act),
                                            act_packed_ptrs.data(), act_sf_ptrs.data(),
                                            d_act_tscales_dn,
                                            static_cast<const int*>(routing.expert_offsets.data),
                                            expanded, eff, ne, stream);
                                        // No D2H sync — pass d_act_tscales_dn directly.
                                        bool ok_down =
                                            run_proj(ly.nvfp4_moe_down_ptr, act_packed_ptrs,
                                                     act_sf_ptrs, d_act_tscales_dn,
                                                     expert_down_base, eff, d);
                                        IMP_CUDA_CHECK_LOG(
                                            cudaFreeAsync(d_act_tscales_dn, stream));
                                        if (ok_down) {
                                            smallM_done = true;
                                        } else {
                                            IMP_LOG_ERROR(
                                                "smallM down dispatch failed; falling back to "
                                                "CUTLASS 3.x");
                                        }
                                    }
                                } else if (ok) {
                                    IMP_LOG_ERROR(
                                        "smallM gate/up dispatch failed; falling back to "
                                        "CUTLASS 3.x");
                                }
                                // Free activation scales if not yet freed (early-exit paths).
                                if (d_act_tscales) {
                                    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_act_tscales, stream));
                                    d_act_tscales = nullptr;
                                }
                            }
                        }
                    }

                    if (!smallM_done && layer == 0)
                        IMP_LOG_INFO("MoE prefill: CUTLASS 3.x NVFP4 grouped (n=%d, expanded=%d)",
                                     n, expanded);

                    if (!smallM_done) {

                    // Active-expert SFA offset table (computed per K_in; different for d vs eff).
                    // Shared across same-K_in projections: gate and up both use K_in=d,
                    // so a single quantize of `gathered_base` + pointer array is reused
                    // for both grouped GEMMs. Down reuses the staging buffer with fresh
                    // quantize for K_in=eff (stream-ordered overwrite).
                    auto quantize_once = [&](const char* a_base, int K_in, std::vector<size_t>& sfa_offsets,
                                             std::vector<uint8_t*>& h_sfa_bases) -> bool {
                        sfa_offsets.assign(ne + 1, 0);
                        h_sfa_bases.assign(ne, nullptr);
                        size_t total_packed = 0;
                        for (int e = 0; e < ne; ++e) {
                            total_packed += static_cast<size_t>(M_per[e]) * K_in / 2;
                            sfa_offsets[e + 1] = sfa_offsets[e] + cutlass_nvfp4_sf_size(M_per[e], K_in);
                        }
                        const size_t total_sfa = sfa_offsets[ne];
                        if (total_packed > moe_.cutlass3x_packed_size || total_sfa > moe_.cutlass3x_sf_size) {
                            IMP_LOG_ERROR(
                                "CUTLASS 3.x MoE staging too small: need packed=%zu sf=%zu, have %zu/%zu",
                                total_packed, total_sfa, moe_.cutlass3x_packed_size, moe_.cutlass3x_sf_size);
                            return false;
                        }
                        uint8_t* all_sfa = static_cast<uint8_t*>(moe_.cutlass3x_sf);
                        for (int e = 0; e < ne; ++e) {
                            if (M_per[e] > 0)
                                h_sfa_bases[e] = all_sfa + sfa_offsets[e];
                        }
                        // Upload SFA pointer array to device (~1 KB).
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(moe_.cutlass3x_sfa_ptrs, h_sfa_bases.data(),
                                                           static_cast<size_t>(ne) * sizeof(uint8_t*),
                                                           cudaMemcpyHostToDevice, stream));
                        // Zero active SFA region (SfAtom pads rows to 128).
                        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(all_sfa, 0, total_sfa, stream));
                        // Fused per-expert quantize in one kernel launch.
                        quantize_fp16_to_nvfp4_cutlass_moe(
                            a_base, moe_.cutlass3x_packed, moe_.cutlass3x_sfa_ptrs,
                            routing.expert_offsets.data ? static_cast<const int*>(routing.expert_offsets.data)
                                                        : nullptr,
                            expanded, K_in, ne, stream);
                        return true;
                    };

                    // Opt-in: device-args wrapper path (Phase 3c-MVP wire).
                    // Uses imp::gemm_grouped_cutlass_3x_nvfp4_device_args, which
                    // builds the per-expert staging buffer in a device kernel
                    // from device-resident d_M_per / d_expert_offsets /
                    // d_sfa_offsets / d_alpha + per-expert ptr arrays. Note:
                    // upstream of this lambda still does the h_offsets D2H+sync,
                    // so this MVP does NOT yet enable CUDA-graph capture of the
                    // prefill path — Phase 3c-full removes that residual sync.
                    // Enable via IMP_NVFP4_DEVICE_ARGS=1.
                    const char* device_args_env = ::getenv("IMP_NVFP4_DEVICE_ARGS");
                    const bool use_device_args =
                        device_args_env && std::atoi(device_args_env) != 0 &&
                        moe_.d_M_per && moe_.d_M_per_count >= ne &&
                        moe_.d_sfa_offsets && moe_.d_B_ptrs_cache &&
                        moe_.d_SFB_ptrs_cache && moe_.d_alpha_full;
                    auto grouped_gemm_device = [&](const std::vector<TensorID>& weight_ids,
                                                   char* c_base, int K_in, int N_out) -> bool {
                        // Per-expert host arrays (registry handles). Built for ALL
                        // experts — CUTLASS skips empty groups via M=0 internally.
                        std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
                        std::vector<float> h_alpha(ne);
                        for (int e = 0; e < ne; ++e) {
                            const auto& h = registry_.handle(weight_ids[e]);
                            h_B_ptrs[e]   = h.payload.cutlass_nvfp4.weight;
                            h_SFB_ptrs[e] = h.payload.cutlass_nvfp4.sf;
                            h_alpha[e]    = h.payload.cutlass_nvfp4.global_scale
                                                ? *h.payload.cutlass_nvfp4.global_scale
                                                : 1.0f;
                        }
                        // Upload host arrays into workspace-cached device
                        // buffers (Phase 3c-full Step 1). No per-call cudaMalloc.
                        cudaMemcpyAsync(moe_.d_B_ptrs_cache,   h_B_ptrs.data(),
                                        ne * sizeof(const void*), cudaMemcpyHostToDevice, stream);
                        cudaMemcpyAsync(moe_.d_SFB_ptrs_cache, h_SFB_ptrs.data(),
                                        ne * sizeof(const void*), cudaMemcpyHostToDevice, stream);
                        cudaMemcpyAsync(moe_.d_alpha_full,     h_alpha.data(),
                                        ne * sizeof(float),       cudaMemcpyHostToDevice, stream);
                        // SFA byte-offset prefix sum on device (Phase 3a kernel).
                        imp::compute_sfa_offsets_device(moe_.d_M_per, moe_.d_sfa_offsets,
                                                        ne, K_in, stream);

                        imp::GroupedNvfp4DeviceArgs dargs{};
                        dargs.d_M_per          = moe_.d_M_per;
                        dargs.d_expert_offsets =
                            static_cast<const int32_t*>(routing.expert_offsets.data);
                        dargs.d_sfa_offsets    = moe_.d_sfa_offsets;
                        dargs.d_alpha          = moe_.d_alpha_full;
                        dargs.base_A_packed    = moe_.cutlass3x_packed;
                        dargs.base_A_sf        = moe_.cutlass3x_sf;
                        dargs.d_B_ptrs         = moe_.d_B_ptrs_cache;
                        dargs.d_SFB_ptrs       = moe_.d_SFB_ptrs_cache;
                        dargs.base_D           = c_base;

                        return imp::gemm_grouped_cutlass_3x_nvfp4_device_args(
                            ne, N_out, K_in, dargs, stream);
                    };

                    // Dispatch a grouped GEMM given already-quantized activations.
                    auto grouped_gemm = [&](const std::vector<TensorID>& weight_ids, char* c_base, int K_in,
                                            int N_out, const std::vector<size_t>& sfa_offsets) {
                        if (use_device_args) {
                            if (layer == 0)
                                IMP_LOG_INFO(
                                    "MoE prefill: CUTLASS 3.x device-args path "
                                    "(opt-in via IMP_NVFP4_DEVICE_ARGS=1)");
                            if (grouped_gemm_device(weight_ids, c_base, K_in, N_out))
                                return;
                            IMP_LOG_ERROR(
                                "device-args dispatch failed; falling back to host-args");
                        }
                        char* all_packed = static_cast<char*>(moe_.cutlass3x_packed);
                        uint8_t* all_sfa = static_cast<uint8_t*>(moe_.cutlass3x_sf);
                        // Active experts only (CUTLASS 3.x wants non-empty groups).
                        std::vector<int> active_M;
                        std::vector<const void*> hA, hSFA, hB, hSFB;
                        std::vector<void*> hD;
                        std::vector<float> hAlpha;
                        active_M.reserve(ne);
                        hA.reserve(ne);
                        hSFA.reserve(ne);
                        hB.reserve(ne);
                        hSFB.reserve(ne);
                        hD.reserve(ne);
                        hAlpha.reserve(ne);
                        for (int e = 0; e < ne; ++e) {
                            if (M_per[e] == 0)
                                continue;
                            const auto& h = registry_.handle(weight_ids[e]);
                            active_M.push_back(M_per[e]);
                            hA.push_back(all_packed + static_cast<size_t>(h_offsets[e]) * K_in / 2);
                            hSFA.push_back(all_sfa + sfa_offsets[e]);
                            hB.push_back(h.payload.cutlass_nvfp4.weight);
                            hSFB.push_back(h.payload.cutlass_nvfp4.sf);
                            hD.push_back(c_base + static_cast<size_t>(h_offsets[e]) * N_out * sizeof(half));
                            hAlpha.push_back(h.payload.cutlass_nvfp4.global_scale
                                                 ? *h.payload.cutlass_nvfp4.global_scale
                                                 : 1.0f);
                        }
                        const int na = static_cast<int>(active_M.size());
                        if (na == 0)
                            return;
                        bool ok = gemm_grouped_cutlass_3x_nvfp4(na, active_M.data(), N_out, K_in, hA.data(),
                                                                hSFA.data(), hB.data(), hSFB.data(),
                                                                hD.data(), hAlpha.data(), stream);
                        if (!ok)
                            IMP_LOG_ERROR("CUTLASS 3.x grouped dispatch failed (K=%d N=%d ne=%d)", K_in,
                                          N_out, na);
                    };

                    // Gate+Up share the same input (gathered_base, K_in=d) → quantize once,
                    // run two grouped GEMMs reusing staging buffers.
                    std::vector<size_t> sfa_offs;
                    std::vector<uint8_t*> sfa_bases;
                    if (quantize_once(gathered_base, d, sfa_offs, sfa_bases)) {
                        // Opt-in fused path: single CUTLASS 3.x dispatch with 2*ne problems.
                        // IMP_NVFP4_FUSED_GATEUP=1 activates it; default is two-call form.
                        // Currently flat-perf vs default (GrpGemm cache in 769effe already
                        // absorbs the per-dispatch overhead). Preserved for future scenarios.
                        const char* fused_env = ::getenv("IMP_NVFP4_FUSED_GATEUP");
                        const bool use_fused = !non_gated_experts &&
                                               fused_env && atoi(fused_env) != 0;
                        bool used_fused = false;

                        if (use_fused) {
                            if (layer == 0)
                                IMP_LOG_INFO("MoE prefill: CUTLASS 3.x gate+up fused (opt-in)");

                            char* all_packed = static_cast<char*>(moe_.cutlass3x_packed);
                            uint8_t* all_sfa_ptr = static_cast<uint8_t*>(moe_.cutlass3x_sf);
                            std::vector<int> active_M;
                            std::vector<const void*> hA, hSFA, hB_gate, hSFB_gate, hB_up, hSFB_up;
                            std::vector<void*> hD_gate, hD_up;
                            std::vector<float> hAlpha_gate, hAlpha_up;
                            active_M.reserve(ne);
                            hA.reserve(ne); hSFA.reserve(ne);
                            hB_gate.reserve(ne); hSFB_gate.reserve(ne);
                            hB_up.reserve(ne); hSFB_up.reserve(ne);
                            hD_gate.reserve(ne); hD_up.reserve(ne);
                            hAlpha_gate.reserve(ne); hAlpha_up.reserve(ne);

                            for (int e = 0; e < ne; ++e) {
                                if (M_per[e] == 0)
                                    continue;
                                const auto& hg = registry_.handle(ly.expert_gate_ids[e]);
                                const auto& hu = registry_.handle(ly.expert_up_ids[e]);
                                active_M.push_back(M_per[e]);
                                hA.push_back(all_packed + static_cast<size_t>(h_offsets[e]) * d / 2);
                                hSFA.push_back(all_sfa_ptr + sfa_offs[e]);
                                hB_gate.push_back(hg.payload.cutlass_nvfp4.weight);
                                hSFB_gate.push_back(hg.payload.cutlass_nvfp4.sf);
                                hD_gate.push_back(expert_gate_base +
                                                  static_cast<size_t>(h_offsets[e]) * eff * sizeof(half));
                                hAlpha_gate.push_back(hg.payload.cutlass_nvfp4.global_scale
                                                          ? *hg.payload.cutlass_nvfp4.global_scale
                                                          : 1.0f);
                                hB_up.push_back(hu.payload.cutlass_nvfp4.weight);
                                hSFB_up.push_back(hu.payload.cutlass_nvfp4.sf);
                                hD_up.push_back(expert_up_base +
                                                static_cast<size_t>(h_offsets[e]) * eff * sizeof(half));
                                hAlpha_up.push_back(hu.payload.cutlass_nvfp4.global_scale
                                                        ? *hu.payload.cutlass_nvfp4.global_scale
                                                        : 1.0f);
                            }
                            const int na = static_cast<int>(active_M.size());
                            if (na > 0) {
                                bool ok = imp::dispatch_gate_up_grouped_fused(
                                    na, active_M.data(), eff, d,
                                    hA.data(), hSFA.data(),
                                    hB_gate.data(), hSFB_gate.data(),
                                    hB_up.data(), hSFB_up.data(),
                                    hD_gate.data(), hD_up.data(),
                                    hAlpha_gate.data(), hAlpha_up.data(),
                                    stream);
                                if (ok) {
                                    used_fused = true;
                                } else {
                                    IMP_LOG_ERROR(
                                        "Fused gate+up dispatch failed; falling back to two-call form");
                                }
                            } else {
                                used_fused = true;  // nothing to do, trivially done
                            }
                        }

                        if (!used_fused) {
                            if (!non_gated_experts)
                                grouped_gemm(ly.expert_gate_ids, expert_gate_base, d, eff, sfa_offs);
                            grouped_gemm(ly.expert_up_ids, expert_up_base, d, eff, sfa_offs);
                        }
                    }

                    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                            moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                            compute_dtype_, cfg.ffn_activation, stream);

                    // Down has a different activation (post-SwiGLU, K_in=eff) → re-quantize.
                    // (A fused silu(gate)*up+quantize kernel was tried but regressed short-prompt
                    //  decode ~11% due to low SM occupancy at small expanded — existing swiglu
                    //  kernel has better per-element parallelism, so keep it separate.)
                    char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                    if (quantize_once(down_act, eff, sfa_offs, sfa_bases)) {
                        grouped_gemm(ly.expert_down_ids, expert_down_base, eff, d, sfa_offs);
                    }
                    }  // !smallM_done

                    // Falls through to scatter (step 7)

                } else if (ly.nvfp4_moe_up_ptr != nullptr && ly.nvfp4_moe_down_ptr != nullptr &&
                           (non_gated_experts || ly.nvfp4_moe_gate_ptr != nullptr) &&
                           moe_.batch_dequant_buf != nullptr &&
                           moe_.batch_dequant_buf_size >= static_cast<size_t>(ne) * fp16_per_expert &&
                           moe_.d_weight_ptrs && moe_.d_weight_ptrs_count >= ne) {
                    // =================================================================
                    // NVFP4→FP16 BATCH DEQUANT + grouped GEMM (fallback when CUTLASS 3.x
                    // can't fire — e.g. allocation failed, force_off env var, llm-compressor
                    // format that goes through the dequant→cuBLAS path for correctness).
                    // Dequants NVFP4 expert weights to FP16, then same grouped GEMM as FP16 batch.
                    // =================================================================
                    if (layer == 0)
                        IMP_LOG_INFO("MoE prefill: NVFP4→FP16 batch path (n=%d, expanded=%d)", n, expanded);

                    std::vector<int32_t> h_offsets(ne + 1);
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                                                       static_cast<size_t>(ne + 1) * sizeof(int32_t),
                                                       cudaMemcpyDeviceToHost, stream));
                    cudaStreamSynchronize(stream);

                    char* buf = static_cast<char*>(moe_.batch_dequant_buf);
                    char* gathered_base = static_cast<char*>(moe_.gathered.data);
                    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
                    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
                    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
                    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

                    auto nvfp4_batch_dequant_gemm = [&](const NvFP4MoEQuantResult& nvfp4, const char* a_base,
                                                        char* c_base, int K_dim, int N_dim) {
                        int64_t rows = nvfp4.N;
                        int64_t cols = nvfp4.K;
                        size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

                        // Dequant all experts NVFP4 → FP16
                        dequantize_nvfp4_moe_to_fp16(nvfp4, buf, stream);

                        std::vector<const void*> b_ptrs(ne);
                        for (int e = 0; e < ne; ++e)
                            b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

                        gemm_moe_batched(a_base, c_base, h_offsets.data(), b_ptrs.data(), K_dim, N_dim,
                                         QType::F16, ne, stream, moe_.d_work_ptrs);
                    };

                    // Gate projection
                    if (!non_gated_experts)
                        nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_gate_ptr, gathered_base, expert_gate_base, d,
                                                 eff);

                    // Up projection
                    nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_up_ptr, gathered_base, expert_up_base, d, eff);

                    // Activation
                    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                            moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                            compute_dtype_, cfg.ffn_activation, stream);

                    // Down projection
                    char* slow_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                    nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_down_ptr, slow_down_act, expert_down_base, eff, d);

                    // Falls through to scatter (step 7)

                } else {
                    // =========================================================================
                    // LEGACY FALLBACK: D2H sync + per-expert or batched GEMM
                    // =========================================================================
                    if (layer == 0)
                        IMP_LOG_INFO("MoE prefill: legacy FP16 fallback path (n=%d, expanded=%d)", n,
                                     expanded);
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
                        auto dequant_expert = [&](const Tensor& packed, QType qtype,
                                                  int expert_idx) -> Tensor {
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
                                    (long)packed.shape[1], (long)packed.shape[2], (unsigned)qtype);
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
                                    void* cached = expert_cache_.get_or_load(ck, host_ptr, expert_raw,
                                                                             stream);
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
                                               const std::vector<TensorID>& fallback_ids, int eidx) {
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
                                        w = expert_cache_.get_or_load(ck, host_ptr, expert_raw, stream);
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
                                Tensor b = use_packed_dequant ? dequant_expert(packed, qtype, eidx)
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
                                                        int N_dim) {
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
                                    expert_gemm(a_view, c_view, packed, qtype, fallback, fallback_ids, e);
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
                                                     expert_gate_base, d, eff);
                            chunked_dequant_gemm(ly.expert_up_packed, ly.expert_up_packed.qtype,
                                                 ly.expert_w_up, ly.expert_up_ids, gathered_base,
                                                 expert_up_base, d, eff);

                            apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data,
                                                    moe_.expert_swiglu.data, non_gated_experts, expanded, eff,
                                                    compute_dtype_, cfg.ffn_activation, stream);

                            {
                                char* dequant_down_act = non_gated_experts ? expert_up_base
                                                                           : expert_swiglu_base;
                                chunked_dequant_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype,
                                                     ly.expert_w_down, ly.expert_down_ids, dequant_down_act,
                                                     expert_down_base, eff, d);
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
                                                ly.expert_gate_ids, e);
                                }

                                {
                                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                                    Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                                                  compute_dtype_, 2, c_shape, true);
                                    expert_gemm(a_view, c_view, ly.expert_up_packed,
                                                ly.expert_up_packed.qtype, ly.expert_w_up, ly.expert_up_ids,
                                                e);
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
                                            e);
                            }
                        }
                    }
                }  // legacy inner scope
            }  // else of can_fp16/fp8_batch/legacy
        }  // FP8/FP16 prefill scope
    }  // else branch of can_fused_q6k + fused Q6_K scope

    // 7+8. Scatter expert outputs back to token positions.
    //      Fused path: token-centric scatter + FP16 convert (+ residual if no shared expert).
    //      Fallback: atomicAdd scatter + FP32->FP16 convert.
    {
        bool has_shared_expert = (ly.w_up_shared.data != nullptr);
        if (routing.token_to_expanded && compute_dtype_ == QType::F16) {
            // Fused token-centric scatter: no atomics, no FP32 intermediate buffer.
            // If no shared expert, also fuse residual add.
            // Gemma-4 FP32 path: defer residual to post_ffn_norm (FP32 accumulator).
            const void* res_ptr = (!has_shared_expert && !residual_fused && !moe_use_fp32_residual) ? r.data
                                                                                                    : nullptr;
            if (fp32_down_buf) {
                // Down GEMM kept FP32 output — use FP32-input scatter variant.
                moe_scatter_fused_residual_fp32in(fp32_down_buf, routing.token_to_expanded,
                                                  static_cast<const float*>(routing.expert_weights.data),
                                                  res_ptr, h.data, n, d, top_k, stream);
            } else {
                moe_scatter_fused_residual(moe_.expert_down.data, routing.token_to_expanded,
                                           static_cast<const float*>(routing.expert_weights.data), res_ptr,
                                           h.data, n, d, top_k, stream);
            }
            if (!has_shared_expert && !moe_use_fp32_residual)
                residual_fused = true;
        } else {
            // Fallback: atomicAdd scatter into FP32 buffer, then convert
            int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
            Tensor expert_down_view(moe_.expert_down.data, compute_dtype_, 2, expert_out_shape, true);
            Tensor scatter_out = slice_rows(moe_.scatter_out, n);
            IMP_CUDA_CHECK_LOG(
                cudaMemsetAsync(scatter_out.data, 0, static_cast<size_t>(n) * d * sizeof(float), stream));
            moe_scatter(expert_down_view, routing, scatter_out, stream);

            int64_t numel = static_cast<int64_t>(n) * d;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == QType::F16) {
                fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<const float*>(
                                                                        moe_.scatter_out.data),
                                                                    static_cast<half*>(h.data), numel);
            } else {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h.data, moe_.scatter_out.data,
                                                   static_cast<size_t>(numel) * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

moe_after_experts:
    // Free diagnostic FP32 expert-down buffer if we malloc'd it ourselves.
    // The persistent moe_.fp32_down_buf is owned by MoEWorkspace — don't free.
    if (fp32_down_buf && fp32_down_buf != moe_.fp32_down_buf) {
        cudaFreeAsync(fp32_down_buf, stream);
    }
    fp32_down_buf = nullptr;
    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    // Gemma 4: sanitize inf/NaN in MoE scatter output before post-norm.
    if (cfg.arch == ModelArch::GEMMA4) {
        sanitize_fp16(static_cast<__half*>(h.data), static_cast<int64_t>(n) * d, stream);
    }

    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_moe_scatter_out", view_tokens(h, n), stream);
    }
    // Gemma 4: apply post_ffw_norm_2 on the MoE branch output (h) BEFORE shared adds.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_2.data != nullptr) {
        rmsnorm(h, ly.ffn_post_norm_2, h, eps, stream, norm_w_off_);
    }
    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_moe_post_norm2", view_tokens(h, n), stream);
    }

    // Gemma 4: re-derive `no` for the shared MLP from the saved residual
    // (which still holds the original hidden state) using ffn_norm — the MoE
    // branch consumed `no` produced from pre_ffw_norm_2 above.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr &&
        ly.w_up_shared.data != nullptr && ly.ffn_norm.data != nullptr) {
        rmsnorm(r, ly.ffn_norm, no, eps, stream, norm_w_off_);
    }

    // Shared expert: enabled by default. Gemma 4: requires post_ffw_norm_1 to be uploaded.
    // DIAGNOSTIC: IMP_NO_SHARED_MLP=1 skips this branch (Gemma-4 NVFP4 audit).
    static const bool s_no_shared_mlp = RuntimeConfig::current().moe.no_shared_mlp;
    if (ly.w_up_shared.data != nullptr && !s_no_shared_mlp) {
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
            auto ctx = GemmContext::make(stream, wcache_, qscratch_, cur_force_fp16_);
            gemm_dispatch(no, ly.w_up_shared, sh_up, ctx);

            if (shared_gated) {
                // Gated: gate + SwiGLU
                Tensor sh_gate(moe_.expert_gate.data, compute_dtype_, 2, sh_shape, true);
                gemm_dispatch(no, ly.w_gate_shared, sh_gate, ctx);
                if (cfg.ffn_activation == FFNActivation::GEGLU)
                    geglu(sh_gate, sh_up, sh_swiglu, stream);
                else
                    swiglu(sh_gate, sh_up, sh_swiglu, stream);
            } else {
                // Non-gated: relu^2(up) in-place [Nemotron-H uses squared ReLU]
                relu_sqr_inplace(sh_up, stream);
            }

            if (layer == 0) {
                Tensor sh_gate_raw(moe_.expert_gate.data, compute_dtype_, 2, sh_shape, true);
                debug_tensor_stats_all("L0_sh_up_raw", sh_up, stream);
                debug_tensor_stats_all("L0_sh_gate_raw", sh_gate_raw, stream);
                debug_tensor_stats_all("L0_sh_swiglu", sh_swiglu, stream);
            }
            // Down projection (reads from sh_up for non-gated since relu² was in-place)
            Tensor& sh_act = shared_gated ? sh_swiglu : sh_up;
            if (layer == 0)
                debug_tensor_stats_all("L0_sh_act_preDown", sh_act, stream);
            gemm_dispatch(sh_act, ly.w_down_shared, sh_down, ctx);
        }

        if (layer == 0) {
            debug_tensor_stats_all("L0_sh_down_raw", sh_down, stream);
            debug_tensor_rows("L0_sh_down_rows", sh_down, stream);
            // Also dump shared expert norm input
            debug_tensor_rows("L0_shared_norm_in", view_tokens(no, n), stream);
        }
        // Gemma 4: shared MLP can overflow FP16 at deep layers. Sanitize inf/NaN
        // to zero before the post-norm so rmsnorm doesn't produce all-zero output.
        if (cfg.arch == ModelArch::GEMMA4) {
            sanitize_fp16(static_cast<__half*>(sh_down.data), static_cast<int64_t>(n) * d, stream);
        }
        // Gemma 4: apply post_ffw_norm_1 on shared MLP output (sh_down).
        if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_1.data != nullptr &&
            !RuntimeConfig::current().gemma4.no_post_ffw_1) {
            rmsnorm(sh_down, ly.ffn_post_norm_1, sh_down, eps, stream, norm_w_off_);
        }

        if (layer == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_shared_post_post_norm1", layer);
            debug_tensor_stats_all(buf, sh_down, stream);
        }
        if (debug_forward_enabled() && layer == 0) {
            debug_tensor_rows("L0_shared_post_norm1", sh_down, stream);
        }
        // Qwen3-Next / Qwen3.6: per-token sigmoid gate on the shared expert output.
        //   gate[r] = sigmoid(sum_d no[r,d] * W_gate_inp_shexp[d])
        //   sh_down[r, :] *= gate[r]
        // Absent in Qwen3 MoE / Gemma-4 (tensor pointer is null → skip).
        static const bool skip_shexp_gate = RuntimeConfig::current().moe.no_shexp_gate;
        if (!skip_shexp_gate && ly.shared_expert_gate_inp.data != nullptr &&
            ly.shared_expert_gate_inp.on_device && compute_dtype_ == QType::F16) {
            shared_expert_gate_scale(no.data, ly.shared_expert_gate_inp.data, sh_down.data, n, d, d, stream);
        }
        // Add shared expert output to hidden (which already has routed expert output)
        elementwise_add(h, sh_down, stream);
    }
    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_combined", view_tokens(h, n), stream);
    }
    if (layer == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "L%d_combined_pre_post_ffn_norm", layer);
        debug_tensor_stats_all(buf, h, stream);
    }

    // Gemma 4: apply post_ffn_norm (combined post-norm) BEFORE residual add.
    // If FP32 accumulator is active, fuse post_ffn_norm + residual add into
    // rmsnorm_fp32_accum_to_fp16_kernel so the residual stays in FP32 precision.
    // Without this, every MoE layer does a FP16 elementwise_add and the downstream
    // forced sync (executor_forward.cu:373-381) clobbers the FP32 accum with
    // FP16-rounded data, accumulating ~1-2% drift per layer over 30 layers.
    const bool moe_fp32_accum = (cfg.arch == ModelArch::GEMMA4 && ly.post_ffn_norm.data != nullptr &&
                                 fp32_accum_buf_ != nullptr && !residual_fused);
    if (moe_fp32_accum) {
        // fp32_hidden_ holds the pre-MoE residual (written by run_attention's
        // FP32 accum path). Kernel: fp32_h += rmsnorm(h) * post_ffn_norm; h = half(fp32_h).
        Tensor fp32_h = view_tokens(fp32_hidden_, n);
        rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
            static_cast<const half*>(h.data), static_cast<const half*>(ly.post_ffn_norm.data),
            static_cast<float*>(fp32_h.data), static_cast<half*>(h.data), cfg.d_model, eps, norm_w_off_);
        if (layer == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_combined_post_post_ffn_norm_fp32accum", layer);
            debug_tensor_stats_all(buf, h, stream);
        }
    } else {
        if (cfg.arch == ModelArch::GEMMA4 && ly.post_ffn_norm.data != nullptr) {
            rmsnorm(h, ly.post_ffn_norm, h, eps, stream, norm_w_off_);
        }
        if (layer == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_combined_post_post_ffn_norm", layer);
            debug_tensor_stats_all(buf, h, stream);
        }

        // 9. Residual connection: hidden += residual
        //    Skipped when decode fast path already fused residual into weighted_sum.
        if (!residual_fused) {
            elementwise_add(h, r, stream);
        }
    }
    if (layer == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "L%d_after_residual_pre_scale", layer);
        debug_tensor_stats_all(buf, h, stream);
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

}  // namespace imp
// Sun Apr 12 06:27:12 PM CEST 2026

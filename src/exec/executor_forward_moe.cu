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

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_forward_moe_internal.h"
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

namespace {

static bool can_decode_fast(int n, const Tensor& expert_up_packed, QType up_qtype, void* dequant_buf,
                            QType compute_dtype) {
    return (n == 1 && expert_up_packed.data != nullptr && dequant_buf != nullptr &&
            compute_dtype == QType::F16 && expert_up_packed.on_device &&
            (up_qtype == QType::Q6_K || up_qtype == QType::Q8_0 || up_qtype == QType::Q4_0 ||
             up_qtype == QType::Q4_K || up_qtype == QType::Q5_K || up_qtype == QType::Q2_K ||
             up_qtype == QType::Q3_K || up_qtype == QType::Q5_1 || up_qtype == QType::NVFP4));
}

}  // anonymous namespace

void GraphExecutor::moe_ffn_phase1_setup_(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    // Phase 4 (MoE host-offload async prefetch). The cache is only initialised
    // when some experts are host-resident; n_slots_ > 0 is therefore the
    // gate. Order matters: drain this layer's pending prefetch before
    // reading the cache, then queue the next layer's prefetch so the
    // prefetch stream gets compute-time overlap.
    if (expert_cache_.n_slots_ > 0) {
        const int top_k_prefetch = runtime_config().moe.prefetch_top_k;
        if (top_k_prefetch > 0) {
            expert_cache_.await_prefetch(layer, stream);
            const int next_layer = layer + 1;
            if (next_layer < model_->config().n_layers) {
                expert_cache_.prefetch_layer(next_layer, top_k_prefetch,
                                             expert_cache_.slot_size_);
            }
        }
    }

    // DIAGNOSTIC (Phase 2 Item 2 follow-up): zero MoE workspace buffers so any
    // legacy-serial-fallback uninit reads become deterministic zero reads.
    // Set IMP_MOE_ZERO_WORKSPACE=1 to enable. Cheap (~1 MiB total memset).
    if (runtime_config().moe.zero_workspace) {
        cudaMemsetAsync(moe_.expert_gate.data, 0, moe_.expert_gate.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_up.data, 0, moe_.expert_up.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_swiglu.data, 0, moe_.expert_swiglu.nbytes(), stream);
        cudaMemsetAsync(moe_.expert_down.data, 0, moe_.expert_down.nbytes(), stream);
        cudaMemsetAsync(moe_.gathered.data, 0, moe_.gathered.nbytes(), stream);
    }
}

void GraphExecutor::moe_ffn_phase2_state_and_norm_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    // Populate per-call context. Subsequent phase helpers read ctx directly.
    ctx.n        = cur_n_tokens_;
    ctx.d        = cfg.d_model;
    ctx.ne       = cfg.n_experts;
    ctx.top_k    = cfg.n_experts_active;
    ctx.eff      = max_expert_eff_;
    ctx.eps      = cfg.rms_norm_eps;
    ctx.es       = dtype_size(compute_dtype_);
    ctx.expanded = ctx.n * ctx.top_k;

    ctx.h  = view_tokens(hidden_, ctx.n);
    ctx.r  = view_tokens(residual_, ctx.n);
    ctx.no = view_tokens(norm_out_, ctx.n);

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
    if (ctx.n == 1 && compute_dtype_ == QType::F16) {
        bool has_up = (ly.nvfp4_moe_up_ptr != nullptr);
        bool has_down = (ly.nvfp4_moe_down_ptr != nullptr);
        ctx.nvfp4_covers_layer = has_up && has_down;
        if (ctx.nvfp4_covers_layer && ly.expert_gate_packed.data != nullptr) {
            ctx.nvfp4_covers_layer = (ly.nvfp4_moe_gate_ptr != nullptr);
        }
    }

    // Pre-check decode fast path (same logic as will_decode_fast below)
    ctx.will_skip_residual_copy = can_decode_fast(ctx.n, ly.expert_up_packed, ly.expert_up_packed.qtype,
                                                  moe_.dequant_buf, compute_dtype_) &&
                                  ly.w_up_shared.data ==
                                      nullptr;  // must not have shared expert for full residual fusion

    if (!ctx.will_skip_residual_copy) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ctx.r.data, ctx.h.data, ctx.h.nbytes(),
                                           cudaMemcpyDeviceToDevice, stream));
    }
    // Fused RMSNorm + Q8_1: skip for NVFP4-covered layers (NVFP4 takes FP16 directly)
    // Gemma-4 with FP32 accum: compute norm from FP32 residual, then quantize to Q8_1
    // separately. The fused kernel reads FP16 h which loses ~0.03% per element that
    // compounds catastrophically through the 128-expert top-8 MoE routing.
    ctx.gemma4_fp32_norm = (cfg.arch == ModelArch::GEMMA4 && fp32_accum_buf_ != nullptr);
    // When FP32 residual accumulator is active AND post_ffn_norm exists, defer the
    // residual add to rmsnorm_fp32_accum_to_fp16_kernel (which keeps fp32_hidden_ in
    // sync + applies overflow scaling). Without this, moe_weighted_sum_residual
    // adds residual in FP16 and the shadow goes stale — measured ~7% drift at L3
    // that compounds to 260% by L29 vs llama.cpp (docs/gemma4_layer_diff.md).
    ctx.moe_use_fp32_residual = (cfg.arch == ModelArch::GEMMA4 && fp32_accum_buf_ != nullptr &&
                                 ly.post_ffn_norm.data != nullptr);
    // Diagnostic: keep MoE down-projection output in FP32 (no FP16 truncation
    // at GEMM output) to isolate precision drift. Allocated below in the FP16
    // batch path; freed at moe_after_experts. Other prefill paths ignore this.
    ctx.fp32_down_active = (cfg.arch == ModelArch::GEMMA4 &&
                            cfg.overrides.gemma4.fp32_expert_down);
    ctx.fp32_down_buf = nullptr;
    ctx.moe_fused_norm_q8 = (ctx.n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                             ctx.h.qtype == QType::F16 && !ctx.nvfp4_covers_layer &&
                             !ctx.gemma4_fp32_norm);
    if (ctx.moe_fused_norm_q8) {
        // Fused: RMSNorm + Q8_1 (also writes FP16 norm_out for gate logits)
        rmsnorm_quantize_q8_1(static_cast<const half*>(ctx.h.data),
                              static_cast<const half*>(norm_w.data),
                              static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf,
                              static_cast<half*>(ctx.no.data), ctx.d, ctx.eps, stream, norm_w_off_);
    } else {
        // Gemma-4 FP32 accum: read FP32 residual directly to avoid FP16 round-trip.
        if (ctx.gemma4_fp32_norm) {
            Tensor fp32_h = view_tokens(fp32_hidden_, ctx.n);
            rmsnorm_fp32_to_fp16(fp32_h, norm_w, ctx.no, ctx.eps, stream, norm_w_off_);
            // Populate Q8_1 buffer from FP16 norm_out for dp4a decode path
            if (ctx.n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr) {
                quantize_fp16_to_q8_1(static_cast<const half*>(ctx.no.data),
                                      static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf,
                                      ctx.d, stream);
            }
        } else {
            rmsnorm(ctx.h, norm_w, ctx.no, ctx.eps, stream, norm_w_off_);
        }
    }
}

void GraphExecutor::moe_ffn_phase3_route_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    // 3. Gate logits + top-k routing
    Tensor router_in = ctx.no;
    // Gemma-4 custom router: logits = gate_inp @ (rmsnorm_noweight(h) * (1/sqrt(d)) * gate_inp_scale)
    // This matches llama.cpp's gemma4-iswa.cpp:151-155.
    // The standard path (router_in = rmsnorm(h, ffn_pre_norm_2)) uses the WRONG norm weight
    // and produces ~2x too small router logits, causing wrong expert selection.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_gate_inp_scale.data != nullptr) {
        // Gemma-4 custom router: logits = gate_inp @ (rmsnorm_noweight(h) * scale * (1/sqrt(d)))
        // Keep router_in in FP32 to prevent precision loss that causes routing instability
        // at later layers (L29). The FP16 intermediate loses enough precision to change
        // expert selection in the 128-expert top-8 MoE.
        if (fp32_accum_buf_ != nullptr && ctx.n == 1) {
            // FP32 router path: rmsnorm(fp32_h, gate_inp_scale) * 1/sqrt(d) → FP32
            // Stays in FP32 all the way through gate GEMV (no FP16 truncation).
            float* fp32_router = static_cast<float*>(moe_.scatter_out.data);
            Tensor fp32_h = view_tokens(fp32_hidden_, ctx.n);
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(ctx.d));
            rmsnorm_fp32_to_fp32(fp32_h, ly.ffn_gate_inp_scale, fp32_router, ctx.n, ctx.d, ctx.eps,
                                 stream, 0.0f);
            {
                int64_t total = static_cast<int64_t>(ctx.n) * ctx.d;
                int thr = 256;
                int blk = static_cast<int>((total + thr - 1) / thr);
                scale_fp32_kernel<<<blk, thr, 0, stream>>>(fp32_router, inv_sqrt_d, total);
            }
            // Gate GEMV directly from FP32 router → FP32 logits (no FP16 truncation).
            // Writes to moe_.gate_logits — same buffer the topk gating reads from.
            {
                Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, ctx.n);
                gemv_gate_fp32_fp32input(static_cast<const half*>(ly.moe_gate.data), fp32_router,
                                         static_cast<float*>(gate_logits_f32.data), ctx.ne, ctx.d, stream);
            }
            ctx.fp32_gate_logits_ready = true;
        } else {
            // FP16 fallback (prefill or non-FP32-accum)
            int64_t ri_shape[2] = {static_cast<int64_t>(ctx.n), static_cast<int64_t>(ctx.d)};
            router_in = Tensor(moe_.scatter_out.data, compute_dtype_, 2, ri_shape, true);
            if (fp32_accum_buf_ != nullptr) {
                Tensor fp32_h = view_tokens(fp32_hidden_, ctx.n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.ffn_gate_inp_scale, router_in, ctx.eps, stream, 0.0f);
            } else {
                rmsnorm(ctx.h, ly.ffn_gate_inp_scale, router_in, ctx.eps, stream, 0.0f);
            }
            int64_t total_elems = static_cast<int64_t>(ctx.n) * ctx.d;
            int threads_s = 256;
            int blocks_s = static_cast<int>((total_elems / 2 + threads_s - 1) / threads_s);
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(ctx.d));
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
    bool use_sigmoid  = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;

    ctx.up_qtype         = ly.expert_up_packed.qtype;
    ctx.will_decode_fast = can_decode_fast(ctx.n, ly.expert_up_packed, ctx.up_qtype, moe_.dequant_buf,
                                           compute_dtype_);
    // Gemma 4: dp4a decode fast path ENABLED by default. dp4a matches llama's
    // Q4_K×Q8_1 accumulation for MoE experts, preventing the routing drift that
    // occurs with FP16 dequant+cuBLAS. Set IMP_G4_NO_DECODE_FAST=1 to disable.
    if (cfg.arch == ModelArch::GEMMA4 && cfg.overrides.gemma4.no_decode_fast) {
        ctx.will_decode_fast = false;
    }

    compute_moe_routing(layer, stream, ctx.n, ctx.d, ctx.ne, ctx.top_k, router_in,
                        ctx.fp32_gate_logits_ready, ctx.will_decode_fast, router_bias_ptr,
                        use_sigmoid, norm_weights, ctx.routing);

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    ctx.use_packed_dequant = (ly.expert_up_packed.data != nullptr && moe_.dequant_buf != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    ctx.non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                             (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (ctx.use_packed_dequant) {
        int64_t ref_eff = ctx.non_gated_experts ? ly.expert_up_packed.shape[1]
                                                : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != ctx.eff || down_eff != ctx.eff) {
            IMP_LOG_ERROR(
                "CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
                "down_packed.shape[2]=%ld. Using packed tensor shapes instead.",
                ctx.eff, (long)ref_eff, (long)down_eff);
            ctx.eff = static_cast<int>(ref_eff);
        }
    }
}

// Cheap precondition mirror for the CUTLASS 3.x NVFP4 device-args fast path
// inside try_run_moe_cutlass3x_nvfp4_prefill_ (lines ~1316–1372). Keep the
// two predicates in sync — if device-args' actual gate flips false at runtime
// while this returns true, the legacy fallback inside the function gathers
// lazily, so output stays correct (at most one wasted decision).
// The path-selection ORDER + arch/config gates are mirrored as a pure function
// `select_moe_prefill_path` in moe_prefill_decision.h, pinned by
// test_routing_decision.cpp (R2 / P1.4).
bool GraphExecutor::moe_cutlass3x_will_use_device_args_(int layer,
                                                        const MoeFfnContext& ctx) const {
    if (runtime_config().moe.no_cutlass3x)
        return false;
    // gpt-oss: device-args path is arch-gated off (no GLU/bias hooks in the
    // fused act+quantize kernel) — keep the mirror in sync.
    if (model_->config().arch == ModelArch::GPT_OSS)
        return false;
    if (!cutlass_grouped_3x_nvfp4_available())
        return false;
    if (!moe_.cutlass3x_packed || !moe_.cutlass3x_sf)
        return false;
    if (!runtime_config().moe.nvfp4_device_args)
        return false;
    if (!moe_.d_M_per || moe_.d_M_per_count < ctx.ne)
        return false;
    if (!moe_.d_sfa_offsets || !moe_.d_B_ptrs_cache || !moe_.d_SFB_ptrs_cache || !moe_.d_alpha_full)
        return false;
    if (!moe_.cutlass3x_sfa_ptrs || moe_.cutlass3x_sfa_ptrs_count < ctx.ne)
        return false;
    // All expert tensors must be CUTLASS_NVFP4 — covers_ids() in the inner
    // function. Mirrored here to avoid the upstream skip-gather firing when
    // the inner path would refuse to take this layer.
    const auto& ly = model_->layer(layer);
    auto covers_ids = [&](const std::vector<TensorID>& ids) {
        if (static_cast<int>(ids.size()) < ctx.ne)
            return false;
        for (int e = 0; e < ctx.ne; ++e) {
            if (ids[e] == kInvalidTensorID)
                return false;
            if (registry_.handle(ids[e]).primary_tier != StorageTier::CUTLASS_NVFP4)
                return false;
        }
        return true;
    };
    if (!covers_ids(ly.expert_up_ids))
        return false;
    if (!covers_ids(ly.expert_down_ids))
        return false;
    if (!ctx.non_gated_experts && !covers_ids(ly.expert_gate_ids))
        return false;
    return true;
}

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    MoeFfnContext ctx;

    moe_ffn_phase1_setup_(layer, stream);
    moe_ffn_phase2_state_and_norm_(layer, stream, ctx);

    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    // Local references into ctx so the remaining body reads unchanged.
    int&     n        = ctx.n;
    int&     d        = ctx.d;
    int&     ne       = ctx.ne;
    int&     top_k    = ctx.top_k;
    int&     eff      = ctx.eff;
    float&   eps      = ctx.eps;
    size_t&  es       = ctx.es;
    int&     expanded = ctx.expanded;
    Tensor&  h        = ctx.h;
    Tensor&  r        = ctx.r;
    Tensor&  no       = ctx.no;
    bool&    residual_fused          = ctx.residual_fused;
    bool&    nvfp4_covers_layer      = ctx.nvfp4_covers_layer;
    bool&    will_skip_residual_copy = ctx.will_skip_residual_copy;
    bool&    moe_fused_norm_q8       = ctx.moe_fused_norm_q8;
    bool&    moe_use_fp32_residual   = ctx.moe_use_fp32_residual;
    void*&   fp32_down_buf           = ctx.fp32_down_buf;
    const bool& fp32_down_active     = ctx.fp32_down_active;

    moe_ffn_phase3_route_(layer, stream, ctx);
    QType&             up_qtype          = ctx.up_qtype;
    bool&              will_decode_fast  = ctx.will_decode_fast;
    MoeRoutingResult&  routing           = ctx.routing;
    bool&              non_gated_experts = ctx.non_gated_experts;
    bool&              use_packed_dequant = ctx.use_packed_dequant;

    // Recompute the FFN-input norm weight (same selection as phase 2). Only the
    // Gemma-4 ggml_prefill path consumes it from here; the RMSNorm itself
    // already ran inside moe_ffn_phase2_state_and_norm_. Declared BEFORE the
    // will_decode_fast goto so the jump does not bypass its initialization.
    const Tensor& norm_w = (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr)
                               ? ly.ffn_pre_norm_2
                           : (ly.ffn_norm.data != nullptr)       ? ly.ffn_norm
                           : (ly.post_attn_norm.data != nullptr) ? ly.post_attn_norm
                                                                 : ly.attn_norm;

    // Decode fast path (n=1, device-resident packed experts): skips
    // gather/scatter + D2H sync. NVFP4 and dp4a/FP16 sub-paths handled
    // internally. Always exits via moe_after_experts.
    if (will_decode_fast) {
        run_moe_decode_fast(layer, stream, n, d, eff, top_k, routing, no, h, r,
                            moe_use_fp32_residual, moe_fused_norm_q8,
                            will_skip_residual_copy, residual_fused);
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
        // gemm.moe_imma_prefill: the grouped INT8 IMMA batch path (fused
        // dequant on tensor cores) supersedes the per-expert fused WMMA/dp4a
        // prefill variants AND the gemma-4 ggml per-token path — skip them so
        // dispatch falls through to the batch path, which tries IMMA per
        // expert tensor. Measured 2026-06-07: Qwen3-30B-A3B pp512 3 968 →
        // 9 970 tok/s (above llama.cpp).
        const bool moe_imma_pref = runtime_config().gemm.moe_imma_prefill;
        if (!moe_imma_pref &&
            try_run_moe_q6k_prefill(layer, stream, n, d, eff, ne, expanded,
                                    non_gated_experts, up_qtype, routing, no)) {
            // Falls through to scatter (step 7)
        // Q4_K/Q5_K fused dp4a prefill: wins when expert_d_ff is small enough
        // that the 3.5× bandwidth savings from reading Q4_K directly outweighs
        // cuBLAS's tiled GEMM efficiency. Measured: +20% at eff=512 (Qwen3.6),
        // -20% at eff=768 (Qwen3-30B). Threshold: eff ≤ 640.
        } else if (eff <= 640 && !moe_imma_pref &&
                   try_run_moe_q4k_prefill(layer, stream, n, d, eff, ne, expanded,
                                            non_gated_experts, up_qtype, routing, no)) {
            // Falls through to scatter (step 7)
        } else if (!moe_imma_pref && cfg.overrides.gemma4.ggml_prefill &&
                   cfg.arch == ModelArch::GEMMA4 &&
                   ly.expert_gate_packed.on_device && ly.expert_up_packed.on_device &&
                   ly.expert_down_packed.on_device &&
                   try_run_moe_gemma4_ggml_prefill(layer, stream, n, d, eff, top_k, up_qtype, eps,
                                                   routing, no, norm_w, h, r,
                                                   moe_use_fp32_residual, residual_fused)) {
            goto moe_after_experts;
        } else {
            // =========================================================================
            // FP16 BATCH or FP8 BATCH PREFILL PATH
            // Pre-check: FP16 batch + device-grouped GEMM is preferred (no D2H sync,
            // simpler pipeline). FP8 batch is only used as fallback when FP16 batch
            // isn't available.
            // =========================================================================

            // Gather: reorder tokens by expert assignment (required for batch/legacy paths).
            // Skipped when the CUTLASS 3.x NVFP4 device-args fast path will fire — that
            // path reads ctx.no via routing.sorted_token_ids in a fused gather+quantize
            // kernel and never touches moe_.gathered. Lazy gather inside the legacy
            // fallback catches the (rare) case where device-args' inner gate flips false
            // at runtime. Saves ~16 MB HBM write per layer on Qwen3-Coder-30B-A3B-NVFP4.
            if (moe_cutlass3x_will_use_device_args_(layer, ctx)) {
                ctx.moe_gather_done = false;
            } else {
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
                    try_run_moe_fp16_batch_prefill(layer, stream, n, d, eff, ne, expanded,
                                                   non_gated_experts, up_qtype, routing,
                                                   fp32_down_active, fp32_down_buf);
                } else if (can_fp8_batch) {
                    try_run_moe_fp8_batch_prefill(layer, stream, n, d, eff, ne, expanded,
                                                  non_gated_experts, up_qtype, routing);

                    // Falls through to scatter (step 7)

                } else if (try_run_moe_cutlass3x_nvfp4_prefill_(layer, stream, ctx)) {
                    // Falls through to scatter (step 7)

                } else if (try_run_moe_nvfp4_dequant_batch_prefill_(layer, stream, ctx)) {
                    // Falls through to scatter (step 7)

                } else {
                    run_moe_legacy_fallback_(layer, stream, ctx);
                }
            }  // else of can_fp16/fp8_batch/legacy
        }  // FP8/FP16 prefill scope
    }  // else branch of can_fused_q6k + fused Q6_K scope

    moe_ffn_phase7_scatter_(layer, stream, ctx);

moe_after_experts:
    moe_ffn_phase8_post_(layer, stream, ctx);
}

// ---------------------------------------------------------------------------
// Phase 7: scatter expert outputs back to token positions.
//   Fused path: token-centric scatter + FP16 convert (+ residual if no shared expert).
//   Fallback:  atomicAdd scatter into FP32 buffer + FP32→FP16 convert.
// ---------------------------------------------------------------------------
void GraphExecutor::moe_ffn_phase7_scatter_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    (void)layer;
    const auto& ly = model_->layer(layer);

    bool has_shared_expert = (ly.w_up_shared.data != nullptr);
    if (ctx.routing.token_to_expanded && compute_dtype_ == QType::F16) {
        // Fused token-centric scatter: no atomics, no FP32 intermediate buffer.
        // If no shared expert, also fuse residual add.
        // Gemma-4 FP32 path: defer residual to post_ffn_norm (FP32 accumulator).
        const void* res_ptr = (!has_shared_expert && !ctx.residual_fused && !ctx.moe_use_fp32_residual)
                                  ? ctx.r.data : nullptr;
        if (ctx.fp32_down_buf) {
            // Down GEMM kept FP32 output — use FP32-input scatter variant.
            moe_scatter_fused_residual_fp32in(ctx.fp32_down_buf, ctx.routing.token_to_expanded,
                                              static_cast<const float*>(ctx.routing.expert_weights.data),
                                              res_ptr, ctx.h.data, ctx.n, ctx.d, ctx.top_k, stream);
        } else {
            moe_scatter_fused_residual(moe_.expert_down.data, ctx.routing.token_to_expanded,
                                       static_cast<const float*>(ctx.routing.expert_weights.data),
                                       res_ptr, ctx.h.data, ctx.n, ctx.d, ctx.top_k, stream);
        }
        if (!has_shared_expert && !ctx.moe_use_fp32_residual)
            ctx.residual_fused = true;
    } else {
        // Fallback: atomicAdd scatter into FP32 buffer, then convert
        int64_t expert_out_shape[2] = {static_cast<int64_t>(ctx.expanded), static_cast<int64_t>(ctx.d)};
        Tensor expert_down_view(moe_.expert_down.data, compute_dtype_, 2, expert_out_shape, true);
        Tensor scatter_out = slice_rows(moe_.scatter_out, ctx.n);
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(scatter_out.data, 0,
                                           static_cast<size_t>(ctx.n) * ctx.d * sizeof(float), stream));
        moe_scatter(expert_down_view, ctx.routing, scatter_out, stream);

        int64_t numel = static_cast<int64_t>(ctx.n) * ctx.d;
        int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute_dtype_ == QType::F16) {
            fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(moe_.scatter_out.data),
                static_cast<half*>(ctx.h.data), numel);
        } else {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ctx.h.data, moe_.scatter_out.data,
                                               static_cast<size_t>(numel) * sizeof(float),
                                               cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 8: post-experts work after `moe_after_experts:` —
//   Gemma-4 sanitize / post_ffw_norm_2 / shared expert FFN /
//   post_ffn_norm (FP32-accum or plain variant) / residual add /
//   debug stats / free routing buffers if owned.
// ---------------------------------------------------------------------------
void GraphExecutor::moe_ffn_phase8_post_(int layer, cudaStream_t stream, MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    // Free diagnostic FP32 expert-down buffer if we malloc'd it ourselves.
    // The persistent moe_.fp32_down_buf is owned by MoEWorkspace — don't free.
    if (ctx.fp32_down_buf && ctx.fp32_down_buf != moe_.fp32_down_buf) {
        cudaFreeAsync(ctx.fp32_down_buf, stream);
    }
    ctx.fp32_down_buf = nullptr;
    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    // Gemma 4: sanitize inf/NaN in MoE scatter output before post-norm.
    if (cfg.arch == ModelArch::GEMMA4) {
        sanitize_fp16(static_cast<__half*>(ctx.h.data), static_cast<int64_t>(ctx.n) * ctx.d, stream);
    }

    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_moe_scatter_out", view_tokens(ctx.h, ctx.n), stream);
    }
    // Gemma 4: apply post_ffw_norm_2 on the MoE branch output (h) BEFORE shared adds.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_2.data != nullptr) {
        rmsnorm(ctx.h, ly.ffn_post_norm_2, ctx.h, ctx.eps, stream, norm_w_off_);
    }
    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_moe_post_norm2", view_tokens(ctx.h, ctx.n), stream);
    }

    // Gemma 4: re-derive `no` for the shared MLP from the saved residual
    // (which still holds the original hidden state) using ffn_norm — the MoE
    // branch consumed `no` produced from pre_ffw_norm_2 above.
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_pre_norm_2.data != nullptr &&
        ly.w_up_shared.data != nullptr && ly.ffn_norm.data != nullptr) {
        rmsnorm(ctx.r, ly.ffn_norm, ctx.no, ctx.eps, stream, norm_w_off_);
    }

    run_shared_expert_ffn(layer, stream, ctx.n, ctx.d, ctx.eps, ctx.no, ctx.h);
    if (debug_forward_enabled() && layer == 0) {
        debug_tensor_rows("L0_combined", view_tokens(ctx.h, ctx.n), stream);
    }
    if (layer == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "L%d_combined_pre_post_ffn_norm", layer);
        debug_tensor_stats_all(buf, ctx.h, stream);
    }

    // Gemma 4: apply post_ffn_norm (combined post-norm) BEFORE residual add.
    // If FP32 accumulator is active, fuse post_ffn_norm + residual add into
    // rmsnorm_fp32_accum_to_fp16_kernel so the residual stays in FP32 precision.
    // Without this, every MoE layer does a FP16 elementwise_add and the downstream
    // forced sync (executor_forward.cu:373-381) clobbers the FP32 accum with
    // FP16-rounded data, accumulating ~1-2% drift per layer over 30 layers.
    const bool moe_fp32_accum = (cfg.arch == ModelArch::GEMMA4 && ly.post_ffn_norm.data != nullptr &&
                                 fp32_accum_buf_ != nullptr && !ctx.residual_fused);
    if (moe_fp32_accum) {
        // fp32_hidden_ holds the pre-MoE residual (written by run_attention's
        // FP32 accum path). Kernel: fp32_h += rmsnorm(h) * post_ffn_norm; h = half(fp32_h).
        Tensor fp32_h = view_tokens(fp32_hidden_, ctx.n);
        rmsnorm_fp32_accum_to_fp16_kernel<<<ctx.n, 256, 0, stream>>>(
            static_cast<const half*>(ctx.h.data), static_cast<const half*>(ly.post_ffn_norm.data),
            static_cast<float*>(fp32_h.data), static_cast<half*>(ctx.h.data), cfg.d_model, ctx.eps,
            norm_w_off_);
        if (layer == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_combined_post_post_ffn_norm_fp32accum", layer);
            debug_tensor_stats_all(buf, ctx.h, stream);
        }
    } else {
        if (cfg.arch == ModelArch::GEMMA4 && ly.post_ffn_norm.data != nullptr) {
            rmsnorm(ctx.h, ly.post_ffn_norm, ctx.h, ctx.eps, stream, norm_w_off_);
        }
        if (layer == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_combined_post_post_ffn_norm", layer);
            debug_tensor_stats_all(buf, ctx.h, stream);
        }

        // 9. Residual connection: hidden += residual
        //    Skipped when decode fast path already fused residual into weighted_sum.
        if (!ctx.residual_fused) {
            elementwise_add(ctx.h, ctx.r, stream);
        }
    }
    if (layer == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "L%d_after_residual_pre_scale", layer);
        debug_tensor_stats_all(buf, ctx.h, stream);
    }

    // 10. Free routing result tensors only if allocated by moe_topk_gating.
    //     When using pre-allocated buffers, memory belongs to moe_.routing_buffers.
    if (ctx.routing.owns_memory) {
        IMP_CUDA_CHECK_LOG(cudaFree(ctx.routing.expert_indices.data));
        IMP_CUDA_CHECK_LOG(cudaFree(ctx.routing.expert_weights.data));
        IMP_CUDA_CHECK_LOG(cudaFree(ctx.routing.sorted_token_ids.data));
        IMP_CUDA_CHECK_LOG(cudaFree(ctx.routing.expert_offsets.data));
    }
}

}  // namespace imp

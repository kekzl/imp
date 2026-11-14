// Small-M NVFP4 dispatch family, moved VERBATIM out of
// executor_gemm_dispatch.cu (that TU sat at the 600-LOC kernel hard
// threshold): producer-side activation quantize, its rmsnorm/swiglu
// wrappers, and the sibling-pair dispatch, split to isolate smallm edits from re-ptxas-ing the whole dispatch
// chain.

#include "exec/executor.h"
#include "exec/gemm_context.h"
#include "quant/nvfp4_gemm.h"
#include "exec/smallm_policy.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_quant.h"
#include "compute/layernorm.h"
#include "exec/executor_debug.h"
#include "exec/executor_kernels.h"
#include "compute/activation.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

namespace {

// Kernel for one sibling launch on the combined N: ws, or v2 in one stripe; kNone declines (v2 would split K
// per tensor, or no kernel takes the shapes).
SmallMKernel sibling_kernel(int impl, const SmallMV2Sibling* sib, int count, int K) {
    int n_total = 0;
    bool all_n64 = true;
    bool all_ws = true;
    for (int i = 0; i < count; ++i) {
        n_total += sib[i].N;
        all_n64 = all_n64 && (sib[i].N % 64) == 0;
        all_ws = all_ws && gemm_nvfp4_smallm_ws_supports(sib[i].N, K);
    }
    const SmallMKernel k = smallm_kernel_for(impl, n_total, K, all_n64, all_ws, /*siblings=*/true);
    if (k == SmallMKernel::kWs || (k == SmallMKernel::kV2 && gemm_nvfp4_smallm_v2_stripes(n_total, K) == 1))
        return k;
    return SmallMKernel::kNone;
}

}  // namespace

bool GraphExecutor::smallm_weight_(const WeightHandle& h, NvFP4QuantResult& out) const {
    if (h.source_data == nullptr)
        return false;
    if (dequant_gpu_supported(h.source_qtype)) {
        // GGUF source (#1897): decode rows read the NVFP4 decode overlay, the same
        // entry the M=1 decode GEMV reads. A GGUF weight with 2+ rows used to take
        // the prefill route and dequantize the whole Q*_K source per step. Prompt
        // rows never take this: the 4-bit overlay would replace full-precision prefill.
        if (!cur_decode_rows_)
            return false;
        const StorageTier decode = (h.decode_tier != StorageTier::Undefined) ? h.decode_tier : h.primary_tier;
        if (decode != StorageTier::NVFP4)
            return false;
        auto it = wcache_.nvfp4.find(h.source_data);
        if (it == wcache_.nvfp4.end() || (it->second.K % 64) != 0)
            return false;
        out = it->second;
        return true;
    }
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4 || h.source_scales == nullptr ||
        ((h.shape[1] * 2) % 64) != 0)
        return false;
    out.packed_data = const_cast<void*>(h.source_data);
    out.micro_scales = h.source_scales;
    out.tensor_scale = h.source_tensor_scale;
    out.N = h.shape[0];
    out.K = h.shape[1] * 2;
    return true;
}

void GraphExecutor::allocate_smallm_scratch(cudaStream_t stream) {
    // Size for decode rows: that is when the GGUF overlay arm is live.
    const bool saved = cur_decode_rows_;
    cur_decode_rows_ = true;
    size_t ws_need = 0;
    size_t xq_need = 0;
    int n_weights = 0;
    for (size_t id = 0; id < registry_.size(); ++id) {
        const WeightHandle& h = registry_.handle(static_cast<TensorID>(id));
        NvFP4QuantResult nv;
        // The LM head has its own batched paths (executor_forward.cu) and
        // would dominate the workspace (vocab x 32 floats).
        if (h.kind == TensorKind::LM_HEAD || !smallm_weight_(h, nv))
            continue;
        const int N = static_cast<int>(nv.N);
        const int K = static_cast<int>(nv.K);
        // A16 (gemm.nvfp4_smallm_a4=false) runs the dequant+HMMA kernel at every shape; the ws kernel needs
        // no workspace.
        const SmallMKernel kern = dispatch_policy().gemm.nvfp4_smallm_a4
                                      ? smallm_kernel_for(kSmallMImpl, N, K)
                                      : ((K % 128) == 0 ? SmallMKernel::kV1 : SmallMKernel::kNone);
        ws_need = std::max(ws_need, smallm_workspace_bytes(kern, N, K));
        xq_need = std::max(xq_need, ((size_t)32 * (K / 2)) + ((size_t)32 * (K / 16)));
        ++n_weights;
    }
    cur_decode_rows_ = saved;
    ws_need = std::max<size_t>(ws_need, 256);  // a non-empty take keeps the arena path when ws needs none
    if (n_weights == 0 || smallm_ws_bytes_ >= ws_need)
        return;
    auto ws = engine_arena().take_bytes(ws_need);
    auto xq = ws.empty() ? ws : engine_arena().take_bytes(xq_need);
    if (ws.empty() || xq.empty()) {
        // Arena closed or under-planned: the lazy growth path (cudaMalloc
        // outside capture) still serves eager forwards; captured ones fall
        // through to their non-small-M route.
        IMP_LOG_WARN(
            "small-M NVFP4 scratch: T2 arena refused %zu + %zu bytes, falling back to lazy allocation",
            ws_need, xq_need);
        ensure_smallm_ws_(ws_need, stream);
        ensure_smallm_xq_(xq_need, stream);
        return;
    }
    smallm_ws_ = ws.data();
    smallm_ws_bytes_ = ws_need;
    smallm_xq_ = xq.data();
    smallm_xq_bytes_ = xq_need;
    smallm_xq_src_ = nullptr;
    smallm_xq_from_producer_ = false;
    smallm_ws_arena_ = true;
    smallm_xq_arena_ = true;
    IMP_LOG_INFO(
        "small-M NVFP4 scratch: %d weights, workspace %zu KiB + activation %zu KiB from the T2 arena",
        n_weights, ws_need / 1024, xq_need / 1024);
}

uint8_t* GraphExecutor::smallm_producer_xq_(TensorID consumer_id, int M, int K, cudaStream_t stream,
                                            uint8_t** scales_out, float* act_scale_out, int k_align) {
    if (!dispatch_policy().gemm.nvfp4_smallm_a4)
        return nullptr;
    if (M < 2 || M > 32 || K <= 0 || (K % k_align) != 0)
        return nullptr;
    if (consumer_id == kInvalidTensorID)
        return nullptr;
    NvFP4QuantResult nv;
    if (!smallm_weight_(registry_.handle(consumer_id), nv) || static_cast<int>(nv.K) != K ||
        smallm_kernel_for(kSmallMImpl, static_cast<int>(nv.N), K) == SmallMKernel::kNone)
        return nullptr;
    const size_t xq_need = ((size_t)32 * (K / 2)) + ((size_t)32 * (K / 16));
    ensure_smallm_xq_(xq_need, stream);
    if (smallm_xq_bytes_ < xq_need)
        return nullptr;
    *scales_out = static_cast<uint8_t*>(smallm_xq_) + ((size_t)32 * (K / 2));
    *act_scale_out = act_scale_of_(consumer_id);
    return static_cast<uint8_t*>(smallm_xq_);
}

float GraphExecutor::act_scale_of_(TensorID id) const {
    return id == kInvalidTensorID ? 1.0f : registry_.handle(id).act_scale;
}

bool GraphExecutor::smallm_xq_holds_(const void* data, int M, int K, float act_scale) const {
    return smallm_xq_src_ == data && smallm_xq_src_m_ == M && smallm_xq_src_k_ == K &&
           smallm_xq_src_scale_ == act_scale;
}

void GraphExecutor::smallm_producer_tag_(const void* out_data, int M, int K, float act_scale) {
    smallm_xq_src_ = out_data;
    smallm_xq_src_m_ = M;
    smallm_xq_src_k_ = K;
    smallm_xq_src_scale_ = act_scale;
    smallm_xq_from_producer_ = true;
}

void GraphExecutor::rmsnorm_for_smallm_(const Tensor& h, const Tensor& w, Tensor& no,
                                        TensorID consumer_id, int n, float eps, cudaStream_t stream,
                                        float weight_offset) {
    // The producer already wrote no + the NVFP4 activations for this (no, w, n) (#2414).
    if (prenorm_matches_(w, no, n)) {
        prenorm_out_ = nullptr;
        return;
    }
    prenorm_out_ = nullptr;
    // No pre-norm weight: the block input is already normalised upstream (Qwen4Exp gated
    // residual hands the block a mixed, hc_norm'ed stream), so the norm is the identity.
    // Without this a null weight reaches the kernel and the process dies on an illegal access.
    if (w.data == nullptr) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(no.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    const int K = static_cast<int>(h.shape[1]);
    uint8_t* xq_scales = nullptr;
    float as = 1.0f;
    uint8_t* xq_packed = smallm_producer_xq_(consumer_id, n, K, stream, &xq_scales, &as);
    if (xq_packed != nullptr &&
        rmsnorm_nvfp4(h, w, no, xq_packed, xq_scales, eps, stream, weight_offset, as)) {
        smallm_producer_tag_(no.data, n, K, as);
        return;
    }
    rmsnorm(h, w, no, eps, stream, weight_offset);
    // The unfused write may have replaced the content behind a still-matching
    // tag (same buffer, same shape, new values) — invalidate it.
    if (smallm_xq_src_ == no.data && smallm_xq_src_m_ == n && smallm_xq_src_k_ == K)
        smallm_xq_from_producer_ = false;
}

bool GraphExecutor::residual_fusion_ok_(int n) const {
    return n >= 2 && n <= 32 && lora_ == nullptr && cvec_ == nullptr && sae_rows_.empty() &&
           !model_->profile().gated_residual && !model_->config().parallel_block &&
           fp32_accum_buf_ == nullptr && !debug_forward_enabled();
}

bool GraphExecutor::add_norm_for_smallm_(const Tensor& a, Tensor& r, Tensor& h, const Tensor& w, Tensor& no,
                                         TensorID consumer_id, int n, float eps, cudaStream_t stream) {
    if (w.data == nullptr || !residual_fusion_ok_(n))
        return false;
    const int K = static_cast<int>(h.shape[1]);
    uint8_t* xq_scales = nullptr;
    float as = 1.0f;
    uint8_t* xq_packed = smallm_producer_xq_(consumer_id, n, K, stream, &xq_scales, &as);
    if (xq_packed == nullptr ||
        !add_rmsnorm_nvfp4(a, r, h, w, no, xq_packed, xq_scales, eps, stream, norm_w_off_, as))
        return false;
    smallm_producer_tag_(no.data, n, K, as);
    prenorm_out_ = no.data;
    prenorm_w_ = w.data;
    prenorm_n_ = n;
    return true;
}

void GraphExecutor::save_residual_(Tensor& r, const Tensor& h, const Tensor& norm_w, const Tensor& no, int n,
                                   cudaStream_t stream) {
    if (!prenorm_matches_(norm_w, no, n))
        device_copy_async(r.data, h.data, h.nbytes(), stream);  // kernel copy — see device_copy_async
}

void GraphExecutor::ffn_residual_add_(int layer, const Tensor& fo, Tensor& r, Tensor& h, Tensor& no, int n,
                                      float eps, cudaStream_t stream) {
    const int nl = layer + 1;
    const bool fuse = nl < model_->config().n_layers && layer_has_attention(nl) && !layer_has_gdn(nl) &&
                      !layer_has_ssm(nl) && model_->layer(layer).layer_out_scale.data == nullptr;
    if (!(fuse && add_norm_for_smallm_(fo, r, h, model_->layer(nl).attn_norm, no, model_->layer(nl).wq_id, n,
                                       eps, stream)))
        elementwise_add_store(fo, r, h, stream);
}

bool GraphExecutor::swiglu_for_smallm_(const Tensor& go, const Tensor& uo, Tensor& so, TensorID consumer_id,
                                       int n, cudaStream_t stream, bool gelu) {
    if (swiglu_epi_.act != nullptr)  // the up GEMM's epilogue did it (swiglu_epi_eligible_)
        return false;
    const int K = static_cast<int>(so.shape[1]);
    // Prefill (#2470): the CUTLASS down GEMM's activation quantize rides in the SwiGLU kernel;
    // so is still written, so a declining dispatch falls back on the same bytes.
    if (!gelu && n > 32 && K % 16 == 0 && so.qtype == QType::F16 &&
        prefill_routes_cutlass_nvfp4_(consumer_id, n)) {
        const float as = act_scale_of_(consumer_id);
        swiglu_quantize_fp16_to_nvfp4_cutlass(go.data, uo.data, so.data, qscratch_.cutlass_act_data,
                                              qscratch_.cutlass_act_sf, n, K, stream, as);
        qscratch_.cutlass_act_scale = as;
        return true;
    }
    uint8_t* xq_scales = nullptr;
    float as = 1.0f;
    // GeGLU: K % 64 (the ws kernel's K; Gemma-4 shared down K 2112).
    uint8_t* xq_packed = smallm_producer_xq_(consumer_id, n, K, stream, &xq_scales, &as, gelu ? 64 : 256);
    if (xq_packed != nullptr &&
        (gelu ? geglu_quantize_nvfp4(go, uo, so, xq_packed, xq_scales, stream, as)
              : swiglu_quantize_nvfp4(go, uo, so, xq_packed, xq_scales, stream, as))) {
        smallm_producer_tag_(so.data, n, K, as);
        return false;
    }
    if (gelu)
        geglu(go, uo, so, stream);
    else
        swiglu(go, uo, so, stream);
    if (smallm_xq_src_ == so.data && smallm_xq_src_m_ == n && smallm_xq_src_k_ == K)
        smallm_xq_from_producer_ = false;
    return false;
}

// The down input in uo's buffer: packed [n, N/2], SfAtom scales from the next 128 B boundary.
static size_t swiglu_epi_sf_offset(int n, int N) {
    return ((static_cast<size_t>(n) * (N / 2)) + 127) & ~size_t(127);
}

// Eligible: SwiGLU prompt rows, gate / up / down on the CUTLASS_NVFP4 tile (not cuBLASLt) at this M (so the
// down takes run_ffn's plain branch), one act scale for gate and up, the down input fits uo's buffer, and
// nothing reads uo / so after the activation (LoRA, calibration, debug dumps).
bool GraphExecutor::swiglu_epi_eligible_(const TransformerLayer& ly, const Tensor& no, const Tensor& go,
                                         const Tensor& uo, int n) const {
    if (n <= 32 || n < imp::cfg::kNvfp4TileMinM || model_->config().ffn_activation != FFNActivation::SWIGLU ||
        lora_ != nullptr || calib_ || cur_spec_verify_ || debug_forward_enabled() || go.qtype != QType::F16 ||
        uo.qtype != QType::F16 || dequant_gpu_supported(ly.w_down.qtype))
        return false;
    for (const TensorID id : {ly.w_gate_id, ly.w_up_id, ly.w_down_id}) {
        if (id == kInvalidTensorID || !prefill_routes_cutlass_nvfp4_(id, n))
            return false;
        const auto& h = registry_.handle(id);
        if (h.primary_tier != StorageTier::CUTLASS_NVFP4 || h.payload.cutlass_nvfp4.sf == nullptr ||
            wcache_.cutlass_mxfp8_prefill.contains(h.source_data))
            return false;
    }
    const auto& hu = registry_.handle(ly.w_up_id);
    const int N = static_cast<int>(hu.shape[0]);
    return act_scale_of_(ly.w_gate_id) == act_scale_of_(ly.w_up_id) && N % 16 == 0 && uo.shape[1] == N &&
           no.shape[1] == hu.shape[1] * 2 &&
           swiglu_epi_sf_offset(n, N) + cutlass_nvfp4_sf_size(n, N) <= uo.nbytes();
}

void GraphExecutor::gate_up_prefill_(const TransformerLayer& ly, const Tensor& no, Tensor& go, Tensor& uo,
                                     int n, const GemmContext& ctx) {
    const int K = static_cast<int>(no.shape[1]);
    // NVFP4 prefill gate / up share the normed input: gate's dispatch quantizes it into the activation
    // scratch, up skips the re-quantize (act-quant hint).
    const bool hint = n > 1 && prefill_routes_cutlass_nvfp4_(ly.w_gate_id, n) &&
                      prefill_routes_cutlass_nvfp4_(ly.w_up_id, n);
    const bool epi = swiglu_epi_eligible_(ly, no, go, uo, n);
    gemm_via_handle_(ly.w_gate_id, no, go, ctx);
    if (epi) {
        // The registry route's alpha and workspace rule (gemm_kernel_cutlass_nvfp4.cpp): same plan, same
        // bits.
        const auto& hu = registry_.handle(ly.w_up_id);
        const auto& cu = hu.payload.cutlass_nvfp4;
        const int N = static_cast<int>(hu.shape[0]);
        const float alpha = (cu.global_scale ? *cu.global_scale : 1.0f) * hu.act_scale;
        const bool no_ws = process_diag_deterministic() || !process_diag_nvfp4_prompt_streamk();
        auto* q = static_cast<uint8_t*>(uo.data);
        uint8_t* sf = q + swiglu_epi_sf_offset(n, N);
        if (gemm_nvfp4_tile_swiglu_quant(qscratch_.cutlass_act_data, qscratch_.cutlass_act_sf, cu.weight,
                                         cu.sf, alpha, go.data, q, sf, act_scale_of_(ly.w_down_id), n, N, K,
                                         ctx.stream, no_ws ? nullptr : qscratch_.cutlass_workspace,
                                         qscratch_.cutlass_workspace_size)) {
            static bool logged = false;  // activity proof, once
            if (!logged) {
                logged = true;
                IMP_LOG_INFO(
                    "swiglu_quant_epilogue: SwiGLU + down NVFP4 quantize in the up GEMM ACTIVE "
                    "(M=%d N=%d K=%d)",
                    n, N, K);
            }
            swiglu_epi_ = {q, sf};
            return;
        }
    }
    gemm_via_handle_(ly.w_up_id, no, uo, hint ? ctx.with_act_quant_hint(no.data, n, K) : ctx);
}

void GraphExecutor::ffn_down_gemm_(TensorID id, const Tensor& so, Tensor& fo, const GemmContext& ctx) {
    if (swiglu_epi_.act != nullptr)
        gemm_nvfp4_prequant_(id, so, swiglu_epi_.act, swiglu_epi_.sf, fo, ctx);
    else
        gemm_via_handle_(id, so, fo, ctx);
}

bool GraphExecutor::try_smallm_pair_dispatch_(TensorID id_a, TensorID id_b, const Tensor& input,
                                              Tensor& out_a, Tensor& out_b, const GemmContext& ctx) {
    const TensorID ids[2] = {id_a, id_b};
    Tensor* outs[2] = {&out_a, &out_b};
    return try_smallm_multi_dispatch_(ids, outs, 2, input, ctx);
}

bool GraphExecutor::try_smallm_multi_dispatch_(const TensorID* ids, Tensor* const* outs, int count,
                                               const Tensor& input, const GemmContext& ctx) {
    // Mirror of the single-tensor smallm eligibility in gemm_via_handle_, applied to EVERY weight,
    // plus: same K, smallm_kernel_for() on the combined N (v2 only when single-stripe), fresh outputs
    // only. Every decline is a plain false; the caller issues the single dispatches.
    if (!dispatch_policy().gemm.nvfp4_smallm_a4 || ctx.spec_verify_small_m || ctx.beta != 0.0f || count < 2 ||
        count > kSmallMV2MaxSiblings)
        return false;
    const int M = static_cast<int>(input.shape[0]);
    // M==1 stays on the fused decode GEMVs; M>32 is prefill.
    if (M < 2 || M > 32 || input.qtype != QType::F16)
        return false;
    NvFP4QuantResult nv[kSmallMV2MaxSiblings];
    SmallMV2Sibling sib[kSmallMV2MaxSiblings];
    int K = 0;
    for (int i = 0; i < count; ++i) {
        if (ids[i] == kInvalidTensorID || outs[i] == nullptr || outs[i]->qtype != QType::F16)
            return false;
        if (!smallm_weight_(registry_.handle(ids[i]), nv[i]))
            return false;
        const int N = static_cast<int>(nv[i].N);
        if (i == 0)
            K = static_cast<int>(nv[i].K);
        // Output row stride must be N: the kernel writes y[m * N + n].
        if (static_cast<int>(nv[i].K) != K || outs[i]->shape[1] != N)
            return false;
        sib[i] = {&nv[i], reinterpret_cast<half*>(outs[i]->data), N};
    }
    const SmallMKernel kern = sibling_kernel(kSmallMImpl, sib, count, K);
    if (kern == SmallMKernel::kNone)
        return false;
    const size_t xq_need = ((size_t)32 * (K / 2)) + ((size_t)32 * (K / 16));
    ensure_smallm_xq_(xq_need, ctx.stream);
    if (smallm_xq_bytes_ < xq_need)
        return false;
    // Same statistic the single path records: every weight consumes `input`.
    if (calib_) {
        for (int i = 0; i < count; ++i)
            calib_->accumulate(cur_layer_, registry_.handle(ids[i]).kind, input, ctx.stream);
    }
    uint8_t* xq_packed = static_cast<uint8_t*>(smallm_xq_);
    uint8_t* xq_scales = xq_packed + ((size_t)32 * (K / 2));
    // One activation quantize serves every sibling: siblings share one act scale (merged at load).
    const float as = act_scale_of_(ids[0]);
    // Quantize dedupe — identical contract to the single-tensor block: a
    // matching scratch tag plus either the caller's act-quant hint or a
    // producer-side tag skips the re-quantize.
    const bool tag_match = smallm_xq_holds_(input.data, M, K, as);
    const bool hint_match = ctx.act_quant_hint_data != nullptr && ctx.act_quant_hint_data == input.data &&
                            ctx.act_quant_hint_m == M && ctx.act_quant_hint_k == K;
    if (!(tag_match && (hint_match || smallm_xq_from_producer_))) {
        quantize_fp16_to_nvfp4_into(input.data, M, K, xq_packed, xq_scales, /*tensor_scale=*/as, ctx.stream);
        smallm_xq_src_ = input.data;
        smallm_xq_src_m_ = M;
        smallm_xq_src_k_ = K;
        smallm_xq_src_scale_ = as;
        smallm_xq_from_producer_ = false;
    }
    NvFP4QuantResult xq;
    xq.packed_data = xq_packed;
    xq.micro_scales = xq_scales;
    xq.tensor_scale = as;
    xq.N = M;
    xq.K = K;
    return kern == SmallMKernel::kWs ? gemm_nvfp4_smallm_ws_multi(sib, count, xq, M, K, ctx.stream)
                                     : gemm_nvfp4_smallm_v2_multi_a4(sib, count, xq, M, K, ctx.stream);
}

}  // namespace imp

// executor_gemm_smallm.cu — the small-M NVFP4 dispatch family of
// GraphExecutor, moved VERBATIM out of executor_gemm_dispatch.cu on
// 2026-08-27 (that TU sat at the 600-LOC kernel hard threshold; this family
// — producer-side activation quantize, its rmsnorm/swiglu wrappers and the
// sibling-pair dispatch — is one coherent unit, and splitting it isolates
// smallm edits from re-ptxas-ing the whole dispatch chain).

#include "exec/executor.h"
#include "exec/gemm_context.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "compute/layernorm.h"
#include "compute/activation.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

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

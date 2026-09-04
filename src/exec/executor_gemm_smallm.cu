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

bool GraphExecutor::smallm_weight_(const WeightHandle& h, NvFP4QuantResult& out) const {
    if (h.source_data == nullptr)
        return false;
    if (dequant_gpu_supported(h.source_qtype)) {
        // GGUF source (#1897): decode rows read the NVFP4 decode overlay,
        // exactly the entry the M=1 decode GEMV reads. A GGUF weight with 2+
        // rows used to take the prefill route and dequantize the whole
        // Q*_K source per step (Qwen3-8B-Q8_0: 52 ms flat from two
        // sequences up, 8 streams slower than 1). Prompt rows never come
        // here: the 4-bit overlay would replace the full-precision prefill.
        if (!cur_decode_rows_)
            return false;
        const StorageTier decode = (h.decode_tier != StorageTier::Undefined) ? h.decode_tier : h.primary_tier;
        if (decode != StorageTier::NVFP4)
            return false;
        auto it = wcache_.nvfp4.find(h.source_data);
        if (it == wcache_.nvfp4.end() || (it->second.K % 128) != 0)
            return false;
        out = it->second;
        return true;
    }
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4 || h.source_scales == nullptr ||
        ((h.shape[1] * 2) % 128) != 0)
        return false;
    out.packed_data = const_cast<void*>(h.source_data);
    out.micro_scales = h.source_scales;
    out.tensor_scale = h.source_tensor_scale;
    out.N = h.shape[0];
    out.K = h.shape[1] * 2;
    return true;
}

void GraphExecutor::allocate_smallm_scratch(cudaStream_t stream) {
    if (!runtime_config().gemm.nvfp4_smallm)
        return;
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
        const bool v2 = runtime_config().gemm.nvfp4_smallm_impl == 2 && (K % 256) == 0 && (N % 64) == 0;
        ws_need = std::max(ws_need, v2 ? gemm_nvfp4_smallm_v2_workspace_bytes(N, K)
                                       : gemm_nvfp4_smallm_workspace_bytes(N));
        xq_need = std::max(xq_need, (size_t)32 * (K / 2) + (size_t)32 * (K / 16));
        ++n_weights;
    }
    cur_decode_rows_ = saved;
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
    smallm_arena_ = true;
    IMP_LOG_INFO(
        "small-M NVFP4 scratch: %d weights, workspace %zu KiB + activation %zu KiB from the T2 arena",
        n_weights, ws_need / 1024, xq_need / 1024);
}

uint8_t* GraphExecutor::smallm_producer_xq_(TensorID consumer_id, int M, int K, cudaStream_t stream,
                                            uint8_t** scales_out) {
    if (!runtime_config().gemm.nvfp4_smallm || cur_spec_verify_ || overlap_prefill_active_)
        return nullptr;
    if (M < 2 || M > 32 || K <= 0 || (K & 255) != 0)
        return nullptr;
    if (consumer_id == kInvalidTensorID)
        return nullptr;
    NvFP4QuantResult nv;
    if (!smallm_weight_(registry_.handle(consumer_id), nv) || static_cast<int>(nv.K) != K)
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
    NvFP4QuantResult nva;
    NvFP4QuantResult nvb;
    if (!smallm_weight_(ha, nva) || !smallm_weight_(hb, nvb))
        return false;
    const int K = static_cast<int>(nva.K);
    if (static_cast<int>(nvb.K) != K || (K % 256) != 0)
        return false;
    const int N1 = static_cast<int>(nva.N);
    const int N2 = static_cast<int>(nvb.N);
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

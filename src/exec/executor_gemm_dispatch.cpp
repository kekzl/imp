// executor_gemm_dispatch.cu — GEMM dispatch entry for the executor.
#include "core/dispatch_policy.h"
#include <stdexcept>
#include <string>
#include <utility>
// Extracted from executor_kernels.cu (D3, structural-debt audit): the
// WeightHandle-keyed GEMM dispatch (gemm_via_handle_) and its uncached
// dequant->cuBLAS safety-net fallback, pulled out of the kernel grab-bag.
#include "exec/executor_kernels.h"
#include "exec/executor_gemv_helpers.h"
#include "exec/gemm_context.h"
#include "exec/gemm_dispatch_fallback.h"
#include "exec/gemm_kernel_registry.h"
#include "exec/weight_dequant.h"
#include "exec/executor.h"
#include "compute/weight_dispatch.h"
#include "core/tensor_kind.h"
#include "core/logging.h"
#include "core/sm120_limits.h"
#include "compute/gemm.h"
#include "quant/mxfp8_sfatom.h"
#include "compute/gemm_q4k.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_mxfp4_f16_tile.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/mxfp4_sfatom.h"
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/mxfp4_gemm.h"
#include "compute/ggml_mmvq.h"
#include "compute/gemm_q8_0_dense.h"
#include "compute/mmq_q8_imma.h"
#include "compute/hadamard.h"
#include "core/pdl.h"
#include "core/process_diag.h"
#include "compute/activation.h"
#include "compute/layernorm.h"

namespace imp {

// IMMA prefill on a GGUF source (Q8_0, Q4_K, Q6_K at 2..32 rows); false = caller dequantizes.
static bool imma_source_prefill(const WeightHandle& h, const Tensor& input, Tensor& output,
                                const GemmContext& ctx, int M, bool decode_rows) {
    // Q8_0 INT8 IMMA fast path (gemm.q8_imma_enabled): fused dequant on int8
    // tensor cores instead of materialize-to-FP16 -> cuBLAS. Covers beta=0 and
    // the beta=1 residual-add form; declines fall through to the dequant
    // fallback. M>=2 (was >=64): below 64 the dequant tax dominates even
    // harder (an M=9 spec-decode verify chunk re-dequantized the entire model
    // every step, #667). The IMMA kernel zero-fills M-tail rows; MoE already runs it at per-expert M~32.
    // Split-K only on decode/verify rows: a prompt row must not change with its chunk size (#2152).
    const bool imma_eligible = input.qtype == QType::F16 && output.qtype == QType::F16 && M >= 2 &&
                               input.stride[0] == h.shape[1] && output.stride[0] == h.shape[0];
    if (ctx.q8_imma_enabled && h.source_qtype == QType::Q8_0 && imma_eligible) {
        // q/k/v share one input: k and v skip the quantize q wrote (tag-checked in the callee).
        const bool reuse_act = ctx.act_quant_hint_matches(input.data, M, static_cast<int>(h.shape[1]));
        if (mmq_q8_imma_gemm(h.source_data, reinterpret_cast<const __half*>(input.data),
                             reinterpret_cast<__half*>(output.data), M, static_cast<int>(h.shape[0]),
                             static_cast<int>(h.shape[1]), ctx.stream, ctx.beta,
                             /*allow_splitk=*/decode_rows, reuse_act))
            return true;
    }
    if (h.source_qtype == QType::Q4_K || h.source_qtype == QType::Q5_K) {
        const Tensor w(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        if (gemm_q45k_tile_rows(input, w, output, ctx, decode_rows))
            return true;
    }
    if (ctx.q4k_imma_prefill && h.source_qtype == QType::Q4_K && imma_eligible) {
        if (mmq_q4k_imma_gemm(h.source_data, reinterpret_cast<const __half*>(input.data),
                              reinterpret_cast<__half*>(output.data), M, static_cast<int>(h.shape[0]),
                              static_cast<int>(h.shape[1]), ctx.stream, ctx.beta, q4k_split_ok(decode_rows)))
            return true;
    }
    // Dense Q6_K above 32 rows stays on dequant + f16-acc cuBLAS (the half-MMA split halves the
    // int8 rate: pp512 6957 vs 3747 tok/s). At 2..32 rows the per-chunk dequant dominates: IMMA.
    if (ctx.q6k_imma_smallm && h.source_qtype == QType::Q6_K && imma_eligible && M <= 32 &&
        !process_diag_deterministic()) {
        if (mmq_q6k_imma_gemm(h.source_data, reinterpret_cast<const __half*>(input.data),
                              reinterpret_cast<__half*>(output.data), M, static_cast<int>(h.shape[0]),
                              static_cast<int>(h.shape[1]), ctx.stream, ctx.beta))
            return true;
    }
    return false;
}

// Last route of gemm_via_handle_. A GGUF source with an SfAtom-less CUTLASS_NVFP4 entry (Phase 3 G3)
// has no CUTLASS form for the M=1 beta != 0 call that lands here: the source serves it.
// M=1 GEMV on the quantized source (dp4a/MMVQ via the kernel registry). False when the source is
// 16-bit, absent, or the registry declines.
static bool source_gemv(const WeightHandle& h, const Tensor& input, Tensor& output, const GemmContext& ctx) {
    const auto* qs = ctx.qscratch;
    if (!qs || h.source_qtype == QType::NONE || h.source_qtype == QType::F16 ||
        h.source_qtype == QType::BF16 || h.source_data == nullptr || output.qtype == QType::F32 ||
        input.stride[0] != h.shape[1])
        return false;
    Tensor src(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = ctx.stream;
    args.weight_payload = &src;
    args.q8_1_buf = qs->q8_1_buf;
    args.d8_buf = qs->d8_buf;
    args.dequant_scratch = qs->dequant;
    args.dequant_scratch_size = qs->dequant_size;
    args.force_mmvq = ctx.force_mmvq;
    args.no_mmvq = ctx.gemm_no_mmvq;
    args.no_mmvq_q8_0 = ctx.gemm_no_mmvq_q8_0;
    args.no_dp4a_gemv = ctx.gemm_no_dp4a_gemv;
    GemmStrategy strat{StorageTier::FP16, h.source_qtype, true};
    return GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok;
}

static void gemm_handle_last_route(const WeightHandle& h, const Tensor& input, Tensor& output,
                                   const GemmContext& ctx, bool decode_rows, void* ws, size_t ws_size) {
    if (h.primary_tier == StorageTier::CUTLASS_NVFP4 && h.payload.cutlass_nvfp4.sf == nullptr &&
        h.source_data != nullptr && dequant_gpu_supported(h.source_qtype)) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("gemm: GGUF %s weight without SfAtom, M=%lld beta=%.0f runs on the source",
                         qtype_name(h.source_qtype), static_cast<long long>(input.shape[0]), ctx.beta);
        }
        gguf_source_fallback(h, input, output, ctx, decode_rows);
        return;
    }
    imp::gemm_dispatch(nullptr, h, input, output, 1.0f, ctx.beta, ws, ws_size, ctx.stream);
}

// One CUTLASS NVFP4 prefill GEMM through the kernel registry: quantizes `input` into the
// shared activation scratch (skipped when ctx says a prior dispatch on the same input
// already did) and declines on a missing SfAtom (GGUF source, Phase 3 G3), a missing scratch
// or a nonzero beta (the epilogue bakes beta=0), so the caller falls through to a beta-honouring route.
static bool dispatch_cutlass_nvfp4_prefill(const CutlassNvFP4Weight& cw, const Tensor& input, Tensor& output,
                                           const GemmContext& ctx, int M) {
    const auto* qs = ctx.qscratch;
    if (!cw.scale_factors || !qs || !qs->cutlass_act_data || !qs->cutlass_act_sf)
        return false;
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = ctx.stream;
    args.weight_payload = &cw;
    args.cutlass_act_data = qs->cutlass_act_data;
    args.cutlass_act_sf = qs->cutlass_act_sf;
    args.cutlass_workspace = qs->cutlass_workspace;
    args.cutlass_workspace_size = qs->cutlass_workspace_size;
    args.act_prequantized = (ctx.act_quant_hint_data != nullptr && ctx.act_quant_hint_data == input.data &&
                             ctx.act_quant_hint_m == M &&
                             ctx.act_quant_hint_k == static_cast<int>(input.shape[1]));
    args.beta = ctx.beta;
    GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, false};
    return GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok;
}

// gemm.nvfp4_gdn_proj_prefill: an FP16-primary handle with an NVFP4 prefill copy (phase 3d,
// keyed by source pointer; the handle's tiers stay FP16 so every other M keeps its route).
// True prefill only: M>32 keeps the batched decode rows on FP16, the spec verify chunk
// stays on the path its acceptance was measured on.
// The source is F16 with tier FP16 (FP8 decode sidecar demoted it) or Undefined (no cache
// entry at all, e.g. gdn_gate and ssm_in on Qwen3.6-35B), never a quantized one.
static bool nvfp4_prefill_copy_routes(const WeightHandle& h, int M, bool spec_verify, const WeightCaches& wc,
                                      const QuantScratch& qs) {
    return M > 32 && !spec_verify && h.source_qtype == QType::F16 &&
           (h.primary_tier == StorageTier::FP16 || h.primary_tier == StorageTier::Undefined) &&
           wc.cutlass_nvfp4_prefill.contains(h.source_data) && qs.cutlass_act_data != nullptr &&
           qs.cutlass_act_sf != nullptr;
}

// One ACTIVE / DECLINED line per tensor kind (log only, a benign race between engines).
static void log_nvfp4_prefill_copy_route(const WeightHandle& h, int M, float beta, bool active) {
    static bool logged[256] = {};
    const auto k = static_cast<unsigned>(h.kind) & 255u;
    if (logged[k])
        return;
    logged[k] = true;
    IMP_LOG_INFO(
        "GDN NVFP4 prefill route %s: %s [%lld x %lld] M=%d beta=%.1f primary_tier=%d source_qtype=%d",
        active ? "ACTIVE" : "DECLINED", tensor_kind_name(h.kind), (long long)h.shape[0],
        (long long)h.shape[1], M, beta, static_cast<int>(h.primary_tier), static_cast<int>(h.source_qtype));
}

static void log_mxfp8_prefill_copy_route(const WeightHandle& h, int M, float beta, bool active) {
    static bool logged[256] = {};
    const auto k = static_cast<unsigned>(h.kind) & 255u;
    if (logged[k])
        return;
    logged[k] = true;
    IMP_LOG_INFO(
        "GDN MXFP8 prefill route %s: %s [%lld x %lld] M=%d beta=%.1f primary_tier=%d source_qtype=%d",
        active ? "ACTIVE" : "DECLINED", tensor_kind_name(h.kind), (long long)h.shape[0],
        (long long)h.shape[1], M, beta, static_cast<int>(h.primary_tier), static_cast<int>(h.source_qtype));
}

// Own MXFP8 GEMM on the quantized activation (column scale fused); false on refusal (N % 8 != 0), the caller
// keeps the handle's previous route. Stream-K borrows the NVFP4 prefill workspace (same stream); none under
// runtime.deterministic.
static bool mxfp8_prefill_gemm(const QuantScratch& qs, const CutlassMxFP8Weight& cw, void* out, int M, int K,
                               cudaStream_t stream) {
    const int N = static_cast<int>(cw.N);
    void* ws = process_diag_deterministic() ? nullptr : qs.cutlass_workspace;
    if (!gemm_mxfp8_tile(qs.mxfp8_act_data, qs.mxfp8_act_sf, cw.data, cw.scale_factors, cw.col_scale, out, M,
                         N, K, stream, /*variant=*/-1, ws, qs.cutlass_workspace_size))
        return false;
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("MXFP8 prefill: own tile GEMM ACTIVE (M=%d N=%d K=%d)", M, N, K);
    }
    return true;
}

// Native MXFP4 GEMM with M > 1 (or fp32 rows): W4A16. Each weight is dequantized to FP16 into the load-time
// scratch (largest non-LM-head weight), then FP16 GEMM with the FP16 activation: Qwen3.5-4B-mxfp4 PPL
// 11.6369 vs W4A4 14.2956 (ppl_4k). The LM head, larger than the scratch, runs in N slices through
// mxfp4_slice_out and lands in its columns by a 2-D copy. The shared workspace is off limits: the SSM/GDN
// buffers are carved from its start (configure_ssm_workspace).
// gemm.mxfp4_tile: the fused W4A16 tile decodes E2M1 per stage (bit-equal to the scratch path below) where
// it wins in situ: M <= 1024 and >= 64 128x128 tiles (Qwen3.5-4B pp2048 gate/up 316 vs 254 + 50 us; N <= 128
// 35 vs ~3 us). fp32 LM-head rows keep the scratch path.
static bool mxfp4_tile_rows(const uint8_t* w, const uint8_t* s, const Tensor& input, Tensor& output, int M,
                            int N, int K, const GemmContext& ctx) {
    const bool det = process_diag_deterministic();  // every row: tile-invariant bits, chunk-invariant
    if (!process_diag_mxfp4_tile() || output.qtype != QType::F16 || input.qtype != QType::F16 ||
        (!det && (M > 1024 || ((M + 127) / 128) * ((N + 127) / 128) < 64)) || input.stride[0] != K ||
        output.stride[0] != N ||
        !gemm_mxfp4_f16_tile(input.data, w, s, output.data, M, N, K, 1.0f, 0.0f,
                             process_diag_cublas_fp16_acc(), ctx.stream))
        return false;
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("gemm.mxfp4_tile: fused MXFP4 W4A16 GEMM ACTIVE (M=%d N=%d K=%d)", M, N, K);
    }
    return true;
}

static void mxfp4_prefill_gemm(const QuantScratch& qs, const WeightHandle& h, const Tensor& input,
                               Tensor& output, const GemmContext& ctx) {
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(h.shape[0]);
    const bool out_f32 = output.qtype == QType::F32;
    const size_t es = out_f32 ? sizeof(float) : sizeof(half);
    const size_t row_w = static_cast<size_t>(K) * sizeof(half);
    const int rows_w = static_cast<int>(qs.mxfp4_wdeq_size / row_w);
    const int whole = N <= rows_w ? N : 0;
    const int rows_o = static_cast<int>(qs.mxfp4_wdeq_size / (static_cast<size_t>(M) * es));
    const int slice = whole > 0 ? whole : (std::min(rows_w, rows_o) / 8) * 8;
    if (ctx.beta != 0.0f || (output.qtype != QType::F16 && !out_f32) || K % 32 != 0 || slice <= 0 ||
        h.payload.mxfp4.weight == nullptr || h.payload.mxfp4.linear_scales == nullptr)
        throw std::runtime_error("MXFP4 GEMM: beta=" + std::to_string(ctx.beta) +
                                 " out=" + qtype_name(output.qtype) + " M=" + std::to_string(M) +
                                 " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                 " outside the W4A16 scratch contract");
    const auto* w = static_cast<const uint8_t*>(h.payload.mxfp4.weight);
    const auto* s = static_cast<const uint8_t*>(h.payload.mxfp4.linear_scales);
    if (mxfp4_tile_rows(w, s, input, output, M, N, K, ctx))
        return;
    for (int n0 = 0; n0 < N; n0 += slice) {
        const int nr = std::min(slice, N - n0);
        dequant_mxfp4_linear_to_fp16(w + (static_cast<size_t>(n0) * (K / 2)),
                                     s + (static_cast<size_t>(n0) * (K / 32)), nr, K, qs.mxfp4_wdeq,
                                     ctx.stream);
        int64_t w_shape[2] = {nr, K};
        Tensor wt(qs.mxfp4_wdeq, QType::F16, 2, w_shape, true);
        if (whole > 0) {
            gemm(input, wt, output, 1.0f, 0.0f, ctx.stream);
            break;
        }
        int64_t o_shape[2] = {M, nr};
        Tensor part(qs.mxfp4_slice_out, output.qtype, 2, o_shape, true);
        gemm(input, wt, part, 1.0f, 0.0f, ctx.stream);
        IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(static_cast<char*>(output.data) + (static_cast<size_t>(n0) * es),
                                             static_cast<size_t>(N) * es, qs.mxfp4_slice_out, nr * es,
                                             nr * es, M, cudaMemcpyDeviceToDevice, ctx.stream));
    }
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("MXFP4 GEMM: W4A16 dequant scratch ACTIVE (M=%d N=%d K=%d, slice %d)", M, N, K, slice);
    }
}

// Conservative mirror of gemm_via_handle_'s M>1 routing: true only when
// the dispatch is guaranteed to reach the CUTLASS NVFP4 prefill block
// (which quantizes input into the shared activation scratch). Every
// earlier-return route in gemm_via_handle_ must answer false here.
bool GraphExecutor::prefill_routes_cutlass_nvfp4_(TensorID id, int M) const {
    if (id == kInvalidTensorID)
        return false;
    const auto& h = registry_.handle(id);
    if (nvfp4_prefill_copy_routes(h, M, cur_spec_verify_, wcache_, qscratch_))
        return true;
    if (h.primary_tier != StorageTier::CUTLASS_NVFP4)
        return false;
    // #1055: small-M verify chunks divert to the batched NVFP4 GEMV overlay
    // (native branch in gemm_via_handle_) — no CUTLASS activation quant.
    if (cur_spec_verify_ && dispatch_policy().speculative.verify_nvfp4_gemm && M <= 4 &&
        h.source_data != nullptr && h.source_scales != nullptr)
        return false;
    StorageTier prefill =
        (h.prefill_tier != StorageTier::Undefined) ? h.prefill_tier : h.primary_tier;
    // FP16 / FP8 prefill-cache hits intercept before the CUTLASS block.
    if (prefill == StorageTier::FP16 && wcache_.fp16.contains(h.source_data))
        return false;
    if (prefill == StorageTier::FP8 && wcache_.fp8.contains(h.source_data))
        return false;
    // GGUF source with NVFP4 decode overlay: prefill dequants the original
    // quant — never reaches the CUTLASS block.
    if (h.source_data != nullptr && dequant_gpu_supported(h.source_qtype))
        return false;
    // The CUTLASS block itself requires the activation scratch.
    return qscratch_.cutlass_act_data != nullptr && qscratch_.cutlass_act_sf != nullptr;
}

// gemm_via_handle_: WeightHandle dispatch for all registered weights. M>1
// routes through weight_dispatch. M=1 beta=0 routes through gemv_dispatch
// or tier-specific handlers. M=1 beta!=0 routes through weight_dispatch
// (cuBLAS with beta). Undefined tier (budget-exhausted) reconstructs the
// weight Tensor from the handle and uses the uncached dequant fallback.
void GraphExecutor::offload_enter_(int layer, int n_layers, cudaStream_t stream) {
    if (!offload_mgr_)
        return;
    offload_mgr_->ensure_layer(layer, stream);
    if (offload_mgr_->is_offloaded(layer))
        rebind_streamed_sources_(layer);
    if (layer + 1 < n_layers)
        offload_mgr_->prefetch_layer(layer + 1);
}

void GraphExecutor::offload_leave_(int layer, cudaStream_t stream) {
    if (!offload_mgr_)
        return;
    offload_mgr_->release_layer(layer, stream);
    if (offload_mgr_->is_offloaded(layer))
        rebind_streamed_sources_(layer);
}

void GraphExecutor::rebind_streamed_sources_(int layer) {
    // The for_each_streamable_matmul set (model/layer_host_keep.h) with its registry ids.
    const auto& L = model_->layer(layer);
    const std::pair<const Tensor*, TensorID> streamed[] = {
        {&L.wq, L.wq_id},         {&L.wk, L.wk_id},     {&L.wv, L.wv_id},        {&L.wo, L.wo_id},
        {&L.w_gate, L.w_gate_id}, {&L.w_up, L.w_up_id}, {&L.w_down, L.w_down_id},
    };
    for (const auto& [w, id] : streamed)
        if (id != kInvalidTensorID)
            registry_.handle(id).source_data = w->data;
}

void GraphExecutor::gemm_via_handle_(TensorID id, const Tensor& input,
                                     Tensor& output, const GemmContext& ctx) {
    const auto& h = registry_.handle(id);

    // Activation calibration ([calibration] enabled). Placed before the tier
    // switch so the statistic is the activation the weight consumes, not
    // whatever a particular tier's kernel happens to materialise.
    if (calib_)
        calib_->accumulate(cur_layer_, h.kind, input, ctx.stream);

    // MXFP8 prefill copy of an F16 GDN projection (gemm.mxfp8_gdn_proj_prefill): the same
    // gate as the NVFP4 copy below, checked first. Quantizes the activation into the MXFP8
    // scratch on every call (ssm_in and gdn_gate share one input; the dedupe is a later lever).
    if (static_cast<int>(input.shape[0]) > 32 && wcache_.cutlass_mxfp8_prefill.contains(h.source_data)) {
        const int Mp = static_cast<int>(input.shape[0]);
        const int Kp = static_cast<int>(input.shape[1]);
        const auto& cw = wcache_.cutlass_mxfp8_prefill.find(h.source_data)->second;
        const auto& qs = qscratch_;
        bool active = ctx.beta == 0.0f && input.qtype == QType::F16 && output.qtype == QType::F16 &&
                      !cur_spec_verify_ && h.source_qtype == QType::F16 &&
                      (h.primary_tier == StorageTier::FP16 || h.primary_tier == StorageTier::Undefined) &&
                      qs.mxfp8_act_data != nullptr && qs.mxfp8_act_sf != nullptr &&
                      static_cast<size_t>(Mp) * Kp <= qs.mxfp8_act_data_size &&
                      cutlass_mxfp8_sf_size(Mp, Kp) <= qs.mxfp8_act_sf_size;
        if (active) {
            quantize_fp16_to_mxfp8_cutlass(input.data, qs.mxfp8_act_data, qs.mxfp8_act_sf, Mp, Kp,
                                           ctx.stream);
            active = mxfp8_prefill_gemm(qs, cw, output.data, Mp, Kp, ctx.stream);
        }
        log_mxfp8_prefill_copy_route(h, Mp, ctx.beta, active);
        if (active)
            return;
    }

    // NVFP4 prefill copy of an F16 GDN projection (gemm.nvfp4_gdn_proj_prefill). Ahead of the
    // tier split: on Qwen3.6-35B ssm_in and gdn_gate sit in no cache (tier Undefined) and
    // ssm_out carries the FP8 sidecar (tier FP16). Declines (beta=1, M<=32, spec verify, no
    // scratch) fall through to the route the handle had before.
    if (static_cast<int>(input.shape[0]) > 32 && wcache_.cutlass_nvfp4_prefill.contains(h.source_data)) {
        const int Mp = static_cast<int>(input.shape[0]);
        auto it = wcache_.cutlass_nvfp4_prefill.find(h.source_data);
        const bool active = ctx.beta == 0.0f && input.qtype == QType::F16 && output.qtype == QType::F16 &&
                            nvfp4_prefill_copy_routes(h, Mp, cur_spec_verify_, wcache_, qscratch_) &&
                            dispatch_cutlass_nvfp4_prefill(it->second, input, output, ctx, Mp);
        log_nvfp4_prefill_copy_route(h, Mp, ctx.beta, active);
        if (active)
            return;
    }

    // Phase 4b freed the source: rebuild it from a copy, never read source_data.
    if (h.source_released) {
        released_source_gemm_(h, input, output, ctx);
        return;
    }

    if (h.primary_tier == StorageTier::Undefined) {
        // Q4_K token GEMV without an FP16 tier (gemm.q4k_in_place): the same kernel the FP16 tier runs.
        if (h.source_qtype == QType::Q4_K && input.shape[0] == 1 && ctx.beta == 0.0f &&
            source_gemv(h, input, output, ctx))
            return;
        Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        // Preserve semantic identity + NVFP4 scale sidecars so the uncached
        // fallback can (a) diagnose which weight leaked and (b) route packed
        // NVFP4 to a real dequant instead of feeding raw bytes to cuBLAS.
        weight.kind = h.kind;
        weight.scales = h.source_scales;
        weight.tensor_scale = h.source_tensor_scale;
        gemm_dispatch_uncached_fallback(input, weight, output, ctx, cur_decode_rows_);
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
                // The MXFP4 GEMV writes fp16: fp32 rows (perplexity LM head, one-row tail) take the prefill
                // route.
                if (output.qtype == QType::F32)
                    break;
                imp::gemv_dispatch(h, input, output, ctx.stream);
                return;
            case StorageTier::FP16: {
                if (source_gemv(h, input, output, ctx))
                    return;
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
        // Verify chunk to the small-M block: speculative.verify_smallm, or a graph-captured verify step
        // under verify_smallm_captured.
        const auto& spec = dispatch_policy().speculative;
        const bool verify_smallm = ctx.spec_verify_small_m &&
                                   (spec.verify_smallm ||
                                    (cur_spec_verify_smallm_ && spec.verify_smallm_captured));

        // Spec-verify chunk (#998): reads the NVFP4 decode overlay in one weight
        // pass per MR<=4 tile instead of dequantizing the source (GGUF K-quants
        // with no direct small-M kernel, e.g. Q6_K, otherwise pay ~7x a decode
        // step per verify). Also aligns verify argmax with the decode path. M cap
        // = largest capture bucket (33). Scoped to dequantable GGUF sources;
        // native-ST NVFP4 weights read NVFP4 directly via the CUTLASS block below
        // (prefill_routes_cutlass_nvfp4_ must answer false here for those).
        // speculative.verify_smallm routes the chunk to the small-M block instead.
        if (ctx.spec_verify_small_m && !verify_smallm && (ctx.beta == 0.0f || ctx.beta == 1.0f) && M <= 33 &&
            input.qtype == QType::F16 && output.qtype == QType::F16 && h.source_data != nullptr &&
            dequant_gpu_supported(h.source_qtype)) {
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
        // #1055: native ST-NVFP4 verify chunks. The CUTLASS prefill block serves
        // them correctly but at low bandwidth for tiny M; the batched GEMV reads
        // the weight once per MR=4 tile, winning only in the single-tile regime:
        // hard cap M<=4, larger chunks stay on CUTLASS. Same weight + linear
        // micro-scales as the M=1 decode GEMV, but NOT the same kernel (decode
        // uses gemv_nvfp4_kpar, verify uses gemm_nvfp4_batched); no argmax parity
        // with non-speculative greedy output either way (docs/LIMITATIONS.md).
        if (ctx.spec_verify_small_m && !verify_smallm && (ctx.beta == 0.0f || ctx.beta == 1.0f) && M <= 4 &&
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

        // Small-M NVFP4 GEMM (gemm.nvfp4_smallm, default on): batched decode at
        // n_seq<=32. impl 2 (default) is the native mxf4nvf4 producer/consumer
        // pipeline on the SAME plain weight bytes the M=1 decode GEMVs read (vs
        // the CUTLASS 128x128 block-scaled tile it replaces); impl 1 keeps the
        // refuted W4A16 dequant+HMMA kernel for A/B. Spec-verify chunks keep their
        // documented paths (#1055). Weight is the native NVFP4 source or, on
        // decode rows of a GGUF source, its NVFP4 decode overlay (smallm_weight_, #1897).
        NvFP4QuantResult nv;
        if (dispatch_policy().gemm.nvfp4_smallm && (!ctx.spec_verify_small_m || verify_smallm) &&
            !overlap_prefill_active_ && M <= 32 && (ctx.beta == 0.0f || ctx.beta == 1.0f) &&
            input.qtype == QType::F16 && output.qtype == QType::F16 && smallm_weight_(h, nv)) {
            const int N = static_cast<int>(nv.N);
            const int K = static_cast<int>(nv.K);
            // A16 (gemm.nvfp4_smallm_a4=false): FP16 activations into the
            // dequant+HMMA kernel, no activation quantize (K%128, M<=32).
            if (!dispatch_policy().gemm.nvfp4_smallm_a4 && (K % 128) == 0) {
                const size_t need16 = gemm_nvfp4_smallm_workspace_bytes(N);
                ensure_smallm_ws_(need16, ctx.stream);
                if (smallm_ws_bytes_ >= need16 &&
                    gemm_nvfp4_smallm(nv, reinterpret_cast<const half*>(input.data),
                                      reinterpret_cast<half*>(output.data), M, N, K, smallm_ws_, ctx.stream,
                                      /*accumulate=*/ctx.beta == 1.0f))
                    return;
            }
            // impl 2 = the native mxf4nvf4 pipeline kernel (v2), impl 1 = the
            // dequant+HMMA kernel on packed activations; unaligned shapes fall back to v1.
            const bool v2 = dispatch_policy().gemm.nvfp4_smallm_impl == 2 && (K % 256) == 0 && (N % 64) == 0;
            const size_t need = v2 ? gemm_nvfp4_smallm_v2_workspace_bytes(N, K)
                                   : gemm_nvfp4_smallm_workspace_bytes(N);
            // Capturing with a too-small workspace falls through (CUTLASS on
            // a native source, the dequant route on a GGUF one);
            // prewarm_smallm_workspace sizes both scratches at init.
            ensure_smallm_ws_(need, ctx.stream);
            // A4: quantizes activation rows into the executor scratch (plain layout,
            // unit tensor scale) and reads both sides packed. The FP16 variant lost to
            // L2 eviction of its large x tile; packed x is far smaller. Scratch sized
            // for K_max on first eager use, like the split-K workspace.
            const size_t xq_need = ((size_t)32 * (K / 2)) + ((size_t)32 * (K / 16));
            ensure_smallm_xq_(xq_need, ctx.stream);
            if (smallm_ws_bytes_ >= need && smallm_xq_bytes_ >= xq_need) {
                uint8_t* xq_packed = static_cast<uint8_t*>(smallm_xq_);
                uint8_t* xq_scales = xq_packed + ((size_t)32 * (K / 2));
                // Shared-activation skip: the call site marked this input as already
                // quantized by the PREVIOUS dispatch (act-quant hint), and the scratch tag
                // confirms it still holds exactly that quantize. Saves one quantize launch
                // + a [M,K] FP16 read per second member of a gate/up, q/k/v or GDN in/z pair.
                const bool tag_match = smallm_xq_src_ == input.data && smallm_xq_src_m_ == M &&
                                       smallm_xq_src_k_ == K;
                const bool hint_match = ctx.act_quant_hint_data != nullptr &&
                                        ctx.act_quant_hint_data == input.data &&
                                        ctx.act_quant_hint_m == M && ctx.act_quant_hint_k == K;
                // Producer fusion (fused rmsnorm/swiglu+quantize) accepts a matching tag
                // without a hint: the producer re-tags on the very write that produced the
                // FP16 buffer, so the pointer cannot hold newer content than the scratch.
                const bool prequant = tag_match && (hint_match || smallm_xq_from_producer_);
                if (!prequant) {
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

        // dp4a dense: computes directly from Q4_K/Q5_K blocks (0.55 B/elem)
        // instead of the FP16 cache (2.0 B/elem). Weight-stationary TILE_M=16, so
        // it re-reads the weight ceil(M/16) times; only wins at small M (<=64,
        // memory-bound). sm_120 smem cap 99 KiB fits K up to ~4400.
        // M <= 32 on a Q4_K / Q5_K source, 2..32 on the other GGUF sources: the in-place kernels read the
        // blocks once (UD-Q4_K_M pp16 Q4_K avg 137 us; the FP16 copy at verify rows: 116 us per 17408 x 5120
        // cuBLAS call). Other qtypes decline there; M = 1 keeps its route.
        const bool k_quant = h.source_qtype == QType::Q4_K || h.source_qtype == QType::Q5_K;
        if (prefill == StorageTier::FP16 && M <= 32 && (M >= 2 || k_quant) && h.source_data != nullptr &&
            !wcache_.q8_planes_only.contains(h.source_data)) {  // dropped Q8_0 source
            Tensor w(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
            if (gemm_uncached_in_place(input, w, output, ctx, cur_decode_rows_))
                return;
        }
        if (prefill == StorageTier::FP16 && ctx.beta == 0.0f && M <= 64) {
            const auto* qs = ctx.qscratch;
            if (qs && qs->q8_1_prefill_buf && qs->d8_prefill_buf) {
                int N = static_cast<int>(h.shape[0]);
                int K = static_cast<int>(h.shape[1]);
                size_t smem_needed = static_cast<size_t>(16) * (K / 32) * 36;
                size_t needed_q8 = static_cast<size_t>(M) * ((K + 31) / 32) * sizeof(block_q8_1);
                size_t needed_d8 = static_cast<size_t>(M) * ((K + 31) / 32) * sizeof(float);
                if (smem_needed <= kSm120SmemOptinBytes && needed_q8 <= qs->q8_1_prefill_bytes &&
                    needed_d8 <= qs->d8_prefill_bytes) {
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
            // FP16-resident or dequantable source with no cache entry: the source IS
            // the weight (e.g. only the fp8_ssm_proj decode sidecar exists). Routes
            // through the uncached fallback exactly as the pre-sidecar Undefined tier
            // did; falling further through would hand cuBLAS a null payload.
            if (h.source_data &&
                (h.source_qtype == QType::F16 || h.source_qtype == QType::BF16 ||
                 dequant_gpu_supported(h.source_qtype))) {
                Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
                weight.kind = h.kind;
                weight.scales = h.source_scales;
                weight.tensor_scale = h.source_tensor_scale;
                gemm_dispatch_uncached_fallback(input, weight, output, ctx, cur_decode_rows_);
                return;
            }
        }
        if (prefill == StorageTier::FP8) {
            auto fp8_it = wcache_.fp8.find(h.source_data);
            if (fp8_it != wcache_.fp8.end()) {
                gemm_fp8_prefill_(h, fp8_it->second, input, output, ctx);
                return;
            }
        }
    }

    // GGUF source with an NVFP4 decode-only overlay (mode 1 additive): for
    // prefill (M>1) must dequant the ORIGINAL Q*_K source to FP16, since
    // running prefill on the 4-bit NVFP4 overlay corrupts the prompt context
    // and degenerates output. Reached only when no FP16/FP8 prefill cache
    // exists (sm_120 disables FP8 prefill, #428). Decode rows read the NVFP4
    // cache instead (M=1 and 2..32-row small-M, #1897). Native NVFP4
    // SafeTensors models are excluded (source_qtype rejected by dequant_gpu_supported()) and use the CUTLASS
    // path.
    if (M > 1 && h.source_data != nullptr && dequant_gpu_supported(h.source_qtype) &&
        (h.primary_tier == StorageTier::CUTLASS_NVFP4 || h.primary_tier == StorageTier::NVFP4)) {
        if (imma_source_prefill(h, input, output, ctx, M, cur_decode_rows_))
            return;
        gguf_source_fallback(h, input, output, ctx, cur_decode_rows_);
        return;
    }

    // ---- CUTLASS NVFP4 prefill via GemmKernelRegistry (qscratch buffers) ----
    if (M > 1 && h.primary_tier == StorageTier::CUTLASS_NVFP4) {
        {
            CutlassNvFP4Weight cw;
            cw.data = h.payload.cutlass_nvfp4.weight;
            cw.scale_factors = h.payload.cutlass_nvfp4.sf;
            cw.tensor_scale = h.payload.cutlass_nvfp4.global_scale
                                  ? *h.payload.cutlass_nvfp4.global_scale
                                  : 1.0f;
            cw.N = h.shape[0];
            cw.K = h.shape[1] * 2;
            // Act-quant dedupe (QKV / gate-up share one normed input, see the
            // with_act_quant_hint call sites) and the beta=1 decline live in the helper.
            if (dispatch_cutlass_nvfp4_prefill(cw, input, output, ctx, M))
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
    // and run cuBLAS. Reached when the fast CUTLASS-NVFP4 path declines and
    // there is no FP16/FP8 prefill companion (native source rejected by
    // dequant_gpu_supported()). Without this, dispatch falls to the generic
    // gemm(), which hands cuBLAS an unsupported FP16xINT8 GEMM
    // (CUBLAS_STATUS_NOT_SUPPORTED) -> silent repeated-token garbage + IMA
    // (hit Qwen3.6-35B-A3B-NVFP4's shared-expert gate/up). Slower than CUTLASS but correct; only fires on
    // decline.
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

    if ((M > 1 || output.qtype == QType::F32) && h.primary_tier == StorageTier::MXFP4) {
        mxfp4_prefill_gemm(qscratch_, h, input, output, ctx);
        return;
    }

    gemm_handle_last_route(h, input, output, ctx, cur_decode_rows_, ws_.shared(), ws_.shared_size());
}

// Producer-side NVFP4 quantize fusion (batched decode): the small-M block
// quantizes its FP16 input into smallm_xq_ once per consumer GROUP (act-
// quant hint dedupes pair members); the producer fusion instead quantizes
// inside the kernel that WRITES the FP16 buffer (fused rmsnorm/swiglu),
// killing the separate launch and the [M,K] re-read. Gate mirrors the
// small-M route conditions; any failure falls back to unfused kernels.
// ensure_smallm_ws_ grows the small-M workspace/xq scratch: never
// allocates while `stream` is capturing (allocate_smallm_scratch sizes
// both from the T2 arena at init; growth past that is the lazy fallback).
// An arena slab is never freed here; it belongs to the arena.
void GraphExecutor::ensure_smallm_ws_(size_t need, cudaStream_t stream) {
    if (need <= smallm_ws_bytes_)
        return;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
        cap = cudaStreamCaptureStatusActive;  // be conservative
    if (cap != cudaStreamCaptureStatusNone)
        return;
    if (smallm_ws_ && !smallm_ws_arena_)
        IMP_CUDA_CHECK_LOG(cudaFree(smallm_ws_));
    if (cudaMalloc(&smallm_ws_, need) == cudaSuccess)
        smallm_ws_bytes_ = need;
    else {
        smallm_ws_ = nullptr;
        smallm_ws_bytes_ = 0;
    }
    smallm_ws_arena_ = false;
}

void GraphExecutor::ensure_smallm_xq_(size_t xq_need, cudaStream_t stream) {
    if (xq_need <= smallm_xq_bytes_)
        return;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
        cap = cudaStreamCaptureStatusActive;  // be conservative
    if (cap != cudaStreamCaptureStatusNone)
        return;
    if (smallm_xq_ && !smallm_xq_arena_)
        IMP_CUDA_CHECK_LOG(cudaFree(smallm_xq_));
    smallm_xq_arena_ = false;
    if (cudaMalloc(&smallm_xq_, xq_need) == cudaSuccess)
        smallm_xq_bytes_ = xq_need;
    else {
        smallm_xq_ = nullptr;
        smallm_xq_bytes_ = 0;
    }
    smallm_xq_src_ = nullptr;  // fresh scratch holds nothing
    smallm_xq_from_producer_ = false;
}

// smallm_producer_xq_/tag_, rmsnorm/swiglu_for_smallm_ and
// try_smallm_pair_dispatch_ live in executor_gemm_smallm.cpp (2026-08-27
// split — see its header comment).

void GraphExecutor::gemm_fp8_prefill_(const WeightHandle& h, const FP8CacheEntry& e, const Tensor& input,
                                      Tensor& output, const GemmContext& ctx) {
    int64_t wshape[2] = {h.shape[0],
                         (h.primary_tier == StorageTier::CUTLASS_NVFP4) ? h.shape[1] * 2 : h.shape[1]};
    Tensor fp8_w(e.weight.data, QType::FP8_E4M3, 2, wshape, true);
    // cuBLASLt has no F16 x FP8 GEMM (status 15): quantize the activation first.
    if (input.qtype == QType::F16 && e.d_scale) {
        if (const float* a_scale = quantize_act_fp8_(h, input, ctx.stream)) {
            Tensor fp8_in(qscratch_.fp8_act, QType::FP8_E4M3, input.ndim, input.shape, true);
            gemm_fp8x8_(fp8_in, fp8_w, output, ctx.beta, a_scale, e.d_scale, ctx.stream);
            return;
        }
    }
    // Native FP8 the activation scratch cannot serve: FP16 weight in the dequant scratch.
    const size_t w_f16 = static_cast<size_t>(wshape[0]) * wshape[1] * sizeof(half);
    if (e.native_source && input.qtype == QType::F16 && qscratch_.dequant &&
        w_f16 <= qscratch_.dequant_size) {
        dequantize_fp8_e4m3_to_fp16(fp8_w.data, qscratch_.dequant, static_cast<int>(wshape[0] * wshape[1]),
                                    e.host_scale, ctx.stream);
        Tensor w16(qscratch_.dequant, QType::F16, 2, wshape, true);
        gemm(input, w16, output, 1.0f, ctx.beta, ctx.stream);
        return;
    }
    static bool s_native_warned = false;
    if (e.native_source && !std::exchange(s_native_warned, true))
        IMP_LOG_ERROR("native FP8 prefill: no activation or dequant scratch (N=%ld K=%ld)",
                      static_cast<long>(wshape[0]), static_cast<long>(wshape[1]));
    gemm_cublaslt(input, fp8_w, output, 1.0f, ctx.beta, nullptr, e.d_scale, ctx.stream);
}

void GraphExecutor::gemm_fp8x8_(const Tensor& a8, const Tensor& w8, Tensor& out, float beta,
                                const float* a_scale, const float* w_scale, cudaStream_t stream) {
    const int M = static_cast<int>(a8.numel() / a8.shape[a8.ndim - 1]);
    const int K = static_cast<int>(a8.shape[a8.ndim - 1]);
    const int N = static_cast<int>(w8.shape[0]);
    // gemm.fp8_tile_min_m: own tile from M 2048 (Qwen3-8B-FP8 pp4096 +0.6 %); below, cuBLASLt (pp512 own -18
    // %).
    const int min_m = process_diag_fp8_tile_min_m();
    if (min_m > 0 && M >= min_m && out.qtype == QType::F16 && a8.is_contiguous() && out.is_contiguous() &&
        a_scale && w_scale &&
        gemm_fp8_tile(a8.data, w8.data, a_scale, w_scale, beta, out.data, M, N, K, stream)) {
        static bool logged = false;  // activity proof, once
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("gemm.fp8_tile_min_m: own FP8 prefill GEMM ACTIVE (M=%d N=%d K=%d beta=%.0f)", M, N,
                         K, beta);
        }
        return;
    }
    gemm_cublaslt(a8, w8, out, 1.0f, beta, a_scale, w_scale, stream);
}

const float* GraphExecutor::quantize_act_fp8_(const WeightHandle& hw, const Tensor& input,
                                              cudaStream_t stream) {
    if (!qscratch_.fp8_act || !input.is_contiguous() ||
        static_cast<size_t>(input.numel()) > qscratch_.fp8_act_size)
        return nullptr;
    if (hw.payload.fp8.d_act_scale &&
        quantize_fp16_to_fp8_e4m3_dscale(input.data, qscratch_.fp8_act, input.numel(),
                                         hw.payload.fp8.d_act_scale, stream))
        return hw.payload.fp8.d_act_scale;
    if (!qscratch_.d_act_scale)
        return nullptr;
    Tensor fp8_in(qscratch_.fp8_act, QType::FP8_E4M3, input.ndim, input.shape, true);
    quantize_fp16_to_fp8_e4m3(input, fp8_in, qscratch_.d_act_scale, stream, qscratch_.d_fp8_block_maxes,
                              qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
    return qscratch_.d_act_scale;
}

}  // namespace imp

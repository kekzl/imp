// MoE batch prefill paths: NVFP4 dequant, Q6K, FP8, FP16, Gemma-4 ggml.
// Extracted from executor_forward_moe.cu for maintainability.

#include "exec/executor.h"
#include "exec/executor_forward_moe_internal.h"
#include "exec/executor_kernels.h"
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
#include "compute/gemm_q4k.h"
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

bool GraphExecutor::try_run_moe_nvfp4_dequant_batch_prefill_(int layer, cudaStream_t stream,
                                                             MoeFfnContext& ctx) {
    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    const size_t fp16_per_expert = static_cast<size_t>(
                                       std::max(ly.expert_up_packed.shape[1] * ly.expert_up_packed.shape[2],
                                                ly.expert_down_packed.shape[1] * ly.expert_down_packed.shape[2])) *
                                   sizeof(half);
    const bool ok = ly.nvfp4_moe_up_ptr != nullptr && ly.nvfp4_moe_down_ptr != nullptr &&
                    (ctx.non_gated_experts || ly.nvfp4_moe_gate_ptr != nullptr) &&
                    moe_.batch_dequant_buf != nullptr &&
                    moe_.batch_dequant_buf_size >= static_cast<size_t>(ctx.ne) * fp16_per_expert &&
                    moe_.d_weight_ptrs && moe_.d_weight_ptrs_count >= ctx.ne;
    if (!ok) return false;

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: NVFP4→FP16 batch path (n=%d, expanded=%d)", ctx.n, ctx.expanded);

    std::vector<int32_t> h_offsets(ctx.ne + 1);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_offsets.data(), ctx.routing.expert_offsets.data,
                                       static_cast<size_t>(ctx.ne + 1) * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    char* buf                = static_cast<char*>(moe_.batch_dequant_buf);
    char* gathered_base      = static_cast<char*>(moe_.gathered.data);
    char* expert_gate_base   = static_cast<char*>(moe_.expert_gate.data);
    char* expert_up_base     = static_cast<char*>(moe_.expert_up.data);
    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
    char* expert_down_base   = static_cast<char*>(moe_.expert_down.data);

    auto nvfp4_batch_dequant_gemm = [&](const NvFP4MoEQuantResult& nvfp4, const char* a_base,
                                        char* c_base, int K_dim, int N_dim) {
        int64_t rows = nvfp4.N;
        int64_t cols = nvfp4.K;
        size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);

        // Dequant all experts NVFP4 → FP16
        dequantize_nvfp4_moe_to_fp16(nvfp4, buf, stream);

        std::vector<const void*> b_ptrs(ctx.ne);
        for (int e = 0; e < ctx.ne; ++e)
            b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

        gemm_moe_batched(a_base, c_base, h_offsets.data(), b_ptrs.data(), K_dim, N_dim, QType::F16,
                         ctx.ne, stream, moe_.d_work_ptrs);
    };

    // Gate projection
    if (!ctx.non_gated_experts)
        nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_gate_ptr, gathered_base, expert_gate_base, ctx.d, ctx.eff);

    // Up projection
    nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_up_ptr, gathered_base, expert_up_base, ctx.d, ctx.eff);

    // Activation
    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                            ctx.non_gated_experts, ctx.expanded, ctx.eff, compute_dtype_,
                            cfg.ffn_activation, stream);

    // Down projection
    char* slow_down_act = ctx.non_gated_experts ? expert_up_base : expert_swiglu_base;
    nvfp4_batch_dequant_gemm(*ly.nvfp4_moe_down_ptr, slow_down_act, expert_down_base, ctx.eff, ctx.d);

    return true;
}

namespace {

// Dump gate logits or routing decisions for diagnostics. Called from
// compute_moe_routing under the appropriate config gates.
void dump_top8_gate_logits(int layer, int n, int ne, const float* d_logits) {
    int last_tok = n - 1;
    std::vector<float> h_logits(ne);
    cudaDeviceSynchronize();
    cudaMemcpy(h_logits.data(), d_logits + last_tok * ne, ne * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(ne);
    for (int i = 0; i < ne; ++i) sorted.emplace_back(h_logits[i], i);
    std::partial_sort(sorted.begin(), sorted.begin() + 8, sorted.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    fprintf(stderr, "[LOGITS] L%02d tok=%d top8_by_value: ", layer, last_tok);
    for (int i = 0; i < 8; ++i)
        fprintf(stderr, "[e=%d v=%.4f] ", sorted[i].second, sorted[i].first);
    fprintf(stderr, "\n");
}

}  // anonymous namespace

bool GraphExecutor::try_run_moe_q6k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                            int ne, int expanded, bool non_gated_experts,
                                            QType up_qtype, const MoeRoutingResult& routing,
                                            const Tensor& no) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    bool can_fused_q6k = (ne > 16 && ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == QType::Q6_K && ly.expert_down_packed.qtype == QType::Q6_K &&
                          compute_dtype_ == QType::F16);
    if (can_fused_q6k && !non_gated_experts)
        can_fused_q6k = (ly.expert_gate_packed.data && ly.expert_gate_packed.on_device &&
                         ly.expert_gate_packed.qtype == QType::Q6_K);

    bool use_tc = can_fused_q6k && (expanded > ne * 12);
    bool use_scalar = can_fused_q6k && !use_tc && (expanded <= ne * 12);
    if (!use_tc && !use_scalar) return false;

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: fused Q6_K %s path (n=%d, expanded=%d)",
                     use_tc ? "TC" : "scalar", n, expanded);

    const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
    const int32_t* d_sorted = static_cast<const int32_t*>(routing.sorted_token_ids.data);
    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

    if (use_tc) {
        // TC path: gather-free via sorted_token_ids indirection.
        if (!non_gated_experts)
            gemm_q6k_fused_moe_prefill_tc(ly.expert_gate_packed.data, no.data, expert_gate_base,
                                          d_offsets, eff, d,
                                          expert_stride(ly.expert_gate_packed,
                                                        ly.expert_gate_packed.qtype),
                                          ne, stream, d_sorted);
        gemm_q6k_fused_moe_prefill_tc(ly.expert_up_packed.data, no.data, expert_up_base, d_offsets,
                                      eff, d, expert_stride(ly.expert_up_packed, up_qtype), ne,
                                      stream, d_sorted);
    } else {
        // Scalar path: gathered buffer materialized first.
        int64_t gath_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
        Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
        moe_gather(no, routing, gathered, stream);
        char* gathered_base = static_cast<char*>(moe_.gathered.data);

        if (!non_gated_experts)
            gemm_q6k_fused_moe_prefill(
                ly.expert_gate_packed.data, gathered_base, expert_gate_base, d_offsets, eff, d,
                expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype), ne, stream);
        gemm_q6k_fused_moe_prefill(ly.expert_up_packed.data, gathered_base, expert_up_base,
                                   d_offsets, eff, d, expert_stride(ly.expert_up_packed, up_qtype),
                                   ne, stream);
    }

    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                            non_gated_experts, expanded, eff, compute_dtype_,
                            cfg.ffn_activation, stream);

    char* fused_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
    if (use_tc) {
        gemm_q6k_fused_moe_prefill_tc(
            ly.expert_down_packed.data, fused_down_act, expert_down_base, d_offsets, d, eff,
            expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype), ne, stream);
    } else {
        gemm_q6k_fused_moe_prefill(
            ly.expert_down_packed.data, fused_down_act, expert_down_base, d_offsets, d, eff,
            expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype), ne, stream);
    }
    return true;
}

static void fused_dp4a_for_qtype(QType qtype, const void* packed, const block_q8_1* q8,
                                 const float* d8, void* out, const int32_t* d_offsets,
                                 int N, int K, size_t stride_bytes, int ne, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q4_K:
            gemm_q4k_dp4a_moe_fused(packed, q8, d8, out, d_offsets, K, N, ne, stride_bytes, stream);
            break;
        case QType::Q5_K:
            gemm_q5k_dp4a_moe_fused(packed, q8, d8, out, d_offsets, K, N, ne, stride_bytes, stream);
            break;
        case QType::Q6_K:
            gemm_q6k_moe_fused(packed, q8, d8, out, d_offsets, K, N, ne, stride_bytes, stream);
            break;
        default:
            break;
    }
}

bool GraphExecutor::try_run_moe_q4k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                            int ne, int expanded, bool non_gated_experts,
                                            QType up_qtype, const MoeRoutingResult& routing,
                                            const Tensor& no) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    auto is_fuseable = [](QType q) {
        return q == QType::Q4_K || q == QType::Q5_K || q == QType::Q6_K;
    };

    bool can_fused = (ne > 16 && ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                      ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                      is_fuseable(up_qtype) && is_fuseable(ly.expert_down_packed.qtype) &&
                      compute_dtype_ == QType::F16 &&
                      moe_.batch_dequant_buf != nullptr);
    if (can_fused && !non_gated_experts)
        can_fused = (ly.expert_gate_packed.data && ly.expert_gate_packed.on_device &&
                     is_fuseable(ly.expert_gate_packed.qtype));

    // Buffer size check: Q8_1 for max(expanded*d, expanded*eff) elements
    if (can_fused) {
        int max_elems = std::max(expanded * d, expanded * eff);
        size_t q8_bytes = static_cast<size_t>(max_elems / 32) * sizeof(block_q8_1);
        size_t d8_bytes = static_cast<size_t>(max_elems / 32) * sizeof(float);
        if (q8_bytes + d8_bytes > moe_.batch_dequant_buf_size)
            can_fused = false;
    }

    if (!can_fused) return false;

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: fused Q4_K/Q5_K dp4a path (n=%d, expanded=%d)", n, expanded);

    const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

    // Step 1: Gather activations
    int64_t gath_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(d)};
    Tensor gathered(moe_.gathered.data, compute_dtype_, 2, gath_shape, true);
    moe_gather(no, routing, gathered, stream);

    // Step 2: Quantize gathered activations to Q8_1 (reuse batch dequant buf)
    int gate_up_elems = expanded * d;
    block_q8_1* q8_buf = reinterpret_cast<block_q8_1*>(moe_.batch_dequant_buf);
    float* d8_buf = reinterpret_cast<float*>(
        reinterpret_cast<char*>(q8_buf) + static_cast<size_t>(gate_up_elems / 32) * sizeof(block_q8_1));
    quantize_fp16_to_q8_1(static_cast<const half*>(moe_.gathered.data), q8_buf, d8_buf,
                          gate_up_elems, stream);

    // Step 3: Gate + up projections (dp4a, fused weight read)
    if (!non_gated_experts)
        fused_dp4a_for_qtype(ly.expert_gate_packed.qtype, ly.expert_gate_packed.data, q8_buf, d8_buf,
                             expert_gate_base, d_offsets, eff, d,
                             expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype), ne, stream);
    fused_dp4a_for_qtype(up_qtype, ly.expert_up_packed.data, q8_buf, d8_buf, expert_up_base,
                         d_offsets, eff, d, expert_stride(ly.expert_up_packed, up_qtype), ne, stream);

    // Step 4: Activation (SwiGLU / GeGLU / ReLU²)
    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                            non_gated_experts, expanded, eff, compute_dtype_,
                            cfg.ffn_activation, stream);

    // Step 5+6: Down projection — dp4a, re-quantize activations to Q8_1
    char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
    QType down_qtype = ly.expert_down_packed.qtype;
    size_t down_stride = expert_stride(ly.expert_down_packed, down_qtype);

    int down_elems = expanded * eff;
    block_q8_1* q8_down = reinterpret_cast<block_q8_1*>(moe_.batch_dequant_buf);
    float* d8_down = reinterpret_cast<float*>(
        reinterpret_cast<char*>(q8_down) + static_cast<size_t>(down_elems / 32) * sizeof(block_q8_1));
    quantize_fp16_to_q8_1(reinterpret_cast<const half*>(down_act), q8_down, d8_down,
                          down_elems, stream);
    fused_dp4a_for_qtype(down_qtype, ly.expert_down_packed.data, q8_down, d8_down,
                         expert_down_base, d_offsets, d, eff, down_stride, ne, stream);
    return true;
}

bool GraphExecutor::try_run_moe_fp8_batch_prefill(int layer, cudaStream_t stream, int n, int d,
                                                  int eff, int ne, int expanded,
                                                  bool non_gated_experts, QType up_qtype,
                                                  const MoeRoutingResult& routing) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: FP8 batch path (n=%d, expanded=%d, buf=%.1f MiB)", n, expanded,
                     moe_.batch_dequant_buf_size / (1024.0 * 1024.0));

    char* buf = static_cast<char*>(moe_.batch_dequant_buf);

    auto chunked_fp8_gemm = [&](const Tensor& packed, QType qtype, const char* a_base_fp16,
                                char* c_base_fp16, int K_dim, int N_dim) {
        int64_t rows = packed.shape[1];
        int64_t cols = packed.shape[2];
        size_t weight_fp8_bytes = static_cast<size_t>(ne) * rows * cols;
        uint8_t* fp8_weights = reinterpret_cast<uint8_t*>(buf);
        uint8_t* fp8_acts = fp8_weights + weight_fp8_bytes;

        dequant_gpu_fp8(packed.data, fp8_weights, qtype, ne * static_cast<int>(rows),
                        static_cast<int>(cols), stream);

        const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
        if (moe_.d_fp8_scales) {
            calibrate_fp8_scales_per_expert(a_base_fp16, K_dim, d_offsets, ne, moe_.d_fp8_scales,
                                            stream);
            quantize_fp16_to_fp8_e4m3_per_expert(a_base_fp16, fp8_acts, K_dim, d_offsets, ne,
                                                 moe_.d_fp8_scales, stream);
        } else {
            quantize_fp16_to_fp8_e4m3_scaled(a_base_fp16, fp8_acts, expanded * K_dim, 1.0f, stream);
        }

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

        gemm_moe_batched(fp8_acts, c_base_fp16, h_offsets.data(), weight_ptrs.data(), K_dim, N_dim,
                         QType::FP8_E4M3, ne, stream, moe_.d_work_ptrs, QType::F16,
                         moe_.d_fp8_scales ? h_act_scales.data() : nullptr);
    };

    char* gathered_base = static_cast<char*>(moe_.gathered.data);
    char* expert_gate_base = static_cast<char*>(moe_.expert_gate.data);
    char* expert_up_base = static_cast<char*>(moe_.expert_up.data);
    char* expert_swiglu_base = static_cast<char*>(moe_.expert_swiglu.data);
    char* expert_down_base = static_cast<char*>(moe_.expert_down.data);

    if (!non_gated_experts)
        chunked_fp8_gemm(ly.expert_gate_packed, ly.expert_gate_packed.qtype, gathered_base,
                         expert_gate_base, d, eff);
    chunked_fp8_gemm(ly.expert_up_packed, up_qtype, gathered_base, expert_up_base, d, eff);
    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                            non_gated_experts, expanded, eff, compute_dtype_, cfg.ffn_activation,
                            stream);
    char* fp8_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
    chunked_fp8_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype, fp8_down_act,
                     expert_down_base, eff, d);
    return true;
}

bool GraphExecutor::try_run_moe_fp16_batch_prefill(int layer, cudaStream_t stream, int n, int d,
                                                   int eff, int ne, int expanded,
                                                   bool non_gated_experts, QType up_qtype,
                                                   const MoeRoutingResult& routing,
                                                   bool fp32_down_active, void*& fp32_down_buf) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

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
        // Prefer the pre-allocated persistent scratch (avoids per-call cudaMallocAsync).
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

    auto debug_dump = [&](const char* name, void* data, int rows, int cols) {
        if (!debug_forward_enabled() || layer != 0) return;
        int64_t sh[2] = {rows, cols};
        Tensor v(data, compute_dtype_, 2, sh, true);
        debug_tensor_stats(name, v, stream);
    };

    if (!non_gated_experts)
        batch_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_packed.qtype, gathered_base,
                           expert_gate_base, d, eff);
    debug_dump("L0_moe_gate_out", moe_.expert_gate.data, expanded, eff);

    batch_dequant_gemm(ly.expert_up_packed, up_qtype, gathered_base, expert_up_base, d, eff);
    debug_dump("L0_moe_up_out", moe_.expert_up.data, expanded, eff);

    apply_expert_activation(moe_.expert_gate.data, moe_.expert_up.data, moe_.expert_swiglu.data,
                            non_gated_experts, expanded, eff, compute_dtype_, cfg.ffn_activation,
                            stream);
    debug_dump("L0_moe_swiglu_out", moe_.expert_swiglu.data, expanded, eff);

    char* down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
    char* down_target =
        fp32_down_active ? static_cast<char*>(fp32_down_buf) : expert_down_base;
    QType down_out_dtype = fp32_down_active ? QType::F32 : QType::F16;
    batch_dequant_gemm(ly.expert_down_packed, ly.expert_down_packed.qtype, down_act, down_target,
                       eff, d, down_out_dtype);
    if (!fp32_down_active) debug_dump("L0_moe_down_out", moe_.expert_down.data, expanded, d);

    return true;
}

bool GraphExecutor::try_run_moe_gemma4_ggml_prefill(int layer, cudaStream_t stream, int n, int d,
                                                    int eff, int top_k, QType up_qtype, float eps,
                                                    const MoeRoutingResult& routing,
                                                    const Tensor& no, const Tensor& norm_w,
                                                    Tensor& h, const Tensor& r,
                                                    bool moe_use_fp32_residual,
                                                    bool& residual_fused) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    if (layer == 0)
        IMP_LOG_INFO("MoE prefill: ggml MMVQ per-token path (n=%d, top_k=%d)", n, top_k);

    half* gate_buf = static_cast<half*>(moe_.expert_gate.data);
    half* up_buf = static_cast<half*>(moe_.expert_up.data);
    half* down_buf = static_cast<half*>(moe_.expert_down.data);

    size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
    size_t up_stride = expert_stride(ly.expert_up_packed, up_qtype);
    size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_packed.qtype);

    // Persistent Q8_1 scratch + FP32 norm scratch (resized lazily).
    static void* s_q8_scratch = nullptr;
    static size_t s_q8_scratch_size = 0;
    size_t q8_needed = std::max(static_cast<size_t>((d + 31) / 32) * 36,
                                static_cast<size_t>((eff + 31) / 32) * 36);
    if (!s_q8_scratch || s_q8_scratch_size < q8_needed) {
        if (s_q8_scratch) cudaFree(s_q8_scratch);
        cudaMalloc(&s_q8_scratch, q8_needed);
        s_q8_scratch_size = q8_needed;
    }
    static float* s_norm_fp32 = nullptr;
    static int s_norm_fp32_d = 0;
    if (!s_norm_fp32 || s_norm_fp32_d < d) {
        if (s_norm_fp32) cudaFree(s_norm_fp32);
        cudaMalloc(&s_norm_fp32, static_cast<size_t>(d) * sizeof(float));
        s_norm_fp32_d = d;
    }

    // M2 from review/phase5_synthesis.md §2.2: batch the per-token expert-
    // index D2H + sync into a single prefetch at function entry. Reduces
    // 2 * n syncs (one per token, three GEMVs each) down to ONE sync per
    // layer call. Restores the prefill graph-capture story up to (but not
    // including) the grouped-mmvq kernel work that would eliminate the
    // remaining sync — see follow-up: replacing this function with a
    // device-indexed grouped mmvq lets the whole layer capture cleanly.
    //
    // top_k is bounded by FFN active-expert count (max 32 per kernel
    // launch guard at line ~2154 in the old per-token form). Use a
    // std::vector since n*top_k can exceed the old 32-entry stack array.
    std::vector<int32_t> h_all_experts(static_cast<size_t>(n) * top_k);
    cudaMemcpyAsync(h_all_experts.data(), routing.expert_indices.data,
                    static_cast<size_t>(n) * top_k * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int t = 0; t < n; t++) {
        const float* tok_weights = static_cast<const float*>(routing.expert_weights.data) +
                                   static_cast<int64_t>(t) * top_k;

        const float* tok_norm_f32 = s_norm_fp32;
        if (fp32_accum_buf_ != nullptr) {
            int64_t tok_shape[2] = {1, static_cast<int64_t>(d)};
            Tensor fp32_tok(static_cast<float*>(fp32_hidden_.data) + static_cast<int64_t>(t) * d,
                            QType::F32, 2, tok_shape, true);
            rmsnorm_fp32_to_fp32(fp32_tok, norm_w, s_norm_fp32, 1, d, eps, stream, norm_w_off_);
        } else {
            tok_norm_f32 = nullptr;
        }

        // Per-token expert IDs: read from the pre-fetched host array (no D2H,
        // no sync). Old form: stack `int32_t h_experts[32]` + per-token
        // cudaMemcpyAsync + cudaStreamSynchronize — M2 batched both above.
        const int32_t* h_experts = h_all_experts.data() + static_cast<size_t>(t) * top_k;

        auto do_mmvq = [&](const uint8_t* w, half* out, QType qt, int rows, int cols) {
            if (tok_norm_f32) {
                ggml_mmvq_q4k_f32(w, tok_norm_f32, out, 1, rows, cols, s_q8_scratch,
                                  s_q8_scratch_size, stream);
                return;
            }
            const half* tok_norm_fp16 = static_cast<const half*>(no.data) +
                                        static_cast<int64_t>(t) * d;
            switch (qt) {
                case QType::Q8_0:
                    ggml_mmvq_q8_0(w, tok_norm_fp16, out, 1, rows, cols, s_q8_scratch,
                                   s_q8_scratch_size, stream);
                    break;
                case QType::Q4_K:
                default:
                    ggml_mmvq_q4k(w, tok_norm_fp16, out, 1, rows, cols, s_q8_scratch,
                                  s_q8_scratch_size, stream);
                    break;
            }
        };
        for (int k = 0; k < top_k; k++) {
            int32_t eid = h_experts[k];
            const uint8_t* gate_w = static_cast<const uint8_t*>(ly.expert_gate_packed.data) +
                                    static_cast<size_t>(eid) * gate_stride;
            const uint8_t* up_w = static_cast<const uint8_t*>(ly.expert_up_packed.data) +
                                  static_cast<size_t>(eid) * up_stride;
            do_mmvq(gate_w, gate_buf + static_cast<int64_t>(k) * eff,
                    ly.expert_gate_packed.qtype, eff, d);
            do_mmvq(up_w, up_buf + static_cast<int64_t>(k) * eff, up_qtype, eff, d);
        }

        half* swiglu_buf = static_cast<half*>(moe_.expert_swiglu.data);
        apply_expert_activation(gate_buf, up_buf, swiglu_buf, /*non_gated=*/false, top_k, eff,
                                compute_dtype_, cfg.ffn_activation, stream);

        for (int k = 0; k < top_k; k++) {
            int32_t eid = h_experts[k];
            const uint8_t* w_down = static_cast<const uint8_t*>(ly.expert_down_packed.data) +
                                    static_cast<size_t>(eid) * down_stride;
            half* act_k = swiglu_buf + static_cast<int64_t>(k) * eff;
            half* out_k = down_buf + static_cast<int64_t>(k) * d;
            switch (ly.expert_down_packed.qtype) {
                case QType::Q5_K:
                    ggml_mmvq_q5k(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                  stream); break;
                case QType::Q8_0:
                    ggml_mmvq_q8_0(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                   stream); break;
                case QType::Q5_1:
                    ggml_mmvq_q5_1(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                   stream); break;
                case QType::Q4_K:
                default:
                    ggml_mmvq_q4k(w_down, act_k, out_k, 1, d, eff, s_q8_scratch, s_q8_scratch_size,
                                  stream); break;
            }
        }

        half* h_tok = static_cast<half*>(h.data) + static_cast<int64_t>(t) * d;
        bool has_shared_expert = (ly.w_up_shared.data != nullptr);
        const void* res_ptr =
            (has_shared_expert || moe_use_fp32_residual)
                ? nullptr
                : static_cast<const void*>(static_cast<const half*>(r.data) +
                                           static_cast<int64_t>(t) * d);
        moe_weighted_sum_residual(down_buf, tok_weights, res_ptr, h_tok, d, top_k, stream);
    }
    if (ly.w_up_shared.data == nullptr && !moe_use_fp32_residual)
        residual_fused = true;
    return true;
}

void GraphExecutor::compute_moe_routing(int layer, cudaStream_t stream, int n, int d, int ne,
                                        int top_k, const Tensor& router_in,
                                        bool fp32_gate_logits_ready, bool will_decode_fast,
                                        const void* router_bias_ptr, bool use_sigmoid,
                                        bool norm_weights, MoeRoutingResult& routing) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    auto run_topk = [&](const Tensor& logits_f32) {
        if (moe_.routing_buffers.pool) {
            moe_topk_gating(logits_f32, top_k, moe_.routing_buffers, routing, stream, use_sigmoid,
                            norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(logits_f32, top_k, routing, stream, use_sigmoid, norm_weights,
                            router_bias_ptr);
        }
    };

    // Fused gate GEMV + topk only profitable when n_experts ≤ warps (8).
    // Higher expert counts (e.g. 128 in Qwen3-Coder) prefer separate
    // gemv_gate_fp32 (128 parallel blocks).
    constexpr int kMaxFusedExperts = 8;
    const std::string& dl = runtime_config().diagnostics.dump_logits_dir;
    bool dump_logits = !dl.empty() && (layer == 29 || dl == "all");

    if (fp32_gate_logits_ready) {
        Tensor gate_logits_f32 = slice_rows(moe_.gate_logits, n);
        if (dump_logits) dump_top8_gate_logits(layer, n, ne, static_cast<const float*>(gate_logits_f32.data));
        run_topk(gate_logits_f32);
    } else if (ne <= kMaxFusedExperts && n == 1 && compute_dtype_ == QType::F16 &&
               ly.moe_gate.qtype == QType::F16 && moe_.routing_buffers.pool && will_decode_fast) {
        // Fused: gate GEMV + softmax/sigmoid + top-k in one kernel
        moe_gate_topk_fused(static_cast<const half*>(ly.moe_gate.data),
                            static_cast<const half*>(router_in.data), ne, d, top_k,
                            moe_.routing_buffers, routing, stream, use_sigmoid, norm_weights,
                            router_bias_ptr);
    } else {
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
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const half*>(gate_logits_tmp.data),
                    static_cast<float*>(gate_logits_f32.data), numel);
            } else {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                                                   static_cast<size_t>(numel) * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream));
            }
        }

        if (debug_forward_enabled() && layer == 0) {
            std::vector<float> rl(ne);
            int last_tok = n - 1;
            cudaMemcpy(rl.data(), static_cast<const float*>(gate_logits_f32.data) + last_tok * ne,
                       ne * sizeof(float), cudaMemcpyDeviceToHost);
            double rsum = 0, rss = 0;
            for (auto v : rl) { rsum += v; rss += v * v; }
            fprintf(stderr,
                    "[DEBUG_FWD] L0_router_logits[%d]: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f\n",
                    last_tok, rsum, std::sqrt(rss), rl[0], rl[1], rl[2]);
        }
        if (dump_logits) dump_top8_gate_logits(layer, n, ne, static_cast<const float*>(gate_logits_f32.data));
        run_topk(gate_logits_f32);
    }

    // Routing decision dump
    if (const std::string& drv = runtime_config().diagnostics.dump_routing_dir; !drv.empty()) {
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
                    layer, last_tok, h_idx[0], h_idx[1], h_idx[2], h_idx[3], h_idx[4], h_idx[5],
                    h_idx[6], h_idx[7], h_wts[0], h_wts[1], h_wts[2], h_wts[3], h_wts[4], h_wts[5],
                    h_wts[6], h_wts[7]);
        }
    }

    // Per-expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data), cfg.expert_weights_scale, n_weights);
    }

    // Gemma-4: per-expert output scale absorbed into routing weights.
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
}

void GraphExecutor::run_moe_decode_fast(int layer, cudaStream_t stream, int n, int d, int eff,
                                        int top_k, const MoeRoutingResult& routing,
                                        const Tensor& no, Tensor& h, const Tensor& r,
                                        bool moe_use_fp32_residual, bool moe_fused_norm_q8,
                                        bool will_skip_residual_copy, bool& residual_fused) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    bool non_gated_experts =
        (ly.expert_gate_packed.data == nullptr &&
         (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));
    QType up_qtype = ly.expert_up_packed.qtype;

    const int32_t* expert_indices = static_cast<const int32_t*>(routing.expert_indices.data);
    const float* expert_weights = static_cast<const float*>(routing.expert_weights.data);

    half* norm_ptr = static_cast<half*>(no.data);
    half* gate_buf = static_cast<half*>(moe_.expert_gate.data);   // [top_k, eff]
    half* up_buf = static_cast<half*>(moe_.expert_up.data);       // [top_k, eff]
    half* act_buf = static_cast<half*>(moe_.expert_swiglu.data);  // [top_k, eff]
    half* down_buf = static_cast<half*>(moe_.expert_down.data);   // [top_k, d]

    // NVFP4 MoE sub-path: takes FP16 input directly, no Q8_1 needed
    bool use_nvfp4_moe = (ly.nvfp4_moe_up_ptr != nullptr && ly.nvfp4_moe_down_ptr != nullptr);
    if (use_nvfp4_moe && !non_gated_experts)
        use_nvfp4_moe = (ly.nvfp4_moe_gate_ptr != nullptr);

    if (use_nvfp4_moe) {
        if (!non_gated_experts) {
            gemv_nvfp4_moe_gate_up_fused(*ly.nvfp4_moe_gate_ptr, *ly.nvfp4_moe_up_ptr, expert_indices,
                                         norm_ptr, gate_buf, up_buf, eff, d, top_k, stream);
            gemv_nvfp4_moe_swiglu_decode(*ly.nvfp4_moe_down_ptr, expert_indices, gate_buf, up_buf,
                                         down_buf, d, eff, /*x_stride=*/eff, top_k, stream);
        } else {
            gemv_nvfp4_moe_decode(*ly.nvfp4_moe_up_ptr, expert_indices, norm_ptr, up_buf, eff, d,
                                  /*x_stride=*/0, top_k, stream);
            int64_t act_shape[2] = {static_cast<int64_t>(top_k), static_cast<int64_t>(eff)};
            Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
            relu_sqr_inplace(up_t, stream);
            gemv_nvfp4_moe_decode(*ly.nvfp4_moe_down_ptr, expert_indices, up_buf, down_buf, d, eff,
                                  /*x_stride=*/eff, top_k, stream);
        }
        bool has_shared_expert = (ly.w_up_shared.data != nullptr);
        const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                                  ? nullptr
                                  : (will_skip_residual_copy ? h.data : r.data);
        moe_weighted_sum_residual(down_buf, expert_weights, res_ptr, h.data, d, top_k, stream);
        if (!has_shared_expert && !moe_use_fp32_residual)
            residual_fused = true;
        return;
    }

    // dp4a/FP16 sub-path: gate+up projection (fused if gated)
    bool use_dp4a = (qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr);

    if (use_dp4a) {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        if (!moe_fused_norm_q8)
            quantize_fp16_to_q8_1(norm_ptr, q8, qscratch_.d8_buf, d, stream);

        size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

        if (!non_gated_experts) {
            size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
            auto gate_up_fused = (up_qtype == QType::Q6_K) ? gemv_q6k_q8_1_moe_gate_up_fused
                               : (up_qtype == QType::Q4_K) ? gemv_q4_k_q8_1_moe_gate_up_fused
                               : (up_qtype == QType::Q5_K) ? gemv_q5_k_q8_1_moe_gate_up_fused
                               : (up_qtype == QType::Q4_0) ? gemv_q4_0_q8_1_moe_gate_up_fused
                               : (up_qtype == QType::Q2_K) ? gemv_q2_k_q8_1_moe_gate_up_fused
                               : (up_qtype == QType::Q3_K) ? gemv_q3_k_q8_1_moe_gate_up_fused
                                                           : gemv_q8_0_q8_1_moe_gate_up_fused;
            gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data, expert_indices, q8,
                          qscratch_.d8_buf, gate_buf, up_buf, eff, d, gate_stride, up_stride_bytes,
                          /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
        } else {
            auto up_gemv = (up_qtype == QType::Q6_K) ? gemv_q6k_q8_1_moe_decode
                         : (up_qtype == QType::Q4_0) ? gemv_q4_0_q8_1_moe_decode
                         : (up_qtype == QType::Q4_K) ? gemv_q4_k_q8_1_moe_decode
                         : (up_qtype == QType::Q5_K) ? gemv_q5_k_q8_1_moe_decode
                         : (up_qtype == QType::Q2_K) ? gemv_q2_k_q8_1_moe_decode
                         : (up_qtype == QType::Q3_K) ? gemv_q3_k_q8_1_moe_decode
                                                     : gemv_q8_0_q8_1_moe_decode;
            up_gemv(ly.expert_up_packed.data, expert_indices, q8, qscratch_.d8_buf, up_buf, eff, d,
                    up_stride_bytes, /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
        }
    } else {
        // FP16 dequant fallback — only Q6_K / Q8_0 wired.
        size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);
        if (!non_gated_experts) {
            size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_packed.qtype);
            if (up_qtype == QType::Q6_K) {
                gemv_q6k_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                           expert_indices, norm_ptr, gate_buf, up_buf, eff, d,
                                           gate_stride, up_stride_bytes, /*x_stride=*/0, top_k, stream);
            } else if (up_qtype == QType::Q8_0) {
                gemv_q8_0_moe_gate_up_fused(ly.expert_gate_packed.data, ly.expert_up_packed.data,
                                            expert_indices, norm_ptr, gate_buf, up_buf, eff, d,
                                            gate_stride, up_stride_bytes, /*x_stride=*/0, top_k, stream);
            } else {
                IMP_LOG_ERROR("MoE non-dp4a gate_up_fused: no kernel for qtype %d", (int)up_qtype);
            }
        } else {
            if (up_qtype == QType::Q6_K) {
                gemv_q6k_moe_decode(ly.expert_up_packed.data, expert_indices, norm_ptr, up_buf, eff, d,
                                    up_stride_bytes, /*x_stride=*/0, top_k, stream);
            } else if (up_qtype == QType::Q8_0) {
                gemv_q8_0_moe_decode(ly.expert_up_packed.data, expert_indices, norm_ptr, up_buf, eff, d,
                                     up_stride_bytes, /*x_stride=*/0, top_k, stream);
            } else {
                IMP_LOG_ERROR("MoE non-dp4a up projection: no kernel for qtype %d", (int)up_qtype);
            }
        }
    }

    // Activation + down projection
    if (use_dp4a) {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        int eff_q8_blocks = eff / 32;
        if (!non_gated_experts) {
            if (cfg.ffn_activation == FFNActivation::GEGLU) {
                if (layer == 0 && runtime_config().diagnostics.debug_forward)
                    fprintf(stderr, "[DEBUG_MoE] Using GEGLU activation for MoE experts\n");
                geglu_quantize_q8_1(gate_buf, up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
            } else {
                swiglu_quantize_q8_1(gate_buf, up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
            }
        } else {
            relu_sqr_quantize_q8_1(up_buf, q8, qscratch_.d8_buf, top_k * eff, stream);
        }
        QType dqt = ly.expert_down_packed.qtype;
        auto down_gemv = (dqt == QType::Q6_K) ? gemv_q6k_q8_1_moe_decode
                       : (dqt == QType::Q4_0) ? gemv_q4_0_q8_1_moe_decode
                       : (dqt == QType::Q4_K) ? gemv_q4_k_q8_1_moe_decode
                       : (dqt == QType::Q5_K) ? gemv_q5_k_q8_1_moe_decode
                       : (dqt == QType::Q2_K) ? gemv_q2_k_q8_1_moe_decode
                       : (dqt == QType::Q3_K) ? gemv_q3_k_q8_1_moe_decode
                       : (dqt == QType::Q5_1) ? gemv_q5_1_q8_1_moe_decode
                                              : gemv_q8_0_q8_1_moe_decode;
        size_t down_stride = expert_stride(ly.expert_down_packed, dqt);
        down_gemv(ly.expert_down_packed.data, expert_indices, q8, qscratch_.d8_buf, down_buf, d, eff,
                  down_stride, /*q8_1_stride=*/eff_q8_blocks, /*d8_stride=*/eff_q8_blocks, top_k, stream);
    } else {
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
            IMP_LOG_ERROR("MoE non-dp4a down projection: no kernel for qtype %d",
                          (int)ly.expert_down_packed.qtype);
        }
    }

    // Fused weighted sum + FP16 output (+ residual if no shared expert)
    bool has_shared_expert = (ly.w_up_shared.data != nullptr);
    const void* res_ptr = (has_shared_expert || moe_use_fp32_residual)
                              ? nullptr
                              : (will_skip_residual_copy ? h.data : r.data);
    moe_weighted_sum_residual(down_buf, expert_weights, res_ptr, h.data, d, top_k, stream);
    if (!has_shared_expert && !moe_use_fp32_residual)
        residual_fused = true;
}

void GraphExecutor::run_shared_expert_ffn(int layer, cudaStream_t stream, int n, int d,
                                          float eps, const Tensor& no, Tensor& h) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);

    // DIAGNOSTIC: moe.no_shared_mlp config flag (was IMP_NO_SHARED_MLP env).
    static const bool s_no_shared_mlp = runtime_config().moe.no_shared_mlp;
    if (ly.w_up_shared.data == nullptr || s_no_shared_mlp) return;

    int eff_shared = static_cast<int>(ly.w_up_shared.shape[0]);
    bool shared_gated = (ly.w_gate_shared.data != nullptr);

    // Reuse moe_.expert_gate, moe_.expert_up, moe_.expert_swiglu as scratch.
    int64_t sh_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(eff_shared)};
    Tensor sh_up(moe_.expert_up.data, compute_dtype_, 2, sh_shape, true);
    Tensor sh_swiglu(moe_.expert_swiglu.data, compute_dtype_, 2, sh_shape, true);

    // Down projection output: [n, d_model]. Reuse moe_.expert_down.
    int64_t sh_down_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
    Tensor sh_down(moe_.expert_down.data, compute_dtype_, 2, sh_down_shape, true);

    auto ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq);
    gemm_via_handle_(ly.w_up_shared_id, no, sh_up, ctx);

    if (shared_gated) {
        Tensor sh_gate(moe_.expert_gate.data, compute_dtype_, 2, sh_shape, true);
        gemm_via_handle_(ly.w_gate_shared_id, no, sh_gate, ctx);
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
    Tensor& sh_act = shared_gated ? sh_swiglu : sh_up;
    if (layer == 0)
        debug_tensor_stats_all("L0_sh_act_preDown", sh_act, stream);
    gemm_via_handle_(ly.w_down_shared_id, sh_act, sh_down, ctx);

    if (layer == 0) {
        debug_tensor_stats_all("L0_sh_down_raw", sh_down, stream);
        debug_tensor_rows("L0_sh_down_rows", sh_down, stream);
        debug_tensor_rows("L0_shared_norm_in", view_tokens(no, n), stream);
    }
    // Gemma-4: shared MLP can overflow FP16 at deep layers; sanitize inf/NaN
    // before post_ffw_norm_1 so rmsnorm doesn't produce all-zero output.
    if (cfg.arch == ModelArch::GEMMA4) {
        sanitize_fp16(static_cast<__half*>(sh_down.data), static_cast<int64_t>(n) * d, stream);
    }
    if (cfg.arch == ModelArch::GEMMA4 && ly.ffn_post_norm_1.data != nullptr &&
        !cfg.overrides.gemma4.no_post_ffw_1) {
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
    // Qwen3-Next / Qwen3.6: per-token sigmoid gate on shared-expert output.
    static const bool skip_shexp_gate = runtime_config().moe.no_shexp_gate;
    if (!skip_shexp_gate && ly.shared_expert_gate_inp.data != nullptr &&
        ly.shared_expert_gate_inp.on_device && compute_dtype_ == QType::F16) {
        shared_expert_gate_scale(no.data, ly.shared_expert_gate_inp.data, sh_down.data, n, d, d, stream);
    }
    elementwise_add(h, sh_down, stream);
}


}  // namespace imp

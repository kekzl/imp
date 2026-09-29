// Prompt logprobs (#2207, #2257): LM head over a row chunk into a caller logits buffer, then one
// fused log-softmax/rank/top-N pass per row (compute/prompt_logprobs_rows.cu). Heads without a
// one-GEMM route keep the per-batch driver of perplexity_nll_partial.

#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "compute/gemm.h"
#include "compute/prompt_logprobs_rows.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/dequant_gpu.h"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

namespace {

constexpr int kPlpNormThreads = 1024;

// Per row: RMSNorm then Q8_1 round trip to FP16 q*d, the math of rmsnorm_quantize_q8_1_kernel
// (gemm_dp4a.cu), so the dequantized-head GEMM sees the dp4a head's activations. d_model <= 32768.
__global__ void __launch_bounds__(kPlpNormThreads) plp_rmsnorm_q8_roundtrip_kernel(
    const half* __restrict__ x, const half* __restrict__ weight, half* __restrict__ out, int d_model,
    float eps, float weight_offset) {
    __shared__ float warp_reduce[32];
    __shared__ float s_inv_rms;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int n_q8_blocks = d_model >> 5;
    const half* xr = x + static_cast<int64_t>(blockIdx.x) * d_model;
    half* orow = out + static_cast<int64_t>(blockIdx.x) * d_model;

    float x_cache[32];
    float sum_sq = 0.0f;
    int n_cached = 0;
    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        const float v = __half2float(xr[b * 32 + lane]);
        x_cache[n_cached++] = v;
        sum_sq += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    int cache_idx = 0;
    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        const float val = x_cache[cache_idx++] * inv_rms *
                          (__half2float(weight[b * 32 + lane]) + weight_offset);
        float amax = fabsf(val);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        const int q = __float2int_rn(val * id);
        orow[b * 32 + lane] = __float2half(static_cast<float>(q) * d);
    }
}

}  // namespace

bool GraphExecutor::prompt_lm_head_chunk_(int row0, int csz, float* lg, int& slab_w, cudaStream_t stream) {
    const auto& cfg = model_->config();
    const int V = cfg.vocab_size;
    const int K = cfg.d_model;
    const int64_t lg_shape[2] = {csz, V};
    Tensor lgt(lg, QType::F32, 2, lg_shape, true);
    Tensor hc = view_tokens(hidden_, row0 + csz).slice(row0, row0 + csz);
    slab_w = V;
    if (lm_head_fp8_(hc, lgt, stream))
        return true;

    // Route of for_each_lm_head_batch_'s dp4a arm only: GGUF head, no FP8/NVFP4 copy, FP16 compute.
    const void* w = model_->output_proj().data;
    const QType qt = model_->out_proj_.qtype;
    const WeightHandle* lm_h = (model_->out_proj_id != kInvalidTensorID)
                                   ? &registry_.handle(model_->out_proj_id)
                                   : nullptr;
    const StorageTier tier = lm_h ? lm_h->primary_tier : StorageTier::Undefined;
    const bool has_fp8 = wcache_.fp8.count(w) != 0;
    const bool nvfp4 = !has_fp8 && (tier == StorageTier::NVFP4 || wcache_.nvfp4.count(w) != 0);
    const bool dp4a = qscratch_.q8_1_buf && compute_dtype_ == QType::F16 && is_dp4a_qtype(qt) &&
                      !dispatch_policy().gemm.no_dp4a_lm && tier != StorageTier::MXFP4 && !has_fp8;
    // Vocab slab = dequant-scratch rows; below 256 rows the per-batch driver wins.
    const int slab = qscratch_.dequant
                         ? static_cast<int>(std::min<size_t>(qscratch_.dequant_size /
                                                                 (static_cast<size_t>(K) * sizeof(half)),
                                                             static_cast<size_t>(V)))
                         : 0;
    if (!w || nvfp4 || !dp4a || !dequant_gpu_supported(qt) || slab < 256 || (K % 32) != 0 || K > 32768)
        return false;

    Tensor noc = view_tokens(norm_out_, csz);
    plp_rmsnorm_q8_roundtrip_kernel<<<csz, kPlpNormThreads, 0, stream>>>(
        static_cast<const half*>(hc.data), static_cast<const half*>(model_->output_norm().data),
        static_cast<half*>(noc.data), K, cfg.rms_norm_eps, norm_w_off_);
    IMP_CUDA_CHECK_LAUNCH();
    const size_t row_bytes = qtype_row_bytes(qt, K);
    for (int n0 = 0; n0 < V; n0 += slab) {
        const int ws = std::min(slab, V - n0);
        dequant_gpu(static_cast<const uint8_t*>(w) + static_cast<size_t>(n0) * row_bytes, qscratch_.dequant,
                    qt, ws, K, stream);
        const int64_t w_shape[2] = {ws, K};
        const int64_t c_shape[2] = {csz, ws};
        Tensor wt(qscratch_.dequant, QType::F16, 2, w_shape, true);
        Tensor ct(lg + static_cast<size_t>(n0) * csz, QType::F32, 2, c_shape, true);
        gemm(noc, wt, ct, 1.0f, 0.0f, stream);
    }
    slab_w = slab;
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("prompt_logprobs: one-GEMM LM head, %d-row chunk, %d-column vocab slabs (%s head)", csz,
                     slab, qtype_name(qt));
    }
    return true;
}

void GraphExecutor::prompt_logprobs_partial(const int32_t* d_targets, int n_rows, int top_n, float* d_lp,
                                            int32_t* d_rank, int32_t* d_top_ids, float* d_top_lp,
                                            float* d_logits, int chunk_rows, cudaStream_t stream) {
    if (!initialized_ || n_rows <= 0 || !d_targets || !d_lp || !d_rank)
        return;
    if (top_n > 0 && (!d_top_ids || !d_top_lp))
        return;
    const auto& cfg = model_->config();
    const int V = cfg.vocab_size;
    auto top_at = [&](auto* p, int row0) { return p ? p + static_cast<int64_t>(row0) * top_n : nullptr; };
    for (int row0 = 0; d_logits && chunk_rows > 0 && row0 < n_rows; row0 += chunk_rows) {
        const int csz = std::min(chunk_rows, n_rows - row0);
        int slab_w = V;
        if (!prompt_lm_head_chunk_(row0, csz, d_logits, slab_w, stream)) {
            if (row0 == 0)
                break;  // no one-GEMM route for this head: the per-batch driver below
            IMP_LOG_ERROR("prompt_logprobs: LM head route changed at row %d", row0);
            return;
        }
        if (cfg.final_logit_softcap > 0.0f && !dispatch_policy().generation.no_logit_softcap) {
            const int64_t total = static_cast<int64_t>(csz) * V;
            logit_softcap_fp32_kernel<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(
                d_logits, cfg.final_logit_softcap, 1.0f / cfg.final_logit_softcap, total);
            IMP_CUDA_CHECK_LAUNCH();
        }
        prompt_logprobs_rows(d_logits, csz, V, slab_w, d_targets + row0, top_n, d_lp + row0, d_rank + row0,
                             top_at(d_top_ids, row0), top_at(d_top_lp, row0), stream);
        if (row0 + csz >= n_rows)
            return;
    }
    // allow_cutlass: same LM head as perplexity_nll_partial, so PPL from these rows matches the tool.
    for_each_lm_head_batch_(n_rows, stream, /*allow_cutlass=*/true, [&](const Tensor& lg, int row0, int csz) {
        prompt_logprobs_rows(static_cast<const float*>(lg.data), csz, V, V, d_targets + row0, top_n,
                             d_lp + row0, d_rank + row0, top_at(d_top_ids, row0), top_at(d_top_lp, row0),
                             stream);
    });
}

}  // namespace imp

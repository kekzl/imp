// Teacher-forced perplexity over a token sequence.
//
// Assumes the caller has just run a prefill of `tokens[0..n-1]` so the
// persistent-workspace `hidden_` holds the final-layer hidden state for ALL n
// positions (forward_logits leaves it intact — it only slices the last token
// for the production LM head). We then apply the (tier-aware) LM head to every
// position in chunks of <= max_logit_tokens_ (logits_ is batch-sized) and
// accumulate the negative log-likelihood of each actual next token.
//
// PPL = exp( (1/(n-1)) * sum_{i=0}^{n-2} -log softmax(logits_i)[tokens_{i+1}] ).
//
// Bench/eval only — does NOT touch the production forward_logits path. Reuses
// gemm_via_handle_ (tier-aware: NVFP4 / FP8 / FP16 / GGUF all handled).

#include "exec/executor.h"
#include "exec/gemm_context.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace imp {

// One block per row. Online max + logsumexp over the vocab, then accumulate
// -logprob(target) into a global double accumulator. Skips the final corpus
// position (no next token to predict).
__global__ void perplexity_nll_kernel(const float* __restrict__ logits,  // [csz, V]
                                       const int32_t* __restrict__ tokens, int chunk_start, int n,
                                       int V, double* __restrict__ nll_accum) {
    int row = blockIdx.x;                 // local row in this chunk
    int global_pos = chunk_start + row;   // position in the corpus
    if (global_pos >= n - 1)
        return;                           // last position has no target
    const float* lg = logits + static_cast<int64_t>(row) * V;
    int target = tokens[global_pos + 1];

    __shared__ float s_max;
    __shared__ double s_sum;
    int tid = threadIdx.x;

    // max over V
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += blockDim.x)
        local_max = fmaxf(local_max, lg[i]);
    __shared__ float red[256];
    red[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        s_max = red[0];
    __syncthreads();
    float mx = s_max;

    // sum exp(x - max)
    double local_sum = 0.0;
    for (int i = tid; i < V; i += blockDim.x)
        local_sum += exp(static_cast<double>(lg[i] - mx));
    __shared__ double redd[256];
    redd[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            redd[tid] += redd[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        double lse = log(redd[0]) + static_cast<double>(mx);
        double logprob = static_cast<double>(lg[target]) - lse;
        atomicAdd(nll_accum, -logprob);
    }
}

double GraphExecutor::perplexity_nll(const int32_t* tokens, int n, cudaStream_t stream) {
    if (!initialized_ || n < 2) {
        IMP_LOG_ERROR("perplexity_nll: not initialized or n < 2 (n=%d)", n);
        return -1.0;
    }
    const auto& cfg = model_->config();
    const int V = cfg.vocab_size;
    const int mb = max_logit_tokens_ > 0 ? max_logit_tokens_ : 1;

    // Ensure the (single-chunk) prefill that populated hidden_ is complete.
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    int32_t* d_tokens = nullptr;
    double* d_nll = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tokens, static_cast<size_t>(n) * sizeof(int32_t)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_nll, sizeof(double)));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_tokens, tokens, static_cast<size_t>(n) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_nll, 0, sizeof(double), stream));

    GemmContext ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config());

    // LM-head tier detection (mirror forward_logits): NVFP4 decode-cache LM heads
    // must use the per-row gemv_nvfp4_kpar_fp32 path, NOT gemm_via_handle_ (which
    // doesn't read the NVFP4 cache → garbage logits). FP16/generic uses the handle.
    const WeightHandle* lm_h =
        (model_->out_proj_id != kInvalidTensorID) ? &registry_.handle(model_->out_proj_id) : nullptr;
    const StorageTier lm_tier = lm_h ? lm_h->primary_tier : StorageTier::Undefined;
    auto lm_nvfp4_it = wcache_.nvfp4.find(model_->output_proj().data);
    const bool lm_nvfp4_secondary = (lm_nvfp4_it != wcache_.nvfp4.end());
    const bool lm_has_fp8 = (wcache_.fp8.count(model_->output_proj().data) != 0);
    const bool lm_is_nvfp4 = !lm_has_fp8 && ((lm_tier == StorageTier::NVFP4) || lm_nvfp4_secondary);
    NvFP4QuantResult nvfp4_lm_r{};
    if (lm_is_nvfp4) {
        if (lm_nvfp4_secondary) {
            nvfp4_lm_r = lm_nvfp4_it->second;
        } else {
            nvfp4_lm_r.packed_data = lm_h->payload.nvfp4.data;
            nvfp4_lm_r.micro_scales = lm_h->payload.nvfp4.block_scales;
            nvfp4_lm_r.tensor_scale =
                (lm_h->payload.nvfp4.tensor_scale != nullptr) ? *lm_h->payload.nvfp4.tensor_scale : 1.0f;
            nvfp4_lm_r.N = cfg.vocab_size;
            nvfp4_lm_r.K = cfg.d_model;
        }
    }

    Tensor hidden_all = view_tokens(hidden_, n);
    for (int c = 0; c < n; c += mb) {
        int csz = (n - c < mb) ? (n - c) : mb;
        Tensor hc = hidden_all.slice(c, c + csz);          // [csz, d]
        Tensor lg = view_tokens(logits_, csz);             // [csz, V]
        if (lm_is_nvfp4) {
            Tensor no_row = view_tokens(norm_out_, 1);
            for (int r = 0; r < csz; ++r) {
                Tensor h_row = hc.slice(r, r + 1);
                Tensor lg_row = lg.slice(r, r + 1);
                rmsnorm(h_row, model_->output_norm(), no_row, cfg.rms_norm_eps, stream, norm_w_off_);
                gemv_nvfp4_kpar_fp32(nvfp4_lm_r, static_cast<const half*>(no_row.data),
                                     static_cast<float*>(lg_row.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else {
            Tensor noc = view_tokens(norm_out_, csz);      // [csz, d]
            rmsnorm(hc, model_->output_norm(), noc, cfg.rms_norm_eps, stream, norm_w_off_);
            gemm_via_handle_(model_->out_proj_id, noc, lg, ctx);
        }
        perplexity_nll_kernel<<<csz, 256, 0, stream>>>(static_cast<const float*>(lg.data), d_tokens, c,
                                                        n, V, d_nll);
    }

    double h_nll = 0.0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_nll, d_nll, sizeof(double), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    cudaFree(d_tokens);
    cudaFree(d_nll);

    double ppl = std::exp(h_nll / static_cast<double>(n - 1));
    IMP_LOG_INFO("perplexity_nll: n=%d  mean_nll=%.4f  PPL=%.4f", n, h_nll / (n - 1), ppl);
    return ppl;
}

}  // namespace imp

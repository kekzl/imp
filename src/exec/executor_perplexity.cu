// Teacher-forced perplexity over a token sequence.
//
// Two entry points:
//  - perplexity_nll(tokens, n): assumes a SINGLE-CHUNK prefill of
//    `tokens[0..n-1]` just ran, so the persistent-workspace `hidden_` holds
//    the final-layer hidden state for ALL n positions (forward_logits leaves
//    it intact — it only slices the last token for the production LM head).
//  - perplexity_nll_partial(...): per-chunk accumulation for CHUNKED prefill
//    (hidden_ only retains the most recent chunk). Driven by the engine's
//    step_prefill_one via begin/end_perplexity_capture — that flow backs
//    imp_perplexity, whose default config resolves to chunked prefill.
// Both apply the (tier-aware) LM head to every position in batches of
// <= max_logit_tokens_ (logits_ is batch-sized) and accumulate the negative
// log-likelihood of each actual next token.
//
// PPL = exp( (1/(n-1)) * sum_{i=0}^{n-2} -log softmax(logits_i)[tokens_{i+1}] ).
//
// Bench/eval only — does NOT touch the production forward_logits path. Reuses
// gemm_via_handle_ (tier-aware: NVFP4 / FP8 / FP16 / GGUF all handled).

#include "exec/executor.h"
#include "exec/executor_gemv_helpers.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <vector>

namespace imp {

// One block per row. Online max + logsumexp over the vocab, then write
// -logprob(target) to the per-position slot. Skips the final corpus position
// (no next token to predict). Per-position writes (host sums in fixed index
// order) instead of a global atomicAdd keep the NLL bit-reproducible —
// cross-block atomic accumulation order varies run-to-run.
__global__ void perplexity_nll_kernel(const float* __restrict__ logits,  // [csz, V]
                                       const int32_t* __restrict__ tokens, int chunk_start, int n,
                                       int V, double* __restrict__ nll_per_pos) {
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
        nll_per_pos[global_pos] = -logprob;
    }
}

// Shared eval-side LM-head driver (perplexity + spec-decode greedy verify):
// applies the tier-aware LM head to hidden_[0..n_rows) in batches of
// max_logit_tokens_, then hands each batch's logits_ view (softcap already
// applied — production parity) to `consume`.
void GraphExecutor::for_each_lm_head_batch_(int n_rows, cudaStream_t stream,
                                            const std::function<void(const Tensor&, int, int)>& consume) {
    if (!initialized_ || n_rows <= 0) {
        return;
    }
    const auto& cfg = model_->config();
    const int V = cfg.vocab_size;
    const int mb = max_logit_tokens_ > 0 ? max_logit_tokens_ : 1;
    const int chunk_len = n_rows;

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
    // Raw-GGUF-quant LM head (tied-embedding Gemma Q4_K etc.): mirror the
    // production use_dp4a_lm arm. Routing these through gemm_via_handle_ at
    // M=1 hits the GGUF GEMV handlers with an FP16 (un-quantized) input and
    // produced an illegal memory access on gemma-3-12b — which poisoned the
    // context and surfaced as the absurd PPL=1.0000 (zeroed NLL buffer).
    const auto out_qtype = model_->out_proj_.qtype;
    const bool use_dp4a_lm = qscratch_.q8_1_buf && compute_dtype_ == QType::F16 &&
                             is_dp4a_qtype(out_qtype) && !runtime_config().gemm.no_dp4a_lm &&
                             lm_tier != StorageTier::MXFP4 && !lm_has_fp8;
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

    Tensor hidden_all = view_tokens(hidden_, chunk_len);
    for (int c = 0; c < chunk_len; c += mb) {
        int csz = (chunk_len - c < mb) ? (chunk_len - c) : mb;
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
        } else if (use_dp4a_lm) {
            // Per-row: fused RMSNorm→Q8_1 quantize + dp4a GEMV (the q8_1
            // scratch holds exactly one row — same as production decode).
            auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
            for (int r = 0; r < csz; ++r) {
                Tensor h_row = hc.slice(r, r + 1);
                Tensor lg_row = lg.slice(r, r + 1);
                rmsnorm_quantize_q8_1(static_cast<const half*>(h_row.data),
                                      static_cast<const half*>(model_->output_norm().data), q8,
                                      qscratch_.d8_buf, nullptr, cfg.d_model, cfg.rms_norm_eps, stream,
                                      norm_w_off_);
                dispatch_gemv_fp32(out_qtype, model_->output_proj().data, q8, qscratch_.d8_buf,
                                   static_cast<float*>(lg_row.data), cfg.vocab_size, cfg.d_model, stream);
            }
        } else {
            Tensor noc = view_tokens(norm_out_, csz);      // [csz, d]
            rmsnorm(hc, model_->output_norm(), noc, cfg.rms_norm_eps, stream, norm_w_off_);
            gemm_via_handle_(model_->out_proj_id, noc, lg, ctx);
        }
        // Final logit softcap (Gemma-2/3/4): production forward_logits applies
        // it before sampling — without it the eval NLL measures a different
        // model than the one being served.
        if (cfg.final_logit_softcap > 0.0f && !runtime_config().generation.no_logit_softcap) {
            int64_t total = static_cast<int64_t>(csz) * V;
            int threads = 256;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            logit_softcap_fp32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<float*>(lg.data), cfg.final_logit_softcap,
                1.0f / cfg.final_logit_softcap, total);
        }
        consume(lg, c, csz);
    }
}

void GraphExecutor::perplexity_nll_partial(const int32_t* d_tokens, int n_total, int chunk_start,
                                           int chunk_len, double* d_nll, cudaStream_t stream) {
    if (!initialized_ || chunk_len <= 0 || n_total < 2 || !d_tokens || !d_nll) {
        return;
    }
    const int V = model_->config().vocab_size;
    for_each_lm_head_batch_(chunk_len, stream, [&](const Tensor& lg, int row0, int csz) {
        perplexity_nll_kernel<<<csz, 256, 0, stream>>>(static_cast<const float*>(lg.data), d_tokens,
                                                       chunk_start + row0, n_total, V, d_nll);
    });
}

// Two-phase row-wise argmax. A single block per row reads ~600 KB of fp32
// logits sequentially (~145 µs/row at vocab 151k); phase 1 splits each row
// across kArgmaxSplits blocks, phase 2 reduces the partials. Tie-break =
// smallest index, matching the production greedy argmax in sampling.cu —
// spec-decode verify must agree with what plain greedy decode would sample.
constexpr int kArgmaxSplits = 16;

__global__ void rowwise_argmax_partial_kernel(const float* __restrict__ logits, int V,
                                              float* __restrict__ pvals,
                                              int* __restrict__ pidxs) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    const int chunk = (V + kArgmaxSplits - 1) / kArgmaxSplits;
    const int begin = split * chunk;
    const int end = min(V, begin + chunk);
    const float* lg = logits + static_cast<int64_t>(row) * V;
    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int best_idx = 0;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        float v = lg[i];
        if (v > best || (v == best && i < best_idx)) {
            best = v;
            best_idx = i;
        }
    }
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] ||
                (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        pvals[row * kArgmaxSplits + split] = s_val[0];
        pidxs[row * kArgmaxSplits + split] = s_idx[0];
    }
}

// One thread per row: 16 partials is a trivial sequential reduce.
__global__ void rowwise_argmax_reduce_kernel(const float* __restrict__ pvals,
                                             const int* __restrict__ pidxs, int n_rows,
                                             int32_t* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows)
        return;
    float best = -FLT_MAX;
    int best_idx = 0;
    for (int s = 0; s < kArgmaxSplits; ++s) {
        const float v = pvals[row * kArgmaxSplits + s];
        const int i = pidxs[row * kArgmaxSplits + s];
        if (v > best || (v == best && i < best_idx)) {
            best = v;
            best_idx = i;
        }
    }
    out[row] = best_idx;
}

// Occurrence counts of the shared history per vocab id (production
// apply_penalties scans the history per vocab thread the same way; doing it
// once here amortizes the scan across all chunk rows).
__global__ void verify_hist_count_kernel(const int32_t* __restrict__ hist, int n_hist, int V,
                                         int32_t* __restrict__ counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;
    int c = 0;
    for (int i = 0; i < n_hist; ++i)
        if (hist[i] == idx) c++;
    counts[idx] = c;
}

// Apply repetition/frequency/presence penalties to a batch of chunk rows.
// Row `row0 + blockIdx.y` predicts the token after chunk position row0+y;
// its penalty set = shared history + d_draft[0..row-1] (the draft tokens the
// eager path would have emitted before this prediction). Formulas identical
// to apply_penalties_kernel in sampling.cu.
__global__ void verify_penalties_kernel(float* __restrict__ logits, int row0, int V,
                                        const int32_t* __restrict__ counts,
                                        const int32_t* __restrict__ draft, float rep_pen,
                                        float freq_pen, float pres_pen) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;
    const int row = row0 + blockIdx.y;
    int count = counts[idx];
    for (int i = 0; i < row; ++i)
        if (draft[i] == idx) count++;
    if (count == 0) return;
    float* lg = logits + static_cast<size_t>(blockIdx.y) * V + idx;
    float logit = *lg;
    if (rep_pen != 1.0f) {
        if (logit > 0.0f)
            logit /= rep_pen;
        else
            logit *= rep_pen;
    }
    logit -= freq_pen * static_cast<float>(count);
    logit -= pres_pen;
    *lg = logit;
}

void GraphExecutor::greedy_argmax_all(int n_rows, int32_t* d_out, cudaStream_t stream,
                                      const int32_t* d_hist, int n_hist, const int32_t* d_draft,
                                      float rep_pen, float freq_pen, float pres_pen) {
    if (!initialized_ || n_rows <= 0 || d_out == nullptr) {
        return;
    }
    const int V = model_->config().vocab_size;
    const int mb = max_logit_tokens_ > 0 ? max_logit_tokens_ : 1;
    const size_t scratch_needed =
        static_cast<size_t>(mb) * kArgmaxSplits * (sizeof(float) + sizeof(int));
    if (verify_argmax_scratch_sz_ < scratch_needed) {
        if (verify_argmax_scratch_)
            IMP_CUDA_CHECK_LOG(cudaFree(verify_argmax_scratch_));
        if (cudaMalloc(&verify_argmax_scratch_, scratch_needed) != cudaSuccess) {
            verify_argmax_scratch_ = nullptr;
            verify_argmax_scratch_sz_ = 0;
            return;
        }
        verify_argmax_scratch_sz_ = scratch_needed;
    }
    float* pvals = static_cast<float*>(verify_argmax_scratch_);
    int* pidxs = reinterpret_cast<int*>(pvals + static_cast<size_t>(mb) * kArgmaxSplits);

    const bool penalties = (rep_pen != 1.0f || freq_pen != 0.0f || pres_pen != 0.0f) &&
                           d_hist != nullptr && d_draft != nullptr;
    if (penalties) {
        if (verify_pen_counts_cap_ < V) {
            if (verify_pen_counts_)
                IMP_CUDA_CHECK_LOG(cudaFree(verify_pen_counts_));
            if (cudaMalloc(&verify_pen_counts_, static_cast<size_t>(V) * sizeof(int32_t)) !=
                cudaSuccess) {
                verify_pen_counts_ = nullptr;
                verify_pen_counts_cap_ = 0;
                return;
            }
            verify_pen_counts_cap_ = V;
        }
        verify_hist_count_kernel<<<(V + 255) / 256, 256, 0, stream>>>(d_hist, n_hist, V,
                                                                      verify_pen_counts_);
    }

    for_each_lm_head_batch_(n_rows, stream, [&](const Tensor& lg, int row0, int csz) {
        if (penalties) {
            dim3 pgrid((V + 255) / 256, csz);
            verify_penalties_kernel<<<pgrid, 256, 0, stream>>>(
                static_cast<float*>(lg.data), row0, V, verify_pen_counts_, d_draft, rep_pen,
                freq_pen, pres_pen);
        }
        dim3 grid(csz, kArgmaxSplits);
        rowwise_argmax_partial_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const float*>(lg.data), V, pvals, pidxs);
        rowwise_argmax_reduce_kernel<<<1, 32, 0, stream>>>(pvals, pidxs, csz, d_out + row0);
    });
}

double GraphExecutor::perplexity_nll(const int32_t* tokens, int n, cudaStream_t stream) {
    if (!initialized_ || n < 2) {
        IMP_LOG_ERROR("perplexity_nll: not initialized or n < 2 (n=%d)", n);
        return -1.0;
    }

    // Ensure the (single-chunk) prefill that populated hidden_ is complete.
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    int32_t* d_tokens = nullptr;
    double* d_nll = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tokens, static_cast<size_t>(n) * sizeof(int32_t)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_nll, static_cast<size_t>(n) * sizeof(double)));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_tokens, tokens, static_cast<size_t>(n) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_nll, 0, static_cast<size_t>(n) * sizeof(double), stream));

    perplexity_nll_partial(d_tokens, n, /*chunk_start=*/0, /*chunk_len=*/n, d_nll, stream);

    // Fixed-order host reduction over per-position NLLs (bit-reproducible).
    std::vector<double> h_nll_pos(static_cast<size_t>(n), 0.0);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_nll_pos.data(), d_nll, static_cast<size_t>(n) * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    cudaFree(d_tokens);
    cudaFree(d_nll);
    double h_nll = 0.0;
    for (int i = 0; i < n - 1; ++i)
        h_nll += h_nll_pos[i];

    double ppl = std::exp(h_nll / static_cast<double>(n - 1));
    IMP_LOG_INFO("perplexity_nll: n=%d  mean_nll=%.4f  PPL=%.4f", n, h_nll / (n - 1), ppl);
    return ppl;
}

}  // namespace imp

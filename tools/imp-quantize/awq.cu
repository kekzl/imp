#include "awq.h"

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

using namespace imp;

namespace imp::awq {

namespace {

constexpr int kThreads = 256;
// 21 points at 0.05 spacing. The objective is smooth in alpha and the whole
// sweep costs a few GEMM-sized passes, so a finer grid buys nothing.
constexpr int kAlphaSteps = 21;

__global__ void scale_cols_kernel(const half* __restrict__ src, half* __restrict__ dst,
                                  const float* __restrict__ s, int64_t rows, int64_t K) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * K;
    if (idx >= total)
        return;
    dst[idx] = __float2half(__half2float(src[idx]) * s[idx % K]);
}

// err += (a_j / s_j)^2 * (scaled_ij - dequant_ij)^2, reduced per block.
__global__ void weighted_sqerr_kernel(const half* __restrict__ scaled, const half* __restrict__ deq,
                                      const float* __restrict__ w_col, int64_t rows, int64_t K,
                                      double* __restrict__ partial) {
    __shared__ double sdata[kThreads];
    const int64_t total = rows * K;
    double acc = 0.0;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const double d = static_cast<double>(__half2float(scaled[idx])) -
                         static_cast<double>(__half2float(deq[idx]));
        acc += static_cast<double>(w_col[idx % K]) * d * d;
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off)
            sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sdata[0];
}

struct DeviceBuf {
    void* p = nullptr;
    ~DeviceBuf() {
        if (p)
            cudaFree(p);
    }
    bool alloc(size_t bytes) { return cudaMalloc(&p, bytes) == cudaSuccess; }
};

}  // namespace

bool search_group_scale(const std::vector<GroupMatrix>& mats, int64_t K, const std::vector<float>& act_mean,
                        SearchResult& out, std::string& err) {
    if (mats.empty() || K <= 0 || static_cast<int64_t>(act_mean.size()) != K) {
        err = "search_group_scale: empty group or activation/K mismatch";
        return false;
    }

    int64_t max_rows = 0;
    for (const auto& m : mats) {
        if (!m.data || m.N <= 0 || static_cast<int64_t>(m.data->size()) != m.N * K) {
            err = "search_group_scale: matrix shape does not match [N, K]";
            return false;
        }
        max_rows = std::max(max_rows, m.N);
    }

    // A channel that never fired carries no information; treat it as the group
    // floor rather than letting a zero drive its scale to zero or infinity.
    double amean = 0.0;
    float amax = 0.0f;
    for (float a : act_mean) {
        amean += a;
        amax = std::max(amax, a);
    }
    amean /= static_cast<double>(K);
    if (!(amean > 0.0) || !(amax > 0.0f)) {
        // No activation signal at all — nothing to protect, keep RTN.
        out.s.assign(static_cast<size_t>(K), 1.0f);
        out.alpha = 0.0f;
        out.err_rtn = out.err_best = 0.0;
        return true;
    }
    const float floor_a = static_cast<float>(amean) * 1e-4f;

    const size_t elems = static_cast<size_t>(max_rows) * static_cast<size_t>(K);
    DeviceBuf d_scaled, d_deq, d_s, d_w, d_partial;
    std::vector<DeviceBuf> d_src(mats.size());
    if (!d_scaled.alloc(elems * sizeof(half)) || !d_deq.alloc(elems * sizeof(half)) ||
        !d_s.alloc(static_cast<size_t>(K) * sizeof(float)) ||
        !d_w.alloc(static_cast<size_t>(K) * sizeof(float))) {
        err = "search_group_scale: cudaMalloc failed for scratch";
        return false;
    }
    for (size_t i = 0; i < mats.size(); i++) {
        const size_t bytes = static_cast<size_t>(mats[i].N) * static_cast<size_t>(K) * sizeof(half);
        if (!d_src[i].alloc(bytes) ||
            cudaMemcpy(d_src[i].p, mats[i].data->data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "search_group_scale: uploading group matrix failed";
            return false;
        }
    }

    const int reduce_blocks = 1024;
    if (!d_partial.alloc(static_cast<size_t>(reduce_blocks) * sizeof(double))) {
        err = "search_group_scale: cudaMalloc failed for reduction";
        return false;
    }
    std::vector<double> h_partial(reduce_blocks);

    std::vector<float> s(static_cast<size_t>(K)), w(static_cast<size_t>(K));
    std::vector<float> best_s(static_cast<size_t>(K), 1.0f);
    double best_err = 0.0, rtn_err = 0.0;
    float best_alpha = 0.0f;
    bool have_best = false;

    for (int step = 0; step < kAlphaSteps; step++) {
        const float alpha = static_cast<float>(step) / static_cast<float>(kAlphaSteps - 1);

        // s_j = (a_j / mean(a))^alpha, then normalised so the geometric span is
        // centred on 1 — a uniform rescale is free (the NVFP4 tensor scale
        // absorbs it exactly), but keeping s near 1 keeps the folded norm
        // weights inside their dtype's range.
        float smin = 3.4e38f, smax = 0.0f;
        for (int64_t j = 0; j < K; j++) {
            const float a = std::max(act_mean[static_cast<size_t>(j)], floor_a);
            const float v = (alpha == 0.0f) ? 1.0f : std::pow(a / static_cast<float>(amean), alpha);
            s[static_cast<size_t>(j)] = v;
            smin = std::min(smin, v);
            smax = std::max(smax, v);
        }
        if (alpha > 0.0f) {
            const float norm = std::sqrt(smin * smax);
            if (norm > 0.0f && std::isfinite(norm))
                for (auto& v : s)
                    v /= norm;
        }
        for (int64_t j = 0; j < K; j++) {
            const float a = act_mean[static_cast<size_t>(j)];
            const float r = a / s[static_cast<size_t>(j)];
            w[static_cast<size_t>(j)] = r * r;
        }
        if (cudaMemcpy(d_s.p, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_w.p, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "search_group_scale: scale upload failed";
            return false;
        }

        double err_alpha = 0.0;
        for (size_t i = 0; i < mats.size(); i++) {
            const int64_t rows = mats[i].N;
            const int64_t total = rows * K;
            const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
            scale_cols_kernel<<<blocks, kThreads>>>(static_cast<const half*>(d_src[i].p),
                                                    static_cast<half*>(d_scaled.p),
                                                    static_cast<const float*>(d_s.p), rows, K);

            int64_t shape[2] = {rows, K};
            Tensor scaled(d_scaled.p, QType::F16, 2, shape, /*on_device=*/true);
            NvFP4QuantResult q;
            quantize_fp16_to_nvfp4(scaled, q);
            dequantize_nvfp4_to_fp16(q, d_deq.p);

            weighted_sqerr_kernel<<<reduce_blocks, kThreads>>>(static_cast<const half*>(d_scaled.p),
                                                               static_cast<const half*>(d_deq.p),
                                                               static_cast<const float*>(d_w.p), rows, K,
                                                               static_cast<double*>(d_partial.p));
            const bool ok = cudaDeviceSynchronize() == cudaSuccess &&
                            cudaMemcpy(h_partial.data(), d_partial.p, h_partial.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost) == cudaSuccess;
            free_nvfp4_result(q);
            if (!ok) {
                err = "search_group_scale: quantize/reduce failed";
                return false;
            }
            for (double v : h_partial)
                err_alpha += v;
        }

        if (step == 0)
            rtn_err = err_alpha;
        // A large alpha can push scaled weights past FP16 range; the objective
        // then reads inf/NaN. Dropping those candidates outright is clearer
        // than relying on NaN comparisons to lose silently.
        if (!std::isfinite(err_alpha))
            continue;
        if (!have_best || err_alpha < best_err) {
            have_best = true;
            best_err = err_alpha;
            best_alpha = alpha;
            best_s = s;
        }
    }

    out.s = std::move(best_s);
    out.alpha = best_alpha;
    out.err_rtn = rtn_err;
    out.err_best = best_err;
    return true;
}

}  // namespace imp::awq

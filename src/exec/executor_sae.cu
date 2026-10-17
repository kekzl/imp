// Sparse autoencoder hook (roadmap row 116, exec/sae.h): after layer sae_->layer the armed rows
// of hidden_ are encoded in tiles of kSaeTile rows; readout rows also get the reconstruction
// error of the unclamped residual, clamped features are written back into the residual.

#include "exec/executor.h"
#include "exec/sae.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace imp {

namespace {

constexpr int kSaeTile = 8;
constexpr int kThreads = 256;

// x[t, k] = residual row rows[base + t], FP32 shadow when the model keeps one.
__global__ void sae_gather_kernel(const half* __restrict__ h16, const float* __restrict__ h32,
                                  const int* __restrict__ rows, int base, int t_n, int d,
                                  float* __restrict__ x) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (k >= d || t >= t_n)
        return;
    const int64_t src = static_cast<int64_t>(rows[base + t]) * d + k;
    x[static_cast<int64_t>(t) * d + k] = h32 ? h32[src] : __half2float(h16[src]);
}

// a[t, j] = pre * (pre > threshold[j]), pre = b_enc[j] + sum_k x[t, k] w_enc[k, j]. One thread
// per feature: w_enc reads coalesce along j, x reads are uniform across the block.
__global__ void sae_encode_kernel(const float* __restrict__ x, const float* __restrict__ w_enc,
                                  const float* __restrict__ b_enc, const float* __restrict__ thr, int t_n,
                                  int d, int width, float* __restrict__ acts) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width)
        return;
    float acc[kSaeTile];
#pragma unroll
    for (int t = 0; t < kSaeTile; t++)
        acc[t] = b_enc[j];
    for (int k = 0; k < d; k++) {
        const float w = w_enc[static_cast<int64_t>(k) * width + j];
#pragma unroll
        for (int t = 0; t < kSaeTile; t++)
            if (t < t_n)
                acc[t] = fmaf(x[static_cast<int64_t>(t) * d + k], w, acc[t]);
    }
    const float th = thr[j];
#pragma unroll
    for (int t = 0; t < kSaeTile; t++)
        if (t < t_n)
            acts[static_cast<int64_t>(t) * width + j] = acc[t] > th ? acc[t] : 0.0f;
}

// err[slot] += (||x - x_hat||^2, ||x||^2) for readout rows, x_hat = b_dec + a w_dec. Zero
// activations are skipped (uniform branch: every thread reads the same a_j).
__global__ void sae_err_kernel(const float* __restrict__ x, const float* __restrict__ acts,
                               const float* __restrict__ w_dec, const float* __restrict__ b_dec,
                               const int* __restrict__ slot, int base, int d, int width,
                               float* __restrict__ err) {
    const int t = blockIdx.y;
    const int s = slot[base + t];
    if (s < 0)
        return;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    float e2 = 0.0f, n2 = 0.0f;
    if (k < d) {
        const float* a = acts + static_cast<int64_t>(t) * width;
        float xh = b_dec[k];
        for (int j = 0; j < width; j++)
            if (a[j] != 0.0f)
                xh = fmaf(a[j], w_dec[static_cast<int64_t>(j) * d + k], xh);
        const float xv = x[static_cast<int64_t>(t) * d + k];
        e2 = (xv - xh) * (xv - xh);
        n2 = xv * xv;
    }
    __shared__ float red[2][kThreads / 32];
    for (int o = 16; o > 0; o >>= 1) {
        e2 += __shfl_down_sync(0xffffffffu, e2, o);
        n2 += __shfl_down_sync(0xffffffffu, n2, o);
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x >> 5] = e2;
        red[1][threadIdx.x >> 5] = n2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float se = 0.0f, sn = 0.0f;
        for (int w = 0; w < kThreads / 32; w++) {
            se += red[0][w];
            sn += red[1][w];
        }
        atomicAdd(err + 2 * static_cast<int64_t>(s), se);
        atomicAdd(err + 2 * static_cast<int64_t>(s) + 1, sn);
    }
}

// Row t: for each clamp (j, c), residual += (c - a_j) w_dec[j] and a_j = c. One block per row.
__global__ void sae_clamp_kernel(half* __restrict__ h16, float* __restrict__ h32, float* __restrict__ acts,
                                 const float* __restrict__ w_dec, const int* __restrict__ rows,
                                 const int* __restrict__ off, const int* __restrict__ ids,
                                 const float* __restrict__ vals, int base, int d, int width) {
    const int t = blockIdx.x;
    const int g = base + t;
    float* a = acts + static_cast<int64_t>(t) * width;
    const int64_t row = static_cast<int64_t>(rows[g]) * d;
    for (int c = off[g]; c < off[g + 1]; c++) {
        const int j = ids[c];
        const float delta = vals[c] - a[j];
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            const float add = delta * w_dec[static_cast<int64_t>(j) * d + k];
            h16[row + k] = __float2half(__half2float(h16[row + k]) + add);
            if (h32)
                h32[row + k] += add;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            a[j] = vals[c];
        __syncthreads();
    }
}

}  // namespace

void GraphExecutor::set_sae_rows(std::vector<SaeRowSpec> rows, cudaStream_t stream) {
    sae_rows_ = std::move(rows);
    sae_n_readout_ = 0;
    if (sae_rows_.empty())
        return;
    if (sae_ == nullptr || vram_alloc_ == nullptr) {
        sae_rows_.clear();
        return;
    }
    // Host mirror of the device index arrays: rows | slot | clamp offsets, then clamp ids.
    const int r_n = static_cast<int>(sae_rows_.size());
    auto& hi = sae_h_int_;
    auto& hf = sae_h_vals_;
    hi.assign(3 * static_cast<size_t>(r_n) + 1, 0);
    hf.clear();
    std::vector<int32_t> ids;
    for (int r = 0; r < r_n; r++) {
        hi[r] = sae_rows_[r].row;
        hi[r_n + r] = sae_rows_[r].readout ? sae_n_readout_++ : -1;
        hi[2 * r_n + r] = static_cast<int>(ids.size());
        for (const auto& [j, v] : sae_rows_[r].clamp)
            if (j >= 0 && j < sae_->width) {
                ids.push_back(j);
                hf.push_back(v);
            }
    }
    hi[3 * static_cast<size_t>(r_n)] = static_cast<int>(ids.size());
    hi.insert(hi.end(), ids.begin(), ids.end());
    auto grow = [&](auto& buf, size_t count, const char* tag) {
        if (buf.count() < count)
            buf = std::remove_reference_t<decltype(buf)>(*vram_alloc_, std::max<size_t>(count, 1), tag);
        return static_cast<bool>(buf);
    };
    const size_t d = static_cast<size_t>(sae_->d_model), w = static_cast<size_t>(sae_->width);
    const bool ok = grow(sae_x_, kSaeTile * d, "sae_x") && grow(sae_acts_, kSaeTile * w, "sae_acts") &&
                    grow(sae_idx_, hi.size(), "sae_idx") &&
                    grow(sae_vals_, std::max<size_t>(hf.size(), 1), "sae_vals") &&
                    grow(sae_out_, std::max(sae_n_readout_, 1) * (w + 2), "sae_out");
    if (!ok) {
        IMP_LOG_ERROR("sae: scratch for %d rows not allocatable, SAE off for this step", r_n);
        sae_rows_.clear();
        return;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(sae_idx_.get(), hi.data(), hi.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    if (!hf.empty())
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(sae_vals_.get(), hf.data(), hf.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    if (sae_n_readout_ > 0)
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(sae_out_.get(), 0,
                                           static_cast<size_t>(sae_n_readout_) * (w + 2) * sizeof(float),
                                           stream));
}

void GraphExecutor::sae_apply_(int n, cudaStream_t stream) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone)
        return;  // the engine runs SAE steps eager; a capture must not bake the edits
    const int d = sae_->d_model, w = sae_->width;
    const int r_n = static_cast<int>(sae_rows_.size());
    half* h16 = static_cast<half*>(view_tokens(hidden_, n).data);
    // Post-norm models (Gemma-3, Gemma-4) carry the residual in fp32_hidden_; hidden_ is its FP16 copy.
    float* h32 = fp32_accum_buf_ ? static_cast<float*>(view_tokens(fp32_hidden_, n).data) : nullptr;
    const int* idx = sae_idx_.get();
    const int* rows = idx;
    const int* slot = idx + r_n;
    const int* off = idx + 2 * static_cast<ptrdiff_t>(r_n);
    const int* ids = idx + 3 * static_cast<ptrdiff_t>(r_n) + 1;
    float* out_err = sae_out_.get() + static_cast<size_t>(sae_n_readout_) * w;
    for (int base = 0; base < r_n; base += kSaeTile) {
        const int t_n = std::min(kSaeTile, r_n - base);
        bool any_ro = false, any_clamp = false;
        for (int t = 0; t < t_n; t++) {
            any_ro = any_ro || sae_rows_[base + t].readout;
            any_clamp = any_clamp || !sae_rows_[base + t].clamp.empty();
        }
        const dim3 gd((d + kThreads - 1) / kThreads, t_n);
        sae_gather_kernel<<<gd, kThreads, 0, stream>>>(h16, h32, rows, base, t_n, d, sae_x_.get());
        IMP_CUDA_CHECK_LAUNCH();
        sae_encode_kernel<<<(w + kThreads - 1) / kThreads, kThreads, 0, stream>>>(sae_x_.get(),
                                                                                  sae_->w_enc.get(),
                                                                                  sae_->b_enc.get(),
                                                                                  sae_->threshold.get(), t_n,
                                                                                  d, w, sae_acts_.get());
        IMP_CUDA_CHECK_LAUNCH();
        if (any_ro) {
            sae_err_kernel<<<gd, kThreads, 0, stream>>>(sae_x_.get(), sae_acts_.get(), sae_->w_dec.get(),
                                                        sae_->b_dec.get(), slot, base, d, w, out_err);
            IMP_CUDA_CHECK_LAUNCH();
        }
        if (any_clamp) {
            sae_clamp_kernel<<<t_n, kThreads, 0, stream>>>(h16, h32, sae_acts_.get(), sae_->w_dec.get(), rows,
                                                           off, ids, sae_vals_.get(), base, d, w);
            IMP_CUDA_CHECK_LAUNCH();
        }
        for (int t = 0; t < t_n; t++) {
            const int s = sae_h_int_[r_n + base + t];
            if (s >= 0)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(sae_out_.get() + static_cast<size_t>(s) * w,
                                                   sae_acts_.get() + static_cast<size_t>(t) * w,
                                                   w * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }
    sae_ran_ = true;
}

bool GraphExecutor::sae_readout(SaeReadout& out, cudaStream_t stream) {
    out = {};
    const bool ran = sae_ran_;
    sae_ran_ = false;
    if (!ran || sae_n_readout_ <= 0)
        return false;
    const size_t w = static_cast<size_t>(sae_->width);
    std::vector<float> host(static_cast<size_t>(sae_n_readout_) * (w + 2));
    if (cudaMemcpyAsync(host.data(), sae_out_.get(), host.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return false;
    out.rows = sae_n_readout_;
    out.width = sae_->width;
    out.acts.assign(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(sae_n_readout_ * w));
    for (int r = 0; r < sae_n_readout_; r++) {
        const size_t e = static_cast<size_t>(sae_n_readout_) * w + 2 * static_cast<size_t>(r);
        const float e2 = host[e], n2 = host[e + 1];
        out.rel_err.push_back(n2 > 0.0f ? std::sqrt(e2 / n2) : 0.0f);
    }
    return true;
}

}  // namespace imp

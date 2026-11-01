// GPTQ for NVFP4 (roadmap row 101, gptq.h). Device layout: W row-major [N, K] FP32; U = the upper
// Cholesky factor of H^-1 (upper_cholesky_of_inverse), column-major, so U(i, j) = U[j * K + i].

#include "gptq.h"

#include "compute/gemm_lt_ex.h"
#include "core/logging.h"
#include "quant/nvfp4_pack.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace imp::gptq {

namespace {

constexpr int kBlock = 128;  // columns per lazy update; a multiple of the 16-column micro-block
constexpr int kThreads = 256;

__constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float e2m1_value(uint8_t code) {
    const float v = kE2M1[code & 7];
    return (code & 8) ? -v : v;
}

// One code per element: the hardware E2M1 rounding the export kernel uses (low nibble of the pair).
__device__ __forceinline__ uint8_t e2m1_code(float scaled) { return nvfp4_pack_pair_hw(scaled, 0.0f) & 0xF; }

// Micro-scale of group [c0, c0 + 16) per row from the current weights; sc[r] = scale * tensor_scale.
__global__ void group_scale_kernel(const float* __restrict__ w, int64_t N, int64_t K, int64_t c0,
                                   float tensor_scale, uint8_t* __restrict__ micro, float* __restrict__ sc) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= N)
        return;
    float amax = 0.0f;
    for (int j = 0; j < kNvfp4MicroBlockSize; j++)
        amax = fmaxf(amax, fabsf(w[r * K + c0 + j]));
    uint8_t fp8 = 0;
    const float s = nvfp4_encode_micro_scale(amax, tensor_scale, &fp8);
    micro[r * (K / kNvfp4MicroBlockSize) + c0 / kNvfp4MicroBlockSize] = fp8;
    sc[r] = s * tensor_scale;
}

// Column i: quantize, record the scaled error, update the rest of the block in place. kAsym (GPTAQ):
// also adds x * P(i, j) with x the pre-quantization value and records the dequantized value in qb.
// Separate instantiations: a runtime branch moved the GPTQ codes (Qwen3-0.6B d10 PPL 27.2857 -> 27.1658).
template <bool kAsym>
__global__ void column_kernel(float* __restrict__ w, int64_t N, int64_t K, int64_t i, int64_t b0, int64_t b1,
                              const float* __restrict__ u, const float* __restrict__ sc,
                              uint8_t* __restrict__ codes, float* __restrict__ err,
                              const float* __restrict__ p, float* __restrict__ qb) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= N)
        return;
    float* row = w + r * K;
    const float x = row[i];
    const uint8_t code = e2m1_code(x / sc[r]);
    codes[r * K + i] = code;
    const float e = (x - e2m1_value(code) * sc[r]) / u[i * K + i];
    err[r * kBlock + (i - b0)] = e;
    if constexpr (kAsym) {
        qb[r * kBlock + (i - b0)] = e2m1_value(code) * sc[r];
        for (int64_t j = i + 1; j < b1; j++)
            row[j] -= e * u[j * K + i] - x * p[j * K + i];
    } else {
        for (int64_t j = i + 1; j < b1; j++)
            row[j] -= e * u[j * K + i];
    }
}

// Column-major K x K: keep alpha * A(i, j) for j > i (strictly upper), zero the rest.
__global__ void triu1_scale_kernel(float* __restrict__ a, int64_t K, float alpha) {
    const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= K * K)
        return;
    a[p] = (p / K > p % K) ? alpha * a[p] : 0.0f;
}

// Round-to-nearest codes of the original weights (the loss baseline), same scale rule.
__global__ void rtn_dequant_kernel(const float* __restrict__ w, int64_t N, int64_t K, float tensor_scale,
                                   float* __restrict__ q) {
    const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t groups = K / kNvfp4MicroBlockSize;
    if (g >= N * groups)
        return;
    const float* src = w + g * kNvfp4MicroBlockSize;
    float amax = 0.0f;
    for (int j = 0; j < kNvfp4MicroBlockSize; j++)
        amax = fmaxf(amax, fabsf(src[j]));
    uint8_t fp8 = 0;
    const float s = nvfp4_encode_micro_scale(amax, tensor_scale, &fp8) * tensor_scale;
    for (int j = 0; j < kNvfp4MicroBlockSize; j++)
        q[g * kNvfp4MicroBlockSize + j] = e2m1_value(e2m1_code(src[j] / s)) * s;
}

// q = dequantized GPTQ codes; packed = low nibble even column.
__global__ void gptq_dequant_pack_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ micro,
                                         int64_t N, int64_t K, float tensor_scale, float* __restrict__ q,
                                         uint8_t* __restrict__ packed) {
    const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N * K / 2)
        return;
    const int64_t r = (2 * p) / K, c = (2 * p) % K;
    const float s = fp8_e4m3_to_float(micro[r * (K / kNvfp4MicroBlockSize) + c / kNvfp4MicroBlockSize]);
    const float scale = (s == 0.0f ? 1.0f / 512.0f : s) * tensor_scale;
    const uint8_t c0 = codes[r * K + c], c1 = codes[r * K + c + 1];
    q[r * K + c] = e2m1_value(c0) * scale;
    q[r * K + c + 1] = e2m1_value(c1) * scale;
    packed[p] = static_cast<uint8_t>(c0 | (c1 << 4));
}

// a = w0 - a (the quantization error), elementwise.
__global__ void residual_kernel(const float* __restrict__ w0, float* __restrict__ a, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = w0[i] - a[i];
}

// A -> J A J (J = row reversal): A(i, j) <-> A(K-1-i, K-1-j) is a[p] <-> a[K*K-1-p].
__global__ void reverse_kernel(float* __restrict__ a, int64_t kk) {
    const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= kk / 2)
        return;
    const float t = a[p];
    a[p] = a[kk - 1 - p];
    a[kk - 1 - p] = t;
}

constexpr int kDiag = 128;  // diagonal block of the blocked Cholesky / inverse, in shared memory
constexpr int kDiagThreads = 1024;

// Unblocked lower Cholesky of the b x b diagonal block at (j0, j0); info = first bad pivot, 1-based.
__global__ void __launch_bounds__(kDiagThreads) potf2_kernel(float* __restrict__ a, int64_t K, int64_t j0,
                                                             int b, int* __restrict__ info) {
    extern __shared__ float s[];  // s[c * b + r] = A(j0 + r, j0 + c)
    for (int e = threadIdx.x; e < b * b; e += blockDim.x)
        s[e] = a[(j0 + e / b) * K + j0 + e % b];
    for (int c = 0; c < b; c++) {
        __syncthreads();
        const float d = s[c * b + c];
        if (!(d > 0.0f) || !isfinite(d)) {
            if (threadIdx.x == 0)
                atomicCAS(info, 0, static_cast<int>(j0 + c + 1));
            return;
        }
        const float piv = sqrtf(d);
        __syncthreads();
        for (int r = c + 1 + threadIdx.x; r < b; r += blockDim.x)
            s[c * b + r] /= piv;
        if (threadIdx.x == 0)
            s[c * b + c] = piv;
        __syncthreads();
        const int m = b - c - 1;
        for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
            const int r = c + 1 + e % m, cc = c + 1 + e / m;
            if (cc <= r)
                s[cc * b + r] -= s[c * b + r] * s[c * b + cc];
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < b * b; e += blockDim.x)
        if (e % b >= e / b)
            a[(j0 + e / b) * K + j0 + e % b] = s[e];
}

// Unblocked inverse of the lower triangular b x b diagonal block at (j0, j0), in place (LAPACK strti2);
// the strict upper triangle is written 0.
__global__ void __launch_bounds__(kDiagThreads) trti2_kernel(float* __restrict__ a, int64_t K, int64_t j0,
                                                             int b) {
    extern __shared__ float s[];  // b * b block, then b scratch
    float* y = s + b * b;
    for (int e = threadIdx.x; e < b * b; e += blockDim.x)
        s[e] = a[(j0 + e / b) * K + j0 + e % b];
    for (int c = b - 1; c >= 0; c--) {
        __syncthreads();
        const float inv = 1.0f / s[c * b + c];
        // x = A(c+1:, c); x <- -inv * T x with T = A(c+1:, c+1:) already inverted.
        for (int r = c + 1 + threadIdx.x; r < b; r += blockDim.x) {
            float acc = 0.0f;
            for (int k = c + 1; k <= r; k++)
                acc += s[k * b + r] * s[c * b + k];
            y[r] = -inv * acc;
        }
        __syncthreads();
        for (int r = c + 1 + threadIdx.x; r < b; r += blockDim.x)
            s[c * b + r] = y[r];
        if (threadIdx.x == 0)
            s[c * b + c] = inv;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < b * b; e += blockDim.x)
        a[(j0 + e / b) * K + j0 + e % b] = e % b >= e / b ? s[e] : 0.0f;
}

unsigned grid(int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

template <typename T>
struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) {
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
            p = nullptr;
    }
    ~Dev() {
        if (p)
            IMP_CUDA_CHECK_LOG(cudaFree(p));
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

// Lower inverse Cholesky factor X = L^-1 (A = L L^T) by recursive halving, GEMMs only:
//   X11 = f(A11), L21 = A21 X11^T, A22 -= L21 L21^T, X22 = f(A22), X21 = -X22 L21 X11.
// Block edges on multiples of kDiag; the strict upper triangle of every finished block is zero.
struct InverseCholesky {
    float* a;
    int64_t K;
    float* tmp;  // (K / 2 + kDiag)^2
    int* info;
    float* at(int64_t i, int64_t j) const { return a + j * K + i; }
    static int64_t split(int64_t n) { return (n / 2 + kDiag - 1) / kDiag * kDiag; }
    bool gemm(cublasOperation_t ta, cublasOperation_t tb, int64_t m, int64_t n, int64_t k, float alpha,
              const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) const {
        return lt_gemm_ex(ta, tb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), &alpha, A,
                          CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                          nullptr) == CUBLAS_STATUS_SUCCESS;
    }

    bool run(int64_t j0, int64_t n) const {
        if (n <= kDiag) {
            potf2_kernel<<<1, kDiagThreads, sizeof(float) * kDiag * kDiag>>>(a, K, j0, static_cast<int>(n),
                                                                             info);
            trti2_kernel<<<1, kDiagThreads, sizeof(float) * (kDiag * kDiag + kDiag)>>>(a, K, j0,
                                                                                       static_cast<int>(n));
            return true;
        }
        const int64_t n1 = split(n), n2 = n - n1;
        const int k = static_cast<int>(K), lt = static_cast<int>(n2);
        float* a21 = at(j0 + n1, j0);
        const float* x11 = at(j0, j0);
        const float* x22 = at(j0 + n1, j0 + n1);
        return run(j0, n1) &&
               // tmp = L21 = A21 X11^T, back into A21
               gemm(CUBLAS_OP_N, CUBLAS_OP_T, n2, n1, n1, 1.0f, a21, k, x11, k, 0.0f, tmp, lt) &&
               cudaMemcpy2DAsync(a21, K * sizeof(float), tmp, n2 * sizeof(float), n2 * sizeof(float), n1,
                                 cudaMemcpyDeviceToDevice) == cudaSuccess &&
               gemm(CUBLAS_OP_N, CUBLAS_OP_T, n2, n2, n1, -1.0f, a21, k, a21, k, 1.0f, at(j0 + n1, j0 + n1),
                    k) &&
               run(j0 + n1, n2) &&
               // tmp = X22 L21, A21 = -tmp X11
               gemm(CUBLAS_OP_N, CUBLAS_OP_N, n2, n1, n2, 1.0f, x22, k, a21, k, 0.0f, tmp, lt) &&
               gemm(CUBLAS_OP_N, CUBLAS_OP_N, n2, n1, n1, -1.0f, tmp, lt, x11, k, 0.0f, a21, k) &&
               cudaMemset2DAsync(at(j0, j0 + n1), K * sizeof(float), 0, n1 * sizeof(float), n2) ==
                   cudaSuccess;
    }
};

// H = R R^T with R upper is the lower Cholesky of J H J, mirrored; H^-1 = R^-T R^-1, so U = R^-1:
// the lower inverse Cholesky factor of J H J, J back. Upper triangle = U on return, lower = 0.
std::expected<void, std::string> upper_cholesky_of_inverse_dev(float* a, int64_t K) {
    const int64_t kk = K * K;
    Dev<int> dinfo(1);
    const int64_t h = K / 2 + kDiag;  // every level: n1, n2 <= K / 2 + kDiag
    Dev<float> tmp(static_cast<size_t>(h * h));
    if (!dinfo.p || !tmp.p || cudaMemset(dinfo.p, 0, sizeof(int)) != cudaSuccess)
        return std::unexpected("Cholesky scratch allocation failed");
    if (cudaFuncSetAttribute(potf2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(float) * kDiag * kDiag)) != cudaSuccess ||
        cudaFuncSetAttribute(trti2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(float) * (kDiag * kDiag + kDiag))) != cudaSuccess)
        return std::unexpected("shared memory opt-in failed");
    reverse_kernel<<<grid(kk / 2), kThreads>>>(a, kk);
    if (!InverseCholesky{a, K, tmp.p, dinfo.p}.run(0, K))
        return std::unexpected("inverse Cholesky panel update failed");
    int info = 0;
    if (cudaMemcpy(&info, dinfo.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || info != 0)
        return std::unexpected("Cholesky of H failed (info " + std::to_string(info) + ")");
    reverse_kernel<<<grid(kk / 2), kThreads>>>(a, kk);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
        return std::unexpected(std::string("Cholesky kernels failed: ") + cudaGetErrorString(e));
    return {};
}

// sum over rows of e_r H e_r^T for the error matrix e [N, K] (row-major), with eh scratch [N, K]
// and a device scalar; negative when a GEMM fails.
double quad_loss(const float* e, const float* H, float* eh, float* d_dot, int64_t N, int64_t K) {
    const float one = 1.0f, zero = 0.0f;
    const int k = static_cast<int>(K), nk = static_cast<int>(N * K);
    // Column-major view: E^T [K, N]; (E H)^T = H E^T (H symmetric); the dot as a 1 x 1 GEMM.
    float dot = 0.0f;
    const bool ok = N * K <= INT32_MAX &&
                    lt_gemm_ex(CUBLAS_OP_N, CUBLAS_OP_N, k, static_cast<int>(N), k, &one, H, CUDA_R_32F, k, e,
                               CUDA_R_32F, k, &zero, eh, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                               nullptr) == CUBLAS_STATUS_SUCCESS &&
                    lt_gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, nk, &one, e, CUDA_R_32F, nk, eh, CUDA_R_32F,
                               nk, &zero, d_dot, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F,
                               nullptr) == CUBLAS_STATUS_SUCCESS &&
                    cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    return ok ? static_cast<double>(dot) : -1.0;
}

}  // namespace

std::expected<void, std::string> upper_cholesky_of_inverse(float* d_a, int64_t K) {
    return upper_cholesky_of_inverse_dev(d_a, K);
}

int layer_of(const std::string& name) {
    const size_t p = name.find(".layers.");
    if (p == std::string::npos)
        return -1;
    size_t i = p + 8;
    int v = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        v = v * 10 + (name[i++] - '0');
        any = true;
    }
    return any && i < name.size() && name[i] == '.' ? v : -1;
}

std::expected<Result, std::string> quantize_nvfp4(const uint16_t* w_fp16, int64_t N, int64_t K,
                                                  std::vector<float> H, float tensor_scale, float damp,
                                                  std::vector<float> dxx, float alpha) {
    if (K % kNvfp4MicroBlockSize != 0)
        return std::unexpected("K = " + std::to_string(K) + " is not a multiple of 16");
    const size_t nk = static_cast<size_t>(N) * static_cast<size_t>(K), kk = static_cast<size_t>(K) * K;
    if (H.size() != kk)
        return std::unexpected("Hessian is not K x K");
    const bool aq = !dxx.empty() && alpha != 0.0f;
    if (aq && dxx.size() != kk)
        return std::unexpected("GPTAQ cross term is not K x K");

    // Host: FP32 weights, dead inputs zeroed, H damped.
    std::vector<float> w(nk);
    for (size_t i = 0; i < nk; i++)
        w[i] = __half2float(__ushort_as_half(w_fp16[i]));
    double mean_diag = 0.0;
    for (int64_t i = 0; i < K; i++)
        mean_diag += H[static_cast<size_t>(i) * K + i];
    mean_diag /= static_cast<double>(K);
    std::vector<float> H0 = H;
    for (int64_t i = 0; i < K; i++) {
        float& d = H[static_cast<size_t>(i) * K + i];
        if (d <= 0.0f) {  // an input channel that never fired: nothing to fit
            d = 1.0f;
            for (int64_t r = 0; r < N; r++)
                w[static_cast<size_t>(r) * K + i] = 0.0f;
            if (aq)  // dXX(:, i) = 0, as the reference code
                std::fill_n(dxx.begin() + static_cast<int64_t>(i * K), K, 0.0f);
        }
        d += static_cast<float>(damp * mean_diag);
    }

    Dev<float> dw(nk), dw0(nk), du(kk), dh0(kk), dq(nk), dsc(static_cast<size_t>(N)),
        derr(static_cast<size_t>(N) * kBlock);
    Dev<uint8_t> dcodes(nk), dmicro(nk / 16), dpacked(nk / 2);
    if (!dw.p || !dw0.p || !du.p || !dh0.p || !dq.p || !dsc.p || !derr.p || !dcodes.p || !dmicro.p ||
        !dpacked.p)
        return std::unexpected("cudaMalloc failed for N = " + std::to_string(N) +
                               ", K = " + std::to_string(K));
    std::vector<float> w0(nk);
    for (size_t i = 0; i < nk; i++)
        w0[i] = __half2float(__ushort_as_half(w_fp16[i]));
    if (cudaMemcpy(dw.p, w.data(), nk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dw0.p, w0.data(), nk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(du.p, H.data(), kk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dh0.p, H0.data(), kk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return std::unexpected("H2D copy failed");

    const int k = static_cast<int>(K);
    if (auto u = upper_cholesky_of_inverse_dev(du.p, K); !u)
        return std::unexpected(u.error());

    // GPTAQ: P = alpha * triu1(dXX U^T) U into dp (column-major), dt scratch; qb = dequantized block.
    Dev<float> dp(aq ? kk : 1), dt(aq ? kk : 1), dqb(aq ? static_cast<size_t>(N) * kBlock : 1);
    if (aq) {
        const float one = 1.0f, zero = 0.0f;
        if (!dp.p || !dt.p || !dqb.p)
            return std::unexpected("cudaMalloc failed for the GPTAQ term, K = " + std::to_string(K));
        if (cudaMemcpy(dt.p, dxx.data(), kk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return std::unexpected("H2D copy failed");
        const bool ok = lt_gemm_ex(CUBLAS_OP_N, CUBLAS_OP_T, k, k, k, &one, dt.p, CUDA_R_32F, k, du.p,
                                   CUDA_R_32F, k, &zero, dp.p, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                                   nullptr) == CUBLAS_STATUS_SUCCESS;
        if (ok)
            triu1_scale_kernel<<<grid(static_cast<int64_t>(kk)), kThreads>>>(dp.p, K, alpha);
        if (!ok ||
            lt_gemm_ex(CUBLAS_OP_N, CUBLAS_OP_N, k, k, k, &one, dp.p, CUDA_R_32F, k, du.p, CUDA_R_32F, k,
                       &zero, dt.p, CUDA_R_32F, k, CUBLAS_COMPUTE_32F, nullptr) != CUBLAS_STATUS_SUCCESS)
            return std::unexpected("GPTAQ P GEMM failed");
    }
    const float* dP = aq ? dt.p : nullptr;

    // Blocks of kBlock columns: per-column quantize + in-block update, then one GEMM for the tail.
    for (int64_t b0 = 0; b0 < K; b0 += kBlock) {
        const int64_t b1 = std::min<int64_t>(b0 + kBlock, K);
        for (int64_t i = b0; i < b1; i++) {
            if (i % kNvfp4MicroBlockSize == 0)
                group_scale_kernel<<<grid(N), kThreads>>>(dw.p, N, K, i, tensor_scale, dmicro.p, dsc.p);
            if (aq)
                column_kernel<true>
                    <<<grid(N), kThreads>>>(dw.p, N, K, i, b0, b1, du.p, dsc.p, dcodes.p, derr.p, dP, dqb.p);
            else
                column_kernel<false><<<grid(N), kThreads>>>(dw.p, N, K, i, b0, b1, du.p, dsc.p, dcodes.p,
                                                            derr.p, nullptr, nullptr);
        }
        if (b1 < K) {
            // W[:, b1:] -= Err [N, B] * U(b0:b1, b1:). Column-major: C [K-b1, N] (ld K) -= A^T E with
            // A = U(b0:b1, b1:) as [B, K-b1] (ld K) and E = Err^T [B, N] (ld kBlock).
            const float minus = -1.0f, one = 1.0f;
            if (lt_gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(K - b1), static_cast<int>(N),
                           static_cast<int>(b1 - b0), &minus, du.p + b1 * K + b0, CUDA_R_32F, k, derr.p,
                           CUDA_R_32F, kBlock, &one, dw.p + b1, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                           nullptr) != CUBLAS_STATUS_SUCCESS)
                return std::unexpected("tail update GEMM failed");
            // GPTAQ: W[:, b1:] += Q [N, B] * P(b0:b1, b1:), Q = the block's dequantized values.
            if (aq && lt_gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(K - b1), static_cast<int>(N),
                                 static_cast<int>(b1 - b0), &one, dP + b1 * K + b0, CUDA_R_32F, k, dqb.p,
                                 CUDA_R_32F, kBlock, &one, dw.p + b1, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                                 nullptr) != CUBLAS_STATUS_SUCCESS)
                return std::unexpected("GPTAQ tail update GEMM failed");
        }
    }

    Result out;
    out.packed.resize(nk / 2);
    out.micro.resize(nk / 16);
    // Losses against the undamped H: RTN of the original weights, then the GPTQ codes.
    rtn_dequant_kernel<<<grid(static_cast<int64_t>(nk / 16)), kThreads>>>(dw0.p, N, K, tensor_scale, dq.p);
    residual_kernel<<<grid(static_cast<int64_t>(nk)), kThreads>>>(dw0.p, dq.p, static_cast<int64_t>(nk));
    out.loss_rtn = quad_loss(dq.p, dh0.p, dw.p, dsc.p, N, K);
    gptq_dequant_pack_kernel<<<grid(static_cast<int64_t>(nk / 2)), kThreads>>>(dcodes.p, dmicro.p, N, K,
                                                                               tensor_scale, dq.p, dpacked.p);
    if (cudaMemcpy(out.packed.data(), dpacked.p, nk / 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.micro.data(), dmicro.p, nk / 16, cudaMemcpyDeviceToHost) != cudaSuccess)
        return std::unexpected("D2H copy failed");
    residual_kernel<<<grid(static_cast<int64_t>(nk)), kThreads>>>(dw0.p, dq.p, static_cast<int64_t>(nk));
    out.loss_gptq = quad_loss(dq.p, dh0.p, dw.p, dsc.p, N, K);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
        return std::unexpected(std::string("GPTQ kernels failed: ") + cudaGetErrorString(e));
    return out;
}

}  // namespace imp::gptq

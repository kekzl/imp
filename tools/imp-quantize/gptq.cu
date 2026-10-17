// GPTQ for NVFP4 (roadmap row 101, gptq.h). Device layout: W row-major [N, K] FP32; U = the upper
// Cholesky factor of H^-1 as cuSOLVER leaves it, column-major, so U(i, j) = U[j * K + i].

#include "gptq.h"

#include "compute/cublas_handle.h"
#include "core/logging.h"
#include "quant/nvfp4_pack.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

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

// Column i: quantize, record the scaled error, update the rest of the block in place.
__global__ void column_kernel(float* __restrict__ w, int64_t N, int64_t K, int64_t i, int64_t b0, int64_t b1,
                              const float* __restrict__ u, const float* __restrict__ sc,
                              uint8_t* __restrict__ codes, float* __restrict__ err) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= N)
        return;
    float* row = w + r * K;
    const float x = row[i];
    const uint8_t code = e2m1_code(x / sc[r]);
    codes[r * K + i] = code;
    const float e = (x - e2m1_value(code) * sc[r]) / u[i * K + i];
    err[r * kBlock + (i - b0)] = e;
    for (int64_t j = i + 1; j < b1; j++)
        row[j] -= e * u[j * K + i];
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

// Lower triangle (column-major) mirrored into the upper one.
__global__ void symmetrize_lower_kernel(float* __restrict__ a, int64_t K) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= K * K)
        return;
    const int64_t i = idx % K, j = idx / K;  // a[j * K + i] = A(i, j)
    if (i < j)
        a[j * K + i] = a[i * K + j];
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

// sum over rows of e_r H e_r^T for the error matrix e [N, K] (row-major), with eh scratch [N, K];
// negative when cuBLAS fails.
double quad_loss(cublasHandle_t h, const float* e, const float* H, float* eh, int64_t N, int64_t K) {
    const float one = 1.0f, zero = 0.0f;
    // Column-major view: E^T [K, N]; (E H)^T = H E^T (H symmetric).
    float dot = 0.0f;
    const bool ok = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(K), static_cast<int>(N),
                                static_cast<int>(K), &one, H, static_cast<int>(K), e, static_cast<int>(K),
                                &zero, eh, static_cast<int>(K)) == CUBLAS_STATUS_SUCCESS &&
                    cublasSdot(h, static_cast<int>(N * K), e, 1, eh, 1, &dot) == CUBLAS_STATUS_SUCCESS;
    return ok ? static_cast<double>(dot) : -1.0;
}

}  // namespace

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
                                                  std::vector<float> H, float tensor_scale, float damp) {
    if (K % kNvfp4MicroBlockSize != 0)
        return std::unexpected("K = " + std::to_string(K) + " is not a multiple of 16");
    const size_t nk = static_cast<size_t>(N) * static_cast<size_t>(K), kk = static_cast<size_t>(K) * K;
    if (H.size() != kk)
        return std::unexpected("Hessian is not K x K");

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
        }
        d += static_cast<float>(damp * mean_diag);
    }

    Dev<float> dw(nk), dw0(nk), du(kk), dh0(kk), dq(nk), dsc(static_cast<size_t>(N)),
        derr(static_cast<size_t>(N) * kBlock);
    Dev<uint8_t> dcodes(nk), dmicro(nk / 16), dpacked(nk / 2);
    Dev<int> dinfo(1);
    if (!dw.p || !dw0.p || !du.p || !dh0.p || !dq.p || !dsc.p || !derr.p || !dcodes.p || !dmicro.p ||
        !dpacked.p || !dinfo.p)
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

    // U = upper Cholesky factor of H^-1: potrf + potri (lower), mirror, potrf (upper).
    cusolverDnHandle_t sol = nullptr;
    if (cusolverDnCreate(&sol) != CUSOLVER_STATUS_SUCCESS)
        return std::unexpected("cuSOLVER handle creation failed");
    cublasHandle_t bl = cublas_handle_create("imp-quantize::gptq", CUBLAS_DEFAULT_MATH);
    struct Handles {
        cusolverDnHandle_t s;
        cublasHandle_t b;
        ~Handles() {
            if (cusolverDnDestroy(s) != CUSOLVER_STATUS_SUCCESS)
                std::fprintf(stderr, "gptq: cuSOLVER handle destroy failed\n");
            cublas_handle_destroy(b);
        }
    } handles{sol, bl};
    const int k = static_cast<int>(K);
    int lw = 0, lw2 = 0;
    cusolverDnSpotrf_bufferSize(sol, CUBLAS_FILL_MODE_LOWER, k, du.p, k, &lw);
    cusolverDnSpotri_bufferSize(sol, CUBLAS_FILL_MODE_LOWER, k, du.p, k, &lw2);
    lw = std::max(lw, lw2);
    cusolverDnSpotrf_bufferSize(sol, CUBLAS_FILL_MODE_UPPER, k, du.p, k, &lw2);
    Dev<float> work(static_cast<size_t>(std::max(lw, lw2)));
    int info = 0;
    auto step_ok = [&](cusolverStatus_t st) {
        return st == CUSOLVER_STATUS_SUCCESS &&
               cudaMemcpy(&info, dinfo.p, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && info == 0;
    };
    if (!work.p || !step_ok(cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_LOWER, k, du.p, k, work.p, lw, dinfo.p)) ||
        !step_ok(cusolverDnSpotri(sol, CUBLAS_FILL_MODE_LOWER, k, du.p, k, work.p, lw, dinfo.p)))
        return std::unexpected("Cholesky inverse of H failed (info " + std::to_string(info) + ")");
    symmetrize_lower_kernel<<<grid(static_cast<int64_t>(kk)), kThreads>>>(du.p, K);
    if (!step_ok(
            cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_UPPER, k, du.p, k, work.p, std::max(lw, lw2), dinfo.p)))
        return std::unexpected("Cholesky of H^-1 failed (info " + std::to_string(info) + ")");

    // Blocks of kBlock columns: per-column quantize + in-block update, then one GEMM for the tail.
    for (int64_t b0 = 0; b0 < K; b0 += kBlock) {
        const int64_t b1 = std::min<int64_t>(b0 + kBlock, K);
        for (int64_t i = b0; i < b1; i++) {
            if (i % kNvfp4MicroBlockSize == 0)
                group_scale_kernel<<<grid(N), kThreads>>>(dw.p, N, K, i, tensor_scale, dmicro.p, dsc.p);
            column_kernel<<<grid(N), kThreads>>>(dw.p, N, K, i, b0, b1, du.p, dsc.p, dcodes.p, derr.p);
        }
        if (b1 < K) {
            // W[:, b1:] -= Err [N, B] * U(b0:b1, b1:). Column-major: C [K-b1, N] (ld K) -= A^T E with
            // A = U(b0:b1, b1:) as [B, K-b1] (ld K) and E = Err^T [B, N] (ld kBlock).
            const float minus = -1.0f, one = 1.0f;
            if (cublasSgemm(bl, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(K - b1), static_cast<int>(N),
                            static_cast<int>(b1 - b0), &minus, du.p + b1 * K + b0, k, derr.p, kBlock, &one,
                            dw.p + b1, k) != CUBLAS_STATUS_SUCCESS)
                return std::unexpected("tail update GEMM failed");
        }
    }

    Result out;
    out.packed.resize(nk / 2);
    out.micro.resize(nk / 16);
    // Losses against the undamped H: RTN of the original weights, then the GPTQ codes.
    rtn_dequant_kernel<<<grid(static_cast<int64_t>(nk / 16)), kThreads>>>(dw0.p, N, K, tensor_scale, dq.p);
    residual_kernel<<<grid(static_cast<int64_t>(nk)), kThreads>>>(dw0.p, dq.p, static_cast<int64_t>(nk));
    out.loss_rtn = quad_loss(bl, dq.p, dh0.p, dw.p, N, K);
    gptq_dequant_pack_kernel<<<grid(static_cast<int64_t>(nk / 2)), kThreads>>>(dcodes.p, dmicro.p, N, K,
                                                                               tensor_scale, dq.p, dpacked.p);
    if (cudaMemcpy(out.packed.data(), dpacked.p, nk / 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out.micro.data(), dmicro.p, nk / 16, cudaMemcpyDeviceToHost) != cudaSuccess)
        return std::unexpected("D2H copy failed");
    residual_kernel<<<grid(static_cast<int64_t>(nk)), kThreads>>>(dw0.p, dq.p, static_cast<int64_t>(nk));
    out.loss_gptq = quad_loss(bl, dq.p, dh0.p, dw.p, N, K);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
        return std::unexpected(std::string("GPTQ kernels failed: ") + cudaGetErrorString(e));
    return out;
}

}  // namespace imp::gptq

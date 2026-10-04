// cuBLASLt NVFP4 (VEC16_UE4M3, TN, COMPUTE_32F) vs imp's CUTLASS sm_120 NVFP4 dense GEMM (#2540).
//
// Both arms read the same packed FP4 bytes and SfAtom scales from quantize_fp16_to_nvfp4_cutlass;
// accuracy is rel-L2 of each arm vs an FP16 cuBLAS GEMM on the unquantized inputs.
// Weights rotate over >= 400 MB of copies (isolated benches otherwise measure L2, 96 MB).
// cuBLASLt: every heuristic candidate (up to 8) timed, best kept. Per cell: 1 s warmup,
// then 10 alternating trials, median per arm. M <= 32 adds imp's decode kernel (smallm v2,
// plain-layout weights, gemm.nvfp4_smallm default); activation quantize excluded in all arms.
//
// Build and run against a `make dev` tree (needs a GPU):
//   docker run --rm --gpus all -v $PWD:/src -w /src imp:toolchain bash -c \
//     'nvcc -O2 -std=c++20 -arch=sm_120a -Isrc tools/analysis/nvfp4_cublaslt_vs_cutlass.cu \
//      build-dev/libimp.a build-dev/libimp_core.a -lcublasLt -lcublas -lcuda -lcurl -lcrypto -ljpeg \
//      -o /tmp/b && /tmp/b'

#include "compute/gemm_cutlass_sm120.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)
#define LK(x)                                                                                     \
    do {                                                                                          \
        cublasStatus_t s_ = (x);                                                                  \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                        \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cublasLtGetStatusName(s_)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

namespace {

struct Shape {
    const char* name;
    int N, K;
};

// Qwen3-14B (hidden 5120, 40/8 heads x 128, ffn 17408) and Qwen3.8-27B (hidden 5120,
// 24/4 heads x 256 with output gate, GDN 16k/48v heads x 128, ffn 17408), deduplicated.
constexpr Shape kShapes[] = {
    {"q/o 14B", 5120, 5120},      {"k/v 14B+27B", 1024, 5120}, {"gate/up", 17408, 5120},
    {"down", 5120, 17408},        {"q+gate 27B", 12288, 5120}, {"o/out 27B", 5120, 6144},
    {"gdn qkv 27B", 10240, 5120}, {"gdn z 27B", 6144, 5120},
};
constexpr int kMs[] = {16, 32, 128, 512, 2048, 8192};
constexpr int kTrials = 10;
constexpr size_t kRotateBytes = 400ull << 20;
constexpr size_t kLtWorkspace = 32ull << 20;

__global__ void fill_half(__half* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        h *= 3266489917u;
        h ^= h >> 16;
        // Uniform [-1,1) plus a sparse outlier channel pattern so blocks carry distinct scales.
        float v = ((h & 0xFFFFFF) / 8388608.0f - 1.0f) * scale;
        if ((h >> 24) == 0)
            v *= 8.0f;
        p[i] = __float2half(v);
    }
}

// acc[0] += sum (x - ref)^2, acc[1] += sum ref^2, acc[2] += sum (x - y)^2, acc[3] += count(x == y)
__global__ void err_stats(const __half* x, const __half* y, const float* ref, size_t n, double* acc) {
    double e = 0, r = 0, d = 0, eq = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float a = __half2float(x[i]), b = __half2float(y[i]), c = ref[i];
        e += (double)(a - c) * (a - c);
        r += (double)c * c;
        d += (double)(a - b) * (a - b);
        eq += (__half_as_ushort(x[i]) == __half_as_ushort(y[i]));
    }
    atomicAdd(&acc[0], e);
    atomicAdd(&acc[1], r);
    atomicAdd(&acc[2], d);
    atomicAdd(&acc[3], eq);
}

// L2 of y vs ref, for the second arm.
__global__ void err_ref(const __half* y, const float* ref, size_t n, double* acc) {
    double e = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float b = __half2float(y[i]);
        e += (double)(b - ref[i]) * (b - ref[i]);
    }
    atomicAdd(&acc[0], e);
}

template <typename Run>
float time_ms(cudaEvent_t a, cudaEvent_t b, cudaStream_t s, int iters, Run&& run) {
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < iters; i++)
        run(i);
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Lt {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    std::vector<cublasLtMatmulAlgo_t> algos;
};

// cuBLAS column-major view of D[M,N] = A[M,K] W[N,K]^T: D^T(NxM) = op_T(W as KxN) * A^T(KxM).
// "A" operand = weight with its SfAtom scales, "B" operand = activation.
Lt make_lt(cublasLtHandle_t lt, int M, int N, int K) {
    Lt r;
    LK(cublasLtMatmulDescCreate(&r.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
    const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    LK(cublasLtMatrixLayoutCreate(&r.la, CUDA_R_4F_E2M1, K, N, K));
    LK(cublasLtMatrixLayoutCreate(&r.lb, CUDA_R_4F_E2M1, K, M, K));
    LK(cublasLtMatrixLayoutCreate(&r.lc, CUDA_R_16F, N, M, N));
    return r;
}

void set_scales(const Lt& r, const void* w_sf, const void* a_sf) {
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sf, sizeof(w_sf)));
    LK(cublasLtMatmulDescSetAttribute(r.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_sf, sizeof(a_sf)));
}

void destroy_lt(Lt& r) {
    cublasLtMatrixLayoutDestroy(r.la);
    cublasLtMatrixLayoutDestroy(r.lb);
    cublasLtMatrixLayoutDestroy(r.lc);
    cublasLtMatmulDescDestroy(r.desc);
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("# device: %s (sm_%d%d), cuBLASLt %zu, CUTLASS available %d\n", prop.name, prop.major, prop.minor,
           cublasLtGetVersion(), (int)imp::cutlass_sm120_nvfp4_available());
    printf(
        "shape\tM\tN\tK\tcutlass_us\tlt_best_us\tlt_h0_us\tlt_algos\tlt_best_idx\tlt_over_cutlass\t"
        "cutlass_tflops\tlt_tflops\trelerr_cutlass\trelerr_lt\trel_diff\tbit_equal_pct\tsmallm_us\t"
        "relerr_smallm\n");

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasLtHandle_t lt;
    LK(cublasLtCreate(&lt));
    cublasHandle_t bl;
    LK(cublasCreate(&bl));
    LK(cublasSetStream(bl, s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    void* lt_ws = nullptr;
    CK(cudaMalloc(&lt_ws, kLtWorkspace));
    double* acc = nullptr;
    CK(cudaMalloc(&acc, 4 * sizeof(double)));

    for (const Shape& sh : kShapes) {
        const int N = sh.N, K = sh.K;
        const size_t w_bytes = (size_t)N * K / 2, w_sf = imp::cutlass_nvfp4_sf_size(N, K);
        const int R = (int)std::clamp<size_t>((kRotateBytes + w_bytes - 1) / w_bytes, 2, 16);

        __half* w16 = nullptr;
        CK(cudaMalloc(&w16, (size_t)N * K * sizeof(__half)));
        fill_half<<<1024, 256, 0, s>>>(w16, (size_t)N * K, 0x1234u + N, 0.05f);
        std::vector<uint8_t*> wd(R), ws(R);
        for (int r = 0; r < R; r++) {
            CK(cudaMalloc(&wd[r], w_bytes));
            CK(cudaMalloc(&ws[r], w_sf));
            CK(cudaMemsetAsync(ws[r], 0, w_sf, s));
        }
        imp::quantize_fp16_to_nvfp4_cutlass(w16, wd[0], ws[0], N, K, s);
        // Plain layout ([N,K/2] nibbles + [N,K/16] E4M3) for the smallm v2 arm.
        const size_t wp_sf = (size_t)N * K / 16;
        std::vector<uint8_t*> wpd(R), wps(R);
        for (int r = 0; r < R; r++) {
            CK(cudaMalloc(&wpd[r], w_bytes));
            CK(cudaMalloc(&wps[r], wp_sf));
        }
        imp::quantize_fp16_to_nvfp4_into(w16, N, K, wpd[0], wps[0], 1.0f, s);
        for (int r = 1; r < R; r++) {
            CK(cudaMemcpyAsync(wd[r], wd[0], w_bytes, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(ws[r], ws[0], w_sf, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(wpd[r], wpd[0], w_bytes, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(wps[r], wps[0], wp_sf, cudaMemcpyDeviceToDevice, s));
        }
        std::vector<imp::NvFP4QuantResult> wp(R);
        for (int r = 0; r < R; r++) {
            wp[r].packed_data = wpd[r];
            wp[r].micro_scales = wps[r];
            wp[r].tensor_scale = 1.0f;
            wp[r].N = N;
            wp[r].K = K;
            wp[r].owned = false;
        }

        for (int M : kMs) {
            const size_t a_sf_bytes = imp::cutlass_nvfp4_sf_size(M, K);
            __half *a16 = nullptr, *d_cut = nullptr, *d_lt = nullptr;
            uint8_t *ad = nullptr, *as = nullptr;
            float* ref = nullptr;
            CK(cudaMalloc(&a16, (size_t)M * K * sizeof(__half)));
            CK(cudaMalloc(&ad, (size_t)M * K / 2));
            CK(cudaMalloc(&as, a_sf_bytes));
            CK(cudaMemsetAsync(as, 0, a_sf_bytes, s));
            CK(cudaMalloc(&d_cut, (size_t)M * N * sizeof(__half)));
            CK(cudaMalloc(&d_lt, (size_t)M * N * sizeof(__half)));
            CK(cudaMalloc(&ref, (size_t)M * N * sizeof(float)));
            fill_half<<<1024, 256, 0, s>>>(a16, (size_t)M * K, 0x9876u + M, 1.0f);
            imp::quantize_fp16_to_nvfp4_cutlass(a16, ad, as, M, K, s);

            const size_t cws_bytes = imp::gemm_nvfp4_cutlass_sm120_workspace(M, N, K);
            void* cws = nullptr;
            if (cws_bytes)
                CK(cudaMalloc(&cws, cws_bytes));
            std::vector<imp::CutlassNvFP4Weight> wt(R);
            for (int r = 0; r < R; r++) {
                wt[r].data = wd[r];
                wt[r].scale_factors = ws[r];
                wt[r].tensor_scale = 1.0f;
                wt[r].N = N;
                wt[r].K = K;
                wt[r].sf_bytes = w_sf;
            }
            bool cut_ok = true;
            auto run_cut = [&](int i) {
                cut_ok &= imp::gemm_nvfp4_cutlass_sm120(ad, as, wt[i % R], d_cut, M, N, K, cws, cws_bytes, s);
            };

            Lt L = make_lt(lt, M, N, K);
            set_scales(L, ws[0], as);
            cublasLtMatmulPreference_t pref;
            LK(cublasLtMatmulPreferenceCreate(&pref));
            LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &kLtWorkspace, sizeof(kLtWorkspace)));
            cublasLtMatmulHeuristicResult_t hr[8] = {};
            int nh = 0;
            cublasLtMatmulAlgoGetHeuristic(lt, L.desc, L.la, L.lb, L.lc, L.lc, pref, 8, hr, &nh);
            cublasLtMatmulPreferenceDestroy(pref);
            const float one = 1.0f, zero = 0.0f;
            int algo = 0;
            auto run_lt = [&](int i) {
                set_scales(L, ws[i % R], as);
                LK(cublasLtMatmul(lt, L.desc, &one, wd[i % R], L.la, ad, L.lb, &zero, d_lt, L.lc, d_lt, L.lc,
                                  &hr[algo].algo, lt_ws, kLtWorkspace, s));
            };

            // FP16 reference on the unquantized inputs, FP32 output.
            LK(cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w16, CUDA_R_16F, K, a16, CUDA_R_16F,
                            K, &zero, ref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            // Warm >1 s (idle downclock ramps over ~1 s).
            const auto t0 = std::chrono::steady_clock::now();
            int wi = 0;
            while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1100)) {
                for (int j = 0; j < 20; j++)
                    run_cut(wi++);
                CK(cudaStreamSynchronize(s));
            }
            const float est = time_ms(e0, e1, s, 10, run_cut);
            const int iters = std::clamp((int)(4.0f / std::max(est, 1e-4f)), 20, 2000);

            float lt_h0 = 0, lt_best = 1e30f;
            int best = -1;
            for (int a = 0; a < nh; a++) {
                algo = a;
                std::vector<float> t;
                for (int k = 0; k < 3; k++)
                    t.push_back(time_ms(e0, e1, s, iters, run_lt));
                const float m = median(t);
                if (a == 0)
                    lt_h0 = m;
                if (m < lt_best) {
                    lt_best = m;
                    best = a;
                }
            }

            std::vector<float> tc, tl;
            for (int k = 0; k < kTrials; k++) {
                tc.push_back(time_ms(e0, e1, s, iters, run_cut));
                if (best >= 0) {
                    algo = best;
                    tl.push_back(time_ms(e0, e1, s, iters, run_lt));
                }
            }
            const float c_ms = median(tc), l_ms = best >= 0 ? median(tl) : 0.0f;

            // smallm v2 arm (M <= 32), same rotation, own trials.
            float sm_ms = 0;
            double sm_err = -1;
            if (M <= 32) {
                uint8_t *xd = nullptr, *xs = nullptr;
                __half* d_sm = nullptr;
                void* sws = nullptr;
                const size_t sws_bytes = imp::gemm_nvfp4_smallm_v2_workspace_bytes(N, K);
                CK(cudaMalloc(&xd, (size_t)M * K / 2));
                CK(cudaMalloc(&xs, (size_t)M * K / 16));
                CK(cudaMalloc(&d_sm, (size_t)M * N * sizeof(__half)));
                if (sws_bytes)
                    CK(cudaMalloc(&sws, sws_bytes));
                imp::quantize_fp16_to_nvfp4_into(a16, M, K, xd, xs, 1.0f, s);
                imp::NvFP4QuantResult xq;
                xq.packed_data = xd;
                xq.micro_scales = xs;
                xq.tensor_scale = 1.0f;
                xq.N = M;
                xq.K = K;
                xq.owned = false;
                bool sm_ok = true;
                auto run_sm = [&](int i) {
                    sm_ok &= imp::gemm_nvfp4_smallm_v2_a4(wp[i % R], xq, reinterpret_cast<half*>(d_sm), M, N,
                                                          K, sws, s);
                };
                std::vector<float> ts;
                for (int k = 0; k < kTrials; k++)
                    ts.push_back(time_ms(e0, e1, s, iters, run_sm));
                sm_ms = sm_ok ? median(ts) : 0.0f;
                run_sm(0);
                CK(cudaMemsetAsync(acc, 0, sizeof(double), s));
                err_ref<<<512, 256, 0, s>>>(d_sm, ref, (size_t)M * N, acc);
                CK(cudaMemcpyAsync(&sm_err, acc, sizeof(sm_err), cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                cudaFree(xd);
                cudaFree(xs);
                cudaFree(d_sm);
                cudaFree(sws);
            }

            // Accuracy on rotation slot 0.
            run_cut(0);
            if (best >= 0) {
                algo = best;
                run_lt(0);
            } else {
                CK(cudaMemsetAsync(d_lt, 0, (size_t)M * N * sizeof(__half), s));
            }
            CK(cudaMemsetAsync(acc, 0, 4 * sizeof(double), s));
            err_stats<<<512, 256, 0, s>>>(d_cut, d_lt, ref, (size_t)M * N, acc);
            double h[4];
            CK(cudaMemcpyAsync(h, acc, sizeof(h), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            CK(cudaMemsetAsync(acc, 0, sizeof(double), s));
            err_ref<<<512, 256, 0, s>>>(d_lt, ref, (size_t)M * N, acc);
            double hl;
            CK(cudaMemcpyAsync(&hl, acc, sizeof(hl), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));

            const double flop = 2.0 * M * N * K;
            printf(
                "%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%d\t%.3f\t%.1f\t%.1f\t%.4f\t%.4f\t%.4f\t%.2f\t%.2f\t%."
                "4f%s\n",
                sh.name, M, N, K, c_ms * 1e3, l_ms * 1e3, lt_h0 * 1e3, nh, best,
                best >= 0 ? l_ms / c_ms : 0.0, flop / (c_ms * 1e9), best >= 0 ? flop / (l_ms * 1e9) : 0.0,
                std::sqrt(h[0] / h[1]), std::sqrt(hl / h[1]), std::sqrt(h[2] / h[1]),
                100.0 * h[3] / ((double)M * N), sm_ms * 1e3, sm_err >= 0 ? std::sqrt(sm_err / h[1]) : -1.0,
                cut_ok ? "" : "\tCUTLASS_FAILED");
            fflush(stdout);

            destroy_lt(L);
            cudaFree(cws);
            cudaFree(a16);
            cudaFree(ad);
            cudaFree(as);
            cudaFree(d_cut);
            cudaFree(d_lt);
            cudaFree(ref);
        }
        cudaFree(w16);
        for (int r = 0; r < R; r++) {
            cudaFree(wd[r]);
            cudaFree(ws[r]);
            cudaFree(wpd[r]);
            cudaFree(wps[r]);
        }
    }
    return 0;
}

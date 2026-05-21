// =============================================================================
// attention_prefill_paths_bench.cu — implementation
// =============================================================================

#include "bench/attention_prefill_paths_bench.h"

#include "compute/attention_cublas.h"
#include "compute/attention_fmha_sm120.h"
#include "core/qtype.h"
#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace imp {

namespace {

constexpr int kWarmup = 3;
constexpr int kReps = 10;

void fill_random_fp16(__half* d_ptr, size_t n) {
    std::vector<__half> host(n);
    for (size_t i = 0; i < n; ++i) {
        // Bounded, deterministic, non-zero values.
        float v = (static_cast<float>((i * 2654435761u) % 1024u) / 1024.0f) - 0.5f;
        host[i] = __float2half(v * 0.125f);
    }
    cudaMemcpy(d_ptr, host.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// FLOPS for causal full-sequence prefill: roughly 2 * (QKᵀ + PV) for the lower
// triangle, i.e. 4 * n_heads * (seq² / 2) * head_dim = 2 * nh * seq² * hd.
double causal_prefill_gflops(int seq, int n_heads, int head_dim, double ms) {
    double flops = 2.0 * n_heads * static_cast<double>(seq) * seq * head_dim;
    return flops / (ms * 1.0e-3) / 1.0e9;
}

}  // namespace

bool attention_prefill_paths_bench(int seq, int n_heads, int n_kv_heads,
                                   int head_dim, AttnPrefillBenchResult* out) {
    if (!out) return false;
    if (seq <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return false;
    if (n_heads % n_kv_heads != 0) return false;

    out->cublas_ms = std::nan("");
    out->fmha_ms = std::nan("");
    out->cublas_gflops = 0.0;
    out->fmha_gflops = 0.0;
    out->cublas_s_workspace_bytes = 0;

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const bool causal = true;

    // ------------------------------------------------------------------------
    // Allocate Q, K, V, O in 2D layout for cuBLAS:
    //   Q: [seq, n_heads * head_dim]
    //   K: [seq, n_kv_heads * head_dim]
    //   V: [seq, n_kv_heads * head_dim]
    //   O: [seq, n_heads * head_dim]
    // FMHA reinterprets the same buffers as [1, seq, nh, hd] / [1, seq, nkv, hd].
    // ------------------------------------------------------------------------
    const size_t q_elems = static_cast<size_t>(seq) * n_heads * head_dim;
    const size_t kv_elems = static_cast<size_t>(seq) * n_kv_heads * head_dim;

    __half* d_Q = nullptr;
    __half* d_K = nullptr;
    __half* d_V = nullptr;
    __half* d_O = nullptr;

    auto bail = [&](const char* why) {
        if (d_Q) cudaFree(d_Q);
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_O) cudaFree(d_O);
        std::fprintf(stderr, "attention_prefill_paths_bench[seq=%d nh=%d nkv=%d hd=%d]: %s\n",
                     seq, n_heads, n_kv_heads, head_dim, why);
        return false;
    };

    if (cudaMalloc(&d_Q, q_elems * sizeof(__half)) != cudaSuccess) return bail("alloc Q");
    if (cudaMalloc(&d_K, kv_elems * sizeof(__half)) != cudaSuccess) return bail("alloc K");
    if (cudaMalloc(&d_V, kv_elems * sizeof(__half)) != cudaSuccess) return bail("alloc V");
    if (cudaMalloc(&d_O, q_elems * sizeof(__half)) != cudaSuccess) return bail("alloc O");

    fill_random_fp16(d_Q, q_elems);
    fill_random_fp16(d_K, kv_elems);
    fill_random_fp16(d_V, kv_elems);
    cudaMemset(d_O, 0, q_elems * sizeof(__half));

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    // ------------------------------------------------------------------------
    // Säule 1: cuBLAS prefill
    //
    // S workspace must hold n_heads * seq * seq elements. The runtime picks
    // FP32 when 2 × FP32-elems ≤ allocated FP16-elems, so allocate enough
    // FP16 for the FP32 path (the production path).
    // ------------------------------------------------------------------------
    {
        const long long s_fp32_elems = static_cast<long long>(n_heads) * seq * seq;
        // Allocate enough FP16 elements so 2*fp32_elems ≤ fp16_elems (use_fp32_s=true).
        const long long s_fp16_elems = 2 * s_fp32_elems;
        const size_t s_bytes = static_cast<size_t>(s_fp16_elems) * sizeof(__half);
        out->cublas_s_workspace_bytes = static_cast<long long>(s_bytes);

        __half* d_S = nullptr;
        if (cudaMalloc(&d_S, s_bytes) != cudaSuccess) {
            std::fprintf(stderr, "  cuBLAS: skip — S workspace %zu MiB alloc failed\n",
                         s_bytes / (1024 * 1024));
            out->cublas_ms = std::nan("");
        } else {
            // No prewarm: handle is lazy-init inside attention_cublas_prefill.
            // Calling prewarm per-iter caused crashes on shape changes (handle
            // internal workspace fragmentation across alloc/free cycles).

            int64_t qkv_shape_2d[2] = {seq, n_heads * head_dim};
            int64_t kv_shape_2d[2] = {seq, n_kv_heads * head_dim};
            int64_t s_shape_3d[3] = {n_heads, seq, seq};
            // Layout S as [n_heads, seq, 2*seq] FP16 so the FP32 fits (matches
            // the runtime's "2*fp32_elems ≤ fp16_elems" check).
            int64_t s_shape_fp16[3] = {n_heads, seq, 2 * seq};

            Tensor Q(d_Q, QType::F16, 2, qkv_shape_2d, /*on_device=*/true);
            Tensor K(d_K, QType::F16, 2, kv_shape_2d, /*on_device=*/true);
            Tensor V(d_V, QType::F16, 2, kv_shape_2d, /*on_device=*/true);
            Tensor O(d_O, QType::F16, 2, qkv_shape_2d, /*on_device=*/true);
            Tensor S(d_S, QType::F16, 3, s_shape_fp16, /*on_device=*/true);

            auto run_cublas = [&]() {
                attention_cublas_prefill(Q, K, V, O, S, n_heads, n_kv_heads, head_dim,
                                        scale, causal, /*softcap=*/0.0f,
                                        /*q_offset=*/0, stream, /*sliding_window=*/0);
            };

            for (int w = 0; w < kWarmup; ++w) run_cublas();
            cudaStreamSynchronize(stream);

            std::vector<double> samples;
            samples.reserve(kReps);
            for (int r = 0; r < kReps; ++r) {
                cudaEventRecord(ev0, stream);
                run_cublas();
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                samples.push_back(static_cast<double>(ms));
            }
            if (cudaGetLastError() != cudaSuccess) {
                std::fprintf(stderr, "  cuBLAS: launch error after timing loop\n");
                out->cublas_ms = std::nan("");
            } else {
                out->cublas_ms = median(samples);
                out->cublas_gflops = causal_prefill_gflops(seq, n_heads, head_dim,
                                                          out->cublas_ms);
            }
            (void)s_shape_3d;  // suppress unused
            cudaFree(d_S);
        }
    }

    // ------------------------------------------------------------------------
    // Säule 2: FMHA prefill
    //
    // FMHA expects 4D layout [batch, seq, nh, hd]. Reuse the same gmem buffers
    // (FP16 contiguous = same memory pattern as cuBLAS).
    // ------------------------------------------------------------------------
    {
        int64_t qo_shape_4d[4] = {1, seq, n_heads, head_dim};
        int64_t kv_shape_4d[4] = {1, seq, n_kv_heads, head_dim};

        Tensor Q4(d_Q, QType::F16, 4, qo_shape_4d, /*on_device=*/true);
        Tensor K4(d_K, QType::F16, 4, kv_shape_4d, /*on_device=*/true);
        Tensor V4(d_V, QType::F16, 4, kv_shape_4d, /*on_device=*/true);
        Tensor O4(d_O, QType::F16, 4, qo_shape_4d, /*on_device=*/true);

        bool ok = true;
        auto run_fmha = [&]() {
            if (!ok) return;
            ok = fmha_sm120_prefill(Q4, K4, V4, O4, scale, causal,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream);
        };

        for (int w = 0; w < kWarmup; ++w) {
            run_fmha();
            if (!ok) break;
        }
        cudaStreamSynchronize(stream);

        if (!ok) {
            std::fprintf(stderr, "  FMHA: kernel returned false (unsupported config)\n");
            out->fmha_ms = std::nan("");
        } else {
            std::vector<double> samples;
            samples.reserve(kReps);
            for (int r = 0; r < kReps; ++r) {
                cudaEventRecord(ev0, stream);
                run_fmha();
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                if (!ok) break;
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                samples.push_back(static_cast<double>(ms));
            }
            if (cudaGetLastError() != cudaSuccess) {
                std::fprintf(stderr, "  FMHA: launch error after timing loop\n");
                out->fmha_ms = std::nan("");
            } else if (samples.size() == kReps) {
                out->fmha_ms = median(samples);
                out->fmha_gflops = causal_prefill_gflops(seq, n_heads, head_dim,
                                                        out->fmha_ms);
            }
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return true;
}

}  // namespace imp

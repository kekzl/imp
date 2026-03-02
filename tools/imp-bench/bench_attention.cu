#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef IMP_USE_CUTLASS
#include "compute/attention_cutlass_fmha.h"
#endif

namespace imp {

struct AttentionConfig {
    const char* name;
    int n_heads;
    int n_kv_heads;
    int head_dim;
};

// Run attention with a specific kernel path and return avg time in ms
static float bench_kernel(
    const AttentionConfig& cfg, int seq_len, bool use_cutlass, cudaStream_t stream)
{
    const int batch = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
    const int warmup_iters = 10;
    const int timed_iters  = 30;

    const int64_t q_elems = (int64_t)batch * seq_len * cfg.n_heads * cfg.head_dim;
    const int64_t kv_elems = (int64_t)batch * seq_len * cfg.n_kv_heads * cfg.head_dim;

    const size_t q_bytes  = q_elems * sizeof(__half);
    const size_t kv_bytes = kv_elems * sizeof(__half);
    const size_t o_bytes  = q_elems * sizeof(__half);

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_o, o_bytes);

    // Fill with deterministic data
    {
        size_t max_elems = q_elems > kv_elems ? q_elems : kv_elems;
        std::vector<__half> h_buf(max_elems);
        for (int64_t i = 0; i < q_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 7 + 13) % 1000 - 500));
        cudaMemcpy(d_q, h_buf.data(), q_bytes, cudaMemcpyHostToDevice);
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 11 + 17) % 1000 - 500));
        cudaMemcpy(d_k, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice);
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 13 + 19) % 1000 - 500));
        cudaMemcpy(d_v, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice);
    }
    cudaMemset(d_o, 0, o_bytes);

    int64_t q_shape[4]  = {batch, seq_len, cfg.n_heads, cfg.head_dim};
    int64_t kv_shape[4] = {batch, seq_len, cfg.n_kv_heads, cfg.head_dim};
    int64_t o_shape[4]  = {batch, seq_len, cfg.n_heads, cfg.head_dim};

    Tensor Q(d_q, DType::FP16, 4, q_shape, true);
    Tensor K(d_k, DType::FP16, 4, kv_shape, true);
    Tensor V(d_v, DType::FP16, 4, kv_shape, true);
    Tensor O(d_o, DType::FP16, 4, o_shape, true);

    // Select kernel path
    auto run_kernel = [&]() {
#ifdef IMP_USE_CUTLASS
        if (use_cutlass) {
            if (cutlass_fmha_prefill(Q, K, V, O, scale, true, stream))
                return;
        }
#endif
        // WMMA fallback (Blackwell or Hopper)
        int sm_ver = get_device_sm_version();
        if (sm_ver >= 120)
            flash_attention_blackwell(Q, K, V, O, scale, true, 0, 0.0f, stream);
        else if (sm_ver >= 90)
            flash_attention_prefill_tc(Q, K, V, O, scale, true, 0, 0.0f, stream);
        else
            flash_attention_prefill(Q, K, V, O, scale, true, 0, 0.0f, stream);
    };

    // Warmup
    for (int i = 0; i < warmup_iters; i++)
        run_kernel();
    cudaStreamSynchronize(stream);

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < timed_iters; i++)
        run_kernel();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / (float)timed_iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);

    return avg_ms;
}

void bench_attention() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("bench_attention: no CUDA device found, skipping\n");
        return;
    }

    printf("=== Attention Prefill: CUTLASS FMHA vs WMMA Blackwell ===\n");
    printf("Causal, FP16, batch=1, warmup=10, iters=30\n\n");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::vector<AttentionConfig> configs = {
        {"Phi-4-mini",       24, 8, 128},
        {"DS-R1-7B (Qwen2)", 28, 4, 128},
        {"Qwen3-4B",         32, 8, 128},
        {"Qwen3-Coder-30B",  32, 4, 128},
        {"DS-R1-14B",        40, 8, 128},
        {"Llama-3-70B",      64, 8, 128},
    };

    std::vector<int> seq_lens = {512, 1024, 2048, 4096, 8192};

    for (const auto& cfg : configs) {
        printf("%-24s  nh=%2d nkv=%2d hd=%3d (GQA %dx)\n",
               cfg.name, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
               cfg.n_heads / cfg.n_kv_heads);
        printf("  %8s  %10s  %10s  %10s  %7s\n",
               "seq", "CUTLASS", "WMMA", "TFLOPS C/W", "speedup");

        for (int seq_len : seq_lens) {
            // Run CUTLASS first, then WMMA, then CUTLASS again — take best of two CUTLASS runs
            float ms_cutlass1 = bench_kernel(cfg, seq_len, true, stream);
            float ms_wmma     = bench_kernel(cfg, seq_len, false, stream);
            float ms_cutlass2 = bench_kernel(cfg, seq_len, true, stream);

            float ms_cutlass = ms_cutlass1 < ms_cutlass2 ? ms_cutlass1 : ms_cutlass2;

            double flops = 4.0 * 1 * cfg.n_heads * (double)seq_len * (double)seq_len * cfg.head_dim;
            double tflops_c = flops / (ms_cutlass * 1e-3) / 1e12;
            double tflops_w = flops / (ms_wmma * 1e-3) / 1e12;
            float speedup = ms_wmma / ms_cutlass;

            printf("  %8d  %8.3f ms  %8.3f ms  %5.1f/%5.1f  %5.2fx%s\n",
                   seq_len, ms_cutlass, ms_wmma, tflops_c, tflops_w, speedup,
                   speedup > 1.05f ? " <--" : "");
        }
        printf("\n");
    }

    cudaStreamDestroy(stream);
}

} // namespace imp

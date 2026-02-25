#include "compute/attention.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace imp {

void bench_attention() {
    // Check for CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("bench_attention: no CUDA device found, skipping\n");
        return;
    }

    // Configuration (Llama-3-8B style)
    const int batch     = 1;
    const int n_heads   = 32;
    const int n_kv_heads = 8;
    const int head_dim  = 128;
    const float scale   = 1.0f / sqrtf(static_cast<float>(head_dim));

    const std::vector<int> seq_lens = {128, 512, 1024, 2048, 4096};

    const int warmup_iters = 3;
    const int timed_iters  = 10;

    printf("=== Flash Attention Prefill Benchmark ===\n");
    printf("Config: n_heads=%d, n_kv_heads=%d, head_dim=%d\n\n", n_heads, n_kv_heads, head_dim);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (int seq_len : seq_lens) {
        // Compute sizes in elements
        const int64_t q_elems = (int64_t)batch * seq_len * n_heads * head_dim;
        const int64_t kv_elems = (int64_t)batch * seq_len * n_kv_heads * head_dim;
        const int64_t o_elems = q_elems;

        const size_t q_bytes  = q_elems * sizeof(__half);
        const size_t kv_bytes = kv_elems * sizeof(__half);
        const size_t o_bytes  = o_elems * sizeof(__half);

        // Allocate device memory
        void* d_q = nullptr;
        void* d_k = nullptr;
        void* d_v = nullptr;
        void* d_o = nullptr;

        cudaMalloc(&d_q, q_bytes);
        cudaMalloc(&d_k, kv_bytes);
        cudaMalloc(&d_v, kv_bytes);
        cudaMalloc(&d_o, o_bytes);

        // Fill Q, K, V with random FP16 values via host
        {
            std::vector<__half> h_buf(q_elems > kv_elems ? q_elems : kv_elems);

            // Fill Q
            for (int64_t i = 0; i < q_elems; i++) {
                float val = 0.01f * (float)((i * 7 + 13) % 1000 - 500);
                h_buf[i] = __float2half(val);
            }
            cudaMemcpy(d_q, h_buf.data(), q_bytes, cudaMemcpyHostToDevice);

            // Fill K
            for (int64_t i = 0; i < kv_elems; i++) {
                float val = 0.01f * (float)((i * 11 + 17) % 1000 - 500);
                h_buf[i] = __float2half(val);
            }
            cudaMemcpy(d_k, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice);

            // Fill V
            for (int64_t i = 0; i < kv_elems; i++) {
                float val = 0.01f * (float)((i * 13 + 19) % 1000 - 500);
                h_buf[i] = __float2half(val);
            }
            cudaMemcpy(d_v, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice);
        }

        cudaMemset(d_o, 0, o_bytes);

        // Build tensor descriptors
        // Q: [batch, seq_q, n_heads, head_dim]
        int64_t q_shape[4] = {batch, seq_len, n_heads, head_dim};
        Tensor Q(d_q, DType::FP16, 4, q_shape, true);

        // K,V: [batch, seq_kv, n_kv_heads, head_dim]
        int64_t kv_shape[4] = {batch, seq_len, n_kv_heads, head_dim};
        Tensor K(d_k, DType::FP16, 4, kv_shape, true);
        Tensor V(d_v, DType::FP16, 4, kv_shape, true);

        // O: [batch, seq_q, n_heads, head_dim]
        int64_t o_shape[4] = {batch, seq_len, n_heads, head_dim};
        Tensor O(d_o, DType::FP16, 4, o_shape, true);

        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            flash_attention_prefill(Q, K, V, O, scale, true, 0, stream);
        }
        cudaStreamSynchronize(stream);

        // Timed iterations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        for (int i = 0; i < timed_iters; i++) {
            flash_attention_prefill(Q, K, V, O, scale, true, 0, stream);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start, stop);
        float avg_ms = total_ms / static_cast<float>(timed_iters);

        // Effective TFLOPS: 4 * batch * n_heads * seq^2 * head_dim / (time_s) / 1e12
        double flops = 4.0 * batch * n_heads * (double)seq_len * (double)seq_len * head_dim;
        double tflops = flops / (avg_ms * 1e-3) / 1e12;

        // Effective bandwidth: (Q + K + V + O bytes) / time_s / 1e9 GB/s
        double total_bytes = (double)(q_bytes + kv_bytes + kv_bytes + o_bytes);
        double bw_gbs = total_bytes / (avg_ms * 1e-3) / 1e9;

        printf("  seq=%-8d  %6.2f ms  %6.1f TFLOPS  %7.1f GB/s\n",
               seq_len, avg_ms, tflops, bw_gbs);

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
    }

    cudaStreamDestroy(stream);
    printf("\n");
}

} // namespace imp

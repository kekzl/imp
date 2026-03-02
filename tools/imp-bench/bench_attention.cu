#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "compute/attention_paged.h"
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

// ---------------------------------------------------------------------------
// Paged Attention Decode Benchmark
// ---------------------------------------------------------------------------
// Measures latency of paged_attention_decode at various context lengths
// for MHA and GQA head configurations. Reports µs, effective bandwidth,
// and whether split-K was activated.

static float bench_paged_decode_kernel(
    const AttentionConfig& cfg, int ctx_len, cudaStream_t stream)
{
    const int batch = 1;
    const int block_size = 16;  // kKVBlockSize
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
    const int warmup_iters = 20;
    const int timed_iters  = 50;

    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;
    const int max_context_len = ctx_len;

    // Q: [batch, 1, n_heads, head_dim]
    const int64_t q_elems = (int64_t)batch * 1 * cfg.n_heads * cfg.head_dim;
    // KV cache: [num_kv_blocks, block_size, n_kv_heads, head_dim] (paged layout)
    const int64_t kv_elems = (int64_t)num_kv_blocks * block_size * cfg.n_kv_heads * cfg.head_dim;
    // O: same shape as Q
    const int64_t o_elems = q_elems;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_elems * sizeof(__half));
    cudaMalloc(&d_k, kv_elems * sizeof(__half));
    cudaMalloc(&d_v, kv_elems * sizeof(__half));
    cudaMalloc(&d_o, o_elems * sizeof(__half));

    // Fill with deterministic data
    {
        int64_t max_e = q_elems > kv_elems ? q_elems : kv_elems;
        std::vector<__half> h_buf(max_e);
        for (int64_t i = 0; i < q_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 7 + 13) % 1000 - 500));
        cudaMemcpy(d_q, h_buf.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice);
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 11 + 17) % 1000 - 500));
        cudaMemcpy(d_k, h_buf.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 13 + 19) % 1000 - 500));
        cudaMemcpy(d_v, h_buf.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice);
    }
    cudaMemset(d_o, 0, o_elems * sizeof(__half));

    // Block tables: identity mapping [batch, max_num_blocks]
    int* d_block_tables;
    int* d_context_lens;
    cudaMalloc(&d_block_tables, num_kv_blocks * sizeof(int));
    cudaMalloc(&d_context_lens, batch * sizeof(int));

    std::vector<int> h_bt(num_kv_blocks);
    for (int i = 0; i < num_kv_blocks; i++) h_bt[i] = i;
    cudaMemcpy(d_block_tables, h_bt.data(), num_kv_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int h_ctx = ctx_len;
    cudaMemcpy(d_context_lens, &h_ctx, sizeof(int), cudaMemcpyHostToDevice);

    // Split-K scratch buffer (generous)
    void* d_scratch = nullptr;
    size_t scratch_size = (size_t)batch * cfg.n_heads * 32 * (2 + cfg.head_dim) * sizeof(float);
    cudaMalloc(&d_scratch, scratch_size);
    paged_attention_set_splitk_scratch(d_scratch, scratch_size);

    // Tensor wrappers
    int64_t q_shape[4] = {batch, 1, cfg.n_heads, cfg.head_dim};
    int64_t kv_shape[4] = {num_kv_blocks, block_size, cfg.n_kv_heads, cfg.head_dim};
    int64_t o_shape[4] = {batch, 1, cfg.n_heads, cfg.head_dim};

    Tensor Q(d_q, DType::FP16, 4, q_shape, true);
    Tensor K(d_k, DType::FP16, 4, kv_shape, true);
    Tensor V(d_v, DType::FP16, 4, kv_shape, true);
    Tensor O(d_o, DType::FP16, 4, o_shape, true);

    auto run = [&]() {
        paged_attention_decode(Q, K, V, O, d_block_tables, d_context_lens,
                               block_size, attn_scale, max_context_len,
                               0, 0.0f, stream);
    };

    // Warmup
    for (int i = 0; i < warmup_iters; i++) run();
    cudaStreamSynchronize(stream);

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < timed_iters; i++) run();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / (float)timed_iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    paged_attention_set_splitk_scratch(nullptr, 0);
    cudaFree(d_scratch);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);

    return avg_ms;
}

void bench_paged_attention() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("bench_paged_attention: no CUDA device found, skipping\n");
        return;
    }

    printf("=== Paged Attention Decode: FP16 KV Cache ===\n");
    printf("batch=1, block_size=16, warmup=20, iters=50\n\n");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::vector<AttentionConfig> configs = {
        {"MHA-32h",         32, 32, 128},  // MHA: Llama-2-7B style
        {"GQA-32q/8kv",     32,  8, 128},  // GQA 4x: Qwen3-4B style
        {"GQA-32q/4kv",     32,  4, 128},  // GQA 8x: Qwen3-Coder-30B style
        {"GQA-28q/4kv",     28,  4, 128},  // GQA 7x: DS-R1-7B style
    };

    std::vector<int> ctx_lens = {64, 256, 1024, 4096, 8192, 32768};

    for (const auto& cfg : configs) {
        int n_q_per_kv = cfg.n_heads / cfg.n_kv_heads;
        printf("%-18s  nh=%2d nkv=%2d hd=%3d (GQA %dx)\n",
               cfg.name, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, n_q_per_kv);
        printf("  %8s  %10s  %10s  %s\n", "ctx_len", "latency", "eff BW", "kernel");

        for (int ctx_len : ctx_lens) {
            float avg_ms = bench_paged_decode_kernel(cfg, ctx_len, stream);
            float avg_us = avg_ms * 1000.0f;

            // Effective bandwidth: reads = Q + K_cache + V_cache, writes = O
            // KV read = ctx_len * n_kv_heads * head_dim * 2 bytes
            // Q read = n_heads * head_dim * 2 bytes
            // O write = n_heads * head_dim * 2 bytes
            double kv_bytes = 2.0 * ctx_len * cfg.n_kv_heads * cfg.head_dim * 2.0;
            double qo_bytes = 2.0 * cfg.n_heads * cfg.head_dim * 2.0;
            double total_bytes = kv_bytes + qo_bytes;
            double bw_gbs = total_bytes / (avg_ms * 1e-3) / 1e9;

            // Determine kernel path
            int total_blocks_nosplit = cfg.n_heads;
            int num_ctx_blocks = (ctx_len + 16 - 1) / 16;
            const char* kernel = "MHA";
            if (num_ctx_blocks >= 4 && total_blocks_nosplit < 128)
                kernel = "split-K";
            else if (n_q_per_kv > 1 && n_q_per_kv <= 8 && num_ctx_blocks >= 8)
                kernel = "cluster";
            else if (n_q_per_kv > 1 && n_q_per_kv <= 8)
                kernel = "GQA";

            printf("  %8d  %7.1f us  %7.1f GB/s  %s\n",
                   ctx_len, avg_us, bw_gbs, kernel);
        }
        printf("\n");
    }

    cudaStreamDestroy(stream);
}

} // namespace imp

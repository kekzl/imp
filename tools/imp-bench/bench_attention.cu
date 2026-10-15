#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "compute/attention_paged.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "core/dispatch_policy.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"

namespace imp {

struct AttentionConfig {
    const char* name;
    int n_heads;
    int n_kv_heads;
    int head_dim;
};

// Run attention with a specific kernel path and return avg time in ms
static float bench_kernel(const AttentionConfig& cfg, int seq_len, bool use_cutlass, cudaStream_t stream) {
    const int batch = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
    const int warmup_iters = 10;
    const int timed_iters = 30;

    const int64_t q_elems = (int64_t)batch * seq_len * cfg.n_heads * cfg.head_dim;
    const int64_t kv_elems = (int64_t)batch * seq_len * cfg.n_kv_heads * cfg.head_dim;

    const size_t q_bytes = q_elems * sizeof(__half);
    const size_t kv_bytes = kv_elems * sizeof(__half);
    const size_t o_bytes = q_elems * sizeof(__half);

    void *d_q, *d_k, *d_v, *d_o;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_q, q_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_k, kv_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_v, kv_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_o, o_bytes));

    // Fill with deterministic data
    {
        size_t max_elems = q_elems > kv_elems ? q_elems : kv_elems;
        std::vector<__half> h_buf(max_elems);
        for (int64_t i = 0; i < q_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 7 + 13) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_q, h_buf.data(), q_bytes, cudaMemcpyHostToDevice));
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 11 + 17) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_k, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice));
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 13 + 19) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_v, h_buf.data(), kv_bytes, cudaMemcpyHostToDevice));
    }
    IMP_CUDA_CHECK_LOG(cudaMemset(d_o, 0, o_bytes));

    int64_t q_shape[4] = {batch, seq_len, cfg.n_heads, cfg.head_dim};
    int64_t kv_shape[4] = {batch, seq_len, cfg.n_kv_heads, cfg.head_dim};
    int64_t o_shape[4] = {batch, seq_len, cfg.n_heads, cfg.head_dim};

    Tensor Q(d_q, QType::F16, 4, q_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, o_shape, true);

    // Default dispatch policy for the microbench — no overrides. Takes the
    // snapshot type rather than RuntimeConfig since F-10; the defaults are the
    // same because DispatchPolicy holds the very same section structs.
    const DispatchPolicy rcfg;

    // Select kernel path
    auto run_kernel = [&]() {
        if (use_cutlass) {
            // Use runtime dispatch (tries MXFP4 -> FP8 -> FP16 FMHA -> Blackwell WMMA)
            attention_prefill_dispatch(Q, K, V, O, scale, true, 0, 0.0f, stream, rcfg);
            return;
        }
        if (!flash_attention_blackwell(Q, K, V, O, scale, true, 0, 0.0f, stream)) {
            fprintf(stderr, "flash_attention_blackwell declined this config (#654) — nothing to bench\n");
            exit(1);
        }
    };

    // Warmup
    for (int i = 0; i < warmup_iters; i++)
        run_kernel();
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    // Timed
    cudaEvent_t start, stop;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&start));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&stop));

    IMP_CUDA_CHECK_LOG(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; i++)
        run_kernel();
    IMP_CUDA_CHECK_LOG(cudaEventRecord(stop, stream));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / (float)timed_iters;

    IMP_CUDA_CHECK_LOG(cudaEventDestroy(start));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(stop));
    IMP_CUDA_CHECK_LOG(cudaFree(d_q));
    IMP_CUDA_CHECK_LOG(cudaFree(d_k));
    IMP_CUDA_CHECK_LOG(cudaFree(d_v));
    IMP_CUDA_CHECK_LOG(cudaFree(d_o));

    return avg_ms;
}

bool bench_attention() {
    int device_count = 0;
    IMP_CUDA_CHECK_LOG(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("bench_attention: no CUDA device found, skipping\n");
        return false;  // measured nothing (#1584)
    }

    printf("=== Attention Prefill: CUTLASS FMHA vs WMMA Blackwell ===\n");
    printf("Causal, FP16, batch=1, warmup=10, iters=30\n\n");

    cudaStream_t stream = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamCreate(&stream));

    std::vector<AttentionConfig> configs = {
        {"Phi-4-mini", 24, 8, 128},      {"DS-R1-7B (Qwen2)", 28, 4, 128}, {"Qwen3-4B", 32, 8, 128},
        {"Qwen3-Coder-30B", 32, 4, 128}, {"DS-R1-14B", 40, 8, 128},        {"Llama-3-70B", 64, 8, 128},
    };

    std::vector<int> seq_lens = {512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (const auto& cfg : configs) {
        printf("%-24s  nh=%2d nkv=%2d hd=%3d (GQA %dx)\n", cfg.name, cfg.n_heads, cfg.n_kv_heads,
               cfg.head_dim, cfg.n_heads / cfg.n_kv_heads);
        printf("  %8s  %10s  %10s  %10s  %7s\n", "seq", "CUTLASS", "WMMA", "TFLOPS C/W", "speedup");

        for (int seq_len : seq_lens) {
            // Run CUTLASS first, then WMMA, then CUTLASS again — take best of two CUTLASS runs
            float ms_cutlass1 = bench_kernel(cfg, seq_len, true, stream);
            float ms_wmma = bench_kernel(cfg, seq_len, false, stream);
            float ms_cutlass2 = bench_kernel(cfg, seq_len, true, stream);

            float ms_cutlass = ms_cutlass1 < ms_cutlass2 ? ms_cutlass1 : ms_cutlass2;

            double flops = 4.0 * 1 * cfg.n_heads * (double)seq_len * (double)seq_len * cfg.head_dim;
            double tflops_c = flops / (ms_cutlass * 1e-3) / 1e12;
            double tflops_w = flops / (ms_wmma * 1e-3) / 1e12;
            float speedup = ms_wmma / ms_cutlass;

            printf("  %8d  %8.3f ms  %8.3f ms  %5.1f/%5.1f  %5.2fx%s\n", seq_len, ms_cutlass, ms_wmma,
                   tflops_c, tflops_w, speedup, speedup > 1.05f ? " <--" : "");
        }
        printf("\n");
    }

    IMP_CUDA_CHECK_LOG(cudaStreamDestroy(stream));

    // ---- FP8 vs MXFP4 FMHA comparison ----
    printf("=== FP8 FMHA vs MXFP4 FMHA (sm_120) ===\n");
    printf("Causal, batch=1, warmup=10, iters=30\n\n");

    cudaStream_t s;
    IMP_CUDA_CHECK_LOG(cudaStreamCreate(&s));

    std::vector<AttentionConfig> cfgs_sm120 = {
        {"Qwen3-4B", 32, 8, 128},
        {"Qwen3-8B", 28, 4, 128},
        {"GQA-MHA", 16, 16, 128},
        {"GQA-8x", 32, 4, 128},
    };
    std::vector<int> seqs = {128, 256, 512, 1024, 2048, 4096};

    for (const auto& c : cfgs_sm120) {
        printf("%-20s  nh=%2d nkv=%2d hd=%3d\n", c.name, c.n_heads, c.n_kv_heads, c.head_dim);
        printf("  %8s  %10s  %10s  %7s\n", "seq", "FP8", "MXFP4", "speedup");

        for (int seq : seqs) {
            const int batch = 1;
            const float sc = 1.0f / sqrtf((float)c.head_dim);
            const int warmup = 10;
            const int iters = 30;

            const int64_t q_elems = (int64_t)batch * seq * c.n_heads * c.head_dim;
            const int64_t kv_elems = (int64_t)batch * seq * c.n_kv_heads * c.head_dim;

            void *dq, *dk, *dv, *dofp8, *domx;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&dq, q_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&dk, kv_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&dv, kv_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&dofp8, q_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&domx, q_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMemset(dq, 0x3C, q_elems * sizeof(half)));  // ~1.0 in FP16
            IMP_CUDA_CHECK_LOG(cudaMemset(dk, 0x3C, kv_elems * sizeof(half)));
            IMP_CUDA_CHECK_LOG(cudaMemset(dv, 0x3C, kv_elems * sizeof(half)));

            int64_t qs[4] = {batch, seq, c.n_heads, c.head_dim};
            int64_t ks[4] = {batch, seq, c.n_kv_heads, c.head_dim};

            Tensor Q(dq, QType::F16, 4, qs, true);
            Tensor K(dk, QType::F16, 4, ks, true);
            Tensor V(dv, QType::F16, 4, ks, true);

            // FP8 FMHA
            float ms_fp8 = -1.0f;
            {
                Tensor O(dofp8, QType::F16, 4, qs, true);
                bool ran = true;
                for (int i = 0; i < warmup; i++)
                    ran = fmha_sm120_fp8_prefill(Q, K, V, O, sc, true, 0, 0.0f, s) && ran;
                if (!ran)
                    fprintf(stderr, "fp8 FMHA refused seq=%d: row reports -1\n", seq);
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
                cudaEvent_t t0, t1;
                IMP_CUDA_CHECK_LOG(cudaEventCreate(&t0));
                IMP_CUDA_CHECK_LOG(cudaEventCreate(&t1));
                IMP_CUDA_CHECK_LOG(cudaEventRecord(t0, s));
                // args checked in warmup
                for (int i = 0; i < iters; i++)
                    (void)fmha_sm120_fp8_prefill(Q, K, V, O, sc, true, 0, 0.0f, s);
                IMP_CUDA_CHECK_LOG(cudaEventRecord(t1, s));
                IMP_CUDA_CHECK_LOG(cudaEventSynchronize(t1));
                IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&ms_fp8, t0, t1));
                ms_fp8 = ran ? ms_fp8 / iters : -1.0f;
                IMP_CUDA_CHECK_LOG(cudaEventDestroy(t0));
                IMP_CUDA_CHECK_LOG(cudaEventDestroy(t1));
            }

            // MXFP4 FMHA
            float ms_mxfp4 = -1.0f;
            {
                Tensor O(domx, QType::F16, 4, qs, true);
                bool ran = true;
                for (int i = 0; i < warmup; i++)
                    ran = fmha_sm120_mxfp4_prefill(Q, K, V, O, sc, true, 0, 0.0f, s) && ran;
                if (!ran)
                    fprintf(stderr, "mxfp4 FMHA refused seq=%d: row reports -1\n", seq);
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(s));
                cudaEvent_t t0, t1;
                IMP_CUDA_CHECK_LOG(cudaEventCreate(&t0));
                IMP_CUDA_CHECK_LOG(cudaEventCreate(&t1));
                IMP_CUDA_CHECK_LOG(cudaEventRecord(t0, s));
                // args checked in warmup
                for (int i = 0; i < iters; i++)
                    (void)fmha_sm120_mxfp4_prefill(Q, K, V, O, sc, true, 0, 0.0f, s);
                IMP_CUDA_CHECK_LOG(cudaEventRecord(t1, s));
                IMP_CUDA_CHECK_LOG(cudaEventSynchronize(t1));
                IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&ms_mxfp4, t0, t1));
                ms_mxfp4 = ran ? ms_mxfp4 / iters : -1.0f;
                IMP_CUDA_CHECK_LOG(cudaEventDestroy(t0));
                IMP_CUDA_CHECK_LOG(cudaEventDestroy(t1));
            }

            float speedup = ms_fp8 / ms_mxfp4;
            printf("  %8d  %8.3f ms  %8.3f ms  %5.2fx%s\n", seq, ms_fp8, ms_mxfp4, speedup,
                   speedup > 1.03f ? " <--" : (speedup < 0.97f ? " REG" : ""));

            IMP_CUDA_CHECK_LOG(cudaFree(dq));
            IMP_CUDA_CHECK_LOG(cudaFree(dk));
            IMP_CUDA_CHECK_LOG(cudaFree(dv));
            IMP_CUDA_CHECK_LOG(cudaFree(dofp8));
            IMP_CUDA_CHECK_LOG(cudaFree(domx));
        }
        printf("\n");
    }
    IMP_CUDA_CHECK_LOG(cudaStreamDestroy(s));
    return true;
}

// Paged Attention Decode Benchmark: measures latency at various context lengths for MHA and
// GQA configs. Reports us, effective bandwidth, and whether split-K activated.

static float bench_paged_decode_kernel(const AttentionConfig& cfg, int ctx_len, cudaStream_t stream,
                                       int block_size = 16) {
    const int batch = 1;
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
    const int warmup_iters = 20;
    const int timed_iters = 50;

    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;
    const int max_context_len = ctx_len;

    // Q: [batch, 1, n_heads, head_dim]
    const int64_t q_elems = (int64_t)batch * 1 * cfg.n_heads * cfg.head_dim;
    // KV cache: [num_kv_blocks, block_size, n_kv_heads, head_dim] (paged layout)
    const int64_t kv_elems = (int64_t)num_kv_blocks * block_size * cfg.n_kv_heads * cfg.head_dim;
    // O: same shape as Q
    const int64_t o_elems = q_elems;

    void *d_q, *d_k, *d_v, *d_o;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_q, q_elems * sizeof(__half)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_k, kv_elems * sizeof(__half)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_v, kv_elems * sizeof(__half)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_o, o_elems * sizeof(__half)));

    // Fill with deterministic data
    {
        int64_t max_e = q_elems > kv_elems ? q_elems : kv_elems;
        std::vector<__half> h_buf(max_e);
        for (int64_t i = 0; i < q_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 7 + 13) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_q, h_buf.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 11 + 17) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_k, h_buf.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice));
        for (int64_t i = 0; i < kv_elems; i++)
            h_buf[i] = __float2half(0.01f * (float)((i * 13 + 19) % 1000 - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_v, h_buf.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice));
    }
    IMP_CUDA_CHECK_LOG(cudaMemset(d_o, 0, o_elems * sizeof(__half)));

    // Block tables: identity mapping [batch, max_num_blocks]
    int* d_block_tables;
    int* d_context_lens;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_tables, num_kv_blocks * sizeof(int)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_context_lens, batch * sizeof(int)));

    std::vector<int> h_bt(num_kv_blocks);
    for (int i = 0; i < num_kv_blocks; i++)
        h_bt[i] = i;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpy(d_block_tables, h_bt.data(), num_kv_blocks * sizeof(int), cudaMemcpyHostToDevice));
    int h_ctx = ctx_len;
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_context_lens, &h_ctx, sizeof(int), cudaMemcpyHostToDevice));

    // Split-K scratch buffer: the split count caps at 128 (workspace_sizes.cpp)
    void* d_scratch = nullptr;
    size_t scratch_size = (size_t)batch * cfg.n_heads * 128 * (2 + cfg.head_dim) * sizeof(float);
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scratch, scratch_size));
    paged_attention_set_splitk_scratch(d_scratch, scratch_size);

    // Tensor wrappers
    int64_t q_shape[4] = {batch, 1, cfg.n_heads, cfg.head_dim};
    int64_t kv_shape[4] = {num_kv_blocks, block_size, cfg.n_kv_heads, cfg.head_dim};
    int64_t o_shape[4] = {batch, 1, cfg.n_heads, cfg.head_dim};

    Tensor Q(d_q, QType::F16, 4, q_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, o_shape, true);

    auto run = [&]() {
        paged_attention_decode(Q, K, V, O, d_block_tables, d_context_lens, block_size, attn_scale,
                               max_context_len, 0, 0.0f, stream);
    };

    // Warmup
    for (int i = 0; i < warmup_iters; i++)
        run();
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    // Timed
    cudaEvent_t start, stop;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&start));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&stop));

    IMP_CUDA_CHECK_LOG(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; i++)
        run();
    IMP_CUDA_CHECK_LOG(cudaEventRecord(stop, stream));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / (float)timed_iters;

    IMP_CUDA_CHECK_LOG(cudaEventDestroy(start));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(stop));
    paged_attention_set_splitk_scratch(nullptr, 0);
    IMP_CUDA_CHECK_LOG(cudaFree(d_scratch));
    IMP_CUDA_CHECK_LOG(cudaFree(d_block_tables));
    IMP_CUDA_CHECK_LOG(cudaFree(d_context_lens));
    IMP_CUDA_CHECK_LOG(cudaFree(d_q));
    IMP_CUDA_CHECK_LOG(cudaFree(d_k));
    IMP_CUDA_CHECK_LOG(cudaFree(d_v));
    IMP_CUDA_CHECK_LOG(cudaFree(d_o));

    return avg_ms;
}

bool bench_paged_attention() {
    int device_count = 0;
    IMP_CUDA_CHECK_LOG(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("bench_paged_attention: no CUDA device found, skipping\n");
        return false;  // measured nothing (#1584)
    }

    printf("=== Paged Attention Decode: FP16 KV Cache ===\n");
    printf("batch=1, block_size=16, warmup=20, iters=50\n\n");

    cudaStream_t stream = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamCreate(&stream));

    std::vector<AttentionConfig> configs = {
        {"MHA-32h", 32, 32, 128},     // MHA: Llama-2-7B style
        {"GQA-32q/8kv", 32, 8, 128},  // GQA 4x: Qwen3-4B style
        {"GQA-32q/4kv", 32, 4, 128},  // GQA 8x: Qwen3-Coder-30B style
        {"GQA-28q/4kv", 28, 4, 128},  // GQA 7x: DS-R1-7B style
    };

    std::vector<int> ctx_lens = {64, 256, 1024, 4096, 8192, 32768};

    for (const auto& cfg : configs) {
        int n_q_per_kv = cfg.n_heads / cfg.n_kv_heads;
        printf("%-18s  nh=%2d nkv=%2d hd=%3d (GQA %dx)\n", cfg.name, cfg.n_heads, cfg.n_kv_heads,
               cfg.head_dim, n_q_per_kv);
        printf("  %8s  %10s  %10s  %s\n", "ctx_len", "latency", "eff BW", "kernel");

        for (int ctx_len : ctx_lens) {
            float avg_ms = bench_paged_decode_kernel(cfg, ctx_len, stream);
            float avg_us = avg_ms * 1000.0f;

            // Effective bandwidth: reads = Q + K_cache + V_cache, writes = O. KV read = ctx_len *
            // n_kv_heads * head_dim * 2 bytes; Q read and O write = n_heads * head_dim * 2 bytes.
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

            printf("  %8d  %7.1f us  %7.1f GB/s  %s\n", ctx_len, avg_us, bw_gbs, kernel);
        }
        printf("\n");
    }

    // Block-size sweep (AUDIT_ARCH_2026 B-5): same kernel at 16/32/64 tokens per block
    // (kv_cache.block_size values). The engine's auto rule (32 for n_kv_heads<=4, else 16) never
    // had this instrument before.
    printf("=== Paged Attention Decode: block-size sweep (FP16 KV) ===\n");
    printf("batch=1, warmup=20, iters=50\n\n");
    for (const auto& cfg : configs) {
        printf("%-18s  nh=%2d nkv=%2d hd=%3d\n", cfg.name, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        printf("  %8s  %4s  %10s  %10s\n", "ctx_len", "bs", "latency", "eff BW");
        for (int ctx_len : {4096, 32768, 131072}) {
            for (int bs : {16, 32, 64}) {
                float avg_ms = bench_paged_decode_kernel(cfg, ctx_len, stream, bs);
                double kv_bytes = 2.0 * ctx_len * cfg.n_kv_heads * cfg.head_dim * 2.0;
                double qo_bytes = 2.0 * cfg.n_heads * cfg.head_dim * 2.0;
                double bw_gbs = (kv_bytes + qo_bytes) / (avg_ms * 1e-3) / 1e9;
                printf("  %8d  %4d  %7.1f us  %7.1f GB/s\n", ctx_len, bs, avg_ms * 1000.0f, bw_gbs);
            }
        }
        printf("\n");
    }

    IMP_CUDA_CHECK_LOG(cudaStreamDestroy(stream));
    return true;
}

}  // namespace imp

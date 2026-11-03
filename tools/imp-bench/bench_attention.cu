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
#include <algorithm>
#include <vector>

#include "compute/attention_fmha_sm120.h"

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
            // Use runtime dispatch (tries FA2 -> FP8 -> FP16 FMHA -> Blackwell WMMA)
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

// Batched paged decode (serving shape): B rows x ctx tokens, shuffled block tables, R disjoint
// KV sets rotated per call so the working set (>= 512 MB) exceeds the 96 MB L2.
struct BatchedDecodeShape {
    const char* name;
    int n_heads, n_kv_heads, head_dim, sliding_window, batch, ctx;
};

// fp8: K/V as E4M3 bytes (paged_attention_decode_fp8, kv_scale 0.05), else F16.
static double bench_paged_decode_batched_kernel(const BatchedDecodeShape& s, bool fp8, cudaStream_t stream,
                                                uint64_t* out_hash) {
    constexpr int kBlock = 16;
    const size_t kv_elem_bytes = fp8 ? 1 : sizeof(__half);
    const int blocks_per_row = (s.ctx + kBlock - 1) / kBlock;
    const int64_t slot_elems = static_cast<int64_t>(kBlock) * s.n_kv_heads * s.head_dim;
    const double call_bytes = 2.0 * s.batch * s.ctx * s.n_kv_heads * s.head_dim * kv_elem_bytes;
    const int n_sets = std::max(1, static_cast<int>((512.0 * 1024 * 1024) / call_bytes + 0.999));
    const int n_blocks = n_sets * s.batch * blocks_per_row;
    const int64_t kv_elems = static_cast<int64_t>(n_blocks) * slot_elems;
    const int64_t q_elems = static_cast<int64_t>(s.batch) * s.n_heads * s.head_dim;

    __half *d_q = nullptr, *d_o = nullptr;
    void *d_k = nullptr, *d_v = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_q, q_elems * sizeof(__half)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_o, q_elems * sizeof(__half)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_k, kv_elems * kv_elem_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_v, kv_elems * kv_elem_bytes));
    {
        std::vector<__half> h(static_cast<size_t>(std::max<int64_t>(q_elems, slot_elems * 64)));
        for (size_t i = 0; i < h.size(); i++)
            h[i] = __float2half(0.01f * static_cast<float>(static_cast<int>((i * 11 + 17) % 1000) - 500));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_q, h.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
        // E4M3 bytes with magnitude codes 0x00..0x4F (0 .. 3.75) and alternating sign: no NaN code.
        std::vector<uint8_t> h8(fp8 ? h.size() : 0);
        for (size_t i = 0; i < h8.size(); i++)
            h8[i] = static_cast<uint8_t>(((i * 11 + 17) % 0x50) | ((i * 7) & 1) << 7);
        const void* src = fp8 ? static_cast<const void*>(h8.data()) : static_cast<const void*>(h.data());
        const int64_t chunk = static_cast<int64_t>(h.size());
        for (int64_t off = 0; off < kv_elems; off += chunk) {
            const int64_t n = std::min(chunk, kv_elems - off);
            const size_t off_b = static_cast<size_t>(off) * kv_elem_bytes;
            IMP_CUDA_CHECK_LOG(
                cudaMemcpy(static_cast<char*>(d_k) + off_b, src, n * kv_elem_bytes, cudaMemcpyHostToDevice));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpy(static_cast<char*>(d_v) + off_b, src, n * kv_elem_bytes, cudaMemcpyHostToDevice));
        }
    }
    // Per set: a disjoint, shuffled slice of the physical blocks (LCG permutation, fixed seed).
    std::vector<int> perm(n_blocks);
    for (int i = 0; i < n_blocks; i++)
        perm[i] = i;
    uint32_t rng = 12345u;
    for (int i = n_blocks - 1; i > 0; i--) {
        rng = (rng * 1664525u) + 1013904223u;
        std::swap(perm[i], perm[rng % static_cast<uint32_t>(i + 1)]);
    }
    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_bt, n_blocks * sizeof(int)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_ctx, s.batch * sizeof(int)));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_bt, perm.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> h_ctx(s.batch, s.ctx);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_ctx, h_ctx.data(), s.batch * sizeof(int), cudaMemcpyHostToDevice));

    void* d_scratch = nullptr;
    const size_t scratch_size = static_cast<size_t>(s.batch) * s.n_heads * 128 * (2 + s.head_dim) *
                                sizeof(float);
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scratch, scratch_size));
    paged_attention_set_splitk_scratch(d_scratch, scratch_size);

    int64_t q_shape[4] = {s.batch, 1, s.n_heads, s.head_dim};
    int64_t kv_shape[4] = {n_blocks, kBlock, s.n_kv_heads, s.head_dim};
    Tensor Q(d_q, QType::F16, 4, q_shape, true);
    const QType kv_type = fp8 ? QType::FP8_E4M3 : QType::F16;
    Tensor K(d_k, kv_type, 4, kv_shape, true);
    Tensor V(d_v, kv_type, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, q_shape, true);
    const float scale = 1.0f / sqrtf(static_cast<float>(s.head_dim));
    auto run = [&](int it) {
        const int set = it % n_sets;
        const int* bt = d_bt + (static_cast<int64_t>(set) * s.batch * blocks_per_row);
        if (fp8)
            paged_attention_decode_fp8(Q, K, V, O, bt, d_ctx, kBlock, scale, 0.05f, s.ctx, s.sliding_window,
                                       0.0f, stream, blocks_per_row);
        else
            paged_attention_decode(Q, K, V, O, bt, d_ctx, kBlock, scale, s.ctx, s.sliding_window, 0.0f,
                                   stream, blocks_per_row);
    };
    // Warm > 1 s (idle downclock), then 400 timed calls.
    cudaEvent_t e0, e1;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&e0));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&e1));
    float warm_ms = 0.0f;
    for (int it = 0; warm_ms < 1200.0f; it += 50) {
        IMP_CUDA_CHECK_LOG(cudaEventRecord(e0, stream));
        for (int j = 0; j < 50; j++)
            run(it + j);
        IMP_CUDA_CHECK_LOG(cudaEventRecord(e1, stream));
        IMP_CUDA_CHECK_LOG(cudaEventSynchronize(e1));
        float ms = 0.0f;
        IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&ms, e0, e1));
        warm_ms += ms;
    }
    // Timed: 100 calls captured into one graph (the decode step replays graphs; eager launches on
    // WSL2 read up to 3x slow on sub-30 us shapes), replayed 4 times.
    constexpr int kGraphCalls = 100;
    constexpr int kReplays = 4;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int it = 0; it < kGraphCalls; it++)
        run(it);
    IMP_CUDA_CHECK_LOG(cudaStreamEndCapture(stream, &graph));
    IMP_CUDA_CHECK_LOG(cudaGraphInstantiate(&graph_exec, graph, 0));
    IMP_CUDA_CHECK_LOG(cudaGraphLaunch(graph_exec, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(e0, stream));
    for (int r = 0; r < kReplays; r++)
        IMP_CUDA_CHECK_LOG(cudaGraphLaunch(graph_exec, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(e1, stream));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(e1));
    constexpr int kIters = kGraphCalls * kReplays;
    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, e0, e1));
    IMP_CUDA_CHECK_LOG(cudaGraphExecDestroy(graph_exec));
    IMP_CUDA_CHECK_LOG(cudaGraphDestroy(graph));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(e0));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(e1));
    // FNV-1a over the last call's O: equal hashes across two builds = bit-identical output.
    {
        std::vector<__half> h_o(static_cast<size_t>(q_elems));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(h_o.data(), d_o, q_elems * sizeof(__half), cudaMemcpyDeviceToHost));
        uint64_t hash = 1469598103934665603ull;
        const auto* bytes = reinterpret_cast<const unsigned char*>(h_o.data());
        for (size_t i = 0; i < h_o.size() * sizeof(__half); i++)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
        *out_hash = hash;
    }

    paged_attention_set_splitk_scratch(nullptr, 0);
    IMP_CUDA_CHECK_LOG(cudaFree(d_scratch));
    IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
    IMP_CUDA_CHECK_LOG(cudaFree(d_ctx));
    IMP_CUDA_CHECK_LOG(cudaFree(d_q));
    IMP_CUDA_CHECK_LOG(cudaFree(d_o));
    IMP_CUDA_CHECK_LOG(cudaFree(d_k));
    IMP_CUDA_CHECK_LOG(cudaFree(d_v));
    return 1000.0 * total_ms / kIters;  // us per call
}

bool bench_paged_attention_batched() {
    int device_count = 0;
    IMP_CUDA_CHECK_LOG(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("bench_paged_attention_batched: no CUDA device found, skipping\n");
        return false;
    }
    cudaStream_t stream = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamCreate(&stream));
    const BatchedDecodeShape shapes[] = {
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 16, 1131},      {"qwen3-8b 32/8/128", 32, 8, 128, 0, 32, 1131},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 8, 1131},       {"qwen3-8b 32/8/128", 32, 8, 128, 0, 4, 1131},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 1131},       {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 640},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 2048},       {"qwen3-8b 32/8/128", 32, 8, 128, 0, 2, 1131},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 16, 4096},      {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 8192},
        {"gemma4-swa 16/8/256", 16, 8, 256, 1024, 16, 1131}, {"gemma4-glb 16/2/512", 16, 2, 512, 0, 16, 1131},
        {"gemma4-glb 16/2/512", 16, 2, 512, 0, 1, 8192},     {"qwen3-14b 40/8/128", 40, 8, 128, 0, 16, 1131},
        {"qwen3-30b 32/4/128", 32, 4, 128, 0, 16, 1131},     {"qwen3-30b 32/4/128", 32, 4, 128, 0, 4, 1131},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 12, 1131},      {"qwen3-8b 32/8/128", 32, 8, 128, 0, 4, 4096},
        {"qwen3-14b 40/8/128", 40, 8, 128, 0, 32, 1131},     {"qwen3-14b 40/8/128", 40, 8, 128, 0, 4, 1131},
        {"qwen3-14b 40/8/128", 40, 8, 128, 0, 1, 1131},      {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 128},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 1, 256},        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 4, 256},
        {"qwen3-8b 32/8/128", 32, 8, 128, 0, 16, 256},       {"qwen3-8b 32/8/128", 32, 8, 128, 0, 4, 512},
    };
    for (const bool fp8 : {false, true}) {
        printf("=== Paged Attention Decode: batched %s KV, block 16, L2-rotated ===\n",
               fp8 ? "FP8 E4M3" : "FP16");
        printf("%-22s %5s %6s %10s %10s  %s\n", "shape", "batch", "ctx", "us/call", "KV GB/s", "O hash");
        for (const auto& s : shapes) {
            uint64_t hash = 0;
            const double us = bench_paged_decode_batched_kernel(s, fp8, stream, &hash);
            const int read_ctx = (s.sliding_window > 0 && s.ctx > s.sliding_window) ? s.sliding_window
                                                                                    : s.ctx;
            const double bytes = 2.0 * s.batch * read_ctx * s.n_kv_heads * s.head_dim *
                                 (fp8 ? 1 : sizeof(__half));
            printf("%-22s %5d %6d %10.2f %10.1f  %016llx\n", s.name, s.batch, s.ctx, us,
                   bytes / (us * 1e-6) / 1e9, static_cast<unsigned long long>(hash));
        }
    }
    IMP_CUDA_CHECK_LOG(cudaStreamDestroy(stream));
    return true;
}

}  // namespace imp

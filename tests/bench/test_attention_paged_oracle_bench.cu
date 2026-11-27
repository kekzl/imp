// Timing cases of tests/test_attention_paged_oracle.cu, built into test-bench (make test-bench), not into the
// default GPU run.

#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include "core/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_attention_paged_oracle_util.cuh"

namespace imp {

// F16 paged decode microbench (default KV dtype for Q8_0 GGUFs / unquantized checkpoints):
// IMP_F16_BENCH_{BATCH=32,CTX=1100,HEADS=32,KV_HEADS=8,HD=128}. Split-K scratch registered
// like the engine. Timing only.
TEST(PagedF16Decode, ServingShapeMicrobench) {
    auto env_int = [](const char* k, int d) {
        const char* v = std::getenv(k);
        return v ? std::atoi(v) : d;
    };
    const int batch = env_int("IMP_F16_BENCH_BATCH", 32);
    const int kv_len = env_int("IMP_F16_BENCH_CTX", 1100);
    const int n_heads = env_int("IMP_F16_BENCH_HEADS", 32);
    const int n_kv_heads = env_int("IMP_F16_BENCH_KV_HEADS", 8);
    const int head_dim = env_int("IMP_F16_BENCH_HD", 128);
    process_diag_set_paged_f16_multitok(env_int("IMP_F16_BENCH_MULTITOK", 4));
    process_diag_set_paged_f16_hpc(env_int("IMP_F16_BENCH_HPC", 0));
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int blocks_per_seq = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks = batch * blocks_per_seq;
    const size_t q_elems = (size_t)batch * n_heads * head_dim;
    const size_t kv_elems = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<half> Qh(q_elems), Kc(kv_elems), Vc(kv_elems);
    lcg_fill(Qh, 0x1F16u, 2.0f);
    lcg_fill(Kc, 0x2F16u, 1.0f);
    lcg_fill(Vc, 0x3F16u, 1.0f);
    std::vector<int> bt((size_t)batch * blocks_per_seq), ctx(batch, kv_len);
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < blocks_per_seq; i++)
            bt[(size_t)b * blocks_per_seq + i] = b * blocks_per_seq + i;
    void* d_q = up(Qh.data(), q_elems * sizeof(half));
    void* d_k = up(Kc.data(), kv_elems * sizeof(half));
    void* d_v = up(Vc.data(), kv_elems * sizeof(half));
    int* d_bt = (int*)up(bt.data(), bt.size() * sizeof(int));
    int* d_ctx = (int*)up(ctx.data(), ctx.size() * sizeof(int));
    void* d_o = nullptr;
    cudaMalloc(&d_o, q_elems * sizeof(half));
    Tensor Q = f16_tensor(d_q, {batch, 1, n_heads, head_dim});
    Tensor K = f16_tensor(d_k, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor V = f16_tensor(d_v, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor O = f16_tensor(d_o, {batch, 1, n_heads, head_dim});
    void* d_scratch = nullptr;
    const size_t scratch_bytes = (size_t)64 << 20;
    cudaMalloc(&d_scratch, scratch_bytes);
    auto launch = [&]() {
        paged_attention_decode({.Q = &Q,
                                .K_cache = &K,
                                .V_cache = &V,
                                .O = &O,
                                .block_tables = d_bt,
                                .context_lens = d_ctx,
                                .block_size = BLOCK_SIZE,
                                .scale = scale,
                                .max_context_len = kv_len,
                                .stream = stream,
                                .max_blocks_per_seq = blocks_per_seq,
                                .splitk = {.ptr = d_scratch, .size = scratch_bytes}});
    };
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float warm_ms = 0.0f;
    for (int round = 0; round < 100000 && warm_ms < 1000.0f; round++) {
        cudaEventRecord(t0, stream);
        for (int i = 0; i < 100; i++)
            launch();
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        warm_ms += ms;
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "F16 paged decode launch";
    const int iters = 200;
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        launch();
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double us = 1000.0 * ms / iters;
    const double bytes = 2.0 * (double)batch * kv_len * n_kv_heads * head_dim * sizeof(half);
    printf(
        "PagedF16Decode: multitok=%d hpc=%d batch=%d ctx=%d heads=%d/%d hd=%d: %.1f us/launch, KV %.1f MB, "
        "%.0f GB/s (warm %.2f "
        "s)\n",
        process_diag_paged_f16_multitok(), process_diag_paged_f16_hpc(), batch, kv_len, n_heads, n_kv_heads,
        head_dim, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9, warm_ms / 1000.0);
    std::vector<float> Oh = read_o(d_o, q_elems);
    size_t nonfinite = 0;
    for (float h : Oh)
        if (!std::isfinite(h))
            nonfinite++;
    EXPECT_EQ(nonfinite, 0u);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_o);
    cudaFree(d_scratch);
    cudaStreamDestroy(stream);
}

// NVFP4 paged decode microbench (hybrid's default decode attention): IMP_NVFP4_BENCH_
// {BATCH=1,CTX=77000,HEADS=24,KV_HEADS=4,HD=256,RING=KV copies, default >= 400 MB}. Random packed
// nibbles + finite UE4M3 scale bytes; prints us/launch and bytes/sec. Finite-output check only.
TEST(PagedNvfp4Decode, LongContextMicrobench) {
    auto env_int = [](const char* k, int d) {
        const char* v = std::getenv(k);
        return v ? std::atoi(v) : d;
    };
    const int batch = env_int("IMP_NVFP4_BENCH_BATCH", 1);
    const int kv_len = env_int("IMP_NVFP4_BENCH_CTX", 77000);
    const int n_heads = env_int("IMP_NVFP4_BENCH_HEADS", 24);
    const int n_kv_heads = env_int("IMP_NVFP4_BENCH_KV_HEADS", 4);
    const int head_dim = env_int("IMP_NVFP4_BENCH_HD", 256);
    const int half_hd = head_dim / 2;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int blocks_per_seq = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks = batch * blocks_per_seq;
    const size_t q_elems = (size_t)batch * n_heads * head_dim;
    const size_t kv_bytes = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads * half_hd;
    const size_t sc_bytes = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads * (head_dim / 16);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<half> Qh(q_elems);
    lcg_fill(Qh, 0x4F9Cu, 2.0f);
    std::vector<uint8_t> Kq(kv_bytes), Vq(kv_bytes), Ks(sc_bytes), Vs(sc_bytes);
    uint32_t s = 0x7654321u;
    for (size_t i = 0; i < kv_bytes; i++) {
        s = s * 1664525u + 1013904223u;
        Kq[i] = (uint8_t)(s >> 24);
        s = s * 1664525u + 1013904223u;
        Vq[i] = (uint8_t)(s >> 24);
    }
    for (size_t i = 0; i < sc_bytes; i++) {
        s = s * 1664525u + 1013904223u;
        Ks[i] = (uint8_t)(0x30 + ((s >> 24) & 0x0F));  // e4m3 codes 0x30..0x3F: 0.25 .. 1.875
        s = s * 1664525u + 1013904223u;
        Vs[i] = (uint8_t)(0x30 + ((s >> 24) & 0x0F));
    }
    // R KV copies, launch i reads copy i % R. Default: >= 400 MB so every launch reads DRAM, not the
    // 96 MB L2 (77k 16/2 = 44 MB fits); IMP_NVFP4_BENCH_RING=1 = the L2-resident reading of #2440.
    const size_t copy_bytes = 2 * (kv_bytes + sc_bytes);
    const int ring = std::max(1, env_int("IMP_NVFP4_BENCH_RING", (int)((400u << 20) / copy_bytes + 1)));
    std::vector<int> bt((size_t)ring * batch * blocks_per_seq), ctx(batch, kv_len);
    for (int r = 0; r < ring; r++)
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < blocks_per_seq; i++)
                bt[((size_t)r * batch + b) * blocks_per_seq + i] = r * num_blocks + b * blocks_per_seq + i;
    void* d_q = up(Qh.data(), q_elems * sizeof(half));
    void *d_k = nullptr, *d_v = nullptr, *d_ks = nullptr, *d_vs = nullptr;
    cudaMalloc(&d_k, ring * kv_bytes);
    cudaMalloc(&d_v, ring * kv_bytes);
    cudaMalloc(&d_ks, ring * sc_bytes);
    cudaMalloc(&d_vs, ring * sc_bytes);
    for (int r = 0; r < ring; r++) {
        cudaMemcpy((uint8_t*)d_k + r * kv_bytes, Kq.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy((uint8_t*)d_v + r * kv_bytes, Vq.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy((uint8_t*)d_ks + r * sc_bytes, Ks.data(), sc_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy((uint8_t*)d_vs + r * sc_bytes, Vs.data(), sc_bytes, cudaMemcpyHostToDevice);
    }
    int* d_bt_ring = (int*)up(bt.data(), bt.size() * sizeof(int));
    int* d_bt = d_bt_ring;
    int ring_pos = 0;
    int* d_ctx = (int*)up(ctx.data(), ctx.size() * sizeof(int));
    void* d_o = nullptr;
    cudaMalloc(&d_o, q_elems * sizeof(half));
    Tensor Q = f16_tensor(d_q, {batch, 1, n_heads, head_dim});
    Tensor K = raw_tensor(d_k, QType::FP4_E2M1, {ring * num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});
    Tensor V = raw_tensor(d_v, QType::FP4_E2M1, {ring * num_blocks, BLOCK_SIZE, n_kv_heads, half_hd});
    Tensor O = f16_tensor(d_o, {batch, 1, n_heads, head_dim});
    // The engine registers a split-K scratch at init; batch-1 long context
    // takes the split-K route only with one (24 CTAs walk 77k tokens
    // serially otherwise, 11.6 ms per launch measured 2026-09-03).
    process_diag_set_paged_nvfp4_multitok(env_int("IMP_NVFP4_BENCH_MULTITOK", 4));
    process_diag_set_paged_nvfp4_hpc(env_int("IMP_NVFP4_BENCH_HPC", 0));
    process_diag_set_paged_nvfp4_mma(env_int("IMP_NVFP4_BENCH_MMA", 64));
    void* d_scratch = nullptr;
    const size_t scratch_bytes = (size_t)64 << 20;
    cudaMalloc(&d_scratch, scratch_bytes);
    auto launch = [&]() {
        d_bt = d_bt_ring + (size_t)ring_pos * batch * blocks_per_seq;
        ring_pos = (ring_pos + 1) % ring;
        paged_attention_decode_nvfp4({.Q = &Q,
                                      .K_cache = &K,
                                      .V_cache = &V,
                                      .O = &O,
                                      .K_scales = (const uint8_t*)d_ks,
                                      .V_scales = (const uint8_t*)d_vs,
                                      .block_tables = d_bt,
                                      .context_lens = d_ctx,
                                      .block_size = BLOCK_SIZE,
                                      .scale = scale,
                                      .max_context_len = kv_len,
                                      .stream = stream,
                                      .max_blocks_per_seq = blocks_per_seq,
                                      .splitk = {.ptr = d_scratch, .size = scratch_bytes}});
    };
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // warm > 1 s of wall time (idle downclock ramps in ~1 s on this box).
    float warm_ms = 0.0f;
    for (int round = 0; round < 100000 && warm_ms < 1000.0f; round++) {
        cudaEventRecord(t0, stream);
        for (int i = 0; i < 100; i++)
            launch();
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        warm_ms += ms;
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "NVFP4 paged decode launch";
    // Median of 7 timed batches: single 3 ms windows scattered 2-3x between runs.
    const int iters = (kv_len >= 20000) ? 100 : 200;
    std::vector<float> batch_ms;
    for (int rep = 0; rep < 7; rep++) {
        cudaEventRecord(t0, stream);
        for (int i = 0; i < iters; i++)
            launch();
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        batch_ms.push_back(ms);
    }
    std::sort(batch_ms.begin(), batch_ms.end());
    const double us = 1000.0 * batch_ms[3] / iters;
    const double bytes = 2.0 * (double)batch * kv_len * n_kv_heads * (half_hd + head_dim / 16);
    printf(
        "PagedNvfp4Decode: ring=%d multitok=%d mma=%d batch=%d ctx=%d heads=%d/%d hd=%d: %.1f "
        "us/launch, KV+scales "
        "%.1f MB, "
        "%.0f GB/s (warm %.2f s)\n",
        ring, process_diag_paged_nvfp4_multitok(), process_diag_paged_nvfp4_mma(), batch, kv_len, n_heads,
        n_kv_heads, head_dim, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9, warm_ms / 1000.0);
    std::vector<float> Oh = read_o(d_o, q_elems);
    size_t nonfinite = 0;
    for (float h : Oh)
        if (!std::isfinite(h))
            nonfinite++;
    EXPECT_EQ(nonfinite, 0u);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ks);
    cudaFree(d_vs);
    cudaFree(d_bt_ring);
    cudaFree(d_ctx);
    cudaFree(d_o);
    process_diag_set_paged_nvfp4_mma(64);
    cudaFree(d_scratch);
    cudaStreamDestroy(stream);
}

// Serving-shape microbench for FP8 paged decode: on the dense Qwen3-14B profile at 32
// streams x ~1.1k context this kernel was 33.4% of kernel time (41800 launches x 190us, ~25%
// DRAM bandwidth). IMP_ATTN_BENCH_{BATCH=32,CTX=1100,HEADS=40,KV_HEADS=8}; asserts finite
// output only, numerics covered by the oracle above.
TEST(PagedFp8Decode, ServingShapeMicrobench) {
    auto env_int = [](const char* k, int d) {
        const char* v = std::getenv(k);
        return v ? std::atoi(v) : d;
    };
    const int batch = env_int("IMP_ATTN_BENCH_BATCH", 32);
    const int kv_len = env_int("IMP_ATTN_BENCH_CTX", 1100);
    const int n_heads = env_int("IMP_ATTN_BENCH_HEADS", 40);
    const int n_kv_heads = env_int("IMP_ATTN_BENCH_KV_HEADS", 8);
    const int head_dim = 128;
    const int multitok = env_int("IMP_ATTN_BENCH_MULTITOK", 4);  // 1 = plain kernel, 4 = multitok
    const int hpc = env_int("IMP_ATTN_BENCH_HPC", 0);  // 0 = auto grouping, -1 = per-head four-token
    process_diag_set_paged_fp8_multitok(multitok);
    process_diag_set_paged_fp8_hpc(hpc);
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int blocks_per_seq = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks = batch * blocks_per_seq;
    const size_t q_elems = (size_t)batch * n_heads * head_dim;
    const size_t kv_bytes = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<half> Qh(q_elems);
    lcg_fill(Qh, 0xBE9Cu, 2.0f);
    // FP8 bytes straight from an LCG, restricted to small-magnitude codes so
    // the softmax stays finite: keep exponent bits low (mask 0x3F keeps
    // |x| <= 1.875 in e4m3).
    std::vector<uint8_t> Kq(kv_bytes), Vq(kv_bytes);
    uint32_t s = 0x1234567u;
    for (size_t i = 0; i < kv_bytes; i++) {
        s = s * 1664525u + 1013904223u;
        Kq[i] = (uint8_t)((s >> 24) & 0x3F);
        s = s * 1664525u + 1013904223u;
        Vq[i] = (uint8_t)((s >> 24) & 0x3F);
    }
    std::vector<int> bt((size_t)batch * blocks_per_seq), ctx(batch, kv_len);
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < blocks_per_seq; i++)
            bt[(size_t)b * blocks_per_seq + i] = b * blocks_per_seq + i;
    void* d_q = up(Qh.data(), q_elems * sizeof(half));
    void* d_k = up(Kq.data(), kv_bytes);
    void* d_v = up(Vq.data(), kv_bytes);
    int* d_bt = (int*)up(bt.data(), bt.size() * sizeof(int));
    int* d_ctx = (int*)up(ctx.data(), ctx.size() * sizeof(int));
    void* d_o = nullptr;
    cudaMalloc(&d_o, q_elems * sizeof(half));
    Tensor Q = f16_tensor(d_q, {batch, 1, n_heads, head_dim});
    Tensor K = raw_tensor(d_k, QType::FP8_E4M3, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor V = raw_tensor(d_v, QType::FP8_E4M3, {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor O = f16_tensor(d_o, {batch, 1, n_heads, head_dim});

    void* d_scratch = nullptr;
    const size_t scratch_bytes = (size_t)64 << 20;
    cudaMalloc(&d_scratch, scratch_bytes);
    auto launch = [&]() {
        paged_attention_decode_fp8({.Q = &Q,
                                    .K_cache = &K,
                                    .V_cache = &V,
                                    .O = &O,
                                    .block_tables = d_bt,
                                    .context_lens = d_ctx,
                                    .block_size = BLOCK_SIZE,
                                    .scale = scale,
                                    .kv_scale = 1.0f,
                                    .max_context_len = kv_len,
                                    .stream = stream,
                                    .max_blocks_per_seq = blocks_per_seq,
                                    .splitk = {.ptr = d_scratch, .size = scratch_bytes}});
    };
    // Warm >1 s (idle downclock ramps in ~1 s on this box): batches of 500
    // launches until a second has passed, whatever the kernel's speed.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float warm_ms = 0.0f;
    while (warm_ms < 1000.0f) {
        cudaEventRecord(t0, stream);
        for (int i = 0; i < 500; i++)
            launch();
        cudaEventRecord(t1, stream);
        cudaEventSynchronize(t1);
        float batch_ms = 0.0f;
        cudaEventElapsedTime(&batch_ms, t0, t1);
        warm_ms += batch_ms;
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "FP8 paged decode launch";
    const int iters = 200;
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        launch();
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    const double us = 1000.0 * ms / iters;
    const double bytes = 2.0 * (double)batch * kv_len * n_kv_heads * head_dim;  // K + V read once
    printf(
        "PagedFp8Decode serving shape: multitok=%d hpc=%d batch=%d ctx=%d heads=%d/%d hd=%d: %.1f us/launch, "
        "KV %.1f MB, %.0f GB/s (warm %.2f s)\n",
        multitok, hpc, batch, kv_len, n_heads, n_kv_heads, head_dim, us, bytes / 1e6,
        bytes / (us * 1e-6) / 1e9, warm_ms / 1000.0);
    process_diag_set_paged_fp8_hpc(0);
    std::vector<float> Oh = read_o(d_o, q_elems);
    size_t nonfinite = 0;
    for (float h : Oh)
        if (!std::isfinite(h))
            nonfinite++;
    EXPECT_EQ(nonfinite, 0u);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_o);
    cudaFree(d_scratch);
    cudaStreamDestroy(stream);
}
}  // namespace imp

// Correctness sweep: compares attention_tiled_streaming_prefill against
// attention_cublas_prefill on the same inputs. Test passes if max-abs-err
// < 5e-3 and max-rel-err < 1e-2 (matches FMHA-test gate).

#include "compute/attention_tiled_streaming.h"
#include "compute/attention_cublas.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace {

struct AttnConfig {
    int seq;
    int n_heads;
    int n_kv_heads;
    int head_dim;
};

void fill_fp16_deterministic(__half* d_ptr, size_t n) {
    std::vector<__half> host(n);
    for (size_t i = 0; i < n; ++i) {
        float v = (static_cast<float>((i * 2654435761u) % 1024u) / 1024.0f) - 0.5f;
        host[i] = __float2half(v * 0.125f);
    }
    cudaMemcpy(d_ptr, host.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

void run_one_shape(const AttnConfig& c) {
    using imp::Tensor;
    using imp::QType;

    const int seq = c.seq;
    const int nh = c.n_heads;
    const int nkv = c.n_kv_heads;
    const int hd = c.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    const size_t q_elems = static_cast<size_t>(seq) * nh * hd;
    const size_t kv_elems = static_cast<size_t>(seq) * nkv * hd;

    __half *d_Q, *d_K, *d_V, *d_O_cublas, *d_O_track;
    cudaMalloc(&d_Q, q_elems * sizeof(__half));
    cudaMalloc(&d_K, kv_elems * sizeof(__half));
    cudaMalloc(&d_V, kv_elems * sizeof(__half));
    cudaMalloc(&d_O_cublas, q_elems * sizeof(__half));
    cudaMalloc(&d_O_track, q_elems * sizeof(__half));

    fill_fp16_deterministic(d_Q, q_elems);
    fill_fp16_deterministic(d_K, kv_elems);
    fill_fp16_deterministic(d_V, kv_elems);

    // cuBLAS reference
    {
        const int64_t s_fp32_elems = static_cast<int64_t>(nh) * seq * seq;
        __half* d_S = nullptr;
        cudaMalloc(&d_S, 2 * s_fp32_elems * sizeof(__half));

        int64_t qkv_2d[2] = {seq, nh * hd};
        int64_t kv_2d[2] = {seq, nkv * hd};
        int64_t s_shape[3] = {nh, seq, 2 * seq};
        Tensor Q(d_Q, QType::F16, 2, qkv_2d, true);
        Tensor K(d_K, QType::F16, 2, kv_2d, true);
        Tensor V(d_V, QType::F16, 2, kv_2d, true);
        Tensor O(d_O_cublas, QType::F16, 2, qkv_2d, true);
        Tensor S(d_S, QType::F16, 3, s_shape, true);
        imp::attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale,
                                       /*causal=*/true, /*softcap=*/0.0f,
                                       /*q_offset=*/0, nullptr,
                                       /*sliding_window=*/0);
        cudaFree(d_S);
    }

    // Track E (under test)
    {
        int64_t q_4d[4] = {1, seq, nh, hd};
        int64_t kv_4d[4] = {1, seq, nkv, hd};
        Tensor Q(d_Q, QType::F16, 4, q_4d, true);
        Tensor K(d_K, QType::F16, 4, kv_4d, true);
        Tensor V(d_V, QType::F16, 4, kv_4d, true);
        Tensor O(d_O_track, QType::F16, 4, q_4d, true);
        bool ok = imp::attention_tiled_streaming_prefill(
            Q, K, V, O, scale, /*causal=*/true, /*sliding_window=*/0,
            /*softcap=*/0.0f, /*q_offset=*/0, nullptr);
        if (!ok) {
            GTEST_SKIP() << "Track E declined this config (expected during ramp-up)";
        }
    }

    cudaDeviceSynchronize();

    // Compare
    std::vector<__half> h_cublas(q_elems), h_track(q_elems);
    cudaMemcpy(h_cublas.data(), d_O_cublas, q_elems * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_track.data(), d_O_track, q_elems * sizeof(__half),
               cudaMemcpyDeviceToHost);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < q_elems; ++i) {
        float a = __half2float(h_cublas[i]);
        float b = __half2float(h_track[i]);
        float abs_e = std::abs(a - b);
        float rel_e = abs_e / (std::abs(a) + 1e-6f);
        if (abs_e > max_abs) max_abs = abs_e;
        if (rel_e > max_rel) max_rel = rel_e;
    }

    EXPECT_LT(max_abs, 5e-3f) << "seq=" << seq << " nh=" << nh << " hd=" << hd;
    EXPECT_LT(max_rel, 1e-2f) << "seq=" << seq << " nh=" << nh << " hd=" << hd;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_cublas); cudaFree(d_O_track);
}

}  // namespace

TEST(TrackE_Correctness, Qwen3_seq512_hd128) { run_one_shape({512, 32, 8, 128}); }
TEST(TrackE_Correctness, Qwen3_seq2048_hd128) { run_one_shape({2048, 32, 8, 128}); }
TEST(TrackE_Correctness, Llama_seq1024_hd128) { run_one_shape({1024, 24, 8, 128}); }
TEST(TrackE_Correctness, Qwen3MHA_seq1024_hd128) { run_one_shape({1024, 32, 32, 128}); }
TEST(TrackE_Correctness, Gemma4SWA_seq1024_hd256) { run_one_shape({1024, 32, 16, 256}); }
TEST(TrackE_Correctness, Gemma4Global_seq1024_hd512) { run_one_shape({1024, 8, 8, 512}); }
TEST(TrackE_Correctness, Llama70B_seq2048_hd128) { run_one_shape({2048, 64, 8, 128}); }

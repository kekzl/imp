#include "model/model.h"
#include "graph/executor.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

namespace imp {

// ---------------------------------------------------------------------------
// Host-side FP16 conversion helpers (bitwise, no CUDA device intrinsics)
// ---------------------------------------------------------------------------

static uint16_t float_to_fp16(float val) {
    uint32_t fbits;
    std::memcpy(&fbits, &val, 4);
    uint32_t f_sign = (fbits >> 31) & 1;
    int f_exp = static_cast<int>((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man = fbits & 0x7FFFFF;
    if ((fbits & 0x7FFFFFFF) == 0) return static_cast<uint16_t>(f_sign << 15);
    if (f_exp > 15) return static_cast<uint16_t>((f_sign << 15) | 0x7C00);
    if (f_exp < -24) return static_cast<uint16_t>(f_sign << 15);
    if (f_exp < -14) {
        int shift = -14 - f_exp;
        uint32_t subnormal_man = (0x800000 | f_man) >> (shift + 13);
        return static_cast<uint16_t>((f_sign << 15) | (subnormal_man & 0x3FF));
    }
    uint16_t h_exp = static_cast<uint16_t>(f_exp + 15);
    uint16_t h_man = static_cast<uint16_t>(f_man >> 13);
    return static_cast<uint16_t>((f_sign << 15) | (h_exp << 10) | h_man);
}

// ---------------------------------------------------------------------------
// Build a synthetic FP16 model for benchmarking
// ---------------------------------------------------------------------------

static std::unique_ptr<Model> make_bench_model(
    int d_model, int d_ff, int n_heads, int n_kv_heads, int n_layers,
    int vocab_size, int max_seq_len)
{
    auto model = std::make_unique<Model>();
    auto& cfg = model->config_;
    cfg.arch = ModelArch::LLAMA;
    cfg.n_layers = n_layers;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.d_model = d_model;
    cfg.d_ff = d_ff;
    cfg.vocab_size = vocab_size;
    cfg.max_seq_len = max_seq_len;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    int head_dim = d_model / n_heads;

    // Helper: create an FP16 weight tensor on host filled with small random values.
    auto make_fp16_weight = [](int rows, int cols, std::mt19937& rng) -> Tensor {
        size_t n = static_cast<size_t>(rows) * cols;
        auto* buf = new uint16_t[n];
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = float_to_fp16(dist(rng));
        }
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        return Tensor(buf, DType::FP16, 2, shape, false);
    };

    // Helper: create 1D FP16 norm weight on host (all 1.0).
    auto make_norm_weight = [](int dim) -> Tensor {
        auto* buf = new uint16_t[dim];
        uint16_t one = float_to_fp16(1.0f);
        for (int i = 0; i < dim; ++i) buf[i] = one;
        int64_t shape[4] = {static_cast<int64_t>(dim), 0, 0, 0};
        return Tensor(buf, DType::FP16, 1, shape, false);
    };

    std::mt19937 rng(42);

    // Token embedding [vocab_size, d_model]
    model->tok_emb_ = make_fp16_weight(vocab_size, d_model, rng);
    model->tok_emb_qtype_ = GGMLQuantType::F16;

    // Output norm [d_model]
    model->out_norm_ = make_norm_weight(d_model);
    model->out_norm_qtype_ = GGMLQuantType::F16;

    // Output projection [vocab_size, d_model]
    model->out_proj_ = make_fp16_weight(vocab_size, d_model, rng);
    model->out_proj_qtype_ = GGMLQuantType::F16;

    // Layers
    model->layers_.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        auto& ly = model->layers_[l];

        ly.wq = make_fp16_weight(n_heads * head_dim, d_model, rng);
        ly.wq_qtype = GGMLQuantType::F16;

        ly.wk = make_fp16_weight(n_kv_heads * head_dim, d_model, rng);
        ly.wk_qtype = GGMLQuantType::F16;

        ly.wv = make_fp16_weight(n_kv_heads * head_dim, d_model, rng);
        ly.wv_qtype = GGMLQuantType::F16;

        ly.wo = make_fp16_weight(d_model, n_heads * head_dim, rng);
        ly.wo_qtype = GGMLQuantType::F16;

        ly.w_gate = make_fp16_weight(d_ff, d_model, rng);
        ly.w_gate_qtype = GGMLQuantType::F16;

        ly.w_up = make_fp16_weight(d_ff, d_model, rng);
        ly.w_up_qtype = GGMLQuantType::F16;

        ly.w_down = make_fp16_weight(d_model, d_ff, rng);
        ly.w_down_qtype = GGMLQuantType::F16;

        ly.attn_norm = make_norm_weight(d_model);
        ly.ffn_norm = make_norm_weight(d_model);
    }

    return model;
}

// ---------------------------------------------------------------------------
// bench_e2e: end-to-end prefill and decode benchmark
// ---------------------------------------------------------------------------

void bench_e2e() {
    // Check for CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("bench_e2e: no CUDA device found, skipping\n");
        return;
    }

    // Model dimensions (small for benchmarking the full pipeline)
    const int d_model    = 256;
    const int d_ff       = 512;
    const int n_heads    = 8;
    const int n_kv_heads = 8;
    const int n_layers   = 4;
    const int vocab_size = 1000;
    const int max_seq_len = 1024;

    const int warmup_iters = 3;
    const int timed_iters  = 10;

    const std::vector<int> seq_lens = {32, 128, 512};

    printf("=== End-to-End Benchmark ===\n");
    printf("Model: d=%d, ff=%d, heads=%d, layers=%d, vocab=%d\n\n",
           d_model, d_ff, n_heads, n_layers, vocab_size);

    // Build and upload model
    auto model = make_bench_model(d_model, d_ff, n_heads, n_kv_heads,
                                  n_layers, vocab_size, max_seq_len);
    if (!model->upload_weights_gpu(DType::FP16, nullptr)) {
        printf("bench_e2e: failed to upload weights to GPU\n");
        return;
    }

    // Initialize executor
    GraphExecutor executor;
    if (!executor.init(*model, DType::FP16, false)) {
        printf("bench_e2e: failed to initialize GraphExecutor\n");
        return;
    }

    // -----------------------------------------------------------------------
    // Prefill benchmark
    // -----------------------------------------------------------------------
    printf("Prefill:\n");

    for (int seq_len : seq_lens) {
        // Create input tokens and positions on host
        std::vector<int32_t> h_tokens(seq_len);
        std::vector<int> h_positions(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            h_tokens[i] = i % vocab_size;
            h_positions[i] = i;
        }

        // Upload to device
        int32_t* d_tokens = nullptr;
        int* d_positions = nullptr;
        cudaMalloc(&d_tokens, seq_len * sizeof(int32_t));
        cudaMalloc(&d_positions, seq_len * sizeof(int));
        cudaMemcpy(d_tokens, h_tokens.data(), seq_len * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_positions, h_positions.data(), seq_len * sizeof(int),
                   cudaMemcpyHostToDevice);

        InferenceState state;
        state.token_ids = d_tokens;
        state.positions = d_positions;
        state.n_tokens = seq_len;
        state.is_prefill = true;
        state.temperature = 0.0f;

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            executor.forward(state, nullptr);
            cudaDeviceSynchronize();
        }

        // Timed iterations
        double total_ms = 0.0;
        for (int i = 0; i < timed_iters; ++i) {
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            executor.forward(state, nullptr);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
        }

        double avg_ms = total_ms / timed_iters;
        double tok_per_sec = (seq_len / avg_ms) * 1000.0;

        printf("  seq=%-8d  %8.2f ms  %8.1f tok/s\n", seq_len, avg_ms, tok_per_sec);

        cudaFree(d_tokens);
        cudaFree(d_positions);
    }

    // -----------------------------------------------------------------------
    // Decode benchmark (single-token steps)
    // -----------------------------------------------------------------------
    printf("\nDecode (single token):\n");

    {
        std::vector<int32_t> h_token = {42};
        std::vector<int> h_position = {0};

        int32_t* d_token = nullptr;
        int* d_position = nullptr;
        cudaMalloc(&d_token, sizeof(int32_t));
        cudaMalloc(&d_position, sizeof(int));
        cudaMemcpy(d_token, h_token.data(), sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_position, h_position.data(), sizeof(int),
                   cudaMemcpyHostToDevice);

        InferenceState state;
        state.token_ids = d_token;
        state.positions = d_position;
        state.n_tokens = 1;
        state.is_prefill = true;  // no KV cache, treat as single-token prefill
        state.temperature = 0.0f;

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            executor.forward(state, nullptr);
            cudaDeviceSynchronize();
        }

        // Timed iterations
        double total_ms = 0.0;
        for (int i = 0; i < timed_iters; ++i) {
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            executor.forward(state, nullptr);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
        }

        double avg_ms = total_ms / timed_iters;
        double tok_per_sec = (1.0 / avg_ms) * 1000.0;

        printf("  step          %8.2f ms  %8.0f tok/s\n", avg_ms, tok_per_sec);

        cudaFree(d_token);
        cudaFree(d_position);
    }

    printf("\n");
}

} // namespace imp

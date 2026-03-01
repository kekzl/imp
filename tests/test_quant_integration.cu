#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "model/model.h"
#include "model/gguf_loader.h"
#include "graph/executor.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>

namespace imp {
namespace {

// ===========================================================================
// Helpers
// ===========================================================================

static bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!HasCudaDevice()) {                                                \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

// Host-side FP16 conversion helpers (bitwise, no CUDA device intrinsics)
static float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp  = (h >> 10) & 0x1F;
    uint16_t man  = h & 0x3FF;
    float result;
    if (exp == 0) {
        if (man == 0) result = 0.0f;
        else result = std::ldexp(static_cast<float>(man) / 1024.0f, -14);
    } else if (exp == 31) {
        result = 0.0f;
    } else {
        result = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f, exp - 15);
    }
    return sign ? -result : result;
}

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

// Build a synthetic Q4_0 block.
// Q4_0: 18 bytes = 2 (fp16 scale) + 16 (packed nibbles for 32 elements).
// nibble_val is used for ALL 32 elements.
static std::vector<uint8_t> make_q4_0_block(float scale, uint8_t nibble_val) {
    std::vector<uint8_t> block(18);
    uint16_t scale_bits = float_to_fp16(scale);
    std::memcpy(block.data(), &scale_bits, 2);
    uint8_t byte_val = static_cast<uint8_t>((nibble_val << 4) | nibble_val);
    for (int i = 0; i < 16; ++i) {
        block[2 + i] = byte_val;
    }
    return block;
}

// Build a synthetic Q8_0 block.
// Q8_0: 34 bytes = 2 (fp16 scale) + 32 (int8 quants).
static std::vector<uint8_t> make_q8_0_block(float scale, int8_t quant_val) {
    std::vector<uint8_t> block(34);
    uint16_t scale_bits = float_to_fp16(scale);
    std::memcpy(block.data(), &scale_bits, 2);
    for (int i = 0; i < 32; ++i) {
        block[2 + i] = static_cast<uint8_t>(quant_val);
    }
    return block;
}

// Build a Model with synthetic weights for testing.
// If use_q4_0 is true, weights are in Q4_0 format; otherwise FP16.
static std::unique_ptr<Model> make_test_model(
    int d_model, int d_ff, int n_heads, int n_kv_heads, int n_layers,
    int vocab_size, bool use_q4_0, float scale = 0.1f, uint8_t nibble = 8)
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
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    int head_dim = d_model / n_heads;

    // Allocate host-side buffers that will persist for the model's lifetime.
    // Store them in static vectors so they outlive the test (cleared in each test).
    // Actually, we'll use the model's mmap trick: allocate a big buffer as mmap_base_.
    // For simplicity, use malloc and set mmap_base_ = nullptr (no mmap cleanup needed).

    // Helper: create an FP16 weight tensor on host filled with small random values.
    auto make_fp16_weight = [](int rows, int cols, std::mt19937& rng) -> std::pair<void*, Tensor> {
        size_t n = static_cast<size_t>(rows) * cols;
        auto* buf = new uint16_t[n];
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = float_to_fp16(dist(rng));
        }
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::FP16, 2, shape, false);
        return {buf, t};
    };

    // Helper: create a Q4_0 weight tensor on host.
    auto make_q4_0_weight = [&](int rows, int cols) -> std::pair<void*, Tensor> {
        int blocks_per_row = cols / 32;
        size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
        size_t total_bytes = total_blocks * 18;
        auto* buf = new uint8_t[total_bytes];

        for (size_t i = 0; i < total_blocks; ++i) {
            auto blk = make_q4_0_block(scale, nibble);
            std::memcpy(buf + i * 18, blk.data(), 18);
        }

        // Tensor shape is logical: [rows, cols]
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::INT4, 2, shape, false);
        return {buf, t};
    };

    // Helper: create 1D FP16 norm weight on host (all 1.0).
    auto make_norm_weight = [](int dim) -> std::pair<void*, Tensor> {
        auto* buf = new uint16_t[dim];
        uint16_t one = float_to_fp16(1.0f);
        for (int i = 0; i < dim; ++i) buf[i] = one;
        int64_t shape[4] = {static_cast<int64_t>(dim), 0, 0, 0};
        Tensor t(buf, DType::FP16, 1, shape, false);
        return {buf, t};
    };

    // Track all host allocations for cleanup.
    // We'll use a shared vector stored as the model's "fake mmap" data.
    // For test cleanup, the caller should delete the model.
    // We store raw pointers in a static thread_local vector and clean up per-model.
    // Actually, simplest: just leak in tests (GTest cleans up process anyway) or
    // we can track via a helper class.

    // For this test, allocate everything and track in a vector we'll attach
    // to the model via a small wrapper. Since Model has no host_alloc tracking,
    // we'll just leak the host buffers (they're small test allocations).

    std::mt19937 rng(42);

    // Token embedding [vocab_size, d_model]
    auto [tok_emb_buf, tok_emb] = make_fp16_weight(vocab_size, d_model, rng);
    model->tok_emb_ = tok_emb;
    model->tok_emb_qtype_ = GGMLQuantType::F16;

    // Output norm [d_model]
    auto [out_norm_buf, out_norm] = make_norm_weight(d_model);
    model->out_norm_ = out_norm;
    model->out_norm_qtype_ = GGMLQuantType::F16;

    // Output projection [vocab_size, d_model]
    auto [out_proj_buf, out_proj] = make_fp16_weight(vocab_size, d_model, rng);
    model->out_proj_ = out_proj;
    model->out_proj_qtype_ = GGMLQuantType::F16;

    // Layers
    model->layers_.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        auto& ly = model->layers_[l];

        if (use_q4_0) {
            // Attention weights in Q4_0
            auto [wq_buf, wq] = make_q4_0_weight(n_heads * head_dim, d_model);
            ly.wq = wq; ly.wq_qtype = GGMLQuantType::Q4_0;

            auto [wk_buf, wk] = make_q4_0_weight(n_kv_heads * head_dim, d_model);
            ly.wk = wk; ly.wk_qtype = GGMLQuantType::Q4_0;

            auto [wv_buf, wv] = make_q4_0_weight(n_kv_heads * head_dim, d_model);
            ly.wv = wv; ly.wv_qtype = GGMLQuantType::Q4_0;

            auto [wo_buf, wo] = make_q4_0_weight(d_model, n_heads * head_dim);
            ly.wo = wo; ly.wo_qtype = GGMLQuantType::Q4_0;

            // FFN weights in Q4_0
            auto [wg_buf, wg] = make_q4_0_weight(d_ff, d_model);
            ly.w_gate = wg; ly.w_gate_qtype = GGMLQuantType::Q4_0;

            auto [wu_buf, wu] = make_q4_0_weight(d_ff, d_model);
            ly.w_up = wu; ly.w_up_qtype = GGMLQuantType::Q4_0;

            auto [wd_buf, wd] = make_q4_0_weight(d_model, d_ff);
            ly.w_down = wd; ly.w_down_qtype = GGMLQuantType::Q4_0;
        } else {
            // Attention weights in FP16
            auto [wq_buf, wq] = make_fp16_weight(n_heads * head_dim, d_model, rng);
            ly.wq = wq; ly.wq_qtype = GGMLQuantType::F16;

            auto [wk_buf, wk] = make_fp16_weight(n_kv_heads * head_dim, d_model, rng);
            ly.wk = wk; ly.wk_qtype = GGMLQuantType::F16;

            auto [wv_buf, wv] = make_fp16_weight(n_kv_heads * head_dim, d_model, rng);
            ly.wv = wv; ly.wv_qtype = GGMLQuantType::F16;

            auto [wo_buf, wo] = make_fp16_weight(d_model, n_heads * head_dim, rng);
            ly.wo = wo; ly.wo_qtype = GGMLQuantType::F16;

            auto [wg_buf, wg] = make_fp16_weight(d_ff, d_model, rng);
            ly.w_gate = wg; ly.w_gate_qtype = GGMLQuantType::F16;

            auto [wu_buf, wu] = make_fp16_weight(d_ff, d_model, rng);
            ly.w_up = wu; ly.w_up_qtype = GGMLQuantType::F16;

            auto [wd_buf, wd] = make_fp16_weight(d_model, d_ff, rng);
            ly.w_down = wd; ly.w_down_qtype = GGMLQuantType::F16;
        }

        // Norm weights always FP16
        auto [an_buf, an] = make_norm_weight(d_model);
        ly.attn_norm = an;
        auto [fn_buf, fn] = make_norm_weight(d_model);
        ly.ffn_norm = fn;
    }

    return model;
}

// Build a Model with synthetic Q8_0 weights for testing.
static std::unique_ptr<Model> make_q8_0_test_model(
    int d_model, int d_ff, int n_heads, int n_kv_heads, int n_layers,
    int vocab_size, float scale = 0.01f, int8_t quant_val = 2)
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
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    int head_dim = d_model / n_heads;

    // Helper: create a Q8_0 weight tensor on host.
    auto make_q8_0_weight = [&](int rows, int cols) -> std::pair<void*, Tensor> {
        int blocks_per_row = cols / 32;
        size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
        size_t total_bytes = total_blocks * 34;
        auto* buf = new uint8_t[total_bytes];

        for (size_t i = 0; i < total_blocks; ++i) {
            auto blk = make_q8_0_block(scale, quant_val);
            std::memcpy(buf + i * 34, blk.data(), 34);
        }

        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::INT8, 2, shape, false);
        return {buf, t};
    };

    // Helper: create an FP16 weight tensor on host filled with small values.
    auto make_fp16_weight = [](int rows, int cols) -> std::pair<void*, Tensor> {
        size_t n = static_cast<size_t>(rows) * cols;
        auto* buf = new uint16_t[n];
        for (size_t i = 0; i < n; ++i) buf[i] = float_to_fp16(0.01f);
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::FP16, 2, shape, false);
        return {buf, t};
    };

    // Helper: create 1D FP16 norm weight on host (all 1.0).
    auto make_norm_weight = [](int dim) -> std::pair<void*, Tensor> {
        auto* buf = new uint16_t[dim];
        uint16_t one = float_to_fp16(1.0f);
        for (int i = 0; i < dim; ++i) buf[i] = one;
        int64_t shape[4] = {static_cast<int64_t>(dim), 0, 0, 0};
        Tensor t(buf, DType::FP16, 1, shape, false);
        return {buf, t};
    };

    // Token embedding in FP16 (small values)
    auto [tok_emb_buf, tok_emb] = make_fp16_weight(vocab_size, d_model);
    model->tok_emb_ = tok_emb;
    model->tok_emb_qtype_ = GGMLQuantType::F16;

    // Output norm
    auto [out_norm_buf, out_norm] = make_norm_weight(d_model);
    model->out_norm_ = out_norm;
    model->out_norm_qtype_ = GGMLQuantType::F16;

    // Output projection in FP16
    auto [out_proj_buf, out_proj] = make_fp16_weight(vocab_size, d_model);
    model->out_proj_ = out_proj;
    model->out_proj_qtype_ = GGMLQuantType::F16;

    // Layers with Q8_0 weights
    model->layers_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        auto& ly = model->layers_[l];

        auto [wq_buf, wq] = make_q8_0_weight(n_heads * head_dim, d_model);
        ly.wq = wq; ly.wq_qtype = GGMLQuantType::Q8_0;

        auto [wk_buf, wk] = make_q8_0_weight(n_kv_heads * head_dim, d_model);
        ly.wk = wk; ly.wk_qtype = GGMLQuantType::Q8_0;

        auto [wv_buf, wv] = make_q8_0_weight(n_kv_heads * head_dim, d_model);
        ly.wv = wv; ly.wv_qtype = GGMLQuantType::Q8_0;

        auto [wo_buf, wo] = make_q8_0_weight(d_model, n_heads * head_dim);
        ly.wo = wo; ly.wo_qtype = GGMLQuantType::Q8_0;

        auto [wg_buf, wg] = make_q8_0_weight(d_ff, d_model);
        ly.w_gate = wg; ly.w_gate_qtype = GGMLQuantType::Q8_0;

        auto [wu_buf, wu] = make_q8_0_weight(d_ff, d_model);
        ly.w_up = wu; ly.w_up_qtype = GGMLQuantType::Q8_0;

        auto [wd_buf, wd] = make_q8_0_weight(d_model, d_ff);
        ly.w_down = wd; ly.w_down_qtype = GGMLQuantType::Q8_0;

        auto [an_buf, an] = make_norm_weight(d_model);
        ly.attn_norm = an;
        auto [fn_buf, fn] = make_norm_weight(d_model);
        ly.ffn_norm = fn;
    }

    return model;
}

// ===========================================================================
// Test 1: Q4_0 weight upload keeps raw quantized bytes on GPU (dp4a GEMV path)
// ===========================================================================
TEST(QuantIntegrationTest, Q4_0WeightUpload) {
    SKIP_IF_NO_CUDA();

    // d_model=32 so each weight row is 1 Q4_0 block (32 elements)
    auto model = make_test_model(
        /*d_model=*/32, /*d_ff=*/64, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32, /*use_q4_0=*/true,
        /*scale=*/0.5f, /*nibble=*/10);

    ASSERT_FALSE(model->gpu_weights_ready());

    // Upload weights to GPU
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));
    EXPECT_TRUE(model->gpu_weights_ready());

    // Check layer 0 wq: raw upload keeps logical shape [N, K] and FP16 dtype
    const auto& ly = model->layer(0);
    EXPECT_TRUE(ly.wq.on_device);
    EXPECT_EQ(ly.wq.dtype, DType::FP16);
    EXPECT_EQ(ly.wq.ndim, 2);
    // wq shape: [n_heads * head_dim, d_model] = [32, 32]
    EXPECT_EQ(ly.wq.shape[0], 32);
    EXPECT_EQ(ly.wq.shape[1], 32);

    // Read back raw Q4_0 data and verify block structure
    // Each Q4_0 block: 18 bytes = 2B (FP16 scale) + 16B (packed nibbles)
    const int blocks_per_row = 1;  // 32 / 32
    const int raw_bytes_per_row = blocks_per_row * 18;
    const int total_raw_bytes = 32 * raw_bytes_per_row;
    std::vector<uint8_t> h_raw(total_raw_bytes);
    cudaMemcpy(h_raw.data(), ly.wq.data, total_raw_bytes,
               cudaMemcpyDeviceToHost);

    for (int row = 0; row < 32; ++row) {
        const uint8_t* blk = h_raw.data() + row * raw_bytes_per_row;
        // Check scale (first 2 bytes = FP16 0.5)
        uint16_t scale_bits;
        std::memcpy(&scale_bits, blk, 2);
        float s = fp16_to_float(scale_bits);
        EXPECT_NEAR(s, 0.5f, 0.01f) << "Scale mismatch at row " << row;
        // Check nibbles (next 16 bytes = 0xAA for nibble 10)
        for (int j = 0; j < 16; ++j) {
            EXPECT_EQ(blk[2 + j], 0xAA)
                << "Nibble byte mismatch at row " << row << " byte " << j;
        }
    }

    // Also check embedding and output are on device
    EXPECT_TRUE(model->token_embedding().on_device);
    EXPECT_TRUE(model->output_proj().on_device);
    EXPECT_TRUE(model->output_norm().on_device);
}

// ===========================================================================
// Test 2: Q8_0 weight upload keeps raw quantized bytes on GPU
// ===========================================================================
TEST(QuantIntegrationTest, Q8_0WeightUpload) {
    SKIP_IF_NO_CUDA();

    // Create a model manually with Q8_0 weights
    auto model = std::make_unique<Model>();
    auto& cfg = model->config_;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 2;
    cfg.d_model = 32;
    cfg.d_ff = 64;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;

    // Create Q8_0 weight for wq: [32, 32] = 32 rows, each with 1 block
    const int rows = 32, cols = 32;
    const float scale = 0.25f;
    const int8_t quant_val = 4;

    int blocks_per_row = cols / 32;
    size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
    size_t raw_bytes = total_blocks * 34;
    auto* q8_buf = new uint8_t[raw_bytes];
    for (size_t i = 0; i < total_blocks; ++i) {
        auto blk = make_q8_0_block(scale, quant_val);
        std::memcpy(q8_buf + i * 34, blk.data(), 34);
    }

    int64_t shape[4] = {rows, cols, 0, 0};
    model->layers_.resize(1);
    model->layers_[0].wq = Tensor(q8_buf, DType::INT8, 2, shape, false);
    model->layers_[0].wq_qtype = GGMLQuantType::Q8_0;

    // Set other weights to minimal FP16 (just need wq for this test)
    auto make_fp16 = [](int r, int c) -> Tensor {
        size_t n = static_cast<size_t>(r) * c;
        auto* buf = new uint16_t[n];
        for (size_t i = 0; i < n; ++i) buf[i] = float_to_fp16(0.01f);
        int64_t s[4] = {static_cast<int64_t>(r), static_cast<int64_t>(c), 0, 0};
        return Tensor(buf, DType::FP16, 2, s, false);
    };
    auto make_norm = [](int d) -> Tensor {
        auto* buf = new uint16_t[d];
        for (int i = 0; i < d; ++i) buf[i] = float_to_fp16(1.0f);
        int64_t s[4] = {static_cast<int64_t>(d), 0, 0, 0};
        return Tensor(buf, DType::FP16, 1, s, false);
    };

    auto& ly = model->layers_[0];
    ly.wk = make_fp16(32, 32); ly.wk_qtype = GGMLQuantType::F16;
    ly.wv = make_fp16(32, 32); ly.wv_qtype = GGMLQuantType::F16;
    ly.wo = make_fp16(32, 32); ly.wo_qtype = GGMLQuantType::F16;
    ly.w_gate = make_fp16(64, 32); ly.w_gate_qtype = GGMLQuantType::F16;
    ly.w_up = make_fp16(64, 32); ly.w_up_qtype = GGMLQuantType::F16;
    ly.w_down = make_fp16(32, 64); ly.w_down_qtype = GGMLQuantType::F16;
    ly.attn_norm = make_norm(32);
    ly.ffn_norm = make_norm(32);

    model->tok_emb_ = make_fp16(32, 32);
    model->tok_emb_qtype_ = GGMLQuantType::F16;
    model->out_norm_ = make_norm(32);
    model->out_norm_qtype_ = GGMLQuantType::F16;
    model->out_proj_ = make_fp16(32, 32);
    model->out_proj_qtype_ = GGMLQuantType::F16;

    // Upload
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    // After upload, wq should be on device with logical shape [N, K]
    // Data is raw Q8_0 bytes (not dequanted FP16)
    EXPECT_TRUE(ly.wq.on_device);
    EXPECT_EQ(ly.wq.dtype, DType::FP16);
    EXPECT_EQ(ly.wq.shape[0], rows);
    EXPECT_EQ(ly.wq.shape[1], cols);

    // Read back raw bytes and verify they match the original Q8_0 data
    std::vector<uint8_t> h_raw(raw_bytes);
    cudaMemcpy(h_raw.data(), ly.wq.data, raw_bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < raw_bytes; ++i) {
        EXPECT_EQ(h_raw[i], q8_buf[i]) << "Raw Q8_0 byte mismatch at " << i;
    }

    // Verify on-the-fly dequant produces correct FP16 values
    void* d_fp16 = nullptr;
    cudaMalloc(&d_fp16, static_cast<size_t>(rows * cols) * sizeof(uint16_t));
    dequant_gpu(ly.wq.data, d_fp16, GGMLQuantType::Q8_0, rows, cols, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint16_t> h_fp16(rows * cols);
    cudaMemcpy(h_fp16.data(), d_fp16,
               static_cast<size_t>(rows * cols) * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_fp16);

    float expected = static_cast<float>(quant_val) * fp16_to_float(float_to_fp16(scale));
    for (int i = 0; i < rows * cols; ++i) {
        float got = fp16_to_float(h_fp16[i]);
        EXPECT_NEAR(got, expected, 0.05f)
            << "Q8_0 on-the-fly dequant mismatch at index " << i;
    }

    delete[] q8_buf;
}

// ===========================================================================
// Test 3: F32 weight upload converts to FP16
// ===========================================================================
TEST(QuantIntegrationTest, F32WeightUpload) {
    SKIP_IF_NO_CUDA();

    auto model = std::make_unique<Model>();
    auto& cfg = model->config_;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 2;
    cfg.d_model = 32;
    cfg.d_ff = 64;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 64;

    // Create F32 embedding [32, 32]
    const int rows = 32, cols = 32;
    auto* f32_buf = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        f32_buf[i] = 0.5f + 0.001f * i;
    }

    int64_t shape[4] = {rows, cols, 0, 0};
    model->tok_emb_ = Tensor(f32_buf, DType::FP32, 2, shape, false);
    model->tok_emb_qtype_ = GGMLQuantType::F32;

    // Minimal other weights
    auto make_fp16 = [](int r, int c) -> Tensor {
        auto* buf = new uint16_t[static_cast<size_t>(r) * c];
        for (int i = 0; i < r * c; ++i) buf[i] = float_to_fp16(0.01f);
        int64_t s[4] = {static_cast<int64_t>(r), static_cast<int64_t>(c), 0, 0};
        return Tensor(buf, DType::FP16, 2, s, false);
    };
    auto make_norm = [](int d) -> Tensor {
        auto* buf = new uint16_t[d];
        for (int i = 0; i < d; ++i) buf[i] = float_to_fp16(1.0f);
        int64_t s[4] = {static_cast<int64_t>(d), 0, 0, 0};
        return Tensor(buf, DType::FP16, 1, s, false);
    };

    model->out_norm_ = make_norm(32);
    model->out_norm_qtype_ = GGMLQuantType::F16;
    model->out_proj_ = make_fp16(32, 32);
    model->out_proj_qtype_ = GGMLQuantType::F16;

    model->layers_.resize(1);
    auto& ly = model->layers_[0];
    ly.wq = make_fp16(32, 32); ly.wq_qtype = GGMLQuantType::F16;
    ly.wk = make_fp16(32, 32); ly.wk_qtype = GGMLQuantType::F16;
    ly.wv = make_fp16(32, 32); ly.wv_qtype = GGMLQuantType::F16;
    ly.wo = make_fp16(32, 32); ly.wo_qtype = GGMLQuantType::F16;
    ly.w_gate = make_fp16(64, 32); ly.w_gate_qtype = GGMLQuantType::F16;
    ly.w_up = make_fp16(64, 32); ly.w_up_qtype = GGMLQuantType::F16;
    ly.w_down = make_fp16(32, 64); ly.w_down_qtype = GGMLQuantType::F16;
    ly.attn_norm = make_norm(32);
    ly.ffn_norm = make_norm(32);

    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    // Token embedding should now be FP16 on device
    EXPECT_TRUE(model->tok_emb_.on_device);
    EXPECT_EQ(model->tok_emb_.dtype, DType::FP16);

    // Read back and verify conversion
    std::vector<uint16_t> h_result(rows * cols);
    cudaMemcpy(h_result.data(), model->tok_emb_.data,
               static_cast<size_t>(rows * cols) * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows * cols; ++i) {
        float original = f32_buf[i];
        float got = fp16_to_float(h_result[i]);
        EXPECT_NEAR(got, original, 0.01f)
            << "F32->FP16 conversion mismatch at index " << i;
    }

    delete[] f32_buf;
}

// ===========================================================================
// Test 4: upload_weights_gpu idempotent (second call returns true early)
// ===========================================================================
TEST(QuantIntegrationTest, UploadIdempotent) {
    SKIP_IF_NO_CUDA();

    auto model = make_test_model(32, 64, 4, 4, 1, 32, false);

    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));
    EXPECT_TRUE(model->gpu_weights_ready());

    // Second call should succeed without re-uploading
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));
    EXPECT_TRUE(model->gpu_weights_ready());
}

// ===========================================================================
// Test 5: FP16 model forward pass through executor (baseline)
// ===========================================================================
TEST(QuantIntegrationTest, FP16ForwardPass) {
    SKIP_IF_NO_CUDA();

    auto model = make_test_model(32, 64, 4, 4, 1, 32, false);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    // Create input tokens on device
    std::vector<int32_t> h_tokens = {1, 5, 10};
    std::vector<int> h_positions = {0, 1, 2};
    int n_tokens = 3;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.temperature = 0.0f; // greedy

    int32_t token = executor.forward(state, nullptr);

    // Token should be in valid range
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 6: Q4_0 quantized model forward pass through executor
// ===========================================================================
TEST(QuantIntegrationTest, Q4_0ForwardPass) {
    SKIP_IF_NO_CUDA();

    // nibble=8 means dequanted values = (8-8)*scale = 0, so weights are zero.
    // Use nibble=9 for small nonzero weights: (9-8)*0.01 = 0.01
    auto model = make_test_model(32, 64, 4, 4, 1, 32, true,
                                  /*scale=*/0.01f, /*nibble=*/9);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    // Input tokens
    std::vector<int32_t> h_tokens = {0, 1};
    std::vector<int> h_positions = {0, 1};
    int n_tokens = 2;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    // Also verify logits are not NaN/Inf
    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    ASSERT_NE(logits.data, nullptr);
    EXPECT_EQ(logits.shape[0], 1);  // prefill: last token only
    EXPECT_EQ(logits.shape[1], 32); // vocab_size

    std::vector<float> h_logits(1 * 32);
    cudaMemcpy(h_logits.data(), logits.data,
               static_cast<size_t>(1 * 32) * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool has_nan = false;
    bool has_inf = false;
    for (size_t i = 0; i < h_logits.size(); ++i) {
        float v = h_logits[i];
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
    }
    EXPECT_FALSE(has_nan) << "Logits contain NaN";
    EXPECT_FALSE(has_inf) << "Logits contain Inf";

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 7: Q4_0 forward is deterministic
// ===========================================================================
TEST(QuantIntegrationTest, Q4_0Deterministic) {
    SKIP_IF_NO_CUDA();

    auto model = make_test_model(32, 64, 4, 4, 1, 32, true, 0.01f, 9);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {3, 7};
    std::vector<int> h_positions = {0, 1};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 2 * sizeof(int32_t));
    cudaMalloc(&d_positions, 2 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 2 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 2 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 2;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token1 = executor.forward(state, nullptr);
    int32_t token2 = executor.forward(state, nullptr);
    EXPECT_EQ(token1, token2) << "Greedy sampling should be deterministic";

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 8: Q4_0 logits shape matches expected dimensions
// ===========================================================================
TEST(QuantIntegrationTest, Q4_0LogitsShape) {
    SKIP_IF_NO_CUDA();

    auto model = make_test_model(32, 64, 4, 4, 1, 32, true, 0.01f, 9);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {1, 2, 3, 4, 5};
    std::vector<int> h_positions = {0, 1, 2, 3, 4};
    int n_tokens = 5;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(logits.ndim, 2);
    EXPECT_EQ(logits.shape[0], 1);  // prefill: last token only
    EXPECT_EQ(logits.shape[1], 32); // vocab_size
    EXPECT_EQ(logits.dtype, DType::FP32); // logits are FP32 for sampling precision
    EXPECT_TRUE(logits.on_device);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 9: Multi-layer Q4_0 model
// ===========================================================================
TEST(QuantIntegrationTest, Q4_0MultiLayer) {
    SKIP_IF_NO_CUDA();

    // 4 layers to test multi-layer quantized forward pass
    auto model = make_test_model(32, 64, 4, 4, 4, 32, true, 0.01f, 9);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {1, 2, 3};
    std::vector<int> h_positions = {0, 1, 2};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 3 * sizeof(int32_t));
    cudaMalloc(&d_positions, 3 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 3 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 3 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 3;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 10: Q8_0 forward pass through executor (on-the-fly dequant)
// ===========================================================================
TEST(QuantIntegrationTest, Q8_0ForwardPass) {
    SKIP_IF_NO_CUDA();

    auto model = make_q8_0_test_model(32, 64, 4, 4, 1, 32, 0.01f, 2);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {0, 1};
    std::vector<int> h_positions = {0, 1};
    int n_tokens = 2;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    // Verify logits are not NaN/Inf
    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    ASSERT_NE(logits.data, nullptr);
    EXPECT_EQ(logits.shape[0], 1);  // prefill: last token only
    EXPECT_EQ(logits.shape[1], 32);

    std::vector<float> h_logits(1 * 32);
    cudaMemcpy(h_logits.data(), logits.data,
               static_cast<size_t>(1 * 32) * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool has_nan = false;
    bool has_inf = false;
    for (size_t i = 0; i < h_logits.size(); ++i) {
        if (std::isnan(h_logits[i])) has_nan = true;
        if (std::isinf(h_logits[i])) has_inf = true;
    }
    EXPECT_FALSE(has_nan) << "Logits contain NaN";
    EXPECT_FALSE(has_inf) << "Logits contain Inf";

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 11: Q8_0 multi-layer forward pass
// ===========================================================================
TEST(QuantIntegrationTest, Q8_0MultiLayer) {
    SKIP_IF_NO_CUDA();

    auto model = make_q8_0_test_model(32, 64, 4, 4, 4, 32, 0.01f, 2);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {1, 2, 3};
    std::vector<int> h_positions = {0, 1, 2};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 3 * sizeof(int32_t));
    cudaMalloc(&d_positions, 3 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 3 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 3 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 3;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 12: Direct quant_gemm_int4 correctness vs CPU reference
// ===========================================================================
TEST(QuantIntegrationTest, QuantGemmInt4LargerMatrix) {
    SKIP_IF_NO_CUDA();

    // Larger test: M=8, K=64, N=16, group_size=32
    constexpr int M = 8;
    constexpr int K = 64;
    constexpr int N = 16;
    constexpr int group_size = 32;
    constexpr int num_groups = K / group_size;  // 2
    constexpr int half_K = K / 2;               // 32

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // A[M, K] in float
    std::vector<float> h_A(M * K);
    for (auto& v : h_A) v = dist(rng);

    // B_quant[N, K/2] packed nibbles (random 0-15)
    std::uniform_int_distribution<int> nibble_dist(0, 15);
    std::vector<uint8_t> h_B(N * half_K);
    for (auto& b : h_B) {
        int lo = nibble_dist(rng);
        int hi = nibble_dist(rng);
        b = static_cast<uint8_t>((hi << 4) | lo);
    }

    // Scales[N, num_groups] in float
    std::vector<float> h_scales(N * num_groups);
    for (auto& s : h_scales) s = 0.1f * dist(rng);

    // CPU reference: dequant B and matmul
    std::vector<float> h_B_dequant(N * K);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int byte_idx = n * half_K + k / 2;
            int nibble = (k % 2 == 0) ? (h_B[byte_idx] & 0x0F)
                                       : ((h_B[byte_idx] >> 4) & 0x0F);
            int group_idx = k / group_size;
            // Use fp16-rounded scale for fair comparison
            float scale = fp16_to_float(float_to_fp16(h_scales[n * num_groups + group_idx]));
            h_B_dequant[n * K + k] = static_cast<float>(nibble - 8) * scale;
        }
    }

    std::vector<float> h_C_ref(M * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // Use fp16-rounded A for fair comparison
                float a = fp16_to_float(float_to_fp16(h_A[m * K + k]));
                sum += a * h_B_dequant[n * K + k];
            }
            h_C_ref[m * N + n] = sum;
        }
    }

    // GPU: upload A as FP16
    std::vector<half> h_A_fp16(M * K);
    for (int i = 0; i < M * K; ++i) h_A_fp16[i] = __float2half(h_A[i]);

    void* d_A = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMemcpy(d_A, h_A_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);

    void* d_B = nullptr;
    cudaMalloc(&d_B, N * half_K);
    cudaMemcpy(d_B, h_B.data(), N * half_K, cudaMemcpyHostToDevice);

    std::vector<half> h_scales_fp16(N * num_groups);
    for (int i = 0; i < N * num_groups; ++i) h_scales_fp16[i] = __float2half(h_scales[i]);
    void* d_scales = nullptr;
    cudaMalloc(&d_scales, N * num_groups * sizeof(half));
    cudaMemcpy(d_scales, h_scales_fp16.data(), N * num_groups * sizeof(half),
               cudaMemcpyHostToDevice);

    void* d_C = nullptr;
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    // Build tensors
    int64_t a_shape[4] = {M, K, 0, 0};
    int64_t b_shape[4] = {N, static_cast<int64_t>(half_K), 0, 0};
    int64_t s_shape[4] = {N, static_cast<int64_t>(num_groups), 0, 0};
    int64_t c_shape[4] = {M, N, 0, 0};

    Tensor t_A(d_A, DType::FP16, 2, a_shape, true);
    Tensor t_B(d_B, DType::INT4, 2, b_shape, true);
    Tensor t_S(d_scales, DType::FP16, 2, s_shape, true);
    Tensor t_C(d_C, DType::FP16, 2, c_shape, true);

    quant_gemm_int4(t_A, t_B, t_S, t_C, nullptr);
    cudaDeviceSynchronize();

    // Read back
    std::vector<half> h_C_gpu(M * N);
    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; ++i) {
        float got = __half2float(h_C_gpu[i]);
        float ref = h_C_ref[i];
        // FP16 accumulation introduces more error for larger matrices
        EXPECT_NEAR(got, ref, 0.5f)
            << "QuantGemmInt4 mismatch at [" << i / N << "," << i % N << "]"
            << ": got " << got << ", ref " << ref;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scales);
    cudaFree(d_C);
}

// ===========================================================================
// Helpers for Q4_K and Q5_K block construction
// ===========================================================================

// Build a synthetic Q4_K block (144 bytes = 256 elements).
// Layout: d[2] + dmin[2] + scales[12] + qs[128]
// Uses uniform scale/min across sub-blocks, constant quant value.
static std::vector<uint8_t> make_q4_k_block(float d_scale, float d_min,
                                             uint8_t sub_scale, uint8_t sub_min,
                                             uint8_t q4_val) {
    std::vector<uint8_t> block(144, 0);

    // d[2]: fp16 super-block scale
    uint16_t d_bits = float_to_fp16(d_scale);
    std::memcpy(block.data(), &d_bits, 2);

    // dmin[2]: fp16 super-block min
    uint16_t dmin_bits = float_to_fp16(d_min);
    std::memcpy(block.data() + 2, &dmin_bits, 2);

    // scales[12]: packed 6-bit scales and mins for 8 sub-blocks
    // Sub-blocks 0-3: sc[sub] low 6 bits = sub_scale, sc[sub+4] low 6 bits = sub_min
    // Sub-blocks 4-7: more complex packing with top bits
    // For simplicity, use values < 16 so they fit in 4 bits for the upper packing.
    uint8_t sc_clamped = sub_scale & 63;  // 6-bit scale
    uint8_t mn_clamped = sub_min & 63;    // 6-bit min
    uint8_t* sc = block.data() + 4;
    // Sub-blocks 0-3: scale in sc[0..3] low 6 bits, min in sc[4..7] low 6 bits
    for (int s = 0; s < 4; ++s) {
        sc[s] = sc_clamped;       // low 6 = scale, top 2 = 0
        sc[s + 4] = mn_clamped;   // low 6 = min,   top 2 = 0
    }
    // Sub-blocks 4-7: sc[8..11] packs remaining scales
    // sc_val  = (sc[sub+4] & 0xF) | ((sc[sub-4] >> 6) << 4) → low 4 of sc[8+s-4]
    // Since we set top 2 of sc[0..3] = 0, the upper nibble comes from that.
    // For uniform values, just set low nibble = scale, high nibble = min.
    for (int s = 0; s < 4; ++s) {
        sc[8 + s] = static_cast<uint8_t>((mn_clamped & 0xF) << 4 | (sc_clamped & 0xF));
    }

    // qs[128]: 4-bit quantized values packed as nibble pairs
    // Q4_K layout: 4 chunks of 64 elements each in 32 bytes
    // First 32 elements in low nibbles, next 32 in high nibbles of same 32 bytes
    uint8_t* qs = block.data() + 16;
    uint8_t q4_clamped = q4_val & 0xF;
    uint8_t byte_val = static_cast<uint8_t>((q4_clamped << 4) | q4_clamped);
    std::memset(qs, byte_val, 128);

    return block;
}

// Build a synthetic Q5_K block (176 bytes = 256 elements).
// Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
static std::vector<uint8_t> make_q5_k_block(float d_scale, float d_min,
                                             uint8_t sub_scale, uint8_t sub_min,
                                             uint8_t q4_val, bool high_bit) {
    std::vector<uint8_t> block(176, 0);

    // d[2]
    uint16_t d_bits = float_to_fp16(d_scale);
    std::memcpy(block.data(), &d_bits, 2);

    // dmin[2]
    uint16_t dmin_bits = float_to_fp16(d_min);
    std::memcpy(block.data() + 2, &dmin_bits, 2);

    // scales[12]: same packing as Q4_K
    uint8_t sc_clamped = sub_scale & 63;
    uint8_t mn_clamped = sub_min & 63;
    uint8_t* sc = block.data() + 4;
    for (int s = 0; s < 4; ++s) {
        sc[s] = sc_clamped;
        sc[s + 4] = mn_clamped;
    }
    for (int s = 0; s < 4; ++s) {
        sc[8 + s] = static_cast<uint8_t>((mn_clamped & 0xF) << 4 | (sc_clamped & 0xF));
    }

    // qh[32]: high bits (5th bit) for 256 elements
    uint8_t* qh = block.data() + 16;
    if (high_bit) {
        std::memset(qh, 0xFF, 32);  // all 5th bits set
    } else {
        std::memset(qh, 0x00, 32);  // all 5th bits clear
    }

    // qs[128]: low 4 bits (same layout as Q4_K)
    uint8_t* qs = block.data() + 48;
    uint8_t q4_clamped = q4_val & 0xF;
    uint8_t byte_val = static_cast<uint8_t>((q4_clamped << 4) | q4_clamped);
    std::memset(qs, byte_val, 128);

    return block;
}

// Build a Model with synthetic Q4_K weights for testing.
// d_model and d_ff must be multiples of 256 (Q4_K block size).
static std::unique_ptr<Model> make_q4_k_test_model(
    int d_model, int d_ff, int n_heads, int n_kv_heads, int n_layers,
    int vocab_size, float d_scale = 0.01f, float d_min = 0.001f,
    uint8_t sub_scale = 2, uint8_t sub_min = 1, uint8_t q4_val = 5)
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
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    int head_dim = d_model / n_heads;

    auto make_q4_k_weight = [&](int rows, int cols) -> std::pair<void*, Tensor> {
        int blocks_per_row = cols / 256;
        size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
        size_t total_bytes = total_blocks * 144;
        auto* buf = new uint8_t[total_bytes];
        for (size_t i = 0; i < total_blocks; ++i) {
            auto blk = make_q4_k_block(d_scale, d_min, sub_scale, sub_min, q4_val);
            std::memcpy(buf + i * 144, blk.data(), 144);
        }
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::INT4, 2, shape, false);
        return {buf, t};
    };

    auto make_fp16_weight = [](int rows, int cols) -> std::pair<void*, Tensor> {
        size_t n = static_cast<size_t>(rows) * cols;
        auto* buf = new uint16_t[n];
        for (size_t i = 0; i < n; ++i) buf[i] = float_to_fp16(0.01f);
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::FP16, 2, shape, false);
        return {buf, t};
    };

    auto make_norm_weight = [](int dim) -> std::pair<void*, Tensor> {
        auto* buf = new uint16_t[dim];
        uint16_t one = float_to_fp16(1.0f);
        for (int i = 0; i < dim; ++i) buf[i] = one;
        int64_t shape[4] = {static_cast<int64_t>(dim), 0, 0, 0};
        Tensor t(buf, DType::FP16, 1, shape, false);
        return {buf, t};
    };

    // Token embedding [vocab_size, d_model] in FP16
    auto [tok_emb_buf, tok_emb] = make_fp16_weight(vocab_size, d_model);
    model->tok_emb_ = tok_emb;
    model->tok_emb_qtype_ = GGMLQuantType::F16;

    // Output norm [d_model]
    auto [out_norm_buf, out_norm] = make_norm_weight(d_model);
    model->out_norm_ = out_norm;
    model->out_norm_qtype_ = GGMLQuantType::F16;

    // Output projection [vocab_size, d_model] in FP16
    auto [out_proj_buf, out_proj] = make_fp16_weight(vocab_size, d_model);
    model->out_proj_ = out_proj;
    model->out_proj_qtype_ = GGMLQuantType::F16;

    model->layers_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        auto& ly = model->layers_[l];

        auto [wq_buf, wq] = make_q4_k_weight(n_heads * head_dim, d_model);
        ly.wq = wq; ly.wq_qtype = GGMLQuantType::Q4_K;

        auto [wk_buf, wk] = make_q4_k_weight(n_kv_heads * head_dim, d_model);
        ly.wk = wk; ly.wk_qtype = GGMLQuantType::Q4_K;

        auto [wv_buf, wv] = make_q4_k_weight(n_kv_heads * head_dim, d_model);
        ly.wv = wv; ly.wv_qtype = GGMLQuantType::Q4_K;

        auto [wo_buf, wo] = make_q4_k_weight(d_model, n_heads * head_dim);
        ly.wo = wo; ly.wo_qtype = GGMLQuantType::Q4_K;

        auto [wg_buf, wg] = make_q4_k_weight(d_ff, d_model);
        ly.w_gate = wg; ly.w_gate_qtype = GGMLQuantType::Q4_K;

        auto [wu_buf, wu] = make_q4_k_weight(d_ff, d_model);
        ly.w_up = wu; ly.w_up_qtype = GGMLQuantType::Q4_K;

        auto [wd_buf, wd] = make_q4_k_weight(d_model, d_ff);
        ly.w_down = wd; ly.w_down_qtype = GGMLQuantType::Q4_K;

        auto [an_buf, an] = make_norm_weight(d_model);
        ly.attn_norm = an;
        auto [fn_buf, fn] = make_norm_weight(d_model);
        ly.ffn_norm = fn;
    }

    return model;
}

// Build a Model with synthetic Q5_K weights for testing.
static std::unique_ptr<Model> make_q5_k_test_model(
    int d_model, int d_ff, int n_heads, int n_kv_heads, int n_layers,
    int vocab_size, float d_scale = 0.01f, float d_min = 0.001f,
    uint8_t sub_scale = 2, uint8_t sub_min = 1,
    uint8_t q4_val = 5, bool high_bit = false)
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
    cfg.max_seq_len = 128;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    int head_dim = d_model / n_heads;

    auto make_q5_k_weight = [&](int rows, int cols) -> std::pair<void*, Tensor> {
        int blocks_per_row = cols / 256;
        size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
        size_t total_bytes = total_blocks * 176;
        auto* buf = new uint8_t[total_bytes];
        for (size_t i = 0; i < total_blocks; ++i) {
            auto blk = make_q5_k_block(d_scale, d_min, sub_scale, sub_min, q4_val, high_bit);
            std::memcpy(buf + i * 176, blk.data(), 176);
        }
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::INT4, 2, shape, false);
        return {buf, t};
    };

    auto make_fp16_weight = [](int rows, int cols) -> std::pair<void*, Tensor> {
        size_t n = static_cast<size_t>(rows) * cols;
        auto* buf = new uint16_t[n];
        for (size_t i = 0; i < n; ++i) buf[i] = float_to_fp16(0.01f);
        int64_t shape[4] = {static_cast<int64_t>(rows), static_cast<int64_t>(cols), 0, 0};
        Tensor t(buf, DType::FP16, 2, shape, false);
        return {buf, t};
    };

    auto make_norm_weight = [](int dim) -> std::pair<void*, Tensor> {
        auto* buf = new uint16_t[dim];
        uint16_t one = float_to_fp16(1.0f);
        for (int i = 0; i < dim; ++i) buf[i] = one;
        int64_t shape[4] = {static_cast<int64_t>(dim), 0, 0, 0};
        Tensor t(buf, DType::FP16, 1, shape, false);
        return {buf, t};
    };

    auto [tok_emb_buf, tok_emb] = make_fp16_weight(vocab_size, d_model);
    model->tok_emb_ = tok_emb;
    model->tok_emb_qtype_ = GGMLQuantType::F16;

    auto [out_norm_buf, out_norm] = make_norm_weight(d_model);
    model->out_norm_ = out_norm;
    model->out_norm_qtype_ = GGMLQuantType::F16;

    auto [out_proj_buf, out_proj] = make_fp16_weight(vocab_size, d_model);
    model->out_proj_ = out_proj;
    model->out_proj_qtype_ = GGMLQuantType::F16;

    model->layers_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        auto& ly = model->layers_[l];

        auto [wq_buf, wq] = make_q5_k_weight(n_heads * head_dim, d_model);
        ly.wq = wq; ly.wq_qtype = GGMLQuantType::Q5_K;

        auto [wk_buf, wk] = make_q5_k_weight(n_kv_heads * head_dim, d_model);
        ly.wk = wk; ly.wk_qtype = GGMLQuantType::Q5_K;

        auto [wv_buf, wv] = make_q5_k_weight(n_kv_heads * head_dim, d_model);
        ly.wv = wv; ly.wv_qtype = GGMLQuantType::Q5_K;

        auto [wo_buf, wo] = make_q5_k_weight(d_model, n_heads * head_dim);
        ly.wo = wo; ly.wo_qtype = GGMLQuantType::Q5_K;

        auto [wg_buf, wg] = make_q5_k_weight(d_ff, d_model);
        ly.w_gate = wg; ly.w_gate_qtype = GGMLQuantType::Q5_K;

        auto [wu_buf, wu] = make_q5_k_weight(d_ff, d_model);
        ly.w_up = wu; ly.w_up_qtype = GGMLQuantType::Q5_K;

        auto [wd_buf, wd] = make_q5_k_weight(d_model, d_ff);
        ly.w_down = wd; ly.w_down_qtype = GGMLQuantType::Q5_K;

        auto [an_buf, an] = make_norm_weight(d_model);
        ly.attn_norm = an;
        auto [fn_buf, fn] = make_norm_weight(d_model);
        ly.ffn_norm = fn;
    }

    return model;
}

// ===========================================================================
// Test 13: Q4_K weight upload keeps raw quantized bytes on GPU
// ===========================================================================
TEST(QuantIntegrationTest, Q4_KWeightUpload) {
    SKIP_IF_NO_CUDA();

    auto model = make_q4_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    const auto& ly = model->layer(0);
    EXPECT_TRUE(ly.wq.on_device);
    EXPECT_EQ(ly.wq.ndim, 2);
    EXPECT_EQ(ly.wq.shape[0], 256);  // n_heads * head_dim
    EXPECT_EQ(ly.wq.shape[1], 256);  // d_model

    // Verify raw bytes are on GPU (read back first block of row 0)
    std::vector<uint8_t> h_raw(144);
    cudaMemcpy(h_raw.data(), ly.wq.data, 144, cudaMemcpyDeviceToHost);

    // Check d (first 2 bytes = fp16 of 0.01)
    uint16_t d_bits;
    std::memcpy(&d_bits, h_raw.data(), 2);
    float d_val = fp16_to_float(d_bits);
    EXPECT_NEAR(d_val, 0.01f, 0.001f);

    // Check dmin (bytes 2-3)
    uint16_t dmin_bits;
    std::memcpy(&dmin_bits, h_raw.data() + 2, 2);
    float dmin_val = fp16_to_float(dmin_bits);
    EXPECT_NEAR(dmin_val, 0.001f, 0.001f);
}

// ===========================================================================
// Test 14: Q4_K forward pass through executor (dp4a GEMV + dequant paths)
// ===========================================================================
TEST(QuantIntegrationTest, Q4_KForwardPass) {
    SKIP_IF_NO_CUDA();

    auto model = make_q4_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {0, 1, 2};
    std::vector<int> h_positions = {0, 1, 2};
    int n_tokens = 3;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    // Verify logits are not NaN/Inf
    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    ASSERT_NE(logits.data, nullptr);
    EXPECT_EQ(logits.shape[0], 1);
    EXPECT_EQ(logits.shape[1], 32);

    std::vector<float> h_logits(32);
    cudaMemcpy(h_logits.data(), logits.data, 32 * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i) {
        EXPECT_FALSE(std::isnan(h_logits[i])) << "Q4_K logit NaN at " << i;
        EXPECT_FALSE(std::isinf(h_logits[i])) << "Q4_K logit Inf at " << i;
    }

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 15: Q4_K deterministic greedy decode
// ===========================================================================
TEST(QuantIntegrationTest, Q4_KDeterministic) {
    SKIP_IF_NO_CUDA();

    auto model = make_q4_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {3, 7};
    std::vector<int> h_positions = {0, 1};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 2 * sizeof(int32_t));
    cudaMalloc(&d_positions, 2 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 2 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 2 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 2;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token1 = executor.forward(state, nullptr);
    int32_t token2 = executor.forward(state, nullptr);
    EXPECT_EQ(token1, token2) << "Q4_K greedy should be deterministic";

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 16: Q4_K multi-layer forward pass
// ===========================================================================
TEST(QuantIntegrationTest, Q4_KMultiLayer) {
    SKIP_IF_NO_CUDA();

    auto model = make_q4_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/4, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {1, 2, 3};
    std::vector<int> h_positions = {0, 1, 2};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 3 * sizeof(int32_t));
    cudaMalloc(&d_positions, 3 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 3 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 3 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 3;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 17: Q5_K weight upload keeps raw quantized bytes on GPU
// ===========================================================================
TEST(QuantIntegrationTest, Q5_KWeightUpload) {
    SKIP_IF_NO_CUDA();

    auto model = make_q5_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    const auto& ly = model->layer(0);
    EXPECT_TRUE(ly.wq.on_device);
    EXPECT_EQ(ly.wq.ndim, 2);
    EXPECT_EQ(ly.wq.shape[0], 256);
    EXPECT_EQ(ly.wq.shape[1], 256);

    // Verify raw bytes on GPU (read back first block)
    std::vector<uint8_t> h_raw(176);
    cudaMemcpy(h_raw.data(), ly.wq.data, 176, cudaMemcpyDeviceToHost);

    uint16_t d_bits;
    std::memcpy(&d_bits, h_raw.data(), 2);
    float d_val = fp16_to_float(d_bits);
    EXPECT_NEAR(d_val, 0.01f, 0.001f);
}

// ===========================================================================
// Test 18: Q5_K forward pass through executor
// ===========================================================================
TEST(QuantIntegrationTest, Q5_KForwardPass) {
    SKIP_IF_NO_CUDA();

    auto model = make_q5_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/1, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {0, 1, 2};
    std::vector<int> h_positions = {0, 1, 2};
    int n_tokens = 3;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    // Verify logits are not NaN/Inf
    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    ASSERT_NE(logits.data, nullptr);
    EXPECT_EQ(logits.shape[0], 1);
    EXPECT_EQ(logits.shape[1], 32);

    std::vector<float> h_logits(32);
    cudaMemcpy(h_logits.data(), logits.data, 32 * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; ++i) {
        EXPECT_FALSE(std::isnan(h_logits[i])) << "Q5_K logit NaN at " << i;
        EXPECT_FALSE(std::isinf(h_logits[i])) << "Q5_K logit Inf at " << i;
    }

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 19: Q5_K multi-layer forward pass
// ===========================================================================
TEST(QuantIntegrationTest, Q5_KMultiLayer) {
    SKIP_IF_NO_CUDA();

    auto model = make_q5_k_test_model(
        /*d_model=*/256, /*d_ff=*/256, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_layers=*/4, /*vocab_size=*/32);
    ASSERT_TRUE(model->upload_weights_gpu(DType::FP16, nullptr));

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*model, DType::FP16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));

    std::vector<int32_t> h_tokens = {1, 2, 3};
    std::vector<int> h_positions = {0, 1, 2};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, 3 * sizeof(int32_t));
    cudaMalloc(&d_positions, 3 * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), 3 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), 3 * sizeof(int),
               cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = 3;
    state.is_prefill = true;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 32);

    cudaFree(d_tokens);
    cudaFree(d_positions);
}

// ===========================================================================
// Test 20: Q4_K GPU dequant correctness (dp4a reference check)
// ===========================================================================
TEST(QuantIntegrationTest, Q4_KDequantCorrectness) {
    SKIP_IF_NO_CUDA();

    // Create a single Q4_K block with known values and verify GPU dequant
    const int rows = 1;
    const int cols = 256;
    const float d_scale = 0.5f;
    const float d_min = 0.1f;
    const uint8_t sub_sc = 3;
    const uint8_t sub_mn = 2;
    const uint8_t q4_val = 7;

    auto blk = make_q4_k_block(d_scale, d_min, sub_sc, sub_mn, q4_val);

    // Upload raw block to GPU
    void* d_raw = nullptr;
    cudaMalloc(&d_raw, 144);
    cudaMemcpy(d_raw, blk.data(), 144, cudaMemcpyHostToDevice);

    // Dequant on GPU
    void* d_fp16 = nullptr;
    cudaMalloc(&d_fp16, cols * sizeof(uint16_t));
    dequant_gpu(d_raw, d_fp16, GGMLQuantType::Q4_K, rows, cols, nullptr);
    cudaDeviceSynchronize();

    // Read back
    std::vector<uint16_t> h_fp16(cols);
    cudaMemcpy(h_fp16.data(), d_fp16, cols * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    // Compute CPU reference for sub-blocks 0-3 (simple packing)
    // val = d * sc_val * q4 - dmin * min_val
    // For sub-blocks 0-3: sc_val = sub_sc, min_val = sub_mn
    float expected_03 = d_scale * static_cast<float>(sub_sc) * static_cast<float>(q4_val)
                       - d_min * static_cast<float>(sub_mn);

    // Check first 128 elements (sub-blocks 0-3)
    for (int i = 0; i < 128; ++i) {
        float got = fp16_to_float(h_fp16[i]);
        EXPECT_NEAR(got, expected_03, 0.05f)
            << "Q4_K dequant mismatch at element " << i;
    }

    cudaFree(d_raw);
    cudaFree(d_fp16);
}

// ===========================================================================
// Test 21: Q5_K GPU dequant correctness
// ===========================================================================
TEST(QuantIntegrationTest, Q5_KDequantCorrectness) {
    SKIP_IF_NO_CUDA();

    const int rows = 1;
    const int cols = 256;
    const float d_scale = 0.5f;
    const float d_min = 0.1f;
    const uint8_t sub_sc = 3;
    const uint8_t sub_mn = 2;
    const uint8_t q4_val = 7;

    // Test with high_bit=false: q5 = q4_val (5th bit is 0)
    auto blk0 = make_q5_k_block(d_scale, d_min, sub_sc, sub_mn, q4_val, false);
    void* d_raw0 = nullptr;
    cudaMalloc(&d_raw0, 176);
    cudaMemcpy(d_raw0, blk0.data(), 176, cudaMemcpyHostToDevice);

    void* d_fp16_0 = nullptr;
    cudaMalloc(&d_fp16_0, cols * sizeof(uint16_t));
    dequant_gpu(d_raw0, d_fp16_0, GGMLQuantType::Q5_K, rows, cols, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint16_t> h_fp16_0(cols);
    cudaMemcpy(h_fp16_0.data(), d_fp16_0, cols * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    // q5 = q4_val | (0 << 4) = q4_val
    float expected_no_high = d_scale * static_cast<float>(sub_sc) * static_cast<float>(q4_val)
                            - d_min * static_cast<float>(sub_mn);

    for (int i = 0; i < 128; ++i) {
        float got = fp16_to_float(h_fp16_0[i]);
        EXPECT_NEAR(got, expected_no_high, 0.05f)
            << "Q5_K dequant (no high) mismatch at " << i;
    }

    // Test with high_bit=true: q5 = q4_val | 16
    auto blk1 = make_q5_k_block(d_scale, d_min, sub_sc, sub_mn, q4_val, true);
    void* d_raw1 = nullptr;
    cudaMalloc(&d_raw1, 176);
    cudaMemcpy(d_raw1, blk1.data(), 176, cudaMemcpyHostToDevice);

    void* d_fp16_1 = nullptr;
    cudaMalloc(&d_fp16_1, cols * sizeof(uint16_t));
    dequant_gpu(d_raw1, d_fp16_1, GGMLQuantType::Q5_K, rows, cols, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint16_t> h_fp16_1(cols);
    cudaMemcpy(h_fp16_1.data(), d_fp16_1, cols * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    // q5 = q4_val | 16 = q4_val + 16
    int q5_val = static_cast<int>(q4_val) | 16;
    float expected_high = d_scale * static_cast<float>(sub_sc) * static_cast<float>(q5_val)
                         - d_min * static_cast<float>(sub_mn);

    for (int i = 0; i < 128; ++i) {
        float got = fp16_to_float(h_fp16_1[i]);
        EXPECT_NEAR(got, expected_high, 0.1f)
            << "Q5_K dequant (high bit) mismatch at " << i;
    }

    // Verify high bit actually makes a difference
    float diff = expected_high - expected_no_high;
    EXPECT_GT(std::abs(diff), 0.1f) << "High bit should change dequant value";

    cudaFree(d_raw0);
    cudaFree(d_fp16_0);
    cudaFree(d_raw1);
    cudaFree(d_fp16_1);
}

} // namespace
} // namespace imp

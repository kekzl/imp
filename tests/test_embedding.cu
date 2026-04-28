#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/embedding.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>

namespace imp {
namespace {

Tensor make_gpu_tensor(const float* host_data, QType dtype,
                       std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    if (dtype == QType::F32) {
        cudaMemcpy(t.data, host_data, t.nbytes(), cudaMemcpyHostToDevice);
    } else if (dtype == QType::F16) {
        std::vector<half> h(t.numel());
        for (int64_t j = 0; j < t.numel(); j++)
            h[j] = __float2half(host_data[j]);
        cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    }
    return t;
}

Tensor alloc_gpu_tensor(QType dtype, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = dtype;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_gpu_tensor(const Tensor& t) {
    std::vector<float> result(t.numel());
    if (t.qtype == QType::F32) {
        cudaMemcpy(result.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    } else if (t.qtype == QType::F16) {
        std::vector<half> h(t.numel());
        cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
        for (int64_t j = 0; j < t.numel(); j++)
            result[j] = __half2float(h[j]);
    }
    return result;
}

void free_gpu_tensor(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// Upload token IDs to device (embedding kernels read from device memory)
int32_t* upload_token_ids(const std::vector<int32_t>& ids) {
    int32_t* d_ids = nullptr;
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return d_ids;
}

// =========================================================================
// FP32 embedding tests
// =========================================================================

TEST(EmbeddingTest, BasicFP32) {
    constexpr int vocab = 4;
    constexpr int d_model = 3;
    std::vector<float> h_table = {
        1.0f, 2.0f, 3.0f,   // token 0
        4.0f, 5.0f, 6.0f,   // token 1
        7.0f, 8.0f, 9.0f,   // token 2
        10.0f, 11.0f, 12.0f // token 3
    };

    std::vector<int32_t> token_ids = {2, 0, 3};
    int n_tokens = static_cast<int>(token_ids.size());
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {n_tokens, d_model});

    embedding_lookup(d_table, d_ids, n_tokens, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Token 2 -> row 2: {7, 8, 9}
    EXPECT_NEAR(h_out[0], 7.0f, 1e-5f);
    EXPECT_NEAR(h_out[1], 8.0f, 1e-5f);
    EXPECT_NEAR(h_out[2], 9.0f, 1e-5f);

    // Token 0 -> row 0: {1, 2, 3}
    EXPECT_NEAR(h_out[3], 1.0f, 1e-5f);
    EXPECT_NEAR(h_out[4], 2.0f, 1e-5f);
    EXPECT_NEAR(h_out[5], 3.0f, 1e-5f);

    // Token 3 -> row 3: {10, 11, 12}
    EXPECT_NEAR(h_out[6], 10.0f, 1e-5f);
    EXPECT_NEAR(h_out[7], 11.0f, 1e-5f);
    EXPECT_NEAR(h_out[8], 12.0f, 1e-5f);

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

TEST(EmbeddingTest, SingleToken) {
    constexpr int vocab = 3;
    constexpr int d_model = 4;
    std::vector<float> h_table = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    std::vector<int32_t> token_ids = {1};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {1, d_model});

    embedding_lookup(d_table, d_ids, 1, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    EXPECT_NEAR(h_out[0], 5.0f, 1e-5f);
    EXPECT_NEAR(h_out[1], 6.0f, 1e-5f);
    EXPECT_NEAR(h_out[2], 7.0f, 1e-5f);
    EXPECT_NEAR(h_out[3], 8.0f, 1e-5f);

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

TEST(EmbeddingTest, RepeatedTokens) {
    constexpr int vocab = 2;
    constexpr int d_model = 4;
    std::vector<float> h_table = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    std::vector<int32_t> token_ids = {1, 1, 1};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {3, d_model});

    embedding_lookup(d_table, d_ids, 3, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);
    for (int t = 0; t < 3; t++) {
        for (int d = 0; d < d_model; d++) {
            EXPECT_NEAR(h_out[t * d_model + d], h_table[1 * d_model + d], 1e-5f)
                << "Mismatch at token " << t << " dim " << d;
        }
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

// =========================================================================
// FP16 embedding tests
// =========================================================================

TEST(EmbeddingTest, FP16Lookup) {
    constexpr int vocab = 4;
    constexpr int d_model = 8;
    std::vector<float> h_table(vocab * d_model);
    for (int i = 0; i < vocab * d_model; i++)
        h_table[i] = 0.1f * static_cast<float>(i);

    std::vector<int32_t> token_ids = {3, 1};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F16, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F16, {2, d_model});

    embedding_lookup(d_table, d_ids, 2, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Token 3 -> row 3
    for (int d = 0; d < d_model; d++) {
        float expected = __half2float(__float2half(h_table[3 * d_model + d]));
        EXPECT_NEAR(h_out[d], expected, 1e-2f)
            << "FP16 mismatch at token 3 dim " << d;
    }

    // Token 1 -> row 1
    for (int d = 0; d < d_model; d++) {
        float expected = __half2float(__float2half(h_table[1 * d_model + d]));
        EXPECT_NEAR(h_out[d_model + d], expected, 1e-2f)
            << "FP16 mismatch at token 1 dim " << d;
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

// =========================================================================
// Large d_model (vectorized path)
// =========================================================================

TEST(EmbeddingTest, LargeDModel) {
    constexpr int vocab = 8;
    constexpr int d_model = 256;
    std::vector<float> h_table(vocab * d_model);
    for (int i = 0; i < vocab * d_model; i++)
        h_table[i] = std::sin(static_cast<float>(i) * 0.01f);

    std::vector<int32_t> token_ids = {5, 0, 7};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {3, d_model});

    embedding_lookup(d_table, d_ids, 3, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    for (int t = 0; t < 3; t++) {
        int tid = token_ids[t];
        for (int d = 0; d < d_model; d++) {
            EXPECT_NEAR(h_out[t * d_model + d], h_table[tid * d_model + d], 1e-5f)
                << "Large d_model mismatch at token " << t << " dim " << d;
        }
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

TEST(EmbeddingTest, NonAlignedDModel) {
    constexpr int vocab = 3;
    constexpr int d_model = 13;
    std::vector<float> h_table(vocab * d_model);
    for (int i = 0; i < vocab * d_model; i++)
        h_table[i] = static_cast<float>(i) * 0.5f;

    std::vector<int32_t> token_ids = {2, 0};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {2, d_model});

    embedding_lookup(d_table, d_ids, 2, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    for (int d = 0; d < d_model; d++) {
        EXPECT_NEAR(h_out[d], h_table[2 * d_model + d], 1e-5f)
            << "Non-aligned mismatch at token 0 dim " << d;
        EXPECT_NEAR(h_out[d_model + d], h_table[0 * d_model + d], 1e-5f)
            << "Non-aligned mismatch at token 1 dim " << d;
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

// =========================================================================
// Device-side embedding lookup
// =========================================================================

TEST(EmbeddingTest, DeviceSideLookup) {
    constexpr int vocab = 4;
    constexpr int d_model = 8;
    std::vector<float> h_table(vocab * d_model);
    for (int i = 0; i < vocab * d_model; i++)
        h_table[i] = static_cast<float>(i);

    int32_t token_id = 2;
    int32_t* d_token_id = nullptr;
    cudaMalloc(&d_token_id, sizeof(int32_t));
    cudaMemcpy(d_token_id, &token_id, sizeof(int32_t), cudaMemcpyHostToDevice);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F16, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F16, {1, d_model});

    embedding_lookup_from_device(d_table, d_token_id, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    for (int d = 0; d < d_model; d++) {
        float expected = __half2float(__float2half(h_table[2 * d_model + d]));
        EXPECT_NEAR(h_out[d], expected, 1e-2f)
            << "Device-side mismatch at dim " << d;
    }

    cudaFree(d_token_id);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

// =========================================================================
// Boundary token IDs: first and last row of embedding table
// =========================================================================

TEST(EmbeddingTest, BoundaryTokenIds) {
    constexpr int vocab = 64;
    constexpr int d_model = 16;
    std::vector<float> h_table(vocab * d_model);
    for (int i = 0; i < vocab * d_model; i++)
        h_table[i] = static_cast<float>(i) * 0.01f;

    // Request first (0) and last (vocab-1) token
    std::vector<int32_t> token_ids = {0, vocab - 1};
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {2, d_model});

    embedding_lookup(d_table, d_ids, 2, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    // Token 0 -> row 0
    for (int d = 0; d < d_model; d++) {
        EXPECT_NEAR(h_out[d], h_table[d], 1e-5f)
            << "Boundary first token mismatch at dim " << d;
    }
    // Token vocab-1 -> last row
    for (int d = 0; d < d_model; d++) {
        EXPECT_NEAR(h_out[d_model + d], h_table[(vocab - 1) * d_model + d], 1e-5f)
            << "Boundary last token mismatch at dim " << d;
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

// =========================================================================
// Large vocab (150k, typical LLM size) — verify correct lookup at scale
// =========================================================================

TEST(EmbeddingTest, LargeVocab) {
    constexpr int vocab = 150000;
    constexpr int d_model = 32;

    // Fill table with deterministic pattern: row r, col c -> r * 0.001 + c
    std::vector<float> h_table(vocab * d_model);
    for (int r = 0; r < vocab; r++) {
        for (int c = 0; c < d_model; c++) {
            h_table[r * d_model + c] = static_cast<float>(r) * 0.001f
                                       + static_cast<float>(c);
        }
    }

    // Sample a few token IDs spread across the range
    std::vector<int32_t> token_ids = {0, 42, 1337, 99999, 149999};
    int n_tokens = static_cast<int>(token_ids.size());
    int32_t* d_ids = upload_token_ids(token_ids);

    Tensor d_table = make_gpu_tensor(h_table.data(), QType::F32, {vocab, d_model});
    Tensor d_out   = alloc_gpu_tensor(QType::F32, {n_tokens, d_model});

    embedding_lookup(d_table, d_ids, n_tokens, d_out);
    cudaDeviceSynchronize();

    auto h_out = read_gpu_tensor(d_out);

    for (int t = 0; t < n_tokens; t++) {
        int tid = token_ids[t];
        for (int d = 0; d < d_model; d++) {
            EXPECT_NEAR(h_out[t * d_model + d], h_table[tid * d_model + d], 1e-4f)
                << "LargeVocab mismatch at token_id=" << tid << " dim " << d;
        }
    }

    cudaFree(d_ids);
    free_gpu_tensor(d_table);
    free_gpu_tensor(d_out);
}

} // namespace
} // namespace imp

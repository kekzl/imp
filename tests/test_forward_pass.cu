#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "graph/executor.h"
#include "model/model.h"
#include "memory/kv_cache.h"
#include "core/logging.h"

#include <vector>
#include <cmath>
#include <random>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helper: allocate a random FP16 GPU tensor
// ---------------------------------------------------------------------------
static Tensor make_random_gpu_tensor(int64_t rows, int64_t cols, unsigned seed = 42) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.compute_strides();
    t.on_device = true;

    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<half> h(rows * cols);
    for (auto& v : h) v = __float2half(dist(gen));

    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

static Tensor make_ones_gpu_tensor(int64_t rows, int64_t cols) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = (rows == 1 && cols > 1) ? 1 : 2;
    if (t.ndim == 1) {
        t.shape[0] = cols;
    } else {
        t.shape[0] = rows;
        t.shape[1] = cols;
    }
    t.compute_strides();
    t.on_device = true;

    std::vector<half> h(rows * cols, __float2half(1.0f));
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

static void free_tensor(Tensor& t) {
    if (t.data && t.on_device) { cudaFree(t.data); t.data = nullptr; }
}

// ---------------------------------------------------------------------------
// Build a minimal 1-layer transformer Model with random FP16 weights.
// Config: d_model=128, n_heads=4, n_kv_heads=4, head_dim=32,
//         d_ff=512, vocab_size=256, 1 layer, SwiGLU activation.
// ---------------------------------------------------------------------------
struct SyntheticModel {
    Model model;
    std::vector<void*> allocs;  // GPU allocations to free

    bool build() {
        auto& cfg = model.config_;
        cfg.arch = ModelArch::LLAMA;
        cfg.n_layers = 1;
        cfg.d_model = 128;
        cfg.d_ff = 512;
        cfg.vocab_size = 256;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 4;
        cfg.head_dim = 32;
        cfg.rms_norm_eps = 1e-5f;
        cfg.rope_theta = 10000.0f;
        cfg.rope_neox = false;
        cfg.max_seq_len = 64;
        cfg.ffn_activation = FFNActivation::SWIGLU;

        // Token embedding: [vocab_size, d_model]
        model.tok_emb_ = make_random_gpu_tensor(cfg.vocab_size, cfg.d_model, 1);
        allocs.push_back(model.tok_emb_.data);

        // Output norm: [d_model] (RMSNorm weights, init to 1.0)
        model.out_norm_ = make_ones_gpu_tensor(1, cfg.d_model);
        allocs.push_back(model.out_norm_.data);

        // Output projection: [vocab_size, d_model]
        model.out_proj_ = make_random_gpu_tensor(cfg.vocab_size, cfg.d_model, 2);
        allocs.push_back(model.out_proj_.data);

        // Single transformer layer
        model.layers_.resize(1);
        auto& ly = model.layers_[0];

        // Attention weights: Q/K/V/O projections [n_heads*head_dim, d_model]
        int qkv_dim = cfg.n_heads * cfg.head_dim;  // = d_model for non-GQA
        int kv_dim = cfg.n_kv_heads * cfg.head_dim;
        ly.wq = make_random_gpu_tensor(qkv_dim, cfg.d_model, 10);
        ly.wk = make_random_gpu_tensor(kv_dim, cfg.d_model, 11);
        ly.wv = make_random_gpu_tensor(kv_dim, cfg.d_model, 12);
        ly.wo = make_random_gpu_tensor(cfg.d_model, qkv_dim, 13);
        allocs.push_back(ly.wq.data);
        allocs.push_back(ly.wk.data);
        allocs.push_back(ly.wv.data);
        allocs.push_back(ly.wo.data);

        // Attention norm: [d_model]
        ly.attn_norm = make_ones_gpu_tensor(1, cfg.d_model);
        allocs.push_back(ly.attn_norm.data);

        // FFN weights: gate [d_ff, d_model], up [d_ff, d_model], down [d_model, d_ff]
        ly.w_gate = make_random_gpu_tensor(cfg.d_ff, cfg.d_model, 20);
        ly.w_up = make_random_gpu_tensor(cfg.d_ff, cfg.d_model, 21);
        ly.w_down = make_random_gpu_tensor(cfg.d_model, cfg.d_ff, 22);
        allocs.push_back(ly.w_gate.data);
        allocs.push_back(ly.w_up.data);
        allocs.push_back(ly.w_down.data);

        // FFN norm: [d_model]
        ly.ffn_norm = make_ones_gpu_tensor(1, cfg.d_model);
        allocs.push_back(ly.ffn_norm.data);

        model.gpu_weights_ready_ = true;
        return true;
    }

    ~SyntheticModel() {
        // Clear model pointers before freeing to avoid double-free in Model dtor
        model.tok_emb_.data = nullptr;
        model.out_norm_.data = nullptr;
        model.out_proj_.data = nullptr;
        for (auto& ly : model.layers_) {
            ly.wq.data = ly.wk.data = ly.wv.data = ly.wo.data = nullptr;
            ly.attn_norm.data = nullptr;
            ly.w_gate.data = ly.w_up.data = ly.w_down.data = nullptr;
            ly.ffn_norm.data = nullptr;
        }
        model.gpu_allocations_.clear();
        for (void* p : allocs) cudaFree(p);
    }
};

// ---------------------------------------------------------------------------
// Test: forward_logits with synthetic model produces correct shape + finite values
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, SyntheticModelForwardLogits) {
    SyntheticModel synth;
    ASSERT_TRUE(synth.build());

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(synth.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    // Prepare input: 4 tokens
    const int n_tokens = 4;
    int32_t h_tokens[4] = {1, 42, 100, 200};
    int h_positions[4] = {0, 1, 2, 3};

    int32_t* d_tokens;
    int* d_positions;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions, n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    // KV cache: 1 layer, 4 kv_heads, head_dim=32, FP16
    KVCache cache(1, 4, 32, DType::FP16, 8);  // 8 blocks * 16 tokens = 128 max

    // Block table: all in block 0
    int h_bt[1] = {0};
    int* d_bt;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMemcpy(d_bt, h_bt, sizeof(int), cudaMemcpyHostToDevice);

    // InferenceState
    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.kv_cache = &cache;
    state.block_tables = d_bt;
    state.max_blocks_per_seq = 1;
    state.max_context_len = n_tokens;

    int h_ctx_len[1] = {n_tokens};
    int* d_ctx_len;
    cudaMalloc(&d_ctx_len, sizeof(int));
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int), cudaMemcpyHostToDevice);
    state.context_lens = d_ctx_len;

    // Run forward pass
    Tensor logits_out;
    executor.forward_logits(state, logits_out, nullptr);
    cudaDeviceSynchronize();

    // Check CUDA errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    // Verify output shape: logits should be [vocab_size] or [n_tokens, vocab_size]
    int vocab = synth.model.config_.vocab_size;
    ASSERT_NE(logits_out.data, nullptr) << "Logits output is null";
    ASSERT_GE(logits_out.numel(), vocab) << "Logits should have at least vocab_size elements";

    // Read logits and verify they are finite (not NaN/Inf)
    std::vector<float> h_logits(vocab);
    if (logits_out.dtype == DType::FP32) {
        cudaMemcpy(h_logits.data(), logits_out.data, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::vector<half> tmp(vocab);
        cudaMemcpy(tmp.data(), logits_out.data, vocab * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < vocab; i++) h_logits[i] = __half2float(tmp[i]);
    }

    int nan_count = 0, inf_count = 0, zero_count = 0;
    for (int i = 0; i < vocab; i++) {
        if (std::isnan(h_logits[i])) nan_count++;
        if (std::isinf(h_logits[i])) inf_count++;
        if (h_logits[i] == 0.0f) zero_count++;
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN logits";
    EXPECT_EQ(inf_count, 0) << "Found " << inf_count << " Inf logits";
    EXPECT_LT(zero_count, vocab) << "All logits are zero — forward pass produced no signal";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    cudaFree(d_bt);
    cudaFree(d_ctx_len);
}

// ---------------------------------------------------------------------------
// Test: decode (n=1) after prefill works correctly
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, SyntheticModelDecodeAfterPrefill) {
    SyntheticModel synth;
    ASSERT_TRUE(synth.build());

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(synth.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    // KV cache
    KVCache cache(1, 4, 32, DType::FP16, 8);

    int h_bt[1] = {0};
    int* d_bt;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMemcpy(d_bt, h_bt, sizeof(int), cudaMemcpyHostToDevice);

    // Step 1: Prefill with 3 tokens
    {
        int32_t h_tokens[3] = {1, 2, 3};
        int h_pos[3] = {0, 1, 2};
        int32_t* d_tok; int* d_pos;
        cudaMalloc(&d_tok, 3 * sizeof(int32_t));
        cudaMalloc(&d_pos, 3 * sizeof(int));
        cudaMemcpy(d_tok, h_tokens, 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, h_pos, 3 * sizeof(int), cudaMemcpyHostToDevice);

        int h_ctx[1] = {3};
        int* d_ctx;
        cudaMalloc(&d_ctx, sizeof(int));
        cudaMemcpy(d_ctx, h_ctx, sizeof(int), cudaMemcpyHostToDevice);

        InferenceState pf_state;
        pf_state.token_ids = d_tok;
        pf_state.positions = d_pos;
        pf_state.n_tokens = 3;
        pf_state.is_prefill = true;
        pf_state.n_sequences = 1;
        pf_state.kv_cache = &cache;
        pf_state.block_tables = d_bt;
        pf_state.max_blocks_per_seq = 1;
        pf_state.max_context_len = 3;
        pf_state.context_lens = d_ctx;

        Tensor pf_logits;
        executor.forward_logits(pf_state, pf_logits, nullptr);
        cudaDeviceSynchronize();
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        cudaFree(d_tok);
        cudaFree(d_pos);
        cudaFree(d_ctx);
    }

    // Step 2: Decode 1 token at position 3
    {
        int32_t h_tokens[1] = {50};
        int h_pos[1] = {3};
        int32_t* d_tok; int* d_pos;
        cudaMalloc(&d_tok, sizeof(int32_t));
        cudaMalloc(&d_pos, sizeof(int));
        cudaMemcpy(d_tok, h_tokens, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, h_pos, sizeof(int), cudaMemcpyHostToDevice);

        int h_ctx[1] = {4};
        int* d_ctx;
        cudaMalloc(&d_ctx, sizeof(int));
        cudaMemcpy(d_ctx, h_ctx, sizeof(int), cudaMemcpyHostToDevice);

        InferenceState dec_state;
        dec_state.token_ids = d_tok;
        dec_state.positions = d_pos;
        dec_state.n_tokens = 1;
        dec_state.is_prefill = false;
        dec_state.n_sequences = 1;
        dec_state.kv_cache = &cache;
        dec_state.block_tables = d_bt;
        dec_state.max_blocks_per_seq = 1;
        dec_state.max_context_len = 4;
        dec_state.context_lens = d_ctx;

        Tensor dec_logits;
        executor.forward_logits(dec_state, dec_logits, nullptr);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "Decode CUDA error: " << cudaGetErrorString(err);

        // Verify finite logits
        int vocab = synth.model.config_.vocab_size;
        std::vector<float> h_logits(vocab);
        if (dec_logits.dtype == DType::FP32) {
            cudaMemcpy(h_logits.data(), dec_logits.data, vocab * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::vector<half> tmp(vocab);
            cudaMemcpy(tmp.data(), dec_logits.data, vocab * sizeof(half), cudaMemcpyDeviceToHost);
            for (int i = 0; i < vocab; i++) h_logits[i] = __half2float(tmp[i]);
        }

        int nan_count = 0;
        for (int i = 0; i < vocab; i++)
            if (std::isnan(h_logits[i])) nan_count++;
        EXPECT_EQ(nan_count, 0) << "Decode produced NaN logits";

        cudaFree(d_tok);
        cudaFree(d_pos);
        cudaFree(d_ctx);
    }

    cudaFree(d_bt);
}

} // anonymous namespace
} // namespace imp

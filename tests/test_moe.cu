#include <gtest/gtest.h>
#include "compute/moe_routing.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

static constexpr float kTolerance = 1e-4f;

// Create a device-resident Tensor from host data. Caller must free with
// free_device_tensor().
Tensor make_device_tensor(const void* host_data, DType dtype,
                          int ndim, const int64_t* shape) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) t.shape[i] = shape[i];
    t.compute_strides();
    t.on_device = true;

    size_t bytes = t.nbytes();
    EXPECT_EQ(cudaMalloc(&t.data, bytes), cudaSuccess);
    if (host_data) {
        EXPECT_EQ(cudaMemcpy(t.data, host_data, bytes,
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    return t;
}

// Allocate a zero-initialized device tensor (no host source).
Tensor make_device_tensor_zeros(DType dtype, int ndim,
                                const int64_t* shape) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) t.shape[i] = shape[i];
    t.compute_strides();
    t.on_device = true;

    size_t bytes = t.nbytes();
    EXPECT_EQ(cudaMalloc(&t.data, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(t.data, 0, bytes), cudaSuccess);
    return t;
}

// Copy a device tensor back to a host vector.
template <typename T>
std::vector<T> to_host(const Tensor& t) {
    size_t n = static_cast<size_t>(t.numel());
    std::vector<T> out(n);
    EXPECT_EQ(cudaMemcpy(out.data(), t.data, n * sizeof(T),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
}

void free_tensor(Tensor& t) {
    if (t.data && t.on_device) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

void free_routing(MoeRoutingResult& r) {
    free_tensor(r.expert_indices);
    free_tensor(r.expert_weights);
    free_tensor(r.sorted_token_ids);
    free_tensor(r.expert_offsets);
}

// ---------------------------------------------------------------------------
// CPU reference: top-k gating with softmax over selected logits
// ---------------------------------------------------------------------------
void cpu_topk_gating(const float* logits, int n_tokens, int n_experts,
                     int top_k, int* expert_indices, float* expert_weights) {
    for (int t = 0; t < n_tokens; t++) {
        std::vector<std::pair<float, int>> scores(n_experts);
        for (int e = 0; e < n_experts; e++) {
            scores[e] = {logits[t * n_experts + e], e};
        }
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first;
                          });
        // Softmax over the top-k logits
        float max_s = scores[0].first;
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            expert_weights[t * top_k + k] = expf(scores[k].first - max_s);
            sum += expert_weights[t * top_k + k];
        }
        for (int k = 0; k < top_k; k++) {
            expert_indices[t * top_k + k] = scores[k].second;
            expert_weights[t * top_k + k] /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Shared test fixture
// ---------------------------------------------------------------------------
class MoERoutingTest : public ::testing::Test {
protected:
    static constexpr int kNTokens  = 4;
    static constexpr int kNExperts = 8;
    static constexpr int kTopK     = 2;

    // Gate logits designed so that the top-2 experts for each token are
    // unambiguous:
    //   token 0 -> experts 1 (10.0) and 3 (8.0)
    //   token 1 -> experts 5 (12.0) and 7 (9.0)
    //   token 2 -> experts 0 (11.0) and 2 (7.0)
    //   token 3 -> experts 4 (15.0) and 6 (14.0)
    std::vector<float> gate_logits = {
        // token 0:  e0   e1    e2   e3   e4   e5   e6   e7
                     1.0, 10.0, 2.0, 8.0, 0.5, 1.5, 0.0, 3.0,
        // token 1:
                     0.0,  1.0, 2.0, 3.0, 0.5, 12.0, 1.0, 9.0,
        // token 2:
                    11.0,  2.0, 7.0, 1.0, 0.0,  0.5, 3.0, 0.0,
        // token 3:
                     0.0,  1.0, 2.0, 3.0, 15.0, 0.5, 14.0, 1.0,
    };

    // Expected top-2 expert indices per token (by descending logit)
    std::vector<int> expected_indices = {
        1, 3,   // token 0
        5, 7,   // token 1
        0, 2,   // token 2
        4, 6,   // token 3
    };

    Tensor d_gate;
    MoeRoutingResult routing{};

    void SetUp() override {
        int64_t shape[2] = {kNTokens, kNExperts};
        d_gate = make_device_tensor(gate_logits.data(), DType::FP32, 2, shape);

        moe_topk_gating(d_gate, kTopK, routing, /*stream=*/nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    void TearDown() override {
        free_tensor(d_gate);
        free_routing(routing);
    }
};

// ---------------------------------------------------------------------------
// Test 1: TopKSelection
// ---------------------------------------------------------------------------
TEST_F(MoERoutingTest, TopKSelection) {
    // Verify shapes
    ASSERT_EQ(routing.expert_indices.ndim, 2);
    ASSERT_EQ(routing.expert_indices.shape[0], kNTokens);
    ASSERT_EQ(routing.expert_indices.shape[1], kTopK);
    ASSERT_EQ(routing.expert_indices.dtype, DType::INT32);

    ASSERT_EQ(routing.expert_weights.ndim, 2);
    ASSERT_EQ(routing.expert_weights.shape[0], kNTokens);
    ASSERT_EQ(routing.expert_weights.shape[1], kTopK);
    ASSERT_EQ(routing.expert_weights.dtype, DType::FP32);

    auto h_indices = to_host<int32_t>(routing.expert_indices);

    // For each token, the set of selected expert ids should match expectations.
    // We compare as sets because the kernel may output them in a different
    // relative order than our reference (e.g. sorted by index rather than by
    // descending score).
    for (int t = 0; t < kNTokens; t++) {
        std::set<int> got_set(h_indices.begin() + t * kTopK,
                              h_indices.begin() + (t + 1) * kTopK);
        std::set<int> exp_set(expected_indices.begin() + t * kTopK,
                              expected_indices.begin() + (t + 1) * kTopK);
        EXPECT_EQ(got_set, exp_set)
            << "Token " << t << " selected wrong experts";
    }

    // Verify the weights correspond to a softmax of the selected logits.
    auto h_weights = to_host<float>(routing.expert_weights);

    std::vector<int>   ref_indices(kNTokens * kTopK);
    std::vector<float> ref_weights(kNTokens * kTopK);
    cpu_topk_gating(gate_logits.data(), kNTokens, kNExperts, kTopK,
                    ref_indices.data(), ref_weights.data());

    for (int t = 0; t < kNTokens; t++) {
        // Build a map from expert_id -> weight for GPU output
        std::map<int, float> gpu_map;
        for (int k = 0; k < kTopK; k++) {
            gpu_map[h_indices[t * kTopK + k]] = h_weights[t * kTopK + k];
        }
        // Build a map from expert_id -> weight for CPU reference
        std::map<int, float> ref_map;
        for (int k = 0; k < kTopK; k++) {
            ref_map[ref_indices[t * kTopK + k]] = ref_weights[t * kTopK + k];
        }

        ASSERT_EQ(gpu_map.size(), ref_map.size());
        for (const auto& [eid, w] : ref_map) {
            auto it = gpu_map.find(eid);
            ASSERT_NE(it, gpu_map.end())
                << "Token " << t << ": expert " << eid << " missing from GPU output";
            EXPECT_NEAR(it->second, w, kTolerance)
                << "Token " << t << ", expert " << eid << " weight mismatch";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: WeightNormalization
// ---------------------------------------------------------------------------
TEST_F(MoERoutingTest, WeightNormalization) {
    auto h_weights = to_host<float>(routing.expert_weights);

    for (int t = 0; t < kNTokens; t++) {
        float sum = 0.0f;
        for (int k = 0; k < kTopK; k++) {
            float w = h_weights[t * kTopK + k];
            // Each individual weight must be in (0, 1]
            EXPECT_GT(w, 0.0f) << "Token " << t << ", slot " << k;
            EXPECT_LE(w, 1.0f + kTolerance)
                << "Token " << t << ", slot " << k;
            sum += w;
        }
        // Sum of weights per token must be 1.0
        EXPECT_NEAR(sum, 1.0f, kTolerance)
            << "Token " << t << " weights do not sum to 1.0";
    }
}

// ---------------------------------------------------------------------------
// Test 3: ExpertOffsets
// ---------------------------------------------------------------------------
TEST_F(MoERoutingTest, ExpertOffsets) {
    ASSERT_EQ(routing.expert_offsets.dtype, DType::INT32);

    // expert_offsets should have n_experts + 1 elements
    int64_t expected_len = kNExperts + 1;
    ASSERT_EQ(routing.expert_offsets.numel(), expected_len);

    auto h_offsets = to_host<int32_t>(routing.expert_offsets);

    // First element must be 0
    EXPECT_EQ(h_offsets[0], 0);

    // Last element must equal total assigned tokens = n_tokens * top_k
    int total = kNTokens * kTopK;
    EXPECT_EQ(h_offsets[kNExperts], total);

    // Offsets must be monotonically non-decreasing (valid prefix sum)
    for (int e = 1; e <= kNExperts; e++) {
        EXPECT_GE(h_offsets[e], h_offsets[e - 1])
            << "expert_offsets is not non-decreasing at index " << e;
    }

    // Differences (counts per expert) must be non-negative
    for (int e = 0; e < kNExperts; e++) {
        int count = h_offsets[e + 1] - h_offsets[e];
        EXPECT_GE(count, 0) << "Negative count for expert " << e;
    }
}

// ---------------------------------------------------------------------------
// Test 4: SortedTokenIds
// ---------------------------------------------------------------------------
TEST_F(MoERoutingTest, SortedTokenIds) {
    int total = kNTokens * kTopK;
    ASSERT_EQ(routing.sorted_token_ids.dtype, DType::INT32);
    ASSERT_EQ(routing.sorted_token_ids.numel(), total);

    auto h_sorted   = to_host<int32_t>(routing.sorted_token_ids);
    auto h_offsets   = to_host<int32_t>(routing.expert_offsets);
    auto h_indices   = to_host<int32_t>(routing.expert_indices);

    // Every entry in sorted_token_ids must be a valid token index [0, n_tokens)
    for (int i = 0; i < total; i++) {
        EXPECT_GE(h_sorted[i], 0) << "Index " << i;
        EXPECT_LT(h_sorted[i], kNTokens) << "Index " << i;
    }

    // Build a set of (token, expert) assignments from expert_indices
    std::set<std::pair<int, int>> expected_assignments;
    for (int t = 0; t < kNTokens; t++) {
        for (int k = 0; k < kTopK; k++) {
            expected_assignments.insert({t, h_indices[t * kTopK + k]});
        }
    }

    // Walk sorted_token_ids, segmented by expert_offsets.
    // For each segment [offsets[e], offsets[e+1]), every token id in that
    // range must have been assigned to expert e.
    std::set<std::pair<int, int>> actual_assignments;
    for (int e = 0; e < kNExperts; e++) {
        int begin = h_offsets[e];
        int end   = h_offsets[e + 1];
        for (int i = begin; i < end; i++) {
            int token_id = h_sorted[i];
            actual_assignments.insert({token_id, e});
        }
    }

    EXPECT_EQ(actual_assignments, expected_assignments)
        << "Sorted token grouping does not match the expert assignments";
}

// ---------------------------------------------------------------------------
// Test 5: GatherScatter
// ---------------------------------------------------------------------------
TEST_F(MoERoutingTest, GatherScatter) {
    static constexpr int kDModel = 16;
    int total = kNTokens * kTopK;

    // ---- Build input tensor [n_tokens, d_model] on device ----
    std::vector<float> h_input(kNTokens * kDModel);
    for (int i = 0; i < kNTokens * kDModel; i++) {
        // Distinct values so we can detect wrong gathers.
        h_input[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }

    int64_t input_shape[2] = {kNTokens, kDModel};
    Tensor d_input = make_device_tensor(h_input.data(), DType::FP32, 2,
                                        input_shape);

    // ---- Gather ----
    int64_t gathered_shape[2] = {total, kDModel};
    Tensor d_gathered = make_device_tensor_zeros(DType::FP32, 2,
                                                 gathered_shape);

    moe_gather(d_input, routing, d_gathered, /*stream=*/nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto h_gathered = to_host<float>(d_gathered);
    auto h_sorted   = to_host<int32_t>(routing.sorted_token_ids);

    // Verify gathered[i] == input[sorted_token_ids[i]] for every row.
    for (int i = 0; i < total; i++) {
        int tok = h_sorted[i];
        ASSERT_GE(tok, 0);
        ASSERT_LT(tok, kNTokens);
        for (int d = 0; d < kDModel; d++) {
            EXPECT_NEAR(h_gathered[i * kDModel + d],
                        h_input[tok * kDModel + d],
                        kTolerance)
                << "Gather mismatch at sorted position " << i
                << ", feature " << d;
        }
    }

    // ---- Scatter (identity expert output = gathered unchanged) ----
    // The scatter should combine contributions for each token, weighted
    // by expert_weights. Since we pass the gathered tensor directly as
    // expert_output (identity transform), the expected output for token t is:
    //   output[t] = sum_k  weight[t][k] * input[t]
    //             = input[t]   (because weights sum to 1)
    int64_t output_shape[2] = {kNTokens, kDModel};
    Tensor d_output = make_device_tensor_zeros(DType::FP32, 2, output_shape);

    moe_scatter(d_gathered, routing, d_output, /*stream=*/nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto h_output  = to_host<float>(d_output);
    auto h_weights = to_host<float>(routing.expert_weights);
    auto h_indices = to_host<int32_t>(routing.expert_indices);
    auto h_offsets = to_host<int32_t>(routing.expert_offsets);

    // Build CPU reference for scatter output.
    // For each token, find which positions in sorted_token_ids correspond to
    // it, look up the weight for that (token, expert) pair, and accumulate.
    std::vector<float> ref_output(kNTokens * kDModel, 0.0f);

    // Build a map from (token, expert) -> weight
    std::map<std::pair<int,int>, float> weight_map;
    for (int t = 0; t < kNTokens; t++) {
        for (int k = 0; k < kTopK; k++) {
            int eid = h_indices[t * kTopK + k];
            float w = h_weights[t * kTopK + k];
            weight_map[{t, eid}] = w;
        }
    }

    for (int e = 0; e < kNExperts; e++) {
        int begin = h_offsets[e];
        int end   = h_offsets[e + 1];
        for (int i = begin; i < end; i++) {
            int tok = h_sorted[i];
            float w = weight_map[{tok, e}];
            for (int d = 0; d < kDModel; d++) {
                // expert_output[i] == input[tok] (identity)
                ref_output[tok * kDModel + d] +=
                    w * h_input[tok * kDModel + d];
            }
        }
    }

    for (int t = 0; t < kNTokens; t++) {
        for (int d = 0; d < kDModel; d++) {
            EXPECT_NEAR(h_output[t * kDModel + d],
                        ref_output[t * kDModel + d],
                        kTolerance)
                << "Scatter mismatch at token " << t << ", feature " << d;
        }
    }

    free_tensor(d_input);
    free_tensor(d_gathered);
    free_tensor(d_output);
}

} // namespace
} // namespace imp

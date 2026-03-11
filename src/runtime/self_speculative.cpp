#include "runtime/self_speculative.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("self_spec %s: %s", msg, cudaGetErrorString(err));
    }
}

SelfSpeculativeDecoder::~SelfSpeculativeDecoder() {
    if (d_tokens_) cudaFree(d_tokens_);
    if (d_positions_) cudaFree(d_positions_);
    if (d_block_table_) cudaFree(d_block_table_);
    if (d_ctx_len_) cudaFree(d_ctx_len_);
}

bool SelfSpeculativeDecoder::init(GraphExecutor* executor,
                                   KVCacheManager* kv_manager,
                                   KVCache* kv_cache,
                                   int n_layers,
                                   const SelfSpecConfig& config) {
    if (!executor || !kv_manager || !kv_cache) {
        IMP_LOG_ERROR("self_spec: null executor/kv_manager/kv_cache");
        return false;
    }
    if (config.spec_k < 1 || config.spec_k > 32) {
        IMP_LOG_ERROR("self_spec: spec_k must be in [1, 32], got %d", config.spec_k);
        return false;
    }

    config_ = config;
    executor_ = executor;
    kv_manager_ = kv_manager;
    kv_cache_ = kv_cache;
    exit_layer_ = (config.exit_layer > 0) ? config.exit_layer : (n_layers / 2);

    // Pre-allocate device buffers for max K+1 tokens
    int max_n = config_.spec_k + 1;
    check_cuda(cudaMalloc(&d_tokens_, max_n * sizeof(int32_t)), "malloc d_tokens");
    check_cuda(cudaMalloc(&d_positions_, max_n * sizeof(int)), "malloc d_positions");
    check_cuda(cudaMalloc(&d_ctx_len_, max_n * sizeof(int)), "malloc d_ctx_len");

    initialized_ = true;
    IMP_LOG_INFO("self_spec: initialized with spec_k=%d, exit_layer=%d/%d",
                 config_.spec_k, exit_layer_, n_layers);
    return true;
}

// ─── Helper: upload block table (single copy) ────────────────────────────────

void SelfSpeculativeDecoder::upload_block_table(int seq_id, cudaStream_t stream) {
    const auto& bt = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(bt.size());
    if (n_blocks > d_block_table_cap_) {
        if (d_block_table_) cudaFree(d_block_table_);
        d_block_table_cap_ = n_blocks + 16;
        check_cuda(cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int)),
                   "malloc d_block_table");
    }
    check_cuda(cudaMemcpyAsync(d_block_table_, bt.data(),
               n_blocks * sizeof(int), cudaMemcpyHostToDevice, stream),
               "memcpy block_table");
}

// ─── Helper: upload replicated block table for batched verify ────────────────

void SelfSpeculativeDecoder::upload_block_table_replicated(
        int seq_id, int n_copies, int max_blocks_per_seq, cudaStream_t stream) {
    const auto& bt = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(bt.size());

    int total_entries = n_copies * max_blocks_per_seq;
    if (total_entries > d_block_table_cap_) {
        if (d_block_table_) cudaFree(d_block_table_);
        d_block_table_cap_ = total_entries + 16;
        check_cuda(cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int)),
                   "malloc d_block_table replicated");
    }

    // Build replicated layout on host
    std::vector<int> replicated(total_entries, 0);
    for (int c = 0; c < n_copies; ++c) {
        for (int b = 0; b < n_blocks && b < max_blocks_per_seq; ++b) {
            replicated[c * max_blocks_per_seq + b] = bt[b];
        }
    }
    check_cuda(cudaMemcpyAsync(d_block_table_, replicated.data(),
               total_entries * sizeof(int), cudaMemcpyHostToDevice, stream),
               "memcpy block_table replicated");
}

// ─── Helper: ensure KV blocks ────────────────────────────────────────────────

void SelfSpeculativeDecoder::ensure_kv_blocks(int seq_id, int ctx_len) {
    int needed = (ctx_len + kKVBlockSize - 1) / kKVBlockSize;
    int have = static_cast<int>(kv_manager_->block_table(seq_id).size());
    while (needed > have) {
        if (kv_manager_->append_block(seq_id) < 0) break;
        have++;
    }
}

// ─── Draft token generation (early exit) ─────────────────────────────────────

std::vector<int32_t> SelfSpeculativeDecoder::draft_tokens(
        int32_t last_token, int position, int seq_id, cudaStream_t stream) {
    std::vector<int32_t> drafts;
    drafts.reserve(config_.spec_k);

    int32_t cur_token = last_token;
    int cur_pos = position;

    for (int k = 0; k < config_.spec_k; ++k) {
        int ctx_len = cur_pos + 1;
        ensure_kv_blocks(seq_id, ctx_len);
        upload_block_table(seq_id, stream);

        check_cuda(cudaMemcpyAsync(d_tokens_, &cur_token, sizeof(int32_t),
                   cudaMemcpyHostToDevice, stream), "memcpy token");
        check_cuda(cudaMemcpyAsync(d_positions_, &cur_pos, sizeof(int),
                   cudaMemcpyHostToDevice, stream), "memcpy pos");
        check_cuda(cudaMemcpyAsync(d_ctx_len_, &ctx_len, sizeof(int),
                   cudaMemcpyHostToDevice, stream), "memcpy ctx_len");

        int n_blocks = static_cast<int>(kv_manager_->block_table(seq_id).size());

        InferenceState state;
        state.token_ids = d_tokens_;
        state.positions = d_positions_;
        state.n_tokens = 1;
        state.kv_cache = kv_cache_;
        state.block_tables = d_block_table_;
        state.context_lens = d_ctx_len_;
        state.max_context_len = ctx_len;
        state.n_sequences = 1;
        state.max_blocks_per_seq = n_blocks;
        state.is_prefill = false;
        state.temperature = 0.0f;  // greedy draft
        state.top_k = 1;
        state.exit_layer = exit_layer_;

        int32_t sampled = executor_->forward(state, stream);
        drafts.push_back(sampled);
        cur_token = sampled;
        cur_pos += 1;
    }

    return drafts;
}

// ─── Batched verification (all K+1 tokens through full model in one pass) ───

SelfSpeculativeDecoder::VerifyResult
SelfSpeculativeDecoder::verify(const std::vector<int32_t>& draft,
                                int32_t last_token, int position, int seq_id,
                                float temperature, float top_p_val, int top_k_val,
                                int seed, cudaStream_t stream) {
    VerifyResult result;
    const int K = static_cast<int>(draft.size());
    if (K == 0) return result;

    const int n_verify = K + 1;

    // Ensure KV blocks for the maximum context length
    int max_ctx = position + K + 1;
    ensure_kv_blocks(seq_id, max_ctx);

    // max_blocks_per_seq MUST match paged_attention's stride:
    //   max_num_blocks = (max_context_len + block_size - 1) / block_size
    int max_blocks_per_seq = (max_ctx + kKVBlockSize - 1) / kKVBlockSize;

    // Upload replicated block table: n_verify copies, each max_blocks_per_seq entries
    upload_block_table_replicated(seq_id, n_verify, max_blocks_per_seq, stream);

    // Build token array: [last_token, draft[0], ..., draft[K-1]]
    std::vector<int32_t> verify_tokens(n_verify);
    verify_tokens[0] = last_token;
    for (int i = 0; i < K; ++i)
        verify_tokens[i + 1] = draft[i];

    // Build positions: [position, position+1, ..., position+K]
    std::vector<int> positions(n_verify);
    for (int i = 0; i < n_verify; ++i)
        positions[i] = position + i;

    // Build context lens: [position+1, position+2, ..., position+K+1]
    std::vector<int> ctx_lens(n_verify);
    for (int i = 0; i < n_verify; ++i)
        ctx_lens[i] = position + i + 1;

    // Upload to device
    check_cuda(cudaMemcpyAsync(d_tokens_, verify_tokens.data(),
               n_verify * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
               "verify memcpy tokens");
    check_cuda(cudaMemcpyAsync(d_positions_, positions.data(),
               n_verify * sizeof(int), cudaMemcpyHostToDevice, stream),
               "verify memcpy positions");
    check_cuda(cudaMemcpyAsync(d_ctx_len_, ctx_lens.data(),
               n_verify * sizeof(int), cudaMemcpyHostToDevice, stream),
               "verify memcpy ctx_lens");

    // Resize workspace for K+1 tokens BEFORE the batched forward
    executor_->resize_workspace(n_verify, stream);

    // Build inference state for batched decode
    InferenceState state;
    state.token_ids = d_tokens_;
    state.positions = d_positions_;
    state.n_tokens = n_verify;
    state.kv_cache = kv_cache_;
    state.block_tables = d_block_table_;
    state.context_lens = d_ctx_len_;
    state.max_context_len = max_ctx;
    state.n_sequences = n_verify;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.is_prefill = false;
    state.temperature = 0.0f;  // greedy for target selection
    state.top_k = 1;
    state.exit_layer = -1;  // full model
    state.per_row_lm_head = true;  // per-row Q8_1 GEMV avoids FP8 per-tensor quantization artifacts

    // forward_batch returns one sampled token per sequence
    std::vector<int32_t> targets = executor_->forward_batch(state, stream);

    // Resize workspace back to 1 for subsequent draft passes
    executor_->resize_workspace(1, stream);

    // Acceptance: compare target[i] with draft[i]
    int n_accepted = 0;
    for (int i = 0; i < K; ++i) {
        if (targets[i] == draft[i]) {
            result.accepted.push_back(draft[i]);
            n_accepted++;
        } else {
            result.n_accepted = n_accepted;
            result.next_token = targets[i];
            return result;
        }
    }

    // All K accepted — target[K] is the bonus token
    result.n_accepted = K;
    result.next_token = targets[K];
    return result;
}

// ─── Full step ───────────────────────────────────────────────────────────────

std::vector<int32_t> SelfSpeculativeDecoder::step(
        int32_t last_token, int position, int seq_id,
        float temperature, float top_p, int top_k, int seed,
        cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("self_spec: not initialized");
        return {};
    }

    // Resize workspace for single-token draft passes
    executor_->resize_workspace(1, stream);

    // 1. Draft K tokens with early exit
    std::vector<int32_t> draft = draft_tokens(last_token, position, seq_id, stream);
    total_drafted_ += draft.size();

    // 2. Verify with full model (batched: all K+1 tokens in one pass)
    VerifyResult vr = verify(draft, last_token, position, seq_id,
                              temperature, top_p, top_k, seed, stream);
    total_accepted_ += vr.n_accepted;

    IMP_LOG_INFO("self_spec: accepted %d/%d draft tokens (cumulative: %lld/%lld = %.1f%%)",
                 vr.n_accepted, static_cast<int>(draft.size()),
                 static_cast<long long>(total_accepted_),
                 static_cast<long long>(total_drafted_),
                 total_drafted_ > 0 ? 100.0 * total_accepted_ / total_drafted_ : 0.0);

    // 3. Combine: accepted + next token
    std::vector<int32_t> output;
    output.reserve(vr.n_accepted + 1);
    for (int i = 0; i < vr.n_accepted; ++i)
        output.push_back(vr.accepted[i]);
    if (vr.next_token >= 0)
        output.push_back(vr.next_token);

    return output;
}

} // namespace imp

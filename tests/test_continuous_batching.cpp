#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "runtime/batch.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "test_cuda_skip.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace imp {
namespace {

// ============================================================================
// BatchBuilder Tests
// ============================================================================

// 1. Build a single decode sequence
TEST(BatchBuilderTest, SingleDecodeSequence) {
    BatchBuilder builder;
    builder.reset();

    int block_table[] = {0, 1, 2};
    builder.add_decode_sequence(/*token=*/42, /*position=*/15, block_table, /*context_len=*/16);

    Batch batch = builder.build();

    EXPECT_EQ(batch.n_sequences, 1);
    EXPECT_EQ(batch.total_tokens, 1);
    EXPECT_EQ(batch.max_blocks_per_seq, 3);

    // Token and position
    ASSERT_EQ(batch.token_ids.size(), 1u);
    EXPECT_EQ(batch.token_ids[0], 42);
    ASSERT_EQ(batch.positions.size(), 1u);
    EXPECT_EQ(batch.positions[0], 15);

    // Context lens
    ASSERT_EQ(batch.context_lens.size(), 1u);
    EXPECT_EQ(batch.context_lens[0], 16);

    // Block tables: [1, 3] padded
    ASSERT_EQ(batch.block_tables.size(), 3u);
    EXPECT_EQ(batch.block_tables[0], 0);
    EXPECT_EQ(batch.block_tables[1], 1);
    EXPECT_EQ(batch.block_tables[2], 2);

    // Seq offsets
    ASSERT_EQ(batch.seq_offsets.size(), 2u);
    EXPECT_EQ(batch.seq_offsets[0], 0);
    EXPECT_EQ(batch.seq_offsets[1], 1);
}

// 2. Build multiple decode sequences (batched)
TEST(BatchBuilderTest, MultipleDecodeSequences) {
    BatchBuilder builder;
    builder.reset();

    // Seq 0: 2 blocks, context 20
    int bt0[] = {5, 10};
    builder.add_decode_sequence(100, 19, std::span(bt0, 2), 20);

    // Seq 1: 3 blocks, context 40
    int bt1[] = {7, 8, 9};
    builder.add_decode_sequence(200, 39, std::span(bt1, 3), 40);

    // Seq 2: 1 block, context 5
    int bt2[] = {12};
    builder.add_decode_sequence(300, 4, std::span(bt2, 1), 5);

    Batch batch = builder.build();

    EXPECT_EQ(batch.n_sequences, 3);
    EXPECT_EQ(batch.total_tokens, 3);
    EXPECT_EQ(batch.max_blocks_per_seq, 3);  // max of {2, 3, 1}

    // Tokens
    EXPECT_EQ(batch.token_ids[0], 100);
    EXPECT_EQ(batch.token_ids[1], 200);
    EXPECT_EQ(batch.token_ids[2], 300);

    // Positions
    EXPECT_EQ(batch.positions[0], 19);
    EXPECT_EQ(batch.positions[1], 39);
    EXPECT_EQ(batch.positions[2], 4);

    // Context lens
    EXPECT_EQ(batch.context_lens[0], 20);
    EXPECT_EQ(batch.context_lens[1], 40);
    EXPECT_EQ(batch.context_lens[2], 5);

    // Padded block tables: [3, 3]
    // Row 0: [5, 10, 0]
    EXPECT_EQ(batch.block_tables[0 * 3 + 0], 5);
    EXPECT_EQ(batch.block_tables[0 * 3 + 1], 10);
    EXPECT_EQ(batch.block_tables[0 * 3 + 2], 0);  // padding
    // Row 1: [7, 8, 9]
    EXPECT_EQ(batch.block_tables[1 * 3 + 0], 7);
    EXPECT_EQ(batch.block_tables[1 * 3 + 1], 8);
    EXPECT_EQ(batch.block_tables[1 * 3 + 2], 9);
    // Row 2: [12, 0, 0]
    EXPECT_EQ(batch.block_tables[2 * 3 + 0], 12);
    EXPECT_EQ(batch.block_tables[2 * 3 + 1], 0);
    EXPECT_EQ(batch.block_tables[2 * 3 + 2], 0);

    // Seq offsets: [0, 1, 2, 3]
    ASSERT_EQ(batch.seq_offsets.size(), 4u);
    EXPECT_EQ(batch.seq_offsets[0], 0);
    EXPECT_EQ(batch.seq_offsets[1], 1);
    EXPECT_EQ(batch.seq_offsets[2], 2);
    EXPECT_EQ(batch.seq_offsets[3], 3);
}

// 3. Build a prefill sequence
TEST(BatchBuilderTest, PrefillSequence) {
    BatchBuilder builder;
    builder.reset();

    int32_t tokens[] = {1, 2, 3, 4, 5};
    int bt[] = {0, 1};
    builder.add_prefill_sequence(std::span(tokens, 5), std::span(bt, 2), /*start_pos=*/0);

    Batch batch = builder.build();

    EXPECT_EQ(batch.n_sequences, 1);
    EXPECT_EQ(batch.total_tokens, 5);
    EXPECT_EQ(batch.max_blocks_per_seq, 2);

    // All 5 tokens
    ASSERT_EQ(batch.token_ids.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(batch.token_ids[i], i + 1);
    }

    // Positions 0..4
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(batch.positions[i], i);
    }

    // Context len = start_pos + n_tokens = 0 + 5 = 5
    EXPECT_EQ(batch.context_lens[0], 5);
}

// 4. Mixed prefill + decode (test reset behavior)
TEST(BatchBuilderTest, ResetClearsPreviousData) {
    BatchBuilder builder;
    builder.reset();

    int bt[] = {0};
    builder.add_decode_sequence(10, 5, std::span(bt, 1), 6);
    Batch b1 = builder.build();
    EXPECT_EQ(b1.n_sequences, 1);

    // Reset and build new batch
    builder.reset();
    builder.add_decode_sequence(20, 10, std::span(bt, 1), 11);
    builder.add_decode_sequence(30, 15, std::span(bt, 1), 16);
    Batch b2 = builder.build();
    EXPECT_EQ(b2.n_sequences, 2);
    EXPECT_EQ(b2.total_tokens, 2);
    EXPECT_EQ(b2.token_ids[0], 20);
    EXPECT_EQ(b2.token_ids[1], 30);
}

// ============================================================================
// GPUBatch Tests
// ============================================================================

// 5. GPUBatch upload and free
TEST(GPUBatchTest, UploadAndFree) {
    SKIP_IF_NO_CUDA();

    BatchBuilder builder;
    builder.reset();

    int bt0[] = {0, 1};
    int bt1[] = {2, 3, 4};
    builder.add_decode_sequence(100, 10, std::span(bt0, 2), 11);
    builder.add_decode_sequence(200, 20, std::span(bt1, 3), 21);

    Batch batch = builder.build();
    ASSERT_EQ(batch.n_sequences, 2);
    ASSERT_EQ(batch.max_blocks_per_seq, 3);

    GPUBatch gpu_batch;
    gpu_batch.upload(batch);
    cudaStreamSynchronize(nullptr);

    EXPECT_TRUE(gpu_batch.is_valid());
    EXPECT_EQ(gpu_batch.n_sequences, 2);
    EXPECT_EQ(gpu_batch.total_tokens, 2);
    EXPECT_EQ(gpu_batch.max_blocks_per_seq, 3);

    // Verify device pointers are non-null
    EXPECT_NE(gpu_batch.d_token_ids, nullptr);
    EXPECT_NE(gpu_batch.d_positions, nullptr);
    EXPECT_NE(gpu_batch.d_block_tables, nullptr);
    EXPECT_NE(gpu_batch.d_context_lens, nullptr);
    EXPECT_NE(gpu_batch.d_seq_offsets, nullptr);  // n_sequences > 1

    // Read back token_ids from GPU
    std::vector<int32_t> token_ids(2);
    cudaMemcpy(token_ids.data(), gpu_batch.d_token_ids, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(token_ids[0], 100);
    EXPECT_EQ(token_ids[1], 200);

    // Read back positions
    std::vector<int> positions(2);
    cudaMemcpy(positions.data(), gpu_batch.d_positions, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(positions[0], 10);
    EXPECT_EQ(positions[1], 20);

    // Read back context_lens
    std::vector<int> ctx_lens(2);
    cudaMemcpy(ctx_lens.data(), gpu_batch.d_context_lens, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(ctx_lens[0], 11);
    EXPECT_EQ(ctx_lens[1], 21);

    // Read back padded block_tables [2, 3]
    std::vector<int> block_tables(6);
    cudaMemcpy(block_tables.data(), gpu_batch.d_block_tables, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    // Row 0: [0, 1, 0]
    EXPECT_EQ(block_tables[0], 0);
    EXPECT_EQ(block_tables[1], 1);
    EXPECT_EQ(block_tables[2], 0);
    // Row 1: [2, 3, 4]
    EXPECT_EQ(block_tables[3], 2);
    EXPECT_EQ(block_tables[4], 3);
    EXPECT_EQ(block_tables[5], 4);

    gpu_batch.free();
    EXPECT_FALSE(gpu_batch.is_valid());
    EXPECT_EQ(gpu_batch.d_token_ids, nullptr);
}

// 6. GPUBatch upload single sequence (no seq_offsets allocation)
TEST(GPUBatchTest, SingleSequenceNoSeqOffsets) {
    SKIP_IF_NO_CUDA();

    BatchBuilder builder;
    builder.reset();

    int bt[] = {0};
    builder.add_decode_sequence(42, 7, std::span(bt, 1), 8);

    Batch batch = builder.build();
    GPUBatch gpu_batch;
    gpu_batch.upload(batch);
    cudaStreamSynchronize(nullptr);

    EXPECT_TRUE(gpu_batch.is_valid());
    EXPECT_EQ(gpu_batch.n_sequences, 1);
    EXPECT_EQ(gpu_batch.d_seq_offsets, nullptr);  // not allocated for single seq

    gpu_batch.free();
}

// ============================================================================
// Scheduler Tests (memory-aware)
// ============================================================================


// ============================================================================
// BatchBuilder with large batch (16 sequences)
// ============================================================================

// 13. Build a batch of 16 decode sequences
TEST(BatchBuilderTest, SixteenDecodeSequences) {
    BatchBuilder builder;
    builder.reset();

    std::vector<std::vector<int>> block_tables(16);
    for (int i = 0; i < 16; i++) {
        int n_blocks = (i % 4) + 1;  // 1..4 blocks
        block_tables[i].resize(n_blocks);
        for (int b = 0; b < n_blocks; b++) {
            block_tables[i][b] = i * 10 + b;
        }

        int ctx_len = n_blocks * 16;  // kKVBlockSize = 16
        int position = ctx_len - 1;
        int32_t token = 1000 + i;

        builder.add_decode_sequence(token, position, block_tables[i], ctx_len);
    }

    Batch batch = builder.build();

    EXPECT_EQ(batch.n_sequences, 16);
    EXPECT_EQ(batch.total_tokens, 16);
    EXPECT_EQ(batch.max_blocks_per_seq, 4);  // max of {1,2,3,4,...}

    // Verify padded block table dimensions
    EXPECT_EQ(static_cast<int>(batch.block_tables.size()), 16 * 4);

    // Check specific entries
    for (int i = 0; i < 16; i++) {
        int n_blocks = (i % 4) + 1;
        for (int b = 0; b < n_blocks; b++) {
            EXPECT_EQ(batch.block_tables[i * 4 + b], i * 10 + b) << "seq=" << i << " block=" << b;
        }
        // Padding should be 0
        for (int b = n_blocks; b < 4; b++) {
            EXPECT_EQ(batch.block_tables[i * 4 + b], 0) << "seq=" << i << " padding block=" << b;
        }
    }
}

// 14. GPUBatch upload of 16 sequences and readback
TEST(GPUBatchTest, SixteenSequenceUpload) {
    SKIP_IF_NO_CUDA();

    BatchBuilder builder;
    builder.reset();

    for (int i = 0; i < 16; i++) {
        int bt[] = {i, i + 100};
        builder.add_decode_sequence(/*token=*/i + 500, /*position=*/i * 10, std::span(bt, 2), (i + 1) * 10);
    }

    Batch batch = builder.build();
    GPUBatch gpu_batch;
    gpu_batch.upload(batch);
    cudaStreamSynchronize(nullptr);

    EXPECT_EQ(gpu_batch.n_sequences, 16);
    EXPECT_EQ(gpu_batch.total_tokens, 16);

    // Read back all token IDs
    std::vector<int32_t> tokens(16);
    cudaMemcpy(tokens.data(), gpu_batch.d_token_ids, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(tokens[i], i + 500);
    }

    // Read back all positions
    std::vector<int> positions(16);
    cudaMemcpy(positions.data(), gpu_batch.d_positions, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(positions[i], i * 10);
    }

    // Read back context lens
    std::vector<int> ctx(16);
    cudaMemcpy(ctx.data(), gpu_batch.d_context_lens, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(ctx[i], (i + 1) * 10);
    }

    gpu_batch.free();
}

// ============================================================================
// Request Lifecycle Tests
// ============================================================================

// 15. Request context_len calculation
TEST(RequestTest, ContextLen) {
    Request req;
    req.input_tokens = {1, 2, 3, 4, 5};
    EXPECT_EQ(req.context_len(), 5);

    req.output_tokens.push_back(100);
    EXPECT_EQ(req.context_len(), 6);

    req.output_tokens.push_back(200);
    req.output_tokens.push_back(300);
    EXPECT_EQ(req.context_len(), 8);
}

// 16. Request status transitions
TEST(RequestTest, StatusTransitions) {
    Request req;
    EXPECT_EQ(req.status, RequestStatus::PENDING);

    req.status = RequestStatus::PREFILLING;
    EXPECT_EQ(req.status, RequestStatus::PREFILLING);

    req.status = RequestStatus::DECODING;
    EXPECT_EQ(req.status, RequestStatus::DECODING);

    req.status = RequestStatus::FINISHED;
    EXPECT_EQ(req.status, RequestStatus::FINISHED);
}


// 18. Batch builder positions with start_pos offset
TEST(BatchBuilderTest, PrefillWithStartPos) {
    BatchBuilder builder;
    builder.reset();

    int32_t tokens[] = {10, 20, 30};
    int bt[] = {0, 1};
    builder.add_prefill_sequence(std::span(tokens, 3), std::span(bt, 2), /*start_pos=*/5);

    Batch batch = builder.build();

    // Positions should be 5, 6, 7
    EXPECT_EQ(batch.positions[0], 5);
    EXPECT_EQ(batch.positions[1], 6);
    EXPECT_EQ(batch.positions[2], 7);

    // Context len = start_pos + n_tokens = 5 + 3 = 8
    EXPECT_EQ(batch.context_lens[0], 8);
}

// 19. GPUBatch double-free safety
TEST(GPUBatchTest, DoubleFree) {
    SKIP_IF_NO_CUDA();

    BatchBuilder builder;
    builder.reset();

    int bt[] = {0};
    builder.add_decode_sequence(1, 0, std::span(bt, 1), 1);

    Batch batch = builder.build();
    GPUBatch gpu_batch;
    gpu_batch.upload(batch);
    cudaStreamSynchronize(nullptr);

    gpu_batch.free();
    EXPECT_FALSE(gpu_batch.is_valid());

    // Second free should be safe (all ptrs are null)
    gpu_batch.free();
    EXPECT_FALSE(gpu_batch.is_valid());
}

// 20. Empty batch upload
TEST(GPUBatchTest, EmptyBatch) {
    SKIP_IF_NO_CUDA();

    Batch empty_batch;
    GPUBatch gpu_batch;
    gpu_batch.upload(empty_batch);

    EXPECT_FALSE(gpu_batch.is_valid());
    EXPECT_EQ(gpu_batch.d_token_ids, nullptr);

    gpu_batch.free();  // should be safe
}

// ============================================================================
// Batched Decode Tests (Phase 3A: multi-sequence decode)
// ============================================================================


// 22. BatchBuilder: batched decode GPU upload with varying context lengths
TEST(GPUBatchTest, BatchedDecodeVaryingContextLens) {
    SKIP_IF_NO_CUDA();

    BatchBuilder builder;
    builder.reset();

    // Simulate 4 concurrent decode sequences with different progress
    struct SeqInfo {
        int32_t token;
        int position;
        int n_blocks;
        int ctx_len;
    };
    SeqInfo seqs[] = {
        {100, 63, 4, 64},    // long context
        {200, 15, 1, 16},    // short context
        {300, 127, 8, 128},  // very long context
        {400, 31, 2, 32},    // medium context
    };

    std::vector<std::vector<int>> block_tables(4);
    for (int i = 0; i < 4; i++) {
        block_tables[i].resize(seqs[i].n_blocks);
        for (int b = 0; b < seqs[i].n_blocks; b++) {
            block_tables[i][b] = i * 100 + b;
        }
        builder.add_decode_sequence(seqs[i].token, seqs[i].position,
                                    std::span(block_tables[i]).first(seqs[i].n_blocks), seqs[i].ctx_len);
    }

    Batch batch = builder.build();
    EXPECT_EQ(batch.n_sequences, 4);
    EXPECT_EQ(batch.total_tokens, 4);
    EXPECT_EQ(batch.max_blocks_per_seq, 8);  // max of {4, 1, 8, 2}

    GPUBatch gpu_batch;
    gpu_batch.upload(batch);
    cudaStreamSynchronize(nullptr);

    EXPECT_TRUE(gpu_batch.is_valid());
    EXPECT_EQ(gpu_batch.n_sequences, 4);
    EXPECT_EQ(gpu_batch.total_tokens, 4);

    // Verify context lengths readback (crucial for paged attention dispatch)
    std::vector<int> ctx(4);
    cudaMemcpy(ctx.data(), gpu_batch.d_context_lens, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(ctx[0], 64);
    EXPECT_EQ(ctx[1], 16);
    EXPECT_EQ(ctx[2], 128);
    EXPECT_EQ(ctx[3], 32);

    // Verify block tables are correctly padded to max_blocks_per_seq=8
    std::vector<int> bt(4 * 8);
    cudaMemcpy(bt.data(), gpu_batch.d_block_tables, 4 * 8 * sizeof(int), cudaMemcpyDeviceToHost);
    // Row 2 (longest): blocks 200..207
    for (int b = 0; b < 8; b++) {
        EXPECT_EQ(bt[2 * 8 + b], 200 + b);
    }
    // Row 1 (shortest, 1 block): block 100, rest padding
    EXPECT_EQ(bt[1 * 8 + 0], 100);
    for (int b = 1; b < 8; b++) {
        EXPECT_EQ(bt[1 * 8 + b], 0) << "padding at seq=1 block=" << b;
    }

    // Verify seq_offsets for multi-sequence decode
    EXPECT_NE(gpu_batch.d_seq_offsets, nullptr);
    std::vector<int> offsets(5);
    cudaMemcpy(offsets.data(), gpu_batch.d_seq_offsets, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i <= 4; i++) {
        EXPECT_EQ(offsets[i], i);
    }

    gpu_batch.free();
}


// ============================================================================
// Scheduler Edge Case Tests
// ============================================================================


// ============================================================================
// Edge Case Tests
// ============================================================================


// 34. Build a batch with a single 1-token request — valid structure
TEST(BatchBuilderTest, SingleToken) {
    BatchBuilder builder;
    builder.reset();

    int32_t tokens[] = {42};
    int bt[] = {0};
    builder.add_prefill_sequence(std::span(tokens, 1), std::span(bt, 1), /*start_pos=*/0);

    Batch batch = builder.build();

    EXPECT_EQ(batch.n_sequences, 1);
    EXPECT_EQ(batch.total_tokens, 1);
    EXPECT_EQ(batch.max_blocks_per_seq, 1);

    ASSERT_EQ(batch.token_ids.size(), 1u);
    EXPECT_EQ(batch.token_ids[0], 42);

    ASSERT_EQ(batch.positions.size(), 1u);
    EXPECT_EQ(batch.positions[0], 0);

    ASSERT_EQ(batch.context_lens.size(), 1u);
    EXPECT_EQ(batch.context_lens[0], 1);  // start_pos(0) + n_tokens(1)

    ASSERT_EQ(batch.block_tables.size(), 1u);
    EXPECT_EQ(batch.block_tables[0], 0);

    ASSERT_EQ(batch.seq_offsets.size(), 2u);
    EXPECT_EQ(batch.seq_offsets[0], 0);
    EXPECT_EQ(batch.seq_offsets[1], 1);
}

// 35. Newly created request has correct default state
TEST(RequestTest, DefaultState) {
    Request req;

    EXPECT_EQ(req.status, RequestStatus::PENDING);
    EXPECT_TRUE(req.output_tokens.empty());
    EXPECT_EQ(req.prefill_offset, 0);
    EXPECT_TRUE(req.input_tokens.empty());
    EXPECT_EQ(req.id, 0);
    EXPECT_EQ(req.max_tokens, 256);
    EXPECT_EQ(req.context_len(), 0);
}

}  // namespace
}  // namespace imp

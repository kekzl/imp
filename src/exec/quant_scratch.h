#pragma once

#include "memory/vram_allocator.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Quantization scratch buffers: pre-allocated per-GEMM scratch space for
// activation quantization, on-the-fly weight dequantization, and dp4a GEMV.
//
// Allocated once during GraphExecutor::allocate_workspaces(), freed by free().
// All members are public for zero-overhead access in the forward pass.
// ---------------------------------------------------------------------------
struct QuantScratch {
    // --- Generic on-the-fly dequant scratch (Q8_0/Q6_K) ---
    void* dequant = nullptr;
    size_t dequant_size = 0;

    // --- FP8 activation quantization scratch ---
    void* fp8_act = nullptr;  // max_tokens * max_dim bytes
    size_t fp8_act_size = 0;
    float* d_act_scale = nullptr;        // 1 float on device
    float* d_fp8_block_maxes = nullptr;  // pre-allocated reduction buffer
    float* d_fp8_absmax = nullptr;       // pre-allocated absmax scalar
    int fp8_max_grid = 0;                // max grid size for reduction

    // --- CUTLASS NVFP4 prefill activation buffers ---
    void* cutlass_act_data = nullptr;  // [max_tokens, max_K/2] packed FP4
    void* cutlass_act_sf = nullptr;    // SfAtom scale factors (NVFP4: UE4M3)
    size_t cutlass_act_data_size = 0;
    size_t cutlass_act_sf_size = 0;
    void* cutlass_workspace = nullptr;  // CUTLASS GEMM workspace
    size_t cutlass_workspace_size = 0;

    // --- MXFP4 activation buffers (SfAtom UE8M0 scales) ---
    // Packed data shares cutlass_act_data (same FP4 nibble format).
    void* mxfp4_act_sf = nullptr;
    size_t mxfp4_act_sf_size = 0;
    void* mxfp4_workspace = nullptr;
    size_t mxfp4_workspace_size = 0;

    // --- dp4a (MMVQ) scratch for quantized input vector (M=1 decode) ---
    // Sized q8_1_rows x q8_1_max_blocks: production decode uses row 0 only;
    // the spec-verify batched LM head (#847 lever 2) quantizes up to
    // q8_1_rows chunk rows at stride q8_1_max_blocks.
    void* q8_1_buf = nullptr;  // block_q8_1 array
    float* d8_buf = nullptr;   // float scale array
    int q8_1_max_blocks = 0;   // max K/32 (per-row stride)
    int q8_1_rows = 1;

    // --- dp4a prefill scratch for M>1 dense GEMM (Q4_K/Q5_K) ---
    // Sized for max_tokens * max_k/32 blocks. Eliminates the FP16 weight
    // cache intermediate: reads Q4_K directly (0.55 B/elem vs 2.0 B/elem).
    void* q8_1_prefill_buf = nullptr;
    float* d8_prefill_buf = nullptr;
    size_t q8_1_prefill_bytes = 0;
    size_t d8_prefill_bytes = 0;

    // --- FFN sparsity mask (Phase 2): 1 bit per Q8 block, packed uint32. ---
    // Sized (q8_1_max_blocks + 31) / 32 uint32s. Tiny (~tens of bytes).
    uint32_t* ffn_block_mask = nullptr;
    int ffn_block_mask_words = 0;

    // --- Split-K paged attention scratch ---
    void* splitk = nullptr;
    size_t splitk_size = 0;

    // --- Sparse decode attention scratch (attention.sparse_topk_tokens) ---
    // budget_blocks == 0 means the feature is inactive for this engine.
    float* sparse_scores = nullptr;      // [max_batch, max_ctx_blocks] block scores
    int* sparse_block_tables = nullptr;  // [max_batch, sparse_budget_blocks]
    int* sparse_context_lens = nullptr;  // [max_batch]
    int sparse_budget_blocks = 0;
    int sparse_sink_blocks = 0;
    int sparse_recent_blocks = 0;
    int sparse_max_ctx_blocks = 0;  // capacity of a sparse_scores row

    // Free all buffers.
    void free(VRAMAllocator* alloc);
};

}  // namespace imp

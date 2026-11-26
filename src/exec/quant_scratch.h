#pragma once

#include "memory/device_owned.h"
#include "memory/span.h"
#include "memory/vram_allocator.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

struct block_q8_1;             // compute/gemm.h
struct Nvfp4DequantWorkspace;  // quant/nvfp4_dequant_ws.h

// Activation scale of an FP8 x FP8 GEMM: one device scalar (checkpoint input_scale), or per_row: one
// scale per activation row (per-token dynamic, quantize_fp8_act_rows). Null d = not quantized.
struct Fp8ActScale {
    const float* d = nullptr;
    bool per_row = false;
    explicit operator bool() const { return d != nullptr; }
};

// Quantization scratch buffers: pre-allocated per-GEMM scratch for activation
// quantization, on-the-fly weight dequant, and dp4a GEMV. Allocated once during
// GraphExecutor::allocate_workspaces(), freed by free(). Move-only: the separately allocated
// buffers are DeviceOwned, the rest are T2 arena views and plain values. All members public for
// zero-overhead forward-pass access.
struct QuantScratch {
    // --- Generic on-the-fly dequant scratch (Q8_0/Q6_K) ---
    DeviceOwned<void> dequant;
    size_t dequant_size = 0;

    // --- FP8 activation quantization scratch ---
    DeviceOwned<void> fp8_act;  // max_tokens * max_dim bytes
    size_t fp8_act_size = 0;
    StableSpan<float> d_act_scale;        // 1 float on device (T2 arena)
    StableSpan<float> d_fp8_block_maxes;  // pre-allocated reduction buffer
    StableSpan<float> d_fp8_absmax;       // pre-allocated absmax scalar
    int fp8_max_grid = 0;                // max grid size for reduction
    StableSpan<float> d_act_row_scales;  // per-token dynamic act scales, act_row_scales_n floats (T2 arena)
    int act_row_scales_n = 0;
    DeviceOwned<void> fp8_tile_ws;  // stream-K partials of the own FP8 tile GEMM (SMs x 64 KiB)
    size_t fp8_tile_ws_size = 0;

    // --- CUTLASS NVFP4 prefill activation buffers ---
    DeviceOwned<void> cutlass_act_data;  // [max_tokens, max_K/2] packed FP4
    DeviceOwned<void> cutlass_act_sf;    // SfAtom scale factors (NVFP4: UE4M3)
    size_t cutlass_act_data_size = 0;
    size_t cutlass_act_sf_size = 0;
    // Act tensor scale the bytes in cutlass_act_data were quantized with: a hinted dedupe reuses
    // them only at the same scale (mutable: GemmContext holds the scratch const).
    mutable float cutlass_act_scale = 1.0f;
    DeviceOwned<void> cutlass_workspace;  // CUTLASS GEMM workspace
    size_t cutlass_workspace_size = 0;

    // --- Native MXFP4 W4A16 GEMM (M > 1): one dense weight dequantized to FP16 (largest non-LM-head N x K)
    // and the LM-head slice output, same size. Own buffers: the shared workspace holds the GDN state. ---
    DeviceOwned<void> mxfp4_wdeq;
    DeviceOwned<void> mxfp4_slice_out;
    size_t mxfp4_wdeq_size = 0;

    // --- MXFP8 GDN prefill activation buffers (gemm.mxfp8_gdn_proj_prefill) ---
    DeviceOwned<void> mxfp8_act_data;  // [max_tokens, max_K] E4M3
    DeviceOwned<void> mxfp8_act_sf;    // SfAtom UE8M0
    size_t mxfp8_act_data_size = 0;
    size_t mxfp8_act_sf_size = 0;

    // dp4a (MMVQ) scratch for quantized input vector (M=1 decode): sized q8_1_rows x
    // q8_1_max_blocks. Production decode uses row 0 only; the spec-verify batched LM head
    // (#847 lever 2) quantizes up to q8_1_rows chunk rows at stride q8_1_max_blocks.
    StableSpan<block_q8_1> q8_1_buf;  // block_q8_1 array
    StableSpan<float> d8_buf;         // float scale array
    int q8_1_max_blocks = 0;   // max K/32 (per-row stride)
    int q8_1_rows = 1;

    // --- dp4a prefill scratch for M>1 dense GEMM (Q4_K/Q5_K) ---
    // Sized for max_tokens * max_k/32 blocks. Eliminates the FP16 weight
    // cache intermediate: reads Q4_K directly (0.55 B/elem vs 2.0 B/elem).
    StableSpan<block_q8_1> q8_1_prefill_buf;
    StableSpan<float> d8_prefill_buf;
    size_t q8_1_prefill_bytes = 0;
    size_t d8_prefill_bytes = 0;

    // gemm_nvfp4's M>1 fallback workspace, the executor's nvfp4_dequant_ws_ (null: none). Reaches
    // gemm_dispatch through GemmContext::qscratch, per call (audit D-9).
    Nvfp4DequantWorkspace* nvfp4_dequant = nullptr;

    // --- Split-K paged attention scratch ---
    StableSpan<std::byte> splitk;
    size_t splitk_size = 0;

    // --- Sparse decode attention scratch (attention.sparse_topk_tokens) ---
    // budget_blocks == 0 means the feature is inactive for this engine.
    StableSpan<float> sparse_scores;      // [max_batch, max_ctx_blocks] block scores
    StableSpan<int> sparse_block_tables;  // [max_batch, sparse_table_blocks]
    StableSpan<int> sparse_context_lens;  // [max_batch]
    int sparse_budget_blocks = 0;
    int sparse_sink_blocks = 0;
    int sparse_recent_blocks = 0;
    int sparse_engage_blocks = 0;   // identity below this many ctx blocks (kSparseMinCtx)
    int sparse_table_blocks = 0;    // row capacity of sparse_block_tables (= engage blocks)
    int sparse_max_ctx_blocks = 0;  // capacity of a sparse_scores row
    int sparse_max_rows = 0;        // row count of the scratch buffers (batch / chunk rows)

    // --- Sparse prefill scratch (attention.sparse_prefill_topk_tokens); budget 0 = off ---
    StableSpan<float> sp_prefill_scores;  // [64, sp_prefill_cap_blocks]
    StableSpan<float> sp_prefill_agg;     // [sp_prefill_cap_blocks]
    StableSpan<int> sp_prefill_table;     // [sp_prefill_cap_blocks]
    StableSpan<int> sp_prefill_ctx;       // [2]
    int sp_prefill_budget_blocks = 0;
    int sp_prefill_sink_blocks = 0;
    int sp_prefill_recent_blocks = 0;
    int sp_prefill_cap_blocks = 0;
    int sp_prefill_rows = 0;

    // Releases every owned buffer and returns the struct to its initial state (arena views nulled).
    void free();
};

}  // namespace imp

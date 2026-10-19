#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

struct DSparkOptions {
    bool fp8 = true;    // q / o / gate / up / down / lm_head as per-row E4M3 (half the propose bytes)
    bool graph = true;  // propose replays one CUDA graph, captured on the second call
};

// DSpark / DFlash block drafter (deepseek-ai/dflash_qwen3_8b_block7, z-lab DFlashDraftModel with
// markov_rank 0): 5 non-causal Qwen3 layers draft block_size tokens in one pass from the target's
// hidden states at target_layer_ids (concatenated, fc + hidden_norm). Reference forward:
// tools/analysis/dspark_reference.py. Batch 1, one sequence; FP16 weights converted from the BF16 checkpoint.
class DSparkDrafter {
public:
    DSparkDrafter();
    ~DSparkDrafter();
    DSparkDrafter(const DSparkDrafter&) = delete;
    DSparkDrafter& operator=(const DSparkDrafter&) = delete;

    // Loads config.json + model.safetensors from `dir`. The checkpoint's embed_tokens / lm_head are kept
    // (identical to Qwen3-8B's). `max_ctx` sizes the context K/V; device memory through `alloc`. "" or a
    // reason.
    std::string load(VRAMAllocator& alloc, const std::string& dir, int target_d_model, int target_layers,
                     int max_ctx, cudaStream_t stream, const DSparkOptions& opt);
    // Proposes served by graph replay so far.
    int graph_replays() const;

    int block_size() const;
    const std::vector<int>& target_layer_ids() const;
    // Columns of one context feature row: target_layer_ids().size() * d_model.
    int feature_dim() const;
    // Bytes the drafter holds on the device after load (weights + K/V + scratch).
    size_t device_bytes() const;

    // Forget the context (new request).
    void reset();
    // Rows of context held, and the position after the last one.
    int ctx_rows() const;
    int next_pos() const;

    // Appends `n` context rows (feature rows with leading dimension `ld` halves, device) at positions
    // pos0..pos0+n-1. Rows at positions >= pos0 already held are dropped first (rollback). n <= max rows.
    void append_ctx(const half* feat, int ld, int n, int pos0, cudaStream_t stream);

    // Drafts block_size tokens after `anchor` (the next token to feed, at position `pos`): block row r
    // predicts position pos + r + 1, the anchor row included (unlike z-lab DFlash, rows 1..). Syncs `stream`.
    std::vector<int32_t> propose(int32_t anchor, int pos, cudaStream_t stream);

    // Rows one append_ctx call takes.
    static constexpr int kMaxAppendRows = 1024;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace imp

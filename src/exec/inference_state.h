#pragma once

#include "compute/rope.h"  // MRopeParams
#include "core/tensor.h"
#include "memory/kv_cache.h"     // KVCache
#include "memory/ssm_state.h"    // SSMState
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <utility>
#include <vector>

namespace imp {

// Constrained-decoding hooks referenced only by pointer here; full definitions live in
// compute/json_constrain.h / schema_constrain.h / regex_constrain.h, included by the TUs
// that dereference them (executor.cu).
class JsonConstrainer;
class SchemaConstrainer;
class RegexConstrainer;
class GrammarConstrainer;

// All the state needed for a single forward pass invocation.
struct InferenceState {
    // Input tokens
    const int32_t* token_ids = nullptr;  // [n_tokens] on device
    const int* positions = nullptr;      // [n_tokens] on device
    int n_tokens = 0;

    // KV cache for paged attention (decode)
    KVCache* kv_cache = nullptr;
    const int* block_tables = nullptr;  // [n_sequences, max_blocks_per_seq] on device (2D padded)
    // SWA-group block tables (kv_cache.swa_sizing): same shape/stride as block_tables, -1
    // holes outside the trailing window. Sliding-window layers read/write through this table;
    // nullptr when off (all layers use block_tables).
    const int* block_tables_swa = nullptr;
    const int* context_lens = nullptr;  // [n_sequences] on device
    int max_context_len = 0;

    // SSM state for Mamba2 layers (nullptr for non-hybrid models)
    SSMState* ssm_state = nullptr;
    int ssm_seq_id = 0;  // sequence ID for SSM state access (single-sequence path)

    // Batched GDN decode: when ssm_n_seq > 1, rows belong to ssm_n_seq DIFFERENT sequences
    // (one token each); ssm_seq_slots is a DEVICE array of their recurrent-state slot ids.
    // The scan is sequential in tokens (cannot parallelize one sequence over its timeline),
    // so separate sequences batch instead; this is what keeps FFN/attention GEMMs off M=1.
    // Stable device buffer, not the values: a CUDA graph captures the pointer; the host
    // refills it per step, so a changed sequence set does not force a re-capture.
    const int* ssm_seq_slots = nullptr;
    int ssm_n_seq = 1;
    // Multi-candidate verify chunk on a hybrid (roadmap gap 5, Stage 3): rows are ssm_n_seq
    // candidate groups of ssm_seq_tokens rows; group c owns rows
    // [c*ssm_seq_tokens, (c+1)*ssm_seq_tokens) and recurrent slot ssm_seq_slots[c].
    // Rows past ssm_n_seq*ssm_seq_tokens are capture-bucket pads. d_chunk_len carries the
    // per-GROUP row count (uniform), not the whole chunk's. 0 = not a grouped chunk.
    int ssm_seq_tokens = 0;
    bool ssm_grouped_chunk() const {
        return is_prefill && ssm_seq_slots != nullptr && ssm_n_seq > 1 && ssm_seq_tokens > 0;
    }
    // Batched speculative verify (docs/plans/2026-09-11-batched-mtp-verify.md): a grouped
    // chunk whose groups are DIFFERENT requests. Group g reads slot ssm_seq_slots[g], commits
    // state at d_chunk_len rows into ssm_out_slots[g] and at d_snap_n rows into
    // ssm_snap_slots[g]. nullptr = in-place commit, group-0 snapshot into spec_snap_slab.
    const int* ssm_out_slots = nullptr;
    const int* ssm_snap_slots = nullptr;
    // Factored spare (compute/gdn_factor.cuh, docs/plans/2026-09-12-factored-verify-spare.md):
    // ssm_fac_out replaces ssm_out_slots, carrying the drafted row as (g,k,delta) per head
    // instead of a second full state slot. ssm_fac_in carries the row a previous verify left
    // for an accepted request. Both are [n_ssm_layers][slot][head][ssm_fac_stride] float,
    // per-layer stride ssm_fac_layer_stride.
    float* ssm_fac_out = nullptr;
    const float* ssm_fac_in = nullptr;
    int ssm_fac_stride = 0;
    int64_t ssm_fac_layer_stride = 0;
    // Conv half of the same spare (compute/ssm_conv_tap.cu): ssm_tap_out stashes the drafted
    // row's conv input; ssm_tap_in + ssm_tap_slots advance accepted slots' windows before
    // this layer's conv reads them.
    void* ssm_tap_out = nullptr;
    const void* ssm_tap_in = nullptr;
    const int* ssm_tap_slots = nullptr;
    int ssm_tap_n = 0;
    int64_t ssm_tap_layer_stride = 0;

    // BitDecoding Phase 3 residual KV cache. Two modes: single-seq scalar or multi-seq array
    // (dispatcher prefers multi-seq whenever the engine sets up the device arrays).
    // Single-seq: kv_seq_id is the request id used to look up ring state via KVCacheManager;
    // -1 disables.
    int kv_seq_id = -1;
    // KVCacheManager owning the residual buffer + ring-state map.
    class KVCacheManager* kv_manager = nullptr;
    // Multi-seq array form: device pointers to per-batch metadata, length n_sequences, built
    // by the engine each forward step before the attention call. Each element matches the
    // corresponding row of block_tables/context_lens. nullptr = multi-seq form inactive.
    const int* d_residual_seq_slots = nullptr;     // [n_sequences] slot in [0, residual_max_seqs)
    const int* d_residual_counts = nullptr;         // [n_sequences] fill_count
    const int* d_residual_write_idxes = nullptr;    // [n_sequences] write_idx
    // Host array of per-batch seq_ids (request ids), used by the KV write path
    // to call KVCacheManager::advance_residual per seq. Length = n_sequences.
    const int* h_residual_seq_ids = nullptr;

    // Batching
    int n_sequences = 1;         // number of sequences in the batch
    int max_blocks_per_seq = 0;  // max blocks per sequence (for 2D block_table indexing)
    // Spec-verify chunk whose attention runs on the batched-decode split-K path (#964):
    // chunk rows are presented as n_sequences same-KV "sequences" with per-row context_lens
    // and row-replicated block_tables. is_prefill stays true so non-attention code keeps
    // chunk-forward semantics.
    bool chunk_decode_attn = false;
    // Spec-verify chunk forward (#998): small-M GEMMs may read the NVFP4 decode overlay
    // (one weight pass per MR tile) instead of the M>1 prefill dequant path, since on GGUF
    // K-quants the per-chunk dequant cost made speculation net-negative. Set for every verify
    // chunk regardless of attention route.
    bool spec_verify_chunk = false;
    const int* seq_offsets =
        nullptr;  // [n_sequences+1] for ragged prefill token offsets (optional, nullptr for decode)

    // Cross-sequence ragged prefill (roadmap 0(d)): rows are the CONCATENATED prefill
    // chunks of n_sequences requests. Row-wise work (GEMMs, norms, RoPE) runs over all rows
    // in one launch; attention and GDN conv drop to a per-seq loop over
    // [h_seq_offsets[i], h_seq_offsets[i+1]), falling back to the fused batched GDN kernel
    // with seq_offsets + ssm_seq_slots. h_seq_offsets != nullptr is the activation condition:
    // engine sets it only for n_sequences>1, is_prefill, none of vision/MTP/spec-verify/
    // logprobs/constraints/ppl capture/SWA sizing/residual KV.
    const int* h_seq_offsets = nullptr;    // HOST [n_sequences+1] row prefix sums
    const int* h_seq_q_offsets = nullptr;  // HOST [n_sequences] per-seq prefill_offset
    const int* h_ssm_slots = nullptr;      // HOST [n_sequences] recurrent-state slots
    bool ragged_prefill() const { return is_prefill && n_sequences > 1 && h_seq_offsets != nullptr; }
    // Mixed prefill+decode step (runtime.prefill_mixed_decode): the LAST n_riders sequences
    // of a ragged prefill state are decoding requests with one row each; run_attention gives
    // them one batched paged-decode launch per layer instead of the per-member prefill
    // dispatch.
    int n_riders = 0;
    int rider_max_context_len = 0;

    // Mode
    bool is_prefill = true;
    // Absolute position of state.positions[0] within the full sequence.
    // 0 means single-chunk prefill or first chunk of a chunked prefill.
    // > 0 means a follow-up chunk: tokens [0, prefill_offset) are already in the KV cache.
    int prefill_offset = 0;

    // Sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;
    float min_p = 0.0f;
    float typical_p = 1.0f;  // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int repeat_last_n = 0;  // How many recent tokens to scan (0 = all)

    // DRY (Don't Repeat Yourself) penalty
    float dry_multiplier = 0.0f;  // 0 = disabled
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 0;                    // 0 = full history
    const int32_t* host_penalty_tokens = nullptr;  // HOST pointer for DRY scanning

    // Mirostat v2 adaptive entropy sampling
    int mirostat = 0;                  // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;         // Target entropy
    float mirostat_eta = 0.1f;         // Learning rate
    mutable float mirostat_mu = 0.0f;  // Running variable (updated by sampling)

    // Token history for penalty computation (device pointer, owned by engine)
    const int32_t* penalty_tokens = nullptr;
    int n_penalty_tokens = 0;
    // Device-side penalty token count (for CUDA graph loop where count grows
    // each iteration). When non-null, forward_decode_async reads the count
    // from *d_n_penalty_tokens instead of n_penalty_tokens.
    const int* d_n_penalty_tokens = nullptr;

    // Logprobs: when true, forward() copies logits to h_logits_pinned_ for CPU extraction
    bool logprobs = false;
    int top_logprobs = 0;

    // JSON mode: when non-null, apply logit mask before sampling
    JsonConstrainer* json_constrainer = nullptr;
    SchemaConstrainer* schema_constrainer = nullptr;
    RegexConstrainer* regex_constrainer = nullptr;
    GrammarConstrainer* grammar_constrainer = nullptr;
    // Output tokens still available to this request (max_tokens - produced). A constrainer
    // can forbid illegal tokens but not force termination, so once only enough tokens remain
    // to close the document, the mask narrows to closers (#1104). -1 = unknown, no narrowing.
    int constrain_remaining_tokens = -1;

    // Logit bias (host-side, applied via cudaMemcpy before sampling)
    const std::pair<int32_t, float>* logit_bias = nullptr;
    int n_logit_bias = 0;

    // Banned tokens: set logits to -inf before sampling (e.g. chat template special tokens)
    const int32_t* banned_tokens = nullptr;  // HOST pointer, small list
    int n_banned_tokens = 0;
    const int32_t* d_banned_tokens = nullptr;  // DEVICE pointer (for CUDA graph path)
    int n_d_banned_tokens = 0;
    // Graph path twin of the host stop mask: these ids are banned only while
    // *d_stop_mask_active != 0, a device flag post_decode_step_kernel writes
    // from think_logic::stop_mask_active. All three set, or none.
    const int32_t* d_stop_mask_tokens = nullptr;
    int n_d_stop_mask_tokens = 0;
    const int* d_stop_mask_active = nullptr;

    // Force token: when >= 0, set ALL logits except this token to -inf.
    // Used by think-budget to force </think> generation via logit manipulation
    // so the token lands correctly in the KV cache (NVIDIA NIM approach).
    int32_t force_token = -1;

    // Vision: when non-null, replace vision_token_id positions with vision embeddings
    const half* vision_embeddings = nullptr;  // [n_vision_tokens, d_model] FP16 on device
    int vision_token_id = -1;                 // <image_soft_token> ID
    int n_vision_tokens = 0;                  // total in the buffer, across all chunks

    // How many image tokens EARLIER chunks already placed. token_ids is one chunk; without
    // this the k-th placeholder of a later chunk would take the k-th embedding of the image,
    // the wrong region, silently. Zero when the prompt is prefilled in one go.
    int vision_emb_offset = 0;

    // DeepStack (Qwen3-VL): extra visual features ADDED at image-token positions after each
    // of the LM's first n_deepstack layers. Indexed by LM layer (0/1/2), not the vision block
    // tapped (blocks 5/11/17). Each entry has the same shape as vision_embeddings.
    static constexpr int kMaxDeepStack = 4;
    const half* deepstack_embeddings[kMaxDeepStack] = {};
    int n_deepstack = 0;

    // M-RoPE: per-token (t,h,w) positions, [3, n_tokens] on device. Present on EVERY step of
    // an M-RoPE model, even text-only (all three rows equal, bit-identical to single-axis).
    // Always-on because rope dispatch branches on this pointer; a flipping branch would bake
    // the wrong rotation into a CUDA-graph replay. Null for non-M-RoPE models.
    MRopeParams mrope;

    // Early exit: run only the first exit_layer layers (-1 = all layers).
    // Used by self-speculative decoding to generate cheap draft tokens.
    int exit_layer = -1;

    // Layer skip: skip layers in [skip_layer_start, skip_layer_end) during forward.
    // Used by self-speculative decoding for better acceptance than pure early exit.
    // Runs layers {0..skip_start-1, skip_end..n_layers-1}. -1 = disabled.
    int skip_layer_start = -1;
    int skip_layer_end = -1;

    // When true, project ALL tokens through the LM head during prefill
    // (normally only the last token is projected). Used by speculative verify.
    bool all_logits = false;

    // When true, bypass FP8 GEMM paths and use dequant→FP16 GEMM instead.
    // Avoids compound FP8 quantization error over many layers (self-spec verify).
    bool force_fp16_gemm = false;

    // When true, use per-row Q8_1 GEMV for LM head instead of batched FP8 GEMM.
    // Avoids FP8 per-tensor quantization artifacts in batched verification.
    bool per_row_lm_head = false;

    // Graph-captured verify chunk (#847). ctx_capacity > 0 makes chunked continuation
    // attention replayable across context growth: gather grids and KV scratch are sized for
    // ctx_capacity, kernels read real lengths from device (context_lens[0], d_past_len)
    // instead of baking max_context_len/prefill_offset. Requires n_sequences==1 and an
    // FA2-served config (GraphExecutor::chunk_capture_supported).
    int ctx_capacity = 0;
    const int* d_past_len = nullptr;  // device int == prefill_offset
    // Device int == real (unpadded) chunk length. Hybrid recurrent-state
    // updates (conv tail, scan final state) read it so the padding rows of a
    // captured verify chunk don't advance the committed state.
    const int* d_chunk_len = nullptr;
    // Speculative verify: second per-sequence recurrent slab, written with the state as of
    // row d_snap_n, alongside the committed one at the real last row. A partial acceptance
    // landing on that row adopts it instead of restoring pre-chunk state and re-forwarding.
    // nullptr disables it; nothing else changes.
    void* spec_snap_slab = nullptr;
    const int* d_snap_n = nullptr;
    // The state as it was BEFORE this chunk. The conv snapshot's leading values
    // come from there; reading them from the live buffer races the commit that
    // another block writes into it.
    void* spec_prev_slab = nullptr;
};

}  // namespace imp

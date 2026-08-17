#pragma once

// Which attention kernel families can serve a given head_dim.
//
// These two questions are asked in two places that must agree: the prefill
// dispatch (executor_attention_prefill.cu), which picks the kernel, and
// max_safe_prefill_chunk (executor_workspace_buffers.cu), which decides how
// large a chunk the engine may hand it. When the second believes a family will
// take the chunk and the first cannot, nothing clamps the chunk and the cuBLAS
// fallback runs off the end of its S-matrix.
//
// That is not hypothetical. The clamp used to return `desired` unclamped
// whenever the context crossed `attention.fmha_prefill_threshold`, without
// asking whether the tiled FMHA serves this head_dim at all. On DeepSeek-V2-Lite
// (MLA, head_dim 192, served by neither FA2 nor FMHA) a perplexity run over a
// 45k-token corpus reached chunk 2048 at ctx 6144, needing 12 582 912 S-matrix
// elements against the 3536x3536 = 12 503 296 allocated, and the defense-in-depth
// check in the dispatch aborted the process with "engine should have prevented
// this". It was right: the engine should have, and could not, because the two
// sides disagreed about FMHA.
//
// Header is CUDA-free on purpose so the CPU lane can test the rules directly.

namespace imp {

// The tiled WMMA FMHA dispatch. Covers the fused head dims, including 512 for
// Gemma-4's global layers.
inline bool fmha_serves_head_dim(int head_dim) {
    return head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 256 || head_dim == 512;
}

// FP16-QK FlashAttention-2. hd=256 rides the stage-1 port and is gated by
// `attention.fa2_hd256`; `attention.fa2_fp16qk == "never"` is the caller's
// opt-out and is checked there, not here.
inline bool fa2_serves_head_dim(int head_dim, bool fa2_hd256_enabled) {
    return head_dim == 128 || (head_dim == 256 && fa2_hd256_enabled);
}

// True when some O(n) family takes this head_dim, i.e. when no S-matrix is
// needed and the chunk size is unconstrained by its capacity.
inline bool o_n_attention_serves_head_dim(int head_dim, bool fa2_hd256_enabled) {
    return fa2_serves_head_dim(head_dim, fa2_hd256_enabled) || fmha_serves_head_dim(head_dim);
}

}  // namespace imp

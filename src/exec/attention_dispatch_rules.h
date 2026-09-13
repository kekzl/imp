#pragma once

// Which attention kernel families can serve a given head_dim: asked in two
// places that must agree, prefill dispatch (executor_attention_prefill.cu,
// picks the kernel) and max_safe_prefill_chunk (executor_workspace_buffers.cu,
// bounds the chunk). If they disagree, nothing clamps the chunk and cuBLAS
// runs off the end of its S-matrix (happened on DeepSeek-V2-Lite MLA,
// hd=192, served by neither FA2 nor FMHA). Header is CUDA-free so the CPU lane tests the rules directly.

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

#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// M-RoPE (Qwen-VL): rotary pairs driven by 3 position axes (text/time, image H, image W)
// instead of one. positions is [3,stride], axis-major.
// Null pointer = opt-out: every pair reads the single `positions` arg (every text-only
// model today). A non-null pointer with equal rows must produce bit-identical angles to
// that path - the invariant a text prompt on a VL model relies on.
struct MRopeParams {
    const int* positions = nullptr;
    int stride = 0;                       // tokens per axis row
    int sec_t = 0, sec_h = 0, sec_w = 0;  // half-counts, summing to rope_dim/2
    // Qwen3-VL interleaves the axes across the spectrum (T,H,W,T,H,W,... then a
    // T tail); Qwen2-VL takes three contiguous blocks. Different dimensions get
    // different angles, so the layouts are not interchangeable.
    bool interleaved = false;
    // Decode's per-token position: generated text advances all 3 M-RoPE axes together, so
    // position = single position + a per-request constant (negative after an image, which
    // costs more tokens than positions).
    // DEVICE array of `stride` entries (not scalar/host-computed): (1) decode position
    // advances device-side inside a captured graph, a baked scalar would freeze the request
    // live at capture time; (2) a decode batch mixes requests, one's offset must not leak
    // into another's step.
    const int* pos_delta = nullptr;
};

// Fused RoPE on Q/K in-place. rope_dim>0: rotate only the first rope_dim dims (else all
// head_dim). neox: true=NeoX/split pairs (i,i+d/2), false=interleaved (2i,2i+1).
// YaRN (ext_factor>0): attn_factor=mscale; corr_dims=[2] host floats, dimension
// boundaries for the ramp (rope_yarn_corr_dims()) - passed by value, a device pointer hangs.
void rope_forward(Tensor& Q, Tensor& K, const int* positions, int head_dim, float theta = 10000.0f,
                  float scaling = 1.0f, int rope_dim = 0, bool neox = false, float ext_factor = 0.0f,
                  float attn_factor = 1.0f, const float* corr_dims = nullptr, cudaStream_t stream = nullptr,
                  const float* longrope_inv_freqs = nullptr, MRopeParams mrope = {});

// Fused QK-norm + RoPE for decode rows (FP16, n_tokens<=64 typical): per-head RMSNorm on
// Q/K then RoPE, one CTA per (head,token).
// Q:[n_tokens,n_heads*head_dim], K:[n_tokens,n_kv_heads*head_dim]; norm weights:[head_dim].
void qknorm_rope_fused(half* Q, half* K, const half* q_norm_weight, const half* k_norm_weight, int n_heads,
                       int n_kv_heads, int head_dim, float eps, const int* positions, float theta = 10000.0f,
                       float scaling = 1.0f, int rope_dim = 0, bool neox = false,
                       cudaStream_t stream = nullptr, float weight_offset = 0.0f, float ext_factor = 0.0f,
                       float attn_factor = 1.0f, const float* corr_dims = nullptr,
                       const float* longrope_inv_freqs = nullptr, MRopeParams mrope = {}, int n_tokens = 1);

// Precompute YaRN correction dimension boundaries: dims[0]=start (below: full NTK
// interpolation), dims[1]=end (above: full extrapolation); linear ramp between.
void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
                         float dims[2]);

// Register RoPE kernels for PDL tail/head overlap.
void rope_pdl_register();

}  // namespace imp

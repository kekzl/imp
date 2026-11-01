#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// Kimi Delta Attention (KDA, fla-org fla/ops/kda): the gated delta rule with a per-key-channel
// decay. Per head, state S[K][V] (same [n_heads, SS, HD] layout as the GDN pool), per token:
//   q, k = l2norm(q), l2norm(k)  (x * rsqrt(sum(x^2) + 1e-6), q also * K^-0.5)
//   g    = lower_bound * sigmoid(exp(A_log[h]) * (f + dt_bias[h, :]))     (log decay, [K])
//   S    = S * exp(g)[:, None];  S += sigmoid(b) * k (v - k^T S)^T;  o = q^T S
// Head dims are fixed at K = V = 128 (Ling-3.0, Kimi-Linear); launchers throw otherwise.
inline constexpr int kKdaHeadDim = 128;

struct KdaScanArgs {
    const float* conv_f32 = nullptr;  // [rows, conv_channels] post conv+SiLU: Q(H*K) | K(H*K) | V(H*V)
    int conv_channels = 0;
    const half* f = nullptr;               // [rows, H*K] raw f_proj output
    const half* beta = nullptr;            // [rows, H] raw b_proj output, sigmoid applied in the kernel
    const float* neg_a = nullptr;          // [H] -exp(A_log): the GDN ssm_a upload transform
    const float* dt_bias = nullptr;        // [H*K]
    float lower_bound = -5.0f;             // kda_lower_bound (safe gate)
    void* h_state = nullptr;               // slot pool base (seq_slots) or one sequence's [H, K, V]
    bool state_bf16 = false;               // state storage BF16, arithmetic FP32
    const int* seq_slots = nullptr;        // device [n_seq] slot ids; nullptr = h_state is one sequence
    int64_t seq_stride = 0;                // state elements between slots
    const int* seq_row_offsets = nullptr;  // device [n_seq+1] ragged row prefix; nullptr = uniform
    const int* d_real_n = nullptr;         // device commit row count (padded chunk), uniform batches only
    half* y = nullptr;                     // [rows, H*V]
    int n_seq = 1;
    int n_tokens = 0;  // rows per sequence (uniform); ignored when ragged
    int n_heads = 0;
};

// Token-serial scan, state in registers: decode, batched decode, short prefill, padded chunks.
void kda_scan_recurrent(const KdaScanArgs& a, cudaStream_t stream);

// Chunked prefill scan (chunk 16, FP32 WY form of fla's naive_chunk_kda): same results as the
// recurrent scan up to FP32 reassociation. No d_real_n (padded chunks take the recurrent scan).
void kda_scan_chunk(const KdaScanArgs& a, cudaStream_t stream);

// Minimum rows per sequence for which callers pick kda_scan_chunk (fla: q_len > 64).
inline constexpr int kKdaChunkMinRows = 64;

}  // namespace imp

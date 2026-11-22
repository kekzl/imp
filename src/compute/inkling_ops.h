#pragma once
// Inkling2-Loop building blocks: fp32 residual stream, fp16 linear inputs and outputs.
// Reference: transformers models/inkling2_loop/modeling_inkling2_loop.py.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp::inkling {

void bf16_to_f16(const uint16_t* in, half* out, size_t n, cudaStream_t s);
void bf16_to_f32(const uint16_t* in, float* out, size_t n, cudaStream_t s);
void f32_to_f16(const float* in, half* out, size_t n, cudaStream_t s);
void f16_to_f32(const half* in, float* out, size_t n, cudaStream_t s);
// out = a + b (b may be null: copy).
void add_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t s);

void embed_rows(const half* table, const int32_t* ids, float* out, int T, int H, cudaStream_t s);
// x * rsqrt(mean(x^2) + eps) [* w]; either output may be null.
void rms_norm(const float* x, const float* w, float* out_f, half* out_h, int T, int H, float eps,
              cudaStream_t s);

// Block attention residual: softmax over the unit-RMS states with nq learned queries (scale H^-0.5),
// out[q] = sum_j p[q][j] * normed_state_j. The last state is the partial block state.
struct MixStates {
    const float* s[5] = {};
    int n = 0;
};
struct MixOut {
    float* o[4] = {};
};
void block_mix(const MixStates& st, const float* query, int nq, const MixOut& out, int T, int H,
               cudaStream_t s);

// Message gate, part 1: cat = [unit_rms(h), unit_rms(msg)] as fp16 [T, 2H].
void gate_cat(const float* h, const float* msg, half* cat, int T, int H, float eps, cudaStream_t s);
// Part 2: out = (1 - tanh(forget)) * h + tanh(gain) * msg.
void gate_apply(const float* h, const float* msg, const half* forget, const half* gain, float* out,
                size_t n, cudaStream_t s);

// Per head: v = x * scale; v *= min(thr / (|v| + 1e-6), 1); log_alpha > 0: v *= 1 + alpha * ln(max((pos + 1) / floor, 1)).
void scaled_l2_clip(const half* x, const float* scale, float* out, int T, int n_heads, int head_dim,
                    float thr, int pos0, float log_alpha, float log_floor, cudaStream_t s);

// Depthwise causal conv, kernel 4, out = (conv + x) / sqrt(5). w [C, 4], tap 3 = current token.
// state [3, C] = the three inputs before this call's first token.
void conv_static(const float* x, const float* w, const float* state, float* out, int T, int C,
                 cudaStream_t s);
// Taps = static_w [4, C] + dyn [T, n_heads * 4] per head, tap 0 = current token; out = (sum + x) / 3.
void conv_dynamic(const float* x, const float* static_w, const half* dyn, const float* state, float* out,
                  int T, int C, int n_heads, cudaStream_t s);
// state <- the last three inputs after this call.
void conv_state_update(const float* x, float* state, int T, int C, cudaStream_t s);

// Key (L2 clip at thr) or value (unit RMS) per head, written as fp16 at slot (pos0 + t) % ring.
void kv_store(const float* x, half* cache, int T, int n_heads, int head_dim, bool is_key, float thr,
              float eps, int pos0, int ring, cudaStream_t s);

struct AttnArgs {
    const float* q = nullptr;     // [T, n_heads, hd]
    const half* k = nullptr;      // [ring or ctx, n_kv, hd]
    const half* v = nullptr;
    const float* rel = nullptr;   // [T, n_heads, d_rel]
    const float* proj = nullptr;  // [d_rel, rel_extent]
    const half* sink = nullptr;   // [T, n_heads]
    float* out = nullptr;         // [T, n_heads, hd]
    int T = 0, n_heads = 0, n_kv = 0, hd = 0, d_rel = 0, rel_extent = 0;
    int pos0 = 0, window = 0, ring = 0;  // window 0 = all earlier keys; ring 0 = slot is the position
    float scale = 1.0f;
};
// softmax over [q.k * scale + bias(distance), sink]; the sink carries no value.
void attention(const AttnArgs& a, cudaStream_t s);

// out = fp16(attn * sigmoid(gate)).
void gate_mul(const float* attn, const half* gate, half* out, size_t n, cudaStream_t s);

// Router: scores = sigmoid(W x); top_k of scores + bias; weights = score / sum * route_scale.
void route_topk(const float* x, const float* w, const float* bias, int32_t* idx, float* weights, int T,
                int H, int n_experts, int top_k, float route_scale, cudaStream_t s);
// gu [slots, 2 * I] with gate / up rows interleaved -> softcap(g, gc) * sigmoid(g) * softcap(u, uc).
void expert_act(const half* gu, half* out, int slots, int I, float gate_cap, float up_cap, cudaStream_t s);
// out[t] = sum_k weights[t, k] * y[t * top_k + k].
void expert_combine(const half* y, const float* weights, float* out, int T, int top_k, int H,
                    cudaStream_t s);
// MXFP8 expert rows: y[slot, row] = sum_k e4m3(w[e, row, k]) * 2^(sc[e, row, k / 32] - 127) * x[slot / x_group, k].
void gemv_mxfp8_slots(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x, half* y,
                      int slots, int rows, int K, int x_group, cudaStream_t s);
// out = cap * tanh(x / cap).
void softcap_f32(const half* x, float* out, size_t n, float cap, cudaStream_t s);

}  // namespace imp::inkling

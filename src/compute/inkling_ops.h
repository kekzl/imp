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

// ---- fused per-site kernels (inkling_fused.cu) ----
// Rows of a stacked projection output: element (t, c) of a sub-block sits at x[t * stride + c].

// Block attention residual, nq <= 4 queries: softmax over the unit-RMS states with the learned
// queries (scale H^-0.5), v_q = sum_j p[q][j] * normed_state_j; the last state is the partial block
// state. mix = false: v = the last state as is. Per query: out_f[q] (optional) = v_q, out_h[q]
// (optional) = fp16(rms_norm(v_q) * norm_w[q]). partial: scratch of mix_partial_floats(T, H) floats.
struct MixArgs {
    MixStates st;
    const float* query = nullptr;  // [nq, H]
    const float* norm_w[4] = {};
    half* out_h[4] = {};
    float* out_f[4] = {};
    float* partial = nullptr;
    int nq = 1;
    bool mix = true;
    float eps = 1e-6f;
};
void mix(const MixArgs& a, int T, int H, cudaStream_t s);
size_t mix_partial_floats(int T, int H);

// Message gates of one layer application, n <= 3 of them: g = down . [unit_rms(h), unit_rms(msg)],
// o = (1 - tanh(forget g)) * h + tanh(gain g) * msg, out_h = fp16(rms_norm(o) * norm_w).
struct GateNormArgs {
    const float* h[3] = {};
    const float* msg[3] = {};
    const half* down[3] = {};    // [rank, 2H]
    const half* forget[3] = {};  // [H, rank]
    const half* gain[3] = {};
    const float* norm_w[3] = {};
    half* out_h[3] = {};
    int n = 1, rank = 0;
    float eps = 1e-6f;
};
// o: fp32 scratch [T, H] per gate; gd: fp32 scratch [T * n * (2 * rank + 2)].
struct GateOut {
    float* o[3] = {};
};
void gate_norm(const GateNormArgs& a, const GateOut& o, float* gd, int T, int H, cudaStream_t s);

// Key or value site: short conv over the projection x (static w [C, 4], or static_w [4, C] + per-head
// taps dyn), L2 clip (key) or unit RMS (value) per head, fp16 write at slot (*d_pos + t) % ring, and
// the conv state [3, C] advanced past the call.
struct KvSiteArgs {
    const half* x = nullptr;
    const float* w = nullptr;
    const float* static_w = nullptr;
    const half* dyn = nullptr;  // [T, stride] sub-block, n_heads * 4 taps
    float* state = nullptr;
    half* cache = nullptr;
    const int* d_pos = nullptr;  // device: position of row 0
    int T = 0, stride = 0, n_heads = 0, hd = 0, ring = 0;
    bool is_key = true;
    float thr = 0.0f, eps = 1e-6f;
};
void kv_site(const KvSiteArgs& a, cudaStream_t s);

struct AttnArgs {
    const half* q = nullptr;  // [T, stride] sub-blocks of one projection output
    const half* rel = nullptr;
    const half* sink = nullptr;
    const half* gate = nullptr;
    const float* q_scale = nullptr;    // [n_heads, hd]
    const float* rel_scale = nullptr;  // [n_heads, d_rel]
    const float* proj = nullptr;       // [rel_extent, d_rel]
    const half* k = nullptr;           // [ring or ctx, n_kv, hd]
    const half* v = nullptr;
    half* out = nullptr;         // [T, n_heads * hd] = attention * sigmoid(gate)
    const int* d_pos = nullptr;  // device: position of row 0
    // T <= 8 with part set: chunked over the keys (32 per chunk in a window, 64 over max_ctx);
    // part holds attention_part_floats(...) floats.
    float* part = nullptr;
    int max_ctx = 0;
    int T = 0, stride = 0, n_heads = 0, n_kv = 0, hd = 0, d_rel = 0, rel_extent = 0;
    int window = 0, ring = 0;  // window 0 = all earlier keys; ring 0 = slot is the position
    float scale = 1.0f, q_thr = 0.0f, rel_thr = 0.0f;
    float log_alpha = 0.0f,
          log_floor = 1.0f;  // alpha > 0: q and rel scaled by 1 + alpha * ln(max((pos + 1) / floor, 1))
};
// q = clip(q * q_scale), rel = clip(rel * rel_scale); softmax over [q.k * scale + rel . proj[distance],
// sink].
void attention(const AttnArgs& a, cudaStream_t s);
// Floats of AttnArgs::part for a model with this window and context.
size_t attention_part_floats(int n_heads, int hd, int window, int max_ctx);

// a = (conv(x) + x) / sqrt(5) with the state [3, C] advanced; out = add ? resid + a : a.
// copy (optional) receives resid.
void conv_residual(const half* x, const float* w, float* state, const float* resid, bool add, float* out,
                   float* copy, int T, int C, cudaStream_t s);
// y = sum_k weights[t, k] * ey[t * top_k + k]; a = (conv(y) + y) / sqrt(5), state advanced; out = resid + a.
// scratch: fp32 [T, C], used above 8 rows.
void moe_tail(const half* ey, const float* weights, const float* w, float* state, const float* resid,
              float* out, float* scratch, int T, int top_k, int C, cudaStream_t s);

// Router: scores = sigmoid(W x); top_k of scores + bias; weights = score / sum * route_scale.
void route_topk(const half* x, const float* w, const float* bias, float* logits, int32_t* idx, float* weights,
                int T, int H, int n_experts, int top_k, float route_scale, cudaStream_t s);
// gu [slots, 2 * I] with gate / up rows interleaved -> softcap(g, gc) * sigmoid(g) * softcap(u, uc).
void expert_act(const half* gu, half* out, int slots, int I, float gate_cap, float up_cap, cudaStream_t s);
// MXFP8 expert rows: y[slot, row] = sum_k e4m3(w[e, row, k]) * 2^(sc[e, row, k / 32] - 127) * x[slot /
// x_group, k].
void gemv_mxfp8_slots(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x, half* y,
                      int slots, int rows, int K, int x_group, cudaStream_t s);
// out = cap * tanh(x / cap).
void softcap_f32(const half* x, float* out, size_t n, float cap, cudaStream_t s);

}  // namespace imp::inkling

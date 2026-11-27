#pragma once

#include "compute/attention_cublas.h"
#include "core/dispatch_policy_fwd.h"

#include <cuda_runtime.h>

namespace imp {

// Prefill attention tier arguments (audit D-11, B-133), policy and Tensors by pointer.
// try_fa2_fp16qk_prefill / fa2_prefill_for_replay: their old parameter order; d_kv_len (device
// kv length, replay) and sinks default to nullptr as before.
struct Fa2PrefillArgs {
    const DispatchPolicy* rcfg = nullptr;
    const Tensor* q = nullptr;
    const Tensor* k = nullptr;
    const Tensor* v = nullptr;
    Tensor* o = nullptr;
    int n = 0;
    int kv_len = 0;
    int nh = 0;
    int nkv = 0;
    int hd = 0;
    float scale = 0.0f;
    int sliding_window = 0;
    float softcap = 0.0f;
    int q_offset = 0;
    cudaStream_t stream = nullptr;
    const int* d_kv_len = nullptr;
    const void* sinks = nullptr;
};

// try_fp4_prefill / try_hd512_row_invariant_prefill: try_fp4_prefill's old parameter order (the
// hd512 tier had hd second); no device kv length.
struct AttnPrefillTierArgs {
    const DispatchPolicy* rcfg = nullptr;
    const Tensor* q = nullptr;
    const Tensor* k = nullptr;
    const Tensor* v = nullptr;
    Tensor* o = nullptr;
    int n = 0;
    int kv_len = 0;
    int nh = 0;
    int nkv = 0;
    int hd = 0;
    float scale = 0.0f;
    int sliding_window = 0;
    float softcap = 0.0f;
    int q_offset = 0;
    cudaStream_t stream = nullptr;
    const void* sinks = nullptr;
};

// The tier view of an FA2 call without a device kv length (the host-length chain).
[[nodiscard]] inline AttnPrefillTierArgs prefill_tier_args(const Fa2PrefillArgs& a) {
    return {.rcfg = a.rcfg,
            .q = a.q,
            .k = a.k,
            .v = a.v,
            .o = a.o,
            .n = a.n,
            .kv_len = a.kv_len,
            .nh = a.nh,
            .nkv = a.nkv,
            .hd = a.hd,
            .scale = a.scale,
            .sliding_window = a.sliding_window,
            .softcap = a.softcap,
            .q_offset = a.q_offset,
            .stream = a.stream,
            .sinks = a.sinks};
}

// The causal cuBLAS S-matrix view of an FA2 call: S is the workspace, the kv length is K's rows.
[[nodiscard]] inline AttnCublasPrefillArgs cublas_prefill_args(const Fa2PrefillArgs& a, Tensor* S) {
    return {.Q = a.q,
            .K = a.k,
            .V = a.v,
            .O = a.o,
            .S = S,
            .n_heads = a.nh,
            .n_kv_heads = a.nkv,
            .head_dim = a.hd,
            .scale = a.scale,
            .causal = true,
            .softcap = a.softcap,
            .q_offset = a.q_offset,
            .stream = a.stream,
            .sliding_window = a.sliding_window,
            .sinks = a.sinks};
}

}  // namespace imp

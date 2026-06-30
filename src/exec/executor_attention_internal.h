#pragma once

// File-local helpers shared across the executor_attention.cu translation unit
// (split out from executor_attention.cu to keep each TU under the file-size
// gate). Included only by executor_attention.cu and its dispatch fragments;
// the static helpers below assume single-TU inclusion.

#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/executor_gemv_helpers.h"
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// Quant type dispatch helpers (file-local)
// ---------------------------------------------------------------------------

// is_dp4a_qtype() and dispatch_dp4a_gemv() are defined in executor_kernels.h

// FP16-QK FA2 for SHORT prefill (seq below fmha_prefill_threshold): replaces
// the materialized cuBLAS+softmax path with the register-resident FA2 kernel
// in f16-QK mode. Same numerical class as the cuBLAS reference (f16 inputs,
// f32 accumulate) — the short-seq e4m3 quality cliff (#511/#512) does not
// apply, and no [n × ctx] S-matrix is materialized. Declined configs
// (hd!=128, non-F16, chunk continuation) return false → caller stays on
// cuBLAS; the fp8 FMHA family is intentionally NOT a fallback here.
static bool try_fa2_fp16qk_prefill(const RuntimeConfig& rcfg, const Tensor& q, const Tensor& k,
                                   const Tensor& v, Tensor& o, int n, int kv_len, int nh, int nkv, int hd,
                                   float scale, int sliding_window, float softcap, int q_offset,
                                   cudaStream_t stream) {
    if (rcfg.attention.fa2_fp16qk == "never" || hd != 128)
        return false;
    // Chunk CONTINUATION (q_offset > 0, queries attend gathered past KV) is
    // declined: the f16-QK kernel produces wrong attention there on the
    // Llama family (teacher-forced NLL 0.29 → 7.13 on Llama-3.2-3B at
    // chunk=64; greedy output token-identical to single-shot once routed to
    // cuBLAS instead). Qwen3 was bit-exact through the same path, so this is
    // a conservative blanket decline until the kernel's q_offset handling is
    // root-caused (issue #548) — first chunks (q_offset == 0, the original #525 use case)
    // keep the fast path.
    // Chunk continuations (q_offset > 0) were declined here as a #553
    // mitigation for catastrophic NLL on the Llama family. Root cause
    // (#548) was NOT this kernel: the pinned prefill staging
    // (h_pf_token_ids_/h_pf_positions_) was rewritten by the host while
    // earlier chunks' H2D copies were still queued — the fully-async FA2
    // path let the host run far enough ahead to hit it, the cuBLAS path's
    // implicit syncs hid it. Fixed via pf_staging_evt_ in
    // engine_scheduler.cpp; kernel-level q_offset parity is locked by
    // FmhaFA2Test.FP16QK_Chunked_*.
    int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
    int64_t kv4s[4] = {1, (int64_t)kv_len, (int64_t)nkv, (int64_t)hd};
    Tensor q4 = q.reshape(4, q4s);
    Tensor k4 = k.reshape(4, kv4s);
    Tensor v4 = v.reshape(4, kv4s);
    Tensor o4 = o.reshape(4, q4s);
    return fmha_sm120_fa2_prefill(q4, k4, v4, o4, scale, /*causal=*/true, sliding_window, softcap, stream,
                                  q_offset, /*fp16_qk=*/true);
}

// Fused QKV GEMV dispatch by quant type (all share identical signatures).
static void dispatch_gemv_qkv_fused(QType qtype, const void* W_q, const void* W_k, const void* W_v,
                                    const block_q8_1* q8_1, const float* d8, half* y_q, half* y_k, half* y_v,
                                    int q_rows, int k_rows, int v_rows, int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_qkv_fused_q6k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                    stream);
            break;
        case QType::Q4_0:
            gemv_qkv_fused_q4_0_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
        case QType::Q4_K:
            gemv_qkv_fused_q4_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
        case QType::Q5_K:
            gemv_qkv_fused_q5_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
        case QType::Q2_K:
            gemv_qkv_fused_q2_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
        case QType::Q3_K:
            gemv_qkv_fused_q3_k_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
        default:
            gemv_qkv_fused_q8_0_q8_1(W_q, W_k, W_v, q8_1, d8, y_q, y_k, y_v, q_rows, k_rows, v_rows, K,
                                     stream);
            break;
    }
}

// dispatch_gemv_residual: from executor_gemv_helpers.h
// get_kv_layer: from executor_helpers.h

// Set L2 persistence hint for KV cache data on the given stream.
// Tells the GPU to prioritize keeping this address range in L2 cache.
// Resets automatically when the stream attribute is overwritten next layer.
static void set_l2_persist_kv(cudaStream_t stream, const void* kv_ptr, size_t kv_bytes) {
    if (!kv_ptr || kv_bytes == 0 || !stream)
        return;
    // Query device limits once. persistingL2CacheMaxSize caps how much of L2 can
    // persist (hitRatio target); accessPolicyMaxWindowSize caps the attribute's
    // address-window extent. Setting num_bytes above the window cap returns
    // cudaErrorInvalidValue, which poisons the stream for every subsequent kernel.
    static size_t max_persist = 0;
    static size_t max_window = 0;
    if (max_persist == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist == 0)
            return;  // L2 persistence not supported
        int mw = 0;
        if (cudaDeviceGetAttribute(&mw, cudaDevAttrMaxAccessPolicyWindowSize, 0) == cudaSuccess && mw > 0) {
            max_window = static_cast<size_t>(mw);
        } else {
            max_window = 128ULL * 1024 * 1024;
        }
    }
    // hitRatio: compare against total KV size so the hardware probabilistically
    // persists a representative subset even when kv_bytes exceeds the window.
    float ratio = (kv_bytes <= max_persist) ? 1.0f
                                            : static_cast<float>(max_persist) / static_cast<float>(kv_bytes);
    size_t window_bytes = kv_bytes < max_window ? kv_bytes : max_window;
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(kv_ptr);
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio = ratio;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    // L2 persistence is a best-effort perf hint. A failed set (e.g. num_bytes vs
    // the per-context persisting-L2 reservation left in a different state by a
    // previously-loaded model in this process) returns cudaErrorInvalidValue,
    // which the runtime records as a STICKY per-context error — it then poisons
    // every subsequent kernel in this forward (the CUTLASS/MoE GEMMs bail on a
    // pending error → degenerate garbage). Drain it immediately so a perf hint
    // can never corrupt correctness. (Cross-model repro: a GDN/SSM model loaded
    // before this one garbled the output until this drain was added.)
    if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess)
        (void)cudaGetLastError();
}

// Alias for the shared clear_l2_policy helper (back-compat name for call sites).
static void clear_l2_persist(cudaStream_t stream) { clear_l2_policy(stream); }

}  // namespace imp

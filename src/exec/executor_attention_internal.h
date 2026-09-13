#pragma once

// File-local helpers shared across the executor_attention.cu TU, split out
// to keep each TU under the file-size gate. Included only by
// executor_attention.cu and its dispatch fragments; static helpers assume single-TU inclusion.

#include "exec/attention_dispatch_rules.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/executor_gemv_helpers.h"
#include "compute/attention_fmha_sm120.h"
#include "core/tensor.h"
#include "core/dispatch_policy.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// Quant type dispatch helpers (file-local)
// ---------------------------------------------------------------------------

// is_dp4a_qtype() and dispatch_dp4a_gemv() are defined in executor_kernels.h

// FP16-QK FA2 for SHORT prefill (seq < fmha_prefill_threshold): register-
// resident FA2 in f16-QK mode instead of materialized cuBLAS+softmax; same
// numerical class as cuBLAS (f16 in, f32 accumulate), avoids the e4m3
// quality cliff (#511/#512), no S-matrix. Declined configs (hd!=128,
// non-F16, chunk continuation) return false; the fp8 FMHA family is NOT a fallback here.
static bool try_fa2_fp16qk_prefill(const DispatchPolicy& rcfg, const Tensor& q, const Tensor& k,
                                   const Tensor& v, Tensor& o, int n, int kv_len, int nh, int nkv, int hd,
                                   float scale, int sliding_window, float softcap, int q_offset,
                                   cudaStream_t stream, const int* d_kv_len = nullptr) {
    if (rcfg.attention.fa2_fp16qk == "never")
        return false;
    // hd=256: stage-1 FA2 port (attention.fa2_hd256, default on since #932).
    // Other head dims decline as before; the kernel wrapper re-checks the gate.
    if (hd != 128 && !(hd == 256 && rcfg.attention.fa2_hd256))
        return false;
    // Chunk CONTINUATION (q_offset>0) declines the f16-QK kernel: wrong
    // attention on the Llama family at chunk boundaries (conservative blanket
    // decline pending kernel-level root cause, #548). First chunks
    // (q_offset==0) keep the fast path.
    // Root cause of the original #553 symptom was NOT this kernel: pinned
    // prefill staging (h_pf_token_ids_/h_pf_positions_) was rewritten by the
    // host while earlier chunks' H2D copies were still queued; the async FA2
    // path exposed it, cuBLAS's implicit syncs hid it. Fixed via
    // pf_staging_evt_ in engine_scheduler.cpp; q_offset parity is locked by FmhaFA2Test.FP16QK_Chunked_*.
    int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
    int64_t kv4s[4] = {1, (int64_t)kv_len, (int64_t)nkv, (int64_t)hd};
    Tensor q4 = q.reshape(4, q4s);
    Tensor k4 = k.reshape(4, kv4s);
    Tensor v4 = v.reshape(4, kv4s);
    Tensor o4 = o.reshape(4, q4s);
    return fmha_sm120_fa2_prefill(q4, k4, v4, o4, scale, /*causal=*/true, sliding_window, softcap, stream,
                                  q_offset, /*fp16_qk=*/true, d_kv_len);
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
    // Query device limits once: persistingL2CacheMaxSize caps persistable L2
    // (hitRatio target), accessPolicyMaxWindowSize caps the attribute's
    // address-window extent. num_bytes above the window cap returns
    // cudaErrorInvalidValue, poisoning the stream for every later kernel.
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
    // L2 persistence is a best-effort perf hint. A failed set (e.g. a
    // different model's leftover per-context persisting-L2 reservation)
    // returns cudaErrorInvalidValue, which the runtime records as a STICKY
    // per-context error, poisoning every later kernel in this forward
    // (CUTLASS/MoE GEMMs bail on a pending error -> degenerate garbage). Drain
    // it immediately so a perf hint can never corrupt correctness.
    if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess)
        (void)cudaGetLastError();
}

// Alias for the shared clear_l2_policy helper (back-compat name for call sites).
static void clear_l2_persist(cudaStream_t stream) { clear_l2_policy(stream); }

}  // namespace imp

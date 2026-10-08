#pragma once

// File-local helpers shared across the executor_attention.cu TU, split out
// to keep each TU under the file-size gate. Included only by
// executor_attention.cu and its dispatch fragments; static helpers assume single-TU inclusion.

#include "exec/attention_dispatch_rules.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/executor_gemv_helpers.h"
#include "compute/attention_cublas.h"
#include "compute/attention_fmha_sm120.h"
#include "compute/attention_apa.h"
#include "compute/attention_ra2.h"
#include "compute/dispatch_paths.h"  // AttnPrefillOuter
#include "compute/kv_gather.h"
#include "core/logging.h"
#include "exec/inference_state.h"
#include "memory/kv_cache.h"
#include "core/tensor.h"
#include "core/dispatch_policy.h"
#include "core/process_diag.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

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
[[nodiscard]] static bool try_fa2_fp16qk_prefill(const DispatchPolicy& rcfg, const Tensor& q, const Tensor& k,
                                   const Tensor& v, Tensor& o, int n, int kv_len, int nh, int nkv, int hd,
                                   float scale, int sliding_window, float softcap, int q_offset,
                                   cudaStream_t stream, const int* d_kv_len = nullptr,
                                   const void* sinks = nullptr) {
    if (rcfg.attention.fa2_fp16qk == "never")
        return false;
    // hd=256: stage-1 FA2 port (attention.fa2_hd256, default on since #932).
    // Other head dims decline as before; the kernel wrapper re-checks the gate.
    if (hd != 64 && hd != 128 && !(hd == 256 && rcfg.attention.fa2_hd256))
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
                                  q_offset, /*fp16_qk=*/true, d_kv_len, static_cast<const half*>(sinks));
}

// APA serves every length unless RA2 is on too: then RA2 below attention.apa_min_kv (faster there), APA
// above (dense prefill per chunk APA/RA2 1.13 at kv 26624, 0.94 at 43008).
template <typename A>
[[nodiscard]] static bool apa_serves_kv(const A& a, int kv_len) {
    return !a.ra2_prefill || kv_len >= a.apa_min_kv;
}

// Chunked prefill: gather past paged KV [0, gather_cap) of the cache dtype into flat FP16 k_full / v_full.
static void gather_past_kv_fp16(KVCache* cache, int kv_layer, half* k_full, half* v_full, const int* past_bt,
                                int gather_cap, int kv_bs, int nkv, int hd,
                                const std::vector<float>& kv_scales, cudaStream_t stream, const int* d_past) {
    const QType kvt = cache->qtype();
    if (kvt == QType::F16) {
        paged_kv_gather_fp16(k_full, static_cast<const half*>(cache->k_ptr(kv_layer, 0)), past_bt, gather_cap,
                             kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_fp16(v_full, static_cast<const half*>(cache->v_ptr(kv_layer, 0)), past_bt, gather_cap,
                             kv_bs, nkv, hd, stream, d_past);
    } else if (kvt == QType::FP8_E4M3) {
        const float kv_scale = kv_layer < (int)kv_scales.size() ? kv_scales[kv_layer] : 1.0f;
        paged_kv_gather_fp8_to_fp16(k_full, static_cast<const __nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                                    past_bt, kv_scale, gather_cap, kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_fp8_to_fp16(v_full, static_cast<const __nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
                                    past_bt, kv_scale, gather_cap, kv_bs, nkv, hd, stream, d_past);
    } else if (kvt == QType::NVFP4) {
        paged_kv_gather_nvfp4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                      static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)), past_bt,
                                      gather_cap, kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_nvfp4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                      static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)), past_bt,
                                      gather_cap, kv_bs, nkv, hd, stream, d_past);
    } else if (kvt == QType::MXFP4_KV) {
        paged_kv_gather_mxfp4_kv_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                         static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                         past_bt, gather_cap, kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_mxfp4_kv_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                         static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                         past_bt, gather_cap, kv_bs, nkv, hd, stream, d_past);
    } else if (kvt == QType::INT8) {  // symmetric 8-bit, per-head FP16 scale
        paged_kv_gather_int8_to_fp16(k_full, static_cast<const int8_t*>(cache->k_ptr(kv_layer, 0)),
                                     static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)), past_bt,
                                     gather_cap, kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_int8_to_fp16(v_full, static_cast<const int8_t*>(cache->v_ptr(kv_layer, 0)),
                                     static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)), past_bt,
                                     gather_cap, kv_bs, nkv, hd, stream, d_past);
    } else {  // INT4: symmetric 4-bit with per-head FP16 scale
        paged_kv_gather_int4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                     static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)), past_bt,
                                     gather_cap, kv_bs, nkv, hd, stream, d_past);
        paged_kv_gather_int4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                     static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)), past_bt,
                                     gather_cap, kv_bs, nkv, hd, stream, d_past);
    }
}

// APA on the paged FP16 cache in chunked prefill, no gather: past keys [0, att_off) through past_bt, the
// current chunk from kk / vv. Gates as try_fp4_prefill's APA arm, plus FP16 cache and no capture replay.
[[nodiscard]] static bool try_apa_paged_prefill(const DispatchPolicy& rcfg, bool cap_replay, const Tensor& q,
                                                KVCache* cache, int kv_layer, const int* past_bt, int kv_bs,
                                                const Tensor& kk, const Tensor& vv, int att_off, Tensor& o,
                                                int n, int nh, int nkv, int hd, float scale,
                                                int sliding_window, float softcap, cudaStream_t stream,
                                                const void* sinks) {
    const auto& a = rcfg.attention;
    if (cap_replay || a.apa_eps <= 0.0f || cache->qtype() != QType::F16 || !apa_serves_kv(a, att_off + n) ||
        sliding_window > 0 || softcap != 0.0f || sinks != nullptr || process_diag_deterministic())
        return false;
    return attention_apa_prefill_paged(q, static_cast<const half*>(cache->k_ptr(kv_layer, 0)),
                                       static_cast<const half*>(cache->v_ptr(kv_layer, 0)), past_bt, kv_bs,
                                       kk, vv, att_off, o, n, att_off + n, nh, nkv, hd, scale, att_off,
                                       a.apa_eps, kv_layer, stream);
}

// Opt-in FP4 prefill tiers, tried before FA2: APA (attention.apa_eps > 0, apa_serves_kv), then
// RA2 (attention.ra2_prefill). Both decline what the kernels do not model (sliding window, softcap,
// sinks), runtime.deterministic (float atomics in the K/V stats), and shapes or scratch they reject;
// UNSET = the caller walks its usual chain.
[[nodiscard]] static AttnPrefillOuter try_fp4_prefill(const DispatchPolicy& rcfg, const Tensor& q,
                                                      const Tensor& k, const Tensor& v, Tensor& o, int n,
                                                      int kv_len, int nh, int nkv, int hd, float scale,
                                                      int sliding_window, float softcap, int q_offset,
                                                      cudaStream_t stream, const void* sinks) {
    const auto& a = rcfg.attention;
    if ((!a.ra2_prefill && a.apa_eps <= 0.0f) || sliding_window > 0 || softcap != 0.0f || sinks != nullptr ||
        process_diag_deterministic())
        return AttnPrefillOuter::UNSET;
    if (a.apa_eps > 0.0f && apa_serves_kv(a, kv_len) &&
        attention_apa_prefill(q, k, v, o, n, kv_len, nh, nkv, hd, scale, q_offset, a.apa_eps, stream))
        return AttnPrefillOuter::APA;
    if (a.ra2_prefill && attention_ra2_prefill(q, k, v, o, n, kv_len, nh, nkv, hd, scale, q_offset, stream))
        return AttnPrefillOuter::RA2;
    return AttnPrefillOuter::UNSET;
}

// hd=512 prefill whose rows do not depend on n or q_offset (#2167): the WMMA FMHA with a forward
// KV scan, used by the single-shot and the chunk-continuation path alike.
// attention.hd512_prefill: "fmha" always, "cublas" never, "auto" under runtime.deterministic.
// False = not selected or declined.
[[nodiscard]] static bool try_hd512_row_invariant_prefill(const DispatchPolicy& rcfg, int hd, const Tensor& q,
                                                          const Tensor& k, const Tensor& v, Tensor& o, int n,
                                                          int kv_len, int nh, int nkv, float scale,
                                                          int sliding_window, float softcap, int q_offset,
                                                          cudaStream_t stream, const void* sinks) {
    const std::string& mode = rcfg.attention.hd512_prefill;
    if (hd != 512 || !(mode == "fmha" || (mode == "auto" && process_diag_deterministic())))
        return false;
    int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
    int64_t kv4s[4] = {1, (int64_t)kv_len, (int64_t)nkv, (int64_t)hd};
    Tensor q4 = q.reshape(4, q4s);
    Tensor k4 = k.reshape(4, kv4s);
    Tensor v4 = v.reshape(4, kv4s);
    Tensor o4 = o.reshape(4, q4s);
    return fmha_sm120_prefill(q4, k4, v4, o4, scale, /*causal=*/true, sliding_window, softcap, stream,
                              q_offset, static_cast<const half*>(sinks), /*fixed_kv_order=*/true);
}

// Capture-replay chunk: FA2 only (the S-matrix/FMHA fallbacks bound work from the baked ctx_len).
// A decline means the engine-side eligibility check and the kernel disagree: throw, never fall back.
static void fa2_prefill_for_replay(const DispatchPolicy& rcfg, const Tensor& q, const Tensor& k,
                                   const Tensor& v, Tensor& o, int n, int kv_capacity, int nh, int nkv,
                                   int hd, float scale, int sliding_window, float softcap, int q_offset,
                                   cudaStream_t stream, const int* d_kv_len, const void* sinks) {
    if (!try_fa2_fp16qk_prefill(rcfg, q, k, v, o, n, kv_capacity, nh, nkv, hd, scale, sliding_window, softcap,
                                q_offset, stream, d_kv_len, sinks))
        throw std::runtime_error("chunked_prefill: FA2 declined a capture-replay chunk");
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
        if (const cudaError_t err = cudaGetDeviceProperties(&prop, 0); err != cudaSuccess) {
            IMP_LOG_WARN("set_l2_persist_kv: device query failed (%s), no L2 hint", cudaGetErrorString(err));
            return;  // uncached: max_persist stays 0 (hint only, output unchanged)
        }
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

// Chunked-prefill per-call K/V gather buffer: K at 0, V at the 256 B-aligned offset.
static size_t chunk_kv_v_offset(size_t kv_bytes) { return (kv_bytes + 255) & ~static_cast<size_t>(255); }
static size_t chunk_kv_bytes(size_t kv_bytes) { return 2 * chunk_kv_v_offset(kv_bytes); }

// Sets *k / *v only when the allocation succeeded; they stay nullptr otherwise.
static void chunk_kv_split(cudaError_t err, half* kv, size_t kv_bytes, half** k, half** v) {
    if (err != cudaSuccess || kv == nullptr) return;
    const size_t v_elems = chunk_kv_v_offset(kv_bytes) / sizeof(half);
    *k = kv;
    *v = kv + v_elems;
}

// The append offsets both buffers by q_offset rows; null is UB (#2288).
static void require_chunk_kv(const half* k, const half* v) {
    if (k == nullptr || v == nullptr)
        throw std::runtime_error("chunked_prefill: K/V gather buffer allocation failed");
}

// DEBUG attention.force_cublas_decode (n == 1): K/V rebuilt from the paged cache, then the
// materialized QK^T path prefill uses. false = setup failed (ERROR logged, ao not written): the
// caller runs paged attention. Sync copies: never under graph capture.
[[nodiscard]] static bool cublas_decode_reference(const InferenceState& state, int layer, int kv_layer,
                                    const int* layer_block_tables, const Tensor& qv, Tensor& ao, int nh,
                                    int nkv, int hd, float scale, float softcap, int sliding_window,
                                    const void* sinks, cudaStream_t stream) {
    KVCache* cache_dbg = state.kv_cache;
    int ctx_len = 0;
    // First failure stops the setup.
    cudaError_t dbg_err = cudaMemcpy(&ctx_len, state.context_lens, sizeof(int), cudaMemcpyDeviceToHost);
    // Allocate temp K/V for all context tokens
    int kv_elems = ctx_len * nkv * hd;
    half *k_flat = nullptr, *v_flat = nullptr;
    if (dbg_err == cudaSuccess)
        dbg_err = cudaMalloc(&k_flat, kv_elems * sizeof(half));
    if (dbg_err == cudaSuccess)
        dbg_err = cudaMalloc(&v_flat, kv_elems * sizeof(half));
    // Copy from paged KV cache to contiguous buffer
    int kv_bs = cache_dbg->block_size();
    int n_blocks = (ctx_len + kv_bs - 1) / kv_bs;
    // Heap-sized to n_blocks — a fixed [1024] stack array overran 4x at
    // the 64K context cap (n_blocks=4096) and silently smashed the frame.
    std::vector<int32_t> h_block_table(n_blocks > 0 ? n_blocks : 1);
    if (dbg_err == cudaSuccess)
        dbg_err = cudaMemcpy(h_block_table.data(), layer_block_tables, n_blocks * sizeof(int32_t),
                             cudaMemcpyDeviceToHost);
    // SWA trailing-free holes (-1) leave their rows zeroed — they sit
    // outside the window mask, so the reference output is unaffected.
    if (dbg_err == cudaSuccess)
        dbg_err = cudaMemset(k_flat, 0, kv_elems * sizeof(half));
    if (dbg_err == cudaSuccess)
        dbg_err = cudaMemset(v_flat, 0, kv_elems * sizeof(half));
    for (int b = 0; dbg_err == cudaSuccess && b < n_blocks; b++) {
        int block_id = h_block_table[b];
        if (block_id < 0)
            continue;
        int toks_in_block = std::min(kv_bs, ctx_len - b * kv_bs);
        size_t row_bytes = static_cast<size_t>(nkv * hd) * sizeof(half);
        half* k_src = static_cast<half*>(cache_dbg->k_ptr(kv_layer, block_id));
        half* v_src = static_cast<half*>(cache_dbg->v_ptr(kv_layer, block_id));
        dbg_err = cudaMemcpy(k_flat + static_cast<ptrdiff_t>(b * kv_bs * nkv * hd), k_src,
                             toks_in_block * row_bytes, cudaMemcpyDeviceToDevice);
        if (dbg_err == cudaSuccess)
            dbg_err = cudaMemcpy(v_flat + static_cast<ptrdiff_t>(b * kv_bs * nkv * hd), v_src,
                                 toks_in_block * row_bytes, cudaMemcpyDeviceToDevice);
    }
    // Reshape for cuBLAS attention: Q[1,nh,hd], K[ctx_len,nkv,hd], V[ctx_len,nkv,hd]
    int64_t k_shape[2] = {ctx_len, static_cast<int64_t>(nkv * hd)};
    int64_t v_shape[2] = {ctx_len, static_cast<int64_t>(nkv * hd)};
    Tensor k_cont(k_flat, QType::F16, 2, k_shape, true);
    Tensor v_cont(v_flat, QType::F16, 2, v_shape, true);
    // n=1 cuBLAS attention: causal=true + q_offset=ctx_len-1 lets the single
    // query see the full context AND keeps the sliding-window mask correct
    // (causal=false/q_offset=0 broke SWA: abs_row=0 never triggered the
    // window). Sinks pass through so this stays a faithful gpt-oss reference (#547).
    {
        int64_t s_shape[3] = {(int64_t)nh, 1, (int64_t)ctx_len};
        half* s_buf = nullptr;
        if (dbg_err == cudaSuccess)
            dbg_err = cudaMalloc(&s_buf, (size_t)nh * ctx_len * sizeof(half));
        if (dbg_err == cudaSuccess) {
            Tensor s_view(s_buf, QType::F16, 3, s_shape, true);
            attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view, nh, nkv, hd, scale, /*causal=*/true,
                                     softcap,
                                     /*q_offset=*/ctx_len - 1, stream, sliding_window, sinks);
        }
        IMP_CUDA_CHECK_LOG(cudaFree(s_buf));
    }
    IMP_CUDA_CHECK_LOG(cudaFree(k_flat));
    IMP_CUDA_CHECK_LOG(cudaFree(v_flat));
    if (dbg_err != cudaSuccess)
        IMP_LOG_ERROR("force_cublas_decode: reference setup failed (%s), layer %d uses paged attention",
                      cudaGetErrorString(dbg_err), layer);
    return dbg_err == cudaSuccess;
}

}  // namespace imp

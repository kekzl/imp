#include "exec/executor.h"
#include "lora/lora_adapter.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "memory/kv_cache_manager.h"
#include "exec/executor_gemv_helpers.h"
#include "exec/executor_debug.h"
#include "exec/gemm_context.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_moe_fused_tc.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_fmha_sm120.h"
#include "compute/attention_paged.h"
#include "compute/kv_gather.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "memory/gdn_state.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/hadamard.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

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
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

// Alias for the shared clear_l2_policy helper (back-compat name for call sites).
static void clear_l2_persist(cudaStream_t stream) { clear_l2_policy(stream); }

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state, cudaStream_t stream) {
    // Configure shared workspace for attention phase
    configure_attn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& prof = model_->profile();
    using AttnVariant = ModelProfile::AttnVariant;
    const auto& ly = model_->layer(layer);
    int n = state.n_tokens;
    int nh = cfg.n_heads;
    // Per-layer head_dim / n_kv_heads (Gemma 4 dual geometry + Nemotron-H hybrid)
    int nkv = (!cfg.n_kv_heads_per_layer.empty() && layer < (int)cfg.n_kv_heads_per_layer.size() &&
               cfg.n_kv_heads_per_layer[layer] > 0)
                  ? cfg.n_kv_heads_per_layer[layer]
                  : cfg.n_kv_heads;
    int hd = (!cfg.head_dim_per_layer.empty() && layer < (int)cfg.head_dim_per_layer.size() &&
              cfg.head_dim_per_layer[layer] > 0)
                 ? cfg.head_dim_per_layer[layer]
                 : (cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh));
    // Gemma 4: derive per-layer n_heads and n_kv_heads from actual tensor shapes.
    // Layer 0 (SWA) wq=[4096,2816] wk=[2048,2816] → 16 Q × hd=256, 8 KV × hd=256
    // Layer 5 (Global) wq=[8192,2816] wk=[1024,2816] → 16 Q × hd=512, 2 KV × hd=512
    // Authoritative source = the loaded tensor shapes; per-layer config can lag.
    if (prof.is_gemma4 && hd > 0 && ly.wq.data != nullptr) {
        int wq_out = static_cast<int>(ly.wq.shape[0]);
        if (wq_out > 0 && (wq_out % hd) == 0) {
            int nh_layer = wq_out / hd;
            if (nh_layer > 0 && nh_layer != nh)
                nh = nh_layer;
        }
        if (ly.wk.data != nullptr) {
            int wk_out = static_cast<int>(ly.wk.shape[0]);
            if (wk_out > 0 && (wk_out % hd) == 0) {
                int nkv_layer = wk_out / hd;
                if (nkv_layer > 0 && nkv_layer != nkv)
                    nkv = nkv_layer;
            }
        }
    }
    float eps = cfg.rms_norm_eps;

    // Sized views for this call (never mutates member tensors).
    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // Qwen3.5 attention: Q projection has ×2 output (Q + output_gate fused).
    // Detect by checking if wq output dim > n_heads * head_dim.
    int q_out_dim = static_cast<int>(ly.wq.shape[0]);
    bool has_attn_output_gate = (q_out_dim > nh * hd);
    int q_actual_dim = nh * hd;  // actual Q dimension (without gate)

    Tensor qv = view_tokens(q_, n);
    Tensor kk = view_tokens(k_, n);
    Tensor vv = view_tokens(v_, n);
    Tensor ao = view_tokens(attn_out_, n);
    Tensor po = view_tokens(proj_out_, n);

    const bool per_layer_shapes = (!cfg.head_dim_per_layer.empty() || !cfg.n_kv_heads_per_layer.empty());

    // Per-layer shape narrowing: Q/K/V/ao workspace tensors are allocated with
    // max shapes (for worst-case layer). For layers with smaller head_dim (Gemma 4
    // SWA), narrow the view so cuBLAS gemm writes with the correct leading dim.
    if (per_layer_shapes) {
        auto narrow_cols = [](Tensor& t, int64_t new_cols) {
            if (t.shape[1] != new_cols) {
                t.shape[1] = new_cols;
                t.compute_strides();
            }
        };
        narrow_cols(qv, static_cast<int64_t>(nh) * hd);
        narrow_cols(kk, static_cast<int64_t>(nkv) * hd);
        narrow_cols(vv, static_cast<int64_t>(nkv) * hd);
        narrow_cols(ao, static_cast<int64_t>(nh) * hd);
    }

    // For Qwen3.5 attention output gate: allocate larger Q buffer AFTER all
    // standard attention buffers to avoid overlap (q_/k_/v_/attn_out_/proj_out_
    // all share the same shared_workspace_ memory).
    Tensor qv_full;
    if (has_attn_output_gate) {
        auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
        size_t es_a = dtype_size(compute_dtype_);
        // Place after proj_out_ (last standard buffer)
        char* after_proj = static_cast<char*>(po.data) +
                           align256(static_cast<size_t>(n) * cfg.d_model * es_a);
        int64_t qfull_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(q_out_dim)};
        qv_full = Tensor(after_proj, compute_dtype_, 2, qfull_shape, true);
    }

    // Per-step diagnostics for n>1 decode debugging (layer 0 only)
    bool debug_attn_steps = (layer == 0 && n > 1 && debug_forward_enabled());
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step0_h_input", h, stream);
    }

    // 1. Save residual for later add-back.
    //    Optimization: for decode (n=1) with dp4a, fuse residual into GEMV.
    //    For prefill (n>1) with FP16 cache, use cuBLAS beta=1 to fuse residual
    //    into the wo projection GEMM — no separate residual save/add/copy needed.
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    // True sandwich norm: post_attn_norm applied inside run_attention AND separate
    // ffn_norm for run_ffn (Gemma-3 pattern). When ffn_norm is absent (Qwen3.5),
    // post_attn_norm serves as FFN input norm in run_ffn — NOT a sandwich norm.
    // Without this check, post_attn_norm is applied TWICE (here + run_ffn fallback).
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr && ly.ffn_norm.data != nullptr);
    // FP32 residual accumulator (Gemma-3 dense + Gemma-4 MoE post-norm architecture).
    // Kernel semantics: fp32_h += rmsnorm(po) * w. llama's build_norm(attn) + residual
    // is mathematically identical for both Gemma-3 and Gemma-4 (normalize-then-add).
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
    if (debug_forward_enabled() && layer <= 1) {
        fprintf(stderr,
                "[DEBUG_FWD] L%d attn: has_post_attn_norm=%d using_fp32_accum=%d "
                "post_attn_norm=%p ffn_norm=%p\n",
                layer, (int)has_post_attn_norm, (int)using_fp32_accum, ly.post_attn_norm.data,
                ly.ffn_norm.data);
    }
    const StorageTier wo_tier = (ly.wo_id != kInvalidTensorID) ? registry_.handle(ly.wo_id).primary_tier
                                                               : StorageTier::Undefined;
    // NVFP4 wo: primary tier OR Phase-3 secondary NVFP4 decode cache
    // (Q8_0/Q6_K/Q5_K models). c8763ad refactor dropped the secondary check.
    const bool wo_nvfp4_secondary = (wcache_.nvfp4.count(ly.wo.data) != 0);
    bool will_fuse_o_nvfp4 = (!has_post_attn_norm && n == 1 && h.qtype == QType::F16 &&
                              (wo_tier == StorageTier::NVFP4 || wo_nvfp4_secondary));
    bool will_fuse_o_residual = (!has_post_attn_norm && !will_fuse_o_nvfp4 && n == 1 &&
                                 qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr &&
                                 h.qtype == QType::F16 && is_dp4a_qtype(ly.wo.qtype));
    bool will_fuse_o_beta1 = (!has_post_attn_norm && !will_fuse_o_residual && !will_fuse_o_nvfp4 && n > 1 &&
                              (wo_tier == StorageTier::FP16 || wo_tier == StorageTier::FP8));
    // Dequant beta=1 path: when force_fp16_gemm bypasses FP8, dequant weights on-the-fly
    bool will_fuse_o_dequant_beta1 = (!has_post_attn_norm && !will_fuse_o_residual && !will_fuse_o_nvfp4 &&
                                      !will_fuse_o_beta1 && n > 1 && qscratch_.dequant != nullptr &&
                                      dequant_gpu_supported(ly.wo.qtype));
    if (!will_fuse_o_residual && !will_fuse_o_beta1 && !will_fuse_o_dequant_beta1 && !will_fuse_o_nvfp4 &&
        !using_fp32_accum) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }

    // For Qwen3.5: Q projection writes to larger buffer (includes gate), then split
    Tensor q_target = has_attn_output_gate ? qv_full : qv;

    // GemmContext for all weight GEMM dispatches in this function.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, runtime_config(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq);

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    For decode (n=1) with matching quant types: fused RMSNorm→Q8_1→QKV GEMV.
    //    This skips the intermediate norm_out FP16 buffer entirely.
    //    Otherwise falls back to separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
        auto* q8 = static_cast<block_q8_1*>(qscratch_.q8_1_buf);
        // MXFP4 decode path: native MXFP4 GEMV with UE8M0 scales
        const WeightHandle* mxfp4_hwq = (ly.wq_id != kInvalidTensorID) ? &registry_.handle(ly.wq_id)
                                                                       : nullptr;
        const WeightHandle* mxfp4_hwk = (ly.wk_id != kInvalidTensorID) ? &registry_.handle(ly.wk_id)
                                                                       : nullptr;
        const WeightHandle* mxfp4_hwv = (ly.wv_id != kInvalidTensorID) ? &registry_.handle(ly.wv_id)
                                                                       : nullptr;
        bool mxfp4_qkv = (!has_attn_output_gate && n == 1 && mxfp4_hwq &&
                          mxfp4_hwq->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwq->payload.mxfp4.linear_scales && mxfp4_hwk &&
                          mxfp4_hwk->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwk->payload.mxfp4.linear_scales && mxfp4_hwv &&
                          mxfp4_hwv->primary_tier == StorageTier::MXFP4 &&
                          mxfp4_hwv->payload.mxfp4.linear_scales);
        // NVFP4 decode path: uses FP16 input (no Q8_1 quantization needed).
        // Two sources: (1) primary NVFP4 storage tier (native NVFP4 models),
        // (2) secondary NVFP4 decode cache populated by Phase 3 for
        // Q8_0/Q6_K/Q5_K models — c8763ad refactor lost this fallback;
        // restore by also probing `wcache_.nvfp4`.
        auto nv_q_it = wcache_.nvfp4.find(ly.wq.data);
        auto nv_k_it = wcache_.nvfp4.find(ly.wk.data);
        auto nv_v_it = wcache_.nvfp4.find(ly.wv.data);
        const bool nv_q_secondary = (nv_q_it != wcache_.nvfp4.end());
        const bool nv_k_secondary = (nv_k_it != wcache_.nvfp4.end());
        const bool nv_v_secondary = (nv_v_it != wcache_.nvfp4.end());
        const bool wq_is_nvfp4 = (mxfp4_hwq && mxfp4_hwq->primary_tier == StorageTier::NVFP4) ||
                                 nv_q_secondary;
        const bool wk_is_nvfp4 = (mxfp4_hwk && mxfp4_hwk->primary_tier == StorageTier::NVFP4) ||
                                 nv_k_secondary;
        const bool wv_is_nvfp4 = (mxfp4_hwv && mxfp4_hwv->primary_tier == StorageTier::NVFP4) ||
                                 nv_v_secondary;
        bool nvfp4_qkv = (!has_attn_output_gate && n == 1 && wq_is_nvfp4 && wk_is_nvfp4 && wv_is_nvfp4);
        // Gemma-4: disable fused QKV when FP32 accum is active — the fused kernel
        // reads FP16 h instead of fp32_hidden_, losing precision through 128-expert routing.
        bool fused_qkv = (!has_attn_output_gate && n == 1 && q8 != nullptr && qscratch_.d8_buf != nullptr &&
                          no.qtype == QType::F16 && ly.wq.qtype == ly.wk.qtype &&
                          ly.wk.qtype == ly.wv.qtype && is_dp4a_qtype(ly.wq.qtype) &&
                          !(using_fp32_accum && prof.is_gemma4));
        if (mxfp4_qkv) {
            // MXFP4 fused QKV: RMSNorm, optional Hadamard, then MXFP4 GEMV
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(mxfp4_hwq->shape[0]);
            int k_rows = static_cast<int>(mxfp4_hwk->shape[0]);
            int v_rows = static_cast<int>(mxfp4_hwv->shape[0]);
            int K = static_cast<int>(mxfp4_hwq->shape[1]);
            int hbs = mxfp4_hwq->payload.mxfp4.hadamard_bs;
            if (hbs > 0 && hadamard_block_size_valid(hbs))
                hadamard_transform_fp16(static_cast<const half*>(no.data), static_cast<half*>(no.data), 1, K,
                                        hbs, stream);
            // Reconstruct CutlassMxFP4Weight structs from handle payloads.
            auto make_mxfp4 = [](const WeightHandle* h) {
                CutlassMxFP4Weight mw;
                mw.data = h->payload.mxfp4.weight;
                mw.scale_factors = h->payload.mxfp4.scales;
                mw.linear_scales = h->payload.mxfp4.linear_scales;
                mw.hadamard_bs = h->payload.mxfp4.hadamard_bs;
                mw.tensor_scale = 1.0f;
                mw.N = static_cast<int>(h->shape[0]);
                mw.K = static_cast<int>(h->shape[1]);
                mw.sf_bytes = cutlass_mxfp4_sf_size(mw.N, mw.K);
                mw.owns_data = false;
                return mw;
            };
            auto mw_q = make_mxfp4(mxfp4_hwq);
            auto mw_k = make_mxfp4(mxfp4_hwk);
            auto mw_v = make_mxfp4(mxfp4_hwv);
            gemv_mxfp4_qkv_fused(mw_q, mw_k, mw_v, static_cast<const half*>(no.data),
                                 static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                 static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else if (nvfp4_qkv) {
            // NVFP4 fused QKV: RMSNorm to FP16, then NVFP4 GEMV (no Q8_1 needed)
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            // Pick the NvFP4QuantResult: secondary cache (Q8_0+NVFP4-decode) is
            // already a populated struct; primary-tier handle needs reconstruction.
            auto from_handle = [](const WeightHandle* hw) {
                NvFP4QuantResult tmp;
                tmp.packed_data = hw->payload.nvfp4.data;
                tmp.micro_scales = hw->payload.nvfp4.block_scales;
                tmp.tensor_scale = (hw->payload.nvfp4.tensor_scale != nullptr)
                                       ? *hw->payload.nvfp4.tensor_scale
                                       : 1.0f;
                tmp.N = static_cast<int>(hw->shape[0]);
                tmp.K = static_cast<int>(hw->shape[1]) * 2;
                return tmp;
            };
            NvFP4QuantResult nv_q = nv_q_secondary ? nv_q_it->second : from_handle(mxfp4_hwq);
            NvFP4QuantResult nv_k = nv_k_secondary ? nv_k_it->second : from_handle(mxfp4_hwk);
            NvFP4QuantResult nv_v = nv_v_secondary ? nv_v_it->second : from_handle(mxfp4_hwv);
            int q_rows = nv_q.N;
            int k_rows = nv_k.N;
            int v_rows = nv_v.N;
            int K = nv_q.K;
            gemv_nvfp4_qkv_fused(nv_q, nv_k, nv_v, static_cast<const half*>(no.data),
                                 static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                 static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else if (fused_qkv) {
            // Fused: RMSNorm + Q8_1 quantization in one kernel (no norm_out write)
            int K = static_cast<int>(ly.wq.shape[1]);
            // LoRA needs the normed projection input materialized — side-write
            // norm_out (the kernel supports it; FFN's variant always does).
            half* lora_no = (lora_ && lora_->has_qkv(layer)) ? static_cast<half*>(no.data) : nullptr;
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                  static_cast<const half*>(ly.attn_norm.data), q8, qscratch_.d8_buf,
                                  lora_no, K, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            dispatch_gemv_qkv_fused(ly.wq.qtype, ly.wq.data, ly.wk.data, ly.wv.data, q8, qscratch_.d8_buf,
                                    static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                                    static_cast<half*>(vv.data), q_rows, k_rows, v_rows, K, stream);
        } else {
            // Separate RMSNorm + dispatch.
            // Gemma-4 FP32 accum path: read FP32 residual directly to avoid the
            // FP16 round-trip that drops ~1-2% precision per layer and causes
            // the last-token hidden state to drift by sign-flip at L29.
            if (using_fp32_accum && prof.is_gemma4) {
                Tensor fp32_h = view_tokens(fp32_hidden_, n);
                rmsnorm_fp32_to_fp16(fp32_h, ly.attn_norm, no, eps, stream, norm_w_off_);
            } else {
                rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);
            }

            // FP8 prefill path: quantize norm_out→FP8 once, 3 separate FP8 GEMMs
            // Use handle tier checks instead of wcache_ probes.
            const WeightHandle* fp8_hwq = (ly.wq_id != kInvalidTensorID) ? &registry_.handle(ly.wq_id)
                                                                         : nullptr;
            const WeightHandle* fp8_hwk = (ly.wk_id != kInvalidTensorID) ? &registry_.handle(ly.wk_id)
                                                                         : nullptr;
            const WeightHandle* fp8_hwv = (ly.wv_id != kInvalidTensorID) ? &registry_.handle(ly.wv_id)
                                                                         : nullptr;
            bool fp8_qkv_available = (fp8_hwq && fp8_hwq->primary_tier == StorageTier::FP8 && fp8_hwk &&
                                      fp8_hwk->primary_tier == StorageTier::FP8 && fp8_hwv &&
                                      fp8_hwv->primary_tier == StorageTier::FP8);
            if (n > 1 && !state.force_fp16_gemm && fp8_qkv_available && qscratch_.fp8_act != nullptr &&
                qscratch_.d_act_scale != nullptr) {
                Tensor fp8_no(qscratch_.fp8_act, QType::FP8_E4M3, no.ndim, no.shape, true);
                quantize_fp16_to_fp8_e4m3(no, fp8_no, qscratch_.d_act_scale, stream,
                                          qscratch_.d_fp8_block_maxes, qscratch_.d_fp8_absmax,
                                          qscratch_.fp8_max_grid);
                // Reconstruct FP8 weight tensors from handle payloads.
                auto make_fp8_tensor = [](const WeightHandle* hw) {
                    int64_t wshape[2] = {hw->shape[0], hw->shape[1]};
                    return Tensor(hw->payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
                };
                Tensor fp8_tq = make_fp8_tensor(fp8_hwq);
                Tensor fp8_tk = make_fp8_tensor(fp8_hwk);
                Tensor fp8_tv = make_fp8_tensor(fp8_hwv);
                gemm_cublaslt(fp8_no, fp8_tq, q_target, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwq->payload.fp8.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_tk, kk, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwk->payload.fp8.d_scale, stream);
                gemm_cublaslt(fp8_no, fp8_tv, vv, 1.0f, 0.0f, qscratch_.d_act_scale,
                              fp8_hwv->payload.fp8.d_scale, stream);
            } else {
                // Try fused K+V path: single strided batched GEMM for both
                // projections. Read via WeightRegistry handle — the wcache_
                // map is no longer the lookup mechanism (it remains the
                // storage owner; cleanup happens via wcache_.clear()).
                // Gemma 4 per-layer shapes break strided-batched K+V layout assumptions.
                const Tensor* fused_kv = nullptr;
                Tensor fused_from_handle;
                if (ly.fused_kv_id != kInvalidTensorID) {
                    const auto& h = registry_.handle(ly.fused_kv_id);
                    if (h.payload.fp16.data) {
                        fused_from_handle = Tensor(h.payload.fp16.data, QType::F16, 2, h.shape, true);
                        fused_kv = &fused_from_handle;
                    }
                }
                if (n > 1 && fused_kv && !per_layer_shapes) {
                    // Q: still separate (different output dim with GQA)
                    gemm_via_handle_(ly.wq_id, no, q_target, ctx);
                    // K+V: one batched cuBLAS call
                    gemm_kv_batched(no, *fused_kv, kk, vv, stream);
                } else {
                    gemm_via_handle_(ly.wq_id, no, q_target, ctx);
                    gemm_via_handle_(ly.wk_id, no, kk, ctx);
                    if (ly.wv.data != nullptr) {
                        gemm_via_handle_(ly.wv_id, no, vv, ctx);
                    }
                    // else: K=V sharing path — vv populated below from kk.
                }
            }
        }

        // Apply Q/K/V biases if present (Qwen2) — fused 3-way for 1 launch
        add_bias_3way(qv, ly.q_bias, kk, ly.k_bias, vv, ly.v_bias, stream);

        // LoRA deltas on the raw projection outputs (pre QK-norm / pre RoPE —
        // matches HF PEFT semantics of wrapping the Linear). Every QKV arm
        // above materializes the normed input in `no` (the fused dp4a arm
        // side-writes it when an adapter is active).
        if (lora_) {
            if (const LoraWeights* w = lora_->get(layer, LoraProj::Q))
                lora_delta_(*w, no.data, qv.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::K))
                lora_delta_(*w, no.data, kk.data, n, stream);
            if (const LoraWeights* w = lora_->get(layer, LoraProj::V))
                lora_delta_(*w, no.data, vv.data, n, stream);
        }
    }

    // Gemma 4: K=V sharing for global attention layers (wv == null).
    // These layers have no V projection — V is aliased from K. Copy K→V here
    // so all downstream code (QK-norm, V-norm, KV-write, attention) sees a
    // valid V tensor.
    if (prof.is_gemma4 && ly.wv.data == nullptr && kk.data != nullptr && vv.data != nullptr) {
        size_t kv_bytes = static_cast<size_t>(n) * nkv * hd * dtype_size(kk.qtype);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(vv.data, kk.data, kv_bytes, cudaMemcpyDeviceToDevice, stream));
    }

    // V-normalization (Gemma 4): per-head RMSNorm with NO learned weight.
    // Matches llama.cpp's `Vcur = ggml_rms_norm(Vcur, eps)` (gemma4-iswa.cpp:82).
    // Required for both K=V-shared global layers and standard SWA layers.
    if (prof.is_gemma4 && v_norm_ones_buf_ != nullptr) {
        int64_t vflat_shape[4] = {static_cast<int64_t>(n) * nkv, hd, 0, 0};
        Tensor v_flat(vv.data, vv.qtype, 2, vflat_shape, true);
        int64_t ones_shape[4] = {hd, 0, 0, 0};
        Tensor ones_w(v_norm_ones_buf_, QType::F16, 1, ones_shape, true);
        rmsnorm(v_flat, ones_w, v_flat, eps, stream, 0.0f);
    }

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step1_after_qkv_q", qv, stream);
        debug_tensor_stats("L0_step1_after_qkv_k", kk, stream);
    }


    // Per-layer RoPE theta and sliding window (Gemma-3: alternating local/global layers)
    float layer_rope_theta = cfg.rope_theta;
    float layer_rope_freq_scale = cfg.rope_freq_scale;
    int layer_sliding_window = cfg.sliding_window;
    // StreamingLLM: only meaningful when this layer also has a sliding window
    // (otherwise full attention covers the full context anyway). Resolved
    // again per-layer below in case Gemma-3 disables SWA on this layer.
    int layer_n_sinks = streaming_n_sinks_;
    if (prof.attn_variant == AttnVariant::GEMMA4_SWA) {
        // Gemma 4: per-layer SWA pattern stored in cfg.swa_layers (1=SWA, 0=global).
        bool is_swa = (layer < (int)cfg.swa_layers.size() && cfg.swa_layers[layer]);
        if (is_swa) {
            layer_rope_theta = (cfg.rope_theta_swa > 0.0f) ? cfg.rope_theta_swa : cfg.rope_local_theta;
            layer_rope_freq_scale = 1.0f;
            // layer_sliding_window stays at cfg.sliding_window (1024)
        } else {
            // Global layer: full attention, model rope_theta, with freq scaling
            layer_sliding_window = 0;
        }
    } else if (prof.attn_variant == AttnVariant::GPTOSS_SWA) {
        // gpt-oss (#547): even layers sliding_attention (window 128), odd
        // layers full attention. Same RoPE (YaRN) on both layer types —
        // only the window toggles.
        bool is_swa = (layer < (int)cfg.swa_layers.size() && cfg.swa_layers[layer]);
        if (!is_swa)
            layer_sliding_window = 0;
    } else if (cfg.sliding_window_pattern > 0) {
        bool is_global = (layer % cfg.sliding_window_pattern) == (cfg.sliding_window_pattern - 1);
        if (is_global) {
            // Global layer: full attention, model-level rope_theta, with freq scaling
            layer_sliding_window = 0;
        } else {
            // Local layer: sliding window, local rope_theta, no freq scaling
            if (cfg.rope_local_theta > 0.0f)
                layer_rope_theta = cfg.rope_local_theta;
            layer_rope_freq_scale = 1.0f;  // no scaling for local layers
        }
    }
    // Apply caller-provided streaming window override and gate sinks on SWA-only layers.
    if (streaming_window_ > 0)
        layer_sliding_window = streaming_window_;
    if (layer_sliding_window <= 0)
        layer_n_sinks = 0;

    // Select LongRoPE frequency table based on context length (nullptr if not longrope)
    const float* longrope_freqs = nullptr;
    if (longrope_short_freqs_) {
        longrope_freqs = (state.max_context_len <= longrope_orig_max_pos_) ? longrope_short_freqs_
                                                                           : longrope_long_freqs_;
    }
    // Gemma 4: per-layer rope_freqs (pre-computed effective frequencies for
    // global layers, see gguf_loader.cpp:1221). llama.cpp's gemma4-iswa.cpp:55-59
    // passes these as freq_factors to ggml_rope_ext on full_attention layers
    // (n_rot=hd, with most pairs effectively zeroed by 1e30 divisors). This
    // matches the proportional-rope schema (ccss000000000000) the converter
    // emits via partial_rotary_factor=0.25.
    if (prof.attn_variant == AttnVariant::GEMMA4_SWA) {
        bool layer_is_swa = (layer < (int)cfg.swa_layers.size() && cfg.swa_layers[layer]);
        if (!layer_is_swa && ly.rope_freqs.data && ly.rope_freqs.on_device) {
            longrope_freqs = static_cast<const float*>(ly.rope_freqs.data);
        }
    }

    // Attention output gate (fused Q + gate projection). Two known layouts:
    //   (a) Per-head interleaved: [Q_h0(hd), Gate_h0(hd), Q_h1(hd), Gate_h1(hd), ...]
    //       — original Qwen 3.5 layout imp was built for.
    //   (b) Feature-dim concat:   [Q_all(nh*hd) | Gate_all(nh*hd)]
    //       — Qwen 3.6 / Qwen3-Next layout used by llama.cpp `qwen3next.cpp`.
    // Select via `IMP_ATTN_GATE_CONCAT=1` — default stays on interleaved for
    // backwards compat with Qwen 3.5 GDN models. Planned: auto-detect via
    // arch or config, once Qwen 3.6 passes an E2E test.
    Tensor attn_gate_buf;
    if (has_attn_output_gate) {
        size_t es_q = dtype_size(compute_dtype_);
        int64_t gate_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(q_actual_dim)};
        attn_gate_buf = Tensor(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);

        const bool use_concat = runtime_config().attention.gate_concat;
        if (use_concat) {
            // Feature-dim concat: Q = src[:, :q_actual_dim]; gate = src[:, q_actual_dim:]
            // One 2D copy each, width = q_actual_dim bytes per row.
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(qv.data,
                                                 static_cast<size_t>(q_actual_dim) * es_q,  // dst pitch
                                                 q_target.data,
                                                 static_cast<size_t>(q_out_dim) * es_q,  // src pitch
                                                 static_cast<size_t>(q_actual_dim) * es_q, n,
                                                 cudaMemcpyDeviceToDevice, stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(attn_gate_buf.data, static_cast<size_t>(q_actual_dim) * es_q,
                                                 static_cast<char*>(q_target.data) +
                                                     static_cast<size_t>(q_actual_dim) * es_q,
                                                 static_cast<size_t>(q_out_dim) * es_q,
                                                 static_cast<size_t>(q_actual_dim) * es_q, n,
                                                 cudaMemcpyDeviceToDevice, stream));
        } else {
            // Per-head interleaved: Qwen 3.5 layout.
            // Q[t, h*hd:(h+1)*hd] = src[t, h*2*hd : h*2*hd+hd]
            // Gate[t, h*hd:(h+1)*hd] = src[t, h*2*hd+hd : (h+1)*2*hd]
            // Replaced nh × 2 cudaMemcpy2DAsync loop with one fused kernel
            // (~656 D2D copies/decode-step → 1 launch on Qwen3.5 GDN W2).
            attn_gate_split_interleaved(q_target.data, qv.data, attn_gate_buf.data, n, nh, hd, q_out_dim,
                                        static_cast<int>(es_q), stream);
        }
    }

    // 4+5+6. QK-norm + RoPE: fused into single kernel for decode (n=1)
    //    For prefill or models without QK-norm, use separate kernels.
    //    For decode with FP16 cache: fuse K-RoPE into KV write (saves 1 launch).
    bool rope_k_deferred = false;  // true when K-RoPE will be fused into KV write
    {
        bool has_qk_norm = (ly.attn_q_norm.data != nullptr && ly.attn_k_norm.data != nullptr);
        // Determine if we can fuse K-RoPE into KV cache write.
        // YaRN models (#547, gpt-oss): the fused kernels
        // (rope_q_only_fp16_kernel / write_kv_cache_rope_fused_kernel) only
        // know theta + 1/scale — no ramp blending, no attn_factor. Prefill
        // K would be YaRN-correct but every decode-written K plain-RoPE →
        // degeneration from token 2. Keep YaRN on the separate full path.
        bool can_fuse_rope_kv = (!state.is_prefill && n == 1 && qv.qtype == QType::F16 && state.kv_cache &&
                                 state.kv_cache->qtype() == QType::F16 && prof.attn_variant != AttnVariant::NOPE &&
                                 cfg.yarn_ext_factor <= 0.0f);
        // Per-layer rope_dim. Gemma 4: both SWA and global layers rotate the
        // full head_dim. Global layers' freq_factors (loaded into
        // longrope_freqs above) zero out most pairs to realize the
        // partial-rotary schedule from the GGUF (ccss000000000000).
        int fused_rope_dim = cfg.rope_dim;
        if (prof.is_gemma4) {
            fused_rope_dim = hd;
        } else if (fused_rope_dim > hd || fused_rope_dim <= 0) {
            fused_rope_dim = hd;
        }
        const bool no_qknorm_fused = runtime_config().attention.no_qknorm_fused;
        if (has_qk_norm && n == 1 && qv.qtype == QType::F16 && !no_qknorm_fused && prof.attn_variant != AttnVariant::NOPE) {
            // Fused: QK-norm + RoPE in one kernel launch (decode only, n=1).
            // Keeps norm intermediate values in FP32 shared memory.
            qknorm_rope_fused(static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                              static_cast<const half*>(ly.attn_q_norm.data),
                              static_cast<const half*>(ly.attn_k_norm.data), nh, nkv, hd, eps,
                              state.positions, layer_rope_theta, layer_rope_freq_scale, fused_rope_dim,
                              cfg.rope_neox, stream, norm_w_off_, cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                              cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, longrope_freqs);
        } else if (can_fuse_rope_kv && !has_qk_norm) {
            // Fused path: Q-only RoPE here, K-RoPE deferred to KV write
            const int effective_rope_dim = fused_rope_dim;
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            rope_q_only_fp16_kernel<<<dim3(1, nh), pairs, 0, stream>>>(static_cast<half*>(qv.data),
                                                                       state.positions, nh, hd,
                                                                       layer_rope_theta, inv_scaling, pairs,
                                                                       cfg.rope_neox, longrope_freqs);
            rope_k_deferred = true;
        } else {
            // Separate path: QK-norm (if present) + RoPE on both Q and K.
            //
            // Some architectures (Qwen3.5-27B-mxfp4) ship `attn_q_norm` /
            // `attn_k_norm` with a smaller dim than `head_dim` — the weight
            // is meant to be applied per (norm_dim)-sized chunk along the
            // head, so a 512-dim head with a 256-dim norm splits into two
            // 256-dim sub-heads sharing the same scale. Detect that by
            // looking at the norm weight's element count and reshape the
            // Q/K view accordingly. When norm_dim == hd (the common case)
            // this is a no-op.
            auto split_norm_dim = [hd](const Tensor& w) -> int {
                int wd = (w.data != nullptr) ? static_cast<int>(w.shape[0]) : hd;
                return (wd > 0 && wd < hd && hd % wd == 0) ? wd : hd;
            };
            if (ly.attn_q_norm.data != nullptr) {
                int q_norm_dim = split_norm_dim(ly.attn_q_norm);
                int64_t q_flat[2] = {static_cast<int64_t>(n) * nh * (hd / q_norm_dim),
                                     static_cast<int64_t>(q_norm_dim)};
                Tensor q_flat_view = qv.reshape(2, q_flat);
                rmsnorm(q_flat_view, ly.attn_q_norm, q_flat_view, eps, stream, norm_w_off_);
            }
            if (ly.attn_k_norm.data != nullptr) {
                int k_norm_dim = split_norm_dim(ly.attn_k_norm);
                int64_t k_flat[2] = {static_cast<int64_t>(n) * nkv * (hd / k_norm_dim),
                                     static_cast<int64_t>(k_norm_dim)};
                Tensor k_flat_view = kk.reshape(2, k_flat);
                rmsnorm(k_flat_view, ly.attn_k_norm, k_flat_view, eps, stream, norm_w_off_);
            }
            int64_t q4r[4] = {1, n, nh, hd};
            int64_t k4r[4] = {1, n, nkv, hd};
            Tensor q4r_t = qv.reshape(4, q4r);
            Tensor k4r_t = kk.reshape(4, k4r);
            // Per-layer rope_dim. Gemma 4: full hd for both SWA and global;
            // global layers' freq_factors (longrope_freqs) realize the
            // partial-rotary schedule from the GGUF.
            int layer_rope_dim = cfg.rope_dim;
            if (prof.is_gemma4) {
                layer_rope_dim = hd;
            } else if (layer_rope_dim > hd || layer_rope_dim <= 0) {
                layer_rope_dim = hd;  // safety clamp
            }
            // NoPE attention (Nemotron-H): position lives in the Mamba layers,
            // rotating Q/K here scrambles positional binding (bag-of-words
            // prompts). QK-norm above still applies; only the rotation is skipped.
            if (prof.attn_variant != AttnVariant::NOPE) {
                rope_forward(q4r_t, k4r_t, state.positions, hd, layer_rope_theta, layer_rope_freq_scale,
                             layer_rope_dim, cfg.rope_neox, cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                             cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, stream, longrope_freqs);
            }
        }
    }

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step2_after_rope_q", qv, stream);
        debug_tensor_stats("L0_step2_after_rope_k", kk, stream);
    }

    // 7. Attention scale.
    //   Standard archs: 1/sqrt(head_dim).
    //   Gemma 4: 1.0 (confirmed by llama.cpp print_info: f_attn_scale = 1.0.
    //                 Q-norm and K-norm absorb the per-element scaling).
    float scale = (prof.is_gemma4) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(hd)));

    // gpt-oss learned attention sinks (#547): per-head logits acting as a
    // virtual extra softmax column. Only the cuBLAS prefill softmax and the
    // FP16 paged decode kernels understand them — prefill routing below
    // forces the cuBLAS path whenever sinks are present.
    const void* attn_sinks = (prof.is_gpt_oss) ? ly.attn_sinks.data : nullptr;

    if (state.is_prefill) {
        // Chunked prefill: when prefill_offset > 0, queries from this chunk must
        // attend to past chunks K/V already in the paged cache. Gather past
        // [0, prefill_offset) KV → contiguous, append current chunk, then run
        // rectangular attention_cublas_prefill with q_offset.
        //
        // Note: cudaMallocAsync per layer here violates CLAUDE.md "No cudaMalloc in hot
        // loops". Acknowledged exception — chunked prefill is excluded from CUDA-graph
        // capture (graphs only capture decode), so the alloc is amortised in the
        // memory pool and runs once per chunk per layer, not per token.
        const int q_offset = state.prefill_offset;
        if (q_offset > 0) {
            KVCache* cache = state.kv_cache;
            QType kvt = cache->qtype();
            // Defense-in-depth: engine resolves out-of-scope models to chunk_size=0,
            // so this code only runs for FP16 / FP8 / NVFP4 / MXFP4_KV KV.
            const bool kvt_ok = (kvt == QType::F16 || kvt == QType::FP8_E4M3 || kvt == QType::NVFP4 ||
                                 kvt == QType::MXFP4_KV || kvt == QType::INT4);
            // sliding_active and attn_shapes_vary are now both supported:
            //   - cuBLAS softmax accepts a sliding_window argument (PR feat(attn): sw),
            //   - the rectangular cuBLAS path dispatches per-layer with nh/nkv/hd.
            // KV dtype remains the only hard gate (the gather kernels are
            // wired for FP16 / FP8 / NVFP4 only).
            if (!kvt_ok) {
                IMP_LOG_ERROR(
                    "chunked_prefill: unsupported KV dtype %d at L%d — engine should have prevented this",
                    (int)kvt, layer);
                std::abort();
            }

            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            int kv_bs = cache->block_size();
            int ctx_len = q_offset + n;
            // attn_scores_ is sized [nh, s_cap, s_cap] (square). Chunked use stores
            // an [nh, n, ctx_len] FP16 matrix (or FP32 = 2× when use_fp32_s). The
            // capacity constraint is `n * ctx_len <= s_cap²`, NOT `ctx_len <= s_cap`
            // (which was the previous guard — overly strict, wrongly aborted at any
            // chunked step where the cumulative ctx_len crossed s_cap even though
            // the actual matrix size still fit). FP32 takes 2× — same gate as
            // attention_cublas_prefill::use_fp32_s; if FP32 would overflow, the
            // attention call falls back to FP16-S automatically. So we only need to
            // guard the FP16 footprint here.
            int s_cap = attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
            int64_t fp16_elems_needed = static_cast<int64_t>(n) * ctx_len;
            int64_t fp16_elems_avail = static_cast<int64_t>(s_cap) * s_cap;
            if (s_cap == 0 || fp16_elems_needed > fp16_elems_avail || n > s_cap) {
                IMP_LOG_ERROR(
                    "chunked_prefill: attn_scores_ capacity %d×%d=%lld too small for "
                    "n=%d × ctx_len=%d = %lld at L%d — engine should have prevented this",
                    s_cap, s_cap, (long long)fp16_elems_avail, n, ctx_len, (long long)fp16_elems_needed,
                    layer);
                std::abort();
            }
            size_t full_bytes = (size_t)ctx_len * nkv * hd * sizeof(half);

            half* k_full = nullptr;
            half* v_full = nullptr;
            cudaMallocAsync(&k_full, full_bytes, stream);
            cudaMallocAsync(&v_full, full_bytes, stream);

            // Gather past KV [0, q_offset) directly into k_full[0..q_offset], v_full[0..q_offset].
            if (kvt == QType::F16) {
                paged_kv_gather_fp16(k_full, static_cast<const half*>(cache->k_ptr(kv_layer, 0)),
                                     state.block_tables, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_fp16(v_full, static_cast<const half*>(cache->v_ptr(kv_layer, 0)),
                                     state.block_tables, q_offset, kv_bs, nkv, hd, stream);
            } else if (kvt == QType::FP8_E4M3) {
                float kv_scale = (!kv_scales_.empty() && kv_layer < (int)kv_scales_.size())
                                     ? kv_scales_[kv_layer]
                                     : 1.0f;
                paged_kv_gather_fp8_to_fp16(k_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
                                            state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_fp8_to_fp16(v_full,
                                            static_cast<const __nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
                                            state.block_tables, kv_scale, q_offset, kv_bs, nkv, hd, stream);
            } else if (kvt == QType::NVFP4) {
                paged_kv_gather_nvfp4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                              state.block_tables, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_nvfp4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                              static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                              state.block_tables, q_offset, kv_bs, nkv, hd, stream);
            } else if (kvt == QType::MXFP4_KV) {
                paged_kv_gather_mxfp4_kv_to_fp16(k_full,
                                                 static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                                 state.block_tables, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_mxfp4_kv_to_fp16(v_full,
                                                 static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                                 static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                                 state.block_tables, q_offset, kv_bs, nkv, hd, stream);
            } else {  // INT4 — symmetric 4-bit with per-head FP16 scale
                paged_kv_gather_int4_to_fp16(k_full, static_cast<const uint8_t*>(cache->k_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                             state.block_tables, q_offset, kv_bs, nkv, hd, stream);
                paged_kv_gather_int4_to_fp16(v_full, static_cast<const uint8_t*>(cache->v_ptr(kv_layer, 0)),
                                             static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                             state.block_tables, q_offset, kv_bs, nkv, hd, stream);
            }

            // Append current chunk's K/V at offset q_offset.
            cudaMemcpyAsync(k_full + (size_t)q_offset * nkv * hd, kk.data,
                            (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_full + (size_t)q_offset * nkv * hd, vv.data,
                            (size_t)n * nkv * hd * sizeof(half), cudaMemcpyDeviceToDevice, stream);

            int64_t kv_full_shape[2] = {(int64_t)ctx_len, (int64_t)(nkv * hd)};
            Tensor k_full_t(k_full, QType::F16, 2, kv_full_shape, /*on_device=*/true);
            Tensor v_full_t(v_full, QType::F16, 2, kv_full_shape, /*on_device=*/true);

            // Chunked prefill: choose FMHA or cuBLAS.
            //
            // REVERTED (#493 routed ALL hd=128 prefill into FA2 "regardless of
            // the threshold" for +7-25% pp512): both fp8-quantizing FMHA
            // kernels (FA2 and the fp8 FMHA) carry per-layer e4m3 score noise
            // that compounds across layers into prompt-blind/degenerate output
            // — every hd=128 model failed the degen suite at short prompts
            // (Qwen3-8B answered "What is 17 + 25?" with "2 5 2 5 2 5...").
            // The FP16 cuBLAS path is the accuracy reference; FMHA stays
            // gated behind the S-matrix-capacity threshold where it has
            // history in production. fp8-attention quality above the
            // threshold is tracked separately (see issue #511).
            const int fmha_threshold = runtime_config().attention.fmha_prefill_threshold;
            const bool chunked_use_fmha = !per_layer_shapes &&  // Gemma-4 hd=512 stays cuBLAS
                                          (fmha_threshold > 0 && ctx_len >= fmha_threshold);

            // Chunked attention routing (#548): prefer the FP16-QK FA2 kernel
            // at EVERY ctx_len, not only below the threshold. It is O(n)
            // memory (no S-matrix) like the fp8 family but carries no e4m3
            // score noise — the long-context chunked path through the
            // e4m3 FMHA family drifted teacher-forced NLL by up to ~25%
            // (ChunkedPrefillTest.LongContext_Chunk_Invariance is the gate).
            // The fp8 family stays as fallback for configs f16-QK declines
            // (hd != 128, fa2_fp16qk=never), and cuBLAS below the threshold.
            if (!per_layer_shapes && !attn_sinks &&
                try_fa2_fp16qk_prefill(runtime_config(), qv, k_full_t, v_full_t, ao, n, ctx_len, nh,
                                       nkv, hd, scale, layer_sliding_window, cfg.attn_logit_softcap,
                                       q_offset, stream)) {
                // chunked prefill: FP16-QK FA2 (no S-matrix, no e4m3 noise)
            } else if (chunked_use_fmha && !attn_sinks) {
                // FMHA: no S-matrix needed, O(n) memory. Reshape to 4D for dispatch.
                int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                int64_t kv4s[4] = {1, (int64_t)ctx_len, (int64_t)nkv, (int64_t)hd};
                int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
                Tensor q4 = qv.reshape(4, q4s);
                Tensor k4 = k_full_t.reshape(4, kv4s);
                Tensor v4 = v_full_t.reshape(4, kv4s);
                Tensor o4 = ao.reshape(4, o4s);
                attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                           cfg.attn_logit_softcap, stream, runtime_config(), q_offset);
            } else {
                // cuBLAS: needs S-matrix for [n × ctx_len]
                attention_cublas_prefill(qv, k_full_t, v_full_t, ao, attn_scores_, nh, nkv, hd, scale,
                                         /*causal=*/true, cfg.attn_logit_softcap, q_offset, stream,
                                         layer_sliding_window, attn_sinks);
            }

            cudaFreeAsync(k_full, stream);
            cudaFreeAsync(v_full, stream);

            // Persist current chunk's K/V (same as non-chunked path)
            write_kv_cache(layer, state, stream);
            goto after_attention;
        }

        // Prefill dispatch (post-Phase-2 + Phase-5 Track D):
        // attn_sinks: only the cuBLAS softmax understands learned sinks.
        const bool force_cublas_attn = per_layer_shapes || attn_sinks != nullptr;
        const bool s_matrix_fits = attn_scores_buf_ != nullptr &&
                                   n <= static_cast<int>(attn_scores_.shape[1]);
        // FMHA only above the S-matrix threshold — see the chunked path above
        // for why the #493 prefer-FA2-at-every-length override was reverted.
        const bool prefer_fmha = !force_cublas_attn &&
                                 (n >= runtime_config().attention.fmha_prefill_threshold);

        // NOTE (#566): sliding-window prefill used to be routed AWAY from the
        // cuBLAS path here (`non_gemma4_sliding`) into the WMMA fallback
        // chain — a relic from before attention_cublas_prefill grew its
        // sliding_window softmax mask. The WMMA chain's hd=256 + window
        // combination computes catastrophically wrong attention (gemma-3-12b
        // at n>1024: teacher-forced PPL 42 vs llama.cpp 1.0; long-prompt
        // recall answered '77' instead of '477'). The cuBLAS masked softmax
        // is the correctness reference and handles the window — keep SWA
        // prefill here whenever the S-matrix fits.
        if (s_matrix_fits && !prefer_fmha) {
            // Below the FMHA threshold: prefer the FP16-QK FA2 kernel over the
            // materialized cuBLAS path (same f16/f32 numerics, O(n) memory).
            if (!force_cublas_attn &&
                try_fa2_fp16qk_prefill(runtime_config(), qv, kk, vv, ao, n, n, nh, nkv, hd, scale,
                                       layer_sliding_window, cfg.attn_logit_softcap, /*q_offset=*/0,
                                       stream)) {
                // handled by FA2 f16
            } else {
                attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale,
                                         /*causal=*/true, cfg.attn_logit_softcap,
                                         /*q_offset=*/0, stream, layer_sliding_window, attn_sinks);
            }
        } else {
            if (attn_sinks) {
                static bool warned_sinks_fmha = false;
                if (!warned_sinks_fmha) {
                    warned_sinks_fmha = true;
                    IMP_LOG_WARN("attention sinks: S-matrix too small (n=%d) — FMHA fallback IGNORES "
                                 "sinks; output will be wrong. Lower the prefill chunk size.",
                                 n);
                }
            }
            // FMHA fallback: tiled O(n) memory chain.
            int64_t q4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            int64_t kv4s[4] = {1, (int64_t)n, (int64_t)nkv, (int64_t)hd};
            int64_t o4s[4] = {1, (int64_t)n, (int64_t)nh, (int64_t)hd};
            Tensor q4 = qv.reshape(4, q4s);
            Tensor k4 = kk.reshape(4, kv4s);
            Tensor v4 = vv.reshape(4, kv4s);
            Tensor o4 = ao.reshape(4, o4s);
            attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, layer_sliding_window,
                                       cfg.attn_logit_softcap, stream, runtime_config(), /*q_offset=*/0);
        }

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream);
    } else {
        // Decode: write new token's K/V to cache first
        if (rope_k_deferred) {
            // Fused: apply RoPE to K during KV cache write (saves 1 kernel launch)
            int kv_layer = get_kv_layer(kv_layer_map_, layer);
            KVCache* cache = state.kv_cache;
            const int kv_block_size_d = cache->block_size();
            int row_elems = nkv * hd;
            int block_stride = kv_block_size_d * row_elems;
            int threads = std::min(row_elems, 256);
            // Per-layer rope_dim (same as prefill rope path above): Gemma 4
            // uses full hd; longrope_freqs encodes the partial-rotary schedule.
            int effective_rope_dim;
            if (prof.is_gemma4) {
                effective_rope_dim = hd;
            } else {
                effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
                if (effective_rope_dim > hd)
                    effective_rope_dim = hd;
            }
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            Tensor kv_view = view_tokens(k_, n);
            Tensor vv_view = view_tokens(v_, n);
            dim3 fused_grid(n, 2);
            write_kv_cache_rope_fused_kernel<<<fused_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv_view.data), static_cast<const half*>(vv_view.data),
                state.positions, state.block_tables, static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)), block_stride, row_elems, kv_block_size_d, n,
                state.max_blocks_per_seq, state.n_sequences, nkv, hd, layer_rope_theta, inv_scaling, pairs,
                cfg.rope_neox, longrope_freqs);
        } else {
            write_kv_cache(layer, state, stream);
        }

        // DEBUG: force cuBLAS attention for decode to isolate paged attention bugs.
        // When enabled, uses the same materialized QK^T path as prefill.
        const bool force_cublas_decode = runtime_config().attention.force_cublas_decode;
        if (force_cublas_decode && n == 1 && attn_scores_buf_) {
            // Reconstruct K/V from cache for this position
            KVCache* cache_dbg = state.kv_cache;
            int kv_layer_dbg = get_kv_layer(kv_layer_map_, layer);
            int ctx_len = 0;
            cudaMemcpy(&ctx_len, state.context_lens, sizeof(int), cudaMemcpyDeviceToHost);
            // Allocate temp K/V for all context tokens
            int kv_elems = ctx_len * nkv * hd;
            half *k_flat = nullptr, *v_flat = nullptr;
            cudaMalloc(&k_flat, kv_elems * sizeof(half));
            cudaMalloc(&v_flat, kv_elems * sizeof(half));
            // Copy from paged KV cache to contiguous buffer
            int kv_bs = cache_dbg->block_size();
            int32_t h_block_table[1024];
            int n_blocks = (ctx_len + kv_bs - 1) / kv_bs;
            cudaMemcpy(h_block_table, state.block_tables, n_blocks * sizeof(int32_t), cudaMemcpyDeviceToHost);
            for (int b = 0; b < n_blocks; b++) {
                int block_id = h_block_table[b];
                int toks_in_block = std::min(kv_bs, ctx_len - b * kv_bs);
                size_t row_bytes = nkv * hd * sizeof(half);
                half* k_src = static_cast<half*>(cache_dbg->k_ptr(kv_layer_dbg, block_id));
                half* v_src = static_cast<half*>(cache_dbg->v_ptr(kv_layer_dbg, block_id));
                cudaMemcpy(k_flat + b * kv_bs * nkv * hd, k_src, toks_in_block * row_bytes,
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(v_flat + b * kv_bs * nkv * hd, v_src, toks_in_block * row_bytes,
                           cudaMemcpyDeviceToDevice);
            }
            // Reshape for cuBLAS attention: Q[1,nh,hd], K[ctx_len,nkv,hd], V[ctx_len,nkv,hd]
            int64_t k_shape[2] = {ctx_len, nkv * hd};
            int64_t v_shape[2] = {ctx_len, nkv * hd};
            Tensor k_cont(k_flat, QType::F16, 2, k_shape, true);
            Tensor v_cont(v_flat, QType::F16, 2, v_shape, true);
            // n=1 cuBLAS attention. causal=true + q_offset=ctx_len-1 makes the
            // single query row see the full context AND keeps the sliding-window
            // mask correct (the old causal=false/q_offset=0 form broke SWA
            // layers: abs_row=0 never triggered the window). Sinks pass through
            // so this debug arm stays a faithful reference for gpt-oss (#547).
            {
                int64_t s_shape[3] = {(int64_t)nh, 1, (int64_t)ctx_len};
                half* s_buf = nullptr;
                cudaMalloc(&s_buf, (size_t)nh * ctx_len * sizeof(half));
                Tensor s_view(s_buf, QType::F16, 3, s_shape, true);
                attention_cublas_prefill(qv, k_cont, v_cont, ao, s_view, nh, nkv, hd, scale, /*causal=*/true,
                                         cfg.attn_logit_softcap, /*q_offset=*/ctx_len - 1, stream,
                                         layer_sliding_window, attn_sinks);
                cudaFree(s_buf);
            }
            cudaFree(k_flat);
            cudaFree(v_flat);
            goto after_attention;
        }

        // Paged attention: Q shape depends on batch size
        int n_seq = state.n_sequences;
        // For decode, n_tokens == n_sequences (one token per seq)
        int64_t qd[4] = {n_seq, 1, nh, hd};
        int64_t od[4] = {n_seq, 1, nh, hd};
        Tensor q4 = qv.reshape(4, qd);
        Tensor o4 = ao.reshape(4, od);

        KVCache* cache = state.kv_cache;
        const int kv_bs = cache->block_size();
        int total_blk = cache->total_blocks();
        QType cache_dtype = cache->qtype();
        int64_t cs[4] = {static_cast<int64_t>(total_blk), static_cast<int64_t>(kv_bs),
                         static_cast<int64_t>(nkv), static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = get_kv_layer(kv_layer_map_, layer);
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        // L2 persistence hint: keep this layer's KV cache in L2 during attention.
        // RTX 5090 has 96 MB L2 — enough for ~3K tokens of KV at FP8.
        set_l2_persist_kv(stream, k_c.data, k_c.nbytes() + v_c.nbytes());

        if (cache_dtype == QType::INT4) {
            // INT4 paged attention with per-head scales and INT4 unpack (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int4(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        state.block_tables, state.context_lens, kv_bs, scale,
                                        state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                        stream, state.max_blocks_per_seq);
        } else if (cache_dtype == QType::NVFP4) {
            // NVFP4 paged attention: packed FP4 + UE4M3 per-group_of_16 scales (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            // BitDecoding TC dispatch opt-in: kv_cache.bitdecoding_qk (legacy
            // IMP_USE_BITDECODING_QK=1) routes to the WMMA-Q.K variant; default
            // keeps the scalar-FFMA path unchanged.
            const bool use_bitdecoding_tc = runtime_config().kv_cache.bitdecoding_qk;
            if (use_bitdecoding_tc) {
                // QW6 from review/phase5_synthesis.md §2.1: gate the entire
                // residual-arg marshalling (8+ trailing args) behind a single
                // null-check. Default path (residual_on=false) uses the
                // function's default arguments — no explicit nullptr/0 marshalling
                // at the call site, no dead per-layer state lookups.
                const bool residual_on = state.kv_manager != nullptr && state.kv_manager->residual_enabled();
                const uint8_t* k_scales = static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0));
                const uint8_t* v_scales = static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0));
                if (!residual_on) {
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, state.block_tables,
                                                    state.context_lens, kv_bs, scale, state.max_context_len,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    state.max_blocks_per_seq);
                } else {
                    // Phase 3b residual read. Two activation paths:
                    //   (multi-seq) state.d_residual_seq_slots != nullptr: kernel
                    //     reads per-batch metadata from the device arrays. Used by
                    //     batched decode.
                    //   (single-seq legacy) state.kv_seq_id >= 0: kernel uses the
                    //     scalar form. Used by single-seq decode that hasn't been
                    //     migrated to the array form (e.g. early-init smoke).
                    const half* k_res = nullptr;
                    const half* v_res = nullptr;
                    int res_count = 0;
                    int res_n = state.kv_manager->residual_n_tokens();
                    int res_widx = 0;
                    const half* k_res_base = nullptr;
                    const half* v_res_base = nullptr;
                    int res_seq_stride_elems = 0;
                    if (state.d_residual_seq_slots != nullptr) {
                        // Multi-seq array form
                        k_res_base = static_cast<const half*>(
                            state.kv_manager->residual_k_layer_base(kv_layer));
                        v_res_base = static_cast<const half*>(
                            state.kv_manager->residual_v_layer_base(kv_layer));
                        res_seq_stride_elems = static_cast<int>(
                            state.kv_manager->residual_seq_stride_bytes() / sizeof(__half));
                    } else if (state.n_sequences == 1 && state.kv_seq_id >= 0) {
                        // Single-seq scalar form
                        k_res = static_cast<const half*>(
                            state.kv_manager->residual_k_ptr(state.kv_seq_id, kv_layer));
                        v_res = static_cast<const half*>(
                            state.kv_manager->residual_v_ptr(state.kv_seq_id, kv_layer));
                        auto rs = state.kv_manager->residual_state(state.kv_seq_id);
                        res_count = rs.fill_count;
                        res_widx = rs.write_idx;
                    }
                    paged_attention_decode_nvfp4_tc(q4, k_c, v_c, o4, k_scales, v_scales, state.block_tables,
                                                    state.context_lens, kv_bs, scale, state.max_context_len,
                                                    layer_sliding_window, cfg.attn_logit_softcap, stream,
                                                    state.max_blocks_per_seq, 0, k_res, v_res, res_count,
                                                    res_n, res_widx, k_res_base, v_res_base,
                                                    res_seq_stride_elems, state.d_residual_seq_slots,
                                                    state.d_residual_counts, state.d_residual_write_idxes,
                                                    state.kv_manager->d_residual_fc_ptr(),
                                                    state.kv_manager->d_residual_widx_ptr());
                }
            } else {
                paged_attention_decode_nvfp4(q4, k_c, v_c, o4,
                                             static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                             static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                             state.block_tables, state.context_lens, kv_bs, scale,
                                             state.max_context_len, layer_sliding_window,
                                             cfg.attn_logit_softcap, stream, state.max_blocks_per_seq);
            }
        } else if (cache_dtype == QType::MXFP4_KV) {
            // MXFP4-KV paged attention: same kernel as NVFP4 but with UE8M0 scale decode.
            // BitDecoding TC path not supported for MXFP4_KV in Slice 2 — always scalar.
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_mxfp4_kv(q4, k_c, v_c, o4,
                                            static_cast<const uint8_t*>(cache->k_scale_ptr(kv_layer, 0)),
                                            static_cast<const uint8_t*>(cache->v_scale_ptr(kv_layer, 0)),
                                            state.block_tables, state.context_lens, kv_bs, scale,
                                            state.max_context_len, layer_sliding_window,
                                            cfg.attn_logit_softcap, stream, state.max_blocks_per_seq);
        } else if (cache_dtype == QType::INT8) {
            // INT8 dp4a paged attention with per-head scales (Split-K enabled)
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_int8(q4, k_c, v_c, o4,
                                        static_cast<const half*>(cache->k_scale_ptr(kv_layer, 0)),
                                        static_cast<const half*>(cache->v_scale_ptr(kv_layer, 0)),
                                        state.block_tables, state.context_lens, kv_bs, scale,
                                        state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                        stream, state.max_blocks_per_seq);
        } else if (cache_dtype == QType::FP8_E4M3) {
            // FP8 paged attention with on-the-fly dequant (Split-K enabled)
            float kv_scale = (!kv_scales_.empty() && kv_layer < static_cast<int>(kv_scales_.size()))
                                 ? kv_scales_[kv_layer]
                                 : 1.0f;
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode_fp8(q4, k_c, v_c, o4, state.block_tables, state.context_lens, kv_bs, scale,
                                       kv_scale, state.max_context_len, layer_sliding_window,
                                       cfg.attn_logit_softcap, stream, state.max_blocks_per_seq);
        } else {
            paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
            paged_attention_decode(q4, k_c, v_c, o4, state.block_tables, state.context_lens, kv_bs, scale,
                                   state.max_context_len, layer_sliding_window, cfg.attn_logit_softcap,
                                   stream, state.max_blocks_per_seq, layer_n_sinks, attn_sinks);
        }
        if (attn_sinks && cache_dtype != QType::F16) {
            static bool warned_sinks_kv = false;
            if (!warned_sinks_kv) {
                warned_sinks_kv = true;
                IMP_LOG_WARN("attention sinks: only the FP16 paged decode kernels apply sinks — "
                             "kv_cache.dtype=%d ignores them; use fp16 KV for this model.",
                             (int)cache_dtype);
            }
        }

        // Clear L2 persistence hint (weights loaded next need L2 space)
        clear_l2_persist(stream);
    }

after_attention:

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step3_after_paged_attn", ao, stream);
        debug_tensor_stats("L0_step3_h_before_oproj", h, stream);
    }

    // Qwen3.5 attention output gate: ao[i] *= sigmoid(gate[i])
    if (has_attn_output_gate) {
        sigmoid_mul(ao, attn_gate_buf, ao, stream);
    }

    // 8+9. O projection + residual connection.
    //    For decode (n=1) with dp4a: fuse residual add into GEMV, write directly
    //    to hidden buffer. When will_fuse_o_residual is set, we skipped the
    //    initial h→r memcpy and use h.data itself as the residual source.
    //    This is safe because h.data is only READ (never written) between the
    //    start of run_attention and this point.
    if (will_fuse_o_nvfp4) {
        // NVFP4 Wo + residual: attn_out (FP16) @ wo_nvfp4^T + residual → hidden
        // Source: secondary cache (Q8_0/Q6_K/Q5_K) is already a populated struct;
        // primary-tier handle needs reconstruction.
        NvFP4QuantResult wo_nvfp4;
        if (wo_nvfp4_secondary) {
            wo_nvfp4 = wcache_.nvfp4.at(ly.wo.data);
        } else {
            const WeightHandle& wo_h = registry_.handle(ly.wo_id);
            wo_nvfp4.packed_data = wo_h.payload.nvfp4.data;
            wo_nvfp4.micro_scales = wo_h.payload.nvfp4.block_scales;
            wo_nvfp4.tensor_scale = (wo_h.payload.nvfp4.tensor_scale != nullptr)
                                        ? *wo_h.payload.nvfp4.tensor_scale
                                        : 1.0f;
            wo_nvfp4.N = static_cast<int>(wo_h.shape[0]);
            wo_nvfp4.K = static_cast<int>(wo_h.shape[1]) * 2;  // packed → logical K
        }
        int M_o = wo_nvfp4.N;
        int K_o = wo_nvfp4.K;
        gemv_nvfp4_residual(wo_nvfp4, static_cast<const half*>(ao.data), static_cast<half*>(h.data),
                            static_cast<const half*>(h.data), M_o, K_o, stream);
    } else if (will_fuse_o_residual) {
        int K_o = static_cast<int>(ly.wo.shape[1]);
        int M_o = static_cast<int>(ly.wo.shape[0]);
        // Separate quant + K-parallel GEMV: higher warp occupancy than inline_quant.
        // quantize_fp16_to_q8_1 is a lightweight kernel (~2 us for d_model=3072).
        // The K-parallel GEMV achieves 48 warps/SM vs inline_quant's ~8 warps/SM.
        const half* attn_fp16 = static_cast<const half*>(ao.data);
        const half* residual_ptr = static_cast<const half*>(h.data);
        quantize_fp16_to_q8_1(attn_fp16, static_cast<block_q8_1*>(qscratch_.q8_1_buf), qscratch_.d8_buf, K_o,
                              stream);
        dispatch_gemv_residual(ly.wo.qtype, ly.wo.data, static_cast<block_q8_1*>(qscratch_.q8_1_buf),
                               qscratch_.d8_buf, static_cast<half*>(h.data), residual_ptr, M_o, K_o, stream);
    } else if (will_fuse_o_beta1 && !cur_force_fp16_ && wo_tier == StorageTier::FP8 &&
               qscratch_.fp8_act != nullptr && qscratch_.d_act_scale != nullptr) {
        // FP8 beta=1: hidden = fp8(attn_out) @ fp8(wo)^T + hidden
        const WeightHandle& wo_h = registry_.handle(ly.wo_id);
        int64_t wshape[2] = {wo_h.shape[0], wo_h.shape[1]};
        Tensor fp8_wo(wo_h.payload.fp8.data, QType::FP8_E4M3, 2, wshape, true);
        Tensor fp8_ao(qscratch_.fp8_act, QType::FP8_E4M3, ao.ndim, ao.shape, true);
        quantize_fp16_to_fp8_e4m3(ao, fp8_ao, qscratch_.d_act_scale, stream, qscratch_.d_fp8_block_maxes,
                                  qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid);
        gemm_cublaslt(fp8_ao, fp8_wo, h, 1.0f, 1.0f, qscratch_.d_act_scale, wo_h.payload.fp8.d_scale, stream);
    } else if (will_fuse_o_beta1 && wo_tier == StorageTier::FP16) {
        // Fused: hidden = attn_out @ wo^T + hidden (cuBLAS beta=1).
        // Safe: hidden is only READ (never written) between attn_norm and here.
        gemm_via_handle_(ly.wo_id, ao, h, ctx.with_beta(1.0f));
    } else if ((will_fuse_o_beta1 || will_fuse_o_dequant_beta1) && qscratch_.dequant != nullptr &&
               dequant_gpu_supported(ly.wo.qtype) &&
               !per_layer_shapes) {  // Gemma 4: workspace stride mismatch with narrow ao
        // Dequant beta=1: dequant weights on-the-fly, then FP16 GEMM + residual
        gemm_via_handle_(ly.wo_id, ao, h, ctx.with_beta(1.0f));
    } else {
        // Fallback: separate O-projection + optional post-norm + residual add.
        // Diagnostic: when IMP_GEMMA4_FP32_GEMM_OUT is set on Gemma-4, after the
        // FP16 GEMM, also produce an FP32 view (fp16_to_fp32). Then route the
        // post-attn-norm through the FP32-input variant. This keeps the proven
        // FP16 GEMM path while letting us validate the FP32-input rmsnorm
        // independently. Any precision win comes from the rmsnorm pre-cast
        // happening once in __half2float vs implicit casts inside the kernel.
        // IMP_GEMMA4_FP32_GEMM_OUT: keep attention output projection in FP32 to
        // preserve cuBLAS's internal FP32 accumulator precision through the
        // post-attention rmsnorm. Uses the cublasGemmEx FP16×FP16→FP32 path
        // (gemm.cu mixed-precision short-circuit). Skips the FP16-only mmvq
        // and dp4a fast paths via output.qtype==FP32 guards in dispatch.
        const bool fp32_attn_out = (prof.is_gemma4 && using_fp32_accum &&
                                    model_->config().overrides.gemma4.fp32_gemm_out);
        void* po_fp32_buf = nullptr;
        if (fp32_attn_out) {
            size_t bytes = static_cast<size_t>(n) * model_->config().d_model * sizeof(float);
            IMP_CUDA_CHECK_LOG(cudaMallocAsync(&po_fp32_buf, bytes, stream));
            int64_t shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(model_->config().d_model)};
            Tensor po_fp32(po_fp32_buf, QType::F32, 2, shape, true);
            gemm_via_handle_(ly.wo_id, ao, po_fp32, ctx);
        } else {
            gemm_via_handle_(ly.wo_id, ao, po, ctx);
        }
        if (debug_attn_steps) {
            debug_tensor_stats_all("L0_ao_pre_wo", view_tokens(ao, n), stream);
            debug_tensor_stats_all("L0_po_after_wo", view_tokens(po, n), stream);
            debug_tensor_rows("po_wo-0", view_tokens(po, n), stream);
            debug_tensor_rows("ao_pre_wo-0", view_tokens(ao, n), stream);
            // dump wo weight shape info
            fprintf(stderr, "[DEBUG_FWD] wo_shape: ndim=%d shape=[%ld,%ld] qtype=%d\n", ly.wo.ndim,
                    (long)ly.wo.shape[0], (long)ly.wo.shape[1], (int)ly.wo.qtype);
        }
        if (has_post_attn_norm && using_fp32_accum) {
            // Sandwich norm with FP32 accumulator (Gemma-3):
            // FP32 residual += attn_out, then post_attn_norm → FP16 hidden.
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            float eps = model_->config().rms_norm_eps;
            if (layer == 0 && debug_attn_steps) {
                fprintf(stderr, "[DEBUG_FWD] L0 fp32_accum_kernel: po=%p h=%p fp32_h=%p d=%d n=%d\n", po.data,
                        h.data, fp32_h.data, model_->config().d_model, n);
            }
            // Add attn output to FP32 accumulator, apply post_attn_norm, write FP16
            // 256 threads: d_model_v = d_model/8 (e.g. 480 for Gemma-3 3840),
            // so 2 iterations/thread. 512 wastes half the threads on idle lanes.
            if (layer == 0 && debug_attn_steps) {
                debug_tensor_stats_all("L0_pre_fp32accum_h", view_tokens(h, n), stream);
                debug_tensor_stats_all("L0_pre_fp32accum_po", view_tokens(po, n), stream);
                debug_tensor_rows("pre_fp32accum_po_rows", view_tokens(po, n), stream);
                debug_tensor_rows("pre_fp32accum_h_rows", view_tokens(h, n), stream);
                // Dump FP32 accumulator state
                {
                    std::vector<float> fp32_tmp(n * model_->config().d_model);
                    cudaMemcpy(fp32_tmp.data(), fp32_h.data, fp32_tmp.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    double fs = 0, fss = 0;
                    for (auto v : fp32_tmp) {
                        fs += v;
                        fss += v * v;
                    }
                    fprintf(stderr, "[DEBUG_FWD] L0_fp32_accum_pre: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f\n",
                            fs, std::sqrt(fss), fp32_tmp[0], fp32_tmp[1], fp32_tmp[2]);
                    // Last row (row n-1)
                    int off = (n - 1) * model_->config().d_model;
                    double rs = 0, rss = 0;
                    for (int i = 0; i < model_->config().d_model; i++) {
                        rs += fp32_tmp[off + i];
                        rss += fp32_tmp[off + i] * fp32_tmp[off + i];
                    }
                    fprintf(stderr,
                            "[DEBUG_FWD] L0_fp32_accum_pre[%d]: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f\n",
                            n - 1, rs, std::sqrt(rss), fp32_tmp[off], fp32_tmp[off + 1], fp32_tmp[off + 2]);
                }
            }
            if (fp32_attn_out) {
                rmsnorm_fp32in_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                    static_cast<const float*>(po_fp32_buf), static_cast<const half*>(ly.post_attn_norm.data),
                    static_cast<float*>(fp32_h.data), static_cast<half*>(h.data), model_->config().d_model,
                    eps, norm_w_off_);
            } else {
                rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                    static_cast<const half*>(po.data), static_cast<const half*>(ly.post_attn_norm.data),
                    static_cast<float*>(fp32_h.data), static_cast<half*>(h.data), model_->config().d_model,
                    eps, norm_w_off_);
            }
            if (layer == 0 && debug_attn_steps) {
                debug_tensor_stats_all("L0_post_fp32accum_h", view_tokens(h, n), stream);
            }
        } else if (has_post_attn_norm && prof.is_gemma4) {
            // Gemma 4 sandwich norm: h = r + post_attn_norm(po).
            // Normalize attention output first, THEN add residual (HF reference order).
            rmsnorm(po, ly.post_attn_norm, po, model_->config().rms_norm_eps, stream, norm_w_off_);
            elementwise_add_store(po, r, h, stream);
        } else if (has_post_attn_norm) {
            // Sandwich norm without FP32 accumulator: h = rmsnorm(po + r)
            // Fused: 3 ops (add_store + rmsnorm + memcpy) → 1 kernel
            add_rmsnorm_inplace(po, r, h, ly.post_attn_norm, model_->config().rms_norm_eps, stream,
                                norm_w_off_);
        } else {
            // Standard pre-norm: h = attn_out + residual
            elementwise_add_store(po, r, h, stream);
        }
        if (po_fp32_buf) {
            cudaFreeAsync(po_fp32_buf, stream);
        }
    }

    // gpt-oss o_proj bias (#547): h holds residual + Wo·ao from whichever arm
    // ran — broadcast-add the bias per token here, after all fused variants.
    if (ly.o_bias.data != nullptr) {
        Tensor hv = view_tokens(h, n);
        add_bias(hv, ly.o_bias, stream);
    }

    // LoRA delta on o_proj: h already holds residual + Wo·ao from whichever
    // arm ran (all fused arms require !has_post_attn_norm), so accumulating
    // s·(ao·A^T)·B^T into h afterwards is exactly PEFT's wrapped-Linear
    // semantics. Sandwich-norm archs (post_attn_norm) would need the delta
    // INSIDE the norm — declined in v1 with a log.
    if (lora_) {
        if (const LoraWeights* w = lora_->get(layer, LoraProj::O)) {
            if (!has_post_attn_norm) {
                lora_delta_(*w, ao.data, h.data, n, stream);
            } else {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    IMP_LOG_WARN("LoRA: o_proj adapter on a post-attn-norm arch is unsupported (v1) — "
                                 "delta skipped");
                }
            }
        }
    }
    if (debug_attn_steps) {
        debug_tensor_stats("L0_step4_after_oproj_residual", h, stream);
        debug_tensor_rows("step4_h-0", view_tokens(h, n), stream);
        debug_tensor_stats_all("L0_step4_post_attn_all", view_tokens(h, n), stream);
    }
}

}  // namespace imp

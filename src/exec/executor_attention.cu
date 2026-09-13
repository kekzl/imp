#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include <stdexcept>
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
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention_paged.h"
#include "compute/dispatch_record.h"  // resolved-path recording (#1205)
#include "compute/kv_gather.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "compute/gdn.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/hadamard.h"
#include "compute/mla_kv_assemble.h"
#include "compute/mla_absorb.h"
#include "exec/sparse_attn_select.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "core/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <utility>

#include "exec/executor_attention_internal.h"
namespace imp {

// File-local attention helpers moved to executor_attention_internal.h
// (try_fa2_fp16qk_prefill, dispatch_gemv_qkv_fused, set_l2_persist_kv,
//  clear_l2_persist) to keep this TU under the file-size gate.

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state, cudaStream_t stream) {
    // Configure shared workspace for attention phase
    configure_attn_workspace(ws_.shared_max_tokens());

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
    // Gemma 4: derive per-layer n_heads/n_kv_heads from actual tensor shapes
    // (layer 0 SWA: 16Q/8KV hd=256; layer 5 global: 16Q/2KV hd=512).
    // Authoritative source is the loaded tensor shapes; per-layer config can lag.
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
    // all share the same ws_.shared() memory).
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

    // Residual save: decode (n=1, dp4a) fuses residual into the GEMV; prefill
    // (n>1, FP16 cache) fuses via cuBLAS beta=1 into wo; FP32-accumulator path
    // keeps the residual in fp32_hidden_, skipping the FP16 copy.
    // True sandwich norm (Gemma-3): post_attn_norm inside run_attention AND a
    // separate ffn_norm in run_ffn. When ffn_norm is absent (Qwen3.5),
    // post_attn_norm serves as the FFN input norm instead, not a sandwich norm;
    // without this check it would apply twice.
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr && ly.ffn_norm.data != nullptr);
    // FP32 residual accumulator (Gemma-3 dense + Gemma-4 MoE post-norm architecture).
    // Kernel semantics: fp32_h += rmsnorm(po) * w. llama's build_norm(attn) + residual
    // is mathematically identical for both Gemma-3 and Gemma-4 (normalize-then-add).
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
    if (debug_forward_enabled() && layer <= 1) {
        IMP_LOG_DEBUG(
            "[DEBUG_FWD] L%d attn: has_post_attn_norm=%d using_fp32_accum=%d "
            "post_attn_norm=%p ffn_norm=%p",
            layer, (int)has_post_attn_norm, (int)using_fp32_accum, ly.post_attn_norm.data, ly.ffn_norm.data);
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
    // Batched-decode residual accumulation on CUTLASS_NVFP4 (gemm.nvfp4_residual_beta1):
    // h += o_proj(ao) via the smallm accumulate path, replacing GEMM-to-scratch
    // + elementwise_add_store and skipping the residual save. Ragged prefill
    // chunks n<=32 take it too (same accumulate, one launch fewer).
    bool will_fuse_o_beta1_nvfp4 =
        (!has_post_attn_norm && !will_fuse_o_residual && !will_fuse_o_nvfp4 && !will_fuse_o_beta1 &&
         !will_fuse_o_dequant_beta1 && n > 1 && n <= 32 && h.qtype == QType::F16 &&
         wo_tier == StorageTier::CUTLASS_NVFP4 && dispatch_policy().gemm.nvfp4_residual_beta1 &&
         !cur_spec_verify_ && !overlap_prefill_active_ && lora_ == nullptr && !using_fp32_accum);
    if (!will_fuse_o_residual && !will_fuse_o_beta1 && !will_fuse_o_dequant_beta1 && !will_fuse_o_nvfp4 &&
        !will_fuse_o_beta1_nvfp4 && !using_fp32_accum) {
        // Kernel copy, not cudaMemcpyAsync: a memcpy node has no programmatic
        // edge, so it cut the PDL chain add -> copy -> norm -> q GEMM once per
        // layer (the GEMM's weight prefetch overlaps the whole chain).
        device_copy_async(r.data, h.data, h.nbytes(), stream);
    }

    // For Qwen3.5: Q projection writes to larger buffer (includes gate), then split
    Tensor q_target = has_attn_output_gate ? qv_full : qv;

    // GemmContext for all weight GEMM dispatches in this function.
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, dispatch_policy(), cur_force_fp16_,
                                 model_->config().overrides.gemma4.force_mmvq, cur_spec_verify_);

    // QKV projections [n,d]@W^T -> [n,proj_dim]. Decode (n=1) with matching
    // quant types: fused RMSNorm->Q8_1->QKV GEMV, skipping the intermediate
    // norm_out FP16 buffer. Otherwise separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
#include "exec/executor_attention_qkv.cu"
    }

    // Gemma 4: K=V sharing for global attention layers (wv==null): no V
    // projection exists, V is aliased from K. Copy K->V here so downstream
    // code (QK-norm, V-norm, KV-write, attention) sees a valid V tensor.
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


    // Per-layer RoPE theta and sliding window (Gemma-3: alternating local/global layers).
    // The window decision itself is centralized in layer_swa_window() (shared with
    // the SWA-aware KV sizing); this block only resolves the per-layer RoPE params.
    float layer_rope_theta = cfg.rope_theta;
    float layer_rope_freq_scale = cfg.rope_freq_scale;
    int layer_sliding_window = layer_swa_window(cfg, prof, layer);
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
        }
        // Global layer: full attention, model rope_theta, with freq scaling.
    } else if (prof.attn_variant == AttnVariant::GPTOSS_SWA) {
        // gpt-oss (#547): even layers sliding_attention (window 128), odd
        // layers full attention. Same RoPE (YaRN) on both layer types —
        // only the window toggles.
    } else if (cfg.sliding_window_pattern > 0) {
        bool is_global = (layer % cfg.sliding_window_pattern) == (cfg.sliding_window_pattern - 1);
        if (!is_global) {
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

    // SWA-aware KV sizing (kv_cache.swa_sizing): windowed layers read/write
    // the small SWA block group through the parallel table (-1 holes outside
    // the trailing window). nullptr table = feature off, every layer shares state.block_tables.
    const int* layer_block_tables =
        (state.block_tables_swa != nullptr && layer_sliding_window > 0) ? state.block_tables_swa
                                                                        : state.block_tables;

    // Select LongRoPE frequency table based on context length (nullptr if not longrope)
    const float* longrope_freqs = nullptr;
    if (longrope_short_freqs_) {
        longrope_freqs = (state.max_context_len <= longrope_orig_max_pos_) ? longrope_short_freqs_
                                                                           : longrope_long_freqs_;
    }
    // Gemma 4: per-layer rope_freqs (pre-computed effective frequencies for
    // global layers, gguf_loader.cpp:1160), matching llama.cpp's
    // gemma4-iswa.cpp passing them as freq_factors to ggml_rope_ext on
    // full_attention layers (n_rot=hd, proportional-rope schema ccss000000000000).
    if (prof.attn_variant == AttnVariant::GEMMA4_SWA) {
        bool layer_is_swa = (layer < (int)cfg.swa_layers.size() && cfg.swa_layers[layer]);
        if (!layer_is_swa && ly.rope_freqs.data && ly.rope_freqs.on_device) {
            longrope_freqs = static_cast<const float*>(ly.rope_freqs.data);
        }
    }

    // Attention output gate (fused Q+gate): per-head interleaved
    // [Q_h0(hd),Gate_h0(hd),...] is the DEFAULT and correct for every checkpoint
    // tested (Qwen3.5, Qwen3.6): matches HF's per-head chunk split, and NVFP4
    // block-scale periodicity on disk confirms it. `attention.gate_concat` is
    // an escape hatch for a feature-dim-concat checkpoint that has not shipped
    // yet; flipping it breaks every staged hybrid today.
    Tensor attn_gate_buf;
    if (has_attn_output_gate) {
        // The gate borrows the SSM z buffer (carved only by
        // configure_ssm_workspace(), which runs only from run_ssm()/run_gdn()).
        // Every gated checkpoint today is recurrent with a recurrent layer before
        // its first attention layer, so the buffer is live here, but this is not
        // guaranteed by construction: getting it wrong writes through a null/stale pointer silently, prefill
        // only.
        if (ssm_z_buf_.data == nullptr) {
            throw std::runtime_error(
                "attention output gate has no buffer: it borrows the SSM z buffer, which this "
                "model never carved (no recurrent layer ran before the first attention layer). "
                "See exec_ssm_z_cols() — the gate needs its own allocation on such a model.");
        }
        size_t es_q = dtype_size(compute_dtype_);
        int64_t gate_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(q_actual_dim)};
        attn_gate_buf = Tensor(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);

        const bool use_concat = dispatch_policy().attention.gate_concat;
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
            // Per-head interleaved (Qwen 3.5 layout): Q[t,h*hd:(h+1)*hd] =
            // src[t,h*2*hd:h*2*hd+hd], Gate = src[t,h*2*hd+hd:(h+1)*2*hd]. One fused
            // kernel replaces an nh x 2 cudaMemcpy2DAsync loop.
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
        // Fused K-RoPE-into-KV-write is declined for: YaRN (#547, gpt-oss) - the
        // fused kernels know only theta+1/scale, no ramp/attn_factor, so decode-
        // written K would go plain-RoPE and degenerate from token 2; MLA - fused
        // kernel doesn't support asymmetric K/V head dims (hd=192, vhd=128);
        // M-RoPE active - the fused kernels take one position array per axis-blind rotation, which would
        // silently misrotate image tokens.
        const bool mrope_active = state.mrope.positions != nullptr || state.mrope.pos_delta != nullptr;
        bool can_fuse_rope_kv = (!state.is_prefill && n == 1 && qv.qtype == QType::F16 && state.kv_cache &&
                                 state.kv_cache->qtype() == QType::F16 &&
                                 prof.attn_variant != AttnVariant::NOPE && cfg.yarn_ext_factor <= 0.0f &&
                                 !cfg.is_mla() && !mrope_active);
        // Per-layer rope_dim. Gemma 4: both SWA and global layers rotate the full
        // head_dim; global layers' freq_factors (loaded as longrope_freqs) zero
        // out most pairs to realize the GGUF's partial-rotary schedule (ccss000000000000).
        int fused_rope_dim = cfg.rope_dim;
        if (prof.is_gemma4) {
            fused_rope_dim = hd;
        } else if (fused_rope_dim > hd || fused_rope_dim <= 0) {
            fused_rope_dim = hd;
        }
        const bool no_qknorm_fused = dispatch_policy().attention.no_qknorm_fused;
        // Fused QK-norm+RoPE covers batched-decode rows too (n<=64, one CTA per
        // head x token), replacing three separate launches per layer at 32
        // streams. Applies the norm weight over the full head; sub-head norm
        // layouts (norm dim < head_dim) stay on the separate path.
        const bool full_head_norm = ly.attn_q_norm.data != nullptr && ly.attn_k_norm.data != nullptr &&
                                    ly.attn_q_norm.shape[0] == hd && ly.attn_k_norm.shape[0] == hd;
        if (has_qk_norm && n <= 64 && (n == 1 || full_head_norm) && qv.qtype == QType::F16 &&
            !no_qknorm_fused && prof.attn_variant != AttnVariant::NOPE) {
            // Fused: QK-norm + RoPE in one kernel launch. Keeps norm
            // intermediate values in FP32 shared memory.
            qknorm_rope_fused(static_cast<half*>(qv.data), static_cast<half*>(kk.data),
                              static_cast<const half*>(ly.attn_q_norm.data),
                              static_cast<const half*>(ly.attn_k_norm.data), nh, nkv, hd, eps,
                              state.positions, layer_rope_theta, layer_rope_freq_scale, fused_rope_dim,
                              cfg.rope_neox, stream, norm_w_off_, cfg.yarn_ext_factor, cfg.yarn_attn_factor,
                              cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, longrope_freqs,
                              state.mrope, n);
        } else if (can_fuse_rope_kv && !has_qk_norm) {
            // Fused path: Q-only RoPE here, K-RoPE deferred to KV write
            const int effective_rope_dim = fused_rope_dim;
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            rope_q_only_fp16_kernel<<<dim3(1, nh), pairs, 0, stream>>>(static_cast<half*>(qv.data),
                                                                       state.positions, nh, hd,
                                                                       layer_rope_theta, inv_scaling, pairs,
                                                                       cfg.rope_neox, longrope_freqs);
            IMP_CUDA_CHECK_LAUNCH();
            rope_k_deferred = true;
        } else {
            // Some archs (Qwen3.5-27B-mxfp4) ship attn_q_norm/attn_k_norm with a
            // smaller dim than head_dim: the weight applies per norm_dim-sized chunk,
            // so a 512-dim head with a 256-dim norm splits into two 256-dim sub-heads
            // sharing the scale. Detected from the norm weight's element count; no-op when norm_dim==hd.
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
                             cfg.yarn_ext_factor > 0.0f ? yarn_corr_dims_ : nullptr, stream, longrope_freqs,
                             state.mrope);
            }
        }
    }

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step2_after_rope_q", qv, stream);
        debug_tensor_stats("L0_step2_after_rope_k", kk, stream);
    }

    // Attention scale: standard archs 1/sqrt(head_dim); Gemma 4 = 1.0 (Q/K-norm
    // absorb the per-element scaling, per llama.cpp f_attn_scale); MLA
    // multiplies by YaRN mscale_adj^2, mscale_adj = 0.1*mscale_all_dim*ln(yarn_factor)+1.0.
    float scale = (prof.is_gemma4) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(hd)));
    if (cfg.is_mla()) scale *= mla_attention_scale_multiplier(cfg);

    // gpt-oss learned attention sinks (#547): per-head logits acting as a
    // virtual extra softmax column. Only the cuBLAS prefill softmax and the
    // FP16 paged decode kernel understand them; prefill forces cuBLAS when sinks are present.
    const void* attn_sinks = (prof.is_gpt_oss) ? ly.attn_sinks.data : nullptr;

    // MLA absorbed-decode (opt-in): populates the per-layer latent cache with
    // this step's RMSNorm'd latent + post-RoPE decoupled key for BOTH prefill
    // and decode, so the cache is warm before the first decode step. Single-sequence only.
    const bool mla_absorb_active =
        dispatch_policy().attention.mla_absorb && cfg.is_mla() && mla_absorb_cache_ != nullptr &&
        state.n_sequences == 1;
    if (mla_absorb_active) {
        half* cache_layer = static_cast<half*>(mla_absorb_cache_) +
                            static_cast<size_t>(layer) * mla_absorb_layer_stride_;
        mla_latent_cache_write(static_cast<const half*>(mla_latent_buf_),
                               static_cast<const half*>(kk.data), cache_layer, state.positions, n, nh,
                               hd, cfg.qk_rope_head_dim, cfg.kv_lora_rank, mla_absorb_max_seq_, stream);
    }

    // Decode attention for `n` one-token rows: historical inline block, now a
    // lambda so ragged prefill can hand its riders (mixed prefill+decode step)
    // to it as ONE batched launch per layer. `state`/`layer_block_tables` are
    // the sub-batch's own (block tables, context lens, positions start at its first sequence).
    auto decode_attend = [&](const InferenceState& state, int n, int row_begin, Tensor qv, Tensor kk,
                             Tensor vv, Tensor ao, const int* layer_block_tables) {
#include "exec/executor_attention_decode.cu"
    };

    if (state.is_prefill && !state.chunk_decode_attn) {
        // Prefill attention for ONE sequence's rows: historical inline dispatch,
        // parameterized on per-sequence geometry so the ragged cross-sequence path
        // can loop it. Non-ragged callers pass bt_flat=nullptr, keeping
        // write_kv_cache on its historical default path.
        auto prefill_attend_seq = [&](int n, int q_offset, int row_begin, Tensor qv, Tensor kk,
                                      Tensor vv, Tensor ao, const int* layer_block_tables,
                                      const int* bt_flat, const int* bt_swa_flat,
                                      const int* seq_positions) {
#include "exec/executor_attention_prefill.cu"
        };
        if (state.ragged_prefill()) {
            // Ragged cross-sequence prefill: loop the per-seq dispatch over
            // row sub-ranges. QKV projection / QK-norm / RoPE above already
            // ran row-wise over the concatenation.
            const size_t es_r = dtype_size(compute_dtype_);
            auto rows_view = [&](const Tensor& t, int64_t cols, int row0, int nrows) -> Tensor {
                int64_t shape[2] = {static_cast<int64_t>(nrows), cols};
                char* p = static_cast<char*>(t.data) + static_cast<size_t>(row0) * cols * es_r;
                return Tensor(p, t.qtype, 2, shape, true);
            };
            const int n_pf = state.n_sequences - state.n_riders;
            for (int s = 0; s < n_pf; ++s) {
                const int rb = state.h_seq_offsets[s];
                const int ns = state.h_seq_offsets[s + 1] - rb;
                if (ns <= 0)
                    continue;
                const int* bt_s = state.block_tables + static_cast<size_t>(s) * state.max_blocks_per_seq;
                prefill_attend_seq(ns, state.h_seq_q_offsets[s], rb,
                                   rows_view(qv, static_cast<int64_t>(nh) * hd, rb, ns),
                                   rows_view(kk, static_cast<int64_t>(nkv) * hd, rb, ns),
                                   rows_view(vv, static_cast<int64_t>(nkv) * hd, rb, ns),
                                   rows_view(ao, static_cast<int64_t>(nh) * hd, rb, ns), bt_s, bt_s,
                                   /*bt_swa_flat=*/nullptr, state.positions + rb);
            }
            // Riders: trailing one-row members decode as one paged batch. Their
            // sub-state starts at sequence n_pf/row rb; ragged geometry is cleared so
            // the decode block sees a plain n_riders-sequence decode step.
            if (state.n_riders > 0) {
                const int rb = state.h_seq_offsets[n_pf];
                const int nr = state.h_seq_offsets[state.n_sequences] - rb;
                InferenceState rst = state;
                rst.is_prefill = false;
                rst.n_tokens = nr;
                rst.n_sequences = state.n_riders;
                rst.block_tables = state.block_tables + static_cast<size_t>(n_pf) * state.max_blocks_per_seq;
                rst.block_tables_swa = nullptr;
                rst.context_lens = state.context_lens + n_pf;
                rst.positions = state.positions + rb;
                rst.max_context_len = state.rider_max_context_len;
                rst.seq_offsets = nullptr;
                rst.h_seq_offsets = nullptr;
                rst.h_seq_q_offsets = nullptr;
                rst.h_ssm_slots = nullptr;
                rst.n_riders = 0;
                decode_attend(rst, nr, rb, rows_view(qv, static_cast<int64_t>(nh) * hd, rb, nr),
                              rows_view(kk, static_cast<int64_t>(nkv) * hd, rb, nr),
                              rows_view(vv, static_cast<int64_t>(nkv) * hd, rb, nr),
                              rows_view(ao, static_cast<int64_t>(nh) * hd, rb, nr), rst.block_tables);
            }
        } else {
            prefill_attend_seq(n, state.prefill_offset, 0, qv, kk, vv, ao, layer_block_tables,
                               /*bt_flat=*/nullptr, /*bt_swa_flat=*/nullptr, /*seq_positions=*/nullptr);
        }
    } else {
        decode_attend(state, n, /*row_begin=*/0, qv, kk, vv, ao, layer_block_tables);
    }

    if (debug_attn_steps) {
        debug_tensor_stats("L0_step3_after_paged_attn", ao, stream);
        debug_tensor_stats("L0_step3_h_before_oproj", h, stream);
    }

    // MLA: V head dim (vhd) is narrower than QK head dim (hd); o_proj expects
    // [n, nh*vhd]. Decode (paged kernel) writes ao COMPACTLY already (shape
    // fixup only). Prefill (cuBLAS/FA2/FMHA) leaves ao as [n,nh,hd] (V
    // zero-padded to hd): needs a per-head [n,nh,hd]->[n,nh,vhd] compaction; a
    // naive narrow would mix head boundaries.
    if (cfg.is_mla() && cfg.v_head_dim > 0 && cfg.v_head_dim != hd) {
        const int vhd = cfg.v_head_dim;
        const int64_t mla_ao_cols = static_cast<int64_t>(nh) * vhd;
        if (state.is_prefill) {
            // Compacts hd-strided -> vhd-compact via a scratch buffer (src/dst must
            // not alias). Prefill isn't CUDA-graph-captured, so the stream-ordered
            // alloc is amortised in the pool.
            void* compact_buf = nullptr;
            const size_t bytes = static_cast<size_t>(n) * mla_ao_cols * sizeof(half);
            IMP_CUDA_CHECK_LOG(cudaMallocAsync(&compact_buf, bytes, stream));
            mla_compact_attn_output(static_cast<const half*>(ao.data),
                                    static_cast<half*>(compact_buf), n, nh, hd, vhd, stream);
            // Copy compacted result back into attn_out_ (dst pitch < src pitch,
            // separate-buffer source — no in-place hazard) and narrow the view.
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ao.data, compact_buf, bytes,
                                               cudaMemcpyDeviceToDevice, stream));
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(compact_buf, stream));
        }
        // Both paths: narrow the view so o_proj sees nh*vhd. (Decode already
        // wrote compact data; prefill just copied compact data into ao.)
        if (ao.shape[1] != mla_ao_cols) {
            ao.shape[1] = mla_ao_cols;
            ao.compute_strides();
        }
    }

    // Qwen3.5 attention output gate: ao[i] *= sigmoid(gate[i])
    if (has_attn_output_gate) {
        sigmoid_mul(ao, attn_gate_buf, ao, stream);
    }

    // O projection + residual: decode (n=1, dp4a) fuses the residual add into
    // the GEMV, writing straight to hidden. When will_fuse_o_residual is set,
    // the initial h->r memcpy is skipped and h.data itself is the residual
    // source (safe: h.data is only read, never written, in this span).
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
    } else if (will_fuse_o_beta1_nvfp4) {
        // Batched-decode NVFP4: hidden += attn_out @ wo^T via the smallm
        // accumulate path (falls through to a beta-honouring handler if the
        // smallm route declines — the cutlass handler refuses beta != 0).
        gemm_via_handle_(ly.wo_id, ao, h, ctx.with_beta(1.0f));
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
        gemm_via_handle_(ly.wo_id, ao, po, ctx);
        if (debug_attn_steps) {
            debug_tensor_stats_all("L0_ao_pre_wo", view_tokens(ao, n), stream);
            debug_tensor_stats_all("L0_po_after_wo", view_tokens(po, n), stream);
            debug_tensor_rows("po_wo-0", view_tokens(po, n), stream);
            debug_tensor_rows("ao_pre_wo-0", view_tokens(ao, n), stream);
            // dump wo weight shape info
            IMP_LOG_DEBUG("[DEBUG_FWD] wo_shape: ndim=%d shape=[%ld,%ld] qtype=%d", ly.wo.ndim,
                          (long)ly.wo.shape[0], (long)ly.wo.shape[1], std::to_underlying(ly.wo.qtype));
        }
        if (has_post_attn_norm && using_fp32_accum) {
            // Sandwich norm with FP32 accumulator (Gemma-3):
            // FP32 residual += attn_out, then post_attn_norm → FP16 hidden.
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            float eps = model_->config().rms_norm_eps;
            if (layer == 0 && debug_attn_steps) {
                IMP_LOG_DEBUG("[DEBUG_FWD] L0 fp32_accum_kernel: po=%p h=%p fp32_h=%p d=%d n=%d", po.data,
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
                    IMP_LOG_DEBUG("[DEBUG_FWD] L0_fp32_accum_pre: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f", fs,
                                  std::sqrt(fss), fp32_tmp[0], fp32_tmp[1], fp32_tmp[2]);
                    // Last row (row n-1)
                    int off = (n - 1) * model_->config().d_model;
                    double rs = 0, rss = 0;
                    for (int i = 0; i < model_->config().d_model; i++) {
                        rs += fp32_tmp[off + i];
                        rss += fp32_tmp[off + i] * fp32_tmp[off + i];
                    }
                    IMP_LOG_DEBUG("[DEBUG_FWD] L0_fp32_accum_pre[%d]: sum=%.4f L2=%.4f [0..2]=%.6f %.6f %.6f",
                                  n - 1, rs, std::sqrt(rss), fp32_tmp[off], fp32_tmp[off + 1],
                                  fp32_tmp[off + 2]);
                }
            }
            rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256, 0, stream>>>(
                static_cast<const half*>(po.data), static_cast<const half*>(ly.post_attn_norm.data),
                static_cast<float*>(fp32_h.data), static_cast<half*>(h.data), model_->config().d_model, eps,
                norm_w_off_);
            IMP_CUDA_CHECK_LAUNCH();
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
    }

    // gpt-oss o_proj bias (#547): h holds residual + Wo·ao from whichever arm
    // ran — broadcast-add the bias per token here, after all fused variants.
    if (ly.o_bias.data != nullptr) {
        Tensor hv = view_tokens(h, n);
        add_bias(hv, ly.o_bias, stream);
    }

    // LoRA delta on o_proj: h already holds residual + Wo.ao from whichever
    // arm ran (all fused arms require !has_post_attn_norm), so accumulating
    // s.(ao.A^T).B^T afterwards matches PEFT's wrapped-Linear semantics.
    // Sandwich-norm archs would need the delta INSIDE the norm; declined in v1 (logged).
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

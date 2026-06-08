#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/activation.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace imp {

// Shared helpers (align256, make_workspace_tensor, vram_alloc, vram_free)
// are defined in executor_helpers.h

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() { free_buffers(); }

bool GraphExecutor::init(const Model& model, QType compute_dtype, bool use_pdl, int max_batch_size,
                         int max_seq_len, bool use_fp8_prefill, int use_nvfp4_decode,
                         bool use_mxfp4_prefill) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    norm_w_off_ = model.config().norm_weight_offset;

    // Gemma 4: allocate a ones buffer for V-normalization (no learned weight).
    // Size = max head_dim (512 for 26B model). Used as rmsnorm weight.
    if (model.profile().is_gemma4) {
        int hd_max = 0;
        for (int v : model.config().head_dim_per_layer)
            hd_max = std::max(hd_max, v);
        if (hd_max == 0)
            hd_max = model.config().head_dim;
        size_t buf_bytes = static_cast<size_t>(hd_max) * sizeof(half);
        if (cudaMalloc(&v_norm_ones_buf_, buf_bytes) == cudaSuccess) {
            // Fill with FP16 1.0 via memset is not possible (FP16 1.0 is 0x3C00).
            // Use a small host buffer + memcpy.
            std::vector<uint16_t> ones(hd_max, 0x3C00);  // FP16 1.0
            cudaMemcpy(v_norm_ones_buf_, ones.data(), buf_bytes, cudaMemcpyHostToDevice);
            IMP_LOG_INFO("Gemma 4: allocated V-norm ones buffer (%d halfs)", hd_max);
        }
    }
    use_pdl_ = use_pdl;
    wcache_.use_fp8 = use_fp8_prefill;
    wcache_.nvfp4_decode_mode = use_nvfp4_decode;
    wcache_.use_mxfp4 = use_mxfp4_prefill;
    // dual_path_quant is set separately by the engine after init
    hints_.prefer_fp8 = use_fp8_prefill;
    hints_.prefer_nvfp4_decode = (use_nvfp4_decode > 0);

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
    has_gdn_ = false;  // detected from tensor presence below
    has_dense_ffn_ = (cfg.d_ff > 0);

    // Compute max expert FFN hidden dim from actual packed tensor shapes.
    // cfg.expert_d_ff may not match the actual tensor dimensions (e.g. Nemotron-H).
    // Gemma 4: ffn_gate_up_exps is fused (shape[1] = 2*expert_d_ff), but it gets
    // split at weight_upload time. Trust cfg.expert_d_ff over the pre-split shape.
    max_expert_eff_ = cfg.expert_d_ff;
    if (has_moe_ && cfg.arch != ModelArch::GEMMA4) {
        for (int li = 0; li < cfg.n_layers; li++) {
            const auto& L = model.layer(li);
            // gate/up packed: shape [n_experts, expert_d_ff, d_model]
            for (const auto* p : {&L.expert_gate_packed, &L.expert_up_packed}) {
                if (p->data && p->ndim >= 3)
                    max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(p->shape[1]));
            }
            // down packed: shape [n_experts, d_model, expert_d_ff]
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3)
                max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(L.expert_down_packed.shape[2]));
        }
        if (max_expert_eff_ != cfg.expert_d_ff) {
            IMP_LOG_WARN("expert_d_ff mismatch: config=%d, actual packed tensors=%d — using %d",
                         cfg.expert_d_ff, max_expert_eff_, max_expert_eff_);
        }
    }

    // Use engine-provided max_seq_len if given, otherwise fall back to model config.
    int effective_seq_len = (max_seq_len > 0) ? max_seq_len : cfg.max_seq_len;
    max_tokens_ = std::min(effective_seq_len, 4096);
    if (max_tokens_ <= 0) {
        max_tokens_ = 4096;
    }

    // Cap max_tokens for hybrid MoE+SSM/GDN models to bound workspace VRAM.
    // SSM state + cuBLAS S-matrix + workspace can exhaust 32 GB VRAM at the
    // model's full max_seq_len. Chunked-prefill on hybrid archs IS supported
    // now (uniform attention shapes across layers), but the cap still keeps
    // single-chunk prefills cheap; the engine clamps effective_chunk to this
    // value and falls into the chunked path for longer prompts. 2048 covers
    // most real prompts in one shot at ~190 MiB shared workspace
    // (attn_scores n_heads × N² is the dominant term).
    if (has_ssm_ && (has_moe_ || has_gdn_)) {
        int capped = has_moe_ ? 2048 : 2048;
        if (max_tokens_ > capped) {
            IMP_LOG_INFO("executor_workspace.cu:%d: Capping max_tokens %d → %d for SSM/GDN hybrid", __LINE__,
                         max_tokens_, capped);
            max_tokens_ = capped;
        }
    }

    // Logits buffer only needs to hold tokens that require LM head projection:
    // - Prefill: 1 (last token only)
    // - Decode:  n_sequences (one per batch slot)
    max_logit_tokens_ = std::max(max_batch_size, 1);

    // Compute shared workspace sizes (no allocation — deferred to allocate_workspaces()).
    // Deferring GPU allocation maximizes VRAM available for expert weight upload.
    compute_shared_sizes(max_tokens_);

    // Build SSM layer index mapping
    if (has_ssm_) {
        ssm_layer_map_.resize(cfg.n_layers, -1);
        int ssm_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).ssm_in.data != nullptr) {
                ssm_layer_map_[i] = ssm_idx++;
            }
        }
        IMP_LOG_INFO("SSM layers: %d out of %d total", ssm_idx, cfg.n_layers);
    }

    // Build GDN layer index mapping (Gated DeltaNet, e.g., Qwen3.5)
    {
        gdn_layer_map_.resize(cfg.n_layers, -1);
        int gdn_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).gdn_gate.data != nullptr) {
                gdn_layer_map_[i] = gdn_idx++;
            }
        }
        if (gdn_idx > 0) {
            has_gdn_ = true;
            IMP_LOG_INFO("GDN layers: %d out of %d total", gdn_idx, cfg.n_layers);
        }
    }

    // Enable Programmatic Dependent Launch on custom kernels if requested.
    if (use_pdl_ && pdl::is_available()) {
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp16_kernel));
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_rope_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp16_to_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp32_to_fp16_kernel));
        // Register compute kernels for PDL overlap (run between GEMMs in hot path)
        layernorm_pdl_register();
        rope_pdl_register();
        activation_pdl_register();
        gemv_pdl_register();
        nvfp4_gemv_pdl_register();
        IMP_LOG_INFO("PDL enabled on executor + compute + GEMV kernels");
    } else if (use_pdl_) {
        IMP_LOG_WARN("PDL requested but not available on this device/CUDA version");
        use_pdl_ = false;
    }

    // SMEM carveout: maximize L1 for bandwidth-bound GEMV kernels (independent of PDL).
    mxfp4_gemv_set_l1_carveout();

    // Precompute YaRN correction dimensions if enabled
    if (cfg.yarn_ext_factor > 0.0f) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int n_dims = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
        int n_ctx_orig = cfg.rope_n_ctx_orig > 0 ? cfg.rope_n_ctx_orig : cfg.max_seq_len;
        rope_yarn_corr_dims(n_dims, n_ctx_orig, cfg.rope_theta, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                            yarn_corr_dims_);
        IMP_LOG_INFO("YaRN corr_dims: [%.1f, %.1f] (n_dims=%d, n_ctx_orig=%d)", yarn_corr_dims_[0],
                     yarn_corr_dims_[1], n_dims, n_ctx_orig);
    }

    // Pre-compute LongRoPE inverse frequencies if enabled (Phi-4)
    if (!cfg.rope_short_factor.empty() && !cfg.rope_long_factor.empty()) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int rd = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
        int pairs = rd / 2;
        longrope_n_pairs_ = pairs;
        longrope_orig_max_pos_ = cfg.rope_scaling_orig_max_pos;

        // inv_freq[i] = 1.0 / (factor[i] * theta^(2i/rd))
        std::vector<float> short_freqs(pairs), long_freqs(pairs);
        for (int i = 0; i < pairs; i++) {
            float base_freq = 1.0f / std::pow(cfg.rope_theta, (2.0f * i) / static_cast<float>(rd));
            short_freqs[i] = base_freq / cfg.rope_short_factor[i];
            long_freqs[i] = base_freq / cfg.rope_long_factor[i];
        }

        cudaError_t e1 = cudaMalloc(&longrope_short_freqs_, pairs * sizeof(float));
        cudaError_t e2 = cudaMalloc(&longrope_long_freqs_, pairs * sizeof(float));
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate LongRoPE frequency buffers: %s",
                          cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
            if (longrope_short_freqs_) {
                cudaFree(longrope_short_freqs_);
                longrope_short_freqs_ = nullptr;
            }
            if (longrope_long_freqs_) {
                cudaFree(longrope_long_freqs_);
                longrope_long_freqs_ = nullptr;
            }
            return false;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpy(longrope_short_freqs_, short_freqs.data(), pairs * sizeof(float),
                                      cudaMemcpyHostToDevice));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(longrope_long_freqs_, long_freqs.data(), pairs * sizeof(float),
                                      cudaMemcpyHostToDevice));

        IMP_LOG_INFO("LongRoPE: %d freq pairs, orig_max_pos=%d", pairs, longrope_orig_max_pos_);
    }

    initialized_ = true;

    IMP_LOG_INFO(
        "GraphExecutor initialized: max_tokens=%d, d_model=%d, "
        "n_layers=%d, dtype=%s, pdl=%s",
        max_tokens_, cfg.d_model, cfg.n_layers, dtype_name(compute_dtype_), use_pdl_ ? "on" : "off");
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2: allocate all GPU workspace buffers (called after weight upload)
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_workspaces(bool experts_on_host) {
    if (!initialized_ || !model_)
        return false;

    if (!allocate_persistent_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Persistent workspace allocation failed — cannot run inference");
        return false;
    }
    if (!allocate_shared_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Shared workspace allocation failed — cannot run inference");
        return false;
    }
    // Always allocate batch dequant buffer — GPU-resident layers need it even
    // when some other layers are host-resident. Without it, ALL layers fall to
    // the serial path (major perf regression; was a correctness regression
    // before the host gate_up split fix since serial path had undefined
    // behavior for Gemma-4 host experts).
    allocate_auxiliary_buffers(/*skip_batch_dequant=*/false);
    (void)experts_on_host;

    return true;
}

size_t GraphExecutor::workspace_estimate() const {
    if (!model_)
        return 0;
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    size_t es = dtype_size(compute_dtype_);

    // Persistent: hidden + residual + norm_out + logits
    size_t persistent = 3 * align256(static_cast<size_t>(max_tokens_) * d * es) +
                        align256(static_cast<size_t>(max_logit_tokens_) * cfg.vocab_size * sizeof(float));

    // Shared: max of phases (already computed in compute_shared_sizes)
    size_t shared = std::max({attn_shared_size_, ffn_shared_size_, moe_shared_size_, ssm_shared_size_});

    // S-matrix is NOT included here — it's optional (flash attention fallback works).
    // This maximizes VRAM available for expert layers during weight upload.
    // S-matrix is allocated opportunistically from remaining VRAM.

    // FP32 accumulator for post-norm models (Gemma-3): 1 × max_tokens × d_model × 4
    bool has_post_norms = (cfg.norm_placement == NormPlacement::POST_NORM);
    size_t fp32_accum = has_post_norms ? align256(static_cast<size_t>(max_tokens_) * d * sizeof(float)) : 0;

    // Auxiliary: compute real estimate from individual buffer sizes
    size_t auxiliary = 0;

    // Dequant scratch: max weight matrix elements × sizeof(half)
    {
        size_t max_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                                  &L.w_up_shared, &L.w_down_shared, &L.ssm_in, &L.ssm_out}) {
                if (w->data)
                    max_elems = std::max(max_elems, static_cast<size_t>(w->numel()));
            }
        }
        auxiliary += max_elems * sizeof(uint16_t);
    }

    // Sampling result (ARGMAX_SCRATCH_BYTES ~16 KiB) + MMVQ scratch + split-K scratch
    int nh_est = cfg.n_heads;
    int hd_est = cfg.head_dim > 0 ? cfg.head_dim : (d / nh_est);
    auxiliary += 16 * 1024;   // sampling
    auxiliary += 256 * 1024;  // MMVQ scratch (conservative)
    auxiliary += static_cast<size_t>(max_logit_tokens_) * nh_est * 32 * (2 + hd_est) *
                 sizeof(float);  // split-K

    // S-matrix for cuBLAS attention fallback — only needed when CUTLASS FMHA
    // is unavailable or unsupported (e.g., softcap, sliding window).
    // Skip for MoE-heavy models where VRAM is tight: WMMA/CUTLASS FMHA
    // doesn't need the S-matrix. This saves up to 256 MiB.
    bool is_moe = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    if (!is_moe) {
        auxiliary += std::min(static_cast<size_t>(nh_est) * max_tokens_ * max_tokens_ * sizeof(half),
                              static_cast<size_t>(256) << 20);
    }

    // Safety margin for FP8 act buffers, misc (reduced for MoE to save VRAM)
    auxiliary += is_moe ? (8ULL << 20) : (32ULL << 20);

    return persistent + shared + fp32_accum + auxiliary;
}

// ---------------------------------------------------------------------------
// Unified workspace allocation
// ---------------------------------------------------------------------------

void GraphExecutor::compute_shared_sizes(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int ff = cfg.d_ff;
    int nh = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    // Attention phase: q, k+v (contiguous for batched GEMM), attn_out, proj_out
    // Check for Q+Gate interleaving (Qwen3.5): Q projection output is 2x larger
    // than standard Q when an attention output gate is present.
    int max_q_out = nh * hd;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& ly = model_->layer(i);
        if (ly.wq.data) {
            int q_dim = static_cast<int>(ly.wq.shape[0]);
            if (q_dim > max_q_out)
                max_q_out = q_dim;
        }
    }
    size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q (de-interleaved)
                        + align256(2 * kv_raw)                                      // k+v contiguous
                        + align256(static_cast<size_t>(max_tokens) * nh * hd * es)  // attn_out
                        + align256(static_cast<size_t>(max_tokens) * d * es)        // proj_out
                        +
                        (max_q_out > nh * hd
                             ? align256(static_cast<size_t>(max_tokens) * max_q_out * es)  // qv_full (Q+Gate)
                             : 0);

    // Dense FFN phase: gate, up, swiglu, ffn_out
    if (has_dense_ffn_ && ff > 0) {
        ffn_shared_size_ = align256(static_cast<size_t>(max_tokens) * ff * es)    // gate_out
                           + align256(static_cast<size_t>(max_tokens) * ff * es)  // up_out
                           + align256(static_cast<size_t>(max_tokens) * ff * es)  // swiglu_out
                           + align256(static_cast<size_t>(max_tokens) * d * es);  // ffn_out
    }

    // MoE phase
    if (has_moe_) {
        int ne = cfg.n_experts;
        int top_k = cfg.n_experts_active;
        int eff = max_expert_eff_;
        int expanded = max_tokens * top_k;

        moe_shared_size_ = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float))    // gate_logits
                           + align256(static_cast<size_t>(expanded) * d * es)                // gathered
                           + align256(static_cast<size_t>(expanded) * eff * es)              // expert_gate
                           + align256(static_cast<size_t>(expanded) * eff * es)              // expert_up
                           + align256(static_cast<size_t>(expanded) * eff * es)              // expert_swiglu
                           + align256(static_cast<size_t>(expanded) * d * es)                // expert_down
                           + align256(static_cast<size_t>(max_tokens) * d * sizeof(float));  // scatter_out
    }

    // SSM phase
    if (has_ssm_) {
        int inner = cfg.ssm_inner_size;
        int n_groups = cfg.ssm_group_count;
        int state_size = cfg.ssm_state_size;
        int n_heads = cfg.ssm_dt_rank;
        int conv_channels = inner + 2 * n_groups * state_size;
        int ssm_in_dim = inner + conv_channels + n_heads;

        size_t proj_elem_size = has_gdn_ ? sizeof(float) : es;
        int fused_total_out = conv_channels + inner + 2 * n_heads;
        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim *
                                    proj_elem_size)  // proj (FP32 for GDN)
                           + align256(static_cast<size_t>(max_tokens) * conv_channels * es)  // xBC
                           + align256(static_cast<size_t>(max_tokens) * inner * es)          // y
                           + align256(static_cast<size_t>(max_tokens) * inner * es)          // z
                           + align256(static_cast<size_t>(max_tokens) * d * es)              // out
                           + align256(static_cast<size_t>(max_tokens) * n_heads * (has_gdn_ ? 2 : 1) *
                                      es)  // dt (2x for GDN: alpha + beta)
                           + (has_gdn_ ? align256(static_cast<size_t>(max_tokens) * fused_total_out * es)
                                       : 0);  // gdn_fused_proj (only on GDN models)
    }
}

bool GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);

    size_t hidden_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    persistent_workspace_ = vram_alloc(vram_alloc_, total, "persistent_workspace");
    if (!persistent_workspace_) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%.1f MiB)", total / (1024.0 * 1024.0));
        return false;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    hidden_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, hidden_sz);
    residual_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, residual_sz);
    norm_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, norm_out_sz);

    {
        int64_t shape[2] = {static_cast<int64_t>(max_logit_tokens_), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, QType::F32, 2, shape, true);
        ptr += logits_sz;
    }

    IMP_LOG_INFO("Persistent workspace: %.2f MiB (hidden+residual+norm+logits)", total / (1024.0 * 1024.0));

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    if (cfg.norm_placement == NormPlacement::POST_NORM) {
        size_t fp32_sz = align256(static_cast<size_t>(max_tokens) * d * sizeof(float));
        cudaError_t e2 = cudaMalloc(&fp32_accum_buf_, fp32_sz);
        if (e2 == cudaSuccess) {
            int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(d)};
            fp32_hidden_ = Tensor(fp32_accum_buf_, QType::F32, 2, shape, true);
            IMP_LOG_INFO("FP32 residual accumulator: %.2f MiB (post-norm architecture)",
                         fp32_sz / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("Failed to allocate FP32 accumulator (%zu bytes): %s — falling back to FP16",
                         fp32_sz, cudaGetErrorString(e2));
        }
    }
    return true;
}

bool GraphExecutor::allocate_shared_workspace(int max_tokens) {
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_, moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0)
        return true;  // no workspace needed

    shared_workspace_ = vram_alloc(vram_alloc_, max_shared, "shared_workspace");
    if (!shared_workspace_) {
        // Shared workspace is critical for GEMV scratch buffers. Fall back to
        // raw cudaMalloc bypassing headroom (Nemotron-30B leaves <headroom free).
        IMP_LOG_WARN("Shared workspace: allocator rejected (%.1f MiB), trying raw cudaMalloc",
                     max_shared / (1024.0 * 1024.0));
        cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate shared workspace (%.1f MiB)", max_shared / (1024.0 * 1024.0));
            return false;
        }
    }
    shared_workspace_size_ = max_shared;
    shared_workspace_max_tokens_ = max_tokens;

    IMP_LOG_INFO(
        "Shared workspace: %.2f MiB = max(attn=%.1f, ffn=%.1f, moe=%.1f, ssm=%.1f MiB) "
        "— saved %.2f MiB vs separate allocation",
        max_shared / (1024.0 * 1024.0), attn_shared_size_ / (1024.0 * 1024.0),
        ffn_shared_size_ / (1024.0 * 1024.0), moe_shared_size_ / (1024.0 * 1024.0),
        ssm_shared_size_ / (1024.0 * 1024.0),
        (attn_shared_size_ + ffn_shared_size_ + moe_shared_size_ + ssm_shared_size_ - max_shared) /
            (1024.0 * 1024.0));

    // Pre-allocate MoE routing buffers (separate from shared workspace)
    if (has_moe_) {
        const auto& cfg = model_->config();
        moe_.routing_buffers.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
    return true;
}

// allocate_auxiliary_buffers(), release_moe_batch_buf(), free_buffers()
// are in executor_workspace_buffers.cu

// pre_dequant_weights() is in executor_pre_dequant.cu
// configure_*_workspace(), resize_workspace(), allocate_decode_workspace(),
// use_workspace(), layer_has_*(), view_tokens(), ensure_logits_pinned()
// are in executor_workspace_config.cu

}  // namespace imp

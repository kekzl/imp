#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#ifdef IMP_USE_CUTLASS
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/attention_cutlass_fmha.h"
#endif
#include "compute/activation.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace imp {

// Helper: round up to 256-byte alignment (used throughout for workspace layout).
static inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// Helper: create a 2D Tensor view at `ptr` and advance ptr by `aligned_sz`.
// Used by configure_*_workspace() and allocate_persistent_workspace() to lay out
// tensors in a contiguous buffer.
static Tensor make_workspace_tensor(char*& ptr, DType dtype, int64_t rows, int64_t cols, size_t aligned_sz) {
    int64_t shape[2] = {rows, cols};
    Tensor t(ptr, dtype, 2, shape, true);
    ptr += aligned_sz;
    return t;
}

// Helper: allocate GPU memory through VRAMAllocator if available, else cudaMalloc.
// Frees through the same path. Used for large persistent allocations.
static void* vram_alloc(VRAMAllocator* alloc, size_t bytes, const char* tag) {
    if (bytes == 0) return nullptr;
    if (alloc) return alloc->allocate(bytes, tag);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
}

static void vram_free(VRAMAllocator* alloc, void* ptr) {
    if (!ptr) return;
    if (alloc) alloc->free(ptr);
    else IMP_CUDA_CHECK_LOG(cudaFree(ptr));
}

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() {
    free_buffers();
}

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl,
                         int max_batch_size, int max_seq_len, bool use_fp8_prefill,
                         int use_nvfp4_decode, bool use_mxfp4_prefill) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    norm_w_off_ = model.config().norm_weight_offset;
    use_pdl_ = use_pdl;
    wcache_.use_fp8 = use_fp8_prefill;
    wcache_.nvfp4_decode_mode = use_nvfp4_decode;
    wcache_.use_mxfp4 = use_mxfp4_prefill;

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
    has_gdn_ = false;  // detected from tensor presence below
    has_dense_ffn_ = (cfg.d_ff > 0);

    // Compute max expert FFN hidden dim from actual packed tensor shapes.
    // cfg.expert_d_ff may not match the actual tensor dimensions (e.g. Nemotron-H).
    max_expert_eff_ = cfg.expert_d_ff;
    if (has_moe_) {
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

    // Cap max_tokens for hybrid MoE+SSM/GDN models to limit workspace.
    // SSM state + cuBLAS S-matrix + workspace can exhaust 32 GB VRAM.
    if (has_ssm_ && (has_moe_ || has_gdn_)) {
        int capped = has_moe_ ? 256 : 512;  // MoE tighter, dense GDN can afford more
        if (max_tokens_ > capped) {
            IMP_LOG_INFO("executor_workspace.cu:%d: Capping max_tokens %d → %d for SSM/GDN hybrid",
                         __LINE__, max_tokens_, capped);
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

    // Precompute YaRN correction dimensions if enabled
    if (cfg.yarn_ext_factor > 0.0f) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int n_dims = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
        int n_ctx_orig = cfg.rope_n_ctx_orig > 0 ? cfg.rope_n_ctx_orig : cfg.max_seq_len;
        rope_yarn_corr_dims(n_dims, n_ctx_orig, cfg.rope_theta,
                            cfg.yarn_beta_fast, cfg.yarn_beta_slow, yarn_corr_dims_);
        IMP_LOG_INFO("YaRN corr_dims: [%.1f, %.1f] (n_dims=%d, n_ctx_orig=%d)",
                     yarn_corr_dims_[0], yarn_corr_dims_[1], n_dims, n_ctx_orig);
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
            long_freqs[i]  = base_freq / cfg.rope_long_factor[i];
        }

        cudaError_t e1 = cudaMalloc(&longrope_short_freqs_, pairs * sizeof(float));
        cudaError_t e2 = cudaMalloc(&longrope_long_freqs_,  pairs * sizeof(float));
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate LongRoPE frequency buffers: %s",
                          cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
            if (longrope_short_freqs_) { cudaFree(longrope_short_freqs_); longrope_short_freqs_ = nullptr; }
            if (longrope_long_freqs_)  { cudaFree(longrope_long_freqs_);  longrope_long_freqs_ = nullptr; }
            return false;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpy(longrope_short_freqs_, short_freqs.data(), pairs * sizeof(float), cudaMemcpyHostToDevice));
        IMP_CUDA_CHECK_LOG(cudaMemcpy(longrope_long_freqs_,  long_freqs.data(),  pairs * sizeof(float), cudaMemcpyHostToDevice));

        IMP_LOG_INFO("LongRoPE: %d freq pairs, orig_max_pos=%d", pairs, longrope_orig_max_pos_);
    }

    initialized_ = true;

    IMP_LOG_INFO("GraphExecutor initialized: max_tokens=%d, d_model=%d, "
                 "n_layers=%d, dtype=%s, pdl=%s",
                 max_tokens_, cfg.d_model, cfg.n_layers,
                 dtype_name(compute_dtype_),
                 use_pdl_ ? "on" : "off");
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2: allocate all GPU workspace buffers (called after weight upload)
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_workspaces(bool experts_on_host) {
    if (!initialized_ || !model_) return false;

    if (!allocate_persistent_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Persistent workspace allocation failed — cannot run inference");
        return false;
    }
    if (!allocate_shared_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Shared workspace allocation failed — cannot run inference");
        return false;
    }
    allocate_auxiliary_buffers(/*skip_batch_dequant=*/experts_on_host);

    return true;
}

size_t GraphExecutor::workspace_estimate() const {
    if (!model_) return 0;
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    size_t es = dtype_size(compute_dtype_);


    // Persistent: hidden + residual + norm_out + logits
    size_t persistent = 3 * align256(static_cast<size_t>(max_tokens_) * d * es)
                      + align256(static_cast<size_t>(max_logit_tokens_) * cfg.vocab_size * sizeof(float));

    // Shared: max of phases (already computed in compute_shared_sizes)
    size_t shared = std::max({attn_shared_size_, ffn_shared_size_,
                              moe_shared_size_, ssm_shared_size_});

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
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_elems = std::max(max_elems, static_cast<size_t>(w->numel()));
            }
        }
        auxiliary += max_elems * sizeof(uint16_t);
    }

    // Sampling result (ARGMAX_SCRATCH_BYTES ~16 KiB) + MMVQ scratch + split-K scratch
    int nh_est = cfg.n_heads;
    int hd_est = cfg.head_dim > 0 ? cfg.head_dim : (d / nh_est);
    auxiliary += 16 * 1024;  // sampling
    auxiliary += 256 * 1024;  // MMVQ scratch (conservative)
    auxiliary += static_cast<size_t>(max_logit_tokens_) * nh_est * 32 * (2 + hd_est) * sizeof(float);  // split-K

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

#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA workspace (LSE buffer + kernel cooperative workspace)
    // Skip for MoE models where prefill uses cuBLAS (compute-light attention)
    if (!is_moe) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        auxiliary += cutlass_fmha_workspace_estimate(1, max_tokens_, cfg.n_heads, hd);
    }
#endif

    return persistent + shared + fp32_accum + auxiliary;
}

// ---------------------------------------------------------------------------
// Unified workspace allocation
// ---------------------------------------------------------------------------

void GraphExecutor::compute_shared_sizes(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int ff  = cfg.d_ff;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);



    // Attention phase: q, k+v (contiguous for batched GEMM), attn_out, proj_out
    // Check for Q+Gate interleaving (Qwen3.5): Q projection output is 2x larger
    // than standard Q when an attention output gate is present.
    int max_q_out = nh * hd;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& ly = model_->layer(i);
        if (ly.wq.data) {
            int q_dim = static_cast<int>(ly.wq.shape[0]);
            if (q_dim > max_q_out) max_q_out = q_dim;
        }
    }
    size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q (de-interleaved)
                       + align256(2 * kv_raw)                                       // k+v contiguous
                       + align256(static_cast<size_t>(max_tokens) * nh * hd * es)   // attn_out
                       + align256(static_cast<size_t>(max_tokens) * d * es)         // proj_out
                       + (max_q_out > nh * hd
                          ? align256(static_cast<size_t>(max_tokens) * max_q_out * es)  // qv_full (Q+Gate)
                          : 0);

    // Dense FFN phase: gate, up, swiglu, ffn_out
    if (has_dense_ffn_ && ff > 0) {
        ffn_shared_size_ = align256(static_cast<size_t>(max_tokens) * ff * es)   // gate_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // up_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // swiglu_out
                          + align256(static_cast<size_t>(max_tokens) * d * es);  // ffn_out
    }

    // MoE phase
    if (has_moe_) {
        int ne    = cfg.n_experts;
        int top_k = cfg.n_experts_active;
        int eff   = max_expert_eff_;
        int expanded = max_tokens * top_k;

        moe_shared_size_ = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float))  // gate_logits
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
        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim * proj_elem_size) // proj (FP32 for GDN)
                          + align256(static_cast<size_t>(max_tokens) * conv_channels * es)   // xBC
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // y
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // z
                          + align256(static_cast<size_t>(max_tokens) * d * es)               // out
                          + align256(static_cast<size_t>(max_tokens) * n_heads * (has_gdn_ ? 2 : 1) * es); // dt (2x for GDN: alpha + beta)
    }
}

bool GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);



    size_t hidden_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz   = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    persistent_workspace_ = vram_alloc(vram_alloc_, total, "persistent_workspace");
    if (!persistent_workspace_) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%.1f MiB)",
                      total / (1024.0 * 1024.0));
        return false;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    hidden_   = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, hidden_sz);
    residual_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, residual_sz);
    norm_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, norm_out_sz);

    {
        int64_t shape[2] = {static_cast<int64_t>(max_logit_tokens_), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += logits_sz;
    }

    IMP_LOG_INFO("Persistent workspace: %.2f MiB (hidden+residual+norm+logits)",
                 total / (1024.0 * 1024.0));

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    if (cfg.norm_placement == NormPlacement::POST_NORM) {
        size_t fp32_sz = align256(static_cast<size_t>(max_tokens) * d * sizeof(float));
        cudaError_t e2 = cudaMalloc(&fp32_accum_buf_, fp32_sz);
        if (e2 == cudaSuccess) {
            int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(d)};
            fp32_hidden_ = Tensor(fp32_accum_buf_, DType::FP32, 2, shape, true);
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
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0) return true;  // no workspace needed

    shared_workspace_ = vram_alloc(vram_alloc_, max_shared, "shared_workspace");
    if (!shared_workspace_) {
        // Shared workspace is critical for GEMV scratch buffers. Fall back to
        // raw cudaMalloc bypassing headroom (Nemotron-30B leaves <headroom free).
        IMP_LOG_WARN("Shared workspace: allocator rejected (%.1f MiB), trying raw cudaMalloc",
                      max_shared / (1024.0 * 1024.0));
        cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate shared workspace (%.1f MiB)",
                          max_shared / (1024.0 * 1024.0));
            return false;
        }
    }
    shared_workspace_size_ = max_shared;
    shared_workspace_max_tokens_ = max_tokens;

    IMP_LOG_INFO("Shared workspace: %.2f MiB = max(attn=%.1f, ffn=%.1f, moe=%.1f, ssm=%.1f MiB) "
                 "— saved %.2f MiB vs separate allocation",
                 max_shared / (1024.0 * 1024.0),
                 attn_shared_size_ / (1024.0 * 1024.0),
                 ffn_shared_size_ / (1024.0 * 1024.0),
                 moe_shared_size_ / (1024.0 * 1024.0),
                 ssm_shared_size_ / (1024.0 * 1024.0),
                 (attn_shared_size_ + ffn_shared_size_ + moe_shared_size_ + ssm_shared_size_
                  - max_shared) / (1024.0 * 1024.0));

    // Pre-allocate MoE routing buffers (separate from shared workspace)
    if (has_moe_) {
        const auto& cfg = model_->config();
        moe_.routing_buffers.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
    return true;
}

void GraphExecutor::allocate_auxiliary_buffers(bool skip_batch_dequant) {
    const auto& cfg = model_->config();

    // Dequant scratch buffer for on-the-fly weight dequantization.
    {
        size_t max_weight_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_weight_elems = std::max(max_weight_elems,
                                                          static_cast<size_t>(w->numel()));
            }
        }
        if (max_weight_elems > 0) {
            qscratch_.dequant_size = max_weight_elems * sizeof(uint16_t);
            qscratch_.dequant = vram_alloc(vram_alloc_, qscratch_.dequant_size, "dequant_scratch");
            if (!qscratch_.dequant) {
                IMP_LOG_ERROR("Failed to allocate dequant scratch (%.1f MiB)",
                              qscratch_.dequant_size / (1024.0 * 1024.0));
                qscratch_.dequant_size = 0;
            } else {
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB",
                             qscratch_.dequant_size / (1024.0 * 1024.0));
            }
        }
    }

    // Sampling result buffer: sized to hold the argmax result plus the
    // multi-block partial reduction scratch (ARGMAX_SCRATCH_BYTES).
    {
        cudaError_t err = cudaMalloc(&d_sample_result_, ARGMAX_SCRATCH_BYTES);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate sampling result buffer: %s",
                          cudaGetErrorString(err));
            d_sample_result_ = nullptr;
        }
    }

    // Pinned host buffer for async sampling D2H copy (avoids stack-variable sync)
    if (!h_sample_pinned_ && d_sample_result_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample pinned buffer failed: %s",
                         cudaGetErrorString(err));
            h_sample_pinned_ = nullptr;
        }
    }

    // MMVQ (dp4a) scratch buffers for quantized input vectors.
    // Find the max Q8_1 block count needed across all uses:
    //   1. Dense GEMV: max_k / 32 blocks (one input vector)
    //   2. MoE down projection: top_k * expert_d_ff / 32 blocks (per-expert quantized activations)
    {
        int max_k = 0;
        int max_moe_down_blocks = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data && w->ndim >= 2) {
                    max_k = std::max(max_k, static_cast<int>(w->shape[1]));
                }
            }
            // MoE expert weight inner dims
            if (L.expert_up_packed.data && L.expert_up_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_up_packed.shape[2]));
            }
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3) {
                int down_k = static_cast<int>(L.expert_down_packed.shape[2]);
                max_k = std::max(max_k, down_k);
                // MoE down projection quantizes top_k expert activations contiguously
                max_moe_down_blocks = std::max(max_moe_down_blocks,
                    cfg.n_experts_active * (down_k / 32));
            }
            if (L.expert_gate_packed.data && L.expert_gate_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_gate_packed.shape[2]));
            }
        }
        int max_blocks = std::max(max_k / 32, max_moe_down_blocks);
        if (max_blocks > 0) {
            qscratch_.q8_1_max_blocks = max_blocks;
            size_t q8_1_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * sizeof(block_q8_1);
            size_t d8_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * sizeof(float);
            cudaError_t err1 = cudaMalloc(&qscratch_.q8_1_buf, q8_1_sz);
            cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d8_buf), d8_sz);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate MMVQ scratch buffers, dp4a path disabled");
                if (qscratch_.q8_1_buf) { cudaFree(qscratch_.q8_1_buf); qscratch_.q8_1_buf = nullptr; }
                if (qscratch_.d8_buf) { cudaFree(qscratch_.d8_buf); qscratch_.d8_buf = nullptr; }
                qscratch_.q8_1_max_blocks = 0;
            } else {
                IMP_LOG_INFO("MMVQ scratch buffers: %.2f KiB (q8_1) + %.2f KiB (d8), max_blocks=%d (max_k=%d, moe_down=%d)",
                             q8_1_sz / 1024.0, d8_sz / 1024.0, max_blocks, max_k, max_moe_down_blocks);
            }
        }
    }

    // Split-K paged attention scratch buffer.
    // Sized for max_batch_size * n_heads * max_splits * (2 + head_dim) floats.
    {
        int nh = cfg.n_heads;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
        // Size splits proportional to max context blocks, capped at 32
        int max_context_blocks = (max_tokens_ + kKVBlockSize - 1) / kKVBlockSize;
        int max_splits = std::min(32, std::max(1, max_context_blocks));
        int partial_stride = 2 + hd;
        int max_batch = max_logit_tokens_;  // = max_batch_size
        size_t sz = static_cast<size_t>(max_batch) * nh * max_splits * partial_stride * sizeof(float);
        cudaError_t err = cudaMalloc(&qscratch_.splitk, sz);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate split-K scratch (%zu bytes), split-K disabled", sz);
            qscratch_.splitk = nullptr;
            qscratch_.splitk_size = 0;
        } else {
            qscratch_.splitk_size = sz;
            IMP_LOG_INFO("Split-K paged attention scratch: %.2f KiB", sz / 1024.0);
        }
    }

    // cuBLAS attention S-matrix workspace: [n_heads, attn_seq, attn_seq] FP16
    // Used for prefill at medium sequence lengths (faster than WMMA flash attention
    // due to higher TC utilization in cuBLAS GEMM). Falls back to flash attention
    // for long sequences or when VRAM-constrained.
    if (!skip_batch_dequant) {
        int nh = cfg.n_heads;
        constexpr size_t kMaxAttnScoresMiB = 256;  // cap at 256 MiB
        size_t max_s_sz = kMaxAttnScoresMiB << 20;
        // max seq = sqrt(budget / (n_heads * sizeof(half)))
        int attn_seq = max_tokens_;
        size_t s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        if (s_sz > max_s_sz) {
            attn_seq = static_cast<int>(std::sqrt(
                static_cast<double>(max_s_sz) / (nh * sizeof(half))));
            attn_seq = (attn_seq / 16) * 16;  // round down to multiple of 16
            if (attn_seq < 32) attn_seq = 0;  // too small to be useful
            s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        }
        if (attn_seq > 0) {
            attn_scores_buf_ = vram_alloc(vram_alloc_, s_sz, "attn_scores");
            if (!attn_scores_buf_) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("Failed to allocate cuBLAS attention S-matrix (%.1f MiB): %s — "
                             "will fall back to WMMA attention for prefill",
                             s_sz / (1024.0 * 1024.0), cudaGetErrorString(e));
                attn_scores_buf_size_ = 0;
            } else {
                attn_scores_buf_size_ = s_sz;
                int64_t s_shape[3] = {static_cast<int64_t>(nh),
                                      static_cast<int64_t>(attn_seq),
                                      static_cast<int64_t>(attn_seq)};
                attn_scores_ = Tensor(attn_scores_buf_, DType::FP16, 3, s_shape, true);
                IMP_LOG_INFO("cuBLAS attention S-matrix: %.2f MiB (%d heads x %d x %d)",
                             s_sz / (1024.0 * 1024.0), nh, attn_seq, attn_seq);
            }
        }
    } else {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (VRAM-constrained, using WMMA/TCGEN05 fallback)");
    }

#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA workspace: pre-allocate LSE + kernel workspace at max dimensions.
    // This ensures the allocations are tracked in the VRAM budget instead of happening
    // lazily (which would cause untracked VRAM growth and potential shared memory swapping).
    {
        int fmha_nh = cfg.n_heads;
        int fmha_hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / fmha_nh);
        size_t fmha_bytes = cutlass_fmha_init_workspace(1, max_tokens_, fmha_nh, fmha_hd);
        if (fmha_bytes > 0) {
            IMP_LOG_INFO("CUTLASS FMHA workspace: %.2f MiB (LSE + kernel)",
                         fmha_bytes / (1024.0 * 1024.0));
        }
    }
#endif

    // MoE dequant and staging buffers
    if (has_moe_) {
        int d   = cfg.d_model;
        int eff = max_expert_eff_;

        // Dequant buffer: 1 expert slot
        {
            size_t expert_fp16_elems = static_cast<size_t>(eff) * d;
            size_t dequant_sz = expert_fp16_elems * sizeof(uint16_t);
            moe_.dequant_buf = vram_alloc(vram_alloc_, dequant_sz, "moe_dequant");
            if (!moe_.dequant_buf) {
                IMP_LOG_ERROR("Failed to allocate MoE dequant buffer (%zu bytes)", dequant_sz);
                moe_.dequant_buf_size = 0;
            } else {
                moe_.dequant_buf_size = dequant_sz;
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)",
                             dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer
        size_t max_expert_raw = 0;
        {
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                auto check = [&](const Tensor& p, GGMLQuantType qt) {
                    if (!p.data || p.ndim < 3) return;
                    size_t rb = ggml_quant_row_bytes(qt, p.shape[2]);
                    size_t expert_raw = static_cast<size_t>(p.shape[1]) * rb;
                    max_expert_raw = std::max(max_expert_raw, expert_raw);
                };
                check(L.expert_up_packed, L.expert_up_qtype);
                check(L.expert_down_packed, L.expert_down_qtype);
                check(L.expert_gate_packed, L.expert_gate_qtype);
            }
            if (max_expert_raw > 0) {
                moe_.raw_staging_buf = vram_alloc(vram_alloc_, max_expert_raw, "moe_staging");
                if (!moe_.raw_staging_buf) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes)", max_expert_raw);
                    moe_.raw_staging_size = 0;
                } else {
                    moe_.raw_staging_size = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
                }
            }
        }

        // LRU expert cache: keeps recently-used host experts on GPU.
        // Only allocated when some experts reside on host (not all fit in VRAM).
        if (max_expert_raw > 0) {
            bool has_host_experts = false;
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                if ((L.expert_up_packed.data && !L.expert_up_packed.on_device) ||
                    (L.expert_down_packed.data && !L.expert_down_packed.on_device) ||
                    (L.expert_gate_packed.data && !L.expert_gate_packed.on_device)) {
                    has_host_experts = true;
                    break;
                }
            }
            if (has_host_experts) {
                // Budget: proportional to free VRAM (15%) instead of flat cap.
                // KV cache + weight caches (FP8/NVFP4) need the remaining VRAM,
                // so expert cache must not over-commit.
                size_t free_mem = 0, total_mem = 0;
                IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
                size_t safety = 128 << 20;  // 128 MiB reserve
                size_t budget = (free_mem > safety) ? free_mem - safety : 0;
                budget = static_cast<size_t>(budget * 0.15);  // 15% of available
                if (expert_cache_.init(max_expert_raw, budget, vram_alloc_)) {
                    IMP_LOG_INFO("Expert LRU cache: %d slots (%.2f MiB / %.2f MiB budget)",
                                 expert_cache_.n_slots_,
                                 expert_cache_.n_slots_ * max_expert_raw / (1024.0 * 1024.0),
                                 budget / (1024.0 * 1024.0));
                }
            }
        }

        // Batch dequant buffer: sized for a chunk of experts (L2-resident strategy).
        // We dequant a chunk of experts to FP16, then immediately GEMM while the
        // FP16 data is still warm in L2 cache (~96 MB on RTX 5090). This avoids
        // writing the FP16 intermediate to DRAM entirely, saving ~5x DRAM traffic.
        // Skip allocation if experts are on host (batch dequant only useful for on-device experts).
        if (!skip_batch_dequant) {
            int targets[] = {cfg.n_experts, cfg.n_experts / 2, 32, 16};
            bool allocated = false;
            for (int ne_try : targets) {
                if (ne_try <= 0) continue;
                ne_try = std::min(ne_try, cfg.n_experts);
                size_t sz = static_cast<size_t>(ne_try) * eff * d * sizeof(half);
                moe_.batch_dequant_buf = vram_alloc(vram_alloc_, sz, "moe_batch_dequant");
                if (!moe_.batch_dequant_buf) {
                    IMP_LOG_DEBUG("MoE dequant buf alloc failed for %d experts", ne_try);
                    continue;
                }
                moe_.batch_dequant_buf_size = sz;
                allocated = true;
                IMP_LOG_INFO("MoE batch dequant buffer: %.2f MiB (%d experts)",
                             sz / (1024.0 * 1024.0), ne_try);
                break;
            }
            if (!allocated) {
                IMP_LOG_INFO("MoE batch dequant buffer: skipped (VRAM insufficient)");
                moe_.batch_dequant_buf = nullptr;
                moe_.batch_dequant_buf_size = 0;
            }
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
        }

        // Pre-allocated device pointer arrays for batched MoE GEMM.
        // 3 arrays × n_experts void pointers = trivial memory (< 4 KB).
        // Eliminates cudaMallocAsync/FreeAsync from the hot path.
        if (cfg.n_experts > 0) {
            size_t ptr_bytes = 3 * static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            cudaError_t err = cudaMalloc(&moe_.d_work_ptrs, ptr_bytes);
            if (err == cudaSuccess) {
                moe_.d_work_ptrs_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE work ptrs alloc failed: %s", cudaGetErrorString(err));
                moe_.d_work_ptrs = nullptr;
                moe_.d_work_ptrs_count = 0;
            }

            // Per-expert FP8 scale buffer (trivial: 128 experts × 4 bytes = 512 bytes).
            size_t scale_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            err = cudaMalloc(&moe_.d_fp8_scales, scale_bytes);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE FP8 scales alloc failed: %s", cudaGetErrorString(err));
                moe_.d_fp8_scales = nullptr;
            }

            // Device-side weight pointer array for device-grouped GEMM.
            size_t wptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            err = cudaMalloc(&moe_.d_weight_ptrs, wptr_bytes);
            if (err == cudaSuccess) {
                moe_.d_weight_ptrs_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE weight ptrs alloc failed: %s", cudaGetErrorString(err));
                moe_.d_weight_ptrs = nullptr;
                moe_.d_weight_ptrs_count = 0;
            }
        }
    }

    // FP8 activation scratch buffers (for FP8 prefill weight cache)
    if (wcache_.use_fp8) {
        int max_dim = cfg.d_model;
        if (cfg.d_ff > 0) max_dim = std::max(max_dim, cfg.d_ff);
        max_dim = std::max(max_dim, cfg.n_heads * (cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads)));
        // SSM dimensions
        if (cfg.ssm_inner_size > 0) {
            int conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
            int ssm_in_dim = cfg.ssm_inner_size + conv_ch + cfg.ssm_dt_rank;
            max_dim = std::max(max_dim, ssm_in_dim);
            max_dim = std::max(max_dim, cfg.ssm_inner_size);
        }
        qscratch_.fp8_act_size = static_cast<size_t>(max_tokens_) * max_dim;
        qscratch_.fp8_act = vram_alloc(vram_alloc_, qscratch_.fp8_act_size, "fp8_activation");
        if (!qscratch_.fp8_act) {
            IMP_LOG_WARN("Failed to allocate FP8 activation buffer (%.1f MiB)",
                         qscratch_.fp8_act_size / (1024.0 * 1024.0));
            qscratch_.fp8_act_size = 0;
        }
        {
            cudaError_t serr = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d_act_scale), sizeof(float));
            if (serr != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate FP8 act scale: %s", cudaGetErrorString(serr));
                qscratch_.d_act_scale = nullptr;
            }
        }
        // Pre-allocate reduction buffers for async FP8 activation quantization.
        // Eliminates per-call cudaMalloc + cudaStreamSynchronize from the hot path.
        if (qscratch_.fp8_act && qscratch_.d_act_scale) {
            int max_n = static_cast<int>(qscratch_.fp8_act_size);  // max elements
            int threads_needed = (max_n + 3) / 4;  // kElemsPerThread=4
            qscratch_.fp8_max_grid = (threads_needed + 255) / 256;  // kBlockSize=256
            cudaError_t e1 = cudaMalloc(&qscratch_.d_fp8_block_maxes, static_cast<size_t>(qscratch_.fp8_max_grid) * sizeof(float));
            cudaError_t e2 = cudaMalloc(&qscratch_.d_fp8_absmax, sizeof(float));
            if (e1 != cudaSuccess || e2 != cudaSuccess || !qscratch_.d_fp8_block_maxes || !qscratch_.d_fp8_absmax) {
                IMP_LOG_WARN("Failed to allocate FP8 reduction buffers — will use sync path");
                if (qscratch_.d_fp8_block_maxes) { cudaFree(qscratch_.d_fp8_block_maxes); qscratch_.d_fp8_block_maxes = nullptr; }
                if (qscratch_.d_fp8_absmax) { cudaFree(qscratch_.d_fp8_absmax); qscratch_.d_fp8_absmax = nullptr; }
                qscratch_.fp8_max_grid = 0;
            }
            IMP_LOG_INFO("FP8 activation scratch: %.2f MiB (max_tokens=%d, max_dim=%d, async reduction grid=%d)",
                         qscratch_.fp8_act_size / (1024.0 * 1024.0), max_tokens_, max_dim, qscratch_.fp8_max_grid);
        }
    }

    // CUTLASS sm_120 NVFP4 activation buffers: pre-allocate for max prefill dimensions.
    // Only needed when NVFP4 decode is active and sm_120 is available.
    if (wcache_.nvfp4_decode_mode > 0 && cutlass_sm120_nvfp4_available()) {
        int max_k = 0;
        int max_n = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data && w->ndim >= 2) {
                    max_n = std::max(max_n, static_cast<int>(w->shape[0]));
                    max_k = std::max(max_k, static_cast<int>(w->shape[1]));
                }
            }
        }
        if (max_k > 0) {
            // Activation packed data: [max_tokens, max_K/2]
            qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * max_k / 2;
            // SfAtom scale factors for activation
            qscratch_.cutlass_act_sf_size = cutlass_nvfp4_sf_size(max_tokens_, max_k);
            // CUTLASS GEMM workspace
            qscratch_.cutlass_workspace_size = gemm_nvfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);

            qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size, "cutlass_act_data");
            qscratch_.cutlass_act_sf = vram_alloc(vram_alloc_, qscratch_.cutlass_act_sf_size, "cutlass_act_sf");
            qscratch_.cutlass_workspace = (qscratch_.cutlass_workspace_size > 0)
                               ? vram_alloc(vram_alloc_, qscratch_.cutlass_workspace_size, "cutlass_workspace")
                               : nullptr;
            if (!qscratch_.cutlass_act_data || !qscratch_.cutlass_act_sf ||
                (qscratch_.cutlass_workspace_size > 0 && !qscratch_.cutlass_workspace)) {
                IMP_LOG_WARN("Failed to allocate CUTLASS NVFP4 activation buffers, native FP4 prefill disabled");
                if (qscratch_.cutlass_act_data) { vram_free(vram_alloc_, qscratch_.cutlass_act_data); qscratch_.cutlass_act_data = nullptr; }
                if (qscratch_.cutlass_act_sf) { vram_free(vram_alloc_, qscratch_.cutlass_act_sf); qscratch_.cutlass_act_sf = nullptr; }
                if (qscratch_.cutlass_workspace) { vram_free(vram_alloc_, qscratch_.cutlass_workspace); qscratch_.cutlass_workspace = nullptr; }
                qscratch_.cutlass_act_data_size = 0;
                qscratch_.cutlass_act_sf_size = 0;
                qscratch_.cutlass_workspace_size = 0;
            } else {
                IMP_LOG_INFO("CUTLASS NVFP4 activation scratch: %.2f MiB (data=%.2f, sf=%.2f, ws=%.2f)",
                             (qscratch_.cutlass_act_data_size + qscratch_.cutlass_act_sf_size + qscratch_.cutlass_workspace_size) / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_data_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_sf_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_workspace_size / (1024.0 * 1024.0));

                // MXFP4 activation buffers: shares packed data with NVFP4, only needs
                // separate UE8M0 scale factors (SFVecSize=32 vs NVFP4's 16).
                if (cutlass_sm120_mxfp4_available()) {
                    qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                    qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);
                    qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
                    qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                                     ? vram_alloc(vram_alloc_, qscratch_.mxfp4_workspace_size, "mxfp4_workspace")
                                     : nullptr;
                    if (!qscratch_.mxfp4_act_sf ||
                        (qscratch_.mxfp4_workspace_size > 0 && !qscratch_.mxfp4_workspace)) {
                        IMP_LOG_WARN("Failed to allocate MXFP4 activation buffers, MXFP4 prefill disabled");
                        if (qscratch_.mxfp4_act_sf) { vram_free(vram_alloc_, qscratch_.mxfp4_act_sf); qscratch_.mxfp4_act_sf = nullptr; }
                        if (qscratch_.mxfp4_workspace) { vram_free(vram_alloc_, qscratch_.mxfp4_workspace); qscratch_.mxfp4_workspace = nullptr; }
                        qscratch_.mxfp4_act_sf_size = 0;
                        qscratch_.mxfp4_workspace_size = 0;
                    } else {
                        IMP_LOG_INFO("CUTLASS MXFP4 activation scratch: sf=%.2f MiB, ws=%.2f MiB",
                                     qscratch_.mxfp4_act_sf_size / (1024.0 * 1024.0),
                                     qscratch_.mxfp4_workspace_size / (1024.0 * 1024.0));
                    }
                }
            }
        }
    }
}

void GraphExecutor::release_moe_batch_buf() {
    if (moe_.batch_dequant_buf) {
        size_t freed = moe_.batch_dequant_buf_size;
        vram_free(vram_alloc_, moe_.batch_dequant_buf);
        moe_.batch_dequant_buf = nullptr;
        moe_.batch_dequant_buf_size = 0;
        IMP_LOG_INFO("Released MoE batch dequant buffer: %.2f MiB (experts on host)",
                     freed / (1024.0 * 1024.0));
    }
}

void GraphExecutor::free_buffers() {
    // Helper: free through VRAMAllocator if pointer was tracked, else cudaFree.
    auto vfree = [this](void*& p) {
        if (p) { vram_free(vram_alloc_, p); p = nullptr; }
    };

    // Free TurboQuant QJL projection
    qjl_destroy(qjl_proj_);

    // Free LongRoPE frequency tables
    if (longrope_short_freqs_) { IMP_CUDA_CHECK_LOG(cudaFree(longrope_short_freqs_)); longrope_short_freqs_ = nullptr; }
    if (longrope_long_freqs_)  { IMP_CUDA_CHECK_LOG(cudaFree(longrope_long_freqs_));  longrope_long_freqs_  = nullptr; }
    longrope_n_pairs_ = 0;
    longrope_orig_max_pos_ = 0;

    // Free all weight caches (FP16, FP8, NVFP4, CUTLASS, fused KV/gate+up, migrated/overflow)
    wcache_.free(vram_alloc_);

    qscratch_.free(vram_alloc_);

    moe_.free(vram_alloc_);
    expert_cache_.destroy();
    if (d_sample_result_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_sample_result_));
        d_sample_result_ = nullptr;
    }
    if (h_sample_pinned_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_sample_pinned_));
        h_sample_pinned_ = nullptr;
    }
    if (h_logits_pinned_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_logits_pinned_));
        h_logits_pinned_ = nullptr;
        h_logits_pinned_size_ = 0;
    }
    vfree(attn_scores_buf_);
    attn_scores_buf_size_ = 0;
#ifdef IMP_USE_CUTLASS
    cutlass_fmha_free_workspace();
#endif
    vfree(shared_workspace_);
    shared_workspace_size_ = 0;
    vfree(persistent_workspace_);
    persistent_workspace_size_ = 0;
    vfree(fp32_accum_buf_);
    ssm_layer_map_.clear();
    initialized_ = false;
}

// pre_dequant_weights() is in executor_pre_dequant.cu
// configure_*_workspace(), resize_workspace(), allocate_decode_workspace(),
// use_workspace(), layer_has_*(), view_tokens(), ensure_logits_pinned()
// are in executor_workspace_config.cu

} // namespace imp

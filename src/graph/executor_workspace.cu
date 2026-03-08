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
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "compute/gemm_cublaslt_nvfp4.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
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

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() {
    free_buffers();
}

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl,
                         int max_batch_size, int max_seq_len, bool use_fp8_prefill,
                         int use_nvfp4_decode) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    norm_w_off_ = model.config().norm_weight_offset;
    use_pdl_ = use_pdl;
    use_fp8_cache_ = use_fp8_prefill;
    use_nvfp4_decode_ = use_nvfp4_decode;

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
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

        cudaMalloc(&longrope_short_freqs_, pairs * sizeof(float));
        cudaMalloc(&longrope_long_freqs_,  pairs * sizeof(float));
        cudaMemcpy(longrope_short_freqs_, short_freqs.data(), pairs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(longrope_long_freqs_,  long_freqs.data(),  pairs * sizeof(float), cudaMemcpyHostToDevice);

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

    allocate_persistent_workspace(max_tokens_);
    allocate_shared_workspace(max_tokens_);
    allocate_auxiliary_buffers(/*skip_batch_dequant=*/experts_on_host);

    return true;
}

size_t GraphExecutor::workspace_estimate() const {
    if (!model_) return 0;
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    size_t es = dtype_size(compute_dtype_);
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

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

    // Auxiliary: dequant scratch, MoE dequant/staging, sampling, split-K, etc.
    size_t auxiliary = 64ULL << 20;  // conservative 64 MiB for misc buffers

#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA workspace (LSE buffer + kernel cooperative workspace)
    int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
    auxiliary += cutlass_fmha_workspace_estimate(1, max_tokens_, cfg.n_heads, hd);
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

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Attention phase: q, k+v (contiguous for batched GEMM), attn_out, proj_out
    size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q
                       + align256(2 * kv_raw)                                       // k+v contiguous
                       + align256(static_cast<size_t>(max_tokens) * nh * hd * es)   // attn_out
                       + align256(static_cast<size_t>(max_tokens) * d * es);        // proj_out

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

        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es)       // proj
                          + align256(static_cast<size_t>(max_tokens) * conv_channels * es)   // xBC
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // y
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // z
                          + align256(static_cast<size_t>(max_tokens) * d * es)               // out
                          + align256(static_cast<size_t>(max_tokens) * n_heads * es);        // dt
    }
}

void GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t hidden_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz   = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    cudaError_t err = cudaMalloc(&persistent_workspace_, total);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%zu bytes): %s",
                      total, cudaGetErrorString(err));
        return;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    hidden_   = make(d, hidden_sz);
    residual_ = make(d, residual_sz);
    norm_out_ = make(d, norm_out_sz);

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
}

void GraphExecutor::allocate_shared_workspace(int max_tokens) {
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0) return;

    cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate shared workspace (%zu bytes): %s",
                      max_shared, cudaGetErrorString(err));
        return;
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
        moe_routing_buffers_.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
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
            dequant_scratch_size_ = max_weight_elems * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&dequant_scratch_, dequant_scratch_size_);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate dequant scratch (%zu bytes): %s",
                              dequant_scratch_size_, cudaGetErrorString(err));
                dequant_scratch_ = nullptr;
                dequant_scratch_size_ = 0;
            } else {
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB",
                             dequant_scratch_size_ / (1024.0 * 1024.0));
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
            q8_1_max_blocks_ = max_blocks;
            size_t q8_1_sz = static_cast<size_t>(q8_1_max_blocks_) * sizeof(block_q8_1);
            size_t d8_sz = static_cast<size_t>(q8_1_max_blocks_) * sizeof(float);
            cudaError_t err1 = cudaMalloc(&q8_1_buf_, q8_1_sz);
            cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&d8_buf_), d8_sz);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate MMVQ scratch buffers, dp4a path disabled");
                if (q8_1_buf_) { cudaFree(q8_1_buf_); q8_1_buf_ = nullptr; }
                if (d8_buf_) { cudaFree(d8_buf_); d8_buf_ = nullptr; }
                q8_1_max_blocks_ = 0;
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
        int max_splits = 32;
        int partial_stride = 2 + hd;
        int max_batch = max_logit_tokens_;  // = max_batch_size
        size_t sz = static_cast<size_t>(max_batch) * nh * max_splits * partial_stride * sizeof(float);
        cudaError_t err = cudaMalloc(&splitk_scratch_, sz);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate split-K scratch (%zu bytes), split-K disabled", sz);
            splitk_scratch_ = nullptr;
            splitk_scratch_size_ = 0;
        } else {
            splitk_scratch_size_ = sz;
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
            cudaError_t err = cudaMalloc(&attn_scores_buf_, s_sz);
            if (err != cudaSuccess) {
                cudaGetLastError();  // clear sticky error from failed cudaMalloc
                IMP_LOG_WARN("Failed to allocate cuBLAS attention S-matrix (%zu bytes, %.1f MiB), "
                             "will fall back to WMMA attention for prefill",
                             s_sz, s_sz / (1024.0 * 1024.0));
                attn_scores_buf_ = nullptr;
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
            cudaError_t err = cudaMalloc(&moe_dequant_buf_, dequant_sz);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate MoE dequant buffer (%zu bytes): %s",
                              dequant_sz, cudaGetErrorString(err));
                moe_dequant_buf_ = nullptr;
                moe_dequant_buf_size_ = 0;
            } else {
                moe_dequant_buf_size_ = dequant_sz;
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)",
                             dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer
        {
            size_t max_expert_raw = 0;
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
                cudaError_t err = cudaMalloc(&moe_raw_staging_buf_, max_expert_raw);
                if (err != cudaSuccess) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes): %s",
                                  max_expert_raw, cudaGetErrorString(err));
                    moe_raw_staging_buf_ = nullptr;
                    moe_raw_staging_size_ = 0;
                } else {
                    moe_raw_staging_size_ = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
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
                cudaError_t err = cudaMalloc(&moe_batch_dequant_buf_, sz);
                if (err != cudaSuccess) {
                    IMP_LOG_DEBUG("MoE dequant buf alloc failed for %d experts: %s",
                                 ne_try, cudaGetErrorString(cudaGetLastError()));
                    continue;
                }
                moe_batch_dequant_buf_size_ = sz;
                allocated = true;
                IMP_LOG_INFO("MoE batch dequant buffer: %.2f MiB (%d experts)",
                             sz / (1024.0 * 1024.0), ne_try);
                break;
            }
            if (!allocated) {
                IMP_LOG_INFO("MoE batch dequant buffer: skipped (VRAM insufficient)");
                moe_batch_dequant_buf_ = nullptr;
                moe_batch_dequant_buf_size_ = 0;
            }
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_batch_dequant_buf_ = nullptr;
            moe_batch_dequant_buf_size_ = 0;
        }

        // Pre-allocated device pointer arrays for batched MoE GEMM.
        // 3 arrays × n_experts void pointers = trivial memory (< 4 KB).
        // Eliminates cudaMallocAsync/FreeAsync from the hot path.
        if (cfg.n_experts > 0) {
            size_t ptr_bytes = 3 * static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            cudaError_t err = cudaMalloc(&d_moe_work_ptrs_, ptr_bytes);
            if (err == cudaSuccess) {
                d_moe_work_ptrs_count_ = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Cleared optional MoE work ptrs alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_work_ptrs_ = nullptr;
                d_moe_work_ptrs_count_ = 0;
            }

            // Per-expert FP8 scale buffer (trivial: 128 experts × 4 bytes = 512 bytes).
            size_t scale_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            err = cudaMalloc(&d_moe_fp8_scales_, scale_bytes);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Cleared optional MoE FP8 scales alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_fp8_scales_ = nullptr;
            }

            // Device-side weight pointer array for device-grouped GEMM.
            size_t wptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            err = cudaMalloc(&d_moe_weight_ptrs_, wptr_bytes);
            if (err == cudaSuccess) {
                d_moe_weight_ptrs_count_ = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Cleared optional MoE weight ptrs alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_weight_ptrs_ = nullptr;
                d_moe_weight_ptrs_count_ = 0;
            }
        }
    }

    // FP8 activation scratch buffers (for FP8 prefill weight cache)
    if (use_fp8_cache_) {
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
        fp8_act_buf_size_ = static_cast<size_t>(max_tokens_) * max_dim;
        cudaError_t err = cudaMalloc(&fp8_act_buf_, fp8_act_buf_size_);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate FP8 activation buffer (%zu bytes): %s",
                         fp8_act_buf_size_, cudaGetErrorString(err));
            fp8_act_buf_ = nullptr;
            fp8_act_buf_size_ = 0;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&d_act_scale_), sizeof(float));
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate FP8 act scale: %s", cudaGetErrorString(err));
            d_act_scale_ = nullptr;
        }
        if (fp8_act_buf_ && d_act_scale_) {
            IMP_LOG_INFO("FP8 activation scratch: %.2f MiB (max_tokens=%d, max_dim=%d)",
                         fp8_act_buf_size_ / (1024.0 * 1024.0), max_tokens_, max_dim);
        }
    }

    // CUTLASS sm_120 NVFP4 activation buffers: pre-allocate for max prefill dimensions.
    // Only needed when NVFP4 decode is active and sm_120 is available.
    if (use_nvfp4_decode_ > 0 && cutlass_sm120_nvfp4_available()) {
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
            cutlass_act_data_size_ = static_cast<size_t>(max_tokens_) * max_k / 2;
            // SfAtom scale factors for activation
            cutlass_act_sf_size_ = cutlass_nvfp4_sf_size(max_tokens_, max_k);
            // CUTLASS GEMM workspace
            cutlass_workspace_size_ = gemm_nvfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);

            cudaError_t err1 = cudaMalloc(&cutlass_act_data_, cutlass_act_data_size_);
            cudaError_t err2 = cudaMalloc(&cutlass_act_sf_, cutlass_act_sf_size_);
            cudaError_t err3 = (cutlass_workspace_size_ > 0)
                               ? cudaMalloc(&cutlass_workspace_, cutlass_workspace_size_)
                               : cudaSuccess;
            if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate CUTLASS NVFP4 activation buffers, native FP4 prefill disabled");
                if (cutlass_act_data_) { cudaFree(cutlass_act_data_); cutlass_act_data_ = nullptr; }
                if (cutlass_act_sf_) { cudaFree(cutlass_act_sf_); cutlass_act_sf_ = nullptr; }
                if (cutlass_workspace_) { cudaFree(cutlass_workspace_); cutlass_workspace_ = nullptr; }
                cutlass_act_data_size_ = 0;
                cutlass_act_sf_size_ = 0;
                cutlass_workspace_size_ = 0;
                cudaGetLastError();  // clear sticky error
            } else {
                IMP_LOG_INFO("CUTLASS NVFP4 activation scratch: %.2f MiB (data=%.2f, sf=%.2f, ws=%.2f)",
                             (cutlass_act_data_size_ + cutlass_act_sf_size_ + cutlass_workspace_size_) / (1024.0 * 1024.0),
                             cutlass_act_data_size_ / (1024.0 * 1024.0),
                             cutlass_act_sf_size_ / (1024.0 * 1024.0),
                             cutlass_workspace_size_ / (1024.0 * 1024.0));
            }
        }
    }
}

void GraphExecutor::release_moe_batch_buf() {
    if (moe_batch_dequant_buf_) {
        size_t freed = moe_batch_dequant_buf_size_;
        cudaFree(moe_batch_dequant_buf_);
        moe_batch_dequant_buf_ = nullptr;
        moe_batch_dequant_buf_size_ = 0;
        IMP_LOG_INFO("Released MoE batch dequant buffer: %.2f MiB (experts on host)",
                     freed / (1024.0 * 1024.0));
    }
}

void GraphExecutor::free_buffers() {
    // Free LongRoPE frequency tables
    if (longrope_short_freqs_) { cudaFree(longrope_short_freqs_); longrope_short_freqs_ = nullptr; }
    if (longrope_long_freqs_)  { cudaFree(longrope_long_freqs_);  longrope_long_freqs_  = nullptr; }
    longrope_n_pairs_ = 0;
    longrope_orig_max_pos_ = 0;

    // Free fused KV weight cache
    for (auto& [idx, tensor] : fused_kv_cache_) {
        if (tensor.data) cudaFree(tensor.data);
    }
    fused_kv_cache_.clear();

    // Free fused gate+up weight cache
    for (auto& [idx, tensor] : fused_gate_up_cache_) {
        if (tensor.data) cudaFree(tensor.data);
    }
    fused_gate_up_cache_.clear();

    // Free FP16 weight cache
    for (auto& [ptr, tensor] : fp16_cache_) {
        cudaFree(tensor.data);
    }
    fp16_cache_.clear();
    fp16_cache_bytes_ = 0;

    // Free NVFP4 decode weight cache
    for (auto& [ptr, result] : nvfp4_cache_) {
        free_nvfp4_result(result);
    }
    nvfp4_cache_.clear();
    nvfp4_cache_bytes_ = 0;

    // Free NVFP4 MoE expert weight cache
    for (auto& [ptr, result] : nvfp4_moe_cache_) {
        free_nvfp4_moe_result(result);
    }
    nvfp4_moe_cache_.clear();
    nvfp4_moe_cache_bytes_ = 0;

    // Free CUTLASS sm_120 NVFP4 weight cache
    for (auto& [ptr, cw] : cutlass_nvfp4_cache_) {
        free_cutlass_nvfp4_weight(cw);
    }
    cutlass_nvfp4_cache_.clear();
    cutlass_nvfp4_cache_bytes_ = 0;

    // Free CUTLASS NVFP4 activation buffers
    if (cutlass_act_data_) {
        cudaFree(cutlass_act_data_);
        cutlass_act_data_ = nullptr;
        cutlass_act_data_size_ = 0;
    }
    if (cutlass_act_sf_) {
        cudaFree(cutlass_act_sf_);
        cutlass_act_sf_ = nullptr;
        cutlass_act_sf_size_ = 0;
    }
    if (cutlass_workspace_) {
        cudaFree(cutlass_workspace_);
        cutlass_workspace_ = nullptr;
        cutlass_workspace_size_ = 0;
    }

    // Free FP8 weight cache
    for (auto& [ptr, entry] : fp8_cache_) {
        if (entry.weight.data) {
            // Check if data pointer is inside a bulk buffer
            bool in_migrated_data = fp8_migrated_data_ &&
                reinterpret_cast<uintptr_t>(entry.weight.data) >= reinterpret_cast<uintptr_t>(fp8_migrated_data_) &&
                reinterpret_cast<uintptr_t>(entry.weight.data) < reinterpret_cast<uintptr_t>(fp8_migrated_data_) + fp8_migrated_data_size_;
            bool in_overflow_data = fp8_overflow_data_ &&
                reinterpret_cast<uintptr_t>(entry.weight.data) >= reinterpret_cast<uintptr_t>(fp8_overflow_data_) &&
                reinterpret_cast<uintptr_t>(entry.weight.data) < reinterpret_cast<uintptr_t>(fp8_overflow_data_) + fp8_overflow_data_size_;
            if (!in_migrated_data && !in_overflow_data) cudaFree(entry.weight.data);
        }
        // d_scale pointers from batched paths point into bulk buffers (freed below)
        if (entry.d_scale) {
            bool in_migrated = fp8_migrated_scales_ &&
                               entry.d_scale >= fp8_migrated_scales_ &&
                               entry.d_scale < fp8_migrated_scales_ + fp8_migrated_count_;
            bool in_overflow = fp8_overflow_scales_ &&
                               entry.d_scale >= fp8_overflow_scales_ &&
                               entry.d_scale < fp8_overflow_scales_ + fp8_overflow_count_;
            if (!in_migrated && !in_overflow) cudaFree(entry.d_scale);
        }
    }
    fp8_cache_.clear();
    fp8_cache_bytes_ = 0;
    if (fp8_migrated_scales_) {
        cudaFree(fp8_migrated_scales_);
        fp8_migrated_scales_ = nullptr;
        fp8_migrated_count_ = 0;
    }
    if (fp8_migrated_data_) {
        cudaFree(fp8_migrated_data_);
        fp8_migrated_data_ = nullptr;
        fp8_migrated_data_size_ = 0;
    }
    if (fp8_overflow_scales_) {
        cudaFree(fp8_overflow_scales_);
        fp8_overflow_scales_ = nullptr;
        fp8_overflow_count_ = 0;
    }
    if (fp8_overflow_data_) {
        cudaFree(fp8_overflow_data_);
        fp8_overflow_data_ = nullptr;
        fp8_overflow_data_size_ = 0;
    }
    if (fp8_act_buf_) {
        cudaFree(fp8_act_buf_);
        fp8_act_buf_ = nullptr;
        fp8_act_buf_size_ = 0;
    }
    if (d_act_scale_) {
        cudaFree(d_act_scale_);
        d_act_scale_ = nullptr;
    }

    moe_routing_buffers_.free();
    if (moe_dequant_buf_) {
        cudaFree(moe_dequant_buf_);
        moe_dequant_buf_ = nullptr;
        moe_dequant_buf_size_ = 0;
    }
    if (moe_raw_staging_buf_) {
        cudaFree(moe_raw_staging_buf_);
        moe_raw_staging_buf_ = nullptr;
        moe_raw_staging_size_ = 0;
    }
    if (moe_batch_dequant_buf_) {
        cudaFree(moe_batch_dequant_buf_);
        moe_batch_dequant_buf_ = nullptr;
    }
    moe_batch_dequant_buf_size_ = 0;
    if (d_moe_work_ptrs_) {
        cudaFree(d_moe_work_ptrs_);
        d_moe_work_ptrs_ = nullptr;
        d_moe_work_ptrs_count_ = 0;
    }
    if (d_moe_fp8_scales_) {
        cudaFree(d_moe_fp8_scales_);
        d_moe_fp8_scales_ = nullptr;
    }
    if (d_moe_weight_ptrs_) {
        cudaFree(d_moe_weight_ptrs_);
        d_moe_weight_ptrs_ = nullptr;
        d_moe_weight_ptrs_count_ = 0;
    }
    if (dequant_scratch_) {
        cudaFree(dequant_scratch_);
        dequant_scratch_ = nullptr;
        dequant_scratch_size_ = 0;
    }
    if (d_sample_result_) {
        cudaFree(d_sample_result_);
        d_sample_result_ = nullptr;
    }
    if (h_logits_pinned_) {
        cudaFreeHost(h_logits_pinned_);
        h_logits_pinned_ = nullptr;
        h_logits_pinned_size_ = 0;
    }
    if (q8_1_buf_) {
        cudaFree(q8_1_buf_);
        q8_1_buf_ = nullptr;
    }
    if (d8_buf_) {
        cudaFree(d8_buf_);
        d8_buf_ = nullptr;
    }
    q8_1_max_blocks_ = 0;
    if (splitk_scratch_) {
        cudaFree(splitk_scratch_);
        splitk_scratch_ = nullptr;
        splitk_scratch_size_ = 0;
    }
    if (attn_scores_buf_) {
        cudaFree(attn_scores_buf_);
        attn_scores_buf_ = nullptr;
        attn_scores_buf_size_ = 0;
    }
#ifdef IMP_USE_CUTLASS
    cutlass_fmha_free_workspace();
#endif
    if (shared_workspace_) {
        cudaFree(shared_workspace_);
        shared_workspace_ = nullptr;
        shared_workspace_size_ = 0;
    }
    if (persistent_workspace_) {
        cudaFree(persistent_workspace_);
        persistent_workspace_ = nullptr;
        persistent_workspace_size_ = 0;
    }
    if (fp32_accum_buf_) {
        cudaFree(fp32_accum_buf_);
        fp32_accum_buf_ = nullptr;
    }
    ssm_layer_map_.clear();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Pre-dequantize quantized weights to FP16 on GPU
// ---------------------------------------------------------------------------

void GraphExecutor::pre_dequant_weights(cudaStream_t stream, size_t cache_budget) {
    if (!initialized_ || !model_) return;

    const auto& cfg = model_->config();
    size_t total_cache_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    // Shared budget across all phases (FP16, FP8 overflow, NVFP4 decode).
    // Computed by Engine from effective_free_vram() minus a runtime reserve.
    size_t remaining_budget = cache_budget;

    // Helper: does this qtype benefit from NVFP4 conversion? (> 4.5 bits/elem)
    auto nvfp4_beneficial = [](GGMLQuantType qt) -> bool {
        switch (qt) {
            case GGMLQuantType::Q8_0: case GGMLQuantType::Q8_K:
            case GGMLQuantType::Q6_K: case GGMLQuantType::Q5_K:
                return true;
            default: return false;
        }
    };

    {
        // --- Phase 1: FP16 weight cache + fused KV + fused gate+up (always) ---
        auto cache_weight = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (fp16_cache_.count(w.data)) return;  // already cached
            if (budget_exhausted) return;

            int rows = static_cast<int>(w.shape[0]);
            int cols = static_cast<int>(w.shape[1]);
            size_t fp16_bytes = static_cast<size_t>(rows) * cols * sizeof(half);

            if (total_cache_bytes + fp16_bytes > remaining_budget) {
                budget_exhausted = true;
                IMP_LOG_INFO("FP16 cache: VRAM budget reached after %d tensors (%.1f / %.1f MiB), "
                             "remaining weights will use on-the-fly dequant",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            void* fp16_buf = nullptr;
            cudaError_t err = cudaMalloc(&fp16_buf, fp16_bytes);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Cleared FP16 cache alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                budget_exhausted = true;
                IMP_LOG_WARN("FP16 cache: cudaMalloc failed after %d tensors (%.1f MiB)",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0));
                return;
            }

            dequant_gpu(w.data, fp16_buf, qtype, rows, cols, stream);

            Tensor fp16_tensor(fp16_buf, DType::FP16, w.ndim, w.shape, true);
            fp16_cache_[w.data] = fp16_tensor;
            total_cache_bytes += fp16_bytes;
            cached_count++;
        };

        // Priority order: attention weights first (critical for cuBLAS prefill),
        // then SSM, shared experts, and dense FFN.  This ensures hybrid models
        // like Nemotron (23 SSM + 6 attention layers) cache all attention weights
        // before SSM weights exhaust the VRAM budget.
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.wq, L.wq_qtype);
            cache_weight(L.wk, L.wk_qtype);
            cache_weight(L.wv, L.wv_qtype);
            cache_weight(L.wo, L.wo_qtype);
        }
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.ssm_in, L.ssm_in_qtype);
            cache_weight(L.ssm_out, L.ssm_out_qtype);
            cache_weight(L.w_gate_shared, L.w_gate_shared_qtype);
            cache_weight(L.w_up_shared, L.w_up_shared_qtype);
            cache_weight(L.w_down_shared, L.w_down_shared_qtype);
            // When NVFP4 decode is active, skip dense FFN FP16 cache for eligible
            // weights.  Decode benefits more from NVFP4 (~47% BW reduction) than
            // prefill loses from on-the-fly dequant.  NVFP4 is also ~3.5x smaller
            // per tensor, so skipping FFN FP16 frees massive VRAM for full NVFP4.
            if (use_nvfp4_decode_ == 0 || !nvfp4_beneficial(L.w_gate_qtype))
                cache_weight(L.w_gate, L.w_gate_qtype);
            if (use_nvfp4_decode_ == 0 || !nvfp4_beneficial(L.w_up_qtype))
                cache_weight(L.w_up, L.w_up_qtype);
            if (use_nvfp4_decode_ == 0 || !nvfp4_beneficial(L.w_down_qtype))
                cache_weight(L.w_down, L.w_down_qtype);
        }

        // Create fused KV weights for strided batched prefill GEMM.
        // Each entry concatenates [wk; wv] as [2*nkv*hd, d_model] FP16 for one layer.
        int fused_kv_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (!L.wk.data || !L.wv.data) continue;
            auto wk_it = fp16_cache_.find(L.wk.data);
            auto wv_it = fp16_cache_.find(L.wv.data);
            if (wk_it == fp16_cache_.end() || wv_it == fp16_cache_.end()) continue;

            int k_rows = static_cast<int>(L.wk.shape[0]);  // nkv * hd
            int K = static_cast<int>(L.wk.shape[1]);        // d_model
            size_t one_sz = static_cast<size_t>(k_rows) * K * sizeof(half);

            // Respect VRAM budget — on WSL2/WDDM, cudaMalloc silently spills to
            // shared (system) memory beyond physical VRAM, causing massive slowdowns.
            if (total_cache_bytes + 2 * one_sz > remaining_budget) break;

            void* fused_buf = nullptr;
            cudaError_t err = cudaMalloc(&fused_buf, 2 * one_sz);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Cleared fused KV alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                break;
            }

            cudaMemcpyAsync(fused_buf, wk_it->second.data, one_sz,
                             cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                             wv_it->second.data, one_sz,
                             cudaMemcpyDeviceToDevice, stream);

            int64_t shape[2] = {2 * k_rows, static_cast<int64_t>(K)};
            fused_kv_cache_[i] = Tensor(fused_buf, DType::FP16, 2, shape, true);
            total_cache_bytes += 2 * one_sz;
            fused_kv_count++;
        }

        // Create fused gate+up weights for strided batched prefill GEMM.
        // Each entry concatenates [w_gate; w_up] as [2*d_ff, d_model] FP16 for one layer.
        int fused_gu_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (!L.w_gate.data || !L.w_up.data) continue;
            // Both must be the same shape (d_ff x d_model)
            if (L.w_gate.shape[0] != L.w_up.shape[0] ||
                L.w_gate.shape[1] != L.w_up.shape[1]) continue;
            auto wg_it = fp16_cache_.find(L.w_gate.data);
            auto wu_it = fp16_cache_.find(L.w_up.data);
            if (wg_it == fp16_cache_.end() || wu_it == fp16_cache_.end()) continue;

            int g_rows = static_cast<int>(L.w_gate.shape[0]);  // d_ff
            int K = static_cast<int>(L.w_gate.shape[1]);        // d_model
            size_t one_sz = static_cast<size_t>(g_rows) * K * sizeof(half);

            // Respect VRAM budget (see fused KV comment above)
            if (total_cache_bytes + 2 * one_sz > remaining_budget) break;

            void* fused_buf = nullptr;
            cudaError_t err = cudaMalloc(&fused_buf, 2 * one_sz);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Cleared fused gate+up alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                break;
            }

            cudaMemcpyAsync(fused_buf, wg_it->second.data, one_sz,
                             cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                             wu_it->second.data, one_sz,
                             cudaMemcpyDeviceToDevice, stream);

            int64_t shape[2] = {2 * g_rows, static_cast<int64_t>(K)};
            fused_gate_up_cache_[i] = Tensor(fused_buf, DType::FP16, 2, shape, true);
            total_cache_bytes += 2 * one_sz;
            fused_gu_count++;
        }

        if (cached_count > 0) {
            cudaStreamSynchronize(stream);
            fp16_cache_bytes_ = total_cache_bytes;
            IMP_LOG_INFO("FP16 weight cache: %d tensors, %.2f MiB (incl. %d fused KV, %d fused gate+up)",
                         cached_count, total_cache_bytes / (1024.0 * 1024.0),
                         fused_kv_count, fused_gu_count);
        }
    }

    // Deduct Phase 1 allocation from shared budget
    remaining_budget = (remaining_budget > total_cache_bytes)
                       ? (remaining_budget - total_cache_bytes) : 0;

    // --- Phase 2: FP8 overflow cache for remaining uncached weights ---
    // Weights already in fp16_cache_ keep their fused KV/gate+up optimizations.
    // FP8 is only used for weights that didn't fit in the FP16 budget (50% smaller).
    // Uses dequant_scratch_ as FP16 staging buffer (stream ordering ensures safety).
    if (use_fp8_cache_) {
        size_t fp8_total = 0;
        int fp8_count = 0;
        bool fp8_exhausted = false;

        // Collect weights to convert
        struct FP8OverflowEntry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            size_t n_elems;
        };
        std::vector<FP8OverflowEntry> fp8_entries;

        auto collect_weight_fp8 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (fp16_cache_.count(w.data)) return;
            if (fp8_cache_.count(w.data)) return;
            if (fp8_exhausted) return;

            size_t n_elems = static_cast<size_t>(w.shape[0]) * w.shape[1];
            size_t fp8_bytes = n_elems;

            if (fp8_total + fp8_bytes + sizeof(float) > remaining_budget) {
                fp8_exhausted = true;
                IMP_LOG_INFO("FP8 overflow: budget reached after %d tensors (%.1f / %.1f MiB)",
                             fp8_count, fp8_total / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            fp8_entries.push_back({w.data, w, qtype, n_elems});
            fp8_total += fp8_bytes + sizeof(float);
            fp8_count++;
        };

        // Same priority order — attention first, then SSM/FFN
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            collect_weight_fp8(L.wq, L.wq_qtype);
            collect_weight_fp8(L.wk, L.wk_qtype);
            collect_weight_fp8(L.wv, L.wv_qtype);
            collect_weight_fp8(L.wo, L.wo_qtype);
        }
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            collect_weight_fp8(L.ssm_in, L.ssm_in_qtype);
            collect_weight_fp8(L.ssm_out, L.ssm_out_qtype);
            collect_weight_fp8(L.w_gate_shared, L.w_gate_shared_qtype);
            collect_weight_fp8(L.w_up_shared, L.w_up_shared_qtype);
            collect_weight_fp8(L.w_down_shared, L.w_down_shared_qtype);
            collect_weight_fp8(L.w_gate, L.w_gate_qtype);
            collect_weight_fp8(L.w_up, L.w_up_qtype);
            collect_weight_fp8(L.w_down, L.w_down_qtype);
        }

        if (!fp8_entries.empty() && dequant_scratch_) {
            // Pre-allocate reusable calibration temp buffers
            int max_grid = 0;
            size_t total_fp8_bytes = 0;
            for (auto& e : fp8_entries) {
                int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                int grid = (threads_needed + 255) / 256;
                if (grid > max_grid) max_grid = grid;
                total_fp8_bytes += e.n_elems;
            }

            float* d_block_maxes = nullptr;
            float* d_absmax = nullptr;
            float* d_scales_all = nullptr;
            cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float));
            cudaMalloc(&d_absmax, sizeof(float));
            cudaMalloc(&d_scales_all, fp8_entries.size() * sizeof(float));

            // Bulk-allocate all FP8 data in one cudaMalloc
            uint8_t* d_fp8_bulk = nullptr;
            cudaError_t bulk_err = cudaMalloc(&d_fp8_bulk, total_fp8_bytes);
            if (bulk_err != cudaSuccess) {
                cudaGetLastError();
                d_fp8_bulk = nullptr;
            }

            int actual_count = 0;
            size_t fp8_offset = 0;
            for (size_t i = 0; i < fp8_entries.size() && d_fp8_bulk; i++) {
                auto& e = fp8_entries[i];
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                // Dequant to dequant_scratch_ (reused each iteration, stream-ordered)
                dequant_gpu(e.weight.data, dequant_scratch_, e.qtype, rows, cols, stream);

                void* fp8_buf = d_fp8_bulk + fp8_offset;
                fp8_offset += e.n_elems;

                // Async calibrate + quantize (no host sync)
                calibrate_and_quantize_fp8_async(
                    dequant_scratch_, fp8_buf, static_cast<int>(e.n_elems),
                    d_block_maxes, max_grid,
                    d_absmax, d_scales_all + static_cast<ptrdiff_t>(i), stream);

                Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.weight.ndim, e.weight.shape, true);
                fp8_cache_[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                actual_count++;
            }

            if (actual_count > 0) {
                cudaStreamSynchronize(stream);
                // Read back scales
                std::vector<float> h_scales(actual_count);
                cudaMemcpy(h_scales.data(), d_scales_all, actual_count * sizeof(float),
                           cudaMemcpyDeviceToHost);
                for (int i = 0; i < actual_count; i++) {
                    auto it = fp8_cache_.find(fp8_entries[i].orig_ptr);
                    if (it != fp8_cache_.end()) {
                        it->second.host_scale = h_scales[i];
                    }
                }
            }

            cudaFree(d_block_maxes);
            cudaFree(d_absmax);
            // Track bulk buffers for cleanup
            fp8_overflow_scales_ = d_scales_all;
            fp8_overflow_count_ = actual_count;
            fp8_overflow_data_ = d_fp8_bulk;
            fp8_overflow_data_size_ = total_fp8_bytes;
            fp8_count = actual_count;
        }

        if (fp8_count > 0) {
            fp8_cache_bytes_ = fp8_total;
            size_t fp16_equivalent = 0;
            for (auto& [ptr, entry] : fp8_cache_) {
                fp16_equivalent += entry.weight.numel() * sizeof(half);
            }
            IMP_LOG_INFO("FP8 overflow cache: %d tensors, %.2f MiB (%.2f MiB saved vs FP16)",
                         fp8_count, fp8_total / (1024.0 * 1024.0),
                         (fp16_equivalent - fp8_total) / (1024.0 * 1024.0));
        } else {
            IMP_LOG_INFO("FP8 prefill: all weights fit in FP16 cache, no FP8 overflow needed");
        }
    }

    // Deduct Phase 2 allocation from shared budget
    remaining_budget = (remaining_budget > fp8_cache_bytes_)
                       ? (remaining_budget - fp8_cache_bytes_) : 0;

    // --- Phase 3: NVFP4 decode weight cache ---
    // Converts eligible weights (> 4.5 bits/elem) to NVFP4 format for faster
    // decode GEMV.  Weights in fp16_cache_ are quantized directly (zero-copy);
    // weights NOT in fp16_cache_ (e.g. dense FFN skipped in Phase 1) are
    // dequantized via dequant_scratch_ as a transient FP16 staging buffer.
    // Mode 2 additionally frees all FP16 cache at the end.
    if (use_nvfp4_decode_ > 0) {
        size_t nvfp4_total = 0;
        int nvfp4_count = 0;
        int nvfp4_from_scratch = 0;
        bool nvfp4_budget_exhausted = false;
        const char* mode_str = (use_nvfp4_decode_ == 1) ? "additive" : "only";

        // Collect eligible weights first, then batch-process async.
        struct NvFP4Entry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            bool from_scratch;
        };
        std::vector<NvFP4Entry> nvfp4_entries;

        auto collect_weight_nvfp4 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (nvfp4_cache_.count(w.data)) return;
            if (nvfp4_budget_exhausted) return;

            int cols = static_cast<int>(w.shape[1]);
            if (cols % 16 != 0) return;

            int rows = static_cast<int>(w.shape[0]);
            size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                                 static_cast<size_t>(rows) * cols / 16 + 4;

            if (nvfp4_total + nvfp4_bytes > remaining_budget) {
                nvfp4_budget_exhausted = true;
                IMP_LOG_INFO("NVFP4 cache: VRAM budget reached after %d tensors "
                             "(%.1f / %.1f MiB), remaining weights use dp4a",
                             nvfp4_count, nvfp4_total / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            bool from_scratch = (fp16_cache_.find(w.data) == fp16_cache_.end());
            if (from_scratch && (!dequant_gpu_supported(qtype) || !dequant_scratch_)) return;
            nvfp4_entries.push_back({w.data, w, qtype, from_scratch});
            nvfp4_total += nvfp4_bytes;
            nvfp4_count++;
            if (from_scratch) nvfp4_from_scratch++;
        };

        // Dense attention + FFN first: every tensor benefits every decode step.
        // MoE experts second: each tensor only benefits 1 of N layers.
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            collect_weight_nvfp4(L.wq, L.wq_qtype);
            collect_weight_nvfp4(L.wk, L.wk_qtype);
            collect_weight_nvfp4(L.wv, L.wv_qtype);
            collect_weight_nvfp4(L.wo, L.wo_qtype);
        }
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            collect_weight_nvfp4(L.ssm_in, L.ssm_in_qtype);
            collect_weight_nvfp4(L.ssm_out, L.ssm_out_qtype);
            collect_weight_nvfp4(L.w_gate_shared, L.w_gate_shared_qtype);
            collect_weight_nvfp4(L.w_up_shared, L.w_up_shared_qtype);
            collect_weight_nvfp4(L.w_down_shared, L.w_down_shared_qtype);
            collect_weight_nvfp4(L.w_gate, L.w_gate_qtype);
            collect_weight_nvfp4(L.w_up, L.w_up_qtype);
            collect_weight_nvfp4(L.w_down, L.w_down_qtype);
        }

        // Batch-process: quantize all collected entries async, single sync at end.
        if (!nvfp4_entries.empty()) {
            // Reusable absmax buffer (overwritten each iteration, stream-ordered)
            float* d_absmax_buf = nullptr;
            cudaMalloc(&d_absmax_buf, sizeof(float));

            // Bulk tensor_scale buffer: one float per entry (read back after sync)
            float* d_tscales_all = nullptr;
            cudaMalloc(&d_tscales_all, nvfp4_entries.size() * sizeof(float));

            for (size_t i = 0; i < nvfp4_entries.size(); i++) {
                auto& e = nvfp4_entries[i];
                const half* fp16_ptr = nullptr;
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                if (e.from_scratch) {
                    // Dequant GGML→FP16 into scratch (stream-ordered, reused each iter)
                    dequant_gpu(e.weight.data, dequant_scratch_, e.qtype, rows, cols, stream);
                    fp16_ptr = reinterpret_cast<const half*>(dequant_scratch_);
                } else {
                    auto it = fp16_cache_.find(e.orig_ptr);
                    fp16_ptr = reinterpret_cast<const half*>(it->second.data);
                }

                Tensor fp16_view(const_cast<half*>(fp16_ptr), DType::FP16, 2,
                                 e.weight.shape, true);

                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf,
                                              d_tscales_all + i,
                                              stream);
                nvfp4_cache_[e.orig_ptr] = result;
            }

            // Single sync for all NVFP4 quantizations
            cudaStreamSynchronize(stream);

            // Bulk read back tensor_scales
            std::vector<float> h_tscales(nvfp4_entries.size());
            cudaMemcpy(h_tscales.data(), d_tscales_all,
                       nvfp4_entries.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < nvfp4_entries.size(); i++) {
                auto it = nvfp4_cache_.find(nvfp4_entries[i].orig_ptr);
                if (it != nvfp4_cache_.end()) {
                    it->second.tensor_scale = h_tscales[i];
                }
            }

            cudaFree(d_absmax_buf);
            cudaFree(d_tscales_all);

            nvfp4_cache_bytes_ = nvfp4_total;
            if (nvfp4_from_scratch > 0) {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (%d from FP16 cache, %d via dequant scratch, mode: %s)",
                             nvfp4_count, nvfp4_total / (1024.0 * 1024.0),
                             nvfp4_count - nvfp4_from_scratch, nvfp4_from_scratch, mode_str);
            } else {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (mode: %s)",
                             nvfp4_count, nvfp4_total / (1024.0 * 1024.0), mode_str);
            }
        }

        // In "only" mode (2), release FP16 cache to save VRAM.
        // Before freeing, migrate FP16 weights to FP8 cache so prefill
        // retains fast FP8 GEMM instead of falling back to on-the-fly dequant.
        // FP8 = half the size of FP16, so net VRAM savings = 50% of FP16 cache.
        if (use_nvfp4_decode_ == 2 && !fp16_cache_.empty()) {
            // Migrate FP16→FP8 for weights not already in fp8_cache_.
            // Batched async: all calibrate+quantize kernels enqueued without
            // per-tensor host sync.  Single sync at the end.
            int migrated = 0;
            size_t migrated_bytes = 0;
            if (use_fp8_cache_) {
                // Collect tensors to migrate
                struct MigrateEntry {
                    const void* orig_ptr;
                    Tensor fp16_tensor;
                    size_t n_elems;
                };
                std::vector<MigrateEntry> to_migrate;
                for (auto& [orig_ptr, fp16_tensor] : fp16_cache_) {
                    if (fp8_cache_.count(orig_ptr)) continue;
                    size_t n = static_cast<size_t>(fp16_tensor.shape[0]) * fp16_tensor.shape[1];
                    to_migrate.push_back({orig_ptr, fp16_tensor, n});
                }

                if (!to_migrate.empty()) {
                    // Find max grid size needed for temp buffers
                    int max_grid = 0;
                    size_t total_fp8_bytes = 0;
                    for (auto& e : to_migrate) {
                        int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                        int grid = (threads_needed + 255) / 256;
                        if (grid > max_grid) max_grid = grid;
                        total_fp8_bytes += e.n_elems;  // FP8 = 1 byte per element
                    }

                    // Pre-allocate reusable temp buffers (once, not per-tensor)
                    float* d_block_maxes = nullptr;
                    float* d_absmax = nullptr;
                    cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float));
                    cudaMalloc(&d_absmax, sizeof(float));

                    // Allocate all scale values in a single buffer
                    float* d_scales_all = nullptr;
                    cudaMalloc(&d_scales_all, to_migrate.size() * sizeof(float));

                    // Bulk-allocate all FP8 data in one cudaMalloc
                    uint8_t* d_fp8_bulk = nullptr;
                    cudaError_t bulk_err = cudaMalloc(&d_fp8_bulk, total_fp8_bytes);
                    if (bulk_err != cudaSuccess) {
                        cudaGetLastError();
                        d_fp8_bulk = nullptr;
                    }

                    size_t fp8_offset = 0;
                    for (size_t i = 0; i < to_migrate.size() && d_fp8_bulk; i++) {
                        auto& e = to_migrate[i];
                        void* fp8_buf = d_fp8_bulk + fp8_offset;
                        fp8_offset += e.n_elems;

                        // Async calibrate + quantize (no host sync)
                        calibrate_and_quantize_fp8_async(
                            e.fp16_tensor.data, fp8_buf, static_cast<int>(e.n_elems),
                            d_block_maxes, max_grid,
                            d_absmax, d_scales_all + i, stream);

                        Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.fp16_tensor.ndim,
                                     e.fp16_tensor.shape, true);
                        fp8_cache_[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                        migrated++;
                        migrated_bytes += e.n_elems + sizeof(float);
                    }

                    // Track bulk buffer for cleanup
                    fp8_migrated_data_ = d_fp8_bulk;
                    fp8_migrated_data_size_ = total_fp8_bytes;

                    // Single sync for ALL migrations
                    if (migrated > 0) {
                        cudaStreamSynchronize(stream);
                        // Read back scales from device to host (for fp8_cache_ host scale field)
                        std::vector<float> h_scales(migrated);
                        cudaMemcpy(h_scales.data(), d_scales_all, migrated * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        int idx = 0;
                        for (size_t i = 0; i < to_migrate.size() && idx < migrated; i++, idx++) {
                            auto it = fp8_cache_.find(to_migrate[i].orig_ptr);
                            if (it != fp8_cache_.end()) {
                                it->second.host_scale = h_scales[idx];
                            }
                        }
                    }

                    cudaFree(d_block_maxes);
                    cudaFree(d_absmax);
                    // d_scales_all stays alive — each entry is pointed to by fp8_cache_
                    fp8_migrated_scales_ = d_scales_all;
                    fp8_migrated_count_ = migrated;
                }
            }

            // Now free FP16 cache
            for (auto& [ptr, tensor] : fp16_cache_) {
                cudaFree(tensor.data);
            }
            size_t freed = fp16_cache_bytes_;
            fp16_cache_.clear();
            fp16_cache_bytes_ = 0;

            // Also free fused KV and gate+up caches (prefill uses individual
            // FP8 Q/K/V weights instead of fused paths)
            for (auto& [idx, tensor] : fused_kv_cache_) {
                if (tensor.data) cudaFree(tensor.data);
            }
            fused_kv_cache_.clear();
            for (auto& [idx, tensor] : fused_gate_up_cache_) {
                if (tensor.data) cudaFree(tensor.data);
            }
            fused_gate_up_cache_.clear();

            // Reclaim freed VRAM for MoE expert caching
            remaining_budget += freed;
            fp8_cache_bytes_ += migrated_bytes;
            IMP_LOG_INFO("NVFP4 only mode: freed FP16 cache (%.2f MiB), migrated %d weights to FP8 (%.2f MiB)",
                         freed / (1024.0 * 1024.0), migrated, migrated_bytes / (1024.0 * 1024.0));
        }

        // --- Phase 3b: Convert NVFP4 weights to CUTLASS sm_120 block-scaled format ---
        // Must be AFTER FP16 free to avoid peak VRAM exceeding physical memory.
        // The CUTLASS cache is a full copy (repacked data + SfAtom scales), so it
        // approximately doubles the NVFP4 cache VRAM.  Budget-aware: stop if VRAM runs out.
        if (nvfp4_count > 0 && cutlass_sm120_nvfp4_available()) {
            size_t ct_budget = (remaining_budget > nvfp4_total)
                               ? (remaining_budget - nvfp4_total) : 0;
            int ct_count = 0;
            size_t ct_total = 0;
            bool ct_exhausted = false;
            for (auto& [ptr, nvfp4] : nvfp4_cache_) {
                if (ct_exhausted) break;
                // Estimate CUTLASS allocation (only scale factors — data is borrowed)
                size_t est = cutlass_nvfp4_sf_size(static_cast<int>(nvfp4.N),
                                                    static_cast<int>(nvfp4.K));
                if (ct_total + est > ct_budget) {
                    ct_exhausted = true;
                    IMP_LOG_INFO("CUTLASS NVFP4 cache: VRAM budget reached after %d tensors "
                                 "(%.1f / %.1f MiB)",
                                 ct_count, ct_total / (1024.0 * 1024.0),
                                 ct_budget / (1024.0 * 1024.0));
                    break;
                }
                CutlassNvFP4Weight cw;
                convert_nvfp4_to_cutlass(nvfp4, cw, stream);
                if (cw.data) {
                    cutlass_nvfp4_cache_[ptr] = cw;
                    ct_total += cw.sf_bytes;
                    ct_count++;
                }
            }
            if (ct_count > 0) {
                cudaStreamSynchronize(stream);
                cutlass_nvfp4_cache_bytes_ = ct_total;
                remaining_budget = (remaining_budget > ct_total + nvfp4_total)
                                   ? (remaining_budget - ct_total - nvfp4_total) : 0;
                IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: %d tensors, %.2f MiB",
                             ct_count, ct_total / (1024.0 * 1024.0));
            }
        }

        // Cache MoE expert weights — done after FP16 free so mode 2 has full budget
        int nvfp4_moe_count = 0;
        size_t nvfp4_moe_total = 0;
        size_t moe_budget = (remaining_budget > nvfp4_total)
                            ? (remaining_budget - nvfp4_total) : 0;
        bool moe_budget_exhausted = false;

        auto cache_moe_expert_nvfp4 = [&](const Tensor& packed, GGMLQuantType qtype) {
            if (!packed.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (nvfp4_moe_cache_.count(packed.data)) return;
            if (moe_budget_exhausted) return;
            if (!packed.on_device) return;
            if (packed.ndim < 3) return;

            int ne = static_cast<int>(packed.shape[0]);
            int rows = static_cast<int>(packed.shape[1]);
            int cols = static_cast<int>(packed.shape[2]);
            if (cols % 16 != 0) return;
            if (!dequant_gpu_supported(qtype) || !dequant_scratch_) return;

            size_t nvfp4_bytes = static_cast<size_t>(ne) * rows * cols / 2 +
                                 static_cast<size_t>(ne) * rows * cols / 16 +
                                 static_cast<size_t>(ne) * sizeof(float);

            if (nvfp4_moe_total + nvfp4_bytes > moe_budget) {
                moe_budget_exhausted = true;
                IMP_LOG_INFO("NVFP4 MoE cache: VRAM budget reached after %d MoE tensors "
                             "(%.1f / %.1f MiB)", nvfp4_moe_count,
                             nvfp4_moe_total / (1024.0 * 1024.0),
                             moe_budget / (1024.0 * 1024.0));
                return;
            }

            NvFP4MoEQuantResult result;
            quantize_packed_experts_to_nvfp4(
                packed.data, qtype, ne, rows, cols,
                dequant_scratch_, result, stream);

            nvfp4_moe_cache_[packed.data] = result;
            nvfp4_moe_total += nvfp4_bytes;
            nvfp4_moe_count++;
        };

        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_moe_expert_nvfp4(L.expert_gate_packed, L.expert_gate_qtype);
            cache_moe_expert_nvfp4(L.expert_up_packed,   L.expert_up_qtype);
            cache_moe_expert_nvfp4(L.expert_down_packed,  L.expert_down_qtype);
        }

        if (nvfp4_moe_count > 0) {
            nvfp4_moe_cache_bytes_ = nvfp4_moe_total;
            IMP_LOG_INFO("NVFP4 MoE cache: %d tensors, %.2f MiB",
                         nvfp4_moe_count, nvfp4_moe_total / (1024.0 * 1024.0));
        } else if (nvfp4_count == 0) {
            IMP_LOG_INFO("NVFP4 decode: no eligible weights found (all ≤ 4.5 bits/elem)");
        }
    }
}

// ---------------------------------------------------------------------------
// Shared workspace configuration (pure pointer arithmetic, no allocation)
// ---------------------------------------------------------------------------

void GraphExecutor::configure_attn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    q_        = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    // K and V are contiguous (no alignment gap) to enable strided batched GEMM.
    // v_.data == k_.data + kv_raw exactly, so output_stride = kv_raw / es.
    {
        size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
        int64_t kv_shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(nkv * hd)};
        k_ = Tensor(ptr, compute_dtype_, 2, kv_shape, true);
        v_ = Tensor(ptr + kv_raw, compute_dtype_, 2, kv_shape, true);
        ptr += align256(2 * kv_raw);
    }
    attn_out_ = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    proj_out_ = make(d,        align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_ffn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d  = cfg.d_model;
    int ff = cfg.d_ff;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    gate_out_   = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    up_out_     = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    swiglu_out_ = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    ffn_out_    = make(d,  align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_moe_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d     = cfg.d_model;
    int ne    = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff   = max_expert_eff_;
    size_t es = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    // gate_logits: FP32
    {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(ne)};
        moe_gate_logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += align256(static_cast<size_t>(max_tokens) * ne * sizeof(float));
    }

    auto make_moe = [&](Tensor& t, int64_t rows, int64_t cols, size_t aligned_sz, DType dt) {
        int64_t shape[2] = {rows, cols};
        t = Tensor(ptr, dt, 2, shape, true);
        ptr += aligned_sz;
    };

    make_moe(moe_gathered_,      expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_expert_gate_,   expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_up_,     expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_swiglu_, expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_down_,   expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_scatter_out_,   max_tokens, d, align256(static_cast<size_t>(max_tokens) * d * sizeof(float)), DType::FP32);
}

void GraphExecutor::configure_ssm_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int state_size = cfg.ssm_state_size;
    int n_heads = cfg.ssm_dt_rank;
    int conv_channels = inner + 2 * n_groups * state_size;
    int ssm_in_dim = inner + conv_channels + n_heads;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    ssm_proj_buf_ = make(ssm_in_dim,    align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es));
    ssm_xBC_buf_  = make(conv_channels,  align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_z_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_out_buf_  = make(d,              align256(static_cast<size_t>(max_tokens) * d * es));
    ssm_dt_buf_   = make(n_heads,        align256(static_cast<size_t>(max_tokens) * n_heads * es));
}

bool GraphExecutor::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0) return true;
    if (new_max_tokens > max_tokens_) new_max_tokens = max_tokens_;  // never exceed init-time max

    // Recompute shared sizes for the new token count
    int saved_max = max_tokens_;
    max_tokens_ = new_max_tokens;
    compute_shared_sizes(new_max_tokens);
    max_tokens_ = saved_max;

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0) return true;

    if (new_shared != shared_workspace_size_) {
        if (shared_workspace_) {
            cudaFreeAsync(shared_workspace_, stream);
        }
        cudaError_t err = cudaMallocAsync(&shared_workspace_, new_shared, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes: %s",
                          new_shared, cudaGetErrorString(err));
            shared_workspace_ = nullptr;
            shared_workspace_size_ = 0;
            return false;
        }
        shared_workspace_size_ = new_shared;
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
}

bool GraphExecutor::layer_has_attention(int layer) const {
    return model_->layer(layer).wq.data != nullptr;
}

bool GraphExecutor::layer_has_ssm(int layer) const {
    return model_->layer(layer).ssm_in.data != nullptr;
}

bool GraphExecutor::layer_has_moe(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.moe_gate.data != nullptr;
}

bool GraphExecutor::layer_has_dense_ffn(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.w_up.data != nullptr && ly.moe_gate.data == nullptr;
}

Tensor GraphExecutor::view_tokens(const Tensor& buf, int n_tokens) const {
    // buf is always [max_tokens_, cols] from allocate_buffers.
    // Return a [n_tokens, cols] view.
    return slice_rows(buf, n_tokens);
}


void GraphExecutor::ensure_logits_pinned(int vocab_size) {
    if (h_logits_pinned_ && h_logits_pinned_size_ >= vocab_size) return;
    if (h_logits_pinned_) cudaFreeHost(h_logits_pinned_);
    cudaHostAlloc(&h_logits_pinned_, vocab_size * sizeof(float), cudaHostAllocDefault);
    h_logits_pinned_size_ = vocab_size;
}

} // namespace imp

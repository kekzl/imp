// Auxiliary buffer allocation and cleanup — extracted from executor_workspace.cu (RF-004).
// Handles: dequant scratch, sampling, MMVQ, split-K, attention S-matrix, FMHA,
// MoE workspace, FP8 activation, CUTLASS NVFP4/MXFP4 activation buffers.

#include "exec/executor.h"
#include "memory/vram_query.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/gemm_scratch.h"  // prewarm_mmvq_scratch
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/sampling.h"
#include "runtime/config.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <algorithm>

namespace imp {

void GraphExecutor::allocate_auxiliary_buffers(bool skip_batch_dequant) {
    const auto& cfg = model_->config();

    // MLA (DeepSeek) persistent QKV scratch: pre-allocate once so the
    // materialized two-step KV projection never calls cudaMallocAsync inside
    // the CUDA-graph-captured decode region. Sized for max_tokens.
    if (cfg.is_mla() && max_tokens_ > 0) {
        const size_t T = static_cast<size_t>(max_tokens_);
        const size_t kva_out = static_cast<size_t>(cfg.kv_lora_rank + cfg.qk_rope_head_dim);
        const size_t kvb_out =
            static_cast<size_t>(cfg.n_heads) * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        auto alloc = [&](void** p, size_t cols, const char* name) {
            size_t sz = T * cols * sizeof(half);
            cudaError_t e = cudaMalloc(p, sz);
            if (e != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate MLA scratch %s (%.1f MiB): %s", name,
                              sz / (1024.0 * 1024.0), cudaGetErrorString(e));
                *p = nullptr;
            }
        };
        alloc(&mla_kv_a_buf_, kva_out, "kv_a");
        alloc(&mla_latent_buf_, static_cast<size_t>(cfg.kv_lora_rank), "latent");
        alloc(&mla_k_rope_buf_, static_cast<size_t>(cfg.qk_rope_head_dim), "k_rope");
        alloc(&mla_kv_b_buf_, kvb_out, "kv_b");
        IMP_LOG_INFO("MLA QKV scratch: kv_a+latent+k_rope+kv_b for max_tokens=%d (graph-safe)",
                     max_tokens_);

        // Phase 3: absorbed-decode latent KV cache. Opt-in via attention.mla_absorb.
        // Requires every layer's kv_b_proj to be FP16 (the absorbed path slices
        // W_UK/W_UV directly from it); skip + warn otherwise so the materialized
        // default stays unaffected.
        if (runtime_config().attention.mla_absorb && mla_absorb_max_seq_ > 0) {
            bool all_fp16 = true;
            for (int li = 0; li < cfg.n_layers; li++) {
                const auto& L = model_->layer(li);
                if (L.kv_b_proj.data == nullptr || L.kv_b_proj.qtype != QType::F16) {
                    all_fp16 = false;
                    break;
                }
            }
            if (!all_fp16) {
                IMP_LOG_WARN("attention.mla_absorb: kv_b_proj is not FP16 on all layers — "
                             "absorbed decode disabled, using materialized MLA path.");
            } else {
                const size_t row_w =
                    static_cast<size_t>(cfg.kv_lora_rank + cfg.qk_rope_head_dim);
                mla_absorb_layer_stride_ = static_cast<size_t>(mla_absorb_max_seq_) * row_w;
                size_t cache_bytes =
                    static_cast<size_t>(cfg.n_layers) * mla_absorb_layer_stride_ * sizeof(half);
                size_t scores_bytes =
                    static_cast<size_t>(cfg.n_heads) * mla_absorb_max_seq_ * sizeof(float);
                cudaError_t e1 = cudaMalloc(&mla_absorb_cache_, cache_bytes);
                cudaError_t e2 = cudaMalloc(&mla_absorb_scores_, scores_bytes);
                if (e1 != cudaSuccess || e2 != cudaSuccess) {
                    IMP_LOG_ERROR("attention.mla_absorb: latent cache alloc failed (%.1f MiB) — "
                                  "falling back to materialized.",
                                  cache_bytes / (1024.0 * 1024.0));
                    if (mla_absorb_cache_) { cudaFree(mla_absorb_cache_); mla_absorb_cache_ = nullptr; }
                    if (mla_absorb_scores_) { cudaFree(mla_absorb_scores_); mla_absorb_scores_ = nullptr; }
                } else {
                    // VRAM comparison vs the materialized per-token KV footprint.
                    const size_t mat_per_tok =
                        static_cast<size_t>(cfg.n_heads) *
                        ((cfg.qk_rope_head_dim + cfg.qk_nope_head_dim) /*K hd*/ +
                         (cfg.qk_rope_head_dim + cfg.qk_nope_head_dim) /*V padded to hd*/);
                    const size_t lat_per_tok = row_w;
                    IMP_LOG_INFO("MLA absorbed latent cache: %.1f MiB (n_layers=%d, max_seq=%d, "
                                 "row=%zu halfs). Per-token: latent %zu vs materialized %zu halfs "
                                 "(%.1fx smaller).",
                                 cache_bytes / (1024.0 * 1024.0), cfg.n_layers, mla_absorb_max_seq_,
                                 row_w, lat_per_tok, mat_per_tok,
                                 static_cast<double>(mat_per_tok) / static_cast<double>(lat_per_tok));
                }
            }
        }
    }

    // Dequant scratch buffer for on-the-fly weight dequantization. Every
    // consumer guards with dequant_gpu_supported(qtype) (GGUF block quants
    // only), so weights outside that set can never need the scratch — on
    // SafeTensors F16/NVFP4 models this skips the whole buffer (~85 MiB on
    // Qwen3-14B-NVFP4).
    {
        size_t max_weight_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                                  &L.w_up_shared, &L.w_down_shared, &L.ssm_in, &L.ssm_out}) {
                if (w->data && dequant_gpu_supported(w->qtype))
                    max_weight_elems = std::max(max_weight_elems, static_cast<size_t>(w->numel()));
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
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB", qscratch_.dequant_size / (1024.0 * 1024.0));
            }
        }
    }

    // Sampling result buffer: sized to hold the result plus the multi-block
    // partial reduction scratch for BOTH the greedy and the top-k/top-p paths
    // (SAMPLE_SCRATCH_BYTES >= ARGMAX_SCRATCH_BYTES).
    {
        cudaError_t err = cudaMalloc(&d_sample_result_, SAMPLE_SCRATCH_BYTES);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate sampling result buffer: %s", cudaGetErrorString(err));
            d_sample_result_ = nullptr;
        }
    }

    // Pinned host buffer for async sampling D2H copy (avoids stack-variable sync)
    if (!h_sample_pinned_ && d_sample_result_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample pinned buffer failed: %s", cudaGetErrorString(err));
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
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                                  &L.w_up_shared, &L.w_down_shared, &L.ssm_in, &L.ssm_out}) {
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
                max_moe_down_blocks = std::max(max_moe_down_blocks, cfg.n_experts_active * (down_k / 32));
            }
            if (L.expert_gate_packed.data && L.expert_gate_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_gate_packed.shape[2]));
            }
        }
        int max_blocks = std::max(max_k / 32, max_moe_down_blocks);
        if (max_blocks > 0) {
            qscratch_.q8_1_max_blocks = max_blocks;
            // Rows: the batched verify LM head quantizes one chunk batch
            // (max_logit_tokens_, floor 8) at a time; cap the multiplier so
            // large-batch servers don't inflate this K-sized scratch.
            qscratch_.q8_1_rows = std::min(std::max(max_logit_tokens_, 8), 16);
            size_t q8_1_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * qscratch_.q8_1_rows *
                             sizeof(block_q8_1);
            size_t d8_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * qscratch_.q8_1_rows *
                           sizeof(float);
            cudaError_t err1 = cudaMalloc(&qscratch_.q8_1_buf, q8_1_sz);
            cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d8_buf), d8_sz);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate MMVQ scratch buffers, dp4a path disabled");
                if (qscratch_.q8_1_buf) {
                    cudaFree(qscratch_.q8_1_buf);
                    qscratch_.q8_1_buf = nullptr;
                }
                if (qscratch_.d8_buf) {
                    cudaFree(qscratch_.d8_buf);
                    qscratch_.d8_buf = nullptr;
                }
                qscratch_.q8_1_max_blocks = 0;
            } else {
                IMP_LOG_INFO(
                    "MMVQ scratch buffers: %.2f KiB (q8_1) + %.2f KiB (d8), max_blocks=%d (max_k=%d, "
                    "moe_down=%d)",
                    q8_1_sz / 1024.0, d8_sz / 1024.0, max_blocks, max_k, max_moe_down_blocks);
            }
        }

        // FFN sparsity mask (Phase 2): one bit per Q8 block, packed uint32.
        if (qscratch_.q8_1_max_blocks > 0) {
            int mask_words = (qscratch_.q8_1_max_blocks + 31) / 32;
            size_t mask_sz = static_cast<size_t>(mask_words) * sizeof(uint32_t);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&qscratch_.ffn_block_mask), mask_sz);
            if (err != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate FFN sparsity mask buffer (%zu bytes): %s",
                             mask_sz, cudaGetErrorString(err));
                qscratch_.ffn_block_mask = nullptr;
                qscratch_.ffn_block_mask_words = 0;
            } else {
                qscratch_.ffn_block_mask_words = mask_words;
            }
        }

        // Pre-warm the file-scope MMVQ Q8_1 quantization scratch used by the
        // ggml_mmvq_q*_kernel hot-path in executor_kernels.cu. Sized for the
        // worst case (max_tokens × max_k) so the hot path never re-allocates
        // (capture-safe). QW1 from review/phase5_synthesis.md §2.1.
        if (max_k > 0 && max_tokens_ > 0) {
            prewarm_mmvq_scratch(max_tokens_, max_k);
        }

        // dp4a prefill scratch: enables direct Q4_K/Q5_K → dp4a GEMM for M>1
        // without the FP16 weight cache intermediate. Only allocated when the
        // model has sub-5-bit dense weights that benefit (Q4_K, Q5_K).
        if (max_blocks > 0 && max_tokens_ > 1) {
            bool has_sub5bit_dense = false;
            for (int i = 0; i < cfg.n_layers && !has_sub5bit_dense; i++) {
                const auto& L = model_->layer(i);
                for (const auto* w : {&L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                                      &L.w_up_shared, &L.w_down_shared,
                                      &L.wq, &L.wk, &L.wv, &L.wo}) {
                    if (w->data && (w->qtype == QType::Q4_K || w->qtype == QType::Q5_K)) {
                        has_sub5bit_dense = true;
                        break;
                    }
                }
            }
            if (has_sub5bit_dense) {
                // dp4a dense prefill only activates at M ≤ 64 (weight-stationary
                // TILE_M=16 re-reads weight ceil(M/16) times — only wins at small M).
                constexpr int kDp4aDenseMaxM = 64;
                int dp4a_m = std::min(max_tokens_, kDp4aDenseMaxM);
                int prefill_max_blocks = dp4a_m * (max_k / 32);
                size_t q8_sz = static_cast<size_t>(prefill_max_blocks) * sizeof(block_q8_1);
                size_t d8_sz = static_cast<size_t>(prefill_max_blocks) * sizeof(float);
                cudaError_t e1 = cudaMalloc(&qscratch_.q8_1_prefill_buf, q8_sz);
                cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d8_prefill_buf), d8_sz);
                if (e1 != cudaSuccess || e2 != cudaSuccess) {
                    IMP_LOG_WARN("dp4a prefill scratch alloc failed (%.1f MiB), FP16 cache fallback",
                                 (q8_sz + d8_sz) / (1024.0 * 1024.0));
                    if (qscratch_.q8_1_prefill_buf) { cudaFree(qscratch_.q8_1_prefill_buf); qscratch_.q8_1_prefill_buf = nullptr; }
                    if (qscratch_.d8_prefill_buf) { cudaFree(qscratch_.d8_prefill_buf); qscratch_.d8_prefill_buf = nullptr; }
                } else {
                    qscratch_.q8_1_prefill_bytes = q8_sz;
                    qscratch_.d8_prefill_bytes = d8_sz;
                    IMP_LOG_INFO("dp4a prefill scratch: %.1f KiB (max_m=%d, max_k=%d)",
                                 (q8_sz + d8_sz) / 1024.0, dp4a_m, max_k);
                }
            }
        }
    }

    // Split-K paged attention scratch buffer.
    // Sized for max_batch_size * n_heads * max_splits * (2 + head_dim) floats.
    {
        int nh = cfg.n_heads;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
        // Size splits proportional to max context blocks, capped at 128 (the
        // GQA tile kernel runs grid.y = n_kv_heads instead of n_heads and
        // recovers parallelism through the split count — see
        // paged_attention_decode_fp8).
        int max_context_blocks = (max_tokens_ + kKVBlockSize - 1) / kKVBlockSize;
        int max_splits = std::min(128, std::max(1, max_context_blocks));
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

    // cuBLAS attention S-matrix workspace: [n_heads, attn_seq, attn_seq] FP16.
    // Only the materialized cuBLAS prefill fallback consumes this. On uniform-
    // shape models without learned sinks whose head_dim FA2 covers (128 always,
    // 256 behind attention.fa2_hd256), FP16-QK FA2 serves ALL prefill at-or-above
    // cuBLAS at every length (hd=128: Qwen3-Coder-30B NVFP4 ~parity pp512, +24%
    // pp1024, +52% pp2048, 2026-06-12; hd=256 rides the #930/#932 port) — the
    // buffer is dead weight there, so skip it (reclaims up to ~380 MiB at
    // batch8/ctx4096). Uniform per-layer shapes (GDN/Mamba2 hybrids: zeros on
    // non-attention layers) take FA2 too since the #932 single-shot refinement.
    // cuBLAS stays the reference for heterogeneous shapes (gemma-4 dual head_dim),
    // learned sinks (gpt-oss), hd=256 with fa2_hd256 off, and the explicit
    // fa2_fp16qk=never opt-out. See the run_attention dispatch (FA2 tried first).
    int hd_for_attn = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
    for (int x : cfg.head_dim_per_layer) {
        if (x > 0) {
            hd_for_attn = x;  // hybrids: first attention layer's head_dim
            break;
        }
    }
    const bool fa2_hd_ok = hd_for_attn == 128 ||
                           (hd_for_attn == 256 && runtime_config().attention.fa2_hd256);
    const bool fa2_serves_all_prefill = fa2_hd_ok &&
                                        runtime_config().attention.fa2_fp16qk != "never" &&
                                        attn_shapes_uniform() &&
                                        !model_->profile().is_gpt_oss;
    if (fa2_serves_all_prefill) {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (FP16-QK FA2 serves all hd=%d prefill — "
                     "no S-matrix needed)",
                     hd_for_attn);
    } else if (!skip_batch_dequant) {
        int nh = cfg.n_heads;
        // 256 MiB: cuBLAS handles short sequences (up to ~1448 attn_seq for
        // 32-head models). Longer sequences auto-route to FMHA via the
        // fmha_prefill_threshold. Reduced from 1024 to free ~768 MiB for KV.
        int cfg_mib = runtime_config().attention.attn_scores_mib;
        size_t kMaxAttnScoresMiB = (cfg_mib > 0) ? static_cast<size_t>(cfg_mib) : 256;
        size_t max_s_sz = kMaxAttnScoresMiB << 20;
        // max seq = sqrt(budget / (n_heads * sizeof(half)))
        int attn_seq = max_tokens_;
        size_t s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        if (s_sz > max_s_sz) {
            attn_seq = static_cast<int>(std::sqrt(static_cast<double>(max_s_sz) / (nh * sizeof(half))));
            attn_seq = (attn_seq / 16) * 16;  // round down to multiple of 16
            if (attn_seq < 32)
                attn_seq = 0;  // too small to be useful
            s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        }
        if (attn_seq > 0) {
            attn_scores_buf_ = vram_alloc(vram_alloc_, s_sz, "attn_scores");
            if (!attn_scores_buf_) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN(
                    "Failed to allocate cuBLAS attention S-matrix (%.1f MiB): %s — "
                    "will fall back to WMMA attention for prefill",
                    s_sz / (1024.0 * 1024.0), cudaGetErrorString(e));
                attn_scores_buf_size_ = 0;
            } else {
                attn_scores_buf_size_ = s_sz;
                int64_t s_shape[3] = {static_cast<int64_t>(nh), static_cast<int64_t>(attn_seq),
                                      static_cast<int64_t>(attn_seq)};
                attn_scores_ = Tensor(attn_scores_buf_, QType::F16, 3, s_shape, true);
                IMP_LOG_INFO("cuBLAS attention S-matrix: %.2f MiB (%d heads x %d x %d)",
                             s_sz / (1024.0 * 1024.0), nh, attn_seq, attn_seq);
            }
        }
    } else {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (VRAM-constrained, using tiled WMMA FMHA fallback)");
    }

    // Auto-derive fmha_prefill_threshold from S-matrix capacity. This only
    // governs the cuBLAS-vs-tiled-FMHA boundary on the configs that still use
    // the S-matrix (hd != 128, per-layer, sinks); on hd=128 FP16-QK FA2 is tried
    // first and the threshold is moot (cap=0 → threshold=1). The dispatch uses
    // `prefer_fmha = (n >= threshold)`, so the threshold is cap+1 — the chunk
    // with n == cap, for which the S-matrix fits exactly, belongs to cuBLAS.
    // (Historical note: the old "cuBLAS ~30% faster than FMHA at n == cap" was
    // measured against the pre-#653/#673/#674 tiled FMHA, NOT FP16-QK FA2; FA2
    // now matches cuBLAS at pp512 and beats it +24%/+52% at pp1024/pp2048 —
    // measured 2026-06-12.)
    if (runtime_config().attention.fmha_prefill_threshold == -1) {
        int auto_threshold = attn_scores_cap() > 0 ? attn_scores_cap() + 1 : 1;
        const_cast<RuntimeConfig::Attention&>(runtime_config().attention).fmha_prefill_threshold =
            auto_threshold;
        IMP_LOG_INFO("auto fmha_prefill_threshold = %d (S-matrix cap + 1)", auto_threshold);
    }

    // MoE dequant and staging buffers
    if (has_moe_) {
        int d = cfg.d_model;
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
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)", dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer. Its only
        // consumers are the `!packed.on_device` branches of the legacy MoE
        // forward, so skip it entirely when every packed expert tensor is
        // device-resident (the common all-on-device load).
        size_t max_expert_raw = 0;
        bool any_host_packed_experts = false;
        {
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                auto check = [&](const Tensor& p, QType qt) {
                    if (!p.data || p.ndim < 3)
                        return;
                    size_t rb = qtype_row_bytes(qt, p.shape[2]);
                    size_t expert_raw = static_cast<size_t>(p.shape[1]) * rb;
                    max_expert_raw = std::max(max_expert_raw, expert_raw);
                    if (!p.on_device)
                        any_host_packed_experts = true;
                };
                check(L.expert_up_packed, L.expert_up_packed.qtype);
                check(L.expert_down_packed, L.expert_down_packed.qtype);
                check(L.expert_gate_packed, L.expert_gate_packed.qtype);
            }
            if (max_expert_raw > 0 && any_host_packed_experts) {
                moe_.raw_staging_buf = vram_alloc(vram_alloc_, max_expert_raw, "moe_staging");
                if (!moe_.raw_staging_buf) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes)", max_expert_raw);
                    moe_.raw_staging_size = 0;
                } else {
                    moe_.raw_staging_size = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
                }
            } else if (max_expert_raw > 0) {
                IMP_LOG_INFO("MoE staging buffer: skipped (all packed experts on device)");
            }
        }

        // LRU expert cache: keeps recently-used host experts on GPU.
        // Only allocated when some experts reside on host (not all fit in VRAM).
        if (max_expert_raw > 0) {
            // gpt-oss exemption: its MXFP4 experts are host-resident here only
            // transiently — pre_dequant converts them to on-device NVFP4 +
            // CUTLASS-grouped before the first forward. They are never
            // host-offloaded at runtime, so the LRU cache must not be allocated
            // (it would shadow the converted device experts → garbage output).
            bool has_host_experts = false;
            if (!model_->profile().is_gpt_oss) {
                for (int li = 0; li < model_->n_layers(); li++) {
                    const auto& L = model_->layer(li);
                    if ((L.expert_up_packed.data && !L.expert_up_packed.on_device) ||
                        (L.expert_down_packed.data && !L.expert_down_packed.on_device) ||
                        (L.expert_gate_packed.data && !L.expert_gate_packed.on_device)) {
                        has_host_experts = true;
                        break;
                    }
                }
            }
            if (has_host_experts && !runtime_config().moe.no_expert_cache) {
                // Budget: proportional to free VRAM (15%) instead of flat cap.
                // KV cache + weight caches (FP8/NVFP4) need the remaining VRAM,
                // so expert cache must not over-commit.
                size_t free_mem = 0, total_mem = 0;
                vram_budget_mem_get_info(&free_mem, &total_mem);
                size_t safety = 128 << 20;  // 128 MiB reserve
                size_t budget = (free_mem > safety) ? free_mem - safety : 0;
                budget = static_cast<size_t>(budget * 0.15);  // 15% of available
                const auto& mcfg = model_->config();
                bool debug_parity = runtime_config().moe.expert_cache_debug_parity;
                if (expert_cache_.init(max_expert_raw, budget, vram_alloc_, mcfg.n_layers,
                                       mcfg.n_experts, debug_parity)) {
                    IMP_LOG_INFO("Expert LRU cache: %d slots (%.2f MiB / %.2f MiB budget)",
                                 expert_cache_.n_slots_,
                                 expert_cache_.n_slots_ * max_expert_raw / (1024.0 * 1024.0),
                                 budget / (1024.0 * 1024.0));
                }
            } else if (has_host_experts) {
                IMP_LOG_INFO("Expert LRU cache disabled via IMP_NO_EXPERT_CACHE (staging fallback)");
            }
        }

        // ST-NVFP4 experts run the CUTLASS 3.x grouped path for ALL prefill
        // sizes (StorageTier::CUTLASS_NVFP4 covers every expert on healthy
        // loads); the buffers below only back the GGUF-family batch paths
        // (FP16-cache batch, Q6_K FP8 batch, IMMA Q8 staging) and the
        // post-CUTLASS NVFP4→FP16 dequant fallback. Skipping them frees
        // ~640 MiB on Qwen3-30B-A3B-NVFP4 (3.5 GiB free at load) — the
        // pathological CUTLASS-decline case falls through to the legacy
        // per-expert path (slow but correct).
        bool experts_st_nvfp4 = true;
        for (int i = 0; i < cfg.n_layers && experts_st_nvfp4; i++) {
            const auto& L = model_->layer(i);
            if (L.expert_up_packed.data && L.expert_up_packed.qtype != QType::NVFP4)
                experts_st_nvfp4 = false;
            if (L.expert_down_packed.data && L.expert_down_packed.qtype != QType::NVFP4)
                experts_st_nvfp4 = false;
        }
        if (experts_st_nvfp4) {
            IMP_LOG_INFO("MoE batch dequant + fp32_down buffers: skipped "
                         "(ST-NVFP4 experts — CUTLASS grouped prefill needs neither)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
            moe_.fp32_down_buf = nullptr;
            moe_.fp32_down_buf_size = 0;
        } else
        // Batch dequant buffer: sized for a chunk of experts (L2-resident strategy).
        // We dequant a chunk of experts to FP16, then immediately GEMM while the
        // FP16 data is still warm in L2 cache (~96 MB on RTX 5090). This avoids
        // writing the FP16 intermediate to DRAM entirely, saving ~5x DRAM traffic.
        // Skip allocation if experts are on host (batch dequant only useful for on-device experts).
        if (!skip_batch_dequant) {
            // Cap at the actual remaining free VRAM minus a reserve for KV
            // cache + workspaces (init_kv_cache runs after this and sees what's
            // left). On Nemotron-H NVFP4 (32 GiB GPU, 22 GiB model) the full
            // n_experts target hit ~1.2 GiB and starved the KV cache → 16-block
            // floor → long-prompt hang. Leaving ≥ 1 GiB free here covers
            // vram_budget reserve (~768 MiB) plus a useful KV cache.
            size_t free_now = 0, total_now = 0;
            vram_budget_mem_get_info(&free_now, &total_now);
            constexpr size_t kPostBufReserve = 1024ULL * 1024 * 1024;
            size_t cap_bytes = (free_now > kPostBufReserve) ? (free_now - kPostBufReserve) : 0;

            int targets[] = {cfg.n_experts, cfg.n_experts / 2, 32, 16};
            bool allocated = false;
            for (int ne_try : targets) {
                if (ne_try <= 0)
                    continue;
                ne_try = std::min(ne_try, cfg.n_experts);
                size_t sz = static_cast<size_t>(ne_try) * eff * d * sizeof(half);
                if (cap_bytes > 0 && sz > cap_bytes) {
                    IMP_LOG_DEBUG("MoE dequant: skipping %d experts (%.0f MiB > cap %.0f MiB)", ne_try,
                                  sz / (1024.0 * 1024.0), cap_bytes / (1024.0 * 1024.0));
                    continue;
                }
                moe_.batch_dequant_buf = vram_alloc(vram_alloc_, sz, "moe_batch_dequant");
                if (!moe_.batch_dequant_buf) {
                    IMP_LOG_DEBUG("MoE dequant buf alloc failed for %d experts", ne_try);
                    continue;
                }
                moe_.batch_dequant_buf_size = sz;
                allocated = true;
                IMP_LOG_INFO("MoE batch dequant buffer: %.2f MiB (%d experts)", sz / (1024.0 * 1024.0),
                             ne_try);
                break;
            }
            if (!allocated) {
                IMP_LOG_INFO("MoE batch dequant buffer: skipped (VRAM insufficient)");
                moe_.batch_dequant_buf = nullptr;
                moe_.batch_dequant_buf_size = 0;
            }
            // Pre-allocate FP32 down-projection scratch (drops per-call
            // cudaMallocAsync at executor_forward_moe.cu:1080). Worst-case sizing
            // matches the per-call: expanded = max_tokens × top_k, d = d_model.
            // Skipped if VRAM insufficient — forward pass falls back to lazy alloc.
            {
                size_t fp32_sz = static_cast<size_t>(max_tokens_) *
                                 static_cast<size_t>(cfg.n_experts_active) *
                                 static_cast<size_t>(d) * sizeof(float);
                size_t free_bytes = 0, total_bytes = 0;
                vram_budget_mem_get_info(&free_bytes, &total_bytes);
                constexpr size_t kReserve = 256ULL * 1024 * 1024;
                if (fp32_sz > 0 && free_bytes > fp32_sz + kReserve) {
                    moe_.fp32_down_buf = vram_alloc(vram_alloc_, fp32_sz, "moe_fp32_down");
                    if (moe_.fp32_down_buf) {
                        moe_.fp32_down_buf_size = fp32_sz;
                        IMP_LOG_INFO("MoE fp32_down scratch: %.2f MiB",
                                     fp32_sz / (1024.0 * 1024.0));
                    } else {
                        moe_.fp32_down_buf = nullptr;
                        moe_.fp32_down_buf_size = 0;
                    }
                }
            }
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
        }

        // CUTLASS 3.x NVFP4 grouped activation staging. Auto-used for prefill
        // (n > 1) on NVFP4-prequant MoE models — 4.6× speedup on Qwen3-Coder-30B-A3B-FP4.
        // Decode (n == 1) keeps using the legacy per-expert GEMV. Max K = d_model,
        // max_expanded = max_tokens * top_k. ~38 MiB on 128-experts / 4096 tokens.
        if (cfg.n_experts > 0) {
            int top_k = cfg.n_experts_active;
            int max_expanded = max_tokens_ * top_k;
            int max_K = std::max(d, eff);  // gate/up use d (=d_model), down uses eff
            // Packed FP4: 1 byte per 2 values
            size_t packed_sz = static_cast<size_t>(max_expanded) * max_K / 2;
            // SFA worst-case: each expert's rows pad to SfAtom row tile (128),
            // so sum over ne of cutlass_nvfp4_sf_size(M_i, max_K) ≤
            // cutlass_nvfp4_sf_size(max_expanded + 128*ne, max_K).
            size_t sf_sz = cutlass_nvfp4_sf_size(max_expanded + 128 * cfg.n_experts, max_K);
            moe_.cutlass3x_packed = vram_alloc(vram_alloc_, packed_sz, "moe_3x_packed");
            moe_.cutlass3x_sf = vram_alloc(vram_alloc_, sf_sz, "moe_3x_sf");
            if (moe_.cutlass3x_packed && moe_.cutlass3x_sf) {
                moe_.cutlass3x_packed_size = packed_sz;
                moe_.cutlass3x_sf_size = sf_sz;
                // Device array of per-expert SFA base pointers for the fused quantize kernel.
                size_t sfa_ptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(uint8_t*);
                cudaError_t err = cudaMalloc(&moe_.cutlass3x_sfa_ptrs, sfa_ptr_bytes);
                if (err == cudaSuccess) {
                    moe_.cutlass3x_sfa_ptrs_count = cfg.n_experts;
                } else {
                    IMP_LOG_WARN("CUTLASS 3.x SFA pointer array alloc failed: %s", cudaGetErrorString(err));
                    moe_.cutlass3x_sfa_ptrs = nullptr;
                }
                IMP_LOG_INFO("CUTLASS 3.x MoE staging: %.2f MiB (packed=%.2f, sf=%.2f) max_expanded=%d",
                             (packed_sz + sf_sz) / (1024.0 * 1024.0), packed_sz / (1024.0 * 1024.0),
                             sf_sz / (1024.0 * 1024.0), max_expanded);
            } else {
                IMP_LOG_WARN("CUTLASS 3.x MoE staging: allocation failed, path disabled");
                if (moe_.cutlass3x_packed) {
                    vram_free(vram_alloc_, moe_.cutlass3x_packed);
                    moe_.cutlass3x_packed = nullptr;
                }
                if (moe_.cutlass3x_sf) {
                    vram_free(vram_alloc_, moe_.cutlass3x_sf);
                    moe_.cutlass3x_sf = nullptr;
                }
            }
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

            // Per-expert device-resident token-count buffer (n_experts × 4 bytes).
            // Populated each forward by compute_M_per_from_offsets_device, replacing
            // the host D2H+sync+loop pattern in the MoE prefill dispatch path.
            size_t m_per_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(int32_t);
            err = cudaMalloc(&moe_.d_M_per, m_per_bytes);
            if (err == cudaSuccess) {
                moe_.d_M_per_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE d_M_per alloc failed: %s", cudaGetErrorString(err));
                moe_.d_M_per = nullptr;
                moe_.d_M_per_count = 0;
            }

            // Compact-alpha output buffer + active-expert counter. Populated by
            // compact_alpha_active. Sized for max n_experts (only first d_na
            // entries used at dispatch).
            err = cudaMalloc(&moe_.d_alpha_compact,
                             static_cast<size_t>(cfg.n_experts) * sizeof(float));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_alpha_compact alloc failed: %s",
                              cudaGetErrorString(err));
                moe_.d_alpha_compact = nullptr;
            }
            err = cudaMalloc(&moe_.d_na, sizeof(int32_t));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_na alloc failed: %s", cudaGetErrorString(err));
                moe_.d_na = nullptr;
            }

            // SFA byte-offsets prefix sum (Phase 3 staging). n_experts+1 int64
            // = trivial (<2 KiB for 128 experts).
            err = cudaMalloc(&moe_.d_sfa_offsets,
                             static_cast<size_t>(cfg.n_experts + 1) * sizeof(int64_t));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_sfa_offsets alloc failed: %s",
                              cudaGetErrorString(err));
                moe_.d_sfa_offsets = nullptr;
            }

            // Phase 3c-full Step 1 — device-args ptr/alpha caches. n_experts ×
            // (2 × sizeof(void*) + sizeof(float)) ≈ 2.5 KiB for 128 experts.
            err = cudaMalloc(&moe_.d_B_ptrs_cache,
                             static_cast<size_t>(cfg.n_experts) * sizeof(const void*));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_B_ptrs_cache alloc failed: %s",
                              cudaGetErrorString(err));
                moe_.d_B_ptrs_cache = nullptr;
            }
            err = cudaMalloc(&moe_.d_SFB_ptrs_cache,
                             static_cast<size_t>(cfg.n_experts) * sizeof(const void*));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_SFB_ptrs_cache alloc failed: %s",
                              cudaGetErrorString(err));
                moe_.d_SFB_ptrs_cache = nullptr;
            }
            err = cudaMalloc(&moe_.d_alpha_full,
                             static_cast<size_t>(cfg.n_experts) * sizeof(float));
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE d_alpha_full alloc failed: %s",
                              cudaGetErrorString(err));
                moe_.d_alpha_full = nullptr;
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
        if (cfg.d_ff > 0)
            max_dim = std::max(max_dim, cfg.d_ff);
        max_dim = std::max(max_dim,
                           cfg.n_heads * (cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads)));
        // SSM dimensions
        if (cfg.ssm_inner_size > 0) {
            int conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
            int ssm_in_dim = cfg.ssm_inner_size + conv_ch + cfg.ssm_dt_rank;
            int gdn_fused_total = conv_ch + cfg.ssm_inner_size + 2 * cfg.ssm_dt_rank;
            max_dim = std::max(max_dim, ssm_in_dim);
            max_dim = std::max(max_dim, gdn_fused_total);
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
            int max_n = static_cast<int>(qscratch_.fp8_act_size);   // max elements
            int threads_needed = (max_n + 3) / 4;                   // kElemsPerThread=4
            qscratch_.fp8_max_grid = (threads_needed + 255) / 256;  // kBlockSize=256
            cudaError_t e1 = cudaMalloc(&qscratch_.d_fp8_block_maxes,
                                        static_cast<size_t>(qscratch_.fp8_max_grid) * sizeof(float));
            cudaError_t e2 = cudaMalloc(&qscratch_.d_fp8_absmax, sizeof(float));
            if (e1 != cudaSuccess || e2 != cudaSuccess || !qscratch_.d_fp8_block_maxes ||
                !qscratch_.d_fp8_absmax) {
                IMP_LOG_WARN("Failed to allocate FP8 reduction buffers — will use sync path");
                if (qscratch_.d_fp8_block_maxes) {
                    cudaFree(qscratch_.d_fp8_block_maxes);
                    qscratch_.d_fp8_block_maxes = nullptr;
                }
                if (qscratch_.d_fp8_absmax) {
                    cudaFree(qscratch_.d_fp8_absmax);
                    qscratch_.d_fp8_absmax = nullptr;
                }
                qscratch_.fp8_max_grid = 0;
            }
            IMP_LOG_INFO(
                "FP8 activation scratch: %.2f MiB (max_tokens=%d, max_dim=%d, async reduction grid=%d)",
                qscratch_.fp8_act_size / (1024.0 * 1024.0), max_tokens_, max_dim, qscratch_.fp8_max_grid);
        }
    }

    // CUTLASS sm_120 NVFP4 activation buffers: pre-allocate for max prefill dimensions.
    // Only needed when NVFP4 decode is active and sm_120 is available.
    if (wcache_.nvfp4_decode_mode > 0 && cutlass_sm120_nvfp4_available()) {
        int max_k = 0;
        int max_n = 0;
        // NVFP4 prequant tensors carry K_packed = K_logical/2 in shape[1].
        // Phase 0b promote runs AFTER this scratch-sizing pass, so the
        // Tensor.qtype is still the on-disk byte type (INT8/U8) here, not
        // QType::NVFP4. Use the model-level cfg flag to detect the format
        // and scale up K accordingly. Without this, layer 11's o_proj on
        // Gemma-4 (K_packed=4096, K_logical=8192) blows past sf scratch
        // (1 MiB) and cudaMemsetAsync poisons the stream with invalid
        // argument, falling back to dequant→cuBLAS for the rest of the
        // forward pass — output collapses to "<strong><strong>..." loops.
        const bool nvfp4_prequant_2d_packed = cfg.is_nvfp4_prequant;
        auto track_2d = [&](const Tensor& w) {
            if (w.data && w.ndim >= 2) {
                max_n = std::max(max_n, static_cast<int>(w.shape[0]));
                int K_dim = static_cast<int>(w.shape[1]);
                if (nvfp4_prequant_2d_packed || w.qtype == QType::NVFP4) {
                    K_dim *= 2;
                }
                max_k = std::max(max_k, K_dim);
            }
        };
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                                  &L.w_up_shared, &L.w_down_shared, &L.ssm_in, &L.ssm_out}) {
                track_2d(*w);
            }
            // MoE expert weights: per-expert tensors are [N, K] 2D. The 3D
            // packed buffers expert_*_packed reshape to [n_experts, N, K] —
            // we only care about (N, K) for activation scratch sizing.
            // For Gemma-4-26B-A4B and similar MoE prequant SafeTensors, the
            // expert down proj has K=8192 (d_ff) while the per-layer scan
            // above only sees K=2816 (d_model) on attention/shared weights.
            // Without this, M=3085+ prefill blows out the SF scratch buffer
            // (sf_bytes=1.6 MiB > scratch=1 MiB), cudaMemsetAsync returns
            // invalid argument, and the stream poisons every downstream
            // kernel — output collapses to <strong><strong>... loops.
            for (const auto* expert_vec : {&L.expert_w_gate, &L.expert_w_up, &L.expert_w_down}) {
                if (!expert_vec->empty())
                    track_2d((*expert_vec)[0]);
            }
            // 3D packed expert buffers: [n_experts, N, K] (or [n_experts, N, K_packed]
            // for NVFP4 prequant where shape[2] is K_packed = K_logical/2).
            for (const auto* w : {&L.expert_gate_packed, &L.expert_up_packed, &L.expert_down_packed}) {
                if (w->data && w->ndim >= 3) {
                    max_n = std::max(max_n, static_cast<int>(w->shape[1]));
                    int K_dim = static_cast<int>(w->shape[2]);
                    // NVFP4 prequant packs two FP4 nibbles per byte, so the
                    // logical K (and thus the GEMM activation K) is 2× shape[2].
                    if (w->qtype == QType::NVFP4)
                        K_dim *= 2;
                    max_k = std::max(max_k, K_dim);
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

            qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size,
                                                    "cutlass_act_data");
            qscratch_.cutlass_act_sf = vram_alloc(vram_alloc_, qscratch_.cutlass_act_sf_size,
                                                  "cutlass_act_sf");
            qscratch_.cutlass_workspace = (qscratch_.cutlass_workspace_size > 0)
                                              ? vram_alloc(vram_alloc_, qscratch_.cutlass_workspace_size,
                                                           "cutlass_workspace")
                                              : nullptr;
            if (!qscratch_.cutlass_act_data || !qscratch_.cutlass_act_sf ||
                (qscratch_.cutlass_workspace_size > 0 && !qscratch_.cutlass_workspace)) {
                IMP_LOG_WARN(
                    "Failed to allocate CUTLASS NVFP4 activation buffers, native FP4 prefill disabled");
                if (qscratch_.cutlass_act_data) {
                    vram_free(vram_alloc_, qscratch_.cutlass_act_data);
                    qscratch_.cutlass_act_data = nullptr;
                }
                if (qscratch_.cutlass_act_sf) {
                    vram_free(vram_alloc_, qscratch_.cutlass_act_sf);
                    qscratch_.cutlass_act_sf = nullptr;
                }
                if (qscratch_.cutlass_workspace) {
                    vram_free(vram_alloc_, qscratch_.cutlass_workspace);
                    qscratch_.cutlass_workspace = nullptr;
                }
                qscratch_.cutlass_act_data_size = 0;
                qscratch_.cutlass_act_sf_size = 0;
                qscratch_.cutlass_workspace_size = 0;
            } else {
                // Pre-zero the SfAtom scale workspace once. The per-call
                // quantize_fp16_nvfp4_cutlass_kernel only writes the M × K_groups
                // valid SF cells; SfAtom padding bytes remain whatever they were
                // before. Zeroing once here lets quantize_fp16_to_nvfp4_cutlass
                // skip its per-call cudaMemsetAsync (saves ~1 launch per CUTLASS
                // NVFP4 GEMM, ~6720 calls / 100 ms in Llama Q8 W1 prefill). Sync
                // because executor init may finish before the first stream-bound
                // call uses this buffer.
                IMP_CUDA_CHECK_LOG(cudaMemset(qscratch_.cutlass_act_sf, 0,
                                              qscratch_.cutlass_act_sf_size));
                IMP_LOG_INFO("CUTLASS NVFP4 activation scratch: %.2f MiB (data=%.2f, sf=%.2f, ws=%.2f)",
                             (qscratch_.cutlass_act_data_size + qscratch_.cutlass_act_sf_size +
                              qscratch_.cutlass_workspace_size) /
                                 (1024.0 * 1024.0),
                             qscratch_.cutlass_act_data_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_sf_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_workspace_size / (1024.0 * 1024.0));

                // MXFP4 activation buffers: shares packed data with NVFP4, only needs
                // separate UE8M0 scale factors (SFVecSize=32 vs NVFP4's 16).
                // Only allocate when the model actually carries MXFP4 weights
                // (or attention.mxfp4 prefill is opt-in enabled). hardware-
                // availability (cutlass_sm120_mxfp4_available) is necessary
                // but not sufficient — was allocating ~0.5 MiB on every NVFP4
                // model regardless of whether MXFP4 path would ever execute.
                bool has_mxfp4_weights = false;
                for (int i = 0; i < cfg.n_layers && !has_mxfp4_weights; i++) {
                    const auto& L = model_->layer(i);
                    if (L.wq.qtype == QType::MXFP4 || L.w_gate.qtype == QType::MXFP4 ||
                        L.w_up.qtype == QType::MXFP4 || L.w_down.qtype == QType::MXFP4 ||
                        L.ssm_in.qtype == QType::MXFP4 || L.ssm_out.qtype == QType::MXFP4 ||
                        L.expert_gate_packed.qtype == QType::MXFP4 ||
                        L.expert_up_packed.qtype == QType::MXFP4 ||
                        L.expert_down_packed.qtype == QType::MXFP4) {
                        has_mxfp4_weights = true;
                    }
                }
                if (cutlass_sm120_mxfp4_available() &&
                    (has_mxfp4_weights || runtime_config().attention.mxfp4 == "always")) {
                    qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                    qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_, max_n,
                                                                                        max_k);
                    qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size,
                                                        "mxfp4_act_sf");
                    qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                                                    ? vram_alloc(vram_alloc_, qscratch_.mxfp4_workspace_size,
                                                                 "mxfp4_workspace")
                                                    : nullptr;
                    if (!qscratch_.mxfp4_act_sf ||
                        (qscratch_.mxfp4_workspace_size > 0 && !qscratch_.mxfp4_workspace)) {
                        IMP_LOG_WARN("Failed to allocate MXFP4 activation buffers, MXFP4 prefill disabled");
                        if (qscratch_.mxfp4_act_sf) {
                            vram_free(vram_alloc_, qscratch_.mxfp4_act_sf);
                            qscratch_.mxfp4_act_sf = nullptr;
                        }
                        if (qscratch_.mxfp4_workspace) {
                            vram_free(vram_alloc_, qscratch_.mxfp4_workspace);
                            qscratch_.mxfp4_workspace = nullptr;
                        }
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

bool GraphExecutor::allocate_nvfp4_dequant_workspace() {
    // Iterate populated NVFP4 weight caches to find the largest single dequant
    // target. The gemm_nvfp4 fallback dequantizes one weight at a time to FP16
    // — the workspace only needs to fit the LARGEST single weight (N × K × 2
    // bytes). MoE caches contribute per-expert weights (callers slice one
    // expert at a time into a synthetic NvFP4QuantResult before invoking
    // gemm_nvfp4 — see executor_forward_moe.cu).
    // Weights above kCap don't get a workspace (an NVFP4 LM head would be
    // ~1.5 GiB) — track the covered maximum separately so ONE oversized
    // tensor no longer disables the workspace for every other weight (the
    // all-or-nothing skip left Nemotron with no workspace at all, which is
    // what made its verify-chunk capture fail: #855 census crash class).
    constexpr size_t kCap = 512ULL * 1024 * 1024;  // 512 MiB
    size_t max_bytes = 0;         // largest dequant target overall
    size_t covered_bytes = 0;     // largest target we will actually cover
    auto consider = [&](int64_t N, int64_t K) {
        size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(half);
        if (bytes > max_bytes)
            max_bytes = bytes;
        if (bytes <= kCap && bytes > covered_bytes)
            covered_bytes = bytes;
    };
    for (const auto& [ptr, qr] : wcache_.nvfp4)
        consider(qr.N, qr.K);
    for (const auto& [ptr, moe] : wcache_.nvfp4_moe)
        consider(moe.N, moe.K);  // single-expert dequant slice
    for (const auto& [ptr, cw] : wcache_.cutlass_nvfp4)
        consider(cw.N, cw.K);
    // SafeTensors NVFP4 prequant: per-tensor and per-expert NVFP4 storage lives
    // on the Layer struct (qtype=NVFP4 with scales sidecar), not in wcache_.
    // The gemm_nvfp4 fallback (executor_forward_moe.cu line ~2369 for MoE
    // experts, and executor_kernels.cu:2052 for dense weights including shared
    // experts) constructs a synthetic NvFP4QuantResult with N=t.shape[0],
    // K=t.shape[1]*2 (logical, since shape[1] is FP4-packed bytes). Iterate
    // every NVFP4 tensor on the layer to find the largest dequant target.
    if (model_ != nullptr) {
        const int n_layers = model_->n_layers();
        for (int li = 0; li < n_layers; ++li) {
            const auto& L = model_->layer(li);
            auto consider_t = [&](const Tensor& t) {
                if (t.qtype != QType::NVFP4 || t.data == nullptr || t.ndim < 2)
                    return;
                // 2D dense weight [N, K_packed] → dequant target N × K_logical
                // 3D MoE packed [n_experts, N, K_packed] → per-expert N × K_logical
                int64_t N = (t.ndim == 2) ? t.shape[0] : t.shape[1];
                int64_t K_packed = (t.ndim == 2) ? t.shape[1] : t.shape[2];
                consider(N, K_packed * 2);
            };
            // Attention weights
            consider_t(L.wq);
            consider_t(L.wk);
            consider_t(L.wv);
            consider_t(L.wo);
            // Dense MLP (used for shared experts in MoE models like Gemma-4)
            consider_t(L.w_gate);
            consider_t(L.w_up);
            consider_t(L.w_down);
            consider_t(L.w_gate_shared);
            consider_t(L.w_up_shared);
            consider_t(L.w_down_shared);
            // MoE per-expert vectors
            for (const auto& t : L.expert_w_gate) consider_t(t);
            for (const auto& t : L.expert_w_up) consider_t(t);
            for (const auto& t : L.expert_w_down) consider_t(t);
            // MoE packed 3D
            consider_t(L.expert_gate_packed);
            consider_t(L.expert_up_packed);
            consider_t(L.expert_down_packed);
        }
    }
    if (max_bytes == 0) {
        // No NVFP4 weights — nothing to do. The fallback won't fire.
        return true;
    }

    // Weights beyond the workspace stay on the lazy-cudaMalloc fallback on
    // non-captured streams; hitting one of them under capture now throws
    // (gemm_nvfp4) and the capture fails cleanly.
    if (max_bytes > kCap) {
        IMP_LOG_WARN(
            "gemm_nvfp4 dequant workspace: largest NVFP4 weight is %.2f MiB > %.0f MiB cap "
            "(covered: %.2f MiB) — prefill graph capture disabled (a captured M>1 fallback "
            "on the oversized weight fails loud).",
            max_bytes / (1024.0 * 1024.0), kCap / (1024.0 * 1024.0),
            covered_bytes / (1024.0 * 1024.0));
        nvfp4_dequant_uncapturable_ = true;  // scheduler will skip prefill-graph capture
    }
    if (covered_bytes == 0)
        return !nvfp4_dequant_uncapturable_;

    nvfp4_dequant_ws_buf_ = vram_alloc(vram_alloc_, covered_bytes, "nvfp4_dequant");
    if (!nvfp4_dequant_ws_buf_) {
        IMP_LOG_WARN("gemm_nvfp4 dequant workspace: alloc failed (%zu bytes) — prefill graph "
                     "capture disabled (M>1 fallback runs eager).",
                     covered_bytes);
        nvfp4_dequant_ws_size_ = 0;
        nvfp4_dequant_uncapturable_ = true;
        return false;
    }
    nvfp4_dequant_ws_size_ = covered_bytes;
    set_nvfp4_dequant_workspace(nvfp4_dequant_ws_buf_, nvfp4_dequant_ws_size_);
    IMP_LOG_INFO(
        "gemm_nvfp4 dequant workspace: %.2f MiB (graph-safe M>1 fallback; "
        "covers dense=%zu, moe=%zu, cutlass=%zu cache entries + per-layer "
        "SafeTensors NVFP4 expert tensors)",
        covered_bytes / (1024.0 * 1024.0), wcache_.nvfp4.size(), wcache_.nvfp4_moe.size(),
        wcache_.cutlass_nvfp4.size());
    return !nvfp4_dequant_uncapturable_;
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
    if (lora_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(lora_scratch_));
        lora_scratch_ = nullptr;
        lora_scratch_sz_ = 0;
    }
    if (verify_argmax_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(verify_argmax_scratch_));
        verify_argmax_scratch_ = nullptr;
        verify_argmax_scratch_sz_ = 0;
    }
    if (verify_pen_counts_) {
        IMP_CUDA_CHECK_LOG(cudaFree(verify_pen_counts_));
        verify_pen_counts_ = nullptr;
        verify_pen_counts_cap_ = 0;
    }
    // Helper: free through VRAMAllocator if pointer was tracked, else cudaFree.
    auto vfree = [this](void*& p) {
        if (p) {
            vram_free(vram_alloc_, p);
            p = nullptr;
        }
    };

    // Free LongRoPE frequency tables
    if (longrope_short_freqs_) {
        IMP_CUDA_CHECK_LOG(cudaFree(longrope_short_freqs_));
        longrope_short_freqs_ = nullptr;
    }
    if (longrope_long_freqs_) {
        IMP_CUDA_CHECK_LOG(cudaFree(longrope_long_freqs_));
        longrope_long_freqs_ = nullptr;
    }
    longrope_n_pairs_ = 0;
    longrope_orig_max_pos_ = 0;

    // Free all weight caches (FP16, FP8, NVFP4, CUTLASS, fused KV/gate+up, migrated/overflow)
    {
        // Registry-owned overlays (Phase 4.2): fused_kv / fused_gate_up
        // storage is now owned by the WeightRegistry handles, not wcache_.
        // The maps below are kept empty by `pre_dequant_weights`. This helper
        // frees any handle whose `owned_bytes > 0`.
        registry_.free_owned_storage(vram_alloc_);
        // Legacy loops — both maps must be empty now. Kept as a defensive
        // fallback in case a code path writes to them without going through
        // the Phase-4 registration helper.
        for (auto& [idx, tensor] : wcache_.fused_kv)
            if (tensor.data)
                vram_free(vram_alloc_, tensor.data);
        wcache_.fused_kv.clear();
        for (auto& [idx, tensor] : wcache_.fused_gate_up)
            if (tensor.data)
                vram_free(vram_alloc_, tensor.data);
        wcache_.fused_gate_up.clear();
        // FP16 cache — entries from the MXFP4 → FP16 decode fallback are
        // sub-pointers into wcache_.fp16_bulk_data (single bulk cudaMalloc);
        // skip the per-tensor cudaFree for those, free the bulk once below.
        for (auto& [ptr, tensor] : wcache_.fp16) {
            if (!tensor.data)
                continue;
            bool in_bulk = wcache_.fp16_bulk_data &&
                           reinterpret_cast<uintptr_t>(tensor.data) >=
                               reinterpret_cast<uintptr_t>(wcache_.fp16_bulk_data) &&
                           reinterpret_cast<uintptr_t>(tensor.data) <
                               reinterpret_cast<uintptr_t>(wcache_.fp16_bulk_data) +
                                   wcache_.fp16_bulk_data_size;
            if (!in_bulk)
                vram_free(vram_alloc_, tensor.data);
        }
        wcache_.fp16.clear();
        wcache_.fp16_bytes = 0;
        if (wcache_.fp16_bulk_data) {
            IMP_CUDA_CHECK_LOG(cudaFree(wcache_.fp16_bulk_data));
            wcache_.fp16_bulk_data = nullptr;
            wcache_.fp16_bulk_data_size = 0;
        }
        // NVFP4 decode cache
        for (auto& [ptr, result] : wcache_.nvfp4)
            free_nvfp4_result(result);
        wcache_.nvfp4.clear();
        wcache_.nvfp4_bytes = 0;
        // NVFP4 MoE expert cache
        for (auto& [ptr, result] : wcache_.nvfp4_moe)
            free_nvfp4_moe_result(result);
        wcache_.nvfp4_moe.clear();
        wcache_.nvfp4_moe_bytes = 0;
        // CUTLASS NVFP4 cache. Entries' scale_factors are sub-pointers into
        // cutlass_sf_slab (sf_borrowed=true) so free_cutlass_nvfp4_weight skips
        // the per-tensor cudaFree; the slab is freed once below.
        for (auto& [ptr, cw] : wcache_.cutlass_nvfp4)
            free_cutlass_nvfp4_weight(cw);
        wcache_.cutlass_nvfp4.clear();
        wcache_.cutlass_nvfp4_bytes = 0;
        if (wcache_.cutlass_sf_slab) {
            IMP_CUDA_CHECK_LOG(cudaFree(wcache_.cutlass_sf_slab));
            wcache_.cutlass_sf_slab = nullptr;
            wcache_.cutlass_sf_slab_size = 0;
        }
        // CUTLASS MXFP4 cache
        for (auto& [ptr, mw] : wcache_.cutlass_mxfp4)
            free_cutlass_mxfp4_weight(mw);
        wcache_.cutlass_mxfp4.clear();
        wcache_.cutlass_mxfp4_bytes = 0;
        // FP8 cache (entries may point into bulk buffers — free entry data only if not in bulk)
        for (auto& [ptr, entry] : wcache_.fp8) {
            if (entry.weight.data) {
                bool in_migrated = wcache_.fp8_migrated_data &&
                                   reinterpret_cast<uintptr_t>(entry.weight.data) >=
                                       reinterpret_cast<uintptr_t>(wcache_.fp8_migrated_data) &&
                                   reinterpret_cast<uintptr_t>(entry.weight.data) <
                                       reinterpret_cast<uintptr_t>(wcache_.fp8_migrated_data) +
                                           wcache_.fp8_migrated_data_size;
                bool in_overflow = wcache_.fp8_overflow_data &&
                                   reinterpret_cast<uintptr_t>(entry.weight.data) >=
                                       reinterpret_cast<uintptr_t>(wcache_.fp8_overflow_data) &&
                                   reinterpret_cast<uintptr_t>(entry.weight.data) <
                                       reinterpret_cast<uintptr_t>(wcache_.fp8_overflow_data) +
                                           wcache_.fp8_overflow_data_size;
                if (!in_migrated && !in_overflow)
                    cudaFree(entry.weight.data);
            }
            if (entry.d_scale) {
                bool in_migrated = wcache_.fp8_migrated_scales &&
                                   entry.d_scale >= wcache_.fp8_migrated_scales &&
                                   entry.d_scale < wcache_.fp8_migrated_scales + wcache_.fp8_migrated_count;
                bool in_overflow = wcache_.fp8_overflow_scales &&
                                   entry.d_scale >= wcache_.fp8_overflow_scales &&
                                   entry.d_scale < wcache_.fp8_overflow_scales + wcache_.fp8_overflow_count;
                if (!in_migrated && !in_overflow)
                    cudaFree(entry.d_scale);
            }
        }
        wcache_.fp8.clear();
        wcache_.fp8_bytes = 0;
        // FP8 bulk buffers
        if (wcache_.fp8_migrated_scales) {
            IMP_CUDA_CHECK_LOG(cudaFree(wcache_.fp8_migrated_scales));
            wcache_.fp8_migrated_scales = nullptr;
            wcache_.fp8_migrated_count = 0;
        }
        if (wcache_.fp8_migrated_data) {
            vram_free(vram_alloc_, wcache_.fp8_migrated_data);
            wcache_.fp8_migrated_data = nullptr;
            wcache_.fp8_migrated_data_size = 0;
        }
        if (wcache_.fp8_overflow_scales) {
            IMP_CUDA_CHECK_LOG(cudaFree(wcache_.fp8_overflow_scales));
            wcache_.fp8_overflow_scales = nullptr;
            wcache_.fp8_overflow_count = 0;
        }
        if (wcache_.fp8_overflow_data) {
            vram_free(vram_alloc_, wcache_.fp8_overflow_data);
            wcache_.fp8_overflow_data = nullptr;
            wcache_.fp8_overflow_data_size = 0;
        }
    }

    qscratch_.free(vram_alloc_);

    moe_.free(vram_alloc_);
    expert_cache_.destroy();

    // Free gemm_nvfp4 dequant workspace and unregister from the free function.
    if (nvfp4_dequant_ws_buf_) {
        set_nvfp4_dequant_workspace(nullptr, 0);
        vram_free(vram_alloc_, nvfp4_dequant_ws_buf_);
        nvfp4_dequant_ws_buf_ = nullptr;
        nvfp4_dequant_ws_size_ = 0;
    }
    if (lm_head_cutlass_ready_) {
        // Frees only the owned SfAtom scales; .data is borrowed from the decode cache.
        free_cutlass_nvfp4_weight(lm_head_cutlass_);
        lm_head_cutlass_ = {};
        lm_head_cutlass_ready_ = false;
    }
    if (mla_absorb_cache_) {
        IMP_CUDA_CHECK_LOG(cudaFree(mla_absorb_cache_));
        mla_absorb_cache_ = nullptr;
    }
    if (mla_absorb_scores_) {
        IMP_CUDA_CHECK_LOG(cudaFree(mla_absorb_scores_));
        mla_absorb_scores_ = nullptr;
    }
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
    if (chunk_capture_k_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_capture_k_));
        chunk_capture_k_ = nullptr;
    }
    if (chunk_capture_v_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_capture_v_));
        chunk_capture_v_ = nullptr;
    }
    chunk_capture_ctx_ = 0;
    if (chunk_eager_k_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_eager_k_));
        chunk_eager_k_ = nullptr;
    }
    if (chunk_eager_v_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_eager_v_));
        chunk_eager_v_ = nullptr;
    }
    chunk_eager_bytes_ = 0;
    ws_.free_buffers();  // shared + persistent workspace (Workspace-owned)
    vfree(fp32_accum_buf_);
    ssm_layer_map_.clear();
    initialized_ = false;
}

bool GraphExecutor::attn_shapes_uniform() const {
    const auto& cfg = model_->config();
    auto uniform = [](const std::vector<int>& v) {
        int ref = 0;
        for (int x : v) {
            if (x <= 0)
                continue;  // zeros mark non-attention layers (GDN/Mamba2 hybrids)
            if (ref == 0)
                ref = x;
            else if (x != ref)
                return false;
        }
        return true;
    };
    return uniform(cfg.head_dim_per_layer) && uniform(cfg.n_kv_heads_per_layer);
}

int GraphExecutor::max_safe_prefill_chunk(int offset, int desired, int kv_bs) const {
    const int s_cap = attn_scores_cap();
    // No S-matrix allocated: the chunked path is served by the O(n) FA2/FMHA
    // family (fa2_serves_all_prefill) — no capacity constraint applies here.
    if (s_cap <= 0 || desired <= 0)
        return desired;
    const auto& cfg = model_->config();
    const bool sinks = model_->profile().is_gpt_oss;
    const bool uniform = attn_shapes_uniform();
    const auto& att = runtime_config().attention;
    // Uniform attention head_dim (first nonzero per-layer value, else global).
    int hd_u = cfg.head_dim > 0 ? cfg.head_dim
                                : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
    for (int x : cfg.head_dim_per_layer) {
        if (x > 0) {
            hd_u = x;
            break;
        }
    }
    // Mirrors the chunked dispatch in executor_attention_prefill.cu:
    // FP16-QK FA2 serves every hd=128 chunk with no S-matrix.
    if (uniform && !sinks && hd_u == 128 && att.fa2_fp16qk != "never")
        return desired;
    // The tiled FMHA dispatch serves chunks whose ctx_len crosses the
    // threshold (and any chunk the S-matrix cannot hold) with no S-matrix.
    if (uniform && !sinks && att.fmha_prefill_threshold > 0 &&
        offset + desired >= att.fmha_prefill_threshold)
        return desired;
    // cuBLAS serves this chunk: n × (offset + n) ≤ s_cap² and n ≤ s_cap.
    // Solve the quadratic for n; floor to a KV-block multiple.
    const double cap2 = static_cast<double>(s_cap) * s_cap;
    const double disc = static_cast<double>(offset) * offset + 4.0 * cap2;
    int n_max = static_cast<int>((std::sqrt(disc) - offset) / 2.0);
    n_max = std::min(n_max, s_cap);
    if (kv_bs > 0)
        n_max = (n_max / kv_bs) * kv_bs;
    return std::min(desired, n_max);
}

bool GraphExecutor::chunk_capture_supported() const {
    const auto& cfg = model_->config();
    if (cfg.is_mla())
        return false;  // absorbed-decode latent cache writes are host-parameterized
    if (model_->profile().is_gpt_oss)
        return false;  // learned sinks — cuBLAS-only chunked attention
    if (longrope_short_freqs_ != nullptr)
        return false;  // host branch on max_context_len picks the freq table
    if (!attn_shapes_uniform())
        return false;
    int hd_u = cfg.head_dim > 0 ? cfg.head_dim
                                : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
    for (int x : cfg.head_dim_per_layer) {
        if (x > 0) {
            hd_u = x;
            break;
        }
    }
    // FP16-QK FA2 is the only device-length chunked attention kernel. hd=256
    // rides the #930 port (d_kv_len is a runtime kernel argument shared by
    // every instance); the GDN/Mamba2 recurrent kernels stop state updates at
    // the device chunk length (d_chunk_len), so hd=256 hybrids (Qwen3.5/3.6)
    // are capture-eligible when the fa2_hd256 flag is on.
    if (hd_u != 128 && !(hd_u == 256 && runtime_config().attention.fa2_hd256))
        return false;
    if (runtime_config().attention.fa2_fp16qk == "never")
        return false;
    // MoE: only the CUTLASS 3.x device-args grouped path records into a
    // graph without host-side routing reads — every other MoE prefill path
    // does a D2H+sync per layer to size the expert GEMMs (capture-illegal;
    // guarded by moe_host_args_capture_guard). Require the static
    // device-args conditions for every MoE layer. Models whose experts live
    // in the data-borrow decode slabs (e.g. Nemotron-H NVFP4: expert ids not
    // in the CUTLASS tier) fall out here until a device-args grouped GEMM
    // exists for that layout (#847 follow-up).
    bool any_moe = false;
    for (int i = 0; i < cfg.n_layers && !any_moe; ++i)
        any_moe = layer_has_moe(i);
    // moe.skip (debug) removes the MoE pass from eager and capture alike —
    // the recorded graph stays consistent, so it does not block capture.
    if (any_moe && !runtime_config().moe.skip) {
        if (runtime_config().moe.no_cutlass3x || !runtime_config().moe.nvfp4_device_args)
            return false;
        if (!cutlass_grouped_3x_nvfp4_available())
            return false;
        const int ne = cfg.n_experts;
        if (!moe_.cutlass3x_packed || !moe_.cutlass3x_sf || !moe_.d_M_per ||
            moe_.d_M_per_count < ne || !moe_.d_sfa_offsets || !moe_.d_B_ptrs_cache ||
            !moe_.d_SFB_ptrs_cache || !moe_.d_alpha_full || !moe_.cutlass3x_sfa_ptrs ||
            moe_.cutlass3x_sfa_ptrs_count < ne)
            return false;
        auto covers = [&](const std::vector<TensorID>& ids) {
            if (static_cast<int>(ids.size()) < ne)
                return false;
            for (int e = 0; e < ne; ++e) {
                if (ids[e] == kInvalidTensorID)
                    return false;
                if (registry_.handle(ids[e]).primary_tier != StorageTier::CUTLASS_NVFP4)
                    return false;
            }
            return true;
        };
        for (int i = 0; i < cfg.n_layers; ++i) {
            if (!layer_has_moe(i))
                continue;
            const auto& ly = model_->layer(i);
            const bool gated = !(ly.expert_gate_packed.data == nullptr &&
                                 (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));
            // Require the per-layer da_cache: without it dispatch_device
            // falls back to per-call H2D from stack vectors, which is not
            // graph-capturable (#860; the fallback also guards itself).
            if (i >= static_cast<int>(moe_.per_layer_da_cache.size()) ||
                !moe_.per_layer_da_cache[i].ready)
                return false;
            if (!covers(ly.expert_up_ids) || !covers(ly.expert_down_ids) ||
                (gated && !covers(ly.expert_gate_ids)))
                return false;
        }
    }
    return true;
}

bool GraphExecutor::ensure_chunk_capture_scratch(int ctx_capacity) {
    if (ctx_capacity <= 0)
        return false;
    if (chunk_capture_k_ && chunk_capture_ctx_ >= ctx_capacity)
        return true;
    const auto& cfg = model_->config();
    // Size for the largest per-layer shape (uniform under the eligibility
    // gate, but the max keeps the scratch safe regardless).
    int nkv_u = 0;
    for (int x : cfg.n_kv_heads_per_layer) nkv_u = std::max(nkv_u, x);
    if (nkv_u <= 0) nkv_u = cfg.n_kv_heads;
    int hd_u = 0;
    for (int x : cfg.head_dim_per_layer) hd_u = std::max(hd_u, x);
    if (hd_u <= 0)
        hd_u = cfg.head_dim > 0 ? cfg.head_dim
                                : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
    if (nkv_u <= 0 || hd_u <= 0)
        return false;
    const size_t bytes = static_cast<size_t>(ctx_capacity) * nkv_u * hd_u * sizeof(half);
    if (chunk_capture_k_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_capture_k_));
        chunk_capture_k_ = nullptr;
    }
    if (chunk_capture_v_) {
        IMP_CUDA_CHECK_LOG(cudaFree(chunk_capture_v_));
        chunk_capture_v_ = nullptr;
    }
    chunk_capture_ctx_ = 0;
    if (cudaMalloc(&chunk_capture_k_, bytes) != cudaSuccess ||
        cudaMalloc(&chunk_capture_v_, bytes) != cudaSuccess) {
        IMP_LOG_WARN("chunk-capture scratch alloc failed (2x %.1f MiB) — captured verify disabled",
                     bytes / (1024.0 * 1024.0));
        if (chunk_capture_k_) {
            IMP_CUDA_CHECK_LOG(cudaFree(chunk_capture_k_));
            chunk_capture_k_ = nullptr;
        }
        return false;
    }
    chunk_capture_ctx_ = ctx_capacity;
    IMP_LOG_INFO("chunk-capture K/V scratch: 2x %.1f MiB (ctx_capacity=%d, nkv=%d, hd=%d)",
                 bytes / (1024.0 * 1024.0), ctx_capacity, nkv_u, hd_u);
    return true;
}

// pre_dequant_weights() is in executor_pre_dequant.cu
// configure_*_workspace(), resize_workspace(), allocate_decode_workspace(),
// use_workspace(), layer_has_*(), view_tokens(), ensure_logits_pinned()
// are in executor_workspace_config.cu

}  // namespace imp

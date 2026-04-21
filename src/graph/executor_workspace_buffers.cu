// Auxiliary buffer allocation and cleanup — extracted from executor_workspace.cu (RF-004).
// Handles: dequant scratch, sampling, MMVQ, split-K, attention S-matrix, FMHA,
// MoE workspace, FP8 activation, CUTLASS NVFP4/MXFP4 activation buffers.

#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "graph/executor_helpers.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/sampling.h"
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
        // Size splits proportional to max context blocks, capped at 64
        // (raised from 32 to support aggressive flash-decode splitting)
        int max_context_blocks = (max_tokens_ + kKVBlockSize - 1) / kKVBlockSize;
        int max_splits = std::min(64, std::max(1, max_context_blocks));
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
            if (has_host_experts && !getenv("IMP_NO_EXPERT_CACHE")) {
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
            } else if (has_host_experts) {
                IMP_LOG_INFO("Expert LRU cache disabled via IMP_NO_EXPERT_CACHE (staging fallback)");
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
                             (packed_sz + sf_sz) / (1024.0 * 1024.0),
                             packed_sz / (1024.0 * 1024.0),
                             sf_sz / (1024.0 * 1024.0),
                             max_expanded);
            } else {
                IMP_LOG_WARN("CUTLASS 3.x MoE staging: allocation failed, path disabled");
                if (moe_.cutlass3x_packed) { vram_free(vram_alloc_, moe_.cutlass3x_packed); moe_.cutlass3x_packed = nullptr; }
                if (moe_.cutlass3x_sf) { vram_free(vram_alloc_, moe_.cutlass3x_sf); moe_.cutlass3x_sf = nullptr; }
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
    if (wcache_.use_fp8) {  // PHASE-3-TODO: mode flag set at init, not weight probe — defer to Phase 4
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
    if (wcache_.nvfp4_decode_mode > 0 && cutlass_sm120_nvfp4_available()) {  // PHASE-3-TODO: mode flag, not weight probe — defer to Phase 4
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
    wcache_.free(vram_alloc_);  // PHASE-3-TODO: lifecycle management, not weight probe — belongs in wcache_

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

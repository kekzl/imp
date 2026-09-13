// Auxiliary buffer allocation and cleanup — extracted from executor_workspace.cu (RF-004).
// Handles: dequant scratch, sampling, MMVQ, split-K, attention S-matrix, FMHA,
// MoE workspace, FP8 activation, CUTLASS NVFP4/MXFP4 activation buffers.

#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/attention_dispatch_rules.h"
#include "exec/dequant_cap.h"
#include "memory/vram_query.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "exec/gemm_scratch.h"  // prewarm_mmvq_scratch
#include "exec/nvfp4_expert_offload.h"
#include "compute/gdn.h"        // gdn_scan_chunkpar_workspace_bytes
#include "compute/gemm_f16_narrow_smallm.h"  // gemm_f16_narrow_smallm_workspace_bytes
#include "compute/gemm.h"       // kGemmCublasWorkspaceBytes, block_q8_1
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "exec/sparse_attn_geometry.h"
#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "memory/engine_arena.h"
#include "exec/workspace_sizes.h"
#include "core/process_diag.h"  // process_diag_prefill_graph_ignore_dequant_cap

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <algorithm>

namespace imp {

// workspace_sizes.h stays CUDA-free and replicates these sizes; static_assert here ties
// both copies together so a layout drift is a compile error, not a bad arena size.
static_assert(kExecBlockQ81Bytes == sizeof(block_q8_1),
              "exec_t2_demand's block_q8_1 stride drifted from compute/gemm.h");
static_assert(kExecCublasWorkspaceBytes == kGemmCublasWorkspaceBytes,
              "exec_t2_demand's cuBLASLt workspace drifted from compute/gemm.h");
static_assert(kExecBenchScratchBytes == kGemmBenchScratchBytes,
              "exec_t2_demand's algo-bench scratch drifted from compute/gemm.h");
static_assert(kExecGrouped3xStagingBytes == kGrouped3xStagingBytes,
              "exec_t2_demand's grouped-3x staging drifted from gemm_cutlass_grouped_3x.h");
static_assert(kExecGrouped3xWorkspaceBytes == kGrouped3xWorkspaceBytes,
              "exec_t2_demand's grouped-3x workspace drifted from gemm_cutlass_grouped_3x.h");

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
        // T2 quartet (A7 step 4b.2, charged as mla_scratch): no degradation contract, both
        // consumers dereference it unconditionally, so a null would fault mid-kernel. Load fails
        // instead: a typed refusal, not a downgrade or crash (MEMORY.md B5.2).
        bool mla_ok = true;
        auto alloc = [&](void** p, size_t cols, const char* name) {
            size_t sz = T * cols * sizeof(half);
            auto slab = engine_arena().take_bytes(sz);
            if (slab.empty()) {
                IMP_LOG_ERROR("MLA scratch %s (%.1f MiB) unavailable from the T2 arena "
                              "(%.1f MiB of %.1f MiB still free) — the plan under-reserved",
                              name, sz / (1024.0 * 1024.0),
                              engine_arena().remaining() / (1024.0 * 1024.0),
                              engine_arena().capacity() / (1024.0 * 1024.0));
                *p = nullptr;
                mla_ok = false;
                return;
            }
            *p = slab.data();
        };
        alloc(&mla_kv_a_buf_, kva_out, "kv_a");
        alloc(&mla_latent_buf_, static_cast<size_t>(cfg.kv_lora_rank), "latent");
        alloc(&mla_k_rope_buf_, static_cast<size_t>(cfg.qk_rope_head_dim), "k_rope");
        alloc(&mla_kv_b_buf_, kvb_out, "kv_b");
        if (!mla_ok) {
            mla_scratch_unservable_ = true;
            return;
        }
        IMP_LOG_INFO("MLA QKV scratch: kv_a+latent+k_rope+kv_b for max_tokens=%d (graph-safe)",
                     max_tokens_);

        // Phase 3 absorbed-decode latent KV cache, opt-in via attention.mla_absorb. Requires
        // every layer's kv_b_proj FP16 (absorbed path slices W_UK/W_UV from it); else skip+warn.
        if (dispatch_policy().attention.mla_absorb && mla_absorb_max_seq_ > 0) {
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
                // Same tier, opposite contract to the quartet above: both consumers null-check and take
                // the materialized path, so this one degrades rather than failing the load. Charged only
                // when mla_absorb is set; n_layers x full max_seq wide, ~1 GiB at ctx 32k.
                auto cache_slab = engine_arena().take_bytes(cache_bytes);
                auto scores_slab = engine_arena().take_bytes(scores_bytes);
                if (cache_slab.empty() || scores_slab.empty()) {
                    IMP_LOG_ERROR("attention.mla_absorb: latent cache (%.1f MiB) unavailable from "
                                  "the T2 arena — falling back to materialized.",
                                  cache_bytes / (1024.0 * 1024.0));
                    mla_absorb_cache_ = nullptr;
                    mla_absorb_scores_ = nullptr;
                } else {
                    mla_absorb_cache_ = cache_slab.data();
                    mla_absorb_scores_ = reinterpret_cast<float*>(scores_slab.data());
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

    // Dequant scratch for on-the-fly weight dequant. Every consumer guards with
    // dequant_gpu_supported(qtype) (GGUF block quants only); SafeTensors F16/NVFP4 models
    // skip this buffer entirely.
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

    // One SAMPLE_SCRATCH_BYTES slot per batched decode sequence (result + multi-block
    // partial scratch, greedy and top-k/top-p; SAMPLE_SCRATCH_BYTES >= ARGMAX_SCRATCH_BYTES).
    // Slot 0 is the single-sequence layout. Allocated x2 (parity halves) so pipelined decode
    // can enqueue step N+1 while step N's gather is still in flight.
    {
        sample_slots_ = std::max(1, max_logit_tokens_);
        // T2 (A7 step 4b.2): max_logit_tokens_ = max(max_batch, 8), the batch not the context,
        // ~1 MiB; exec_t2_demand charges it as sample_scratch. No fallback: a null buffer means
        // "no batched sampling", zeroing sample_slots_ (AUDIT B53).
        auto slab = engine_arena().take_bytes(2 * static_cast<size_t>(sample_slots_) *
                                              SAMPLE_SCRATCH_BYTES);
        if (slab.empty()) {
            IMP_LOG_ERROR("Failed to obtain the sampling result buffer from the T2 arena");
            d_sample_result_ = nullptr;
            sample_slots_ = 0;
        } else {
            d_sample_result_ = reinterpret_cast<int32_t*>(slab.data());
        }
    }

    // Pinned host buffer for async sampling D2H copy (avoids stack-variable
    // sync) — one int32 per batched slot.
    if (h_sample_pinned_.empty() && d_sample_result_) {
        // T5b (memory/host_pinned.h). Same failure contract as before: an empty
        // buffer disables the async D2H path, which every consumer tests for.
        h_sample_pinned_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                 2 * sizeof(int32_t) * sample_slots_);
        if (h_sample_pinned_.empty())
            IMP_LOG_WARN("pinned sample buffer unavailable — async sample D2H disabled");
    }

    // Row-batched sampler args: pinned staging + device mirror (one H2D per
    // decode step for the whole batch).
    if (h_row_args_.empty() && d_sample_result_ && sample_slots_ > 0) {
        // Device mirror is engine-persistent (T2), sized from max_logit_tokens_, ~115 KiB, reused
        // every decode step. No direct-allocation fallback: null falls back to per-row sampling
        // (AUDIT B47, A7 step 4b.2).
        // A reconfigure strands the superseded slab in the bump arena (bounded, one-time; same
        // trade as the MMVQ tenant): 115 KiB against 120 MiB arena slack.
        auto slab = engine_arena().take_bytes(2 * sizeof(TopkRowArgs) * sample_slots_);
        h_row_args_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                            2 * sizeof(TopkRowArgs) * sample_slots_);
        if (h_row_args_.empty() || slab.empty()) {
            IMP_LOG_WARN("row-batched sampler args alloc failed — falling back to per-row sampling");
            h_row_args_.reset();
            d_row_args_ = nullptr;
        } else {
            d_row_args_ = reinterpret_cast<TopkRowArgs*>(slab.data());
        }
        // Greedy + penalty row staging (same tier, same fallback contract:
        // an empty slab just means those rows launch per-row as before).
        auto gslab = engine_arena().take_bytes(2 * sizeof(GreedyRowArgs) * sample_slots_);
        h_greedy_args_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                               2 * sizeof(GreedyRowArgs) * sample_slots_);
        if (h_greedy_args_.empty() || gslab.empty()) {
            h_greedy_args_.reset();
            d_greedy_args_ = nullptr;
        } else {
            d_greedy_args_ = reinterpret_cast<GreedyRowArgs*>(gslab.data());
        }
        auto pslab = engine_arena().take_bytes(2 * sizeof(PenaltyRowArgs) * sample_slots_);
        h_pen_args_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                            2 * sizeof(PenaltyRowArgs) * sample_slots_);
        if (h_pen_args_.empty() || pslab.empty()) {
            h_pen_args_.reset();
            d_pen_args_ = nullptr;
        } else {
            d_pen_args_ = reinterpret_cast<PenaltyRowArgs*>(pslab.data());
        }
    }

    // Per-parity gather-done events for the pipelined split gather/wait.
    for (int p = 0; p < 2; ++p) {
        if (!sample_gather_evt_[p] &&
            cudaEventCreateWithFlags(&sample_gather_evt_[p], cudaEventDisableTiming) != cudaSuccess) {
            sample_gather_evt_[p] = nullptr;
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
            // T2 (A7 step 4b.2): engine-persistent, sized once from max K and batch, reused every
            // decode step. exec_t2_demand charges the dp4a-staging family as quant_scratch. No
            // fallback: consumers null-check and take the FP16 GEMV path (AUDIT B47).
            auto q8_slab = engine_arena().take_bytes(q8_1_sz);
            auto d8_slab = engine_arena().take_bytes(d8_sz);
            if (q8_slab.empty() || d8_slab.empty()) {
                IMP_LOG_WARN("MMVQ scratch unavailable from the T2 arena, dp4a path disabled");
                qscratch_.q8_1_buf = nullptr;
                qscratch_.d8_buf = nullptr;
                qscratch_.q8_1_max_blocks = 0;
            } else {
                qscratch_.q8_1_buf = q8_slab.data();
                qscratch_.d8_buf = reinterpret_cast<float*>(d8_slab.data());
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
            // Same family, same tier: tens of bytes, engine-lifetime, and the
            // sparsity path in executor_ffn.cu already tests the pointer.
            auto slab = engine_arena().take_bytes(mask_sz);
            if (slab.empty()) {
                IMP_LOG_WARN("FFN sparsity mask unavailable from the T2 arena (%zu bytes)", mask_sz);
                qscratch_.ffn_block_mask = nullptr;
                qscratch_.ffn_block_mask_words = 0;
            } else {
                qscratch_.ffn_block_mask = reinterpret_cast<uint32_t*>(slab.data());
                qscratch_.ffn_block_mask_words = mask_words;
            }
        }

        // Pre-warm the file-scope MMVQ Q8_1 quantization scratch used by the
        // ggml_mmvq_q*_kernel hot-path in executor_kernels.cu. Sized for the
        // worst case (max_tokens × max_k) so the hot path never re-allocates
        // (capture-safe). QW1 from the phase-5 review §2.1 (archived in #604).
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
                // Third member of the same family. executor_gemm_dispatch.cu
                // checks both pointers AND both sizes before routing, so a
                // closed arena degrades to the FP16 weight cache.
                auto q8_slab = engine_arena().take_bytes(q8_sz);
                auto d8_slab = engine_arena().take_bytes(d8_sz);
                if (q8_slab.empty() || d8_slab.empty()) {
                    IMP_LOG_WARN("dp4a prefill scratch unavailable from the T2 arena (%.1f MiB), "
                                 "FP16 cache fallback",
                                 (q8_sz + d8_sz) / (1024.0 * 1024.0));
                    qscratch_.q8_1_prefill_buf = nullptr;
                    qscratch_.d8_prefill_buf = nullptr;
                } else {
                    qscratch_.q8_1_prefill_buf = q8_slab.data();
                    qscratch_.d8_prefill_buf = reinterpret_cast<float*>(d8_slab.data());
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
        // Size splits proportional to max context blocks, capped at 128: the GQA tile kernel
        // runs grid.y = n_kv_heads and recovers parallelism via split count (paged_attention_decode_fp8).
        // Charged from the resolved block size (set_kv_block_size); kKVBlockSize sizes the wrong
        // geometry for n_kv_heads <= 4 models under kv_cache.block_size (B-7).
        int max_context_blocks = (max_tokens_ + kv_block_size_ - 1) / kv_block_size_;
        int max_splits = std::min(128, std::max(1, max_context_blocks));
        int partial_stride = 2 + hd;
        int max_batch = max_logit_tokens_;  // = max_batch_size
        size_t sz = static_cast<size_t>(max_batch) * nh * max_splits * partial_stride * sizeof(float);
        // T2 (A7 step 4b.2), charged as splitk_scratch: engine-persistent, sized from shape +
        // batch + context. paged_attention_set_splitk_scratch treats null as "no split-K"; the
        // kernel re-checks size before use, so a closed arena only costs the split path.
        auto slab = engine_arena().take_bytes(sz);
        if (slab.empty()) {
            IMP_LOG_WARN("Split-K scratch unavailable from the T2 arena (%zu bytes), split-K "
                         "disabled", sz);
            qscratch_.splitk = nullptr;
            qscratch_.splitk_size = 0;
        } else {
            qscratch_.splitk = slab.data();
            qscratch_.splitk_size = sz;
            IMP_LOG_INFO("Split-K paged attention scratch: %.2f KiB", sz / 1024.0);
        }
    }

    // Sparse decode attention scratch (attention.sparse_topk_tokens).
    // Scores row capacity is the max-context block count so a captured graph
    // stays in-bounds while the context grows during replay.
    {
        const auto& acfg = dispatch_policy().attention;
        if (acfg.sparse_topk_tokens > 0) {
            // Scores row capacity must cover the max context, not max_tokens_ (the per-forward chunk
            // cap): sizing from max_tokens_ silently disables the feature past 4k context (dispatch
            // gate checks max_blocks_per_seq against this capacity).
            const int max_ctx_tokens = (mla_absorb_max_seq_ > 0) ? mla_absorb_max_seq_ : max_tokens_;
            // +16: spec verify row tables carry 16 slack blocks past the context ceiling
            // (engine_spec_capture.cpp table_cap "+16"); dispatch gate compares stride against this.
            // Every token->block conversion must use the cache's real block size, not kKVBlockSize:
            // n_kv_heads <= 4 models use block size 32, not 16, or budgets silently double (#1819).
            const int kv_bs = kv_block_size_;
            const SparseGeometry geo =
                sparse_geometry(acfg.sparse_topk_tokens, acfg.sparse_sink_tokens,
                                acfg.sparse_recent_tokens, acfg.sparse_min_ctx, max_ctx_tokens, kv_bs);
            const int max_ctx_blocks = geo.max_ctx_blocks;
            const int sink_blocks = geo.sink_blocks;
            const int recent_blocks = geo.recent_blocks;
            const int budget_blocks = geo.budget_blocks;
            if (geo.budget_raised) {
                IMP_LOG_WARN("attention.sparse_topk_tokens: budget below sink+recent, raised to %d "
                             "blocks (%d tokens)",
                             budget_blocks, budget_blocks * kv_bs);
            }
            // Identity below sparse_min_ctx (the selection's win only outgrows
            // its overhead past ~12k measured); the table rows must hold an
            // identity copy up to that length.
            const int engage_blocks = geo.engage_blocks;
            const int table_blocks = engage_blocks;
            // Row capacity covers batched decode AND spec verify chunks: chunk rows are sequences,
            // up to the 33-row chunk cap (engine_spec_capture.cpp chunk_cap = max(bucket_max, k+1, 33)).
            // Sizing from max_batch alone starves any chunk wider than the batch.
            const int max_batch = std::max(max_logit_tokens_, 33);
            const size_t scores_sz = (size_t)max_batch * max_ctx_blocks * sizeof(float);
            const size_t bt_sz = (size_t)max_batch * table_blocks * sizeof(int);
            const size_t ctx_sz = (size_t)max_batch * sizeof(int);
            auto scores_slab = engine_arena().take_bytes(scores_sz);
            auto bt_slab = engine_arena().take_bytes(bt_sz);
            auto ctx_slab = engine_arena().take_bytes(ctx_sz);
            if (scores_slab.empty() || bt_slab.empty() || ctx_slab.empty()) {
                IMP_LOG_WARN("sparse decode attention scratch unavailable from the T2 arena "
                             "(%.1f KiB) - feature disabled",
                             (scores_sz + bt_sz + ctx_sz) / 1024.0);
            } else {
                qscratch_.sparse_scores = reinterpret_cast<float*>(scores_slab.data());
                qscratch_.sparse_block_tables = reinterpret_cast<int*>(bt_slab.data());
                qscratch_.sparse_context_lens = reinterpret_cast<int*>(ctx_slab.data());
                qscratch_.sparse_budget_blocks = budget_blocks;
                qscratch_.sparse_sink_blocks = sink_blocks;
                qscratch_.sparse_recent_blocks = recent_blocks;
                qscratch_.sparse_engage_blocks = engage_blocks;
                qscratch_.sparse_table_blocks = table_blocks;
                qscratch_.sparse_max_ctx_blocks = max_ctx_blocks;
                qscratch_.sparse_max_rows = max_batch;
                qscratch_.sparse_score_meanstd = acfg.sparse_score_meanstd;
                qscratch_.sparse_score_std_coef = acfg.sparse_score_std_coef;
                IMP_LOG_INFO("Sparse decode attention: budget %d blocks (%d tokens), sink %d + "
                             "recent %d blocks, engage above %d tokens, score %s, scratch %.1f KiB",
                             budget_blocks, budget_blocks * kv_bs, sink_blocks, recent_blocks,
                             engage_blocks * kv_bs, acfg.sparse_score_meanstd ? "mean+std" : "minmax",
                             (scores_sz + bt_sz + ctx_sz) / 1024.0);
            }
        }
    }

    // cuBLAS attention S-matrix workspace [n_heads, attn_seq, attn_seq] FP16; only the
    // materialized cuBLAS prefill fallback uses it. Skip when FP16-QK FA2 serves all prefill
    // (uniform per-layer shapes, no learned sinks, head_dim 128 or 256 with fa2_hd256).
    // cuBLAS remains the reference for heterogeneous shapes, learned sinks, hd=256 without
    // fa2_hd256, and fa2_fp16qk=never.
    if (fa2_serves_all_prefill()) {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (FP16-QK FA2 serves all prefill — "
                     "no S-matrix needed)");
    } else if (!skip_batch_dequant) {
        int nh = cfg.n_heads;
        // 256 MiB: cuBLAS handles short sequences (up to ~1448 attn_seq for
        // 32-head models). Longer sequences auto-route to FMHA via the
        // fmha_prefill_threshold. Reduced from 1024 to free ~768 MiB for KV.
        int cfg_mib = dispatch_policy().attention.attn_scores_mib;
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

    // Auto-derive fmha_prefill_threshold from S-matrix capacity. Only matters for configs
    // still using the S-matrix (hd != 128, per-layer, sinks); hd=128 tries FP16-QK FA2 first
    // so the threshold is moot (cap=0 -> threshold=1). Dispatch uses prefer_fmha = (n >=
    // threshold), so threshold = cap+1: the chunk with n == cap fits the S-matrix and goes to cuBLAS.
    if (dispatch_policy().attention.fmha_prefill_threshold == -1) {
        int auto_threshold = attn_scores_cap() > 0 ? attn_scores_cap() + 1 : 1;
        const_cast<cfg::Attention&>(dispatch_policy().attention).fmha_prefill_threshold = auto_threshold;
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

        // Staging buffer for host->device expert weight transfer, used only by the legacy MoE
        // forward's !packed.on_device branches. Skip entirely when every packed expert tensor is
        // device-resident.
        size_t max_expert_raw = 0;
        bool any_host_packed_experts = false;
        bool nvfp4_host_experts = false;
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
            // NVFP4-prequant checkpoints have no 3-D packed tensor here: experts are per-expert 2-D
            // tensors and Phase 3 stamps the packed slot later. Size the pool from those instead, off
            // the full slot (packed weights + micro-scales).
            // qtype is still INT8 here: promotion to NVFP4 runs in pre_dequant_weights()
            // (init_kv_cache), after this. is_nvfp4_prequant comes from config.json, already known.
            if (model_->config().is_nvfp4_prequant) {
                for (int li = 0; li < model_->n_layers(); li++) {
                    const auto& L = model_->layer(li);
                    auto check_nv = [&](const std::vector<Tensor>& experts) {
                        if (experts.empty() || !experts[0].data || experts[0].ndim != 2)
                            return;
                        // Per-expert 2-D NVFP4 weights are stored [N, K/2].
                        const auto layout =
                            nvfp4_slot_layout(experts[0].shape[0], experts[0].shape[1] * 2);
                        if (layout.slot_bytes() == 0)
                            return;
                        max_expert_raw = std::max(max_expert_raw, layout.slot_bytes());
                        if (!experts[0].on_device) {
                            any_host_packed_experts = true;
                            nvfp4_host_experts = true;
                        }
                    };
                    check_nv(L.expert_w_gate);
                    check_nv(L.expert_w_up);
                    check_nv(L.expert_w_down);
                }
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
            // gpt-oss exemption: its MXFP4 experts are host-resident here only transiently;
            // pre_dequant converts them to on-device NVFP4 + CUTLASS-grouped before the first forward.
            // Never host-offloaded at runtime, so the LRU cache must not be allocated (would shadow
            // the converted device experts -> garbage output).
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
                // NVFP4-prequant experts are per-expert 2-D at this point, so
                // the loop above cannot see them.
                has_host_experts = has_host_experts || nvfp4_host_experts;
            }
            if (has_host_experts && !dispatch_policy().moe.no_expert_cache) {
                // Budget: proportional to free VRAM (15%) instead of flat cap.
                // KV cache + weight caches (FP8/NVFP4) need the remaining VRAM,
                // so expert cache must not over-commit.
                size_t free_mem = 0, total_mem = 0;
                vram_budget_mem_get_info(&free_mem, &total_mem);
                size_t safety = 128 << 20;  // 128 MiB reserve
                size_t budget = (free_mem > safety) ? free_mem - safety : 0;
                int pct = dispatch_policy().moe.expert_cache_budget_pct;
                pct = std::clamp(pct, 1, 90);
                budget = static_cast<size_t>(budget * (pct / 100.0));
                const auto& mcfg = model_->config();
                bool debug_parity = dispatch_policy().moe.expert_cache_debug_parity;
                if (expert_cache_.init(max_expert_raw, budget, vram_alloc_, mcfg.n_layers,
                                       mcfg.n_experts, debug_parity, nvfp4_host_experts)) {
                    IMP_LOG_INFO("Expert LRU cache: %d slots (%.2f MiB / %.2f MiB budget)",
                                 expert_cache_.n_slots_,
                                 expert_cache_.n_slots_ * max_expert_raw / (1024.0 * 1024.0),
                                 budget / (1024.0 * 1024.0));
                    // Slot-index buffer for the host-offload decode path: one
                    // block of top_k int32 per projection.
                    int idx_count = kExpertProjCount * std::max(1, mcfg.n_experts_active);
                    size_t idx_bytes = static_cast<size_t>(idx_count) * sizeof(int32_t);
                    moe_.d_slot_idx = static_cast<int32_t*>(
                        vram_alloc(vram_alloc_, idx_bytes, "moe_slot_idx"));
                    moe_.d_slot_idx_count = moe_.d_slot_idx ? idx_count : 0;
                }
            } else if (has_host_experts) {
                IMP_LOG_INFO("Expert LRU cache disabled via moe.no_expert_cache (staging fallback)");
            }

            // Whole-layer staging buffer for NVFP4 host prefill, sized for one layer and reused
            // across layers (forward is sequential, each layer overwrites the previous).
            // Batches per-expert transfers (~768 KiB + ~96 KiB, below PCIe bandwidth) into one
            // ~110 MiB projection transfer. Only effective with moe.pin_host_experts: large
            // transfers pay off only from a pinned source.
            const auto& stage_cfg = model_->config();
            if (nvfp4_host_experts && stage_cfg.n_experts > 0 &&
                dispatch_policy().moe.pin_host_experts) {
                const size_t proj_bytes =
                    static_cast<size_t>(stage_cfg.n_experts) * max_expert_raw;
                const size_t total = static_cast<size_t>(kExpertProjCount) * proj_bytes;
                // Only worth it if it fits comfortably: this runs on a model
                // that already did not fit, so never take the last of VRAM.
                size_t free_mem = 0, total_mem = 0;
                vram_budget_mem_get_info(&free_mem, &total_mem);
                if (total + (512u << 20) < free_mem) {
                    moe_.layer_stage_buf = vram_alloc(vram_alloc_, total, "moe_layer_stage");
                    if (moe_.layer_stage_buf) {
                        moe_.layer_stage_proj_bytes = proj_bytes;
                        moe_.layer_stage_size = total;
                        moe_.layer_stage_experts = stage_cfg.n_experts;
                        IMP_LOG_INFO(
                            "MoE layer staging buffer: %.2f MiB (%d experts x 3 projections, "
                            "one layer at a time)",
                            total / (1024.0 * 1024.0), stage_cfg.n_experts);

                        // SfAtom scales + per-expert pointer arrays, so the staged layer takes the CUTLASS
                        // device-args path instead of the per-expert dequant fallback. Sized from the largest
                        // projection (gate/up and down differ).
                        const int64_t d_model = stage_cfg.d_model;
                        const int64_t eff_ff =
                            stage_cfg.expert_d_ff > 0 ? stage_cfg.expert_d_ff : stage_cfg.d_ff;
                        const size_t sf_gate = cutlass_nvfp4_sf_size(
                            static_cast<int>(eff_ff), static_cast<int>(d_model));
                        const size_t sf_down = cutlass_nvfp4_sf_size(
                            static_cast<int>(d_model), static_cast<int>(eff_ff));
                        const size_t sf_proj =
                            static_cast<size_t>(stage_cfg.n_experts) * std::max(sf_gate, sf_down);
                        const size_t sf_total =
                            static_cast<size_t>(kExpertProjCount) * sf_proj;
                        const size_t ptr_count =
                            static_cast<size_t>(kExpertProjCount) * stage_cfg.n_experts;
                        moe_.layer_stage_sf = vram_alloc(vram_alloc_, sf_total, "moe_layer_stage_sf");
                        moe_.layer_stage_b_ptrs = static_cast<const void**>(
                            vram_alloc(vram_alloc_, ptr_count * sizeof(void*), "moe_stage_bptr"));
                        moe_.layer_stage_sfb_ptrs = static_cast<const void**>(
                            vram_alloc(vram_alloc_, ptr_count * sizeof(void*), "moe_stage_sfbptr"));
                        moe_.layer_stage_alpha = static_cast<float*>(
                            vram_alloc(vram_alloc_, ptr_count * sizeof(float), "moe_stage_alpha"));
                        if (moe_.layer_stage_sf && moe_.layer_stage_b_ptrs &&
                            moe_.layer_stage_sfb_ptrs && moe_.layer_stage_alpha) {
                            moe_.layer_stage_sf_proj_bytes = sf_proj;
                            moe_.layer_stage_sf_size = sf_total;
                            IMP_LOG_INFO(
                                "MoE layer staging: CUTLASS device-args view enabled "
                                "(+%.2f MiB SfAtom)",
                                sf_total / (1024.0 * 1024.0));
                        } else {
                            // Staging still works, it just falls back to the
                            // per-expert dequant path for the GEMM.
                            moe_.layer_stage_sf_proj_bytes = 0;
                            IMP_LOG_WARN(
                                "MoE layer staging: SfAtom view alloc failed (%.2f MiB) — "
                                "prefill keeps the per-expert dequant path",
                                sf_total / (1024.0 * 1024.0));
                        }
                    }
                } else {
                    IMP_LOG_INFO(
                        "MoE layer staging buffer: skipped, needs %.2f MiB and %.2f MiB free "
                        "— prefill stays on the per-expert path",
                        total / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
                }
            }
        }

        // ST-NVFP4 experts run the CUTLASS 3.x grouped path for all prefill sizes; the buffers
        // below only back GGUF-family batch paths (FP16-cache batch, Q6_K FP8 batch, IMMA Q8
        // staging) and the post-CUTLASS NVFP4->FP16 dequant fallback. Skipping them frees VRAM;
        // CUTLASS-decline falls through to the legacy per-expert path (slow but correct).
        bool experts_st_nvfp4 = true;
        for (int i = 0; i < cfg.n_layers && experts_st_nvfp4; i++) {
            const auto& L = model_->layer(i);
            if (L.expert_up_packed.data && L.expert_up_packed.qtype != QType::NVFP4)
                experts_st_nvfp4 = false;
            if (L.expert_down_packed.data && L.expert_down_packed.qtype != QType::NVFP4)
                experts_st_nvfp4 = false;
        }
        if (experts_st_nvfp4) {
            IMP_LOG_INFO(
                "MoE batch dequant buffer: skipped "
                "(ST-NVFP4 experts — CUTLASS grouped prefill does not need it)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
        } else
        // Batch dequant: sized for a chunk of experts (L2-resident strategy). Dequant a chunk to
        // FP16 then GEMM while still warm in L2 (~96 MB on RTX 5090), avoiding an FP16 round-trip
        // to DRAM. Skipped when experts are host-resident (only useful for on-device experts).
        if (!skip_batch_dequant) {
            // Cap at remaining free VRAM minus a reserve for KV cache + workspaces (init_kv_cache
            // runs after this and sees what's left). Leave >= 1 GiB free: covers vram_budget reserve
            // (~768 MiB) plus a useful KV cache.
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
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
        }

        // CUTLASS 3.x NVFP4 grouped activation staging: auto-used for prefill (n > 1) on
        // NVFP4-prequant MoE models. Decode (n == 1) keeps the legacy per-expert GEMV.
        // max K = d_model, max_expanded = max_tokens * top_k.
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
                // Device array of per-expert SFA base pointers for the fused quantize kernel. Last of the
                // n_experts-sized MoE pointer arrays to move; same tier, same moe_arrays charge. The
                // fused quantize kernel treats null as "per-expert SFA bases unavailable".
                size_t sfa_ptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(uint8_t*);
                if (auto sl = engine_arena().take_bytes(sfa_ptr_bytes); !sl.empty()) {
                    moe_.cutlass3x_sfa_ptrs = reinterpret_cast<uint8_t**>(sl.data());
                    moe_.cutlass3x_sfa_ptrs_count = cfg.n_experts;
                } else {
                    IMP_LOG_WARN("CUTLASS 3.x SFA pointer array unavailable from the T2 arena");
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
            // T2 for the cluster below (A7 step 4b.2): each sized from n_experts, never freed per
            // request, and each caller treats null as "this optional path is off". No direct-
            // allocation fallback; exec_t2_demand charges them as moe_arrays (AUDIT B47).
            size_t ptr_bytes = 3 * static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            if (auto sl = engine_arena().take_bytes(ptr_bytes); !sl.empty()) {
                moe_.d_work_ptrs = reinterpret_cast<void**>(sl.data());
                moe_.d_work_ptrs_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE work ptrs unavailable from the T2 arena");
                moe_.d_work_ptrs = nullptr;
                moe_.d_work_ptrs_count = 0;
            }

            // Per-expert FP8 scale buffer (trivial: 128 experts × 4 bytes = 512 bytes).
            size_t scale_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            if (auto sl = engine_arena().take_bytes(scale_bytes); !sl.empty())
                moe_.d_fp8_scales = reinterpret_cast<float*>(sl.data());
            else
                moe_.d_fp8_scales = nullptr;

            // Per-expert device-resident token-count buffer (n_experts × 4 bytes).
            // Populated each forward by compute_M_per_from_offsets_device, replacing
            // the host D2H+sync+loop pattern in the MoE prefill dispatch path.
            size_t m_per_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(int32_t);
            if (auto sl = engine_arena().take_bytes(m_per_bytes); !sl.empty()) {
                moe_.d_M_per = reinterpret_cast<int32_t*>(sl.data());
                moe_.d_M_per_count = cfg.n_experts;
            } else {
                moe_.d_M_per = nullptr;
                moe_.d_M_per_count = 0;
            }

            // Compact-alpha output buffer + active-expert counter. Populated by
            // compact_alpha_active. Sized for max n_experts (only first d_na
            // entries used at dispatch).
            if (auto sl = engine_arena().take_bytes(
                    static_cast<size_t>(cfg.n_experts) * sizeof(float));
                !sl.empty())
                moe_.d_alpha_compact = reinterpret_cast<float*>(sl.data());
            else
                moe_.d_alpha_compact = nullptr;
            if (auto sl = engine_arena().take_bytes(sizeof(int32_t)); !sl.empty())
                moe_.d_na = reinterpret_cast<int32_t*>(sl.data());
            else
                moe_.d_na = nullptr;

            // SFA byte-offsets prefix sum (Phase 3 staging). n_experts+1 int64
            // = trivial (<2 KiB for 128 experts).
            if (auto sl = engine_arena().take_bytes(
                    static_cast<size_t>(cfg.n_experts + 1) * sizeof(int64_t));
                !sl.empty())
                moe_.d_sfa_offsets = reinterpret_cast<int64_t*>(sl.data());
            else
                moe_.d_sfa_offsets = nullptr;

            // Phase 3c-full Step 1 — device-args ptr/alpha caches. n_experts ×
            // (2 × sizeof(void*) + sizeof(float)) ≈ 2.5 KiB for 128 experts.
            const size_t ptr_sz = static_cast<size_t>(cfg.n_experts) * sizeof(const void*);
            const size_t alpha_sz = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            if (auto sl = engine_arena().take_bytes(ptr_sz); !sl.empty())
                moe_.d_B_ptrs_cache = reinterpret_cast<const void**>(sl.data());
            else
                moe_.d_B_ptrs_cache = nullptr;
            if (auto sl = engine_arena().take_bytes(ptr_sz); !sl.empty())
                moe_.d_SFB_ptrs_cache = reinterpret_cast<const void**>(sl.data());
            else
                moe_.d_SFB_ptrs_cache = nullptr;
            if (auto sl = engine_arena().take_bytes(alpha_sz); !sl.empty())
                moe_.d_alpha_full = reinterpret_cast<float*>(sl.data());
            else
                moe_.d_alpha_full = nullptr;

            // Device-side weight pointer array for device-grouped GEMM.
            if (auto sl = engine_arena().take_bytes(
                    static_cast<size_t>(cfg.n_experts) * sizeof(void*));
                !sl.empty()) {
                moe_.d_weight_ptrs = reinterpret_cast<void**>(sl.data());
                moe_.d_weight_ptrs_count = cfg.n_experts;
            } else {
                moe_.d_weight_ptrs = nullptr;
                moe_.d_weight_ptrs_count = 0;
            }
        }
    }

    // Chunk-parallel GDN prefill scan workspace (gdn.chunkpar_scan): five per-(chunk, head)
    // strip arrays + FP32 inter-strip state, ~42 MiB at 32 value heads. Engine lifetime; on
    // failure the route degrades to the fused scan (dispatch checks nullptr).
    if (has_gdn_ && dispatch_policy().gdn.chunkpar_scan && cfg.ssm_dt_rank > 0) {
        gdn_chunkpar_ws_bytes_ = gdn_scan_chunkpar_workspace_bytes(cfg.ssm_dt_rank);
        gdn_chunkpar_ws_ = vram_alloc(vram_alloc_, gdn_chunkpar_ws_bytes_, "gdn_chunkpar_ws");
        if (!gdn_chunkpar_ws_) {
            IMP_LOG_WARN("gdn_chunkpar workspace unavailable (%.1f MiB) — fused scan route",
                         gdn_chunkpar_ws_bytes_ / (1024.0 * 1024.0));
            gdn_chunkpar_ws_bytes_ = 0;
        }
    }

    // GDN alpha/beta narrow GEMM workspace (gdn.alpha_beta_smallm): split-K partials +
    // tickets for N = 2 x n_heads. Kernel resets its tickets, so zeroing here is the whole
    // init. nullptr keeps the two cuBLAS calls.
    if (has_gdn_ && dispatch_policy().gdn.alpha_beta_smallm && cfg.ssm_dt_rank > 0) {
        gdn_ab_ws_bytes_ = gemm_f16_narrow_smallm_workspace_bytes(2 * cfg.ssm_dt_rank);
        gdn_ab_ws_ = vram_alloc(vram_alloc_, gdn_ab_ws_bytes_, "gdn_ab_ws");
        if (gdn_ab_ws_) {
            IMP_CUDA_CHECK_LOG(cudaMemset(gdn_ab_ws_, 0, gdn_ab_ws_bytes_));
        } else {
            IMP_LOG_WARN("gdn alpha/beta workspace unavailable (%zu B) - two-call route", gdn_ab_ws_bytes_);
            gdn_ab_ws_bytes_ = 0;
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
            int conv_ch = cfg.ssm_conv_channels();
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
            // T2 (A7 step 4b.2): engine-lifetime, sized from init-time shapes; each caller degrades
            // on null (the reduction pair falls back to the sync path via its own log line). No
            // direct-allocation fallback; exec_t2_demand charges them as fp8_reduction.
            auto sl = engine_arena().take_bytes(sizeof(float));
            qscratch_.d_act_scale = sl.empty() ? nullptr : reinterpret_cast<float*>(sl.data());
            if (!qscratch_.d_act_scale)
                IMP_LOG_WARN("FP8 act scale unavailable from the T2 arena");
        }
        // Pre-allocate reduction buffers for async FP8 activation quantization.
        // Eliminates per-call cudaMalloc + cudaStreamSynchronize from the hot path.
        if (qscratch_.fp8_act && qscratch_.d_act_scale) {
            int max_n = static_cast<int>(qscratch_.fp8_act_size);   // max elements
            int threads_needed = (max_n + 3) / 4;                   // kElemsPerThread=4
            qscratch_.fp8_max_grid = (threads_needed + 255) / 256;  // kBlockSize=256
            auto sl_bm = engine_arena().take_bytes(
                static_cast<size_t>(qscratch_.fp8_max_grid) * sizeof(float));
            auto sl_am = engine_arena().take_bytes(sizeof(float));
            if (sl_bm.empty() || sl_am.empty()) {
                IMP_LOG_WARN("FP8 reduction buffers unavailable from the T2 arena — sync path");
                qscratch_.d_fp8_block_maxes = nullptr;
                qscratch_.d_fp8_absmax = nullptr;
                qscratch_.fp8_max_grid = 0;
            } else {
                qscratch_.d_fp8_block_maxes = reinterpret_cast<float*>(sl_bm.data());
                qscratch_.d_fp8_absmax = reinterpret_cast<float*>(sl_am.data());
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
        // NVFP4 prequant tensors carry K_packed = K_logical/2 in shape[1]. Phase 0b promote runs
        // after this scratch-sizing pass, so Tensor.qtype is still the on-disk byte type here,
        // not QType::NVFP4. Use the model-level cfg flag to scale K accordingly: otherwise sf
        // scratch undersizes and cudaMemsetAsync poisons the stream, collapsing output to loops.
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
            // MoE expert weights are per-expert [N, K] 2D; the 3D packed buffers reshape to
            // [n_experts, N, K], only (N, K) matters for activation scratch sizing.
            // Expert down proj K (d_ff) can exceed the per-layer scan's K (d_model, attention/shared
            // weights); missing this blows out SF scratch and poisons the stream, collapsing output.
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
                // Pre-zero the SfAtom scale workspace once: quantize_fp16_nvfp4_cutlass_kernel only
                // writes the M x K_groups valid SF cells, leaving padding as whatever was there before.
                // Zeroing once here lets the per-call path skip its own cudaMemsetAsync. Sync because
                // executor init may finish before the first stream-bound call uses this buffer.
                IMP_CUDA_CHECK_LOG(cudaMemset(qscratch_.cutlass_act_sf, 0,
                                              qscratch_.cutlass_act_sf_size));
                IMP_LOG_INFO("CUTLASS NVFP4 activation scratch: %.2f MiB (data=%.2f, sf=%.2f, ws=%.2f)",
                             (qscratch_.cutlass_act_data_size + qscratch_.cutlass_act_sf_size +
                              qscratch_.cutlass_workspace_size) /
                                 (1024.0 * 1024.0),
                             qscratch_.cutlass_act_data_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_sf_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_workspace_size / (1024.0 * 1024.0));

                // MXFP4 activation buffers share packed data with NVFP4, only need separate UE8M0 scale
                // factors (SFVecSize=32 vs NVFP4's 16). Allocate only when the model carries MXFP4
                // weights (or attention.mxfp4 prefill is opt-in enabled); hardware availability alone is
                // not sufficient.
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
                    (has_mxfp4_weights || dispatch_policy().attention.mxfp4 == "always")) {
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
    // Largest single dequant target across NVFP4 weight caches: gemm_nvfp4 dequantizes one
    // weight at a time to FP16, so the workspace only needs to fit the largest single weight
    // (N x K x 2 bytes). Weights above kCap (512 MiB) get no workspace; track the covered
    // maximum separately so one oversized tensor doesn't disable the workspace for every
    // other weight (#855).
    constexpr size_t kCap = 512ULL * 1024 * 1024;  // 512 MiB
    size_t max_bytes = 0;         // largest ELIGIBLE dequant target
    size_t covered_bytes = 0;     // largest target we will actually cover
    size_t lm_head_bytes = 0;     // the excluded plane, for the log
    // LM head is not a candidate for the fallback this cap guards: at M > 1 it is served by
    // the smallm v2 kernel, the CUTLASS NVFP4 GEMM, or the batched K-par GEMV, none of which
    // dequantizes the weight. Counting it would let one oversized plane disable prefill graph
    // capture for every plane that can use it. See exec/dequant_cap.h.
    const void* lm_head_ptr = model_ ? model_->output_proj().data : nullptr;
    auto consider = [&](int64_t N, int64_t K) {
        size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(half);
        if (bytes > max_bytes)
            max_bytes = bytes;
        if (bytes <= kCap && bytes > covered_bytes)
            covered_bytes = bytes;
    };
    auto consider_plane = [&](const void* ptr, int64_t N, int64_t K) {
        const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(half);
        if (ptr != nullptr && ptr == lm_head_ptr) {
            lm_head_bytes = std::max(lm_head_bytes, bytes);
            return;
        }
        consider(N, K);
    };
    for (const auto& [ptr, qr] : wcache_.nvfp4)
        consider_plane(ptr, qr.N, qr.K);
    for (const auto& [ptr, moe] : wcache_.nvfp4_moe)
        consider_plane(ptr, moe.N, moe.K);  // single-expert dequant slice
    for (const auto& [ptr, cw] : wcache_.cutlass_nvfp4)
        consider_plane(ptr, cw.N, cw.K);
    // SafeTensors NVFP4 prequant: per-tensor/per-expert NVFP4 storage lives on the Layer
    // struct (qtype=NVFP4 with scales sidecar), not in wcache_. gemm_nvfp4 fallback
    // constructs a synthetic NvFP4QuantResult with N=t.shape[0], K=t.shape[1]*2 (logical,
    // since shape[1] is FP4-packed). Iterate every NVFP4 tensor on the layer for the largest.
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
    if (max_bytes == 0 && lm_head_bytes == 0) {
        // No NVFP4 weights — nothing to do. The fallback won't fire.
        return true;
    }

    // Weights beyond the workspace stay on the lazy-cudaMalloc fallback on
    // non-captured streams; hitting one of them under capture now throws
    // (gemm_nvfp4) and the capture fails cleanly.
    DequantCapInputs cap_in;
    cap_in.max_eligible_bytes = max_bytes;
    cap_in.max_excluded_bytes = lm_head_bytes;
    cap_in.cap_bytes = kCap;
    cap_in.ignore_cap = imp::process_diag_prefill_graph_ignore_dequant_cap();
    const DequantCapDecision cap = dequant_cap_decide(cap_in);
    IMP_LOG_INFO("prefill dequant cap: %.0f MiB of eligible planes vs %.0f MiB cap -> graph %s "
                 "(LM head %.0f MiB excluded, it never takes the M>1 dequant fallback; covered %.0f MiB)",
                 max_bytes / (1024.0 * 1024.0), kCap / (1024.0 * 1024.0),
                 cap.graph_capture_ok ? "capturable" : "disabled",
                 lm_head_bytes / (1024.0 * 1024.0), covered_bytes / (1024.0 * 1024.0));
    if (cap.over_cap) {
        IMP_LOG_WARN(
            "gemm_nvfp4 dequant workspace: largest eligible NVFP4 weight is %.2f MiB > %.0f MiB cap "
            "(covered: %.2f MiB) — prefill graph capture %s (a captured M>1 fallback "
            "on the oversized weight fails loud).",
            max_bytes / (1024.0 * 1024.0), kCap / (1024.0 * 1024.0),
            covered_bytes / (1024.0 * 1024.0),
            cap_in.ignore_cap ? "KEPT ENABLED by diagnostics.prefill_graph_ignore_dequant_cap"
                              : "disabled");
    }
    if (!cap.graph_capture_ok)
        nvfp4_dequant_uncapturable_ = true;  // scheduler will skip prefill-graph capture
    if (covered_bytes == 0)
        return !nvfp4_dequant_uncapturable_;

    // T2: taken from the engine-persistent arena, sized to include exactly this buffer
    // (exec/workspace_sizes.h) so pre-dequant cache build doesn't leave nothing behind
    // (AUDIT B23). Falls back to direct allocation when the arena is closed (bare
    // GraphExecutor in tests) or short, like the MMVQ tenant.
    {
        auto slab = engine_arena().take_bytes(covered_bytes);
        if (!slab.empty()) {
            nvfp4_dequant_ws_buf_ = slab.data();
            nvfp4_dequant_ws_from_arena_ = true;
        } else {
            nvfp4_dequant_ws_buf_ = vram_alloc(vram_alloc_, covered_bytes, "nvfp4_dequant");
            nvfp4_dequant_ws_from_arena_ = false;
            if (nvfp4_dequant_ws_buf_)
                IMP_LOG_WARN("nvfp4_dequant: engine arena could not supply %.2f MiB "
                             "(remaining %.2f MiB) — fell back to a direct allocation",
                             covered_bytes / (1024.0 * 1024.0),
                             engine_arena().remaining() / (1024.0 * 1024.0));
        }
    }
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
    // T2 arena slabs (allocate_smallm_scratch) die with the arena.
    if (smallm_ws_ && !smallm_arena_)
        IMP_CUDA_CHECK_LOG(cudaFree(smallm_ws_));
    smallm_ws_ = nullptr;
    smallm_ws_bytes_ = 0;
    if (smallm_xq_ && !smallm_arena_)
        IMP_CUDA_CHECK_LOG(cudaFree(smallm_xq_));
    smallm_xq_ = nullptr;
    smallm_xq_bytes_ = 0;
    smallm_arena_ = false;
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

    vfree(gdn_chunkpar_ws_);
    gdn_chunkpar_ws_bytes_ = 0;
    vfree(gdn_ab_ws_);
    gdn_ab_ws_bytes_ = 0;

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
        // Registry-owned overlays (Phase 4.2): fused_kv / fused_gate_up storage is owned by
        // WeightRegistry handles, not wcache_. The maps below stay empty (pre_dequant_weights);
        // this helper frees any handle with owned_bytes > 0.
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
        // Per-(layer, projection) SfAtom slabs from the MoE phase. Their
        // slices are sf_borrowed, so this is the only owner.
        for (void* slab : wcache_.owned_sf_slabs)
            vram_free(vram_alloc_, slab);
        wcache_.owned_sf_slabs.clear();
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
                bool in_ssm_sidecar =
                    wcache_.fp8_ssm_sidecar_data &&
                    reinterpret_cast<uintptr_t>(entry.weight.data) >=
                        reinterpret_cast<uintptr_t>(wcache_.fp8_ssm_sidecar_data) &&
                    reinterpret_cast<uintptr_t>(entry.weight.data) <
                        reinterpret_cast<uintptr_t>(wcache_.fp8_ssm_sidecar_data) +
                            wcache_.fp8_ssm_sidecar_data_size;
                if (!in_migrated && !in_overflow && !in_ssm_sidecar)
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
        if (wcache_.fp8_ssm_sidecar_data) {
            vram_free(vram_alloc_, wcache_.fp8_ssm_sidecar_data);
            wcache_.fp8_ssm_sidecar_data = nullptr;
            wcache_.fp8_ssm_sidecar_data_size = 0;
        }
        if (wcache_.fp8_ssm_sidecar_row_scales) {
            IMP_CUDA_CHECK_LOG(cudaFree(wcache_.fp8_ssm_sidecar_row_scales));
            wcache_.fp8_ssm_sidecar_row_scales = nullptr;
        }
    }

    qscratch_.free(vram_alloc_);

    moe_.free(vram_alloc_);
    expert_cache_.destroy();

    // Free gemm_nvfp4 dequant workspace and unregister from the free function.
    if (nvfp4_dequant_ws_buf_) {
        set_nvfp4_dequant_workspace(nullptr, 0);
        if (!nvfp4_dequant_ws_from_arena_)
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
    // MLA quartet and absorbed cache are arena-owned since A7 step 4b.2: no frees here;
    // ~Engine closes the arena after every executor teardown.
    mla_kv_a_buf_ = nullptr;
    mla_latent_buf_ = nullptr;
    mla_k_rope_buf_ = nullptr;
    mla_kv_b_buf_ = nullptr;
    mla_absorb_cache_ = nullptr;
    mla_absorb_scores_ = nullptr;
    if (d_sample_result_) {
        // Arena-owned since A7 step 4b.2 — no free here; ~Engine closes the arena
        // after every executor teardown.
        d_sample_result_ = nullptr;
    }
    // T5b owners: reset() releases, and doing it twice is a no-op — the
    // hand-written cudaFreeHost pairs are gone (memory/host_pinned.h).
    h_sample_pinned_.reset();
    h_row_args_.reset();
    h_greedy_args_.reset();
    h_pen_args_.reset();
    d_greedy_args_ = nullptr;  // arena-owned, like d_row_args_ below
    d_pen_args_ = nullptr;
    n_pending_greedy_rows_ = 0;
    n_pending_pen_rows_ = 0;
    if (d_banned_cache_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_banned_cache_));
        d_banned_cache_ = nullptr;
        banned_cache_src_ = nullptr;
        banned_cache_n_ = 0;
        banned_cache_capacity_ = 0;
    }
    if (d_row_args_) {
        // Arena-owned since A7 step 4b.2 — no free here. The arena is closed by
        // ~Engine, after every executor teardown.
        d_row_args_ = nullptr;
    }
    for (int p = 0; p < 2; ++p) {
        if (sample_gather_evt_[p]) {
            IMP_CUDA_CHECK_LOG(cudaEventDestroy(sample_gather_evt_[p]));
            sample_gather_evt_[p] = nullptr;
        }
    }
    sample_parity_ = 0;
    h_logits_pinned_.reset();
    h_logits_pinned_size_ = 0;
    vfree(attn_scores_buf_);
    attn_scores_buf_size_ = 0;
    // Arena-owned since A7 step 4b.2 — ~Engine closes the region; re-arm only.
    chunk_capture_k_ = nullptr;
    chunk_capture_v_ = nullptr;
    chunk_capture_ctx_ = 0;
    // cudaFreeAsync, not cudaFree: the grow path (executor_attention_prefill.cu) allocates
    // these with cudaMallocAsync, and a sync free on a stream-ordered allocation returns
    // success without returning the block to the async mempool (#834). Null stream + sync,
    // matching ~Model.
    if (chunk_eager_k_) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(chunk_eager_k_, nullptr));
        chunk_eager_k_ = nullptr;
    }
    if (chunk_eager_v_) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(chunk_eager_v_, nullptr));
        chunk_eager_v_ = nullptr;
    }
    if (chunk_eager_bytes_ > 0)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(nullptr));  // retire the frees for the pool
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

bool GraphExecutor::fa2_serves_all_prefill() const {
    const auto& cfg = model_->config();
    int hd_for_attn = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
    for (int x : cfg.head_dim_per_layer) {
        if (x > 0) {
            hd_for_attn = x;  // hybrids: first attention layer's head_dim
            break;
        }
    }
    const bool fa2_hd_ok = hd_for_attn == 128 ||
                           (hd_for_attn == 256 && dispatch_policy().attention.fa2_hd256);
    return fa2_hd_ok && dispatch_policy().attention.fa2_fp16qk != "never" &&
           attn_shapes_uniform() && !model_->profile().is_gpt_oss;
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
    const auto& att = dispatch_policy().attention;
    // Uniform attention head_dim (first nonzero per-layer value, else global).
    int hd_u = cfg.head_dim > 0 ? cfg.head_dim
                                : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
    for (int x : cfg.head_dim_per_layer) {
        if (x > 0) {
            hd_u = x;
            break;
        }
    }
    // #1675: a sink model routes straight to the FP16 WMMA FMHA tier
    // (attention_dispatch.cu:65-71) since #992, and that tier needs no
    // S-matrix. All three no-clamp returns below excluded sinks, so gpt-oss
    // took the quadratic clamp for a reason the dispatch stopped having - the
    // chunk collapsing with offset, at no benefit. The condition mirrors the
    // dispatch exactly: if the tier declines, the dispatch throws rather than
    // falling back to something that would need the S-matrix.
    if (sinks && uniform && att.fmha_sm120 != "never" && fmha_serves_head_dim(hd_u))
        return desired;
    // Mirrors the chunked dispatch in executor_attention_prefill.cu:
    // FP16-QK FA2 serves every hd=128 chunk with no S-matrix.
    if (uniform && !sinks && hd_u == 128 && att.fa2_fp16qk != "never")
        return desired;
    // Tiled FMHA dispatch serves chunks whose ctx_len crosses the threshold (or that the
    // S-matrix cannot hold) with no S-matrix, but only for head dims it covers. Without this
    // check, an unclamped head_dim (e.g. MLA's 192) reaches the cuBLAS fallback and aborts on
    // its own S-matrix bound. See attention_dispatch_rules.h.
    if (uniform && !sinks && fmha_serves_head_dim(hd_u) && att.fmha_prefill_threshold > 0 &&
        offset + desired >= att.fmha_prefill_threshold)
        return desired;
    // Heterogeneous per-layer shapes (e.g. Gemma-4 dual head_dim 256/512): every layer is
    // served at any chunk x ctx (hd 128/256 via FA2 per-layer, hd=512 via cuBLAS in
    // workspace-sized q-row slices, tiled FMHA covers the rest), so no global clamp is needed.
    // A global clamp would shrink every layer's chunk to the smallest S-matrix capacity.
    if (!uniform && !sinks) {
        bool all_served = true;
        for (int x : cfg.head_dim_per_layer) {
            if (x <= 0)
                continue;  // non-attention layers (GDN/Mamba2 hybrids)
            const bool fa2 = fa2_serves_head_dim(x, att.fa2_hd256) && att.fa2_fp16qk != "never";
            const bool fmha = fmha_serves_head_dim(x);
            if (!fa2 && !fmha) {
                all_served = false;
                break;
            }
        }
        if (all_served)
            return desired;
    }
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
    // FP16-QK FA2 is the only device-length chunked attention kernel. hd=256 rides the #930
    // port (d_kv_len is a runtime kernel arg); GDN/Mamba2 recurrent kernels stop state
    // updates at the device chunk length, so hd=256 hybrids are capture-eligible only when
    // fa2_hd256 is on.
    if (hd_u != 128 && !(hd_u == 256 && dispatch_policy().attention.fa2_hd256))
        return false;
    if (dispatch_policy().attention.fa2_fp16qk == "never")
        return false;
    // MoE: only the CUTLASS 3.x device-args grouped path records into a graph without
    // host-side routing reads; every other MoE prefill path does a D2H+sync per layer to size
    // expert GEMMs (capture-illegal, guarded by moe_host_args_capture_guard). Require static
    // device-args conditions for every MoE layer; models with experts outside the CUTLASS
    // tier fall out here until a device-args grouped GEMM exists for that layout (#847).
    bool any_moe = false;
    for (int i = 0; i < cfg.n_layers && !any_moe; ++i)
        any_moe = layer_has_moe(i);
    // moe.skip (debug) removes the MoE pass from eager and capture alike —
    // the recorded graph stays consistent, so it does not block capture.
    if (any_moe && !dispatch_policy().moe.skip) {
        if (dispatch_policy().moe.no_cutlass3x || !dispatch_policy().moe.nvfp4_device_args)
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
    // T2 (A7 step 4b.2, the last exec/ holdout): charges 2 x capture_ctx_cap x nkv x hd
    // halves once (not a growing staircase of takes), matching what the caller asks for
    // (engine_spec_capture.cpp clamps the cap to max_seq_len). Both pointers are baked into
    // the captured verify graph, so cudaFree here would hand a replay a freed address (AUDIT B13).
    chunk_capture_ctx_ = 0;
    auto k_slab = engine_arena().take_bytes(bytes);
    auto v_slab = engine_arena().take_bytes(bytes);
    if (k_slab.empty() || v_slab.empty()) {
        IMP_LOG_WARN("chunk-capture scratch unavailable from the T2 arena (2x %.1f MiB, %.1f MiB "
                     "free) — captured verify disabled",
                     bytes / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
        chunk_capture_k_ = nullptr;
        chunk_capture_v_ = nullptr;
        return false;
    }
    chunk_capture_k_ = reinterpret_cast<half*>(k_slab.data());
    chunk_capture_v_ = reinterpret_cast<half*>(v_slab.data());
    chunk_capture_ctx_ = ctx_capacity;
    IMP_LOG_INFO("chunk-capture K/V scratch: 2x %.1f MiB (ctx_capacity=%d, nkv=%d, hd=%d)",
                 bytes / (1024.0 * 1024.0), ctx_capacity, nkv_u, hd_u);
    return true;
}

// pre_dequant_weights() is in executor_pre_dequant.cu
// configure_*_workspace(), resize_workspace(), allocate_decode_workspace(),
// use_workspace(), layer_has_*(), view_tokens(), ensure_logits_pinned()
// are in executor_workspace_config.cu

// ---------------------------------------------------------------------------
// Prefill/decode overlap: per-slot quant scratches + the slot switch that
// swaps them (docs/plans/2026-08-27-prefill-decode-overlap.md).
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_decode_qscratch(int max_batch) {
    if (qscratch_decode_ready_)
        return true;
    if (max_batch <= 0)
        return false;
    // Mirror the prefill-sized originals at decode-batch rows. A null
    // original means that family is unused on this model — the copy stays
    // null and its consumers keep declining exactly as they do today.
    size_t total = 0;
    if (qscratch_.fp8_act && qscratch_.fp8_act_size > 0 && max_tokens_ > 0) {
        const size_t per_row = qscratch_.fp8_act_size / static_cast<size_t>(max_tokens_);
        qscratch_decode_.fp8_act_size = per_row * max_batch;
        auto sl = engine_arena().take_bytes(qscratch_decode_.fp8_act_size);
        auto sc = engine_arena().take_bytes(sizeof(float));
        int max_n = static_cast<int>(qscratch_decode_.fp8_act_size);
        qscratch_decode_.fp8_max_grid = (((max_n + 3) / 4) + 255) / 256;
        auto bm = engine_arena().take_bytes(static_cast<size_t>(qscratch_decode_.fp8_max_grid) *
                                            sizeof(float));
        auto am = engine_arena().take_bytes(sizeof(float));
        if (sl.empty() || sc.empty() || bm.empty() || am.empty()) {
            IMP_LOG_WARN("decode qscratch: fp8 family unavailable from the T2 arena");
            return false;
        }
        qscratch_decode_.fp8_act = sl.data();
        qscratch_decode_.d_act_scale = reinterpret_cast<float*>(sc.data());
        qscratch_decode_.d_fp8_block_maxes = reinterpret_cast<float*>(bm.data());
        qscratch_decode_.d_fp8_absmax = reinterpret_cast<float*>(am.data());
        total += qscratch_decode_.fp8_act_size;
    }
    if (qscratch_.q8_1_buf && qscratch_.q8_1_max_blocks > 0) {
        qscratch_decode_.q8_1_max_blocks = qscratch_.q8_1_max_blocks;
        qscratch_decode_.q8_1_rows = max_batch;
        size_t q8_sz = static_cast<size_t>(qscratch_decode_.q8_1_max_blocks) * max_batch *
                       sizeof(block_q8_1);
        size_t d8_sz = static_cast<size_t>(qscratch_decode_.q8_1_max_blocks) * max_batch *
                       sizeof(float);
        auto q8 = engine_arena().take_bytes(q8_sz);
        auto d8 = engine_arena().take_bytes(d8_sz);
        if (q8.empty() || d8.empty()) {
            IMP_LOG_WARN("decode qscratch: q8_1 family unavailable from the T2 arena");
            return false;
        }
        qscratch_decode_.q8_1_buf = q8.data();
        qscratch_decode_.d8_buf = reinterpret_cast<float*>(d8.data());
        total += q8_sz + d8_sz;
    }
    qscratch_decode_ready_ = true;
    IMP_LOG_INFO("decode qscratch: per-slot fp8/q8 copies, %.1f KiB for max_batch=%d",
                 total / 1024.0, max_batch);
    return true;
}

void GraphExecutor::use_workspace(int slot) {
    const int prev = ws_.active();
    ws_.use_workspace(slot);
    if (!qscratch_decode_ready_ || ws_.active() == prev)
        return;
    if (slot == 1) {
        qscratch_prefill_save_ = {qscratch_.fp8_act,      qscratch_.fp8_act_size,
                                  qscratch_.d_act_scale,  qscratch_.d_fp8_block_maxes,
                                  qscratch_.d_fp8_absmax, qscratch_.fp8_max_grid,
                                  qscratch_.q8_1_buf,     qscratch_.d8_buf,
                                  qscratch_.q8_1_rows};
        if (qscratch_decode_.fp8_act) {
            qscratch_.fp8_act = qscratch_decode_.fp8_act;
            qscratch_.fp8_act_size = qscratch_decode_.fp8_act_size;
            qscratch_.d_act_scale = qscratch_decode_.d_act_scale;
            qscratch_.d_fp8_block_maxes = qscratch_decode_.d_fp8_block_maxes;
            qscratch_.d_fp8_absmax = qscratch_decode_.d_fp8_absmax;
            qscratch_.fp8_max_grid = qscratch_decode_.fp8_max_grid;
        }
        if (qscratch_decode_.q8_1_buf) {
            qscratch_.q8_1_buf = qscratch_decode_.q8_1_buf;
            qscratch_.d8_buf = qscratch_decode_.d8_buf;
            qscratch_.q8_1_rows = qscratch_decode_.q8_1_rows;
        }
    } else {
        qscratch_.fp8_act = qscratch_prefill_save_.fp8_act;
        qscratch_.fp8_act_size = qscratch_prefill_save_.fp8_act_size;
        qscratch_.d_act_scale = qscratch_prefill_save_.d_act_scale;
        qscratch_.d_fp8_block_maxes = qscratch_prefill_save_.d_fp8_block_maxes;
        qscratch_.d_fp8_absmax = qscratch_prefill_save_.d_fp8_absmax;
        qscratch_.fp8_max_grid = qscratch_prefill_save_.fp8_max_grid;
        qscratch_.q8_1_buf = qscratch_prefill_save_.q8_1_buf;
        qscratch_.d8_buf = qscratch_prefill_save_.d8_buf;
        qscratch_.q8_1_rows = qscratch_prefill_save_.q8_1_rows;
    }
}

bool GraphExecutor::has_gguf_nvfp4_overlay() const {
    // "GGUF class" for the overlap gate = any registered weight whose source is a
    // GPU-dequantable GGUF qtype (decodes via the dp4a/dequant scratches prefill shares).
    // wcache_.nvfp4 is the wrong predicate: it's also populated on native-NVFP4 models.
    for (size_t i = 0; i < registry_.size(); ++i)
        if (dequant_gpu_supported(registry_.handle(static_cast<TensorID>(i)).source_qtype))
            return true;
    return false;
}

}  // namespace imp

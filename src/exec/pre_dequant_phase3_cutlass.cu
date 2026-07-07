// Pre-dequant Phase 3 (CUTLASS / MXFP4): NVFP4→CUTLASS sm_120 conversion,
// native MXFP4 registration, and the MXFP4→FP16 decode fallback.
// Split out of pre_dequant_phase3_nvfp4_decode.cu to keep each .cu under the
// kernel file-size threshold. See pre_dequant_internal.h / quant_pipeline.h
// for shared declarations.

#include "exec/executor.h"
#include "memory/vram_query.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <cstring>
#include <vector>

namespace imp {

using imp::pre_dequant_internal::deduct_budget;

// Phase 3b: convert NVFP4 weights into CUTLASS sm_120 block-scaled format.
// Must run after FP16-free; the CUTLASS cache approximately doubles NVFP4
// VRAM (repacked data + SfAtom scales).  Budget-aware: stops if VRAM
// budget runs out and emits an info line.
void QuantPipeline::nvfp4_decode_convert_cutlass_(const ModelConfig& cfg, size_t& remaining_budget,
                                                  cudaStream_t stream) {
    // After incremental mode, remaining_budget is stale.  Use actual free VRAM.
    size_t ct_budget;
    if (wcache_->nvfp4_decode_mode == 2) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        size_t free_mem = 0, total_mem = 0;
        vram_budget_mem_get_info(&free_mem, &total_mem);
        // Intentionally NOT using dctx.safety_reserve here: populating
        // cutlass_nvfp4 in mode 2 destabilised CUDA-graph capture on
        // Qwen3-14B Q6_K (bimodal 97 vs 145 tok/s decode across trials).
        // The dense in-loop safety relaxation already delivers the +15%
        // decode win; the CUTLASS path stays conservative until the
        // capture-failure root cause is understood.
        size_t kCtReserve = vram_reserve_floor(total_mem);
        ct_budget = (free_mem > kCtReserve) ? (free_mem - kCtReserve) : 0;
    } else {
        ct_budget = (remaining_budget > wcache_->nvfp4_bytes)
                        ? (remaining_budget - wcache_->nvfp4_bytes)
                        : 0;
    }
    // SfAtom scale factors share ONE slab allocation (each entry borrows a
    // sub-region) instead of a per-tensor cudaMalloc+cudaMemsetAsync (#734). MoE
    // experts additionally convert in ONE batched launch per (layer,projection)
    // group instead of one launch per expert: on a 128-expert MoE that collapses
    // ~18.6k convert launches into ~144. Pass 1 sizes the slab (contiguous MoE
    // groups first, then per-tensor dense / non-contiguous entries) under the
    // SAME skip + VRAM-budget rules; pass 2 converts each into its slab offset.
    constexpr size_t kSfAlign = 256;  // keep each entry's SF base CUTLASS-aligned
    auto align_up = [](size_t x, size_t a) { return (x + a - 1) / a * a; };

    // Identify contiguous MoE expert groups from the model. The grouping comes
    // from the model because wcache_->nvfp4_moe is not populated until a later
    // phase. The loader's contiguity invariant is re-checked here; any group
    // that is non-contiguous, non-uniform, or partly absent from the decode
    // cache falls back to the per-tensor path below (correctness over speed).
    struct MoeGroup {
        const void* base_ms;
        int ne, N, K;
        size_t sf_per_expert, slab_off;
        std::vector<float> tscale;
        std::vector<const void*> data_ptrs;
    };
    std::vector<MoeGroup> moe_groups;
    std::unordered_set<const void*> grouped;  // expert data ptrs handled by a group
    auto try_group = [&](const std::vector<Tensor>& experts) {
        int ne = static_cast<int>(experts.size());
        if (ne < 2 || !experts[0].data || !experts[0].scales)
            return;
        const int N = static_cast<int>(experts[0].shape[0]);
        const int Kp = static_cast<int>(experts[0].shape[1]);  // packed K/2
        const int K = Kp * 2;
        const size_t e_ms = static_cast<size_t>(N) * (K / 16);  // micro-scale bytes/expert
        for (int e = 0; e < ne; ++e) {
            const Tensor& w = experts[e];
            if (!w.data || !w.scales || static_cast<int>(w.shape[0]) != N ||
                static_cast<int>(w.shape[1]) != Kp)
                return;  // non-uniform shape
            if (static_cast<const char*>(w.scales) !=
                static_cast<const char*>(experts[0].scales) + static_cast<size_t>(e) * e_ms)
                return;  // micro-scales not contiguous → per-tensor fallback
            if (!wcache_->nvfp4.count(w.data))
                return;  // not a registered CUTLASS candidate
        }
        MoeGroup g;
        g.base_ms = experts[0].scales;
        g.ne = ne;
        g.N = N;
        g.K = K;
        g.sf_per_expert = cutlass_nvfp4_sf_size(N, K);
        g.slab_off = 0;
        g.tscale.resize(ne);
        g.data_ptrs.resize(ne);
        for (int e = 0; e < ne; ++e) {
            g.tscale[e] = experts[e].tensor_scale;
            g.data_ptrs[e] = experts[e].data;
            grouped.insert(experts[e].data);
        }
        moe_groups.push_back(std::move(g));
    };
    for (int i = 0; i < cfg.n_layers; ++i) {
        const auto& L = model_->layer(i);
        try_group(L.expert_w_gate);
        try_group(L.expert_w_up);
        try_group(L.expert_w_down);
    }

    // Pass 1: size the slab. Groups first (each placed contiguously so the
    // batched kernel can stride by sf_per_expert), then per-tensor entries.
    std::vector<std::pair<const void*, size_t>> ct_included;  // (src ptr, slab offset)
    size_t ct_total = 0;       // running padded slab size = real SF VRAM
    size_t n_groups_inc = 0;   // leading moe_groups that fit the budget
    int ct_skipped_dead = 0;
    bool ct_exhausted = false;
    for (auto& g : moe_groups) {
        size_t group_sf = align_up(static_cast<size_t>(g.ne) * g.sf_per_expert, kSfAlign);
        if (ct_total + group_sf > ct_budget) {
            ct_exhausted = true;
            break;
        }
        g.slab_off = ct_total;
        ct_total += group_sf;
        ++n_groups_inc;
    }
    if (!ct_exhausted) {
        for (auto& [ptr, nvfp4] : wcache_->nvfp4) {
            if (grouped.count(ptr))
                continue;  // converted by a batched MoE group above
            // G3 (Stage 1.4): skip the CUTLASS SF buffer for weights whose M>1
            // prefill uses IMMA raw-read on the GGUF source — the CUTLASS GEMM
            // path is never reached for them, so the SF buffer is dead VRAM.
            // Today that is Q8_0 with q8_imma_enabled. CUTLASS stays for native-
            // NVFP4 (prefill IS CUTLASS) and Q6_K/Q5_K. Decode is unaffected:
            // decode_tier stays NVFP4 (the plain wcache_->nvfp4 GEMV).
            const auto* pe = storage_plan_.entry_of(ptr);
            if (pe && pe->source_qtype == QType::Q8_0 && runtime_config().gemm.q8_imma_enabled) {
                ++ct_skipped_dead;
                continue;
            }
            size_t est = align_up(
                cutlass_nvfp4_sf_size(static_cast<int>(nvfp4.N), static_cast<int>(nvfp4.K)), kSfAlign);
            if (ct_total + est > ct_budget) {
                ct_exhausted = true;
                break;
            }
            ct_included.emplace_back(ptr, ct_total);
            ct_total += est;
        }
    }
    if (ct_exhausted)
        IMP_LOG_INFO("CUTLASS NVFP4 cache: VRAM budget reached (%.1f / %.1f MiB)",
                     ct_total / (1024.0 * 1024.0), ct_budget / (1024.0 * 1024.0));

    int ct_count = 0;
    if (ct_total > 0) {
        void* sf_slab = nullptr;
        IMP_CUDA_CHECK_LOG(cudaMalloc(&sf_slab, ct_total));
        // Zero every entry's SfAtom padding rows at once (convert kernels write
        // only valid (n, k_group) cells).
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(sf_slab, 0, ct_total, stream));
        wcache_->cutlass_sf_slab = sf_slab;
        wcache_->cutlass_sf_slab_size = ct_total;
        auto* slab = static_cast<uint8_t*>(sf_slab);
        // Pass 2a: one batched convert per MoE group → per-expert borrowed slices.
        for (size_t gi = 0; gi < n_groups_inc; ++gi) {
            const MoeGroup& g = moe_groups[gi];
            convert_nvfp4_moe_scales_to_sfatom(g.base_ms, slab + g.slab_off, g.ne, g.N, g.K, stream);
            for (int e = 0; e < g.ne; ++e) {
                CutlassNvFP4Weight cw;
                cw.data = g.data_ptrs[e];
                cw.scale_factors = slab + g.slab_off + static_cast<size_t>(e) * g.sf_per_expert;
                cw.tensor_scale = g.tscale[e];
                cw.N = g.N;
                cw.K = g.K;
                cw.sf_bytes = g.sf_per_expert;
                cw.sf_borrowed = true;
                wcache_->cutlass_nvfp4[g.data_ptrs[e]] = cw;
                ++ct_count;
            }
        }
        // Pass 2b: per-tensor convert for dense / non-contiguous entries.
        for (auto& [ptr, off] : ct_included) {
            auto it = wcache_->nvfp4.find(ptr);
            if (it == wcache_->nvfp4.end())
                continue;
            CutlassNvFP4Weight cw;
            convert_nvfp4_to_cutlass_borrowed(it->second, cw, slab + off, stream);
            if (cw.data) {
                wcache_->cutlass_nvfp4[ptr] = cw;
                ct_count++;
            }
        }
    }
    if (ct_count > 0) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        wcache_->cutlass_nvfp4_bytes = ct_total;
        deduct_budget(remaining_budget, ct_total + wcache_->nvfp4_bytes);
        IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: %d tensors, %.2f MiB (skipped %d "
                     "IMMA-prefill weights — dead SF buffer)",
                     ct_count, ct_total / (1024.0 * 1024.0), ct_skipped_dead);
    } else if (ct_skipped_dead > 0) {
        IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: 0 tensors (skipped %d IMMA-prefill "
                     "weights — dead SF buffer)",
                     ct_skipped_dead);
    }
}

// Phase 3c-native: register MXFP4 GGUF weights directly in CUTLASS cache.
// Bypasses NVFP4 — the GGUF data is unpacked into E2M1 + SfAtom UE8M0 on
// GPU. Allocates the MXFP4 activation scratch once if any layer carries
// MXFP4 weights, then runs an optional NVFP4->MXFP4 conversion pass for
// models with `use_mxfp4`.
void QuantPipeline::nvfp4_decode_convert_mxfp4_and_native_(const ModelConfig& cfg, cudaStream_t stream) {
    // These bypass NVFP4 entirely — the GGUF data is unpacked into
    // separate E2M1 data + SfAtom UE8M0 scales on GPU.
    // For native MXFP4, allocate activation buffers if not already done.
    if (cutlass_sm120_mxfp4_available()) {
        // Check if any layer has MXFP4 weights
        bool has_mxfp4 = false;
        auto check_mxfp4 = [&](const Tensor&, QType qt) {
            if (qt == QType::MXFP4)
                has_mxfp4 = true;
        };
        for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
            const auto& L = model_->layer(i);
            check_mxfp4(L.wq, L.wq.qtype);
            check_mxfp4(L.wk, L.wk.qtype);
            check_mxfp4(L.w_gate, L.w_gate.qtype);
            check_mxfp4(L.ssm_in, L.ssm_in.qtype);
            check_mxfp4(L.ssm_out, L.ssm_out.qtype);
        }

        // Allocate MXFP4 scratch if needed and not already allocated
        if (has_mxfp4 && !qscratch_->mxfp4_act_sf) {
            int max_k = 0, max_n = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                if (L.wq.data && L.wq.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.wq.shape[0]);
                    max_k = std::max(max_k, (int)L.wq.shape[1]);
                }
                if (L.w_gate.data && L.w_gate.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.w_gate.shape[0]);
                    max_k = std::max(max_k, (int)L.w_gate.shape[1]);
                }
                if (L.w_down.data && L.w_down.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.w_down.shape[0]);
                    max_k = std::max(max_k, (int)L.w_down.shape[1]);
                }
                if (L.ssm_in.data && L.ssm_in.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.ssm_in.shape[0]);
                    max_k = std::max(max_k, (int)L.ssm_in.shape[1]);
                }
                if (L.ssm_out.data && L.ssm_out.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.ssm_out.shape[0]);
                    max_k = std::max(max_k, (int)L.ssm_out.shape[1]);
                }
            }
            if (max_k > 0) {
                qscratch_->mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                qscratch_->mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_,
                                                                                    max_n, max_k);
                qscratch_->mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_->mxfp4_act_sf_size,
                                                    "mxfp4_act_sf");
                qscratch_->mxfp4_workspace = (qscratch_->mxfp4_workspace_size > 0)
                                                ? vram_alloc(vram_alloc_,
                                                             qscratch_->mxfp4_workspace_size,
                                                             "mxfp4_workspace")
                                                : nullptr;
                // Also need CUTLASS activation data buffer
                if (!qscratch_->cutlass_act_data) {
                    qscratch_->cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                    qscratch_->cutlass_act_data = vram_alloc(vram_alloc_,
                                                            qscratch_->cutlass_act_data_size,
                                                            "cutlass_act_data");
                }
                IMP_LOG_INFO("Native MXFP4: allocated activation scratch (sf=%.2f MiB)",
                             qscratch_->mxfp4_act_sf_size / (1024.0 * 1024.0));
            }
        }
    }

    // Convert NVFP4 weights to MXFP4 (UE8M0 scales) if MXFP4 prefill is enabled.
    // Same packed FP4 data (borrowed), only allocates new scale factor buffers.
    // Note: Hadamard rotation requires MR-GPTQ pre-rotated weights (SafeTensors).
    // For GGUF models, we use direct scale conversion (no rotation).
    if (wcache_->use_mxfp4 && qscratch_->mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
        int mx_count = 0;
        size_t mx_total = 0;
        for (auto& [ptr, nvfp4] : wcache_->nvfp4) {
            // Only convert weights where K is multiple of 32 (MXFP4 requirement)
            if (nvfp4.K % 32 != 0)
                continue;
            CutlassMxFP4Weight mw;
            convert_nvfp4_to_mxfp4_cutlass(nvfp4, mw, stream);
            if (mw.data) {
                wcache_->cutlass_mxfp4[ptr] = mw;
                mx_total += mw.sf_bytes;
                mx_count++;
            }
        }
        if (mx_count > 0) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            wcache_->cutlass_mxfp4_bytes = mx_total;
            IMP_LOG_INFO("CUTLASS sm_120 MXFP4 weight cache: %d tensors, %.2f MiB", mx_count,
                         mx_total / (1024.0 * 1024.0));
        }
    }
}

// Native MXFP4 GGUF unpack + FP16 fallback dequant. Registers MXFP4 weights
// directly in the CUTLASS cache, then for GDN / forced-fallback models
// dequants them into a bulk FP16 buffer and rewrites model weight pointers
// so the dispatch path sees FP16 instead of raw MXFP4 blocks.
void QuantPipeline::nvfp4_decode_mxfp4_fp16_fallback_(const ModelConfig& cfg, cudaStream_t stream) {
int mx_native = 0;
size_t mx_native_bytes = 0;
auto register_if_mxfp4 = [&](const Tensor& w, QType qt, bool is_attn = true) {
    if (qt != QType::MXFP4 || !w.data || !w.on_device)
        return;
    if (w.ndim < 2 || w.shape[1] % 32 != 0)
        return;
    if (wcache_->cutlass_mxfp4.count(w.data))
        return;  // already registered
    CutlassMxFP4Weight mw;
    if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
        mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
        wcache_->cutlass_mxfp4[w.data] = mw;
        mx_native_bytes += mw.sf_bytes + static_cast<size_t>(w.shape[0]) * (w.shape[1] / 2);
        mx_native++;
    }
};
for (int i = 0; i < cfg.n_layers; i++) {
    const auto& L = model_->layer(i);
    register_if_mxfp4(L.wq, L.wq.qtype, true);
    register_if_mxfp4(L.wk, L.wk.qtype, true);
    register_if_mxfp4(L.wv, L.wv.qtype, true);
    register_if_mxfp4(L.wo, L.wo.qtype, true);
    register_if_mxfp4(L.w_up, L.w_up.qtype, false);
    register_if_mxfp4(L.w_gate, L.w_gate.qtype, false);
    register_if_mxfp4(L.w_down, L.w_down.qtype, false);
    // GDN-specific weights (Qwen3.5)
    register_if_mxfp4(L.ssm_in, L.ssm_in.qtype, true);
    register_if_mxfp4(L.ssm_out, L.ssm_out.qtype, true);
    register_if_mxfp4(L.gdn_gate, L.gdn_gate.qtype, true);
    register_if_mxfp4(L.gdn_alpha, L.gdn_alpha.qtype, true);
    register_if_mxfp4(L.gdn_beta, L.gdn_beta.qtype, true);
}
register_if_mxfp4(model_->output_proj(), model_->out_proj_.qtype);
if (mx_native > 0) {
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    wcache_->cutlass_mxfp4_bytes += mx_native_bytes;
    wcache_->use_mxfp4 = true;
    // unpack_mxfp4_gguf compacts the GGUF raw blocks IN PLACE inside the
    // model's source buffers — a second engine on this model handle cannot
    // re-run the unpack (it would read the already-compacted layout as raw
    // blocks → illegal access, #830). Engine::init rejects it up front.
    const_cast<Model*>(model_)->mark_sources_consumed();
    IMP_LOG_INFO("Native MXFP4 GGUF: %d tensors, %.2f MiB (direct → CUTLASS)", mx_native,
                 mx_native_bytes / (1024.0 * 1024.0));

    // Sync and check for errors from unpack kernels
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            IMP_LOG_ERROR("MXFP4 unpack error: %s", cudaGetErrorString(e));
    }

    // Check if MXFP4 GEMV is available (linear_scales populated).
    // GDN models force the FP16 fallback because — although every
    // linear projection in executor_ssm_gdn.cu *does* now go through
    // gemm_dispatch — the MXFP4 prefill dispatch path for GDN-shape
    // weights (notably Qwen3.5-27B's ssm_out at K=6144 N=5120, and
    // FFN at K=17408 N=5120) hits a cuBLAS-INTERNAL_ERROR (status
    // 14) cascade we have not yet root-caused. Tracking in
    // qwen35_27b_mxfp4_ima_2026_04_25.md. Until that's resolved,
    // honor the historical fallback path.
    bool force_fallback = runtime_config().attention.mxfp4_fp16_fallback;
    bool has_gdn = (cfg.ssm_inner_size > 0);
    bool mxfp4_gemv_available = !force_fallback && !has_gdn;
    for (auto& [p, m] : wcache_->cutlass_mxfp4)
        if (!m.linear_scales) {
            mxfp4_gemv_available = false;
            break;
        }

    if (mxfp4_gemv_available) {
        IMP_LOG_INFO("MXFP4 GEMV: all %d weights have linear_scales, skipping FP16 fallback",
                     mx_native);
    }

    // Dequant MXFP4 → FP16 for decode (only when MXFP4 GEMV not available).
    // Single bulk allocation to avoid CUDA heap fragmentation.
    //
    // Phase A2 (`docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md`):
    // when attention.mxfp4_fp16_cache_policy == "pruned", skip
    // tensor slots that aren't read on the dispatch hot path —
    // MoE expert_*_packed (consumed only by executor_forward_moe.cu's
    // pre-cached FP16 path, which the MXFP4 batch-dequant route
    // bypasses) and the LM head out_proj_ (routed through
    // generic-dequant). For Qwen3.5-27B MXFP4 this shrinks
    // the FP16 fallback from ~48 GiB to ~8-12 GiB and unblocks
    // load on 32 GiB VRAM.
    std::unordered_set<const void*> pruned_skip_ptrs;
    const bool pruned_policy =
        (runtime_config().attention.mxfp4_fp16_cache_policy == "pruned");
    if (pruned_policy) {
        for (int li = 0; li < cfg.n_layers; ++li) {
            const auto& L = model_->layer(li);
            if (L.expert_gate_packed.data)
                pruned_skip_ptrs.insert(L.expert_gate_packed.data);
            if (L.expert_up_packed.data)
                pruned_skip_ptrs.insert(L.expert_up_packed.data);
            if (L.expert_down_packed.data)
                pruned_skip_ptrs.insert(L.expert_down_packed.data);
        }
        if (model_->output_proj().data)
            pruned_skip_ptrs.insert(model_->output_proj().data);
    }
    size_t pruned_skipped_bytes = 0;
    int pruned_skipped_count = 0;

    size_t fp16_total = 0;
    if (!mxfp4_gemv_available) {
        for (auto& [p, m] : wcache_->cutlass_mxfp4) {
            if (wcache_->fp16.count(p))
                continue;
            size_t b = static_cast<size_t>(m.N) * m.K * sizeof(half);
            if (pruned_policy && pruned_skip_ptrs.count(p)) {
                pruned_skipped_bytes += b;
                pruned_skipped_count++;
                continue;
            }
            fp16_total += b;
        }
        if (pruned_policy && pruned_skipped_count > 0) {
            IMP_LOG_INFO(
                "MXFP4 FP16 cache pruning: skipping %d MoE/LM-head tensors "
                "(%.2f GiB saved)",
                pruned_skipped_count,
                pruned_skipped_bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    void* d_fp16_bulk = nullptr;
    if (fp16_total > 0) {
        // Pre-flight VRAM check: WSL2/WDDM cudaMalloc happily pages over
        // the device boundary into host RAM. cuBLASLt then fails at
        // runtime when it can't allocate its internal workspace → status
        // 14 (INVALID_VALUE) followed by a confusing downstream illegal
        // memory access (observed on Qwen3.5-27B-mxfp4 GDN where the
        // 12 GiB MXFP4 raw + 48 GiB FP16 fallback exceed 32 GiB VRAM).
        // Refuse the alloc instead of paging — keeps the failure mode
        // legible.
        size_t free_mem = 0, total_mem = 0;
        vram_budget_mem_get_info(&free_mem, &total_mem);
        constexpr size_t kRuntimeHeadroom = static_cast<size_t>(2) * 1024 * 1024 * 1024;
        bool oversubscribe = (free_mem <= kRuntimeHeadroom ||
                              fp16_total + kRuntimeHeadroom > free_mem);
        // The "force anyway despite oversubscription" path is gone —
        // attention.mxfp4_fp16_fallback is a plain bool now. If the
        // user explicitly opts in via imp.conf the oversubscribe
        // check still gates them; this matches the previous
        // IMP_MXFP4_FP16_FALLBACK=1 semantics. The legacy
        // =force escape hatch is obsolete.
        bool allow_force = false;
        if (oversubscribe && !allow_force) {
            IMP_LOG_ERROR(
                "MXFP4 FP16 fallback would oversubscribe VRAM "
                "(need %.1f GiB + %.1f GiB runtime headroom, %.1f GiB free). "
                "Model is too large for this GPU with the FP16 decode "
                "fallback. Use a smaller quant or a smaller model. "
                "Set IMP_MXFP4_FP16_FALLBACK=force to attempt anyway "
                "(may IMA at first decode forward).",
                fp16_total / (1024.0 * 1024.0 * 1024.0),
                kRuntimeHeadroom / (1024.0 * 1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0 * 1024.0));
            // Skip the alloc — wcache_->fp16 stays empty for these weights.
            // Downstream code will detect the missing entries and bail
            // with its own diagnostic instead of silently corrupting state.
            fp16_total = 0;
        } else if (oversubscribe) {
            IMP_LOG_WARN(
                "MXFP4 FP16 fallback: forcing oversubscribed alloc "
                "(IMP_MXFP4_FP16_FALLBACK=force, %.1f GiB > %.1f GiB free)",
                fp16_total / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    if (fp16_total > 0) {
        cudaError_t ae = cudaMalloc(&d_fp16_bulk, fp16_total);
        if (ae != cudaSuccess) {
            IMP_LOG_ERROR("MXFP4 FP16 bulk alloc failed: %s (%.1f MiB)", cudaGetErrorString(ae),
                          fp16_total / (1024.0 * 1024.0));
            d_fp16_bulk = nullptr;
        } else {
            // Track the bulk for shutdown cleanup. Each fp16 Tensor written
            // below points to a sub-range of this allocation, so we cannot
            // cudaFree the sub-pointers — only the bulk base.
            wcache_->fp16_bulk_data = d_fp16_bulk;
            wcache_->fp16_bulk_data_size = fp16_total;
        }
    }

    if (d_fp16_bulk) {
        size_t offset = 0;
        for (auto& [ptr, mw] : wcache_->cutlass_mxfp4) {
            if (wcache_->fp16.count(ptr))
                continue;
            // Honor the same pruning filter the fp16_total compute
            // pass used; otherwise the offset accounting drifts.
            if (pruned_policy && pruned_skip_ptrs.count(ptr))
                continue;
            size_t fp16_bytes = static_cast<size_t>(mw.N) * mw.K * sizeof(half);
            void* d_fp16 = static_cast<char*>(d_fp16_bulk) + offset;
            offset += fp16_bytes;

            // GPU-side dequant via dequant_mxfp4_to_fp16. The previous CPU
            // fallback indexed `(r*bpr+b)*17` + `blk[16]`, which assumed
            // GGUF interleaved 17-byte block layout — but weight_upload.cu
            // splits to [data(N*bpr*16) | scales(N*bpr)] before GPU upload,
            // so the CPU path read scale bytes from inside data and produced
            // garbage FP16. The GPU kernel below reads the split layout
            // correctly (data first, scales at offset N*K/2).
            dequant_mxfp4_to_fp16(ptr, mw.N, mw.K, d_fp16, stream);
            int64_t shape[2] = {mw.N, mw.K};
            wcache_->fp16[ptr] = Tensor(d_fp16, QType::F16, 2, shape, true);
        }
    }  // end if (d_fp16_bulk)

    if (fp16_total > 0) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess)
                IMP_LOG_ERROR("MXFP4 dequant kernel error: %s", cudaGetErrorString(e));
        }
        IMP_LOG_INFO("MXFP4 decode fallback: dequant → FP16 cache %.2f MiB",
                     fp16_total / (1024.0 * 1024.0));

        // Replace model weight tensor pointers with FP16 data.
        // This ensures ALL code paths (GEMV, direct gemm, etc.) see
        // valid FP16 data instead of raw MXFP4 blocks.
        auto replace_weight = [&](Tensor& w, QType& qt) {
            auto it = wcache_->fp16.find(w.data);
            if (it != wcache_->fp16.end() && qt == QType::MXFP4) {
                w = it->second;
                qt = QType::F16;
            }
        };
        for (int i = 0; i < cfg.n_layers; i++) {
            TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
            replace_weight(L.wq, L.wq.qtype);
            replace_weight(L.wk, L.wk.qtype);
            replace_weight(L.wv, L.wv.qtype);
            replace_weight(L.wo, L.wo.qtype);
            replace_weight(L.w_up, L.w_up.qtype);
            replace_weight(L.w_gate, L.w_gate.qtype);
            replace_weight(L.w_down, L.w_down.qtype);
            // GDN-specific weights (Qwen3.5)
            replace_weight(L.ssm_in, L.ssm_in.qtype);
            replace_weight(L.ssm_out, L.ssm_out.qtype);
            replace_weight(L.gdn_gate, L.gdn_gate.qtype);
            replace_weight(L.gdn_alpha, L.gdn_alpha.qtype);
            replace_weight(L.gdn_beta, L.gdn_beta.qtype);
        }
        replace_weight(const_cast<Model*>(model_)->out_proj_,
                       const_cast<Model*>(model_)->out_proj_.qtype);
        // Tok-embed table — when weight tying is on (Qwen3.5-4B / others), it
        // shares the same GPU storage as out_proj_, so its data pointer is
        // already a key in wcache_->fp16. Without this replace, the embedding
        // lookup reads the raw MXFP4 bytes as FP16 → garbage hidden state from
        // token 0 → garbage logits → token-0 spam output. Also harmless when
        // the embedding is FP16 (the lookup misses, qtype guard is satisfied
        // anyway).
        replace_weight(const_cast<Model*>(model_)->tok_emb_,
                       const_cast<Model*>(model_)->tok_emb_.qtype);
        IMP_LOG_INFO("MXFP4 → FP16: replaced %d weight tensor pointers",
                     (int)wcache_->fp16.size());
    }
}
}

}  // namespace imp

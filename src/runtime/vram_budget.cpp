#include "runtime/vram_budget.h"
#include "runtime/engine.h"  // EngineConfig full definition
#include "runtime/storage_planner.h"
#include "compute/gemm_cutlass_sm120.h"  // cutlass_nvfp4_sf_size (SfAtom padding)
#include "core/logging.h"
#include "memory/vram_query.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace imp {

NativeCacheDemand compute_native_cache_demand(const Model& model) {
    NativeCacheDemand d;
    const auto& mcfg = model.config();
    if (!mcfg.is_nvfp4_prequant)
        return d;

    // Mirrors phase 3b's slab sizing (pre_dequant_phase3_cutlass.cu): each
    // slab entry is align_up(cutlass_nvfp4_sf_size(N, K), 256); contiguous
    // MoE expert groups occupy align_up(ne × sf_size, 256) as ONE entry.
    // Wire shapes store the PACKED byte dim (K/2 for e2m1 pairs), so
    // logical K = 2 × shape[1]. The previous elems/16+256 heuristic was
    // NOT an upper bound under SfAtom padding (rows→128, K→64 elems).
    constexpr size_t kSfAlign = 256;
    auto align_up = [](size_t x, size_t a) { return (x + a - 1) / a * a; };

    auto add_dense = [&](const Tensor& w) {
        if (!w.data || w.ndim < 2)
            return;
        if (w.ndim >= 3) {
            // Packed 3D experts [ne, N, K/2] (GGUF / Gemma-4 loaders): one
            // contiguous group entry + the transient per-(layer,proj) copy.
            int ne = static_cast<int>(w.shape[0]);
            int N = static_cast<int>(w.shape[1]);
            int64_t K = static_cast<int64_t>(w.shape[2]) * 2;
            d.sf_bytes += align_up(static_cast<size_t>(ne) *
                                       cutlass_nvfp4_sf_size(N, static_cast<int>(K)),
                                   kSfAlign);
            size_t slab = static_cast<size_t>(ne) * N * (K / 2) +
                          static_cast<size_t>(ne) * N * (K / 16) +
                          static_cast<size_t>(ne) * sizeof(float);
            d.moe_slab_bytes = std::max(d.moe_slab_bytes, slab);
            return;
        }
        int N = static_cast<int>(w.shape[0]);
        int64_t K = static_cast<int64_t>(w.shape[1]) * 2;
        d.sf_bytes += align_up(cutlass_nvfp4_sf_size(N, static_cast<int>(K)), kSfAlign);
    };
    auto add_experts = [&](const std::vector<Tensor>& ws) {
        if (ws.empty() || !ws[0].data || ws[0].ndim < 2)
            return;
        int ne = static_cast<int>(ws.size());
        int N = static_cast<int>(ws[0].shape[0]);
        int64_t Kp = ws[0].shape[1];
        int64_t K = Kp * 2;
        d.sf_bytes += align_up(static_cast<size_t>(ne) *
                                   cutlass_nvfp4_sf_size(N, static_cast<int>(K)),
                               kSfAlign);
        // Transient contiguous-copy slab (phase 3-moe copy branch: packed +
        // micro-scales + tensor-scales, matching add_bytes there). The
        // zero-copy borrow branch needs ~none — this is the upper bound.
        size_t slab = static_cast<size_t>(ne) * N * Kp + static_cast<size_t>(ne) * N * (K / 16) +
                      static_cast<size_t>(ne) * sizeof(float);
        d.moe_slab_bytes = std::max(d.moe_slab_bytes, slab);
    };

    // The same tensor set phase 0b registers for the decode cache
    // (pre_dequant_phase0_nvfp4_loader.cu register_prequant) — including
    // ssm_in/ssm_out/gdn_gate, which the previous inline scan omitted
    // (under-count on GDN hybrids, i.e. Qwen3.6-35B itself).
    add_dense(model.output_proj());
    for (int i = 0; i < mcfg.n_layers; i++) {
        const auto& L = model.layer(i);
        add_dense(L.wq);
        add_dense(L.wk);
        add_dense(L.wv);
        add_dense(L.wo);
        add_dense(L.w_gate);
        add_dense(L.w_up);
        add_dense(L.w_down);
        add_dense(L.w_gate_shared);
        add_dense(L.w_up_shared);
        add_dense(L.w_down_shared);
        add_dense(L.ssm_in);
        add_dense(L.ssm_out);
        add_dense(L.gdn_gate);
        if (L.expert_gate_packed.data)
            add_dense(L.expert_gate_packed);
        else
            add_experts(L.expert_w_gate);
        if (L.expert_up_packed.data)
            add_dense(L.expert_up_packed);
        else
            add_experts(L.expert_w_up);
        if (L.expert_down_packed.data)
            add_dense(L.expert_down_packed);
        else
            add_experts(L.expert_w_down);
    }
    return d;
}

VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram, int swa_live_tokens, int n_swa_layers,
                               size_t mandatory_cache_prealloc) {
    VRAMBudget budget;
    const auto& mcfg = model.config();

    // --- 1. Classify model quantization ---
    auto qtype = model.layer(0).wq.qtype;
    bool sub_8bit = (qtype == QType::Q4_0 || qtype == QType::Q4_K || qtype == QType::Q5_0 ||
                     qtype == QType::Q5_K || qtype == QType::Q3_K || qtype == QType::Q2_K ||
                     qtype == QType::Q4_1 || qtype == QType::Q5_1);

    // --- 2. Choose strategy ---
    if (config.use_nvfp4_decode == 0) {
        budget.strategy = VRAMBudget::FP16_ONLY;
    } else if (sub_8bit) {
        budget.strategy = VRAMBudget::NVFP4_DECODE_ONLY;
    } else {
        budget.strategy = VRAMBudget::FP8_PREFILL_NVFP4_DECODE;
    }

    // --- 3. Compute available VRAM ---
    // Feature-aware reserve instead of flat 1 GiB.
    budget.reserve_bytes = 256ULL * 1024 * 1024;  // base: cuBLAS + driver
    if (config.use_cuda_graphs)
        budget.reserve_bytes += 256ULL * 1024 * 1024;
    if (config.use_green_contexts)
        budget.reserve_bytes += 128ULL * 1024 * 1024;
    if (config.use_fp8_prefill)
        budget.reserve_bytes += 128ULL * 1024 * 1024;
    budget.reserve_bytes = std::max(budget.reserve_bytes, static_cast<size_t>(512ULL * 1024 * 1024));
    // At least 10% of total VRAM as headroom — but skipped for second-pass
    // NVFP4 (mode 2). In that mode the NVFP4 MoE caching path already enforced
    // its own ~1 GiB reserve before this runs; piling another 10% on top here
    // pushed `available` to 0 on Nemotron-H NVFP4 (32 GiB GPU → 3.2 GiB extra
    // reserve > what the MoE pass left free), which collapsed the KV cache to
    // the 16-block floor and left long-prompt requests stuck in pending_.
    size_t total_vram = 0;
    {
        size_t f;
        vram_budget_mem_get_info(&f, &total_vram);
    }
    // Budget-planner knobs (imp.conf [vram]): clamp once here so direct
    // EngineConfig embedders (tests, C-API) get the same envelope.
    const float kv_fraction = std::clamp(config.kv_fraction, 0.05f, 0.95f);
    const int reserve_floor_pct = std::clamp(config.vram_reserve_floor_pct, 0, 50);
    if (config.use_nvfp4_decode != 2) {
        budget.reserve_bytes =
            std::max(budget.reserve_bytes, vram_reserve_floor(total_vram, reserve_floor_pct));
    }

    // Estimate SSM footprint
    size_t ssm_footprint = 0;
    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model.layer(i).ssm_in.data != nullptr)
                n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_footprint = static_cast<size_t>(n_ssm) * config.max_batch_size *
                            (static_cast<unsigned long>(conv_ch) * std::max(mcfg.ssm_conv_kernel - 1, 0) *
                                 sizeof(float) +
                             static_cast<size_t>(n_heads) * hd_ssm * mcfg.ssm_state_size *
                                 dtype_size(config.ssm_state_dtype));
        }
    }

    size_t available = free_vram;
    size_t overhead = budget.reserve_bytes + ssm_footprint;
    available = (available > overhead) ? (available - overhead) : 0;

    // --- 4. Compute KV cache per-block cost ---
    int bs = config.kv_block_size > 0 ? config.kv_block_size : kKVBlockSize;
    size_t single_block_bytes;
    // Packed 4-bit KV dtypes: 2 elements per byte (FP4 nibbles or INT4 packed).
    // NVFP4 was historically missing from this OR-chain — fell through to the
    // dtype_size() fallback which returns 0 for QType::NVFP4, silently zeroing
    // out NVFP4's KV-cache budget contribution. Pre-existing pre-MXFP4-KV; the
    // Slice 2 spec reviewer flagged it during the MXFP4-KV scope review.
    if (config.kv_cache_dtype == QType::INT4 || config.kv_cache_dtype == QType::NVFP4 ||
        config.kv_cache_dtype == QType::MXFP4_KV) {
        single_block_bytes = static_cast<size_t>(bs) * mcfg.n_kv_heads * head_dim / 2;
    } else {
        single_block_bytes = static_cast<size_t>(bs) * mcfg.n_kv_heads * head_dim *
                             dtype_size(config.kv_cache_dtype);
    }
    // Per-token scale overhead folded into the per-layer block bytes.
    size_t scale_per_block = 0;
    if (config.kv_cache_dtype == QType::INT8 || config.kv_cache_dtype == QType::INT4) {
        scale_per_block = static_cast<size_t>(bs) * mcfg.n_kv_heads * sizeof(half);
    } else if (config.kv_cache_dtype == QType::NVFP4 || config.kv_cache_dtype == QType::MXFP4_KV) {
        scale_per_block = static_cast<size_t>(bs) * mcfg.n_kv_heads * (head_dim / 16);
    }
    const size_t per_layer_block_bytes = (single_block_bytes + scale_per_block) * 2;  // K+V

    // SWA-aware sizing (kv_cache.swa_sizing): sliding-window layers hold a
    // fixed live span (window + slack + burst/chunk peak) per sequence slot —
    // charge them batch-shaped up front, exactly like the SSM state slabs,
    // and let only the global layers scale with context below.
    int n_global_layers = n_kv_layers;
    if (swa_live_tokens > 0 && n_swa_layers > 0 && n_swa_layers <= n_kv_layers) {
        n_global_layers = n_kv_layers - n_swa_layers;
        int swa_blocks_per_seq = (swa_live_tokens + bs - 1) / bs + 1;
        int batch = config.max_batch_size > 0 ? config.max_batch_size : 1;
        budget.swa_max_blocks = swa_blocks_per_seq * batch;
        size_t swa_footprint = static_cast<size_t>(budget.swa_max_blocks) * per_layer_block_bytes *
                               static_cast<size_t>(n_swa_layers);
        available = (available > swa_footprint) ? (available - swa_footprint) : 0;
        IMP_LOG_INFO("VRAM budget: SWA group %d blocks × %d layers = %.1f MiB "
                     "(live span %d tokens, %d global layers keep full context)",
                     budget.swa_max_blocks, n_swa_layers, swa_footprint / (1024.0 * 1024.0),
                     swa_live_tokens, n_global_layers);
    }

    // K+V (2x) per GLOBAL layer — SWA layers were charged batch-shaped above.
    size_t per_block_total = per_layer_block_bytes * n_global_layers;

    int blocks_per_seq = (config.max_seq_len + bs - 1) / bs;
    int needed_blocks = blocks_per_seq * config.max_batch_size;

    // --- 5. Estimate NVFP4-eligible weight cache size ---
    // Eligibility policy is the shared nvfp4_beneficial() in core/qtype.h —
    // the same predicate the pre-dequant phases use, so the estimate and the
    // actual cache build can't drift.
    const bool decode_all = config.nvfp4_decode_all;
    size_t nvfp4_elems = 0;
    auto count_nvfp4 = [&](const Tensor& w, QType qt) {
        if (!w.data || !nvfp4_beneficial(qt, decode_all))
            return;
        if (w.shape[1] % 16 != 0)
            return;
        nvfp4_elems += static_cast<size_t>(w.shape[0]) * w.shape[1];
    };

    count_nvfp4(model.output_proj(), model.out_proj_.qtype);
    for (int i = 0; i < mcfg.n_layers; i++) {
        const auto& L = model.layer(i);
        count_nvfp4(L.wq, L.wq.qtype);
        count_nvfp4(L.wk, L.wk.qtype);
        count_nvfp4(L.wv, L.wv.qtype);
        count_nvfp4(L.wo, L.wo.qtype);
        count_nvfp4(L.w_gate, L.w_gate.qtype);
        count_nvfp4(L.w_up, L.w_up.qtype);
        count_nvfp4(L.w_down, L.w_down.qtype);
        count_nvfp4(L.ssm_in, L.ssm_in.qtype);
        count_nvfp4(L.ssm_out, L.ssm_out.qtype);
        count_nvfp4(L.w_gate_shared, L.w_gate_shared.qtype);
        count_nvfp4(L.w_up_shared, L.w_up_shared.qtype);
        count_nvfp4(L.w_down_shared, L.w_down_shared.qtype);
    }

    size_t nvfp4_estimate = nvfp4_elems / 2 + nvfp4_elems / 16;
    size_t cutlass_sf_estimate = nvfp4_elems / 16;

    // Cross-check the heuristic estimate against the StoragePlanner's
    // projected total (source-qtype-aware). Diverging numbers indicate the
    // heuristic missed a tier (e.g. Q4_K weights that the planner routes to
    // the FP16 cache but the heuristic treats as 0 because nvfp4_beneficial
    // is false). The heuristic drives allocation; the plan built here is
    // UNCONSTRAINED (no vram_budget_bytes hint, so no downgrade loop and
    // plan.failed can never be set) — it measures ideal per-tensor demand,
    // which the GGUF branch below folds into the weight-cache reserve (#875).
    // The budget-constrained plan the phases actually consult is built
    // separately in QuantPipeline::build (executor_pre_dequant.cu).
    {
        PlanHints hints;
        hints.prefer_nvfp4_decode = (config.use_nvfp4_decode > 0);
        hints.dual_path_attn_fp8_ffn_nvfp4 = config.dual_path_quant;
        hints.prefer_fp8 = config.use_fp8_prefill;
        StoragePlan plan = plan_storage(model, mcfg, hints);
        IMP_LOG_INFO(
            "VRAM budget: planner projects %.1f MiB (%zu entries), heuristic "
            "nvfp4=%.1f MiB cutlass_sf=%.1f MiB.",
            plan.projected_vram_bytes / (1024.0 * 1024.0), plan.entries.size(),
            nvfp4_estimate / (1024.0 * 1024.0), cutlass_sf_estimate / (1024.0 * 1024.0));

        // Native NVFP4 checkpoints: the GGUF-oriented count above sees the
        // source as non-beneficial and yields 0, but phase 3 still needs
        // post-KV headroom for (a) the persistent CUTLASS SfAtom SF cache
        // (~elems/16 across ALL NVFP4 weights incl. experts — model qtypes
        // are still wire dtypes here, so sum it from the PLAN, which is
        // source-aware), (b) the transient per-(layer,proj) MoE contiguous-
        // copy slab, and (c) the free-VRAM reserves phases 0-3 re-derive for
        // themselves (max(total/10, 256 MiB)) plus executor workspaces that
        // allocate between KV init and phase 3. With the estimate at 0 the
        // KV clamp below ate ALL post-weight VRAM at auto max_batch_size and
        // both caches got zero budget — Qwen3-Coder-30B-FP4 @45k served
        // 31.8 tok/s / 1265 ms TTFT instead of 314.7 / 106 (both caches are
        // free-VRAM-driven at phase 3, so reserving room here is the fix).
        if (mcfg.is_nvfp4_prequant) {
            // Both the count above AND the plan classify by source qtype,
            // which at budget time is still the wire dtype (NVFP4-ness lives
            // in the nvfp4_scratch_ sidecars until phase 0 promotes it) — so
            // compute_native_cache_demand scans the projection tensors
            // directly, no qtype filter: on an NVFP4-prequant checkpoint
            // these are exactly the quantized set.
            NativeCacheDemand demand = compute_native_cache_demand(model);
            // Floors are only handed to phase 3 when the balloon physically
            // guaranteed the bytes — an unbacked floor on a genuinely tight
            // card would over-commit the SF slab cudaMalloc. Phase 3b (SF)
            // runs before phase 3-moe, so an sf-only balloon still floors SF.
            budget.mandatory_sf_bytes =
                (mandatory_cache_prealloc >= demand.sf_bytes) ? demand.sf_bytes : 0;
            budget.mandatory_moe_bytes =
                (mandatory_cache_prealloc >= demand.total()) ? demand.moe_slab_bytes : 0;
            size_t phase3_reserve =
                vram_reserve_floor(total_vram, reserve_floor_pct) + 1024ULL * 1024 * 1024;
            // Bytes already physically held by the Engine's balloon are
            // excluded from free_vram — charge KV only for the UNCOVERED
            // remainder (normally 0) so the demand isn't counted twice.
            size_t uncovered = (demand.total() > mandatory_cache_prealloc)
                                   ? demand.total() - mandatory_cache_prealloc
                                   : 0;
            size_t native_need = uncovered + phase3_reserve;
            if (native_need > cutlass_sf_estimate) {
                IMP_LOG_INFO("VRAM budget: native-NVFP4 weight-cache reserve %.1f MiB "
                             "(sf=%.1f, moe_slab=%.1f, prealloc-covered=%.1f, "
                             "phase3+workspaces=%.1f)",
                             native_need / (1024.0 * 1024.0), demand.sf_bytes / (1024.0 * 1024.0),
                             demand.moe_slab_bytes / (1024.0 * 1024.0),
                             std::min(demand.total(), mandatory_cache_prealloc) / (1024.0 * 1024.0),
                             phase3_reserve / (1024.0 * 1024.0));
                cutlass_sf_estimate = native_need;
            }
        } else if (per_block_total > 0) {
            // GGUF sources the heuristic can't see: Q4_K/Q3_K weights are not
            // nvfp4_beneficial (estimate 0) but the planner routes them to the
            // FP16 cache — exactly the divergence the log above warns about.
            // Without reserving that demand the KV backstop below eats ALL
            // post-weight VRAM and phases 1/3 build 0 tensors: Ornith-35B
            // Q4_K_M served 11 tok/s via on-the-fly dequant with a 159k-token
            // KV pool nobody asked for (#874 follow-up). Same failure class as
            // the native-NVFP4 branch above, same fix: fold the real demand
            // into cutlass_sf_estimate so both the strategy math and the
            // backstop see it.
            //
            // Gated on a 2x divergence so heuristic-covered models (Q6_K/Q8
            // dense — the tuned bench configs) keep their KV pools unchanged.
            size_t heuristic = nvfp4_estimate + cutlass_sf_estimate;
            if (plan.projected_vram_bytes > 2 * heuristic) {
                size_t want =
                    plan.projected_vram_bytes + vram_reserve_floor(total_vram, reserve_floor_pct);
                // Never squeeze the pool below one full max_seq_len sequence
                // (or the min-KV floor, whichever is larger) — long-context
                // must stay servable; concurrency is bounded by scheduler
                // admission, not by pool size.
                int floor_tok = config.min_kv_tokens > 0
                                    ? config.min_kv_tokens
                                    : std::min(16384, config.max_seq_len * 4);
                int guarantee_blocks = std::max(blocks_per_seq, (floor_tok + bs - 1) / bs);
                size_t guarantee_bytes = static_cast<size_t>(guarantee_blocks) * per_block_total;
                size_t cap = (available > guarantee_bytes) ? available - guarantee_bytes : 0;
                size_t reserve = std::min(want, cap);
                if (reserve > heuristic) {
                    IMP_LOG_INFO(
                        "VRAM budget: planner-driven weight-cache reserve %.1f MiB "
                        "(plan=%.1f MiB, heuristic=%.1f MiB, kv guarantee=%d blocks)",
                        reserve / (1024.0 * 1024.0),
                        plan.projected_vram_bytes / (1024.0 * 1024.0),
                        heuristic / (1024.0 * 1024.0), guarantee_blocks);
                    cutlass_sf_estimate = reserve - nvfp4_estimate;
                }
            }
        }
    }

    // --- 6. Allocate based on strategy ---
    switch (budget.strategy) {
        case VRAMBudget::NVFP4_DECODE_ONLY: {
            budget.fp8_cache_bytes = 0;
            budget.nvfp4_cache_bytes = nvfp4_estimate;
            size_t weight_total = nvfp4_estimate + cutlass_sf_estimate;
            size_t kv_available = (available > weight_total) ? (available - weight_total) : 0;
            budget.kv_cache_bytes = static_cast<size_t>(kv_available * kv_fraction);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            budget.nvfp4_second_pass = false;
            break;
        }
        case VRAMBudget::FP8_PREFILL_NVFP4_DECODE: {
            // NVFP4 decode cache is critical for performance — ensure it fits first.
            // FP8 prefill cache is nice-to-have but not essential (fallback: dequant on-the-fly).
            budget.nvfp4_cache_bytes = nvfp4_estimate;
            // KV target = kv_fraction (default 0.8) of available for BOTH
            // modes. Mode 2 (native-NVFP4 second pass) previously used 0.1 and
            // skipped the needed_blocks floor — intended to leave room for an
            // FP8 prefill cache. But on sm_120 FP8 prefill is unavailable
            // (native NVFP4 uses the CUTLASS NVFP4 GEMM, no FP8 weight cache is
            // built), so the 90% reserve was dead VRAM: an NVFP4 SafeTensors
            // model starved its KV pool to ~24K tokens while 16 GB sat free —
            // the long-context/agentic default was effectively broken.
            // target_blocks (below) still clamps mode 2 to needed_blocks, so
            // the fraction only lifts KV up to the per-sequence need and leaves
            // the remainder for FP8 (computed post-clamp) when it IS enabled —
            // no regression for fp8-prefill configs.
            budget.kv_cache_bytes = static_cast<size_t>(available * kv_fraction);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            budget.kv_max_blocks = std::max(budget.kv_max_blocks, needed_blocks);
            // FP8 budget is computed below — after the kv_max_blocks clamp /
            // min_kv_blocks enforcement — so it reflects the FINAL KV size.
            budget.nvfp4_second_pass = (config.use_nvfp4_decode == 2);
            break;
        }
        case VRAMBudget::FP16_ONLY: {
            budget.fp8_cache_bytes = 0;
            budget.nvfp4_cache_bytes = 0;
            budget.kv_cache_bytes = static_cast<size_t>(available * kv_fraction);
            budget.kv_max_blocks = (per_block_total > 0)
                                       ? static_cast<int>(budget.kv_cache_bytes / per_block_total)
                                       : needed_blocks;
            budget.nvfp4_second_pass = false;
            break;
        }
    }
    budget.kv_max_blocks = std::max(budget.kv_max_blocks, 16);

    int target_blocks = (config.use_nvfp4_decode == 2 || budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY)
                            ? needed_blocks
                            : needed_blocks * 2;
    budget.kv_max_blocks = std::min(budget.kv_max_blocks, target_blocks);
    budget.kv_max_blocks = std::max(budget.kv_max_blocks, 16);

    // Hard backstop: never request more KV blocks than the post-weight VRAM can
    // physically hold. The strategy above can raise kv_max_blocks toward
    // needed_blocks (= max_batch_size × max_seq_len); with the VRAM-aware auto
    // max_batch_size (#736) that product blows past real headroom on a
    // small-weight / large-card config — dense Q8 on a 32 GB card sizes batch
    // auto→25, so 25 × 16384-token slots = 25600 blocks (57.6 GB) and the KV
    // cudaMalloc OOMs at context creation. The KV pool is paged with scheduler
    // admission control (Scheduler::can_allocate), so clamping the pool to what
    // fits only bounds concurrency under load — no single sequence is
    // under-served. Subtract the NVFP4 decode + CUTLASS-SF caches, which share
    // the same post-weight headroom (FP8 is computed from the remainder below).
    if (per_block_total > 0) {
        size_t weight_caches = nvfp4_estimate + cutlass_sf_estimate;
        size_t kv_room = (available > weight_caches) ? (available - weight_caches) : available;
        int max_fit_blocks = std::max(static_cast<int>(kv_room / per_block_total), 16);
        if (budget.kv_max_blocks > max_fit_blocks) {
            IMP_LOG_INFO("VRAM budget: KV clamped %d → %d blocks to fit post-weight VRAM "
                         "(paged pool; concurrency bounded by scheduler admission)",
                         budget.kv_max_blocks, max_fit_blocks);
            budget.kv_max_blocks = max_fit_blocks;
        }
    }

    // Enforce minimum KV token budget. Auto default: 16K tokens or 4x max_seq_len,
    // whichever is smaller (capped to not exceed what VRAM can physically hold).
    int min_kv_tok = config.min_kv_tokens;
    bool user_requested_min = (min_kv_tok > 0);
    if (!user_requested_min) {
        min_kv_tok = std::min(16384, config.max_seq_len * 4);
    }
    int min_kv_blocks = (min_kv_tok + bs - 1) / bs;
    const int min_kv_blocks_wanted = min_kv_blocks;  // pre-cap: what the floor asked for
    int max_affordable = (per_block_total > 0) ? static_cast<int>(available / per_block_total)
                                               : budget.kv_max_blocks;
    // Defensive cap for auto mode (leaves room for weight caches). When the
    // user explicitly sets min_kv_tokens, respect their request up to the
    // physical max_affordable — they're opting into a tighter weight-cache
    // budget in exchange for more context.
    int cap = user_requested_min ? max_affordable : static_cast<int>(max_affordable * kv_fraction);
    min_kv_blocks = std::min(min_kv_blocks, cap);
    if (budget.kv_max_blocks < min_kv_blocks) {
        IMP_LOG_INFO("VRAM budget: raising KV from %d to %d blocks (min_kv_tokens=%d)", budget.kv_max_blocks,
                     min_kv_blocks, min_kv_tok);
        budget.kv_max_blocks = min_kv_blocks;
        budget.kv_cache_bytes = static_cast<size_t>(min_kv_blocks) * per_block_total;
    }
    // Loud when the pool STILL can't hold the KV floor after every clamp: any
    // request longer than the pool is rejected/cancelled at admission while
    // /v1/models keeps advertising max_seq_len. Observed: --max-batch 64 on
    // Qwen3.6-35B-A3B-NVFP4 (32 GB card) collapsed KV to 16 blocks = 512
    // tokens — every longer prompt came back finish_reason=cancelled with no
    // hint why. Batch-scaled workspaces are the usual culprit.
    if (budget.kv_max_blocks < min_kv_blocks_wanted) {
        IMP_LOG_WARN(
            "VRAM budget: KV pool holds only %d tokens (< %d-token floor; requested context "
            "%d). Longer requests will be rejected or cancelled at admission. Lower "
            "max_batch_size (batch-scaled workspaces are the usual VRAM consumer) or "
            "runtime.max_seq_len, or use a smaller model/quant.",
            budget.kv_max_blocks * bs, min_kv_tok, config.max_seq_len);
    }

    // FP8 prefill: use remaining VRAM after NVFP4 decode + the *final* KV size.
    // Computing this earlier (against the unclamped kv_max_blocks) silently
    // zeroed FP8 in mode 1 because the 0.8 kv_fraction filled the budget on
    // paper, even though `target_blocks` clamped the actual KV allocation
    // back down. The post-clamp computation lets mode 1 (additive) populate
    // both caches when VRAM allows — Qwen3-14B Q6_K mode 1 default flags
    // previously cached fp8=0 tensors and paid ~28 % prefill for it.
    if (budget.strategy == VRAMBudget::FP8_PREFILL_NVFP4_DECODE && config.use_fp8_prefill) {
        size_t kv_actual = static_cast<size_t>(budget.kv_max_blocks) * per_block_total;
        size_t nvfp4_actual = nvfp4_estimate + cutlass_sf_estimate;
        size_t remaining_for_fp8 = (available > kv_actual + nvfp4_actual)
                                       ? (available - kv_actual - nvfp4_actual)
                                       : 0;
        // For native NVFP4 (nvfp4_elems=0 because already FP4), compute FP8
        // cache size from the actual weight sizes. Each dense weight needs
        // N×K×1 bytes FP8. This enables FP8 prefill for NVFP4 SafeTensors.
        size_t fp8_size_cap = nvfp4_elems;
        if (fp8_size_cap == 0 && mcfg.is_nvfp4_prequant)
            fp8_size_cap = remaining_for_fp8;
        budget.fp8_cache_bytes = std::min(fp8_size_cap, remaining_for_fp8);
    }
    // With FP8 prefill resolved off (the sm_120 auto default), the strategy's
    // KV math above still applies but no FP8 cache will ever be built (Phase 2
    // is gated on use_fp8) — report 0 instead of a phantom reservation.

    const char* strat_name = (budget.strategy == VRAMBudget::FP8_PREFILL_NVFP4_DECODE)
                                 ? (config.use_fp8_prefill ? "FP8_PREFILL_NVFP4_DECODE"
                                                           : "NVFP4_DECODE (fp8 prefill off)")
                             : (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) ? "NVFP4_DECODE_ONLY"
                                                                                  : "FP16_ONLY";
    IMP_LOG_INFO(
        "VRAM budget: strategy=%s, available=%.1f MiB, "
        "kv=%d blocks (%.1f MiB), nvfp4=%.1f MiB, fp8=%.1f MiB, "
        "reserve=%.0f MiB, second_pass=%s",
        strat_name, available / (1024.0 * 1024.0), budget.kv_max_blocks,
        (per_block_total > 0 ? budget.kv_max_blocks * per_block_total : 0) / (1024.0 * 1024.0),
        nvfp4_estimate / (1024.0 * 1024.0), budget.fp8_cache_bytes / (1024.0 * 1024.0),
        budget.reserve_bytes / (1024.0 * 1024.0), budget.nvfp4_second_pass ? "yes" : "no");

    return budget;
}

}  // namespace imp

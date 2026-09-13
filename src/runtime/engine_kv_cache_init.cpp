// Engine init phase: paged KV cache allocation. Declaration in engine.h.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "memory/vram_query.h"
#include "memory/library_reserve_cache.h"
#include "memory/plan.h"
#include "memory/ssm_state_size.h"
#include "runtime/plan_shadow.h"
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "core/logging.h"
#include "compute/mmq_q8_imma.h"  // mmq_q8_imma_set_plane_budget
#include "compute/sampling.h"     // SAMPLE_SCRATCH_BYTES (constrained-pipeline region)
#include "runtime/serving_metadata_layout.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace imp {

// Stable model identity hash: FNV-1a over config scalars + weight bytes
// (LM head, embeddings, layer-0/mid Q proj) - distinguishes same-shape fine-tunes.
// Gates the persisted prefix cache only (cold path, twice per process).
uint64_t Engine::model_fingerprint_() const {
    auto fnv = [](uint64_t h, const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    };
    uint64_t h = 0xcbf29ce484222325ULL;
    const auto& c = model_->config();
    const uint32_t ids[] = {
        static_cast<uint32_t>(std::to_underlying(c.arch)), static_cast<uint32_t>(c.n_layers),
        static_cast<uint32_t>(c.n_heads), static_cast<uint32_t>(c.n_kv_heads),
        static_cast<uint32_t>(c.d_model), static_cast<uint32_t>(c.d_ff),
        static_cast<uint32_t>(c.vocab_size), static_cast<uint32_t>(c.head_dim),
        static_cast<uint32_t>(c.n_experts), static_cast<uint32_t>(c.n_experts_active),
        static_cast<uint32_t>(c.is_nvfp4_prequant), static_cast<uint32_t>(c.is_mxfp4_prequant)};
    h = fnv(h, ids, sizeof(ids));
    h = fnv(h, &c.rope_theta, sizeof(c.rope_theta));

    auto sample = [&](const Tensor& t) {
        if (!t.data)
            return;
        size_t n = std::min<size_t>(t.nbytes(), 512);
        if (n == 0)
            return;
        std::vector<uint8_t> buf(n);
        if (t.on_device) {
            if (cudaMemcpy(buf.data(), t.data, n, cudaMemcpyDeviceToHost) != cudaSuccess)
                return;
        } else {
            std::memcpy(buf.data(), t.data, n);
        }
        h = fnv(h, buf.data(), n);
    };
    sample(model_->output_proj());
    sample(model_->token_embedding());
    sample(model_->layer(0).wq);
    if (c.n_layers > 1)
        sample(model_->layer(c.n_layers / 2).wq);
    return h;
}

bool Engine::init_kv_cache() {
    const auto& mcfg = model_->config();
    int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);

    int n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model_->layer(i).wq.data != nullptr && model_->layer(i).gdn_gate.data == nullptr)
            kv_layer_map[i] = n_attn_layers++;
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;
        for (int i = 0; i < mcfg.n_layers; i++)
            kv_layer_map[i] = i;
    }
    int n_kv_layers = n_attn_layers;
    IMP_LOG_INFO("KV cache layers: %d attention out of %d total", n_kv_layers, mcfg.n_layers);

    // Resolved in init_resolve_kv_block_size_() before the executor sizes its
    // workspaces - anything sized there needs the real value, not kKVBlockSize.
    // Fallback keeps this function correct if that resolver was skipped.
    if (config_.kv_block_size <= 0)
        config_.kv_block_size = kKVBlockSize;
    const int kv_bs = config_.kv_block_size;
    int blocks_per_seq = (config_.max_seq_len + kv_bs - 1) / kv_bs;

    // SWA-aware KV sizing gate (kv_cache.swa_sizing): sliding-window layers get
    // a small dedicated block group instead of full-length KV. Resolved before
    // the VRAM budget so it charges window-cost, not context-cost.
    swa_sizing_active_ = false;
    swa_window_max_ = 0;
    int n_swa_layers = 0;
    const SwaSizingMode swa_mode = runtime_config_.kv_cache.swa_sizing_mode();
    if (swa_mode != SwaSizingMode::Off) {
        const auto& prof = model_->profile();
        for (int i = 0; i < mcfg.n_layers; i++) {
            if (kv_layer_map[i] < 0)
                continue;  // non-attention layer
            int w = layer_swa_window(mcfg, prof, i);
            if (w > 0) {
                n_swa_layers++;
                swa_window_max_ = std::max(swa_window_max_, w);
            }
        }
        const char* off_reason = nullptr;
        if (n_swa_layers == 0)
            off_reason = "model has no sliding-window layers";
        else if (n_swa_layers == n_kv_layers && swa_window_max_ >= config_.max_seq_len)
            off_reason = "window >= max_seq_len (nothing to save)";
        else if (config_.kv_cache_dtype == QType::INT8 || config_.kv_cache_dtype == QType::INT4)
            off_reason = "INT8/INT4 KV lacks the per-layer cache path";
        else if (prof.is_ssm || prof.is_gdn)
            off_reason = "hybrid recurrent model (conservative)";
        else if (mcfg.is_mla())
            off_reason = "MLA attention";
        else if (config_.streaming_kv_enabled)
            off_reason = "StreamingLLM is enabled";
        else if (config_.use_green_contexts)
            off_reason = "green contexts (cross-stream block reuse unordered)";
        else if (runtime_config_.runtime.deterministic)
            off_reason = "deterministic mode (unbounded graph loop would be burst-chunked)";
        else if (swa_mode == SwaSizingMode::Auto && config_.use_prefix_caching &&
                 runtime_config_.kv_cache.swa_snapshot_mb <= 0)
            off_reason = "auto mode yields to prefix caching (freed window blocks cannot back "
                         "prefix reuse; set kv_cache.swa_snapshot_mb to combine, or "
                         "kv_cache.swa_sizing=on to force the KV savings)";
        if (off_reason) {
            IMP_LOG_INFO("kv_cache.swa_sizing=%s ignored: %s",
                         runtime_config_.kv_cache.swa_sizing.c_str(), off_reason);
            swa_window_max_ = 0;
            n_swa_layers = 0;
        } else {
            swa_sizing_active_ = true;
            // Slack must cover the deepest speculative rollback (verify
            // chunks roll back rejected drafts) plus the partial boundary
            // block. Sized from the spec config so the assert can't trip.
            const auto& sc = runtime_config_.speculative;
            int spec_depth = std::max({sc.k, sc.suffix_k_max, sc.mtp_k + 1, kJumpRowsCap});
            swa_slack_tokens_ = std::max(2 * kv_bs, spec_depth + kv_bs);
            // Longest on-device burst span (graph decode loop) plus the
            // largest prefill chunk the live window must ride through.
            int chunk_peak = config_.prefill_chunk_size > 0 ? config_.prefill_chunk_size : 2048;
            int burst_peak = runtime_config_.runtime.decode_burst > 0
                                 ? runtime_config_.runtime.decode_burst
                                 : 512;
            swa_burst_cap_tokens_ = std::max(chunk_peak, burst_peak);
            // Prefix caching cannot reuse freed window blocks. With a SWA
            // snapshot budget the two coexist (window restored at reuse
            // boundary); otherwise honor the forced opt-in by disabling it.
            if (config_.use_prefix_caching && runtime_config_.kv_cache.swa_snapshot_mb <= 0) {
                config_.use_prefix_caching = false;
                IMP_LOG_INFO("kv_cache.swa_sizing=on: prefix caching disabled (freed window "
                             "blocks cannot back prefix reuse; set kv_cache.swa_snapshot_mb "
                             "to combine)");
            }
            // StreamingLLM auto-enable frees middle blocks of the GLOBAL
            // table, redundant and conflicting here.
            config_.streaming_kv_auto = false;
        }
    }
    const int swa_live_tokens =
        swa_sizing_active_ ? swa_window_max_ + swa_slack_tokens_ + swa_burst_cap_tokens_ : 0;

    // VRAM budget: KV charges the full measured cache demand, phase 3 floors at
    // it (no held-back reserve, AUDIT B62). library_reserve_mb caches the first
    // forward's actual claim (stable per model/quant/library); explicit config wins.
    if (config_.library_reserve_mb < 0 && runtime_config_.vram.library_reserve_cache != "off") {
        const std::string path = runtime_config_.vram.library_reserve_cache.empty()
                                     ? library_reserve_cache_default_path()
                                     : runtime_config_.vram.library_reserve_cache;
        LibraryReserveKey key;
        key.model_fingerprint = model_fingerprint_();
        key.nvfp4_decode_mode = config_.use_nvfp4_decode;
        key.fp8_prefill = config_.use_fp8_prefill;
        cudaRuntimeGetVersion(&key.cuda_runtime_version);
        library_reserve_key_ = key;
        library_reserve_cache_path_ = path;
        bool remembered_found = false;
        const size_t remembered = library_reserve_cache_load(path, key, &remembered_found);
        if (remembered_found) {
            // A recorded ZERO reserve is valid (not "missing") - models whose
            // first forward claims nothing (AUDIT B70).
            config_.library_reserve_mb = static_cast<int>(remembered >> 20);
            IMP_LOG_INFO("library reserve: %d MiB from the measurement cache (%s) — the default "
                         "constant is %zu MiB",
                         config_.library_reserve_mb, path.c_str(),
                         kMeasuredLibraryReserveBytes >> 20);
        } else if (!path.empty()) {
            // No entry: log the constant charge HERE, before pools are sized -
            // reporting only after the first forward would be too late, the KV
            // pool would already be sized around an unwanted reserve.
            IMP_LOG_INFO("library reserve: no measurement for this model in %s — planning with the "
                         "%zu MiB constant. It is recorded after the first forward; mount that "
                         "path (or set vram.library_reserve_cache) to keep it across restarts.",
                         path.c_str(), kMeasuredLibraryReserveBytes >> 20);
        }
    }

    // Reserved pool slots: mc verify W-1 + batched-verify spares (recomputed after the clamp).
    int ssm_reserved_slots = spec_mc_reserved_slots_() + batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size);
    auto vram_budget = compute_vram_budget(*model_, config_, n_kv_layers, head_dim, effective_free_vram(),
                                           swa_live_tokens, n_swa_layers, &native_cache_demand(),
                                           ssm_reserved_slots, runtime_config_.gemm.q8_imma_enabled,
                                           runtime_config_.gemm.moe_imma_prefill);
    // Cap the IMMA prefill planes at what the pass just charged: they are taken
    // lazily on each Q8_0 weight's first prefill, so uncapped the cache would
    // take whatever KV leaves free, varying per start (#1899). Set before sizing.
    mmq_q8_imma_set_plane_budget(vram_budget.imma_plane_bytes);
    // KV block count comes from plan_memory(), not the live-free-derived pass
    // (B62/B65/B66). The residual clamp below can only shrink it further, never
    // grow it, so a plan wrong about the device cannot overcommit.
    int max_blocks = 0;
    {
        ShadowPlanProbe probe;
        probe.distributable_bytes = effective_free_vram();
        // Steady-state demand only: transient init headroom is free again by
        // pool-sizing time. Charging it starved optional caches to a 128-block
        // floor while GiBs sat idle (#1765).
        probe.weight_cache_demand =
            vram_budget.weight_cache_estimate_bytes > vram_budget.weight_cache_transient_bytes
                ? vram_budget.weight_cache_estimate_bytes - vram_budget.weight_cache_transient_bytes
                : 0;
        // The IMMA prefill planes are mandatory in the same sense: the plan
        // grants them, mmq_q8_imma is capped at the grant, and the pool must
        // not be sized over bytes the first prefill will take (#1899).
        probe.mandatory_cache_bytes = vram_budget.mandatory_sf_bytes + vram_budget.mandatory_moe_bytes +
                                      vram_budget.imma_plane_bytes;
        probe.ssm_state_bytes = vram_budget.ssm_footprint_bytes;
        // The recurrent snapshot store cudaMallocs server.recurrent_snapshot_mb
        // AFTER KV sizing, so charge it here: whole slots of one sequence's
        // state, never more than the budget.
        if (mcfg.ssm_inner_size > 0 && config_.use_prefix_caching &&
            runtime_config_.server.recurrent_snapshot_mb > 0 && vram_budget.ssm_footprint_bytes > 0) {
            const size_t slots =
                static_cast<size_t>(config_.max_batch_size) + static_cast<size_t>(std::max(0, ssm_reserved_slots));
            const size_t per_seq = slots > 0 ? vram_budget.ssm_footprint_bytes / slots : 0;
            const size_t budget = static_cast<size_t>(runtime_config_.server.recurrent_snapshot_mb) << 20;
            // Whole slots only, stopped at budget: device charge is
            // floor(budget/per_seq)*per_seq, not raw budget. Host tier
            // (server.recurrent_snapshot_host_mb) is pinned HOST memory, not VRAM.
            probe.recurrent_snapshot_bytes = per_seq > 0 ? (budget / per_seq) * per_seq : 0;
        }
        if (executor_) {
            probe.engine_persistent_bytes = executor_->workspace_estimate();
            probe.workspace_estimate_available = true;
        }
        probe.vision_tower_unmodelled = !config_.mmproj_path.empty();
        // Use config_.library_reserve_mb, NOT the runtime-config field: the
        // loader above writes the remembered measurement into the former only
        // (AUDIT B70).
        probe.library_reserve_bytes = config_.library_reserve_mb < 0
                                          ? kMeasuredLibraryReserveBytes
                                          : static_cast<size_t>(config_.library_reserve_mb) << 20;
        probe.n_kv_layers = n_kv_layers;
        probe.n_swa_layers = n_swa_layers;
        probe.swa_live_tokens = swa_live_tokens;
        probe.max_batch_size = config_.max_batch_size;
        probe.max_seq_len = config_.max_seq_len;
        probe.kv_block_size = kv_bs;
        probe.min_kv_tokens = config_.min_kv_tokens;
        probe.kv_block_bytes_per_layer =
            kv_block_bytes_per_layer(config_.kv_cache_dtype, kv_bs, mcfg.n_kv_heads, head_dim);

        PlanResult plan = plan_memory(shadow_plan_input(probe));
        IMP_LOG_INFO("%s", shadow_plan_report(probe, plan, vram_budget.kv_max_blocks).c_str());

        if (!plan.ok) {
            clamp_max_batch_to_plan_(probe, plan, ssm_reserved_slots, vram_budget.kv_max_blocks);
            ssm_reserved_slots = spec_mc_reserved_slots_() + batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size);
        }

        if (config_.kv_cache_max_blocks > 0) {
            max_blocks = config_.kv_cache_max_blocks;  // operator pin wins over both
        } else if (plan.ok) {
            max_blocks = plan.plan.kv.blocks;
            if (max_blocks != vram_budget.kv_max_blocks) {
                // Plan and live pass may legitimately differ (plan charges what
                // the live read cannot see) - logged so divergence isn't silent.
                // Live figure is pre-floor: kv_max_blocks may already be raised by min_kv_tokens (#1747).
                const int live_raw = vram_budget.kv_blocks_pre_floor > 0 ? vram_budget.kv_blocks_pre_floor
                                                                         : vram_budget.kv_max_blocks;
                if (live_raw != vram_budget.kv_max_blocks) {
                    IMP_LOG_INFO("KV blocks: plan %d (live pass sized %d, raised to %d by min_kv_tokens)",
                                 max_blocks, live_raw, vram_budget.kv_max_blocks);
                } else {
                    IMP_LOG_INFO("KV blocks: plan %d (live pass sized %d)", max_blocks, live_raw);
                }
            }
        } else {
            // Plan refuses this configuration: fails the load only when an
            // explicit --vram-budget is installed (check further down, D8).
            // Otherwise falls back to the live pass (best-effort, not a refusal).
            max_blocks = vram_budget.kv_max_blocks;
            IMP_LOG_WARN("KV blocks: the plan rejects this configuration — falling back to the "
                         "live-derived %d blocks. The report above says what it could not fit.",
                         max_blocks);
        }
    }

    // Sparse decode attention's key min/max pool (nkv*hd*4 bytes/block/layer)
    // is allocated AFTER this sizing and must be priced in: unpriced it is a
    // silent 6-12% overcommit that spills on WSL2/WDDM rather than failing (#1103).
    if (runtime_config_.attention.sparse_topk_tokens > 0) {
        const QType kvt = config_.kv_cache_dtype;
        const bool eligible = (kvt == QType::F16 || kvt == QType::FP8_E4M3) && !mcfg.is_mla() &&
                              !runtime_config_.speculative.token_recycling &&
                              config_.prefix_cache_path.empty() && mcfg.head_dim_per_layer.empty() &&
                              !swa_sizing_active_;
        if (eligible) {
            const size_t kv_per_block =
                static_cast<size_t>(n_kv_layers) *
                kv_block_bytes_per_layer(kvt, kv_bs, mcfg.n_kv_heads, head_dim);
            const size_t mm_per_block = static_cast<size_t>(n_kv_layers) * mcfg.n_kv_heads *
                                        static_cast<size_t>(head_dim) * 4;  // (min,max) halves
            if (config_.kv_cache_max_blocks > 0) {
                IMP_LOG_WARN("attention.sparse_topk_tokens: the key min/max pool adds %.1f MiB ON TOP "
                             "of the pinned kv_cache.max_blocks pool — pin with that headroom or "
                             "WSL2/WDDM spills silently",
                             static_cast<double>(mm_per_block) * max_blocks / (1024.0 * 1024.0));
            } else {
                // Auto-sized pools: charged post-plan like the BitDecoding
                // residual buffer - deflating the block count here broke the
                // admission guarantee. Pricing inside plan_memory is the open follow-up.
                IMP_LOG_INFO("sparse decode attention: key min/max pool adds %.1f MiB after the KV "
                             "sizing (%.2f%% of the K+V pool)",
                             static_cast<double>(mm_per_block) * max_blocks / (1024.0 * 1024.0),
                             kv_per_block > 0 ? 100.0 * mm_per_block / kv_per_block : 0.0);
            }
        }
    }

    // Weight caches build BEFORE the KV pool (A7 step 6.4): caches have bounded
    // demand and go first, KV (elastic) takes what's left - reversed order can
    // starve caches to 0 free and spill on WSL2/WDDM (AUDIT B23).
    // The profile gates below must stay with the caches: they set flags the
    // cache build and graph capture depend on (wcache_->use_fp8).
    // GDN detection
    {
        if (model_->profile().is_gdn) {
            if (config_.use_cuda_graphs) {
                IMP_LOG_INFO("GDN model: CUDA graphs enabled (recurrent state in-place)");
            } else {
                IMP_LOG_INFO(
                    "GDN model: CUDA graphs disabled (disabled earlier by caller or expert offload)");
            }
            // GDN recurrent state accumulates precision errors per token. FP8
            // E4M3's 3-bit mantissa amplifies these through the delta-rule scan,
            // causing degenerate output after ~50 tokens in multi-turn chat.
            if (config_.use_fp8_prefill) {
                if (config_.dual_path_quant) {
                    IMP_LOG_WARN(
                        "GDN + dual-path: attention weights forced to FP16 (not FP8) — "
                        "recurrent state needs FP16 precision. FFN weights still use NVFP4.");
                } else {
                    IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
                }
                config_.use_fp8_prefill = 0;
                executor_->disable_fp8_prefill();
            }
        }
    }

    // (Gemma 4 FP8 prefill disabled earlier, before executor init)

    // Pure Mamba2 SSM layers (ssm_in without gdn_gate) are capture-safe: the
    // scan is stream-async and the state lives in one pool allocated once, so
    // graph replay writes it in place exactly as eager. runtime.cuda_graphs=never opts out.

    // Dequant weights → FP16/FP8/NVFP4 caches
    executor_->pre_dequant_weights(stream_, vram_budget);
    dequant_done_ = true;

    // Requires two facts that exist only now: Phase 0 (inside pre_dequant) has
    // labelled host-resident NVFP4 experts, and the expert cache was sized in
    // init_weights(). Refusing here, not at weight-upload time, is the point.
    executor_->verify_host_expert_placement();

    // KV takes the MEASURED residual, not a predicted one: this can only shrink
    // the pool relative to the budget's projection, never grow it, so it cannot
    // overcommit, and keeps the allocator headroom free for later allocations.
    const size_t per_block_total_bytes =
        static_cast<size_t>(n_kv_layers) *
        kv_block_bytes_per_layer(config_.kv_cache_dtype, kv_bs, mcfg.n_kv_heads, head_dim);
    // Ceiling for a growable pool: starts at the plan's commit (may be less than
    // the live-derived kv_max_blocks due to conservative reserves) and grows
    // toward the live figure under admission pressure - without this max() growable
    // was a no-op on planned loads.
    // kv_cache.max_blocks is a ceiling, not a floor: with growable on by default,
    // a pinned pool must never grow past the pin.
    const int kv_blocks_planned = config_.kv_cache_max_blocks > 0
                                      ? max_blocks
                                      : std::max(max_blocks, vram_budget.kv_max_blocks);
    if (per_block_total_bytes > 0) {
        size_t free_now = 0, total_now = 0;
        vram_budget_mem_get_info(&free_now, &total_now);
        // IMMA prefill planes are still OUTSTANDING here (taken on each Q8_0
        // weight's first prefill, during warmup). Charging them here keeps the
        // residual pass from handing the pool bytes the next forward claims (#1899).
        const size_t imma_used = mmq_q8_imma_plane_bytes_used();
        const size_t imma_outstanding = vram_budget.imma_plane_bytes > imma_used
                                            ? vram_budget.imma_plane_bytes - imma_used
                                            : 0;
        if (imma_outstanding > 0) {
            IMP_LOG_INFO(
                "KV cache: holding %.0f MiB of the post-cache residual for the IMMA "
                "prefill planes (taken on the first prefill)",
                imma_outstanding / (1024.0 * 1024.0));
            free_now = free_now > imma_outstanding ? free_now - imma_outstanding : 0;
        }
        const size_t headroom = vram_allocator_headroom(total_now);
        const int max_blocks_planned = kv_blocks_planned;
        const auto sizing =
            kv_blocks_from_residual(free_now, headroom, per_block_total_bytes, max_blocks, 16);
        if (sizing.clamped) {
            IMP_LOG_INFO("KV cache: %d -> %d blocks from the measured post-cache residual "
                         "(%.0f MiB free, %.0f MiB allocator headroom kept)",
                         max_blocks, sizing.blocks, free_now / (1024.0 * 1024.0),
                         headroom / (1024.0 * 1024.0));
            max_blocks = sizing.blocks;
        }
        // The floor is a rescue, not a size: requests longer than it will be
        // cancelled at admission even though the load reports success. The hard
        // failure below fires only with an explicit --vram-budget (#1251).
        if (sizing.floored) {
            // Kept, not just logged: the server has to be able to answer for
            // this after the log line has scrolled away.
            kv_pool_floored_ = true;
            IMP_LOG_WARN(
                "KV cache: only %.0f MiB was left after the weight caches, and the allocator "
                "keeps %.0f MiB of it as headroom — nothing remained to size the pool from, so "
                "it fell back to the %d-block floor (%.0f tokens) instead of the planned %d "
                "blocks. Requests longer than %.0f tokens will be cancelled at admission. "
                "Lower the weight-cache demand (moe.reserve_mib, --kv-fp8) or raise --vram-budget.",
                free_now / (1024.0 * 1024.0), headroom / (1024.0 * 1024.0), sizing.blocks,
                static_cast<double>(sizing.blocks) * kv_bs, max_blocks_planned,
                static_cast<double>(sizing.blocks) * kv_bs);
        }
        // Quiet half of the same fault: pool is real-sized (not floored) but
        // holds less than one max_seq_len sequence - load reports success while
        // every full-length request is cancelled at admission (#1251).
        // Only warn for an operator-set max_seq_len: an AUTO value is expected to
        // be undercut by this clamp (init_compute_max_seq_len_ sizes from raw
        // free VRAM on purpose), so warning there would bury real faults.
        if (vram_budget_bytes() == 0 && max_seq_len_explicit_ &&
            kv_pool_verdict(sizing, config_.max_seq_len, kv_bs) ==
                KvPoolVerdict::ShortOfOneSequence) {
            const int need_blocks = kv_blocks_per_sequence(config_.max_seq_len, kv_bs);
            const double need_mib =
                double(need_blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
            const double have_mib =
                double(sizing.blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
            IMP_LOG_WARN(
                "KV cache: the pool ends up at %d blocks (%.0f MiB, %.0f tokens) but the "
                "requested max_seq_len=%d needs %d blocks (%.0f MiB). Every full-length request "
                "will be cancelled at admission even though this load reports success. Lower "
                "runtime.max_seq_len (imp-cli: --max-seq-len; imp-server: --set "
                "runtime.max_seq_len=N, the flag is CLI-only and exits 1 there, #1681), lower "
                "the weight-cache demand (moe.reserve_mib, --kv-fp8), or "
                "free at least %.0f MiB for the KV pool.",
                sizing.blocks, have_mib, static_cast<double>(sizing.blocks) * kv_bs, config_.max_seq_len,
                need_blocks, need_mib, need_mib - have_mib);
        }
    }

    // I6, plan-time half: an explicit --vram-budget too small for one
    // full-length sequence must fail HERE (naming the arithmetic), not load
    // successfully and cancel every request later.
    // Only when a budget is installed - without one this stays the pre-existing best-effort path.
    if (vram_budget_bytes() > 0 && per_block_total_bytes > 0) {
        const int blocks_per_seq = kv_blocks_per_sequence(config_.max_seq_len, kv_bs);
        if (max_blocks < blocks_per_seq) {
            const double need_mib =
                double(blocks_per_seq) * double(per_block_total_bytes) / (1024.0 * 1024.0);
            const double have_mib =
                double(max_blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
            IMP_LOG_ERROR(
                "--vram-budget %zu MiB is too small for this model: the KV pool ends up at %d "
                "blocks (%.0f MiB) but one max_seq_len=%d sequence needs %d blocks (%.0f MiB). "
                "Every request would be cancelled at admission. Raise --vram-budget by at least "
                "%.0f MiB, or lower runtime.max_seq_len (imp-server: --set "
                "runtime.max_seq_len=N; --max-seq-len is imp-cli only, #1681).",
                vram_budget_bytes() >> 20, max_blocks, have_mib, config_.max_seq_len, blocks_per_seq,
                need_mib, need_mib - have_mib);
            return false;
        }
    }

    // The clamp above answers "what fits right now"; a growable pool keeps the
    // pre-clamp number as its ceiling and commits the clamped one, so a reading
    // skewed by another process still releasing VRAM is not final.
    const int kv_ceiling_blocks = runtime_config_.kv_cache.growable ? kv_blocks_planned : 0;
    // What the pool was actually built with; the retry loop below may lower it.
    int kv_ceiling_effective = kv_ceiling_blocks;
    if (kv_ceiling_blocks > 0) {
        // Commit a fraction on purpose when the operator asked: the clamp above
        // answers "what fits" from a reading that is wrong in both directions on
        // this platform, so starting under it is the only way to stay resident.
        const int pct = std::clamp(runtime_config_.kv_cache.growable_initial_pct, 1, 100);
        if (pct < 100) {
            const int initial = std::max(16, max_blocks / 100 * pct);
            if (initial < max_blocks) {
                IMP_LOG_INFO("KV cache: committing %d%% of the pool up front (%d of %d blocks)", pct, initial,
                             max_blocks);
                max_blocks = initial;
            }
        }
    }
    if (kv_ceiling_blocks > max_blocks) {
        IMP_LOG_INFO(
            "KV cache: growable, starting at %d blocks with a %d-block ceiling "
            "(%.0f -> %.0f tokens as VRAM frees)",
            max_blocks, kv_ceiling_blocks, static_cast<double>(max_blocks) * kv_bs,
            static_cast<double>(kv_ceiling_blocks) * kv_bs);
    }

    {
        QType kv_dtype = config_.kv_cache_dtype;
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks *
                          kv_block_bytes_per_layer(kv_dtype, kv_bs, mcfg.n_kv_heads, head_dim);
        IMP_LOG_INFO(
            "KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
            "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
            max_blocks, static_cast<double>(max_blocks) * kv_bs,
            static_cast<double>(total_kv) / (1024.0 * 1024.0), dtype_name(kv_dtype), n_kv_layers,
            mcfg.n_layers, mcfg.n_kv_heads, head_dim, kv_bs);
    }

    // Per-layer KV shape path (Gemma 4 dual attention geometry): build per-layer
    // nkv/hd arrays restricted to attention layers (hybrid models may have non-attn layers).
    // SWA sizing also requires the per-layer path (per-layer region capacities).
    auto make_kv_cache = [&](int blocks, int ceiling) -> std::unique_ptr<KVCache> {
        std::unique_ptr<KVCache> kv_cache;
        if ((!mcfg.head_dim_per_layer.empty() || swa_sizing_active_) &&
            config_.kv_cache_dtype != QType::INT8 && config_.kv_cache_dtype != QType::INT4) {
            std::vector<int> per_layer_nkv(n_kv_layers, 0);
            std::vector<int> per_layer_hd(n_kv_layers, 0);
            std::vector<char> per_layer_swa(swa_sizing_active_ ? n_kv_layers : 0, 0);
            for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
                int attn_nkv = (l < (int)mcfg.n_kv_heads_per_layer.size()) ? mcfg.n_kv_heads_per_layer[l]
                                                                           : mcfg.n_kv_heads;
                if (kv_layer_map[l] < 0)
                    continue;  // non-attention layer (SSM/GDN)
                if (attn_nkv <= 0)
                    attn_nkv = mcfg.n_kv_heads;
                per_layer_nkv[k] = attn_nkv;
                per_layer_hd[k] = (l < (int)mcfg.head_dim_per_layer.size() && mcfg.head_dim_per_layer[l] > 0)
                                      ? mcfg.head_dim_per_layer[l]
                                      : head_dim;
                if (swa_sizing_active_)
                    per_layer_swa[k] = layer_swa_window(mcfg, model_->profile(), l) > 0 ? 1 : 0;
                k++;
            }
            kv_cache = std::make_unique<KVCache>(n_kv_layers, per_layer_nkv, per_layer_hd,
                                                 config_.kv_cache_dtype, blocks, kv_bs, &vram_alloc_,
                                                 per_layer_swa,
                                                 swa_sizing_active_ ? vram_budget.swa_max_blocks : 0,
                                                 ceiling);
        } else {
            kv_cache = std::make_unique<KVCache>(n_kv_layers, mcfg.n_kv_heads, head_dim,
                                                 config_.kv_cache_dtype, blocks, kv_bs, &vram_alloc_,
                                                 ceiling);
        }
        return kv_cache;
    };

    // #1662: a pool that does not fit is a smaller pool, not a dead process.
    // Every sizing input above is a projection and can still be wrong (#1631,
    // #1662); halving down to the 16-block floor is the backstop.
    // Only the pool retries - weight caches are already built and the model's
    // source tensors may be consumed, so a retry one level up (imp_context_create)
    // cannot rebuild this engine.
    std::unique_ptr<KVCache> kv_cache;
    {
        constexpr int kKVFloorBlocks = 16;
        const int planned = max_blocks;
        int attempt_blocks = max_blocks;
        int attempt_ceiling = kv_ceiling_blocks;
        for (;;) {
            try {
                kv_cache = make_kv_cache(attempt_blocks, attempt_ceiling);
                break;
            } catch (const std::exception& e) {
                if (attempt_blocks <= kKVFloorBlocks) {
                    IMP_LOG_ERROR(
                        "KV cache: %d blocks is the %d-block floor and it still does not "
                        "fit - giving up (%s)",
                        attempt_blocks, kKVFloorBlocks, e.what());
                    throw;
                }
                const int next = std::max(kKVFloorBlocks, attempt_blocks / 2);
                const double short_mib = static_cast<double>(n_kv_layers) * (attempt_blocks - next) *
                                         kv_block_bytes_per_layer(config_.kv_cache_dtype, kv_bs,
                                                                  mcfg.n_kv_heads, head_dim) /
                                         (1024.0 * 1024.0);
                IMP_LOG_WARN(
                    "KV cache: %d blocks did not fit (planned %d), retrying at %d blocks "
                    "- %.1f MiB less (%s)",
                    attempt_blocks, planned, next, short_mib, e.what());
                attempt_blocks = next;
                // The ceiling is a reservation of the same pool, so a retry
                // that only halves the committed part reserves the same
                // address space again and fails the same way.
                if (attempt_ceiling > 0)
                    attempt_ceiling = std::max(next, attempt_ceiling / 2);
            }
        }
        if (attempt_blocks != max_blocks) {
            max_blocks = attempt_blocks;
            kv_ceiling_effective = attempt_ceiling;
            IMP_LOG_WARN("KV cache: serving %d blocks (%.0f tokens) instead of the planned %d", max_blocks,
                         static_cast<double>(max_blocks) * kv_bs, planned);
        }
    }
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));
    // A successful allocation proves nothing on WSL2/WDDM: the pool can sit in
    // host memory at a sixth of the bandwidth with /health ok and nothing logged
    // (#1103, AUDIT_arch_2026 B-6). One cheap copy inside the fresh pool, then a
    // WARN and a gauge, never a refusal - the threshold is one driver on one card.
    {
        const double gbps = kv_cache_raw_->probe_residency();
        if (gbps > 0.0 && gbps < kKvPoolSpillGbps) {
            IMP_LOG_WARN(
                "KV cache: pool copy bandwidth %.0f GB/s, below %.0f GB/s: the pool has most likely "
                "spilled into host memory (WDDM oversubscription) and decode will run at a fraction of "
                "the card's bandwidth. Free the VRAM other processes hold, or pin a smaller pool with "
                "kv_cache.max_blocks",
                gbps, kKvPoolSpillGbps);
        } else if (gbps > 0.0) {
            IMP_LOG_INFO("KV cache: pool copy bandwidth %.0f GB/s (resident)", gbps);
        }
    }
    if (swa_sizing_active_) {
        kv_manager_->enable_swa_sizing(swa_window_max_, swa_slack_tokens_);
        swa_sizing_active_ = kv_manager_->swa_sizing_enabled();
    }

    // BitDecoding Phase 3: residual FP16 cache (opt-in). Ring state (write_idx/
    // fill_count per slot) lives in device memory, updated by
    // advance_residual_state_kernel at the end of forward_logits - keeps the
    // whole path graph-capture-safe.
    {
        const auto& rcfg = runtime_config_;
        int residual_n = rcfg.kv_cache.bitdecoding_residual_tokens;
        if (residual_n > 0 && config_.kv_cache_dtype == QType::NVFP4) {
            int max_seqs = config_.max_batch_size > 0 ? config_.max_batch_size : 1;
            if (kv_manager_->enable_residual_buffer(max_seqs, residual_n, &vram_alloc_)) {
                // Persistent batch→slot lookup buffer (graph-safe). [max_batch_size] ints.
                size_t slot_bytes = static_cast<size_t>(max_seqs) * sizeof(int);
                cudaMalloc(&d_kv_slot_buf_, slot_bytes);
                std::vector<int> init_slots(max_seqs, -1);
                cudaMemcpy(d_kv_slot_buf_, init_slots.data(), slot_bytes, cudaMemcpyHostToDevice);
                d_kv_slot_last_uploaded_.assign(max_seqs, -1);
                // Same treatment for the multi-sequence metadata (#1648): must
                // be allocated ONCE, not per decode step - a captured
                // forward_logits graph bakes the address and nothing invalidates it on reuse.
                size_t meta_bytes = static_cast<size_t>(3) * max_seqs * sizeof(int);
                if (cudaMalloc(&residual_meta_d_buf_, meta_bytes) == cudaSuccess) {
                    residual_meta_capacity_ = max_seqs;
                    std::vector<int> init_meta(3 * static_cast<size_t>(max_seqs), 0);
                    cudaMemcpy(residual_meta_d_buf_, init_meta.data(), meta_bytes, cudaMemcpyHostToDevice);
                }
            }
        } else if (residual_n > 0) {
            IMP_LOG_INFO("kv_cache.bitdecoding_residual_tokens=%d ignored (only active with kv_cache_dtype=NVFP4)",
                         residual_n);
        }
    }

    // attention.sparse_topk_tokens: per-block key min/max metadata pool.
    // Gates: F16/FP8/NVFP4 KV only (NVFP4 unpacks nibbles + UE4M3 scale, #1818);
    // non-MLA; token_recycling off (copy_blocks_device doesn't copy metadata);
    // enable_key_minmax also refuses per-layer geometry and growable pools.
    // A refused gate disables the feature loudly and changes nothing else.
    if (runtime_config_.attention.sparse_topk_tokens > 0) {
        const QType kvt = config_.kv_cache_dtype;
        const char* refuse = nullptr;
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 && kvt != QType::NVFP4)
            refuse = "KV dtype (needs f16, fp8 or nvfp4)";
        else if (mcfg.is_mla())
            refuse = "MLA model";
        else if (runtime_config_.speculative.token_recycling)
            refuse = "speculative.token_recycling";
        else if (!config_.prefix_cache_path.empty())
            refuse = "persistent prefix cache (disk-restored blocks bypass the KV write path and "
                     "would carry empty metadata)";
        if (refuse) {
            IMP_LOG_WARN("attention.sparse_topk_tokens=%d ignored: %s",
                         runtime_config_.attention.sparse_topk_tokens, refuse);
        } else if (!kv_cache_raw_->enable_key_minmax()) {
            IMP_LOG_WARN("attention.sparse_topk_tokens=%d ignored: metadata pool unavailable "
                         "(per-layer KV geometry, growable pool, or allocation failure)",
                         runtime_config_.attention.sparse_topk_tokens);
        }
    }

    if (config_.use_prefix_caching) {
        kv_manager_->set_prefix_caching_enabled(true);
        // cache_control/cache_prompt pin budget: percent of the pool,
        // floor of 1 block when enabled at all.
        int pin_pct = std::min(std::max(config_.prefix_pin_budget_pct, 0), 100);
        int pin_budget =
            pin_pct > 0 ? std::max(1, kv_manager_->kv_cache()->total_blocks() * pin_pct / 100) : 0;
        kv_manager_->set_pin_budget_blocks(pin_budget);
        IMP_LOG_INFO("Prefix caching enabled (pin budget %d blocks)", pin_budget);
        // Persistent cache is dense-only: restored KV blocks are usable for
        // hybrids only together with a recurrent-state snapshot, and snapshots
        // are not persisted (the recurrent-snapshot store below must also come up).
        if (mcfg.ssm_inner_size == 0 && !config_.prefix_cache_path.empty()) {
            int restored = kv_manager_->load_prefix_cache(config_.prefix_cache_path,
                                                          model_fingerprint_(), stream_);
            if (restored > 0)
                IMP_LOG_INFO("Restored %d prefix cache blocks from %s", restored,
                             config_.prefix_cache_path.c_str());
        }
    }

    executor_->set_kv_layer_map(std::move(kv_layer_map));

    if (offload_mgr_)
        executor_->set_offload_manager(offload_mgr_.get());
    scheduler_->set_kv_manager(kv_manager_.get());

    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr)
                n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_conv_channels();
            int n_heads = mcfg.ssm_dt_rank;
            int hd = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_state_ = std::make_unique<SSMState>();
            // Lazy: reserve every slot, commit one per admitted sequence
            // (scheduler admission gate below). The plan above charged the
            // whole slab either way.
            const bool ssm_pool_ok = ssm_state_->init(n_ssm, config_.max_batch_size, conv_ch,
                                                      mcfg.ssm_conv_kernel, n_heads, hd, mcfg.ssm_state_size,
                                                      config_.ssm_state_dtype, &vram_alloc_,
                                                      ssm_reserved_slots,
                                                      runtime_config_.vram.lazy_commit ? vmm_backend()
                                                                                       : nullptr,
                                                      batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size));
            if (ssm_pool_ok && ssm_state_->lazy() && scheduler_)
                scheduler_->set_admission_gate([this] { return recurrent_slot_admissible_(); });
            if (must_refuse_without_ssm_state(n_ssm, ssm_pool_ok)) {
                // NOT "continuing without it": a GDN/SSM layer with a missing
                // recurrent state reads a null slab and produces garbage for
                // every request. SSMState::init already logged bytes/slots/lever.
                ssm_state_.reset();
                throw std::runtime_error(
                    "SSM/GDN state pool allocation failed and this model has " +
                    std::to_string(n_ssm) +
                    " recurrent layers; refusing to serve without it (see the SSM/GDN state "
                    "pool line above for the shortfall and the lever)");
            } else if (ssm_reserved_slots > 0) {
                IMP_LOG_INFO("SSM state: %d slot(s) reserved past max_batch_size=%d: %d multi-candidate verify "
                             "(mtp_tree_width=%d), %d batched-verify spares (lazy), %.1f MiB each",
                             ssm_reserved_slots, config_.max_batch_size, spec_mc_reserved_slots_(),
                             runtime_config_.speculative.mtp_tree_width, batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size),
                             ssm_state_->per_seq_bytes() / (1024.0 * 1024.0));
            }
            // Slot table for batched GDN decode. Allocated once and kept at a
            // stable address so a captured decode graph does not have to be
            // rebuilt when the set of active sequences changes.
            if (ssm_state_ && config_.max_batch_size > 0) {
                const size_t bytes = static_cast<size_t>(config_.max_batch_size) * sizeof(int);
                if (cudaMalloc(&d_ssm_seq_slots_, bytes) != cudaSuccess) {
                    IMP_LOG_WARN("batched GDN decode: slot table alloc failed — staying single-sequence");
                    d_ssm_seq_slots_ = nullptr;
                } else {
                    cudaMemset(d_ssm_seq_slots_, 0, bytes);
                    h_ssm_seq_slots_.assign(static_cast<size_t>(config_.max_batch_size), 0);
                }
            }
        }

        // Recurrent-state snapshots: KV block reuse alone cannot skip prefill
        // for a recurrent model (state at the skip boundary would be zero), so
        // hybrid prefix caching needs the snapshot store, or it must turn back off.
        if (kv_manager_->prefix_caching_enabled()) {
            int budget_mb = runtime_config_.server.recurrent_snapshot_mb;
            if (ssm_state_ && budget_mb > 0) {
                recurrent_snapshots_ = std::make_unique<RecurrentSnapshotStore>();
                                recurrent_snapshots_->init(
                    ssm_state_->per_seq_bytes(), static_cast<size_t>(budget_mb) << 20,
                    static_cast<size_t>(std::max(runtime_config_.server.recurrent_snapshot_host_mb, 0))
                        << 20);
                if (recurrent_snapshots_->enabled()) {
                    scheduler_->set_prefix_reuse_limit(
                        [this](Request& r) { return hybrid_prefix_reuse_limit_(r); });
                } else {
                    recurrent_snapshots_.reset();
                }
            }
            if (!recurrent_snapshots_) {
                kv_manager_->set_prefix_caching_enabled(false);
                IMP_LOG_INFO(
                    "Prefix caching disabled for recurrent model (snapshot store off — "
                    "server.recurrent_snapshot_mb=%d)",
                    budget_mb);
            }
        }
    }

    // kv_cache.swa_snapshot_mb: under SWA sizing, global-layer KV blocks alone
    // cannot back a prefix-cache hit (windowed layers' earlier blocks were
    // trailing-freed, leaving holes). The store keeps the packed window at each
    // prefill-end prefix hash and restores it at admission; without a working store, fall back to prefix caching off.
    if (swa_sizing_active_ && kv_manager_->prefix_caching_enabled()) {
        const int budget_mb = runtime_config_.kv_cache.swa_snapshot_mb;
        if (budget_mb > 0 && kv_manager_->enable_swa_snapshots()) {
            const size_t slab_bytes = kv_manager_->swa_snapshot_bytes();
            if (cudaMalloc(&swa_snap_slab_, slab_bytes) == cudaSuccess) {
                swa_snapshots_ = std::make_unique<RecurrentSnapshotStore>();
                swa_snapshots_->init(slab_bytes, static_cast<size_t>(budget_mb) << 20);
                if (swa_snapshots_->enabled()) {
                    scheduler_->set_prefix_reuse_limit(
                        [this](Request& r) { return swa_prefix_reuse_limit_(r); });
                    IMP_LOG_INFO("SWA snapshots: %d MiB budget, %zu KiB/snapshot, capacity %d",
                                 budget_mb, slab_bytes >> 10, swa_snapshots_->capacity());
                } else {
                    swa_snapshots_.reset();
                    IMP_CUDA_CHECK_LOG(cudaFree(swa_snap_slab_));
                    swa_snap_slab_ = nullptr;
                }
            }
        }
        if (!swa_snapshots_) {
            kv_manager_->set_prefix_caching_enabled(false);
            config_.use_prefix_caching = false;
            // A budget below one snapshot silently costs prefix caching, worse
            // than swa_snapshot_mb=0 (which keeps caching, drops SWA savings).
            // Name the required size or the two cases are indistinguishable in the log.
            const size_t need_mb = (kv_manager_->swa_snapshot_bytes() + (1u << 20) - 1) >> 20;
            if (budget_mb > 0 && static_cast<size_t>(budget_mb) < need_mb) {
                IMP_LOG_WARN("Prefix caching DISABLED: kv_cache.swa_snapshot_mb=%d is below one "
                             "snapshot (%zu MiB). Set it to >=%zu to run SWA sizing AND prefix "
                             "caching together, or to 0 to keep prefix caching and drop the SWA "
                             "savings.",
                             budget_mb, need_mb, need_mb);
            } else {
                IMP_LOG_INFO("Prefix caching disabled under SWA sizing (snapshot store off — "
                             "kv_cache.swa_snapshot_mb=%d)",
                             budget_mb);
            }
        }
    }

    cudaStreamSynchronize(stream_);

    // Coverage check: for prequant MoE, the nvfp4_moe decode cache is
    // all-or-nothing (mirrors executor_forward_moe.cu nvfp4_covers_layer). One
    // uncovered layer falls to host-args legacy, which throws under graph capture.
    if (mcfg.is_nvfp4_prequant && mcfg.n_experts > 0) {
        int moe_layers = 0, covered = 0;
        for (int i = 0; i < mcfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            bool has_experts = L.expert_up_packed.data != nullptr ||
                               (!L.expert_w_up.empty() && L.expert_w_up[0].data != nullptr);
            if (!has_experts)
                continue;
            moe_layers++;
            bool ok = L.nvfp4_moe_up_ptr != nullptr && L.nvfp4_moe_down_ptr != nullptr;
            if (ok && L.expert_gate_packed.data != nullptr)
                ok = L.nvfp4_moe_gate_ptr != nullptr;
            if (ok)
                covered++;
        }
        if (moe_layers > 0 && covered == moe_layers) {
            IMP_LOG_INFO("NVFP4 decode caches: FULL (%d/%d MoE layers) — decode graph "
                         "capture eligible",
                         covered, moe_layers);
        } else if (moe_layers > 0) {
            IMP_LOG_WARN(
                "NVFP4 decode caches: PARTIAL (%d/%d MoE layers covered) — decode "
                "CUDA-graph capture will abort and decode runs per-step (~10x slower). "
                "Remedies: lower runtime.max_seq_len or max_batch_size (both shrink the "
                "workspaces/KV competing for cache VRAM), or check the [vram] knobs.",
                covered, moe_layers);
        }
    }

    // Pre-allocate the gemm_nvfp4 fallback dequant workspace, sized from
    // wcache_.nvfp4 (populated above, so must come AFTER pre_dequant_weights).
    // Lets the M>1 fallback path run inside CUDA stream capture without cudaMalloc.
    (void)executor_->allocate_nvfp4_dequant_workspace();
    // CUTLASS NVFP4 LM head for batched-decode GEMM: only n>1 decode consumes
    // it, so skip the 19-47 MiB SfAtom-scale VRAM when max_batch_size==1 (the
    // perplexity harness lazy-builds its own). Must run AFTER pre_dequant_weights
    // and BEFORE decode-graph capture so the captured topology includes it.
    if (config_.max_batch_size > 1)
        executor_->build_lm_head_cutlass_(stream_);
    if (config_.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq,
                                /*with_swa_tables=*/swa_sizing_active_);
    // The "can the verify run" gate lives inside the callee: gating here on the
    // spare slot count left speculative.factored_spare permanently disabled
    // (it reserves none by design).
    if (!ensure_batch_verify_bufs_())  // init-time staging
        IMP_LOG_WARN("spec-batch: staging buffers unavailable - batched verify stays off");
    {
        d_penalty_tokens_capacity_ = static_cast<size_t>(config_.max_seq_len);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_WARN("Failed to pre-allocate penalty token buffer");
            d_penalty_tokens_capacity_ = 0;
        }
        // Batched-decode penalty histories (see engine.h). max_batch_size
        // slots x max_seq_len tokens; 32 x 4096 = 512 KiB, 64 x 131072 = 32 MiB.
        penalty_hist_slots_ =
            std::min(config_.max_batch_size, (int)imp::PenaltyAppendArgs::kMaxRows);
        penalty_hist_cap_ = config_.max_seq_len;
        if (penalty_hist_slots_ > 1 && penalty_hist_cap_ > 0) {
            d_penalty_hist_ = static_cast<int32_t*>(vram_alloc_.allocate(
                (size_t)penalty_hist_slots_ * penalty_hist_cap_ * sizeof(int32_t),
                "penalty_hist"));
            if (d_penalty_hist_) {
                penalty_hist_state_.assign(penalty_hist_slots_, {});
            } else {
                IMP_LOG_WARN("penalty_hist: alloc failed — batched sampling keeps the "
                             "per-row upload path");
                penalty_hist_slots_ = 0;
                penalty_hist_cap_ = 0;
            }
        }
    }

    {
        init_serving_metadata_pool_(max_blocks, kv_ceiling_effective, kv_bs);

        // T5b: an empty buffer means "no staging", which the prefill path
        // already tests for (memory/host_pinned.h).
        h_pf_positions_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                static_cast<size_t>(config_.max_seq_len) * sizeof(int));
        h_pf_token_ids_ = PinnedBuffer::acquire(
            cuda_host_pinned_allocator(),
            static_cast<size_t>(config_.max_seq_len) * sizeof(int32_t));
        if (cudaEventCreateWithFlags(&pf_staging_evt_, cudaEventDisableTiming) != cudaSuccess)
            pf_staging_evt_ = nullptr;
        // The constrained pipeline's pinned landing for the sampled token and
        // its ready event: engine-lifetime, so a first json request does not
        // allocate while serving (I2 gate phase B counted the lazy acquire).
        cpipe_.h_token = PinnedBuffer::acquire(cuda_host_pinned_allocator(), sizeof(int32_t));
        if (cudaEventCreateWithFlags(&cpipe_.ev, cudaEventDisableTiming) != cudaSuccess)
            cpipe_.ev = nullptr;
    }

    // Report memory
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                         (total_mem - free_mem) / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
                         free_mem / (1024.0 * 1024.0));
        vram_alloc_.report();
    }

    return true;
}

// Serving metadata pool: one allocation at init per forward path (token ids,
// positions, block tables, context lengths), so nothing allocates while
// serving (I2, MEMORY.md A3.2). Each region sized to its path's worst case:
//   serial prefill      one chunk of max_seq_len rows, one block table
//   ragged prefill      max_seq_len rows total, max_batch_size tables
//   graph loops         one block table each for sync/async/constrained loops,
//                       plus the pipeline's sampled token, position, context
// Regions are disjoint: the async loop keeps its table across scheduler steps,
// so it cannot share with a ragged prefill running in between. Layout: runtime/serving_metadata_layout.h.
void Engine::init_serving_metadata_pool_(int max_blocks, int kv_ceiling_effective, int kv_bs) {
    // A block table can grow to the entire KV pool (max_blocks) or to the
    // CEILING of a growable pool - "the entire pool" is a moving number as the
    // pool grows past its initial commit. The async loop sizes from the
    // request's max_tokens ceiling (bounded by max_seq_len, not the pool).
    // Four bytes per block of a pool that may not exist: cheap to over-allocate.
    const int bt_cap = std::max(
        {max_blocks, kv_ceiling_effective, (config_.max_seq_len + kv_bs - 1) / kv_bs});
    const int seq_cap = std::max(1, config_.max_batch_size);
    const auto lay = ServingMetadataLayout::compute(config_.max_seq_len, seq_cap, bt_cap, swa_sizing_active_,
                                                    SAMPLE_SCRATCH_BYTES);
    prefill_pool_size_ = lay.total;
    prefill_pool_ = vram_alloc_.allocate(prefill_pool_size_, "prefill_pool");
    if (!prefill_pool_) {
        IMP_LOG_WARN("Failed to pre-allocate the serving metadata pool, will use per-request malloc");
        return;
    }
    auto* base = static_cast<char*>(prefill_pool_);
    auto ints = [base](size_t o) { return reinterpret_cast<int*>(base + o); };
    d_pf_token_ids_ = reinterpret_cast<int32_t*>(base + lay.pf_tok);
    d_pf_positions_ = ints(lay.pf_pos);
    d_pf_block_tables_ = ints(lay.pf_bt);
    d_pf_context_lens_ = ints(lay.pf_ctx);
    d_rg_token_ids_ = reinterpret_cast<int32_t*>(base + lay.rg_tok);
    d_rg_positions_ = ints(lay.rg_pos);
    d_rg_block_tables_ = ints(lay.rg_bt);
    d_rg_context_lens_ = ints(lay.rg_ctx);
    d_rg_seq_offsets_ = ints(lay.rg_soff);
    d_rg_ssm_slots_ = ints(lay.rg_slots);
    d_gl_block_tables_ = ints(lay.gl_bt);
    d_agl_block_tables_ = ints(lay.agl_bt);
    d_cp_block_tables_ = ints(lay.cp_bt);
    d_cp_token_ = reinterpret_cast<int32_t*>(base + lay.cp_token);
    d_cp_pos_ = ints(lay.cp_pos);
    d_cp_ctx_ = ints(lay.cp_ctx);
    if (swa_sizing_active_) {
        d_pf_block_tables_swa_ = ints(lay.pf_bt_swa);
        d_gl_block_tables_swa_ = ints(lay.gl_bt_swa);
        d_agl_block_tables_swa_ = ints(lay.agl_bt_swa);
    }
    pool_bt_cap_ = bt_cap;
    rg_rows_cap_ = config_.max_seq_len;
    rg_seq_cap_ = seq_cap;
    // M-RoPE position uploads (engine_qwen3vl.cpp) are sized once here, up to
    // what bind_mrope_ can be asked for: prefill rows to the executor's chunk
    // ceiling, decode rows to the batch ceiling. bind_mrope_ never regrows a fitting buffer.
    if (model_->config_.has_mrope()) {
        const int rows = std::max(1, executor_ ? executor_->max_tokens() : config_.max_seq_len);
        d_mrope_prefill_ = static_cast<int32_t*>(
            vram_alloc_.allocate(static_cast<size_t>(3) * rows * sizeof(int32_t), "mrope_positions"));
        mrope_prefill_cap_ = d_mrope_prefill_ ? rows : 0;
        d_mrope_decode_ = static_cast<int32_t*>(
            vram_alloc_.allocate(static_cast<size_t>(seq_cap) * sizeof(int32_t), "mrope_delta"));
        mrope_decode_cap_ = d_mrope_decode_ ? seq_cap : 0;
    }
    IMP_LOG_INFO("Serving metadata pool: %.1f MiB (rows %d, %d tables x %d blocks)",
                 prefill_pool_size_ / (1024.0 * 1024.0), config_.max_seq_len, seq_cap + 4, bt_cap);
}

}  // namespace imp

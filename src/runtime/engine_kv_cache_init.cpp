// Engine init phase: paged KV cache allocation.
// Decides block geometry (block_size=16), allocates blocks per KV dtype
// (FP16/FP8/INT8/INT4/NVFP4/MXFP4), wires up KVCacheManager. Also
// initialises the BitDecoding residual FP16 cache (opt-in), SSM/GDN
// state, pre-dequant weight caches, decode batch pool, prefill metadata
// pool + pinned staging, and reports VRAM usage.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Method remains Engine:: with declaration in engine.h.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "memory/vram_query.h"
#include "memory/library_reserve_cache.h"
#include "memory/plan.h"
#include "runtime/plan_shadow.h"
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include <utility>

namespace imp {

// Stable identity hash of the loaded model: FNV-1a over config identity
// scalars plus a small sample of real weight bytes (LM head, token embeddings,
// layer-0 and mid-layer Q projections). The weight sample is what distinguishes
// two same-shape fine-tunes (identical config) that the geometry checks cannot.
// Used only to gate the persisted prefix cache (cold-path, twice per process).
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

    // Build KV layer mapping for hybrid models
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

    // Auto-select block size
    if (config_.kv_block_size <= 0) {
        config_.kv_block_size = (mcfg.n_kv_heads <= 4 && mcfg.n_kv_heads > 0) ? 32 : kKVBlockSize;
        IMP_LOG_INFO("KV block size: auto → %d (n_kv_heads=%d)", config_.kv_block_size, mcfg.n_kv_heads);
    }
    const int kv_bs = config_.kv_block_size;
    int blocks_per_seq = (config_.max_seq_len + kv_bs - 1) / kv_bs;

    // ── SWA-aware KV sizing gate (kv_cache.swa_sizing) ────────────────
    // Sliding-window layers get a small dedicated block group instead of
    // full-length KV. Resolve the gate + geometry before the VRAM budget so
    // the budget charges SWA layers window-cost, not context-cost.
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
            // Prefix caching cannot reuse freed window blocks on its own.
            // With a SWA snapshot budget the two coexist (the store below
            // restores the window at the reuse boundary); without one, only
            // an explicit "on" reaches this point — honor the forced opt-in
            // by disabling prefix caching.
            if (config_.use_prefix_caching && runtime_config_.kv_cache.swa_snapshot_mb <= 0) {
                config_.use_prefix_caching = false;
                IMP_LOG_INFO("kv_cache.swa_sizing=on: prefix caching disabled (freed window "
                             "blocks cannot back prefix reuse; set kv_cache.swa_snapshot_mb "
                             "to combine)");
            }
            // StreamingLLM auto-enable frees middle blocks of the GLOBAL
            // table — redundant and conflicting here.
            config_.streaming_kv_auto = false;
        }
    }
    const int swa_live_tokens =
        swa_sizing_active_ ? swa_window_max_ + swa_slack_tokens_ + swa_burst_cap_tokens_ : 0;

    // VRAM budget. Nothing is held back from effective_free_vram() any more
    // (the balloon is gone, AUDIT B62): the prequant reserve charges KV the full
    // measured cache demand and phase 3 is floored at it.
    // Charge what the first forward actually claimed LAST time, if we know
    // (AUDIT B41/B49). The plan needs this number before the forward that
    // produces it, so a single run cannot both measure and use it — but the
    // value is stable per (model, quant path, library stack) and invariant to
    // batch and context, so remembering it is enough. Explicit
    // vram.library_reserve_mb always wins; a miss leaves the constant in place.
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
            // `remembered > 0` was the test here, which discarded a recorded
            // ZERO and charged the 3900 MiB constant instead — on exactly the
            // models whose first forward claims nothing. B43 fixed this shape in
            // the reporter; the loader kept it (AUDIT B70).
            config_.library_reserve_mb = static_cast<int>(remembered >> 20);
            IMP_LOG_INFO("library reserve: %d MiB from the measurement cache (%s) — the default "
                         "constant is %zu MiB",
                         config_.library_reserve_mb, path.c_str(),
                         kMeasuredLibraryReserveBytes >> 20);
        } else if (!path.empty()) {
            // No entry: the plan is about to charge the constant. Say so HERE,
            // before the pools are sized, instead of only reporting the
            // mismatch after the first forward — by then the KV pool has
            // already been sized around a reserve the model may not want.
            IMP_LOG_INFO("library reserve: no measurement for this model in %s — planning with the "
                         "%zu MiB constant. It is recorded after the first forward; mount that "
                         "path (or set vram.library_reserve_cache) to keep it across restarts.",
                         path.c_str(), kMeasuredLibraryReserveBytes >> 20);
        }
    }

    auto vram_budget = compute_vram_budget(*model_, config_, n_kv_layers, head_dim,
                                           effective_free_vram(), swa_live_tokens, n_swa_layers,
                                           &native_cache_demand());
    // A7 step 2 — APPLIED. The KV block count now comes from plan_memory(), not
    // from the live-free-derived pass. What made that safe is three changes, in
    // this order: the balloon stopped hiding bytes from the live read (B62), the
    // unexplained `* 2` that was the whole live-vs-plan divergence went (B65),
    // and V8 now asserts the live pass never exceeds the plan (B66). Today the
    // two produce the SAME number on every measured config, so this changes
    // where the number comes from rather than what it is — and the
    // measured-residual clamp further down can still only shrink it, which is
    // what keeps a plan that is wrong about the device from overcommitting.
    int max_blocks = 0;
    {
        ShadowPlanProbe probe;
        probe.distributable_bytes = effective_free_vram();
        probe.weight_cache_demand = vram_budget.weight_cache_estimate_bytes;
        probe.mandatory_cache_bytes =
            vram_budget.mandatory_sf_bytes + vram_budget.mandatory_moe_bytes;
        probe.ssm_state_bytes = vram_budget.ssm_footprint_bytes;
        if (executor_) {
            probe.engine_persistent_bytes = executor_->workspace_estimate();
            probe.workspace_estimate_available = true;
        }
        probe.vision_tower_unmodelled = !config_.mmproj_path.empty();
        // config_.library_reserve_mb, NOT the runtime-config field: the loader
        // above writes the REMEMBERED measurement into the former, and reading
        // the latter here meant the plan kept charging the 3900 MiB constant
        // while the live pass used the measured 0. Two fields, one number, and
        // the plan was on the wrong one (AUDIT B70).
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

        const PlanResult plan = plan_memory(shadow_plan_input(probe));
        IMP_LOG_INFO("%s", shadow_plan_report(probe, plan, vram_budget.kv_max_blocks).c_str());

        if (config_.kv_cache_max_blocks > 0) {
            max_blocks = config_.kv_cache_max_blocks;  // operator pin wins over both
        } else if (plan.ok) {
            max_blocks = plan.plan.kv.blocks;
            if (max_blocks != vram_budget.kv_max_blocks) {
                // Not a failure — the two are allowed to differ, and the plan is
                // the one that charges what the live read cannot see. Logged
                // because a silent divergence is how the old pass drifted.
                IMP_LOG_INFO("KV blocks: plan %d (live pass would have said %d)", max_blocks,
                             vram_budget.kv_max_blocks);
            }
        } else {
            // The plan refuses this configuration. D8 argues that should fail
            // the load, and it does when an explicit --vram-budget is installed
            // (the check further down). Without one, falling back to the live
            // pass keeps the pre-existing best-effort behaviour rather than
            // turning a plan gap into a refusal to serve.
            max_blocks = vram_budget.kv_max_blocks;
            IMP_LOG_WARN("KV blocks: the plan rejects this configuration — falling back to the "
                         "live-derived %d blocks. The report above says what it could not fit.",
                         max_blocks);
        }
    }

    // ── Weight caches are built BEFORE the KV pool (A7 step 6.4) ──────
    // Measured on gpt-oss-20b-mxfp4 at server defaults: sizing KV first from
    // an ESTIMATE of the cache demand left the card at exactly 0 MiB free —
    // the estimate was ~1.6 GiB low, and the caches took the difference out
    // of the reserve. At 0 free, WSL2/WDDM spills into host memory (the exact
    // hazard VRAMAllocator's own docstring names) and decode collapses from
    // ~345 tok/s to 25 tok/s. Leaving as little as 1 GiB free restores it.
    // So the caches, whose demand is bounded by the model, go first, and the
    // KV pool — the elastic tier — takes what is actually left (AUDIT B23).
    //
    // The two profile gates below must come with them: both feed flags the
    // cache build reads (wcache_->use_fp8) or that graph capture depends on.
    // GDN detection
    {
        if (model_->profile().is_gdn) {
            if (config_.use_cuda_graphs) {
                IMP_LOG_INFO("GDN model: CUDA graphs enabled (recurrent state in-place)");
            } else {
                IMP_LOG_INFO(
                    "GDN model: CUDA graphs disabled (disabled earlier by caller or expert offload)");
            }
            // GDN recurrent state accumulates small precision errors per token.
            // FP8 E4M3 (3-bit mantissa) amplifies these through the delta rule
            // scan, causing degenerate output after ~50 special tokens in
            // multi-turn chat.  Force FP16 weights for GDN prefill.
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

    // Pure Mamba2 SSM layers (ssm_in without gdn_gate) used to demote graphs
    // here, on the assumption that the recurrent state was not capture-safe.
    // It is: the scan's device work is all stream-async, and the state lives in
    // one pool allocated once, so replay writes it in place exactly as an eager
    // step does. Measured 2026-08-12 on this box, decode, spec off:
    //   Nemotron-3.5-Lightning  126.2 -> 365.6 tok/s
    //   Nemotron-3-Nano         127.2 -> 381.7 tok/s
    // with 45/45 degen_suite, a clean 700-token generation, multi-turn, and
    // four concurrent requests keeping their states apart. `AUDIT_ARCH` called
    // this "eager decode by design" and supported-models.md called the model
    // "arch-limited" — both were describing this demotion, not the
    // architecture. `runtime.cuda_graphs=never` remains the way out.

    // Dequant weights → FP16/FP8/NVFP4 caches
    executor_->pre_dequant_weights(stream_, vram_budget);
    dequant_done_ = true;

    // Both facts this needs exist only now: Phase 0 (inside pre_dequant) has
    // labelled host-resident NVFP4 experts, and the expert cache was sized in
    // init_weights(). Refusing here rather than at weight-upload time is the
    // point — see verify_host_expert_placement().
    executor_->verify_host_expert_placement();

    // KV now takes the MEASURED residual, not a predicted one. This can only
    // shrink the pool relative to the budget's projection, never grow it, so
    // it cannot overcommit; and it keeps the allocator's headroom free, which
    // is what everything allocated after this point needs to succeed.
    const size_t per_block_total_bytes =
        static_cast<size_t>(n_kv_layers) *
        kv_block_bytes_per_layer(config_.kv_cache_dtype, kv_bs, mcfg.n_kv_heads, head_dim);
    if (per_block_total_bytes > 0) {
        size_t free_now = 0, total_now = 0;
        vram_budget_mem_get_info(&free_now, &total_now);
        const size_t headroom = vram_allocator_headroom(total_now);
        const int max_blocks_planned = max_blocks;
        const auto sizing =
            kv_blocks_from_residual(free_now, headroom, per_block_total_bytes, max_blocks, 16);
        if (sizing.clamped) {
            IMP_LOG_INFO("KV cache: %d -> %d blocks from the measured post-cache residual "
                         "(%.0f MiB free, %.0f MiB allocator headroom kept)",
                         max_blocks, sizing.blocks, free_now / (1024.0 * 1024.0),
                         headroom / (1024.0 * 1024.0));
            max_blocks = sizing.blocks;
        }
        // The floor is a rescue, not a size: nothing was left to size the pool
        // from, so every request longer than the floor will be cancelled at
        // admission while the load still reports success. The hard failure
        // below only fires with an explicit --vram-budget, so without this the
        // default path learns about it from cancelled generations (#1251).
        if (sizing.floored) {
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
        // The quiet half of the same fault: the pool is a real size — not the
        // floor — and still holds less than one max_seq_len sequence, so the
        // load reports success and every full-length request is cancelled at
        // admission. With an explicit --vram-budget the check below turns this
        // into a hard failure; without one the path stays best-effort by
        // design, so it has to at least say so (#1251).
        //
        // Only for an operator-set max_seq_len. An AUTO value is a projection
        // that this clamp is *expected* to undercut — init_compute_max_seq_len_
        // sizes the GGUF path from raw free VRAM on purpose and leaves the
        // overshoot for exactly this clamp to absorb. Warning there would fire
        // on healthy loads and bury the case that is a fault.
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
                "--max-seq-len, lower the weight-cache demand (moe.reserve_mib, --kv-fp8), or "
                "free at least %.0f MiB for the KV pool.",
                sizing.blocks, have_mib, static_cast<double>(sizing.blocks) * kv_bs,
                config_.max_seq_len, need_blocks, need_mib, need_mib - have_mib);
        }
    }

    // I6, plan-time half: an explicit --vram-budget that cannot hold one
    // full-length sequence produces a process that loads fine and then cancels
    // every request ("needs 32 KV blocks but cache capacity is 16"). That is
    // the right runtime answer but the wrong time to learn it — the load has
    // already been paid for. Fail here, naming the arithmetic, so the operator
    // can raise the budget instead of reading scheduler errors.
    //
    // Only when a budget is installed. Without one this is the pre-existing
    // best-effort path and must keep its current behaviour.
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
                "%.0f MiB, or lower --max-seq-len.",
                vram_budget_bytes() >> 20, max_blocks, have_mib, config_.max_seq_len,
                blocks_per_seq, need_mib, need_mib - have_mib);
            return false;
        }
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
    std::unique_ptr<KVCache> kv_cache;
    if ((!mcfg.head_dim_per_layer.empty() || swa_sizing_active_) &&
        config_.kv_cache_dtype != QType::INT8 && config_.kv_cache_dtype != QType::INT4) {
        std::vector<int> per_layer_nkv(n_kv_layers, 0);
        std::vector<int> per_layer_hd(n_kv_layers, 0);
        std::vector<char> per_layer_swa(swa_sizing_active_ ? n_kv_layers : 0, 0);
        for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
            // Only attention layers get KV cache entries
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
        kv_cache = std::make_unique<KVCache>(n_kv_layers, per_layer_nkv, per_layer_hd, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &vram_alloc_, per_layer_swa,
                                             swa_sizing_active_ ? vram_budget.swa_max_blocks : 0);
    } else {
        kv_cache = std::make_unique<KVCache>(n_kv_layers, mcfg.n_kv_heads, head_dim, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &vram_alloc_);
    }
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));
    if (swa_sizing_active_) {
        kv_manager_->enable_swa_sizing(swa_window_max_, swa_slack_tokens_);
        swa_sizing_active_ = kv_manager_->swa_sizing_enabled();
    }

    // BitDecoding Phase 3: residual FP16 cache (opt-in).
    //
    // Ring state (write_idx / fill_count per slot) lives in device memory
    // (kv_manager_->d_residual_widx_ptr / d_residual_fc_ptr). Updated by a
    // tiny advance_residual_state_kernel at the end of forward_logits; the
    // residual write/read kernels read the state at execution time. This
    // makes the whole path graph-capture-safe — graphs stay enabled.
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
            }
        } else if (residual_n > 0) {
            IMP_LOG_INFO("kv_cache.bitdecoding_residual_tokens=%d ignored (only active with kv_cache_dtype=NVFP4)",
                         residual_n);
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
        // Persistent cache is dense-only: restored KV blocks are only usable
        // for hybrids together with a recurrent-state snapshot, and snapshots
        // are not persisted. (For hybrids the recurrent-snapshot store below
        // must also come up, or caching is turned back off.)
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

    // SSM state
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
            if (!ssm_state_->init(n_ssm, config_.max_batch_size, conv_ch, mcfg.ssm_conv_kernel, n_heads, hd,
                                  mcfg.ssm_state_size, config_.ssm_state_dtype, &vram_alloc_)) {
                IMP_LOG_WARN("Failed to init SSM state, continuing without it");
                ssm_state_.reset();
            }
        }

        // Recurrent-state snapshots: KV block reuse alone cannot skip prefill
        // for a recurrent model (the state at the skip boundary would be
        // zero), so hybrid prefix caching needs the snapshot store. Without
        // it, turn prefix caching back off — retaining hashed blocks that can
        // never be reused just churns the pool.
        if (kv_manager_->prefix_caching_enabled()) {
            int budget_mb = runtime_config_.server.recurrent_snapshot_mb;
            if (ssm_state_ && budget_mb > 0) {
                recurrent_snapshots_ = std::make_unique<RecurrentSnapshotStore>();
                recurrent_snapshots_->init(ssm_state_->per_seq_bytes(),
                                           static_cast<size_t>(budget_mb) << 20);
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

    // SWA window snapshots (kv_cache.swa_snapshot_mb): under SWA sizing,
    // global-layer KV blocks alone cannot back a prefix-cache hit — the
    // windowed layers' earlier blocks were trailing-freed, so the reused
    // prefix would leave their window as holes. The store keeps the packed
    // window at each prefill-end prefix hash and restores it at admission.
    // Without a working store the coexistence the gate allowed is invalid —
    // fall back to prefix caching off (mirrors the hybrid rule above).
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
            // Say WHY and WHAT TO DO. A budget below one snapshot silently costs
            // prefix caching — strictly worse than swa_snapshot_mb=0, which keeps
            // caching and yields the SWA savings instead. Without the required
            // size in the message there is no way to tell those apart from a log.
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

    // Coverage check: for prequant MoE models the nvfp4_moe decode cache is
    // all-or-nothing (predicate mirrors executor_forward_moe.cu
    // nvfp4_covers_layer) — one uncovered layer makes decode fall to the
    // host-args legacy path, which throws under CUDA-graph capture and
    // aborts the WHOLE decode graph. Surface partial coverage loudly.
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

    // Pre-allocate the gemm_nvfp4 fallback dequant workspace. Sized from
    // wcache_.nvfp4 which is populated by pre_dequant_weights above, so this
    // call must come AFTER. Lets the M>1 fallback path (used by future
    // multi-token verify / spec-decode) run inside CUDA stream capture
    // without crashing on cudaMalloc.
    (void)executor_->allocate_nvfp4_dequant_workspace();
    // Build the CUTLASS NVFP4 LM head for batched-decode tensor-core GEMM. Only
    // batched decode (n>1) consumes it, so skip the SfAtom-scale VRAM (19-47 MiB
    // depending on vocab×d_model) when max_batch_size can never reach n>1; the
    // perplexity harness lazy-builds it for its own measurement path. Must come
    // AFTER pre_dequant_weights so the NVFP4 decode cache exists, and before
    // decode-graph capture so the captured topology includes it.
    if (config_.max_batch_size > 1)
        executor_->build_lm_head_cutlass_(stream_);
    if (config_.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq,
                                /*with_swa_tables=*/swa_sizing_active_);
    {
        d_penalty_tokens_capacity_ = static_cast<size_t>(config_.max_seq_len);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_WARN("Failed to pre-allocate penalty token buffer");
            d_penalty_tokens_capacity_ = 0;
        }
    }

    // Pre-allocate prefill metadata pool (avoids per-request cudaMallocAsync)
    {
        size_t tok_bytes = config_.max_seq_len * sizeof(int32_t);
        size_t pos_bytes = config_.max_seq_len * sizeof(int);
        // A single request's block_table can grow to the entire KV cache
        // pool (max_blocks), not just max_seq_len/block_size. Size from
        // max_blocks so the H2D copy at the prefill metadata upload site
        // doesn't overflow on long-cumulative-KV requests.
        size_t bt_bytes = static_cast<size_t>(max_blocks) * sizeof(int);
        size_t swa_bt_bytes = swa_sizing_active_ ? bt_bytes : 0;
        size_t cl_bytes = sizeof(int);
        prefill_pool_size_ = tok_bytes + pos_bytes + bt_bytes + swa_bt_bytes + cl_bytes;
        prefill_pool_ = vram_alloc_.allocate(prefill_pool_size_, "prefill_pool");
        if (prefill_pool_) {
            auto* base = static_cast<char*>(prefill_pool_);
            d_pf_token_ids_ = reinterpret_cast<int32_t*>(base);
            d_pf_positions_ = reinterpret_cast<int*>(base + tok_bytes);
            d_pf_block_tables_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes);
            if (swa_sizing_active_)
                d_pf_block_tables_swa_ =
                    reinterpret_cast<int*>(base + tok_bytes + pos_bytes + bt_bytes);
            d_pf_context_lens_ =
                reinterpret_cast<int*>(base + tok_bytes + pos_bytes + bt_bytes + swa_bt_bytes);
        } else {
            IMP_LOG_WARN("Failed to pre-allocate prefill pool, will use per-request malloc");
        }

        // Pinned host staging buffers for prefill
        // T5b: an empty buffer means "no staging", which the prefill path already
        // tests for (memory/host_pinned.h).
        h_pf_positions_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                static_cast<size_t>(config_.max_seq_len) * sizeof(int));
        h_pf_token_ids_ = PinnedBuffer::acquire(
            cuda_host_pinned_allocator(),
            static_cast<size_t>(config_.max_seq_len) * sizeof(int32_t));
        if (cudaEventCreateWithFlags(&pf_staging_evt_, cudaEventDisableTiming) != cudaSuccess)
            pf_staging_evt_ = nullptr;
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


}  // namespace imp

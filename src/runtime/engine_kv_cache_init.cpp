// Engine init phase: paged KV cache allocation. Declaration in engine.h.

#include "runtime/engine.h"
#include "memory/recurrent_snapshot_store.h"
#include "runtime/engine_internal.h"
#include "runtime/kv_fp8_scale_init.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "memory/vram_query.h"
#include "memory/library_reserve_cache.h"
#include "memory/plan.h"
#include "memory/ssm_state_size.h"
#include "exec/expert_batch_policy.h"
#include "runtime/plan_shadow.h"
#include "runtime/prompt_tail.h"  // base_prefill_chunk
#include "runtime/snapshot_host_default.h"
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "core/logging.h"
#include "compute/mmq_q8_imma.h"  // mmq_q8_imma_set_plane_budget
#include "compute/sampling.h"     // SAMPLE_SCRATCH_BYTES (constrained-pipeline region)
#include "runtime/serving_metadata_layout.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace imp {

namespace {

// The planes are optional (weights past the cap prefill via dequant), one max_seq_len sequence is
// not: the plan capped them against its projection, the measured residual is lower by the allocator
// headroom and cache estimates (856 of 1333 blocks, Qwen3-4B Q8_0 at --vram-budget 9000).
// Returns the plane bytes still outstanding after giving back what `need` requires.
size_t yield_imma_planes_to_one_sequence(VRAMBudget& budget, size_t used, size_t free_now, size_t need,
                                         int max_seq_len) {
    size_t outstanding = budget.imma_plane_bytes > used ? budget.imma_plane_bytes - used : 0;
    const size_t have = free_now > outstanding ? free_now - outstanding : 0;
    if (outstanding == 0 || have >= need)
        return outstanding;
    const size_t give = std::min(outstanding, need - have);
    budget.imma_plane_bytes -= give;
    mmq_q8_imma_set_plane_budget(budget.imma_plane_bytes);
    IMP_LOG_INFO(
        "KV cache: IMMA prefill planes cut by %.0f MiB to %.0f MiB so one max_seq_len=%d "
        "sequence fits the measured residual",
        give / (1024.0 * 1024.0), budget.imma_plane_bytes / (1024.0 * 1024.0), max_seq_len);
    return outstanding - give;
}

// AUTO max_seq_len = min(resolver bound, what the built pool holds; a growable pool its ceiling).
// The resolver runs before weights and caches exist, so its value is only an upper bound.
void clamp_auto_max_seq_len_to_pool(int& max_seq_len, int pool_blocks, int kv_bs) {
    const int pool_tokens = pool_blocks * kv_bs;
    if (pool_tokens <= 0 || pool_tokens >= max_seq_len)
        return;
    IMP_LOG_INFO("max_seq_len: auto %d -> %d (KV pool %d blocks x %d)", max_seq_len, pool_tokens, pool_blocks,
                 kv_bs);
    max_seq_len = pool_tokens;
}

// Engine state the init_kv_cache phase helpers read; `ec` is written where a phase turns a feature
// off (prefix caching, StreamingLLM auto, FP8 prefill) or loads the remembered library reserve;
// `rc` only by snapshot_host_budget_bytes (its host-tier default).
struct KvInitCtx {
    Model& model;
    const ModelConfig& mcfg;
    EngineConfig& ec;
    RuntimeConfig& rc;
};

// Attention-layer geometry every sizing phase reads, fixed once the SWA gate has run.
struct KvShape {
    std::vector<int> layer_map;  // model layer -> KV slot, -1 = owns none
    int n_kv_layers = 0;
    int head_dim = 0;
    int kv_bs = 0;
    int n_swa_layers = 0;
    int swa_live_tokens = 0;
    bool swa_active = false;            // the gate's verdict; enable_swa_sizing may still drop it
    size_t layer_block_bytes = 0;       // kv_block_bytes_layer_mean() of one layer
    size_t minmax_bytes_per_layer = 0;  // sparse decode key min/max metadata per block
};

std::vector<int> map_kv_layers(const Model& model, int& n_attn_layers) {
    const auto& mcfg = model.config();
    n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model.layer(i).wq.data == nullptr || model.layer(i).gdn_gate.data != nullptr)
            continue;
        // Gemma-4 KV sharing: the shared layer reads its source's slot and owns none.
        const int src = mcfg.kv_source(i);
        kv_layer_map[i] = (src >= 0 && src < i) ? kv_layer_map[src] : n_attn_layers++;
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;
        for (int i = 0; i < mcfg.n_layers; i++)
            kv_layer_map[i] = i;
    }
    IMP_LOG_INFO("KV cache layers: %d attention out of %d total", n_attn_layers, mcfg.n_layers);
    return kv_layer_map;
}

const char* swa_sizing_off_reason(const KvInitCtx& c, int n_swa_layers, int n_kv_layers, int window_max) {
    const auto& prof = c.model.profile();
    if (n_swa_layers == 0)
        return "model has no sliding-window layers";
    if (n_swa_layers == n_kv_layers && window_max >= c.ec.max_seq_len)
        return "window >= max_seq_len (nothing to save)";
    if (c.ec.kv_cache_dtype == QType::INT8 || c.ec.kv_cache_dtype == QType::INT4)
        return "INT8/INT4 KV lacks the per-layer cache path";
    if (prof.is_ssm || prof.is_gdn)
        return "hybrid recurrent model (conservative)";
    if (c.mcfg.is_mla())
        return "MLA attention";
    if (c.ec.streaming_kv_enabled)
        return "StreamingLLM is enabled";
    if (c.rc.runtime.deterministic)
        return "deterministic mode (unbounded graph loop would be burst-chunked)";
    if (!c.rc.kv_cache.swa_sizing_wanted(c.ec.use_prefix_caching))
        return "auto mode yields to prefix caching (freed window blocks cannot back "
               "prefix reuse; set kv_cache.swa_snapshot_mb to combine, or "
               "kv_cache.swa_sizing=on to force the KV savings)";
    return nullptr;
}

struct SwaGate {
    bool active = false;
    int window_max = 0;
    int n_layers = 0;
    int slack_tokens = 0;
    int burst_cap_tokens = 0;
    int chunk_extra_tokens = 0;
};

// SWA-aware KV sizing gate (kv_cache.swa_sizing): sliding-window layers get
// a small dedicated block group instead of full-length KV. Resolved before
// the VRAM budget so it charges window-cost, not context-cost.
SwaGate resolve_swa_sizing(const KvInitCtx& c, const std::vector<int>& kv_layer_map, int n_kv_layers,
                           int kv_bs, const GraphExecutor* executor,
                           const std::function<int()>& resolve_chunk) {
    SwaGate g;
    if (c.rc.kv_cache.swa_sizing_mode() == SwaSizingMode::Off)
        return g;
    for (int i = 0; i < c.mcfg.n_layers; i++) {
        if (kv_layer_map[i] < 0 || c.mcfg.layer_kv_shared(i))
            continue;  // non-attention or KV-shared layer (no slot of its own)
        int w = layer_swa_window(c.mcfg, c.model.profile(), i);
        if (w > 0) {
            g.n_layers++;
            g.window_max = std::max(g.window_max, w);
        }
    }
    if (const char* off_reason = swa_sizing_off_reason(c, g.n_layers, n_kv_layers, g.window_max)) {
        IMP_LOG_INFO("kv_cache.swa_sizing=%s ignored: %s", c.rc.kv_cache.swa_sizing.c_str(), off_reason);
        return SwaGate{};
    }
    g.active = true;
    // Slack must cover the deepest speculative rollback (verify
    // chunks roll back rejected drafts) plus the partial boundary
    // block. Sized from the spec config so the assert can't trip.
    const auto& sc = c.rc.speculative;
    int spec_depth = std::max({cfg::kSpecDraftK, cfg::kSpecSuffixKMax, sc.mtp_k + 1});
    g.slack_tokens = std::max(2 * kv_bs, spec_depth + kv_bs);
    // Per sequence: the on-device decode burst (graph loop clamp). Once per pool: the
    // prefill chunk the scheduler runs (resolver incl. --set runtime.prefill_chunk_size,
    // executor cap, block floor); serial prefill trims each chunk on enqueue.
    const int max_rows = executor ? executor->max_tokens() : 0;
    const int resolved = resolve_chunk();
    int chunk_peak = max_rows > 0 ? base_prefill_chunk(resolved, max_rows, kv_bs) : resolved;
    if (chunk_peak <= 0)
        chunk_peak = 2048;
    const int burst_peak = kDecodeBurst;
    g.burst_cap_tokens = burst_peak;
    g.chunk_extra_tokens = std::max(0, chunk_peak - burst_peak);
    // Prefix caching cannot reuse freed window blocks. With a SWA
    // snapshot budget the two coexist (window restored at reuse
    // boundary); otherwise honor the forced opt-in by disabling it.
    if (c.ec.use_prefix_caching && c.rc.kv_cache.swa_snapshot_mb <= 0) {
        c.ec.use_prefix_caching = false;
        IMP_LOG_INFO(
            "kv_cache.swa_sizing=on: prefix caching disabled (freed window "
            "blocks cannot back prefix reuse; set kv_cache.swa_snapshot_mb "
            "to combine)");
    }
    // StreamingLLM auto-enable frees middle blocks of the GLOBAL
    // table, redundant and conflicting here.
    c.ec.streaming_kv_auto = false;
    return g;
}

// library_reserve_mb caches the first forward's actual claim (stable per model/quant/library);
// explicit config wins. key_out/path_out name the entry the first forward records.
void load_library_reserve(const KvInitCtx& c, const std::function<uint64_t()>& fingerprint,
                          LibraryReserveKey& key_out, std::string& path_out) {
    if (c.ec.library_reserve_mb >= 0 || c.rc.vram.library_reserve_cache == "off")
        return;
    const std::string path = c.rc.vram.library_reserve_cache.empty() ? library_reserve_cache_default_path()
                                                                     : c.rc.vram.library_reserve_cache;
    LibraryReserveKey key;
    key.model_fingerprint = fingerprint();
    key.nvfp4_decode_mode = c.ec.use_nvfp4_decode;
    key.fp8_prefill = c.ec.use_fp8_prefill;
    // Failure keeps the default 0: matches no cache entry written by a real runtime.
    IMP_CUDA_CHECK_LOG(cudaRuntimeGetVersion(&key.cuda_runtime_version));
    key_out = key;
    path_out = path;
    bool remembered_found = false;
    const size_t remembered = library_reserve_cache_load(path, key, &remembered_found);
    if (remembered_found) {
        // A recorded ZERO reserve is valid (not "missing") - models whose
        // first forward claims nothing (AUDIT B70).
        c.ec.library_reserve_mb = static_cast<int>(remembered >> 20);
        IMP_LOG_INFO(
            "library reserve: %d MiB from the measurement cache (%s) — the default "
            "constant is %zu MiB",
            c.ec.library_reserve_mb, path.c_str(), kMeasuredLibraryReserveBytes >> 20);
    } else if (!path.empty()) {
        // No entry: log the constant charge HERE, before pools are sized -
        // reporting only after the first forward would be too late, the KV
        // pool would already be sized around an unwanted reserve.
        IMP_LOG_INFO(
            "library reserve: no measurement for this model in %s — planning with the "
            "%zu MiB constant. It is recorded after the first forward; mount that "
            "path (or set vram.library_reserve_cache) to keep it across restarts.",
            path.c_str(), kMeasuredLibraryReserveBytes >> 20);
    }
}

// Sparse decode key min/max pool: allocated after the sizing from the same block count, so
// the plan prices it per block (#1103 silent-spill class, #2360). Same rule as the enable gate
// below; per-layer geometry and SWA pools refuse it in enable_key_minmax.
size_t sparse_minmax_plan_bytes(const KvInitCtx& c, int head_dim, bool swa_active) {
    const auto& at = c.rc.attention;
    if ((at.sparse_topk_tokens > 0 || at.sparse_prefill_topk_tokens > 0) &&
        !sparse_minmax_refusal(c.ec.kv_cache_dtype, c.mcfg.is_mla() && at.mla_absorb,
                               c.rc.speculative.token_recycling, !c.ec.prefix_cache_path.empty()) &&
        c.mcfg.head_dim_per_layer.empty() && !swa_active)
        return KVCache::minmax_block_bytes(c.mcfg.n_kv_heads, head_dim);
    return 0;
}

// The recurrent snapshot store cudaMallocs server.recurrent_snapshot_mb
// AFTER KV sizing, so charge it here: whole slots of one sequence's
// state, never more than the budget.
size_t recurrent_snapshot_plan_bytes(const KvInitCtx& c, size_t ssm_footprint_bytes, int ssm_reserved_slots) {
    if (c.mcfg.ssm_inner_size <= 0 || !c.ec.use_prefix_caching || c.rc.server.recurrent_snapshot_mb <= 0 ||
        ssm_footprint_bytes == 0)
        return 0;
    const size_t slots = static_cast<size_t>(c.ec.max_batch_size) +
                         static_cast<size_t>(std::max(0, ssm_reserved_slots));
    const size_t per_seq = slots > 0 ? ssm_footprint_bytes / slots : 0;
    const size_t budget = static_cast<size_t>(c.rc.server.recurrent_snapshot_mb) << 20;
    // Whole slots only, stopped at budget: device charge is
    // floor(budget/per_seq)*per_seq, not raw budget. Host tier
    // (server.recurrent_snapshot_host_mb) is pinned HOST memory, not VRAM.
    return per_seq > 0 ? (budget / per_seq) * per_seq : 0;
}

// plan_memory() input. `distributable` and `spec_decode_bytes` are read by the caller in this order.
ShadowPlanProbe build_shadow_probe(const KvInitCtx& c, const VRAMBudget& vb, const KvShape& kv,
                                   const GraphExecutor* executor, size_t distributable,
                                   size_t spec_decode_bytes, size_t mtp_workspace_bytes,
                                   int ssm_reserved_slots) {
    ShadowPlanProbe probe;
    probe.distributable_bytes = distributable;
    // Steady-state demand only: transient init headroom is free again by
    // pool-sizing time. Charging it starved optional caches to a 128-block
    // floor while GiBs sat idle (#1765).
    probe.weight_cache_demand = vb.weight_cache_estimate_bytes > vb.weight_cache_transient_bytes
                                    ? vb.weight_cache_estimate_bytes - vb.weight_cache_transient_bytes
                                    : 0;
    // The IMMA prefill planes are mandatory in the same sense: the plan
    // grants them, mmq_q8_imma is capped at the grant, and the pool must
    // not be sized over bytes the first prefill will take (#1899).
    probe.mandatory_cache_bytes = vb.mandatory_sf_bytes + vb.mandatory_moe_bytes + vb.imma_plane_bytes;
    probe.ssm_state_bytes = vb.ssm_footprint_bytes;
    probe.recurrent_snapshot_bytes = recurrent_snapshot_plan_bytes(c, vb.ssm_footprint_bytes,
                                                                   ssm_reserved_slots);
    // Hybrid verify state scratch (#2452): 2 x one sequence's recurrent state, taken at init.
    probe.spec_decode_bytes = spec_decode_bytes;
    probe.mtp_workspace_bytes = mtp_workspace_bytes;
    if (executor) {
        // allocate_workspaces() ran before distributable was read: charge only the part of the
        // estimate not yet resident (Qwen3.8-27B: 887 estimated, 749 held, #2365).
        const size_t est = executor->workspace_estimate();
        const size_t held = executor->workspace_allocated();
        probe.engine_persistent_bytes = est > held ? est - held : 0;
        probe.engine_persistent_resident_bytes = held;
        probe.workspace_estimate_available = true;
    }
    probe.kv_growable_ceiling_blocks = c.rc.kv_cache.growable ? vb.kv_max_blocks : 0;
    probe.vision_tower_unmodelled = !c.ec.mmproj_path.empty();
    // Use config_.library_reserve_mb, NOT the runtime-config field: the
    // loader above writes the remembered measurement into the former only
    // (AUDIT B70).
    probe.library_reserve_bytes = c.ec.library_reserve_mb < 0
                                      ? kMeasuredLibraryReserveBytes
                                      : static_cast<size_t>(c.ec.library_reserve_mb) << 20;
    probe.n_kv_layers = kv.n_kv_layers;
    probe.n_swa_layers = kv.n_swa_layers;
    probe.swa_live_tokens = kv.swa_live_tokens;
    probe.swa_shared_tokens = c.ec.swa_shared_tokens;
    probe.max_batch_size = c.ec.max_batch_size;
    probe.kv_seqs = admitted_batch(c.ec.max_batch_size, c.rc.runtime.max_batch_size);
    probe.max_seq_len = c.ec.max_seq_len;
    probe.kv_block_size = kv.kv_bs;
    probe.min_kv_tokens = c.ec.min_kv_tokens;
    probe.kv_block_bytes_per_layer = kv.layer_block_bytes;
    probe.kv_meta_block_bytes_per_layer = kv.minmax_bytes_per_layer;
    return probe;
}

// KV block count comes from plan_memory(), not the live-free-derived pass
// (B62/B65/B66). The residual clamp below can only shrink it further, never
// grow it, so a plan wrong about the device cannot overcommit.
int kv_blocks_from_plan(const EngineConfig& ec, const PlanResult& plan, const VRAMBudget& vb) {
    if (ec.kv_cache_max_blocks > 0)
        return ec.kv_cache_max_blocks;  // operator pin wins over both
    if (plan.ok) {
        const int max_blocks = plan.plan.kv.blocks;
        if (max_blocks != vb.kv_max_blocks) {
            // Plan and live pass may legitimately differ (plan charges what
            // the live read cannot see) - logged so divergence isn't silent.
            // Live figure is pre-floor: kv_max_blocks may already be raised by min_kv_tokens (#1747).
            const int live_raw = vb.kv_blocks_pre_floor > 0 ? vb.kv_blocks_pre_floor : vb.kv_max_blocks;
            if (live_raw != vb.kv_max_blocks) {
                IMP_LOG_INFO("KV blocks: plan %d (live pass sized %d, raised to %d by min_kv_tokens)",
                             max_blocks, live_raw, vb.kv_max_blocks);
            } else {
                IMP_LOG_INFO("KV blocks: plan %d (live pass sized %d)", max_blocks, live_raw);
            }
        }
        return max_blocks;
    }
    // Plan refuses this configuration: fails the load only when an
    // explicit --vram-budget is installed (check further down, D8).
    // Otherwise falls back to the live pass (best-effort, not a refusal).
    IMP_LOG_WARN(
        "KV blocks: the plan rejects this configuration — the pool ceiling falls back to "
        "the live pass's %d blocks (the 'KV cache:' lines below report what is committed "
        "and usable). The report above says what the plan could not fit.",
        vb.kv_max_blocks);
    return vb.kv_max_blocks;
}

// The plan sized auto pools with the metadata per block; a pinned pool is the operator's count.
void log_sparse_minmax_pool(const EngineConfig& ec, const KvShape& kv, int max_blocks) {
    if (kv.minmax_bytes_per_layer == 0)
        return;
    const double mm_mib = static_cast<double>(kv.minmax_bytes_per_layer) * kv.n_kv_layers * max_blocks /
                          (1024.0 * 1024.0);
    if (ec.kv_cache_max_blocks > 0) {
        IMP_LOG_WARN(
            "attention.sparse_topk_tokens: the key min/max pool adds %.1f MiB ON TOP "
            "of the pinned kv_cache.max_blocks pool - pin with that headroom or "
            "WSL2/WDDM spills silently",
            mm_mib);
    } else {
        IMP_LOG_INFO(
            "sparse decode attention: key min/max pool %.1f MiB for %d blocks, priced in the KV "
            "plan (%.2f%% of the K+V pool)",
            mm_mib, max_blocks,
            kv.layer_block_bytes > 0 ? 100.0 * kv.minmax_bytes_per_layer / kv.layer_block_bytes : 0.0);
    }
}

void apply_gdn_precision_gates(const KvInitCtx& c, GraphExecutor* executor) {
    if (!c.model.profile().is_gdn)
        return;
    if (c.ec.use_cuda_graphs) {
        IMP_LOG_INFO("GDN model: CUDA graphs enabled (recurrent state in-place)");
    } else {
        IMP_LOG_INFO("GDN model: CUDA graphs disabled (disabled earlier by caller or expert offload)");
    }
    // GDN recurrent state accumulates precision errors per token. FP8
    // E4M3's 3-bit mantissa amplifies these through the delta-rule scan,
    // causing degenerate output after ~50 tokens in multi-turn chat.
    if (c.ec.use_fp8_prefill) {
        if (c.ec.dual_path_quant) {
            IMP_LOG_WARN(
                "GDN + dual-path: attention weights forced to FP16 (not FP8) — "
                "recurrent state needs FP16 precision. FFN weights still use NVFP4.");
        } else {
            IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
        }
        c.ec.use_fp8_prefill = false;
        executor->disable_fp8_prefill();
    }
}

void warn_host_expert_batch(const EngineConfig& ec, const GraphExecutor& executor, bool experts_on_host) {
    if (const std::string w = host_expert_batch_warning(ec.max_batch_size, experts_on_host,
                                                        executor.device_expert_cache_tiered(),
                                                        executor.expert_cache_bytes(),
                                                        executor.expert_cache_slots_per_layer());
        !w.empty())
        IMP_LOG_WARN("%s", w.c_str());
}

// KV takes the MEASURED residual, not a predicted one: this can only shrink
// the pool relative to the budget's projection, never grow it, so it cannot
// overcommit, and keeps the allocator headroom free for later allocations.
int size_kv_from_residual(VRAMBudget& vb, const EngineConfig& ec, const KvShape& kv,
                          size_t per_block_total_bytes, int max_blocks, int kv_blocks_planned,
                          bool max_seq_len_explicit, bool& pool_floored) {
    if (per_block_total_bytes == 0)
        return max_blocks;
    const int kv_bs = kv.kv_bs;
    size_t free_now = 0, total_now = 0;
    // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
    (void)vram_budget_mem_get_info(&free_now, &total_now);
    // IMMA prefill planes are still OUTSTANDING here (taken on each Q8_0
    // weight's first prefill, during warmup). Charging them here keeps the
    // residual pass from handing the pool bytes the next forward claims (#1899).
    const size_t imma_used = mmq_q8_imma_plane_bytes_used();
    const size_t imma_outstanding = yield_imma_planes_to_one_sequence(
        vb, imma_used, free_now,
        (static_cast<size_t>(kv_blocks_per_sequence(ec.max_seq_len, kv_bs)) * per_block_total_bytes) +
            vram_allocator_headroom(total_now),
        ec.max_seq_len);
    if (imma_outstanding > 0) {
        IMP_LOG_INFO(
            "KV cache: holding %.0f MiB of the post-cache residual for the IMMA "
            "prefill planes (taken on the first prefill)",
            imma_outstanding / (1024.0 * 1024.0));
        free_now = free_now > imma_outstanding ? free_now - imma_outstanding : 0;
    }
    const size_t headroom = vram_allocator_headroom(total_now);
    const int max_blocks_planned = kv_blocks_planned;
    const auto sizing = kv_blocks_from_residual(free_now, headroom, per_block_total_bytes, max_blocks, 16);
    if (sizing.clamped) {
        IMP_LOG_INFO(
            "KV cache: %d -> %d blocks from the measured post-cache residual "
            "(%.0f MiB free, %.0f MiB allocator headroom kept)",
            max_blocks, sizing.blocks, free_now / (1024.0 * 1024.0), headroom / (1024.0 * 1024.0));
        max_blocks = sizing.blocks;
    }
    // The floor is a rescue, not a size: requests longer than it will be
    // cancelled at admission even though the load reports success. The hard
    // failure below fires only with an explicit --vram-budget (#1251).
    if (sizing.floored) {
        // Kept, not just logged: the server has to be able to answer for
        // this after the log line has scrolled away.
        pool_floored = true;
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
    // free VRAM on purpose, then clamped to the built pool), so warning there would bury real faults.
    if (vram_budget_bytes() == 0 && max_seq_len_explicit &&
        kv_pool_verdict(sizing, ec.max_seq_len, kv_bs) == KvPoolVerdict::ShortOfOneSequence) {
        const int need_blocks = kv_blocks_per_sequence(ec.max_seq_len, kv_bs);
        const double need_mib = double(need_blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
        const double have_mib = double(sizing.blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
        IMP_LOG_WARN(
            "KV cache: the pool ends up at %d blocks (%.0f MiB, %.0f tokens) but the "
            "requested max_seq_len=%d needs %d blocks (%.0f MiB). Every full-length request "
            "will be cancelled at admission even though this load reports success. Lower "
            "runtime.max_seq_len (imp-cli: --max-seq-len; imp-server: --set "
            "runtime.max_seq_len=N, the flag is CLI-only and exits 1 there, #1681), lower "
            "the weight-cache demand (moe.reserve_mib, --kv-fp8), or "
            "free at least %.0f MiB for the KV pool.",
            sizing.blocks, have_mib, static_cast<double>(sizing.blocks) * kv_bs, ec.max_seq_len, need_blocks,
            need_mib, need_mib - have_mib);
    }
    return max_blocks;
}

// I6, plan-time half: an explicit --vram-budget too small for one
// full-length sequence must fail HERE (naming the arithmetic), not load
// successfully and cancel every request later.
// Only when a budget is installed - without one this stays the pre-existing best-effort path.
bool kv_pool_fits_vram_budget(const EngineConfig& ec, int kv_bs, size_t per_block_total_bytes, int max_blocks,
                              bool max_seq_len_explicit) {
    if (vram_budget_bytes() == 0 || per_block_total_bytes == 0)
        return true;
    const int blocks_per_req = kv_blocks_per_sequence(ec.max_seq_len, kv_bs);
    // Operator-set max_seq_len only: an AUTO one is clamped to the built pool below.
    if (!max_seq_len_explicit || max_blocks >= blocks_per_req)
        return true;
    const double need_mib = double(blocks_per_req) * double(per_block_total_bytes) / (1024.0 * 1024.0);
    const double have_mib = double(max_blocks) * double(per_block_total_bytes) / (1024.0 * 1024.0);
    IMP_LOG_ERROR(
        "--vram-budget %zu MiB is too small for this model: the KV pool ends up at %d "
        "blocks (%.0f MiB) but one max_seq_len=%d sequence needs %d blocks (%.0f MiB). "
        "Every request would be cancelled at admission. Raise --vram-budget by at least "
        "%.0f MiB, or lower runtime.max_seq_len (imp-server: --set "
        "runtime.max_seq_len=N; --max-seq-len is imp-cli only, #1681).",
        vram_budget_bytes() >> 20, max_blocks, have_mib, ec.max_seq_len, blocks_per_req, need_mib,
        need_mib - have_mib);
    return false;
}

// Returns the blocks to commit up front; a growable pool may start under the clamped count.
int growable_initial_blocks(const RuntimeConfig& rc, int max_blocks, int kv_ceiling_blocks, int kv_bs) {
    if (kv_ceiling_blocks > 0) {
        // Commit a fraction on purpose when the operator asked: the clamp above
        // answers "what fits" from a reading that is wrong in both directions on
        // this platform, so starting under it is the only way to stay resident.
        const int pct = std::clamp(rc.kv_cache.growable_initial_pct, 1, 100);
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
    return max_blocks;
}

void log_kv_pool_summary(const KvInitCtx& c, const KvShape& kv, int max_blocks) {
    QType kv_dtype = c.ec.kv_cache_dtype;
    size_t total_kv = static_cast<size_t>(kv.n_kv_layers) * max_blocks * kv.layer_block_bytes;
    IMP_LOG_INFO(
        "KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
        "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
        max_blocks, static_cast<double>(max_blocks) * kv.kv_bs,
        static_cast<double>(total_kv) / (1024.0 * 1024.0), dtype_name(kv_dtype), kv.n_kv_layers,
        c.mcfg.n_layers, c.mcfg.n_kv_heads, kv.head_dim, kv.kv_bs);
}

// Per-layer KV shape path (Gemma 4 dual attention geometry): build per-layer
// nkv/hd arrays restricted to attention layers (hybrid models may have non-attn layers).
// SWA sizing also requires the per-layer path (per-layer region capacities).
std::unique_ptr<KVCache> make_kv_cache(const KvInitCtx& c, const KvShape& kv, VRAMAllocator* alloc,
                                       int swa_max_blocks, int blocks, int ceiling) {
    const auto& mcfg = c.mcfg;
    const int n_kv_layers = kv.n_kv_layers;
    if ((!mcfg.head_dim_per_layer.empty() || kv.swa_active) && c.ec.kv_cache_dtype != QType::INT8 &&
        c.ec.kv_cache_dtype != QType::INT4) {
        std::vector<int> per_layer_nkv(n_kv_layers, 0);
        std::vector<int> per_layer_hd(n_kv_layers, 0);
        std::vector<char> per_layer_swa(kv.swa_active ? n_kv_layers : 0, 0);
        for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
            int attn_nkv = (l < (int)mcfg.n_kv_heads_per_layer.size()) ? mcfg.n_kv_heads_per_layer[l]
                                                                       : mcfg.n_kv_heads;
            if (kv.layer_map[l] < 0 || mcfg.layer_kv_shared(l))
                continue;  // non-attention (SSM/GDN) or KV-shared layer
            if (attn_nkv <= 0)
                attn_nkv = mcfg.n_kv_heads;
            per_layer_nkv[k] = attn_nkv;
            per_layer_hd[k] = (l < (int)mcfg.head_dim_per_layer.size() && mcfg.head_dim_per_layer[l] > 0)
                                  ? mcfg.head_dim_per_layer[l]
                                  : kv.head_dim;
            if (kv.swa_active)
                per_layer_swa[k] = layer_swa_window(mcfg, c.model.profile(), l) > 0 ? 1 : 0;
            k++;
        }
        return std::make_unique<KVCache>(n_kv_layers, per_layer_nkv, per_layer_hd, c.ec.kv_cache_dtype,
                                         blocks, kv.kv_bs, alloc, per_layer_swa,
                                         kv.swa_active ? swa_max_blocks : 0, ceiling);
    }
    return std::make_unique<KVCache>(n_kv_layers, mcfg.n_kv_heads, kv.head_dim, c.ec.kv_cache_dtype, blocks,
                                     kv.kv_bs, alloc, ceiling);
}

// #1662: a pool that does not fit is a smaller pool, not a dead process.
// Every sizing input above is a projection and can still be wrong (#1631,
// #1662); halving down to the 16-block floor is the backstop.
// Only the pool retries - weight caches are already built and the model's
// source tensors may be consumed, so a retry one level up (imp_context_create)
// cannot rebuild this engine.
std::unique_ptr<KVCache> allocate_kv_pool(const KvInitCtx& c, const KvShape& kv, VRAMAllocator* alloc,
                                          int swa_max_blocks, int kv_ceiling_blocks, int& max_blocks,
                                          int& kv_ceiling_effective) {
    std::unique_ptr<KVCache> kv_cache;
    constexpr int kKVFloorBlocks = 16;
    const int planned = max_blocks;
    int attempt_blocks = max_blocks;
    int attempt_ceiling = kv_ceiling_blocks;
    for (;;) {
        try {
            kv_cache = make_kv_cache(c, kv, alloc, swa_max_blocks, attempt_blocks, attempt_ceiling);
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
            const double short_mib = static_cast<double>(kv.n_kv_layers) * (attempt_blocks - next) *
                                     kv.layer_block_bytes / (1024.0 * 1024.0);
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
                     static_cast<double>(max_blocks) * kv.kv_bs, planned);
    }
    return kv_cache;
}

// A successful allocation proves nothing on WSL2/WDDM: the pool can sit in
// host memory at a sixth of the bandwidth with /health ok and nothing logged
// (#1103, AUDIT_ARCH_2026 B-6). One cheap copy inside the fresh pool, then a
// WARN and a gauge, never a refusal - the threshold is one driver on one card.
void log_kv_pool_residency(KVCache& kv_cache) {
    const double gbps = kv_cache.probe_residency();
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

// attention.sparse_topk_tokens: per-block key mean/std metadata pool. Gate rule:
// sparse_minmax_refusal (shared with the plan charge above); enable_key_minmax also
// refuses per-layer geometry; a growable pool grows the metadata with it. A refusal disables
// the feature loudly.
void enable_sparse_key_minmax(const KvInitCtx& c, KVCache& kv_cache) {
    const auto& at = c.rc.attention;
    if (at.sparse_topk_tokens <= 0 && at.sparse_prefill_topk_tokens <= 0)
        return;
    const char* refuse = sparse_minmax_refusal(c.ec.kv_cache_dtype, c.mcfg.is_mla() && at.mla_absorb,
                                               c.rc.speculative.token_recycling,
                                               !c.ec.prefix_cache_path.empty());
    if (refuse) {
        IMP_LOG_WARN("attention.sparse_topk_tokens=%d / sparse_prefill_topk_tokens=%d ignored: %s",
                     at.sparse_topk_tokens, at.sparse_prefill_topk_tokens, refuse);
    } else if (!kv_cache.enable_key_minmax()) {
        IMP_LOG_WARN(
            "attention.sparse_topk_tokens=%d / sparse_prefill_topk_tokens=%d ignored: metadata pool "
            "unavailable (per-layer KV geometry or allocation failure)",
            at.sparse_topk_tokens, at.sparse_prefill_topk_tokens);
    }
}

void enable_prefix_caching(const KvInitCtx& c, KVCacheManager& kvm,
                           const std::function<uint64_t()>& fingerprint, cudaStream_t stream) {
    if (!c.ec.use_prefix_caching)
        return;
    kvm.set_prefix_caching_enabled(true);
    // cache_control/cache_prompt pin budget: percent of the pool,
    // floor of 1 block when enabled at all.
    int pin_pct = std::min(std::max(c.ec.prefix_pin_budget_pct, 0), 100);
    int pin_budget = pin_pct > 0 ? std::max(1, kvm.kv_cache()->total_blocks() * pin_pct / 100) : 0;
    kvm.set_pin_budget_blocks(pin_budget);
    IMP_LOG_INFO("Prefix caching enabled (pin budget %d blocks)", pin_budget);
    // Persistent cache is dense-only: restored KV blocks are usable for
    // hybrids only together with a recurrent-state snapshot, and snapshots
    // are not persisted (the recurrent-snapshot store below must also come up).
    if (c.mcfg.ssm_inner_size == 0 && !c.ec.prefix_cache_path.empty() &&
        prefix_cache_restore_allowed(c.ec.kv_cache_dtype, c.rc)) {
        int restored = kvm.load_prefix_cache(c.ec.prefix_cache_path,
                                             prefix_cache_fingerprint(fingerprint(), c.ec.kv_cache_dtype,
                                                                      c.rc),
                                             stream);
        if (restored > 0)
            IMP_LOG_INFO("Restored %d prefix cache blocks from %s", restored, c.ec.prefix_cache_path.c_str());
    }
}

// The Engine members the recurrent-state phase builds.
struct RecurrentStateOut {
    std::unique_ptr<SSMState>& ssm_state;
    int*& d_seq_slots;
    std::vector<int>& h_seq_slots;
    std::unique_ptr<RecurrentSnapshotStore>& snapshots;
};

void init_ssm_state_pool(const KvInitCtx& c, int n_ssm, int ssm_reserved_slots, VRAMAllocator* alloc,
                         Scheduler* scheduler, Scheduler::AdmissionGateFn gate, RecurrentStateOut& out) {
    const auto& mcfg = c.mcfg;
    int conv_ch = mcfg.ssm_conv_channels();
    int n_heads = mcfg.ssm_dt_rank;
    int hd = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
    out.ssm_state = std::make_unique<SSMState>();
    // Lazy: reserve every slot, commit one per admitted sequence
    // (scheduler admission gate below). The plan above charged the
    // whole slab either way.
    const bool ssm_pool_ok = out.ssm_state->init(
        n_ssm, c.ec.max_batch_size, conv_ch, mcfg.ssm_conv_kernel, n_heads, hd, mcfg.ssm_state_size,
        c.ec.ssm_state_dtype, alloc, ssm_reserved_slots, c.rc.vram.lazy_commit ? vmm_backend() : nullptr,
        batch_verify_spare_slots(c.rc, &c.model, c.ec.max_batch_size), c.model.ple_state_bytes());
    if (ssm_pool_ok && out.ssm_state->lazy() && scheduler)
        scheduler->set_admission_gate(std::move(gate));
    if (must_refuse_without_ssm_state(n_ssm, ssm_pool_ok)) {
        // NOT "continuing without it": a GDN/SSM layer with a missing
        // recurrent state reads a null slab and produces garbage for
        // every request. SSMState::init already logged bytes/slots/lever.
        out.ssm_state.reset();
        throw std::runtime_error("SSM/GDN state pool allocation failed and this model has " +
                                 std::to_string(n_ssm) +
                                 " recurrent layers; refusing to serve without it (see the SSM/GDN state "
                                 "pool line above for the shortfall and the lever)");
    } else if (ssm_reserved_slots > 0) {
        IMP_LOG_INFO("SSM state: %d verify spare slots past batch %d (lazy), %.1f MiB each",
                     ssm_reserved_slots, c.ec.max_batch_size, out.ssm_state->per_seq_bytes() / 1048576.0);
    }
    // Slot table for batched GDN decode. Allocated once and kept at a
    // stable address so a captured decode graph does not have to be
    // rebuilt when the set of active sequences changes.
    if (out.ssm_state && c.ec.max_batch_size > 0) {
        const size_t bytes = static_cast<size_t>(c.ec.max_batch_size) * sizeof(int);
        if (cudaMalloc(&out.d_seq_slots, bytes) != cudaSuccess) {
            IMP_LOG_WARN("batched GDN decode: slot table alloc failed — staying single-sequence");
            out.d_seq_slots = nullptr;
        } else {
            // Log only: the scheduler uploads the table before every batched step.
            IMP_CUDA_CHECK_LOG(cudaMemset(out.d_seq_slots, 0, bytes));
            out.h_seq_slots.assign(static_cast<size_t>(c.ec.max_batch_size), 0);
        }
    }
}

// Recurrent-state snapshots: KV block reuse alone cannot skip prefill
// for a recurrent model (state at the skip boundary would be zero), so
// hybrid prefix caching needs the snapshot store, or it must turn back off.
void init_recurrent_snapshots(const KvInitCtx& c, KVCacheManager& kvm, Scheduler* scheduler,
                              Scheduler::PrefixReuseLimitFn reuse_limit, RecurrentStateOut& out) {
    if (!kvm.prefix_caching_enabled())
        return;
    int budget_mb = c.rc.server.recurrent_snapshot_mb;
    if (out.ssm_state && budget_mb > 0) {
        out.snapshots = std::make_unique<RecurrentSnapshotStore>();
        out.snapshots->init(out.ssm_state->per_seq_bytes(), static_cast<size_t>(budget_mb) << 20,
                            snapshot_host_budget_bytes(c.rc));
        if (out.snapshots->enabled()) {
            scheduler->set_prefix_reuse_limit(std::move(reuse_limit));
        } else {
            out.snapshots.reset();
        }
    }
    if (!out.snapshots) {
        kvm.set_prefix_caching_enabled(false);
        IMP_LOG_INFO(
            "Prefix caching disabled for recurrent model (snapshot store off — "
            "server.recurrent_snapshot_mb=%d)",
            budget_mb);
    }
}

void init_recurrent_state(const KvInitCtx& c, KVCacheManager& kvm, int ssm_reserved_slots,
                          VRAMAllocator* alloc, Scheduler* scheduler, Scheduler::AdmissionGateFn gate,
                          Scheduler::PrefixReuseLimitFn reuse_limit, RecurrentStateOut out) {
    if (c.mcfg.ssm_inner_size <= 0)
        return;
    int n_ssm = 0;
    for (int i = 0; i < c.mcfg.n_layers; i++)
        if (c.model.layer(i).ssm_in.data != nullptr)
            n_ssm++;
    if (n_ssm > 0)
        init_ssm_state_pool(c, n_ssm, ssm_reserved_slots, alloc, scheduler, std::move(gate), out);
    // PLE conv rows live in the SSM slab tail (executor_ple.cpp): no slab, no PLE state.
    if (const size_t ple = c.model.ple_state_bytes();
        ple > 0 && (!out.ssm_state || out.ssm_state->extra_bytes() < ple))
        throw std::runtime_error("PLE model without the per-slot conv rows in the SSM/GDN state slab");
    init_recurrent_snapshots(c, kvm, scheduler, std::move(reuse_limit), out);
}

// kv_cache.swa_snapshot_mb: under SWA sizing, global-layer KV blocks alone
// cannot back a prefix-cache hit (windowed layers' earlier blocks were
// trailing-freed, leaving holes). The store keeps the packed window at each
// prefill-end prefix hash and restores it at admission; without a working store, fall back to prefix caching
// off.
void init_swa_snapshots(const KvInitCtx& c, bool swa_active, KVCacheManager& kvm, Scheduler* scheduler,
                        Scheduler::PrefixReuseLimitFn reuse_limit,
                        std::unique_ptr<RecurrentSnapshotStore>& store, void*& slab) {
    if (!swa_active || !kvm.prefix_caching_enabled())
        return;
    const int budget_mb = c.rc.kv_cache.swa_snapshot_mb;
    if (budget_mb > 0 && kvm.enable_swa_snapshots()) {
        const size_t slab_bytes = kvm.swa_snapshot_bytes();
        if (cudaMalloc(&slab, slab_bytes) == cudaSuccess) {
            store = std::make_unique<RecurrentSnapshotStore>();
            store->init(slab_bytes, static_cast<size_t>(budget_mb) << 20);
            if (store->enabled()) {
                scheduler->set_prefix_reuse_limit(std::move(reuse_limit));
                log_swa_snapshot_store(*store, budget_mb);
            } else {
                store.reset();
                IMP_CUDA_CHECK_LOG(cudaFree(slab));
                slab = nullptr;
            }
        }
    }
    if (store)
        return;
    kvm.set_prefix_caching_enabled(false);
    c.ec.use_prefix_caching = false;
    // A budget below one snapshot silently costs prefix caching, worse
    // than swa_snapshot_mb=0 (which keeps caching, drops SWA savings).
    // Name the required size or the two cases are indistinguishable in the log.
    const size_t need_mb = (kvm.swa_snapshot_bytes() + (1u << 20) - 1) >> 20;
    if (budget_mb > 0 && static_cast<size_t>(budget_mb) < need_mb) {
        IMP_LOG_WARN(
            "Prefix caching DISABLED: kv_cache.swa_snapshot_mb=%d is below one "
            "snapshot (%zu MiB). Set it to >=%zu to run SWA sizing AND prefix "
            "caching together, or to 0 to keep prefix caching and drop the SWA "
            "savings.",
            budget_mb, need_mb, need_mb);
    } else {
        IMP_LOG_INFO(
            "Prefix caching disabled under SWA sizing (snapshot store off — "
            "kv_cache.swa_snapshot_mb=%d)",
            budget_mb);
    }
}

void build_post_cache_gemm_buffers(const EngineConfig& ec, GraphExecutor& executor, cudaStream_t stream) {
    // Pre-allocate the gemm_nvfp4 fallback dequant workspace, sized from
    // wcache_.nvfp4 (populated above, so must come AFTER pre_dequant_weights).
    // Lets the M>1 fallback path run inside CUDA stream capture without cudaMalloc.
    (void)executor.allocate_nvfp4_dequant_workspace();
    // NVFP4 tile-GEMM LM head for batched-decode GEMM: only n>1 decode consumes
    // it, so skip the 19-47 MiB SfAtom-scale VRAM when max_batch_size==1 (the
    // perplexity harness lazy-builds its own). Must run AFTER pre_dequant_weights
    // and BEFORE decode-graph capture so the captured topology includes it.
    if (ec.max_batch_size > 1)
        executor.build_lm_head_cutlass_(stream);
    if (ec.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");
}

void log_gpu_memory(VRAMAllocator& alloc) {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
        IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                     (total_mem - free_mem) / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
                     free_mem / (1024.0 * 1024.0));
    alloc.report();
}

}  // namespace

// Phases in #1103 order: every VRAM read sees the bytes the steps before it took
// (layer map, SWA gate, budget, plan, weight caches, measured residual, pool, state).
bool Engine::init_kv_cache() {
    const auto& mcfg = model_->config();
    const KvInitCtx ctx{*model_, mcfg, config_, runtime_config_};
    KvShape kv;
    kv.head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
    kv.layer_map = map_kv_layers(*model_, kv.n_kv_layers);
    const int n_kv_layers = kv.n_kv_layers;
    const int head_dim = kv.head_dim;

    // Resolved in init_resolve_kv_block_size_() before the executor sizes its
    // workspaces - anything sized there needs the real value, not kKVBlockSize.
    // Fallback keeps this function correct if that resolver was skipped.
    if (config_.kv_block_size <= 0)
        config_.kv_block_size = kKVBlockSize;
    const int kv_bs = config_.kv_block_size;
    kv.kv_bs = kv_bs;
    int blocks_per_seq = (config_.max_seq_len + kv_bs - 1) / kv_bs;

    const SwaGate swa = resolve_swa_sizing(ctx, kv.layer_map, n_kv_layers, kv_bs, executor_.get(),
                                           [this] { return resolve_prefill_chunk_size_(); });
    swa_sizing_active_ = swa.active;
    swa_window_max_ = swa.window_max;
    if (swa.active) {
        swa_slack_tokens_ = swa.slack_tokens;
        swa_burst_cap_tokens_ = swa.burst_cap_tokens;
    }
    kv.n_swa_layers = swa.n_layers;
    kv.swa_active = swa_sizing_active_;
    kv.swa_live_tokens = swa_sizing_active_ ? swa_window_max_ + swa_slack_tokens_ + swa_burst_cap_tokens_ : 0;
    config_.swa_shared_tokens = swa_sizing_active_ ? swa.chunk_extra_tokens : 0;
    kv.layer_block_bytes = kv_block_bytes_layer_mean(mcfg, config_.kv_cache_dtype, kv_bs, head_dim,
                                                     kv.swa_active);

    // VRAM budget: KV charges the full measured cache demand, phase 3 floors at
    // it (no held-back reserve, AUDIT B62).
    const auto fingerprint = [this] { return model_fingerprint_(); };
    load_library_reserve(ctx, fingerprint, library_reserve_key_, library_reserve_cache_path_);

    // Reserved pool slots: mc verify W-1 + batched-verify spares (recomputed after the clamp).
    int ssm_reserved_slots = batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size);
    config_.kv_ceiling_seqs =
        kv_ceiling_seqs(admitted_batch(config_.max_batch_size, runtime_config_.runtime.max_batch_size),
                        experts_on_host_ && executor_ && executor_->expert_cache_nvfp4_slots());
    // MTP draft workspace: allocated at the end of init (take_planned_mtp_workspace_), so both passes
    // size KV without it; the live pass sees it as already spent.
    mtp_workspace_planned_bytes_ = mtp_workspace_plan_bytes_();
    const size_t live_free = effective_free_vram();
    auto vram_budget = compute_vram_budget(*model_, config_, n_kv_layers, head_dim,
                                           live_free > mtp_workspace_planned_bytes_
                                               ? live_free - mtp_workspace_planned_bytes_
                                               : 0,
                                           kv.swa_live_tokens, kv.n_swa_layers, &native_cache_demand(),
                                           ssm_reserved_slots, /*q8_imma_prefill=*/true,
                                           runtime_config_.gemm.moe_imma_prefill);
    // Cap the IMMA prefill planes at what the pass just charged: they are taken
    // lazily on each Q8_0 weight's first prefill, so uncapped the cache would
    // take whatever KV leaves free, varying per start (#1899). Set before sizing.
    mmq_q8_imma_set_plane_budget(vram_budget.imma_plane_bytes);
    kv.minmax_bytes_per_layer = sparse_minmax_plan_bytes(ctx, head_dim, kv.swa_active);

    const size_t distributable = effective_free_vram();
    const size_t spec_decode_bytes = spec_state_scratch_plan_bytes_(vram_budget.ssm_footprint_bytes,
                                                                    ssm_reserved_slots);
    ShadowPlanProbe probe = build_shadow_probe(ctx, vram_budget, kv, executor_.get(), distributable,
                                               spec_decode_bytes, mtp_workspace_planned_bytes_,
                                               ssm_reserved_slots);
    PlanResult plan = plan_memory(shadow_plan_input(probe));
    IMP_LOG_INFO("%s", shadow_plan_report(probe, plan, vram_budget.kv_max_blocks).c_str());
    if (!plan.ok) {
        clamp_max_batch_to_plan_(probe, plan, ssm_reserved_slots, vram_budget.kv_max_blocks);
        ssm_reserved_slots = batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size);
    }
    // Mode-2 weight caches stop at the plan's grant and leave its KV pool free (vram_budget.h).
    apply_plan_to_weight_caches(vram_budget, plan, probe.distributable_bytes, mcfg.is_nvfp4_prequant);
    int max_blocks = kv_blocks_from_plan(config_, plan, vram_budget);
    kv_replan_arm(kv_replan_, probe, vram_budget, max_blocks, plan.ok,
                  config_.library_reserve_mb < 0 && config_.kv_cache_max_blocks <= 0);
    log_sparse_minmax_pool(config_, kv, max_blocks);

    // Weight caches build BEFORE the KV pool (A7 step 6.4): caches have bounded
    // demand and go first, KV (elastic) takes what's left - reversed order can
    // starve caches to 0 free and spill on WSL2/WDDM (AUDIT B23).
    // The profile gates below must stay with the caches: they set flags the
    // cache build and graph capture depend on (wcache_->use_fp8).
    apply_gdn_precision_gates(ctx, executor_.get());

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
    if (!executor_->init_device_expert_cache() && experts_on_host_)
        demote_graphs_(GraphDemotionReason::ExpertsOnHost);
    warn_host_expert_batch(config_, *executor_, experts_on_host_);

    const size_t per_block_total_bytes = static_cast<size_t>(n_kv_layers) * kv.layer_block_bytes;
    // Ceiling for a growable pool: starts at the plan's commit (may be less than
    // the live-derived kv_max_blocks due to conservative reserves) and grows
    // toward the live figure under admission pressure - without this max() growable
    // was a no-op on planned loads.
    // kv_cache.max_blocks is a ceiling, not a floor: with growable on by default,
    // a pinned pool must never grow past the pin.
    const int kv_blocks_planned = config_.kv_cache_max_blocks > 0
                                      ? max_blocks
                                      : std::max(max_blocks, vram_budget.kv_max_blocks);
    max_blocks = size_kv_from_residual(vram_budget, config_, kv, per_block_total_bytes, max_blocks,
                                       kv_blocks_planned, max_seq_len_explicit_, kv_pool_floored_);
    if (!kv_pool_fits_vram_budget(config_, kv_bs, per_block_total_bytes, max_blocks, max_seq_len_explicit_))
        return false;

    // The clamp above answers "what fits right now"; a growable pool keeps the
    // pre-clamp number as its ceiling and commits the clamped one, so a reading
    // skewed by another process still releasing VRAM is not final.
    const int kv_ceiling_blocks = runtime_config_.kv_cache.growable ? kv_blocks_planned : 0;
    // What the pool was actually built with; the retry loop below may lower it.
    int kv_ceiling_effective = kv_ceiling_blocks;
    max_blocks = growable_initial_blocks(runtime_config_, max_blocks, kv_ceiling_blocks, kv_bs);
    log_kv_pool_summary(ctx, kv, max_blocks);

    std::unique_ptr<KVCache> kv_cache = allocate_kv_pool(ctx, kv, &vram_alloc_, vram_budget.swa_max_blocks,
                                                         kv_ceiling_blocks, max_blocks, kv_ceiling_effective);
    kv_replan_set_pool(kv_replan_, per_block_total_bytes, kv_ceiling_blocks, kv_ceiling_effective);
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));
    if (!max_seq_len_explicit_)
        clamp_auto_max_seq_len_to_pool(config_.max_seq_len, std::max(max_blocks, kv_ceiling_effective),
                                       kv_bs);
    log_kv_pool_residency(*kv_cache_raw_);
    if (swa_sizing_active_) {
        kv_manager_->enable_swa_sizing(swa_window_max_, swa_slack_tokens_);
        swa_sizing_active_ = kv_manager_->swa_sizing_enabled();
    }
    enable_sparse_key_minmax(ctx, *kv_cache_raw_);
    enable_prefix_caching(ctx, *kv_manager_, fingerprint, stream_);

    executor_->set_kv_layer_map(kv.layer_map);
    apply_checkpoint_fp8_kv_scales(*executor_, *model_, kv.layer_map, config_.kv_cache_dtype,
                                   runtime_config_);

    if (offload_mgr_)
        executor_->set_offload_manager(offload_mgr_.get());
    scheduler_->set_kv_manager(kv_manager_.get());

    init_recurrent_state(
        ctx, *kv_manager_, ssm_reserved_slots, &vram_alloc_, scheduler_.get(),
        [this] { return recurrent_slot_admissible_(); },
        [this](Request& r) { return hybrid_prefix_reuse_limit_(r); },
        {ssm_state_, d_ssm_seq_slots_, h_ssm_seq_slots_, recurrent_snapshots_});
    init_swa_snapshots(
        ctx, swa_sizing_active_, *kv_manager_, scheduler_.get(),
        [this](Request& r) { return swa_prefix_reuse_limit_(r); }, swa_snapshots_, swa_snap_slab_);

    IMP_CUDA_CHECK_BOOL(cudaStreamSynchronize(stream_));

    check_moe_decode_cache_coverage_();

    build_post_cache_gemm_buffers(config_, *executor_, stream_);

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq,
                                /*with_swa_tables=*/swa_sizing_active_);
    // The "can the verify run" gate lives inside the callee: gating here on the
    // spare slot count left speculative.factored_spare permanently disabled
    // (it reserves none by design).
    if (!ensure_batch_verify_bufs_(stream_))  // init-time staging
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
        (void)pf_staging_evt_.create(cudaEventDisableTiming);  // stays null on failure
        // The constrained pipeline's pinned landing for the sampled token and
        // its ready event: engine-lifetime, so a first json request does not
        // allocate while serving (I2 gate phase B counted the lazy acquire).
        cpipe_.h_token = PinnedBuffer::acquire(cuda_host_pinned_allocator(), sizeof(int32_t));
        (void)cpipe_.ev.create(cudaEventDisableTiming);  // stays null on failure
    }

    log_gpu_memory(vram_alloc_);

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
    d_agl_block_tables_ = ints(lay.agl_bt);
    d_cp_block_tables_ = ints(lay.cp_bt);
    d_cp_token_ = reinterpret_cast<int32_t*>(base + lay.cp_token);
    d_cp_pos_ = ints(lay.cp_pos);
    d_cp_ctx_ = ints(lay.cp_ctx);
    if (swa_sizing_active_) {
        d_pf_block_tables_swa_ = ints(lay.pf_bt_swa);
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

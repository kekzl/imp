#include "memory/plan.h"

#include <algorithm>
#include <format>

namespace imp {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;

size_t mib(size_t bytes) { return bytes / (1024 * 1024); }

void push(std::vector<PlanLine>& v, const char* name, RegionTag tag, size_t bytes) {
    if (bytes > 0)
        v.push_back(PlanLine{name, tag, bytes});
}

std::string fmt_lever(const char* key, long long from, long long to) {
    return std::format("{} {} -> {}", key, from, to);
}

}  // namespace

size_t MemoryPlan::total() const {
    size_t sum = context_reserve + library_reserve + model_resident + optional_caches +
                 engine_persistent + forward_scratch + kv.bytes + kv.swa_bytes;
    for (const auto& p : pools)
        sum += p.bytes;
    return sum;
}

std::vector<PlanLine> MemoryPlan::lines() const {
    std::vector<PlanLine> v;
    push(v, "model weights + mandatory caches", RegionTag::ModelResident, model_resident);
    push(v, "optional weight caches", RegionTag::ModelResident, optional_caches);
    push(v, "KV pool", RegionTag::KvBlockPool, kv.bytes);
    push(v, "KV pool (SWA group)", RegionTag::SwaBlockPool, kv.swa_bytes);
    push(v, "engine-persistent", RegionTag::EnginePersistent, engine_persistent);
    push(v, "forward scratch", RegionTag::ForwardScratch, forward_scratch);
    for (const auto& p : pools)
        push(v, p.name, p.tag, p.bytes);
    push(v, "CUDA context + driver", RegionTag::Other, context_reserve);
    push(v, "library reserve (measured)", RegionTag::Other, library_reserve);
    std::stable_sort(v.begin(), v.end(),
                     [](const PlanLine& a, const PlanLine& b) { return a.bytes > b.bytes; });
    return v;
}

std::string PlanFailure::report() const {
    // std::format, not snprintf into a fixed buffer: a plan line name long
    // enough to overrun 256 bytes used to be silently truncated, and this
    // report is what a user reads when the engine refuses to start.
    std::string out = std::format("Cannot fit this configuration in the {} MiB budget.\n\n", mib(budget));
    out += std::format("  {:<36} {:>10}\n", "requested", "MiB");
    for (const auto& l : lines)
        out += std::format("    {:<34} {:>10.0f}\n", l.name, l.bytes / kMiB);
    out += std::format("    {:<34} {:>10}\n", "", "-----");
    out += std::format("    {:<34} {:>10.0f}   over by {:.0f} MiB\n\n", "total", requested / kMiB,
                       over_by / kMiB);

    if (!levers.empty()) {
        out += "  the largest levers\n";
        for (const auto& lv : levers)
            out += std::format("    {:<38} frees {:>6.0f} MiB\n", lv.change, lv.frees / kMiB);
    }
    return out;
}

PlanResult plan_memory(const PlanInput& in) {
    PlanResult res;
    MemoryPlan& p = res.plan;

    const int bs = in.limits.kv_block_size > 0 ? in.limits.kv_block_size : 16;
    const int batch = std::max(1, in.limits.max_batch_size);
    const int seq = std::max(0, in.limits.max_seq_len);

    // ── 1. Fixed charges, subtracted before anything is distributed. ──
    // Order matters: these are not negotiable and every one of them was, in the
    // old code, either invisible (library reserve) or re-derived per phase from
    // a live free-VRAM reading (everything else).
    p.context_reserve = in.context_bytes;
    p.library_reserve = in.library.bytes;

    // ── 2. Model-resident. The mandatory cache subset is never traded; the
    // rest of the cache demand is a preference, and is charged in step 5. ──
    // This line used to commit weight_cache_bytes whole, and that made the plan
    // reject configurations the engine serves without trouble: on
    // Qwen3.6-35B-A3B-NVFP4 at 32k it wanted 6553 MiB of caches against 6083 MiB
    // distributable and failed by 4703 — while naming "drop the optional weight
    // caches, frees 4285 MiB" as its own largest lever. A charge the plan would
    // itself drop under pressure is not a commitment (AUDIT B69).
    const size_t mandatory_caches =
        std::min(in.model.mandatory_cache_bytes, in.model.weight_cache_bytes);
    const size_t optional_cache_demand = in.model.weight_cache_bytes - mandatory_caches;
    p.model_resident = in.model.weight_bytes + mandatory_caches;

    // ── 3. Engine-persistent + scratch + the fixed pools. ──
    p.engine_persistent = in.engine_persistent_bytes + in.features.vision_tower_bytes +
                          in.features.spec_decode_bytes;
    p.forward_scratch = in.forward_scratch_bytes;

    push(p.pools, "SSM/GDN state", RegionTag::SsmState, in.features.ssm_state_bytes);
    push(p.pools, "recurrent snapshots", RegionTag::RecurrentSnapshots,
         in.features.recurrent_snapshot_bytes);
    push(p.pools, "residual FP16 ring", RegionTag::ResidualRing, in.features.residual_ring_bytes);

    // ── 4. SWA group: batch-shaped, charged before the global pool. ──
    int n_global_layers = in.model.n_kv_layers;
    const size_t per_layer = in.limits.kv_block_bytes_per_layer;
    if (in.features.swa_live_tokens > 0 && in.features.n_swa_layers > 0 &&
        in.features.n_swa_layers <= in.model.n_kv_layers) {
        n_global_layers = in.model.n_kv_layers - in.features.n_swa_layers;
        const int per_seq = (in.features.swa_live_tokens + bs - 1) / bs + 1;
        p.kv.swa_blocks = per_seq * batch;
        p.kv.swa_bytes = static_cast<size_t>(p.kv.swa_blocks) * per_layer *
                         static_cast<size_t>(in.features.n_swa_layers);
    }

    // ── 5. Everything above is committed. KV takes the COMPUTED residual. ──
    // This is the structural change: the residual is arithmetic over declared
    // demand, not a measurement of what happens to be free at this instant.
    const size_t committed = p.context_reserve + p.library_reserve + p.model_resident +
                             p.engine_persistent + p.forward_scratch + p.kv.swa_bytes +
                             [&] {
                                 size_t s = 0;
                                 for (const auto& q : p.pools)
                                     s += q.bytes;
                                 return s;
                             }();

    const size_t per_block = per_layer * static_cast<size_t>(std::max(0, n_global_layers));
    p.kv.blocks_per_seq = (seq + bs - 1) / bs;

    if (committed >= in.budget_bytes || per_block == 0) {
        // Not even the fixed charges fit, or there is no global KV to size.
        p.kv.blocks = 0;
        p.kv.bytes = 0;
    } else {
        const size_t residual = in.budget_bytes - committed;
        // The engine builds the weight caches BEFORE the KV pool (A7 step 6.4)
        // and KV then takes the measured remainder, so the plan grants the
        // optional caches first — but never below one advertised sequence, which
        // is the floor D8 already refuses under. That ordering is the policy;
        // stating it here is the point of having a plan at all.
        const size_t kv_floor_bytes = static_cast<size_t>(p.kv.blocks_per_seq) * per_block;
        const size_t grantable = residual > kv_floor_bytes ? residual - kv_floor_bytes : 0;
        p.optional_caches = std::min(optional_cache_demand, grantable);

        const size_t kv_residual = residual - p.optional_caches;
        const int want_blocks = p.kv.blocks_per_seq * batch;
        const int fits_blocks = static_cast<int>(kv_residual / per_block);
        p.kv.blocks = std::min(want_blocks, fits_blocks);
        p.kv.bytes = static_cast<size_t>(p.kv.blocks) * per_block;
    }

    const int floor_blocks =
        in.limits.min_kv_tokens > 0 ? (in.limits.min_kv_tokens + bs - 1) / bs : 0;
    p.kv.below_floor = floor_blocks > 0 && p.kv.blocks < floor_blocks;

    // ── 6. Fit or fail. ──
    const size_t total = p.total();
    // The floor is a hard requirement, not a preference: a pool that cannot
    // hold one advertised sequence rejects requests at admission while
    // /v1/models keeps advertising max_seq_len. Better to fail at load.
    const bool floor_ok = !p.kv.below_floor;
    if (total <= in.budget_bytes && floor_ok) {
        res.ok = true;
        return res;
    }

    PlanFailure& f = res.failure;
    f.budget = in.budget_bytes;
    f.lines = p.lines();
    if (p.kv.below_floor) {
        // Report what it would have to cost to honour the floor, so the
        // arithmetic in the message adds up to the shortfall the operator has
        // to close.
        const size_t need = static_cast<size_t>(floor_blocks) * per_block;
        f.requested = total - p.kv.bytes + need;
        for (auto& l : f.lines)
            if (l.tag == RegionTag::KvBlockPool)
                l.bytes = need;
    } else {
        f.requested = total;
    }
    f.over_by = f.requested > in.budget_bytes ? f.requested - in.budget_bytes : 0;

    // Levers, largest first. Each is a knob the operator actually has.
    if (seq > 256 && per_block > 0) {
        const size_t frees =
            static_cast<size_t>(p.kv.blocks_per_seq - ((seq / 2) + bs - 1) / bs) * per_block * batch;
        f.levers.push_back(PlanLever{fmt_lever("runtime.max_seq_len", seq, seq / 2), frees});
    }
    if (batch > 1) {
        const size_t frees = p.kv.bytes / static_cast<size_t>(batch) + p.kv.swa_bytes / static_cast<size_t>(batch);
        f.levers.push_back(
            PlanLever{fmt_lever("runtime.max_batch_size", batch, batch - 1), frees});
    }
    if (p.kv.bytes > 0) {
        // FP8 KV halves the pool; the accuracy trade is per-family (see
        // docs/vram_audit.md 2026-06-12 Lever C), so it is offered, not applied.
        f.levers.push_back(PlanLever{std::string("kv_cache.dtype f16 -> fp8"), p.kv.bytes / 2});
    }
    if (in.model.weight_cache_bytes > in.model.mandatory_cache_bytes) {
        f.levers.push_back(PlanLever{std::string("drop the optional weight caches"),
                                     in.model.weight_cache_bytes - in.model.mandatory_cache_bytes});
    }
    std::stable_sort(f.levers.begin(), f.levers.end(),
                     [](const PlanLever& a, const PlanLever& b) { return a.frees > b.frees; });
    if (f.levers.size() > 3)
        f.levers.resize(3);

    return res;
}

}  // namespace imp

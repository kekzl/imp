#pragma once

#include <cstddef>

namespace imp {

// ─────────────────────────────────────────────────────────────────────
// Budget-aware VRAM query — the "pretend the GPU is only X MiB" view.
//
// Multiple imp-server processes sharing one GPU need each process to size
// itself (weight caches, KV clamp, expert offload, workspaces, upload gates)
// against ITS slice of the card, not against whatever cudaMemGetInfo happens
// to report. EngineConfig.vram_budget_mb declares that slice;
// vram_budget_mem_get_info() is the drop-in replacement for cudaMemGetInfo
// at every SIZING/decision site (diagnostic/audit sites keep the raw call).
//
// Semantics (budget installed):
//   my_used = free_at_install − free_now   (baseline delta: covers ALL of
//             this process's allocations — async pool, plain cudaMalloc,
//             cuBLAS-internal — without per-site tracking)
//   free'   = min(free_now, budget − my_used)
//   total'  = budget
// so used' = total' − free' = my_used: a consistent virtual small GPU. The
// total'-capping also right-sizes the various total/10 headroom heuristics.
//
// A neighbour process allocating AFTER our install inflates my_used and
// shrinks our view — the conservative direction (we size smaller, never
// overcommit the card). Frees by this process shrink my_used again.
//
// Best-effort hard cap, not an OS limit: allocations that bypass the sizing
// gates (small fixed buffers, cuBLAS handles) still land outside the budget;
// leave ~1 GiB of real headroom between the sum of budgets and the card.
//
// Not thread-safe against concurrent install; install once from
// Engine::init (single-engine-per-process is the supported deployment).
// Budget 0 = uncapped (raw cudaMemGetInfo passthrough).
// ─────────────────────────────────────────────────────────────────────

// Install (or clear, budget_mb=0) the process-wide budget and snapshot the
// baseline. Called from Engine::init once the config is resolved.
void vram_budget_install(size_t budget_mb);

// Installed budget in bytes (0 = uncapped).
size_t vram_budget_bytes();

// Device-used at the moment the view was installed: the CUDA primary context
// plus anything a neighbour process already held. NOT this process's model
// memory, and NOT charged against the budget — the budget covers what imp
// allocates after init. Snapshotted even when uncapped.
size_t vram_used_at_install_bytes();

// What this process has allocated since the install, i.e. the same baseline
// delta the budget view sizes from. This — not device-used — is the number a
// budget is a cap on, so it is what "--vram-budget respected" has to be
// measured against. 0 if the view was never installed.
size_t vram_own_used_bytes();

// High water of vram_own_used_bytes(). Sampled at every sizing site, which is
// the phase the peak forms in; serving adds nothing to it (I2).
size_t vram_own_peak_bytes();

// cudaMemGetInfo with the budget view applied. Either out pointer may be
// null. Returns false (zeros) if the raw query fails.
bool vram_budget_mem_get_info(size_t* free_bytes, size_t* total_bytes);

// Canonical free-VRAM reserve floor for the sizing phases: pct% of the
// (budget-visible) total, floored at 256 MiB. Keeps the WSL2 shared-memory
// spill guard in ONE place — the budget pass and every pre-dequant phase
// used to re-derive this independently (and one copy had drifted to a
// 1 MiB floor). Pass the `total` from vram_budget_mem_get_info so the
// floor scales with a --vram-budget slice, not the physical card.
// The pct parameter is a budget-planner knob (imp.conf vram.reserve_floor_pct,
// consumed only inside compute_vram_budget); the pre-dequant phases keep the
// default so their internal safety floors stay independent of the knob.
// Headroom the VRAMAllocator enforces on every allocation >=16 MiB
// (can_allocate: free >= bytes + headroom). This is a HARD constraint, not a
// policy knob: a plan that leaves less free than this cannot be executed, the
// allocation is simply refused. Engine::init and the VRAM budget must agree on
// it — they used to disagree by 1118 MiB on 32 GB, and the KV pool happily
// consumed the difference (#1103).
constexpr int kAllocatorHeadroomPct = 5;

inline size_t vram_allocator_headroom(size_t total_bytes) {
    return total_bytes * static_cast<size_t>(kAllocatorHeadroomPct) / 100;
}

inline size_t vram_reserve_floor(size_t total_bytes, int pct = 10) {
    const size_t floor_bytes = 256ULL * 1024 * 1024;
    pct = pct < 0 ? 0 : (pct > 50 ? 50 : pct);
    const size_t share = total_bytes * static_cast<size_t>(pct) / 100;
    return share > floor_bytes ? share : floor_bytes;
}

// Outcome of sizing the KV pool from the measured post-cache residual.
struct KvResidualSizing {
    int blocks = 0;         // block count to use
    bool clamped = false;   // residual was smaller than the planned pool
    bool floored = false;   // residual could not cover even `floor_blocks`
};

// The KV pool is sized from what is left AFTER the weight caches are built,
// minus the allocator headroom above — which is a hard constraint, not a
// policy knob. The consequence is easy to miss and has now cost two bugs:
// anyone who sets VRAM aside *for* this pool must set aside the headroom ON
// TOP of the pool, or the residual they leave is entirely headroom, `room`
// evaluates to 0, and this returns `floor_blocks` — a rescue, not a size.
// #1103 was that mismatch between Engine::init and the VRAM budget (1118 MiB
// on a 32 GiB card); #1251 was the same mismatch in the NVFP4 MoE cache's
// reserve, which left 1264 MiB against a 1630 MiB headroom and pinned a
// 35B model to a 512-token pool that cancelled every generation.
//
// `floored` distinguishes "the pool is smaller than planned" (normal: the
// plan is a projection, the residual is the truth) from "there was nothing
// left to size from" (always an operator-visible fault), so the caller can
// be quiet about the first and loud about the second.
// Blocks one max_seq_len sequence occupies. 0 when either input is unset,
// which reads as "no requirement to check against".
inline int kv_blocks_per_sequence(int max_seq_len, int block_size) {
    if (max_seq_len <= 0 || block_size <= 0)
        return 0;
    return (max_seq_len + block_size - 1) / block_size;
}

// What the operator has to be told about the pool this sizing produced.
//
// `Floored` has had its own message since #1251: nothing was left to size
// from. `ShortOfOneSequence` is the quiet half of the same fault and had
// none — the pool is a real size, just too small for a single max_seq_len
// request, so the load reports success and every full-length generation is
// cancelled at admission instead. Both are operator faults; only the first
// was audible.
enum class KvPoolVerdict {
    Sufficient,          // holds at least one full-length sequence
    ShortOfOneSequence,  // sized, but no full-length request can be admitted
    Floored,             // nothing was left to size from
};

inline KvPoolVerdict kv_pool_verdict(const KvResidualSizing& sizing, int max_seq_len,
                                     int block_size) {
    if (sizing.floored)
        return KvPoolVerdict::Floored;
    const int need = kv_blocks_per_sequence(max_seq_len, block_size);
    if (need > 0 && sizing.blocks < need)
        return KvPoolVerdict::ShortOfOneSequence;
    return KvPoolVerdict::Sufficient;
}

inline KvResidualSizing kv_blocks_from_residual(size_t free_bytes, size_t headroom_bytes,
                                                size_t per_block_bytes, int planned_blocks,
                                                int floor_blocks) {
    KvResidualSizing out;
    out.blocks = planned_blocks;
    if (per_block_bytes == 0 || planned_blocks <= 0)
        return out;
    const size_t room = (free_bytes > headroom_bytes) ? (free_bytes - headroom_bytes) : 0;
    const size_t fits_sz = room / per_block_bytes;
    const int fits = fits_sz > static_cast<size_t>(planned_blocks)
                         ? planned_blocks
                         : static_cast<int>(fits_sz);
    if (fits >= planned_blocks)
        return out;
    out.clamped = true;
    out.floored = fits < floor_blocks;
    out.blocks = out.floored ? floor_blocks : fits;
    return out;
}

}  // namespace imp

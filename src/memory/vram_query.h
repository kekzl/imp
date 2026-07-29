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

}  // namespace imp

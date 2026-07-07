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

// cudaMemGetInfo with the budget view applied. Either out pointer may be
// null. Returns false (zeros) if the raw query fails.
bool vram_budget_mem_get_info(size_t* free_bytes, size_t* total_bytes);

// Canonical free-VRAM reserve floor for the sizing phases: 10% of the
// (budget-visible) total, floored at 256 MiB. Keeps the WSL2 shared-memory
// spill guard in ONE place — the budget pass and every pre-dequant phase
// used to re-derive this independently (and one copy had drifted to a
// 1 MiB floor). Pass the `total` from vram_budget_mem_get_info so the
// floor scales with a --vram-budget slice, not the physical card.
inline size_t vram_reserve_floor(size_t total_bytes) {
    const size_t floor_bytes = 256ULL * 1024 * 1024;
    const size_t tenth = total_bytes / 10;
    return tenth > floor_bytes ? tenth : floor_bytes;
}

}  // namespace imp

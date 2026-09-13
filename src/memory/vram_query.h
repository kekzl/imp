#pragma once

#include <cstddef>

namespace imp {

// Budget-aware VRAM query, the "pretend the GPU is only X MiB" view. Multiple
// imp-server processes sharing one GPU need each to size itself against ITS slice, not
// whatever cudaMemGetInfo reports. EngineConfig.vram_budget_mb declares that slice;
// vram_budget_mem_get_info() is the drop-in replacement for cudaMemGetInfo at every
// SIZING/decision site (diagnostic/audit sites keep the raw call).
// Semantics (budget installed):
//   my_used = free_at_install - free_now   (baseline delta: covers ALL of this
//             process's allocations without per-site tracking)
//   free'   = min(free_now, budget - my_used)
//   total'  = budget
// so used' = total' - free' = my_used: a consistent virtual small GPU.
// A neighbour process allocating AFTER install inflates my_used and shrinks our view,
// the conservative direction (never overcommit). Frees by this process shrink my_used
// again.
// Best-effort hard cap, not an OS limit: allocations bypassing the sizing gates (small
// fixed buffers, cuBLAS handles) still land outside the budget; leave ~1 GiB of real
// headroom between the sum of budgets and the card.
// Not thread-safe against concurrent install; install once from Engine::init
// (single-engine-per-process is the supported deployment). Budget 0 = uncapped.

// Install (or clear, budget_mb=0) the process-wide budget and snapshot the
// baseline. Called from Engine::init once the config is resolved.
void vram_budget_install(size_t budget_mb);

// Installed budget in bytes (0 = uncapped).
size_t vram_budget_bytes();

// Device-used at the moment the view was installed: the CUDA primary context plus
// anything a neighbour process already held. NOT this process's model memory, and NOT
// charged against the budget (the budget covers what imp allocates after init).
// Snapshotted even when uncapped.
size_t vram_used_at_install_bytes();

// What this process has allocated since install, the same baseline delta the budget view
// sizes from. This, not device-used, is what a budget caps, so it is what
// "--vram-budget respected" must be measured against. 0 if the view was never installed.
size_t vram_own_used_bytes();

// High water of vram_own_used_bytes(). Sampled at every sizing site, which is
// the phase the peak forms in; serving adds nothing to it (I2).
size_t vram_own_peak_bytes();

// cudaMemGetInfo with the budget view applied. Either out pointer may be
// null. Returns false (zeros) if the raw query fails.
bool vram_budget_mem_get_info(size_t* free_bytes, size_t* total_bytes);
// The same view with the lazy pools' pending charge left in `free`, for an accounting
// reader that wants what the device holds rather than what a planner may take. One read,
// so it cannot race a slot commit the way separately reading adjusted-free then adding
// the ledger back would.
bool vram_budget_mem_get_info_ex(size_t* free_bytes, size_t* total_bytes, bool exclude_pending);

// Bytes the planner charged that no pool has committed yet: a lazy slab's reservation
// minus what it has mapped so far. Free VRAM shows them as free, and they are not:
// whoever sizes from a free reading (KV plan, weight caches, KV growth cap) must leave
// them alone, or the slot commit they were charged for spills when it comes. Lazy pools
// add their reservation at init, subtract each commit, add back each decommit;
// vram_budget_mem_get_info() subtracts the total from what it reports free.
void vram_reserved_uncommitted_add(std::ptrdiff_t delta_bytes);
size_t vram_reserved_uncommitted_bytes();

// Canonical free-VRAM reserve floor for sizing phases: pct% of the (budget-visible)
// total, floored at 256 MiB. Keeps the WSL2 shared-memory spill guard in ONE place; pass
// the `total` from vram_budget_mem_get_info so the floor scales with a --vram-budget
// slice, not the physical card. pct is a budget-planner knob
// (vram.reserve_floor_pct, compute_vram_budget only); pre-dequant phases keep the
// default so their internal safety floors stay independent of the knob.
// Headroom the VRAMAllocator enforces on every allocation >=16 MiB (can_allocate: free
// >= bytes + headroom). A HARD constraint, not a policy knob: a plan leaving less free
// than this cannot be executed, the allocation is simply refused. Engine::init and the
// VRAM budget must agree on it (#1103).
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

// Whether a weight upload consumed more device free VRAM than the checkpoint on disk can
// account for. The upload is the first moment a co-tenant on the card is observable at
// all: under WSL2/WDDM the driver reports the whole card as free until a process
// allocates, so both vram_used_at_install_bytes() and the plan's free-VRAM read are
// blind to a neighbour, and the KV pool that follows is sized from the shrunken residual
// while the load reports success (MEMORY.md B8). Consuming LESS than the file is
// ordinary (host-resident experts, dropped sources), so this check is one-sided by
// construction. on_disk_bytes == 0 means "could not size the checkpoint", not a fault.
inline bool upload_exceeds_checkpoint(size_t consumed_bytes, size_t on_disk_bytes) {
    if (on_disk_bytes == 0)
        return false;
    return consumed_bytes > on_disk_bytes + on_disk_bytes / 4;
}

// Outcome of sizing the KV pool from the measured post-cache residual.
struct KvResidualSizing {
    int blocks = 0;         // block count to use
    bool clamped = false;   // residual was smaller than the planned pool
    bool floored = false;   // residual could not cover even `floor_blocks`
};

// The KV pool is sized from what's left AFTER weight caches are built, minus the
// allocator headroom (a hard constraint, not a policy knob). Anyone reserving VRAM FOR
// this pool must set the headroom aside ON TOP of the pool, or the residual left is
// entirely headroom, `room` evaluates to 0, and this returns floor_blocks, a rescue, not
// a size (#1103 was this mismatch between Engine::init and the VRAM budget; #1251 was
// the same mismatch in the NVFP4 MoE cache's reserve).
// `floored` distinguishes "the pool is smaller than planned" (normal) from "there was
// nothing left to size from" (always an operator-visible fault).
// Blocks one max_seq_len sequence occupies; 0 when either input is unset ("no
// requirement to check against").
inline int kv_blocks_per_sequence(int max_seq_len, int block_size) {
    if (max_seq_len <= 0 || block_size <= 0)
        return 0;
    return (max_seq_len + block_size - 1) / block_size;
}

// Why an explicit kv_cache.block_size cannot be served, nullptr when it can. Every
// paged kernel takes the block size at runtime, but two tile it: the FP8 split-K decode
// kernel dispatches on block_size % 16 == 0, and the NVFP4 TC path maps one 16-token
// WMMA tile per block chunk. Above 256, a block (the prefix cache's reuse granularity
// and a sequence's minimum footprint) no longer buys anything the split-K count did not
// already have (AUDIT B-5).
inline const char* kv_block_size_error(int block_size) {
    if (block_size < 16)
        return "is below 16, the token tile of the FP8 and NVFP4 tensor-core decode kernels";
    if (block_size % 16 != 0)
        return "is not a multiple of 16 (attention_paged_fp8_tile.cu dispatches on block_size % 16 == 0)";
    if (block_size > 256)
        return "is above 256; a block is the prefix cache's reuse granularity and a sequence's minimum "
               "footprint";
    return nullptr;
}

// Times n device-to-device copies issued back to back on the legacy default stream, one
// event pair around the batch, returning bandwidth in GB/s counting every byte twice
// (read plus write). 0 when a copy cannot run.
// This is the residency test the platform needs: on WSL2/WDDM a successful allocation
// proves nothing, and a pool the driver spilled into host memory serves at a fraction of
// the bandwidth with no error anywhere.
// The same set is copied for warm_ms first: an idle card sits at floor clocks, and a
// cold pass can read as resident when it isn't. What warm-up cannot fix is a set whose
// traffic fits the 96 MB L2: span enough memory to exceed it, or read the result as
// "resident, L2-served".
struct DeviceCopy {
    void* dst;
    const void* src;
    size_t bytes;
};
double device_copy_bandwidth_gbps(const DeviceCopy* copies, size_t n, int warm_ms = 300);
inline double device_copy_bandwidth_gbps(void* dst, const void* src, size_t bytes, int warm_ms = 300) {
    const DeviceCopy c{dst, src, bytes};
    return device_copy_bandwidth_gbps(&c, 1, warm_ms);
}

// What the operator has to be told about the pool this sizing produced. Floored has had
// its own message since #1251: nothing was left to size from. ShortOfOneSequence is the
// quiet half of the same fault and had none: the pool is a real size, just too small for
// a single max_seq_len request, so the load reports success and every full-length
// generation is cancelled at admission instead. Both are operator faults; only the first
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

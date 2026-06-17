#pragma once

// Façade that consolidates VRAM ownership for the imp Engine.
//
// Phase 5 Track C of
// docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md
//
// Before this façade, Engine carried a `VRAMAllocator vram_alloc_` field
// directly and the budget + storage-plan free functions lived in
// `src/runtime/{vram_budget,storage_planner}.cpp`. This left memory-related
// ownership scattered across Engine + two anonymous free functions.
//
// MemoryManager bundles them into a single owner: the Engine has one
// `MemoryManager memory_manager_` field, and code that previously called
//   vram_alloc(engine.vram_alloc_, ...)
//   compute_vram_budget(model, config, ...)
//   plan_storage(model, cfg, hints)
// now reaches them via `engine.memory_manager()` (allocator) and the
// matching wrapper methods (`compute_budget`, `plan_storage_for`).
//
// The underlying modules (vram_allocator, device_allocator, pinned_allocator,
// vram_budget, storage_planner) are unchanged. This is a façade, not an
// implementation merge: existing free functions still work; callers just
// reach the allocator through the façade now.
//
// PinnedAllocator and DeviceAllocator are exposed via lazy accessors. The
// imp Engine does not currently instantiate them on construction (no path
// uses them yet); making them lazy keeps Engine init cheap and avoids
// allocating a 64-MiB pinned pool until something actually asks for it.

#include "memory/device_allocator.h"
#include "memory/pinned_allocator.h"
#include "memory/vram_allocator.h"
#include "runtime/storage_planner.h"
#include "runtime/vram_budget.h"

#include <cstddef>
#include <memory>

namespace imp {

class Model;
struct ModelConfig;
struct EngineConfig;

class MemoryManager {
public:
    MemoryManager() = default;
    ~MemoryManager() = default;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    // -- VRAM allocator ----------------------------------------------------
    // Owned directly: Engine calls init() before use.
    VRAMAllocator& vram_allocator() noexcept { return vram_alloc_; }
    const VRAMAllocator& vram_allocator() const noexcept { return vram_alloc_; }

    // -- Pinned host allocator (lazy) -------------------------------------
    // Constructed on first access. Default pool size from PinnedAllocator
    // (currently 64 MiB).
    PinnedAllocator& pinned_allocator() {
        if (!pinned_alloc_)
            pinned_alloc_ = std::make_unique<PinnedAllocator>();
        return *pinned_alloc_;
    }
    bool has_pinned_allocator() const noexcept { return pinned_alloc_ != nullptr; }

    // -- Device (cudaMallocAsync pool) allocator (lazy) -------------------
    // Constructed on first access.
    DeviceAllocator& device_allocator() {
        if (!device_alloc_)
            device_alloc_ = std::make_unique<DeviceAllocator>();
        return *device_alloc_;
    }
    bool has_device_allocator() const noexcept { return device_alloc_ != nullptr; }

    // -- Budget + storage planning (wrap free functions) ------------------
    static VRAMBudget compute_budget(const Model& model, const EngineConfig& config, int n_kv_layers,
                                     int head_dim, size_t free_vram) {
        return compute_vram_budget(model, config, n_kv_layers, head_dim, free_vram);
    }

    static StoragePlan plan_storage_for(const Model& model, const ModelConfig& cfg, const PlanHints& hints) {
        return plan_storage(model, cfg, hints);
    }

private:
    VRAMAllocator vram_alloc_;
    std::unique_ptr<PinnedAllocator> pinned_alloc_;
    std::unique_ptr<DeviceAllocator> device_alloc_;
};

}  // namespace imp

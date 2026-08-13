#pragma once

// T1 (model-resident) and T2 (engine-persistent) of the lifetime taxonomy
// (docs/internals/MEMORY.md §A2/§A3.3).
//
// A bump arena over ONE Region. Allocations are handed out by advancing an
// offset and are never individually freed — the whole arena is released at
// once. That is the point: the failure mode a bump arena makes structurally
// impossible is the per-object leak, because there is no per-object free to
// forget. It is also why every span it hands out is stable for the arena's
// lifetime (I3): nothing inside ever moves.
//
// Two instances exist in the engine, not one:
//   - model-resident: weights + the pre-dequant weight caches. Released
//     wholesale on model unload, which is what makes server.model_swap a
//     bounded operation rather than a process restart.
//   - engine-persistent: workspaces, cuBLAS/CUTLASS scratch, graph buffers.
//     Survives a model swap.

#include "memory/backend.h"
#include "memory/span.h"

#include <cstddef>
#include <mutex>

namespace imp {

class ArenaAllocator {
public:
    ArenaAllocator() = default;
    ~ArenaAllocator();

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Acquire `capacity` bytes from `backend`. Fails if already open.
    [[nodiscard]] MemError open(Backend& backend, size_t capacity, RegionTag tag);

    // Release the whole region. Every span previously handed out dangles
    // afterwards — that is the contract, and it is why the arena's lifetime is
    // tied to a phase boundary rather than to any individual consumer.
    void close();

    bool is_open() const { return region_.valid(); }

    // Bump-allocate `bytes`, aligned. Returns an empty span when the arena is
    // exhausted; the caller decides (I6 — exhaustion is a value, not a crash).
    StableSpan<std::byte> take_bytes(size_t bytes, size_t alignment = 256);

    template <class T>
    StableSpan<T> take(size_t count, size_t alignment = alignof(T) < 256 ? 256 : alignof(T)) {
        auto raw = take_bytes(count * sizeof(T), alignment);
        if (raw.empty())
            return StableSpan<T>();
        return raw.as<T>();
    }

    // Rewind to empty WITHOUT releasing the region. Only legal at a phase
    // boundary; every outstanding span is invalidated.
    void reset();

    size_t capacity() const;
    size_t used() const;
    size_t remaining() const;
    // High-water of `used` since open(). The number the planner wants back:
    // it is what the arena actually needed, as opposed to what it was given.
    size_t high_water() const;
    RegionTag tag() const { return tag_; }

    // Bumped by close()/reset(). Consumers that cache derived pointers across
    // a model swap compare this instead of inventing their own mechanism —
    // the executor already does exactly this with workspace_generation.
    uint64_t generation() const;

private:
    mutable std::mutex mu_;
    Region region_;
    size_t offset_ = 0;
    size_t high_water_ = 0;
    uint64_t generation_ = 0;
    RegionTag tag_ = RegionTag::Other;
};

}  // namespace imp

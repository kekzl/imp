#pragma once

// T4 (forward-scratch) of the lifetime taxonomy
// (docs/internals/MEMORY.md §A2/§A3.3).
//
// A LIFO stack over one Region. A forward pass opens a Mark on entry, every
// intermediate takes from the stack, and the Mark's destructor rewinds. Two
// failure modes are structurally impossible: it cannot fragment (LIFO), and it
// cannot leak (the mark unwinds on the exception path too).
//
// It also produces the number the planner actually needs. Today the executor
// derives its shared workspace from max(attn, ffn, moe, ssm) heuristics
// recomputed in three places; a stack reports its true high-water mark from
// warmup, so the plan can size it from a measurement instead of an estimate.
//
// This is what removes the per-request cudaMallocAsync traffic that measures at
// +190 MiB of steady-state allocation across all three reference configs — the
// entire I2 violation surface.

#include "memory/backend.h"
#include "memory/span.h"

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace imp {

class ScratchStack {
public:
    ScratchStack() = default;
    ~ScratchStack();

    ScratchStack(const ScratchStack&) = delete;
    ScratchStack& operator=(const ScratchStack&) = delete;

    [[nodiscard]] MemError open(Backend& backend, size_t capacity, RegionTag tag);
    void close();
    bool is_open() const;

    // RAII rewind point. Destroying a Mark returns everything taken since it
    // was created. Marks must be destroyed in reverse creation order; anything
    // else is a programming error and is caught, not tolerated.
    class Mark {
    public:
        Mark() = default;
        ~Mark();
        Mark(Mark&& o) noexcept : stack_(o.stack_), offset_(o.offset_), depth_(o.depth_) {
            o.stack_ = nullptr;
        }
        Mark& operator=(Mark&&) = delete;
        Mark(const Mark&) = delete;
        Mark& operator=(const Mark&) = delete;

    private:
        friend class ScratchStack;
        Mark(ScratchStack* s, size_t offset, uint32_t depth)
            : stack_(s), offset_(offset), depth_(depth) {}

        ScratchStack* stack_ = nullptr;
        size_t offset_ = 0;
        uint32_t depth_ = 0;
    };

    [[nodiscard]] Mark mark();

    // Empty span when exhausted — the caller decides (I6). A kernel that
    // cannot get its scratch falls back or the request is rejected; nothing
    // reaches for the driver.
    StableSpan<std::byte> take_bytes(size_t bytes, size_t alignment = 256);

    template <class T>
    StableSpan<T> take(size_t count, size_t alignment = alignof(T) < 256 ? 256 : alignof(T)) {
        auto raw = take_bytes(count * sizeof(T), alignment);
        if (raw.empty())
            return StableSpan<T>();
        return raw.as<T>();
    }

    size_t capacity() const;
    size_t used() const;
    // Peak `used` since open(). This is the plan input.
    size_t high_water() const;
    // Takes that returned empty because the stack was exhausted. Non-zero
    // means the plan under-provisioned this stack — a report, not a crash.
    uint64_t exhaustion_count() const;

private:
    friend class Mark;
    void rewind_(size_t offset, uint32_t depth);

    mutable std::mutex mu_;
    Region region_;
    size_t offset_ = 0;
    size_t high_water_ = 0;
    uint32_t depth_ = 0;
    uint64_t exhaustions_ = 0;
};

}  // namespace imp

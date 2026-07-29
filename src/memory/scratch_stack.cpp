#include "memory/scratch_stack.h"
#include "core/logging.h"

#include <algorithm>
#include <cstdlib>

namespace imp {

namespace {
size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }
}  // namespace

ScratchStack::~ScratchStack() { close(); }

MemError ScratchStack::open(Backend& backend, size_t capacity, RegionTag tag) {
    std::lock_guard<std::mutex> lock(mu_);
    if (region_.valid())
        return MemError::InvalidArgument;
    if (capacity == 0)
        return MemError::InvalidArgument;

    auto res = backend.acquire(capacity, 256, tag);
    if (!res)
        return res.error;
    region_ = std::move(res.region);
    offset_ = 0;
    high_water_ = 0;
    depth_ = 0;
    exhaustions_ = 0;
    return MemError::Ok;
}

void ScratchStack::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid())
        return;
    if (depth_ != 0) {
        IMP_LOG_ERROR("ScratchStack::close with %u live marks — a forward pass did not unwind",
                      depth_);
#ifndef NDEBUG
        std::abort();
#endif
    }
    region_.reset();
    offset_ = 0;
    depth_ = 0;
}

bool ScratchStack::is_open() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.valid();
}

ScratchStack::Mark ScratchStack::mark() {
    std::lock_guard<std::mutex> lock(mu_);
    return Mark(this, offset_, ++depth_);
}

void ScratchStack::rewind_(size_t offset, uint32_t depth) {
    std::lock_guard<std::mutex> lock(mu_);
    if (depth != depth_) {
        // Out-of-order rewind. Tolerating it would silently hand live scratch
        // back to the next taker, which is a use-after-free with no crash.
        IMP_LOG_ERROR("ScratchStack: mark released out of order (depth %u, expected %u) — LIFO "
                      "discipline violated",
                      depth, depth_);
#ifndef NDEBUG
        std::abort();
#endif
        return;
    }
    offset_ = offset;
    --depth_;
}

StableSpan<std::byte> ScratchStack::take_bytes(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
        return StableSpan<std::byte>();

    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid())
        return StableSpan<std::byte>();

    const size_t start = align_up(offset_, alignment);
    if (start > region_.committed() || bytes > region_.committed() - start) {
        ++exhaustions_;
        return StableSpan<std::byte>();
    }

    auto* base = static_cast<std::byte*>(region_.base()) + start;
    offset_ = start + bytes;
    high_water_ = std::max(high_water_, offset_);
    return StableSpan<std::byte>(detail::StableKey{}, base, bytes);
}

size_t ScratchStack::capacity() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.committed();
}

size_t ScratchStack::used() const {
    std::lock_guard<std::mutex> lock(mu_);
    return offset_;
}

size_t ScratchStack::high_water() const {
    std::lock_guard<std::mutex> lock(mu_);
    return high_water_;
}

uint64_t ScratchStack::exhaustion_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exhaustions_;
}

ScratchStack::Mark::~Mark() {
    if (stack_)
        stack_->rewind_(offset_, depth_);
}

}  // namespace imp

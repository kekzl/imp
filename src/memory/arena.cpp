#include "memory/arena.h"

#include <algorithm>

namespace imp {

namespace {
size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }
}  // namespace

ArenaAllocator::~ArenaAllocator() { close(); }

MemError ArenaAllocator::open(Backend& backend, size_t capacity, RegionTag tag) {
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
    tag_ = tag;
    return MemError::Ok;
}

void ArenaAllocator::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid())
        return;
    region_.reset();
    offset_ = 0;
    high_water_ = 0;
    ++generation_;
}

void ArenaAllocator::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    offset_ = 0;
    ++generation_;
}

StableSpan<std::byte> ArenaAllocator::take_bytes(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
        return StableSpan<std::byte>();

    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid())
        return StableSpan<std::byte>();

    const size_t start = align_up(offset_, alignment);
    if (start > region_.committed() || bytes > region_.committed() - start)
        return StableSpan<std::byte>();

    auto* base = static_cast<std::byte*>(region_.base()) + start;
    offset_ = start + bytes;
    high_water_ = std::max(high_water_, offset_);
    return StableSpan<std::byte>(detail::StableKey{}, base, bytes);
}

size_t ArenaAllocator::capacity() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.committed();
}

size_t ArenaAllocator::used() const {
    std::lock_guard<std::mutex> lock(mu_);
    return offset_;
}

size_t ArenaAllocator::remaining() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.committed() > offset_ ? region_.committed() - offset_ : 0;
}

size_t ArenaAllocator::high_water() const {
    std::lock_guard<std::mutex> lock(mu_);
    return high_water_;
}

uint64_t ArenaAllocator::generation() const {
    std::lock_guard<std::mutex> lock(mu_);
    return generation_;
}

}  // namespace imp

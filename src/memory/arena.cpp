#include "memory/arena.h"

#include "memory/mem_account.h"
#include "memory/vram_query.h"

#include <algorithm>

namespace imp {

namespace {
size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }
}  // namespace

ArenaAllocator::~ArenaAllocator() { close(); }

MemError ArenaAllocator::open(Backend& backend, size_t capacity, RegionTag tag, bool lazy) {
    std::lock_guard<std::mutex> lock(mu_);
    if (region_.valid())
        return MemError::InvalidArgument;
    if (capacity == 0)
        return MemError::InvalidArgument;

    if (lazy) {
        auto res = backend.acquire_growable(capacity, 0, 256, tag);
        if (res) {
            region_ = std::move(res.region);
            lazy_ = true;
            backend_ = &backend;
            vram_reserved_uncommitted_add(static_cast<std::ptrdiff_t>(region_.reserved()));
            offset_ = 0;
            high_water_ = 0;
            tag_ = tag;
            return MemError::Ok;
        }
        if (res.error != MemError::NotGrowable)
            return res.error;
        // No VMM on this device: a fixed region is the honest fallback.
    }
    auto res = backend.acquire(capacity, 256, tag);
    if (!res)
        return res.error;
    region_ = std::move(res.region);
    lazy_ = false;
    backend_ = nullptr;
    offset_ = 0;
    high_water_ = 0;
    tag_ = tag;
    return MemError::Ok;
}

void ArenaAllocator::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid())
        return;
    if (lazy_) {
        vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(region_.reserved() - region_.committed()));
        MemAccount::instance().note("engine_arena", -static_cast<std::ptrdiff_t>(region_.committed()));
    }
    region_.reset();
    lazy_ = false;
    backend_ = nullptr;
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
    if (start > region_.reserved() || bytes > region_.reserved() - start)
        return StableSpan<std::byte>();

    // Lazy: back the prefix this take reaches into. The commit rounds up to
    // the backend's granule, so a staircase of small takes commits once per
    // granule rather than once per take.
    if (const size_t end = start + bytes; end > region_.committed()) {
        if (!lazy_ || !backend_)
            return StableSpan<std::byte>();
        const size_t before = region_.committed();
        const size_t granule = std::max<size_t>(backend_->granularity(), 256);
        const size_t target = std::min(align_up(end, granule), region_.reserved());
        const MemError e = backend_->commit(region_, target);
        const size_t added = region_.committed() > before ? region_.committed() - before : 0;
        if (added > 0) {
            vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(added));
            MemAccount::instance().note("engine_arena", static_cast<std::ptrdiff_t>(added));
        }
        if (e != MemError::Ok || region_.committed() < end)
            return StableSpan<std::byte>();
    }

    auto* base = static_cast<std::byte*>(region_.base()) + start;
    offset_ = start + bytes;
    high_water_ = std::max(high_water_, offset_);
    return StableSpan<std::byte>(detail::StableKey{}, base, bytes);
}

size_t ArenaAllocator::capacity() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.reserved();
}

size_t ArenaAllocator::committed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.committed();
}

size_t ArenaAllocator::used() const {
    std::lock_guard<std::mutex> lock(mu_);
    return offset_;
}

size_t ArenaAllocator::remaining() const {
    std::lock_guard<std::mutex> lock(mu_);
    return region_.reserved() > offset_ ? region_.reserved() - offset_ : 0;
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

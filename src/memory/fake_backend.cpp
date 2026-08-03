#include "memory/fake_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace imp {

namespace {

void* aligned_base(void* raw, size_t alignment) {
    auto addr = reinterpret_cast<uintptr_t>(raw);
    auto aligned = (addr + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
    return reinterpret_cast<void*>(aligned);
}

}  // namespace

FakeBackend::FakeBackend(size_t capacity_bytes, bool growable)
    : capacity_(capacity_bytes), growable_(growable) {
    stats_.capacity = capacity_bytes;
}

FakeBackend::~FakeBackend() {
    // A test that leaks a Region would otherwise leak host memory too. Regions
    // are RAII, so anything still here is the test's bug — free it so the leak
    // shows up as a failed conservation assert, not as an ASan report.
    for (auto& l : live_)
        std::free(l.raw);
    live_.clear();
    for (auto& q : quarantine_)
        std::free(q.raw);
    quarantine_.clear();
}

void FakeBackend::fail_acquisition(uint64_t nth, MemError err) {
    std::lock_guard<std::mutex> lock(mu_);
    fail_at_ = nth ? acquire_ordinal_ + nth : 0;
    fail_err_ = err;
}

size_t FakeBackend::journal_live_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    long long live = 0;
    for (const auto& e : journal_) {
        switch (e.op) {
            case AllocEvent::Op::Acquire:
            case AllocEvent::Op::Commit:
                live += static_cast<long long>(e.bytes);
                break;
            case AllocEvent::Op::Release:
            case AllocEvent::Op::Decommit:
                live -= static_cast<long long>(e.bytes);
                break;
        }
    }
    return live > 0 ? static_cast<size_t>(live) : 0;
}

size_t FakeBackend::live_regions() const {
    std::lock_guard<std::mutex> lock(mu_);
    return live_.size();
}

bool FakeBackend::is_poisoned(const void* base, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(base);
    for (size_t i = 0; i < bytes; ++i)
        if (p[i] != kPoison)
            return false;
    return true;
}

BackendStats FakeBackend::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

void FakeBackend::record_(AllocEvent::Op op, RegionTag tag, size_t bytes, const void* base) {
    journal_.push_back(AllocEvent{++seq_, op, alloc_phase(), tag, bytes, base});
}

MemError FakeBackend::do_acquire(size_t bytes, size_t alignment, RegionTag tag, void** out_base,
                                 size_t* out_reserved) {
    std::lock_guard<std::mutex> lock(mu_);
    ++acquire_ordinal_;
    if (fail_at_ && acquire_ordinal_ >= fail_at_) {
        fail_at_ = 0;
        return fail_err_;
    }

    if (capacity_ && stats_.live_bytes + bytes > capacity_)
        return MemError::OutOfMemory;

    void* raw = std::malloc(bytes + alignment);
    if (!raw)
        return MemError::OutOfMemory;
    void* base = aligned_base(raw, alignment);

    live_.push_back(Live{raw, base, bytes, bytes, tag});
    stats_.live_bytes += bytes;
    stats_.reserved_bytes += bytes;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
    stats_.acquire_count++;
    record_(AllocEvent::Op::Acquire, tag, bytes, base);

    *out_base = base;
    *out_reserved = bytes;
    return MemError::Ok;
}

MemError FakeBackend::do_acquire_growable(size_t reserve_bytes, size_t initial_commit,
                                          size_t alignment, RegionTag tag, void** out_base) {
    if (!growable_)
        return MemError::NotGrowable;

    std::lock_guard<std::mutex> lock(mu_);
    ++acquire_ordinal_;
    if (fail_at_ && acquire_ordinal_ >= fail_at_) {
        fail_at_ = 0;
        return fail_err_;
    }

    // Only the COMMITTED bytes count against capacity — that is the whole
    // point of a reservation, and the property the KV pool will rely on.
    if (capacity_ && stats_.live_bytes + initial_commit > capacity_)
        return MemError::OutOfMemory;

    // The reservation is backed for real so the base address is stable for the
    // full range, exactly as cuMemAddressReserve promises on device.
    void* raw = std::malloc(reserve_bytes + alignment);
    if (!raw)
        return MemError::OutOfMemory;
    void* base = aligned_base(raw, alignment);

    live_.push_back(Live{raw, base, initial_commit, reserve_bytes, tag});
    stats_.live_bytes += initial_commit;
    stats_.reserved_bytes += reserve_bytes;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
    stats_.acquire_count++;
    record_(AllocEvent::Op::Acquire, tag, initial_commit, base);

    *out_base = base;
    return MemError::Ok;
}

MemError FakeBackend::commit(Region& region, size_t new_committed) {
    if (!growable_)
        return MemError::NotGrowable;
    if (!region.valid())
        return MemError::InvalidArgument;
    if (new_committed > region.reserved())
        return MemError::InvalidArgument;

    std::lock_guard<std::mutex> lock(mu_);
    auto it = std::find_if(live_.begin(), live_.end(),
                           [&](const Live& l) { return l.base == region.base(); });
    if (it == live_.end())
        return MemError::InvalidArgument;

    const size_t old = it->committed;
    if (new_committed > old) {
        const size_t grow = new_committed - old;
        if (capacity_ && stats_.live_bytes + grow > capacity_)
            return MemError::OutOfMemory;
        stats_.live_bytes += grow;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
        record_(AllocEvent::Op::Commit, it->tag, grow, it->base);
    } else if (new_committed < old) {
        const size_t shrink = old - new_committed;
        // Decommitted pages read back as poison, mirroring the device where
        // they are unmapped and touching them faults.
        std::memset(static_cast<char*>(it->base) + new_committed, kPoison, shrink);
        stats_.live_bytes -= std::min(stats_.live_bytes, shrink);
        record_(AllocEvent::Op::Decommit, it->tag, shrink, it->base);
    }
    it->committed = new_committed;
    set_committed_(region, new_committed);
    return MemError::Ok;
}

void FakeBackend::do_release(void* base, size_t committed, size_t reserved, RegionTag tag) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = std::find_if(live_.begin(), live_.end(), [&](const Live& l) { return l.base == base; });
    if (it == live_.end())
        return;  // double release — conservation replay will show it

    // Poison, then quarantine rather than free: a test that wants to prove the
    // buffer was really returned has to be able to read it back. Freeing here
    // would make is_poisoned() a use-after-free in the test itself.
    std::memset(it->base, kPoison, it->reserved);
    quarantine_.push_back(*it);
    live_.erase(it);
    while (quarantine_.size() > kQuarantineDepth) {
        std::free(quarantine_.front().raw);
        quarantine_.erase(quarantine_.begin());
    }

    stats_.live_bytes -= std::min(stats_.live_bytes, committed);
    stats_.reserved_bytes -= std::min(stats_.reserved_bytes, reserved);
    stats_.release_count++;
    record_(AllocEvent::Op::Release, tag, committed, base);
}

}  // namespace imp

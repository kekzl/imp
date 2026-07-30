// GraphSlotPool — the T2 slot pool behind the conditional graph loop
// (docs/MEMORY_ARCHITECTURE.md A7 step 5.3).
//
// CPU lane on purpose. The pool's device side goes through Backend and its
// pinned-host side through HostPinnedAllocator, so both halves substitute and
// the layout arithmetic — the part that can silently hand out overlapping
// buffers — is testable without a GPU. CI has no GPU runner, so a GPU-lane
// test here would never actually run.

#include <gtest/gtest.h>

#include "memory/fake_backend.h"
#include "memory/graph_slots.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using namespace imp;

namespace {

// Host-heap stand-in for cudaHostAlloc(cudaHostAllocMapped). The "device view"
// is the same pointer, which is all the pool's arithmetic needs.
class FakeHostPinned final : public HostPinnedAllocator {
public:
    bool alloc(size_t bytes, HostPinnedKind kind, void** out_host, void** out_device) override {
        if (fail_next_) {
            fail_next_ = false;
            return false;
        }
        void* p = std::malloc(bytes);
        if (!p)
            return false;
        bytes_ = bytes;
        kind_ = kind;
        ++allocs_;
        *out_host = p;
        // Only a mapped allocation has a device view, same as the real thing.
        if (out_device)
            *out_device = (kind == HostPinnedKind::Mapped) ? p : nullptr;
        return true;
    }
    void free(void* host) override {
        std::free(host);
        ++frees_;
    }

    void fail_next() { fail_next_ = true; }
    int allocs() const { return allocs_; }
    int frees() const { return frees_; }
    size_t bytes() const { return bytes_; }
    HostPinnedKind kind() const { return kind_; }

private:
    bool fail_next_ = false;
    int allocs_ = 0;
    int frees_ = 0;
    size_t bytes_ = 0;
    HostPinnedKind kind_ = HostPinnedKind::Plain;
};

GraphSlotCaps default_caps() {
    GraphSlotCaps c;
    c.max_steps = 256;
    c.penalty_slots = 512;
    c.stop_ids = 8;
    return c;
}

// Every device buffer a view points at, with its length.
std::vector<std::pair<const std::byte*, size_t>> device_extents(const GraphSlotView& v,
                                                                const GraphSlotCaps& caps) {
    auto E = [](const void* p, size_t n) {
        return std::make_pair(static_cast<const std::byte*>(p), n);
    };
    return {
        E(v.sample_scratch, kGraphSlotSampleScratchBytes),
        E(v.position, sizeof(int)),
        E(v.context_len, sizeof(int)),
        E(v.step_counter, sizeof(int)),
        E(v.step_limit, sizeof(int)),
        E(v.think_limit, sizeof(int)),
        E(v.think_count, sizeof(int)),
        E(v.in_think, sizeof(int)),
        E(v.think_exit_step, sizeof(int)),
        E(v.content_after_think, sizeof(int)),
        E(v.penalty_count, sizeof(int)),
        E(v.stop_ids, static_cast<size_t>(caps.stop_ids) * sizeof(int32_t)),
        E(v.penalty_ring, static_cast<size_t>(caps.penalty_slots) * sizeof(int32_t)),
    };
}

bool overlaps(const std::pair<const std::byte*, size_t>& a,
              const std::pair<const std::byte*, size_t>& b) {
    return a.first < b.first + b.second && b.first < a.first + a.second;
}

}  // namespace

TEST(GraphSlotPool, OpensAndReportsItsShape) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;

    ASSERT_EQ(pool.open(backend, host, default_caps(), 4), MemError::Ok);
    EXPECT_TRUE(pool.is_open());
    EXPECT_EQ(pool.num_slots(), 4);
    EXPECT_EQ(pool.free_slots(), 4);
    EXPECT_GT(pool.device_bytes(), 0u);
    EXPECT_EQ(host.allocs(), 1);  // ONE pinned allocation for all slots
    EXPECT_EQ(pool.declines(), 0u);
}

TEST(GraphSlotPool, RejectsNonsenseCapacities) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;

    GraphSlotCaps c = default_caps();
    c.max_steps = 0;
    EXPECT_EQ(pool.open(backend, host, c, 4), MemError::InvalidArgument);
    EXPECT_EQ(pool.open(backend, host, default_caps(), 0), MemError::InvalidArgument);
    EXPECT_FALSE(pool.is_open());
}

TEST(GraphSlotPool, OpeningTwiceIsAnError) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;

    ASSERT_EQ(pool.open(backend, host, default_caps(), 2), MemError::Ok);
    EXPECT_EQ(pool.open(backend, host, default_caps(), 2), MemError::InvalidArgument);
}

TEST(GraphSlotPool, PinnedHostFailureLeavesThePoolClosed) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;

    host.fail_next();
    EXPECT_EQ(pool.open(backend, host, default_caps(), 4), MemError::OutOfMemory);
    EXPECT_FALSE(pool.is_open());
    // And the device region it had already taken went back, rather than
    // stranding a Region on a failed open.
    EXPECT_EQ(backend.live_regions(), 0u);
}

// The one that would catch a layout arithmetic bug: no two buffers in a slot,
// and no two slots, may overlap — and everything must sit inside the region.
TEST(GraphSlotPool, SlotBuffersDoNotOverlapWithinOrAcrossSlots) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 4), MemError::Ok);

    std::vector<GraphSlotLease> leases;
    std::vector<std::pair<const std::byte*, size_t>> all;
    for (int i = 0; i < 4; ++i) {
        GraphSlotLease l = pool.acquire(caps);
        ASSERT_TRUE(l.valid()) << "slot " << i;
        for (auto& e : device_extents(l.view(), caps))
            all.push_back(e);
        leases.push_back(std::move(l));
    }

    for (size_t i = 0; i < all.size(); ++i) {
        for (size_t j = i + 1; j < all.size(); ++j) {
            EXPECT_FALSE(overlaps(all[i], all[j]))
                << "buffers " << i << " and " << j << " overlap";
        }
    }

    // Every extent inside the pool's device allocation.
    const std::byte* lo = all[0].first;
    const std::byte* hi = all[0].first + all[0].second;
    for (auto& e : all) {
        lo = std::min(lo, e.first);
        hi = std::max(hi, e.first + e.second);
    }
    EXPECT_LE(static_cast<size_t>(hi - lo), pool.device_bytes());
}

TEST(GraphSlotPool, HostBuffersDoNotOverlapAcrossSlots) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 3), MemError::Ok);

    std::vector<GraphSlotLease> leases;
    std::vector<std::pair<const std::byte*, size_t>> all;
    for (int i = 0; i < 3; ++i) {
        GraphSlotLease l = pool.acquire(caps);
        ASSERT_TRUE(l.valid());
        const GraphSlotView& v = l.view();
        auto E = [](const void* p, size_t n) {
            return std::make_pair(static_cast<const std::byte*>(p), n);
        };
        all.push_back(E(v.h_ring, static_cast<size_t>(caps.max_steps) * sizeof(int32_t)));
        all.push_back(E(v.h_step_counter, sizeof(int)));
        all.push_back(E(v.h_burst_done, sizeof(int)));
        all.push_back(E(v.h_decode_scratch, sizeof(int32_t)));
        leases.push_back(std::move(l));
    }

    for (size_t i = 0; i < all.size(); ++i)
        for (size_t j = i + 1; j < all.size(); ++j)
            EXPECT_FALSE(overlaps(all[i], all[j])) << i << " vs " << j;

    // The mapped host buffers are writable through the host pointer — this is
    // what the runner does to reset the ring before a burst.
    for (auto& l : leases) {
        std::memset(l.view().h_ring, 0, static_cast<size_t>(caps.max_steps) * sizeof(int32_t));
        *l.view().h_step_counter = 0;
        *l.view().h_burst_done = 0;
    }
}

// I3: a slot that comes back must come back at the same address, because the
// previous burst's captured graph baked those addresses in.
TEST(GraphSlotPool, ReleasedSlotReturnsWithIdenticalAddresses) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 1), MemError::Ok);

    void* first_scratch = nullptr;
    int32_t* first_ring = nullptr;
    int32_t* first_h_ring = nullptr;
    {
        GraphSlotLease l = pool.acquire(caps);
        ASSERT_TRUE(l.valid());
        first_scratch = l.view().sample_scratch;
        first_ring = l.view().penalty_ring;
        first_h_ring = l.view().h_ring;
        EXPECT_EQ(pool.free_slots(), 0);
    }
    EXPECT_EQ(pool.free_slots(), 1);  // the lease returned it on destruction

    GraphSlotLease again = pool.acquire(caps);
    ASSERT_TRUE(again.valid());
    EXPECT_EQ(again.view().sample_scratch, first_scratch);
    EXPECT_EQ(again.view().penalty_ring, first_ring);
    EXPECT_EQ(again.view().h_ring, first_h_ring);
}

TEST(GraphSlotPool, DeclinesWhenExhaustedAndSaysSo) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 2), MemError::Ok);

    GraphSlotLease a = pool.acquire(caps);
    GraphSlotLease b = pool.acquire(caps);
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());

    GraphSlotLease c = pool.acquire(caps);
    EXPECT_FALSE(c.valid());  // caller falls back to allocating for itself
    EXPECT_EQ(pool.declines_exhausted(), 1u);
    EXPECT_EQ(pool.declines_too_small(), 0u);

    a.release();
    GraphSlotLease d = pool.acquire(caps);
    EXPECT_TRUE(d.valid());
}

TEST(GraphSlotPool, DeclinesWhenARequestExceedsACapacity) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 2), MemError::Ok);

    GraphSlotCaps too_many_steps = caps;
    too_many_steps.max_steps = caps.max_steps + 1;
    EXPECT_FALSE(pool.acquire(too_many_steps).valid());

    GraphSlotCaps too_much_penalty = caps;
    too_much_penalty.penalty_slots = caps.penalty_slots + 1;
    EXPECT_FALSE(pool.acquire(too_much_penalty).valid());

    GraphSlotCaps too_many_stops = caps;
    too_many_stops.stop_ids = caps.stop_ids + 1;
    EXPECT_FALSE(pool.acquire(too_many_stops).valid());

    EXPECT_EQ(pool.declines_too_small(), 3u);
    EXPECT_EQ(pool.declines_exhausted(), 0u);
    EXPECT_EQ(pool.free_slots(), 2);  // declines take nothing
}

TEST(GraphSlotPool, ClosedPoolDeclinesWithoutCounting) {
    GraphSlotPool pool;
    EXPECT_FALSE(pool.acquire(default_caps()).valid());
    EXPECT_EQ(pool.declines(), 0u);
}

TEST(GraphSlotPool, MovingALeaseDoesNotReleaseTwice) {
    FakeBackend backend;
    FakeHostPinned host;
    GraphSlotPool pool;
    const GraphSlotCaps caps = default_caps();
    ASSERT_EQ(pool.open(backend, host, caps, 1), MemError::Ok);

    {
        GraphSlotLease a = pool.acquire(caps);
        ASSERT_TRUE(a.valid());
        GraphSlotLease b = std::move(a);
        EXPECT_FALSE(a.valid());
        EXPECT_TRUE(b.valid());
        EXPECT_EQ(pool.free_slots(), 0);
    }
    EXPECT_EQ(pool.free_slots(), 1);

    // Move-assign over a live lease releases the one being overwritten.
    GraphSlotLease held = pool.acquire(caps);
    ASSERT_TRUE(held.valid());
    EXPECT_EQ(pool.free_slots(), 0);
    held = GraphSlotLease{};
    EXPECT_EQ(pool.free_slots(), 1);
}

TEST(GraphSlotPool, CloseReturnsTheDeviceRegionAndTheHostBlock) {
    FakeBackend backend;
    FakeHostPinned host;
    {
        GraphSlotPool pool;
        ASSERT_EQ(pool.open(backend, host, default_caps(), 4), MemError::Ok);
        EXPECT_EQ(backend.live_regions(), 1u);
    }
    EXPECT_EQ(backend.live_regions(), 0u);
    EXPECT_EQ(host.frees(), 1);
    EXPECT_EQ(backend.journal_live_bytes(), 0u);
}

// PinnedBuffer — the owner for T5's engine-persistent half
// (docs/MEMORY_ARCHITECTURE.md §A2, memory/host_pinned.h).
//
// CPU lane on purpose, and it is the whole reason HostPinnedAllocator is an
// interface: the 26 call sites this type exists for are pinned-host buffers in
// the decode path, and the property that has to hold — freed exactly once, never
// after a move — is an ownership property, not a CUDA one. A GPU-lane test would
// never run in CI (no GPU runner) and would prove less.

#include <gtest/gtest.h>

#include "memory/host_pinned.h"

#include <cstdlib>
#include <utility>
#include <vector>

using namespace imp;

namespace {

// Host-heap stand-in for cudaHostAlloc. Counts every edge so a leak or a double
// free is an assertion rather than something valgrind might notice later.
class CountingHostPinned final : public HostPinnedAllocator {
public:
    bool alloc(size_t bytes, HostPinnedKind kind, void** out_host, void** out_device) override {
        if (fail_next_) {
            fail_next_ = false;
            return false;
        }
        void* p = std::malloc(bytes ? bytes : 1);
        if (!p)
            return false;
        ++allocs_;
        last_bytes_ = bytes;
        last_kind_ = kind;
        *out_host = p;
        if (out_device)
            *out_device = (kind == HostPinnedKind::Mapped) ? p : nullptr;
        return true;
    }
    void free(void* host) override {
        ++frees_;
        freed_.push_back(host);
        std::free(host);
    }

    void fail_next() { fail_next_ = true; }
    int allocs() const { return allocs_; }
    int frees() const { return frees_; }
    size_t last_bytes() const { return last_bytes_; }
    HostPinnedKind last_kind() const { return last_kind_; }
    const std::vector<void*>& freed() const { return freed_; }

private:
    bool fail_next_ = false;
    int allocs_ = 0;
    int frees_ = 0;
    size_t last_bytes_ = 0;
    HostPinnedKind last_kind_ = HostPinnedKind::Plain;
    std::vector<void*> freed_;
};

}  // namespace

TEST(PinnedBuffer, DefaultConstructedIsEmptyAndFreesNothing) {
    CountingHostPinned a;
    {
        PinnedBuffer b;
        EXPECT_TRUE(b.empty());
        EXPECT_FALSE(static_cast<bool>(b));
        EXPECT_EQ(b.data(), nullptr);
        EXPECT_EQ(b.device(), nullptr);
        EXPECT_EQ(b.bytes(), 0u);
    }
    EXPECT_EQ(a.frees(), 0) << "an empty buffer must not call free — that is the double-free path";
}

TEST(PinnedBuffer, AcquireFreesExactlyOnceAtScopeExit) {
    CountingHostPinned a;
    void* raw = nullptr;
    {
        PinnedBuffer b = PinnedBuffer::acquire(a, 4096);
        ASSERT_FALSE(b.empty());
        EXPECT_EQ(a.allocs(), 1);
        EXPECT_EQ(a.frees(), 0);
        EXPECT_EQ(b.bytes(), 4096u);
        raw = b.data();
    }
    EXPECT_EQ(a.frees(), 1);
    ASSERT_EQ(a.freed().size(), 1u);
    EXPECT_EQ(a.freed()[0], raw) << "it must free the pointer it was given, not some other";
}

// Exhaustion is a value, not an exception and not an abort (I6). Every existing
// pinned-host call site degrades on a null, so the owner has to preserve that.
TEST(PinnedBuffer, FailedAcquireIsAnEmptyBufferNotAThrow) {
    CountingHostPinned a;
    a.fail_next();
    PinnedBuffer b = PinnedBuffer::acquire(a, 1024);
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(a.allocs(), 0);
    b.reset();
    EXPECT_EQ(a.frees(), 0) << "a failed acquire owns nothing and must free nothing";
}

// The fake would happily hand back a 1-byte block for a 0-byte request, so this
// pins the OWNER's behaviour rather than the allocator's: a zero-size request is
// a caller bug, and a buffer whose bytes() disagrees with its allocation is how
// an overrun gets through.
TEST(PinnedBuffer, ZeroBytesIsRefusedWithoutAskingTheAllocator) {
    CountingHostPinned a;
    PinnedBuffer b = PinnedBuffer::acquire(a, 0);
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(a.allocs(), 0) << "the allocator must not even be called";
    EXPECT_EQ(a.frees(), 0);
}

TEST(PinnedBuffer, MoveTransfersOwnershipAndTheSourceFreesNothing) {
    CountingHostPinned a;
    {
        PinnedBuffer src = PinnedBuffer::acquire(a, 256);
        ASSERT_FALSE(src.empty());
        void* raw = src.data();

        PinnedBuffer dst = std::move(src);
        EXPECT_TRUE(src.empty()) << "a moved-from buffer must not still point at the memory";
        EXPECT_EQ(src.bytes(), 0u);
        EXPECT_EQ(dst.data(), raw);
        EXPECT_EQ(dst.bytes(), 256u);
        EXPECT_EQ(a.frees(), 0) << "moving is not releasing";
    }
    EXPECT_EQ(a.frees(), 1) << "exactly one free for one allocation, after a move";
}

TEST(PinnedBuffer, MoveAssignmentReleasesTheTargetFirst) {
    CountingHostPinned a;
    PinnedBuffer first = PinnedBuffer::acquire(a, 128);
    PinnedBuffer second = PinnedBuffer::acquire(a, 256);
    ASSERT_EQ(a.allocs(), 2);
    void* first_raw = first.data();

    first = std::move(second);
    EXPECT_EQ(a.frees(), 1) << "the overwritten buffer must be released, not leaked";
    ASSERT_EQ(a.freed().size(), 1u);
    EXPECT_EQ(a.freed()[0], first_raw);
    EXPECT_EQ(first.bytes(), 256u);
}

TEST(PinnedBuffer, SelfMoveAssignmentDoesNotFreeTheBufferItKeeps) {
    CountingHostPinned a;
    PinnedBuffer b = PinnedBuffer::acquire(a, 64);
    void* raw = b.data();
    PinnedBuffer& alias = b;
    b = std::move(alias);
    EXPECT_EQ(a.frees(), 0);
    EXPECT_EQ(b.data(), raw) << "self-move must be a no-op, not a use-after-free";
}

TEST(PinnedBuffer, ResetIsIdempotent) {
    CountingHostPinned a;
    PinnedBuffer b = PinnedBuffer::acquire(a, 512);
    b.reset();
    b.reset();
    b.reset();
    EXPECT_EQ(a.frees(), 1) << "reset twice must not free twice";
    EXPECT_TRUE(b.empty());
}

// Plain vs Mapped is not cosmetic: cudaHostGetDevicePointer fails on a buffer
// that was not allocated mapped, so the kind has to reach the allocator and the
// device view must exist only for Mapped.
TEST(PinnedBuffer, KindReachesTheAllocatorAndOnlyMappedHasADeviceView) {
    CountingHostPinned a;
    PinnedBuffer plain = PinnedBuffer::acquire(a, 32, HostPinnedKind::Plain);
    ASSERT_FALSE(plain.empty());
    EXPECT_EQ(a.last_kind(), HostPinnedKind::Plain);
    EXPECT_EQ(plain.device(), nullptr);

    PinnedBuffer mapped = PinnedBuffer::acquire(a, 32, HostPinnedKind::Mapped);
    ASSERT_FALSE(mapped.empty());
    EXPECT_EQ(a.last_kind(), HostPinnedKind::Mapped);
    EXPECT_NE(mapped.device(), nullptr);
}

TEST(PinnedBuffer, TypedAccessorsViewTheSameMemory) {
    CountingHostPinned a;
    PinnedBuffer b = PinnedBuffer::acquire(a, 4 * sizeof(int), HostPinnedKind::Mapped);
    ASSERT_FALSE(b.empty());
    b.as<int>()[2] = 42;
    EXPECT_EQ(static_cast<const int*>(b.data())[2], 42);
    EXPECT_EQ(b.device_as<int>(), b.device());
}

// The default is Plain. Getting this backwards would silently hand mapped
// memory to every staging buffer that only needs an explicit copy.
TEST(PinnedBuffer, DefaultKindIsPlain) {
    CountingHostPinned a;
    PinnedBuffer b = PinnedBuffer::acquire(a, 16);
    ASSERT_FALSE(b.empty());
    EXPECT_EQ(a.last_kind(), HostPinnedKind::Plain);
}

// The tests above substitute the allocator, so none of them exercises the real
// one at all. This one does, and what it can prove is bounded — stated here
// rather than left to be assumed:
//
//   - CPU-only lane (what CI runs): cudaHostAlloc fails without a device, and the
//     contract is "empty buffer, no crash" (I6). That is the assertion.
//   - GPU box: a Mapped buffer carries a device view and a Plain one does not.
//
// What NO lane here catches, verified by mutation: a build that passes
// cudaHostAllocMapped for BOTH kinds. PinnedBuffer only ever exposes a device
// view for Mapped, and under UVA the underlying query succeeds either way — see
// the HostPinnedKind comment in host_pinned.h.
TEST(PinnedBuffer, RealAllocatorEitherFailsCleanlyOrMapsOnlyWhenAsked) {
    HostPinnedAllocator& a = cuda_host_pinned_allocator();

    PinnedBuffer plain = PinnedBuffer::acquire(a, 256, HostPinnedKind::Plain);
    if (plain.empty()) {
        // CPU-only lane: the driver said no and nothing blew up.
        EXPECT_EQ(plain.data(), nullptr);
        EXPECT_EQ(plain.bytes(), 0u);
    } else {
        EXPECT_EQ(plain.device(), nullptr)
            << "a Plain allocation must not carry a device view — that is the flag mutation";
        PinnedBuffer mapped = PinnedBuffer::acquire(a, 256, HostPinnedKind::Mapped);
        ASSERT_FALSE(mapped.empty());
        EXPECT_NE(mapped.device(), nullptr);
    }
}

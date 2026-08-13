// PinnedBuffer — the owner for T5's engine-persistent half
// (docs/internals/MEMORY.md §A2, memory/host_pinned.h).
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

// ── HostRegistration ─────────────────────────────────────────────────
// Pinning memory imp does not own. The registrar is substitutable for a reason
// found by mutation testing: with the real one, a device-less machine fails every
// registration, so every path collapses to "empty" and a reset() that forgot to
// clear its pointer passed. Against the fake below the ownership is actually
// pinned. The leak this type prevents is the asymmetric one — a page-locked
// region left behind by an early return does not show up as missing bytes.

// Host-heap stand-in for cudaHostRegister: it never touches the driver, so the
// ownership paths below are reachable on a machine with no GPU. Without this the
// CPU lane proves nothing here — mutation testing confirmed a reset() that
// forgot to clear its pointer and a dropped null guard both survived.
class FakeRegistrar final : public HostRegistrar {
public:
    bool register_read_only(void* ptr, size_t bytes) override {
        if (fail_next_) {
            fail_next_ = false;
            return false;
        }
        ++registers_;
        last_ptr_ = ptr;
        last_bytes_ = bytes;
        return true;
    }
    void unregister(void* ptr) override {
        ++unregisters_;
        unregistered_.push_back(ptr);
    }
    void fail_next() { fail_next_ = true; }
    int registers() const { return registers_; }
    int unregisters() const { return unregisters_; }
    void* last_ptr() const { return last_ptr_; }
    size_t last_bytes() const { return last_bytes_; }
    const std::vector<void*>& unregistered() const { return unregistered_; }

private:
    bool fail_next_ = false;
    int registers_ = 0;
    int unregisters_ = 0;
    void* last_ptr_ = nullptr;
    size_t last_bytes_ = 0;
    std::vector<void*> unregistered_;
};

TEST(HostRegistration, RegistersOnceAndUnregistersExactlyOnce) {
    FakeRegistrar r;
    std::vector<char> mem(4096);
    {
        HostRegistration h = HostRegistration::acquire_read_only(mem.data(), mem.size(), r);
        ASSERT_FALSE(h.empty());
        EXPECT_EQ(r.registers(), 1);
        EXPECT_EQ(r.unregisters(), 0);
        EXPECT_EQ(h.data(), mem.data());
        EXPECT_EQ(h.bytes(), mem.size());
        EXPECT_EQ(r.last_bytes(), mem.size());
    }
    EXPECT_EQ(r.unregisters(), 1);
    ASSERT_EQ(r.unregistered().size(), 1u);
    EXPECT_EQ(r.unregistered()[0], mem.data());
}

TEST(HostRegistration, ResetIsIdempotentAndDoesNotUnregisterTwice) {
    FakeRegistrar r;
    std::vector<char> mem(1024);
    HostRegistration h = HostRegistration::acquire_read_only(mem.data(), mem.size(), r);
    h.reset();
    h.reset();
    h.reset();
    EXPECT_EQ(r.unregisters(), 1) << "unregistering twice is an error, not a no-op";
    EXPECT_TRUE(h.empty());
}

TEST(HostRegistration, FailedRegistrationOwnsNothing) {
    FakeRegistrar r;
    std::vector<char> mem(1024);
    r.fail_next();
    HostRegistration h = HostRegistration::acquire_read_only(mem.data(), mem.size(), r);
    EXPECT_TRUE(h.empty());
    h.reset();
    EXPECT_EQ(r.unregisters(), 0);
}

TEST(HostRegistration, MovedFromRegistrationDoesNotUnregister) {
    FakeRegistrar r;
    std::vector<char> mem(1024);
    {
        HostRegistration src = HostRegistration::acquire_read_only(mem.data(), mem.size(), r);
        HostRegistration dst = std::move(src);
        EXPECT_TRUE(src.empty());
        EXPECT_FALSE(dst.empty());
        EXPECT_EQ(r.unregisters(), 0);
    }
    EXPECT_EQ(r.unregisters(), 1) << "one registration, one unregister, after a move";
}

TEST(HostRegistration, MoveAssignmentReleasesTheTargetFirst) {
    FakeRegistrar r;
    std::vector<char> a(64), b(64);
    HostRegistration first = HostRegistration::acquire_read_only(a.data(), a.size(), r);
    HostRegistration second = HostRegistration::acquire_read_only(b.data(), b.size(), r);
    first = std::move(second);
    EXPECT_EQ(r.unregisters(), 1);
    ASSERT_EQ(r.unregistered().size(), 1u);
    EXPECT_EQ(r.unregistered()[0], a.data()) << "the overwritten registration must be released";
    EXPECT_EQ(first.data(), b.data());
}

TEST(HostRegistration, NullOrZeroIsRefusedWithoutTouchingTheRegistrar) {
    FakeRegistrar r;
    std::vector<char> mem(4096);
    EXPECT_TRUE(HostRegistration::acquire_read_only(nullptr, 4096, r).empty());
    EXPECT_TRUE(HostRegistration::acquire_read_only(mem.data(), 0, r).empty());
    EXPECT_EQ(r.registers(), 0);
}

TEST(HostRegistration, DefaultConstructedIsEmpty) {
    HostRegistration r;
    EXPECT_TRUE(r.empty());
    EXPECT_FALSE(static_cast<bool>(r));
    EXPECT_EQ(r.data(), nullptr);
    EXPECT_EQ(r.bytes(), 0u);
}

TEST(HostRegistration, EitherRegistersOrStaysEmptyAndSurvivesRelease) {
    std::vector<char> mem(64 * 1024);
    HostRegistration r = HostRegistration::acquire_read_only(mem.data(), mem.size());
    if (r.empty()) {
        // CPU-only lane: no device, so registration fails and nothing blew up.
        EXPECT_EQ(r.bytes(), 0u);
    } else {
        EXPECT_EQ(r.data(), mem.data());
        EXPECT_EQ(r.bytes(), mem.size());
    }
    r.reset();
    r.reset();  // idempotent — a second unregister would be an error
    EXPECT_TRUE(r.empty());
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

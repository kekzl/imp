#pragma once

// T5's engine-persistent half: pinned host memory with an owner
// (docs/MEMORY_ARCHITECTURE.md §A2, and the correction to it recorded there).
//
// `Backend` covers DEVICE memory only, which left every pinned host buffer in
// the engine allocating through the driver directly — 26 acquisition sites in
// 11 files at the time this was written, the largest single class remaining on
// the I1 allowlist and the only one with no tier to move to.
//
// Why it is not simply "T5" as A2 first described it. That row reads "transient
// host-staging, load only, failure mode made impossible: surviving load
// (asserted at phase transition)". Most of these buffers cannot obey that: a
// pinned staging buffer for the per-step D2H gather exists PRECISELY so that it
// is pinned once and reused every decode step, so it must survive into
// `AllocPhase::Serving` by construction. Asserting it away would delete the
// optimisation. So T5 has two halves, and this is the engine-persistent one;
// the transient load staging A2 described keeps its own discipline.
//
// The seam is an interface rather than a free function for the same reason
// `Backend` is: it makes the whole thing testable in the CPU-only CI lane
// (§A6). `graph_slots.h` introduced it for the slot pool; it lives here now
// because it is a tier, not a detail of one pool.

#include <cstddef>
#include <utility>

namespace imp {

// Whether the allocation must also be visible to the device through a mapped
// pointer. The engine uses both: `Mapped` for the buffers a kernel or a
// captured graph reads in place (the conditional-graph ring, step counters),
// `Plain` for staging that is only ever the target of an explicit copy.
//
// Stated honestly, because it was mutation-tested: on a unified-virtual-address
// platform the two are NOT distinguishable through this interface. Only `Mapped`
// is ever handed a device view here, and `cudaHostGetDevicePointer` succeeds
// under UVA even for an allocation made without the mapped flag — so a build
// that wrongly passed `cudaHostAllocMapped` for everything would pass every test
// in tests/test_host_pinned.cpp. The distinction is therefore a statement of
// intent (request mapping only where a device-side view is actually used, and
// keep the seam correct on a platform where it does matter), not a behaviour the
// CPU lane can pin. Whether the flag costs anything measurable on sm_120 is
// unmeasured.
enum class HostPinnedKind { Plain, Mapped };

class HostPinnedAllocator {
public:
    virtual ~HostPinnedAllocator() = default;
    // On success writes the host pointer, and for `Mapped` its device-side
    // view. `out_device` is left null for `Plain`. Never throws: exhaustion is
    // a false return, which every caller must already handle because that is
    // what a failed cudaHostAlloc looked like before (I6).
    [[nodiscard]] virtual bool alloc(size_t bytes, HostPinnedKind kind, void** out_host,
                                     void** out_device) = 0;
    virtual void free(void* host) = 0;
};

// cudaHostAlloc(Default|Mapped) + cudaHostGetDevicePointer + cudaFreeHost.
HostPinnedAllocator& cuda_host_pinned_allocator();

// ─────────────────────────────────────────────────────────────────────
// PinnedBuffer — the owner. Move-only RAII over one pinned host allocation,
// so a call site holds a member instead of an alloc/free pair it has to
// remember to match. Same relationship to HostPinnedAllocator that Region has
// to Backend, and the same reason: the per-object leak stops being possible
// because there is no per-object free left to forget.
//
// An empty buffer is the failure value. `data()` returns null, `bytes()` zero,
// and `operator bool` is false — the state every consumer of these buffers
// already tests for, since a failed pinned allocation has always degraded to a
// slower path rather than being fatal.
// ─────────────────────────────────────────────────────────────────────
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { reset(); }

    PinnedBuffer(PinnedBuffer&& other) noexcept { steal_(std::move(other)); }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            steal_(std::move(other));
        }
        return *this;
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Empty on failure — deliberately not an exception and not an abort.
    static PinnedBuffer acquire(HostPinnedAllocator& alloc, size_t bytes,
                                HostPinnedKind kind = HostPinnedKind::Plain);

    // Release the memory. Idempotent, and safe on a moved-from buffer.
    void reset();

    void* data() const { return host_; }
    template <class T>
    T* as() const {
        return static_cast<T*>(host_);
    }
    // The device-side view. Null unless acquired as Mapped.
    void* device() const { return device_; }
    template <class T>
    T* device_as() const {
        return static_cast<T*>(device_);
    }
    size_t bytes() const { return bytes_; }
    bool empty() const { return host_ == nullptr; }
    explicit operator bool() const { return host_ != nullptr; }

private:
    void steal_(PinnedBuffer&& o) noexcept;

    HostPinnedAllocator* owner_ = nullptr;
    void* host_ = nullptr;
    void* device_ = nullptr;
    size_t bytes_ = 0;
};

}  // namespace imp

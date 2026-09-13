#pragma once

// T5's engine-persistent half: pinned host memory with an owner (MEMORY.md A2).
// `Backend` covers DEVICE memory only, which left every pinned host buffer allocating
// through the driver directly (the largest remaining I1-allowlist class with no tier).
// Not simply "T5" as A2 first described it (that row assumes transient load-only
// staging): a pinned staging buffer for the per-step D2H gather is pinned once and
// reused every decode step, so it must survive into Serving by construction. T5
// therefore has two halves; this is the engine-persistent one, and load-only staging
// keeps its own discipline.
// An interface, like Backend, so the whole thing is testable in the CPU-only CI lane
// (A6); introduced for the slot pool but lives here because it's a tier, not a pool
// detail.

#include <cstddef>
#include <utility>

namespace imp {

// Whether the allocation must also be device-visible through a mapped pointer. `Mapped`
// for buffers a kernel or captured graph reads in place; `Plain` for staging that is
// only ever an explicit copy target.
// Not distinguishable on a UVA platform through this interface alone
// (cudaHostGetDevicePointer succeeds under UVA even without the mapped flag, so a build
// that always passed cudaHostAllocMapped would still pass every test); the distinction
// is a statement of intent, kept correct for a platform where it does matter.
enum class HostPinnedKind { Plain, Mapped };

class HostPinnedAllocator {
public:
    virtual ~HostPinnedAllocator() = default;
    // On success writes the host pointer, and for Mapped its device-side view (out_device
    // left null for Plain). Never throws: exhaustion is a false return, which every caller
    // must already handle (I6).
    [[nodiscard]] virtual bool alloc(size_t bytes, HostPinnedKind kind, void** out_host,
                                     void** out_device) = 0;
    virtual void free(void* host) = 0;
};

// cudaHostAlloc(Default|Mapped) + cudaHostGetDevicePointer + cudaFreeHost.
HostPinnedAllocator& cuda_host_pinned_allocator();

// PinnedBuffer: the owner. Move-only RAII over one pinned host allocation, so a call
// site holds a member instead of an alloc/free pair to remember. Same relationship to
// HostPinnedAllocator that Region has to Backend: no per-object free left to forget.
// An empty buffer is the failure value (data() null, bytes() zero, operator bool false),
// the state every consumer already tests for since a failed pinned allocation has always
// degraded to a slower path rather than being fatal.
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

// HostRegistration: pinning memory imp does NOT own. cudaHostRegister page-locks an
// existing mapping (imp's case: the mmap'd GGUF, read-only, so H2D copies can DMA). Not
// an allocation, only something to un-register, so it is a separate type rather than a
// flag on PinnedBuffer. Same discipline: move-only, releases exactly once, a failed
// registration is an empty object.
// Prevents the asymmetric leak: an early return between register and unregister leaves a
// page-locked region behind that nothing owns and does not show up as missing bytes.
// An interface (like Backend/HostPinnedAllocator) so the CPU-only CI lane can pin the
// ownership behaviour even though every registration fails there without a device; the
// driver call itself is the only untested line.
class HostRegistrar {
public:
    virtual ~HostRegistrar() = default;
    [[nodiscard]] virtual bool register_read_only(void* ptr, size_t bytes) = 0;
    virtual void unregister(void* ptr) = 0;
};

// cudaHostRegister(cudaHostRegisterReadOnly) + cudaHostUnregister.
HostRegistrar& cuda_host_registrar();

class HostRegistration {
public:
    HostRegistration() = default;
    ~HostRegistration() { reset(); }

    HostRegistration(HostRegistration&& o) noexcept : reg_(o.reg_), ptr_(o.ptr_), bytes_(o.bytes_) {
        o.reg_ = nullptr;
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    HostRegistration& operator=(HostRegistration&& o) noexcept {
        if (this != &o) {
            reset();
            reg_ = o.reg_;
            ptr_ = o.ptr_;
            bytes_ = o.bytes_;
            o.reg_ = nullptr;
            o.ptr_ = nullptr;
            o.bytes_ = 0;
        }
        return *this;
    }
    HostRegistration(const HostRegistration&) = delete;
    HostRegistration& operator=(const HostRegistration&) = delete;

    // Empty on failure; the caller carries on with an unpinned mapping, which is
    // what every current call site already does.
    static HostRegistration acquire_read_only(void* ptr, size_t bytes,
                                              HostRegistrar& reg = cuda_host_registrar());

    void reset();

    void* data() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    bool empty() const { return ptr_ == nullptr; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    HostRegistrar* reg_ = nullptr;
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

}  // namespace imp

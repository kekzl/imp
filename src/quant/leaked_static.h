#pragma once

// Owners for process-global statics. A static destructor runs after the CUDA runtime is gone, so
// a static DeviceOwned that still held a buffer at exit would call cudaFree into a dead driver.
// LeakedStatic<T> constructs T and never destroys it; the registered reset hook
// (core/cuda_static_reset.h) is the only release, while the context is alive.
//   StaticDeviceOwned<int> s_counter_own;            // file scope
//   s_counter_own.adopt(p, bytes, nullptr);          // lazy acquire
//   void my_reset() { s_counter_own.reset(); }       // IMP_REGISTER_CUDA_STATIC_RESET(my_reset)
// Lives in quant/ because compute/ and quant/ both use it (layer order compute > quant > memory).

#include <cstddef>
#include <utility>

#include "memory/device_owned.h"

namespace imp {

// Holds a T for the whole process and skips ~T at exit. Not copyable, not movable.
template <typename T>
class LeakedStatic {
public:
    LeakedStatic() : value_() {}
    ~LeakedStatic() {}  // NOLINT(modernize-use-equals-default): value_ is left alive on purpose

    LeakedStatic(const LeakedStatic&) = delete;
    LeakedStatic& operator=(const LeakedStatic&) = delete;

    T& operator*() noexcept { return value_; }
    const T& operator*() const noexcept { return value_; }
    T* operator->() noexcept { return &value_; }
    const T* operator->() const noexcept { return &value_; }

private:
    union {
        T value_;
    };
};

// One device allocation owned by a static. Same calls as DeviceOwned<T>; released only by reset()
// or by taking a new buffer, never at process exit.
template <typename T>
class StaticDeviceOwned {
public:
    StaticDeviceOwned() = default;

    StaticDeviceOwned& operator=(DeviceOwned<T>&& o) noexcept {
        *own_ = std::move(o);
        return *this;
    }

    void adopt(T* ptr, size_t bytes, VRAMAllocator* alloc) noexcept { own_->adopt(ptr, bytes, alloc); }
    void reset() noexcept { own_->reset(); }

    T* get() const noexcept { return own_->get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(*own_); }
    size_t bytes() const noexcept { return own_->bytes(); }

private:
    LeakedStatic<DeviceOwned<T>> own_;
};

}  // namespace imp

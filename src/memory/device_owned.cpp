#include "memory/device_owned.h"

#include <atomic>

namespace imp {

namespace {
std::atomic<DeviceOwnedReleaseHook> g_release_hook{nullptr};
}  // namespace

DeviceOwnedReleaseHook set_device_owned_release_hook(DeviceOwnedReleaseHook hook) {
    return g_release_hook.exchange(hook, std::memory_order_acq_rel);
}

namespace detail {

void device_owned_release(void* ptr, DeviceFamily family, VRAMAllocator* alloc, cudaStream_t stream) {
    if (ptr == nullptr || family == DeviceFamily::None)
        return;
    if (const DeviceOwnedReleaseHook hook = g_release_hook.load(std::memory_order_acquire)) {
        hook(ptr, family, alloc, stream);
        return;
    }
    switch (family) {
        case DeviceFamily::Vram:
            alloc->free(ptr);
            break;
        case DeviceFamily::Plain:
            device_free_all(ptr);
            break;
        case DeviceFamily::Async:
            device_free_all_async(stream, ptr);
            break;
        case DeviceFamily::None:
            break;
    }
}

}  // namespace detail

}  // namespace imp

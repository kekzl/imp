#include "memory/weight_owned.h"

#include "core/logging.h"
#include "memory/mem_account.h"

#include <algorithm>
#include <atomic>
#include <utility>

namespace imp {

namespace {
std::atomic<WeightOwnedReleaseHook> g_release_hook{nullptr};

// The one release of WeightOwned.
cudaError_t weight_owned_release(void* ptr, cudaStream_t stream) {
    if (const WeightOwnedReleaseHook hook = g_release_hook.load(std::memory_order_acquire))
        return hook(ptr, stream);
    return weights_free(ptr, stream);
}
}  // namespace

WeightOwnedReleaseHook set_weight_owned_release_hook(WeightOwnedReleaseHook hook) {
    return g_release_hook.exchange(hook, std::memory_order_acq_rel);
}

WeightOwned::~WeightOwned() { IMP_CUDA_CHECK_LOG(reset()); }

WeightOwned& WeightOwned::operator=(WeightOwned&& o) noexcept {
    if (this != &o) {
        IMP_CUDA_CHECK_LOG(reset());
        steal_(o);
    }
    return *this;
}

WeightOwned WeightOwned::allocate(size_t bytes, cudaStream_t stream, bool in_scope) {
    void* p = nullptr;
    if (bytes == 0 || weights_alloc(&p, bytes, stream, in_scope) != cudaSuccess)
        return {};
    return adopt(p, bytes, stream);
}

WeightOwned WeightOwned::adopt(void* ptr, size_t bytes, cudaStream_t release_stream) noexcept {
    WeightOwned out;
    if (ptr != nullptr) {
        out.ptr_ = ptr;
        out.bytes_ = bytes;
        out.stream_ = release_stream;
    }
    return out;
}

cudaError_t WeightOwned::reset() noexcept {
    void* p = std::exchange(ptr_, nullptr);
    const cudaStream_t s = std::exchange(stream_, nullptr);
    bytes_ = 0;
    return p != nullptr ? weight_owned_release(p, s) : cudaSuccess;
}

void* WeightOwned::release() noexcept {
    bytes_ = 0;
    stream_ = nullptr;
    return std::exchange(ptr_, nullptr);
}

void WeightOwned::steal_(WeightOwned& o) noexcept {
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
    stream_ = std::exchange(o.stream_, nullptr);
}

WeightSet::~WeightSet() {
    cudaError_t first = cudaSuccess;
    if (release_all(nullptr, &first) > 0)
        IMP_CUDA_CHECK_LOG(first);
}

void WeightSet::adopt(void* ptr, size_t bytes) {
    if (ptr != nullptr)
        handles_.push_back(WeightOwned::adopt(ptr, bytes, nullptr));
}

void WeightSet::adopt(WeightOwned&& h) {
    if (!h)
        return;
    h.set_release_stream(nullptr);
    handles_.push_back(std::move(h));
}

WeightOwned WeightSet::extract(const void* ptr) noexcept {
    WeightOwned out;
    if (ptr == nullptr)
        return out;
    const auto it = std::ranges::find_if(handles_, [ptr](const WeightOwned& h) { return h.get() == ptr; });
    if (it != handles_.end()) {
        out = std::move(*it);
        handles_.erase(it);
    }
    return out;
}

bool WeightSet::contains(const void* ptr) const noexcept {
    return ptr != nullptr &&
           std::ranges::any_of(handles_, [ptr](const WeightOwned& h) { return h.get() == ptr; });
}

int WeightSet::release_all(cudaStream_t stream, cudaError_t* first_err) noexcept {
    int failed = 0;
    for (WeightOwned& h : handles_) {
        const cudaError_t e = h.reset_on(stream);
        if (e != cudaSuccess && failed++ == 0 && first_err != nullptr)
            *first_err = e;
    }
    handles_.clear();
    return failed;
}

}  // namespace imp

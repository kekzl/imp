#pragma once

// Owning handles for weight memory (invariant I5): a block from weights_alloc
// (memory/mem_account.h) is released by weights_free and nothing else, so the WEIGHTS account
// and the stream-ordered pool see every release (#834: cudaFree on such a block parks it).
// Separate from DeviceOwned for two reasons: the release reports a cudaError_t (teardown counts
// failures instead of logging 1 line per tensor), and one block is released on different
// streams over its life (the upload stream for a temporary, the executor's stream for a dropped
// source, the null stream at teardown).
//   WeightOwned tmp = WeightOwned::adopt(d_raw, bytes, stream);   // freed on `stream`
//   WeightSet set; set.adopt(d_out, bytes);                       // freed at teardown, null stream
//   WeightOwned src = set.extract(ptr); (void)src.reset_on(stream);  // a source leaves early
// Move-only: a copy would be a double free.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace imp {

// Test seam (CPU lane has no device): a non-null hook receives every release instead of
// weights_free. Returns the previous hook. Never set in production.
using WeightOwnedReleaseHook = cudaError_t (*)(void* ptr, cudaStream_t stream);
WeightOwnedReleaseHook set_weight_owned_release_hook(WeightOwnedReleaseHook hook);

class WeightOwned {
public:
    WeightOwned() = default;
    ~WeightOwned();

    WeightOwned(const WeightOwned&) = delete;
    WeightOwned& operator=(const WeightOwned&) = delete;

    WeightOwned(WeightOwned&& o) noexcept { steal_(o); }
    WeightOwned& operator=(WeightOwned&& o) noexcept;

    // weights_alloc on `stream`, released on the same stream. Empty on failure or bytes == 0.
    [[nodiscard]] static WeightOwned allocate(size_t bytes, cudaStream_t stream, bool in_scope = true);

    // Takes over a block weights_alloc produced; null ptr = empty.
    [[nodiscard]] static WeightOwned adopt(void* ptr, size_t bytes, cudaStream_t release_stream) noexcept;

    // The stream the release is ordered on from now on.
    void set_release_stream(cudaStream_t stream) noexcept { stream_ = stream; }

    // Frees early on the stored stream; the weights_free status. Idempotent (empty: cudaSuccess).
    cudaError_t reset() noexcept;
    // Frees early on `stream` instead of the stored one.
    cudaError_t reset_on(cudaStream_t stream) noexcept {
        stream_ = stream;
        return reset();
    }

    // Leaves the weights family without a free: the caller uncharges WEIGHTS and frees it.
    [[nodiscard]] void* release() noexcept;

    void* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    size_t bytes() const noexcept { return bytes_; }
    cudaStream_t release_stream() const noexcept { return stream_; }

private:
    void steal_(WeightOwned& o) noexcept;

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    cudaStream_t stream_ = nullptr;
};

// The weight blocks of one model, in adoption order. Each handle's release stream is the null
// stream (teardown); a block that leaves earlier is taken out and released on a named stream.
class WeightSet {
public:
    WeightSet() = default;
    ~WeightSet();

    WeightSet(const WeightSet&) = delete;
    WeightSet& operator=(const WeightSet&) = delete;
    WeightSet(WeightSet&&) noexcept = default;
    WeightSet& operator=(WeightSet&&) noexcept = default;

    // Appends a block weights_alloc produced; null ptr is ignored.
    void adopt(void* ptr, size_t bytes);
    void adopt(WeightOwned&& h);

    // Moves the handle of base pointer `ptr` out (order of the rest kept); empty when not held.
    [[nodiscard]] WeightOwned extract(const void* ptr) noexcept;

    [[nodiscard]] bool contains(const void* ptr) const noexcept;
    size_t size() const noexcept { return handles_.size(); }
    bool empty() const noexcept { return handles_.empty(); }
    // Base pointer i in adoption order (the weight snapshot records an upload's delta by index).
    void* ptr_at(size_t i) const noexcept { return handles_[i].get(); }

    // Frees every block front to back on `stream` and empties the set. Returns the number of
    // failed frees; *first_err receives the first failure.
    int release_all(cudaStream_t stream, cudaError_t* first_err = nullptr) noexcept;

private:
    std::vector<WeightOwned> handles_;
};

}  // namespace imp

#pragma once

// An owning handle for a VRAMAllocator allocation.
//
// `VRAMAllocator` says of itself, in its own destructor, that it is "a tracker,
// not an owner" (vram_allocator.cu). Ownership therefore lives with the caller,
// spelled as a raw pointer member plus a free somewhere else, and that spelling
// is what the 2026-08-21 campaign kept finding defects in: four device pointers
// freed through an allocator that had not produced them (#1505), a 128 MiB
// cudaMallocAsync block released with cudaFree so it never returned to the async
// pool, and 469 direct allocation sites outside src/memory/ that
// tools/check_alloc_pairs.py has to re-derive from source text on every run.
//
// A grep-based gate catches those after they are written. This type makes them
// not compile: the allocation carries the allocator that produced it, and the
// only way to release it is through that same allocator.
//
// `AUDIT.md` R7 records that earlier audits referred to a `VramOwned` type that
// did not exist. It exists now.
//
//   VramOwned<int32_t> buf(vram_alloc_, n, "banned_tokens");
//   if (!buf) return nullptr;              // allocation failed, nothing leaked
//   cudaMemcpyAsync(buf.get(), ..., buf.bytes(), ...);
//   // freed by ~VramOwned, through vram_alloc_, exactly once
//
// Move-only on purpose. A copy would be a double free, and there is no sane
// deep-copy semantic for device memory that this type should be choosing.

#include <cstddef>
#include <utility>

#include "memory/vram_allocator.h"

namespace imp {

template <typename T>
class VramOwned {
public:
    VramOwned() = default;

    // Allocates `count` elements. On failure the handle is empty and converts
    // to false; it never throws, because the allocation sites this replaces are
    // in paths that check and degrade rather than abort.
    VramOwned(VRAMAllocator& alloc, size_t count, const char* tag, bool bypass_headroom = false)
        : alloc_(&alloc), count_(count) {
        if (count_ == 0)
            return;
        ptr_ = static_cast<T*>(alloc_->allocate(count_ * sizeof(T), tag, bypass_headroom));
        if (ptr_ == nullptr)
            count_ = 0;
    }

    ~VramOwned() { reset(); }

    VramOwned(const VramOwned&) = delete;
    VramOwned& operator=(const VramOwned&) = delete;

    VramOwned(VramOwned&& o) noexcept
        : alloc_(o.alloc_), ptr_(o.ptr_), count_(o.count_) {
        o.alloc_ = nullptr;
        o.ptr_ = nullptr;
        o.count_ = 0;
    }

    VramOwned& operator=(VramOwned&& o) noexcept {
        if (this != &o) {
            reset();
            alloc_ = o.alloc_;
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.alloc_ = nullptr;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    // Frees early. Idempotent, and safe on a moved-from handle.
    void reset() noexcept {
        if (ptr_ != nullptr && alloc_ != nullptr)
            alloc_->free(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    size_t count() const noexcept { return count_; }
    size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    VRAMAllocator* alloc_ = nullptr;
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace imp

#pragma once

// L3 of the memory architecture (docs/MEMORY_ARCHITECTURE.md §A3.4): the typed
// views that make I3 — "stable addresses for graph-captured memory" — a
// property of the type system instead of a comment the next refactor ignores.
//
// CUDA Graphs bake device pointers into the captured graph. Prefill and decode
// are both graphified, so any buffer touched inside a captured region must live
// somewhere whose address is guaranteed stable for the graph's lifetime. The
// mechanism is deliberately small:
//
//   DeviceSpan<T>  — a view of device memory. Says nothing about stability.
//   StableSpan<T>  — a view whose address is guaranteed stable for the lifetime
//                    of the region it came from.
//
// StableSpan widens to DeviceSpan implicitly. There is NO conversion the other
// way, no StableSpan(T*) constructor, and no as_stable() escape hatch: the only
// way to obtain one is from a tier allocator that can actually make the promise
// (passkey idiom below). A graph-capturable kernel wrapper takes StableSpan, so
// handing it a relocatable buffer does not compile.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace imp {

class ArenaAllocator;
class ScratchStack;
class BlockPool;
template <class T>
class StableSpan;

namespace detail {

// Passkey: only the tier allocators can construct one, so only they can build a
// StableSpan. Adding a friend here is the single, greppable, reviewable act of
// granting something the right to promise address stability.
class StableKey {
private:
    StableKey() = default;
    friend class ::imp::ArenaAllocator;
    friend class ::imp::ScratchStack;
    friend class ::imp::BlockPool;
    // StableSpan itself, so subspan()/as<>() can re-wrap a view it ALREADY
    // holds. That mints no new guarantee — you cannot reach this without a
    // StableSpan in hand, which only an allocator above can have given you.
    template <class T>
    friend class ::imp::StableSpan;
    // Deliberately no test-only friend: tests obtain their spans from a real
    // allocator over a FakeBackend, so the guarantee has no hole to widen.
};

}  // namespace detail

// ── DeviceSpan: a view. Cheap, copyable, promises nothing. ────────────
template <class T>
class DeviceSpan {
public:
    DeviceSpan() = default;
    DeviceSpan(T* p, size_t n) : p_(p), n_(n) {}

    T* data() const { return p_; }
    size_t size() const { return n_; }
    size_t size_bytes() const { return n_ * sizeof(T); }
    bool empty() const { return n_ == 0; }
    explicit operator bool() const { return p_ != nullptr && n_ > 0; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// ── StableSpan: a view with an address-stability guarantee. ───────────
template <class T>
class StableSpan {
public:
    StableSpan() = default;
    // Constructible only with a passkey — see detail::StableKey.
    StableSpan(detail::StableKey, T* p, size_t n) : p_(p), n_(n) {}

    T* data() const { return p_; }
    size_t size() const { return n_; }
    size_t size_bytes() const { return n_ * sizeof(T); }
    bool empty() const { return n_ == 0; }
    explicit operator bool() const { return p_ != nullptr && n_ > 0; }

    // Widening to a plain view is always sound: dropping a guarantee is safe.
    operator DeviceSpan<T>() const { return DeviceSpan<T>(p_, n_); }

    // Sub-views inherit the guarantee — they point into the same stable region.
    StableSpan<T> subspan(size_t offset, size_t count) const {
        if (offset >= n_)
            return StableSpan<T>();
        const size_t avail = n_ - offset;
        return StableSpan<T>(detail::StableKey{}, p_ + offset, count < avail ? count : avail);
    }
    StableSpan<T> first(size_t count) const { return subspan(0, count); }

    // Reinterpret a byte span as a typed one. Only from std::byte, so this
    // cannot be used to launder an arbitrary pointer into a StableSpan: you
    // still need a stable byte span to start from.
    template <class U>
    StableSpan<U> as() const
        requires std::is_same_v<std::remove_const_t<T>, std::byte>
    {
        return StableSpan<U>(detail::StableKey{}, reinterpret_cast<U*>(p_), n_ / sizeof(U));
    }

private:
    friend class StableSpan<const T>;
    T* p_ = nullptr;
    size_t n_ = 0;
};

}  // namespace imp

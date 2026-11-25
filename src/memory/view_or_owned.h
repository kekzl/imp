#pragma once

// L3 handle for the arena-or-private buffer (MEMORY.md A3.4, I3 + I5): normally a slice of a tier
// allocator (a StableSpan, never freed individually), and a private allocation this handle frees
// when the tier refused or the need outgrew the slice. Replaces "raw pointer + bool from_arena".
// Holds a view or an owner, never both. No allocation policy inside: the caller asks the tier,
// then falls back.
//   ViewOrOwned<void> ws;
//   if (auto s = engine_arena().take_bytes(n); !s.empty()) moved = ws.set_view(s);
//   else moved = ws.set_owned(DeviceOwned<void>::allocate(alloc, n, "ws"));
//   if (moved) bump_graph_epoch();          // the caller's policy, the handle only reports
//   kernel(ws.get(), ...);                  // raw pointer at the launch
// The setters and reset() return true when a non-null address was held before the call and get()
// no longer returns it: the one case in which a captured graph can hold a stale pointer.

#include <cstddef>
#include <type_traits>
#include <utility>

#include "memory/device_owned.h"
#include "memory/span.h"

namespace imp {

template <typename T>
class ViewOrOwned {
public:
    // The element type of the view: take_bytes() spans serve T = void.
    using view_element = std::conditional_t<std::is_void_v<T>, std::byte, T>;
    using view_type = StableSpan<view_element>;

    ViewOrOwned() = default;
    ~ViewOrOwned() = default;  // owned_ releases itself, a view is never freed

    ViewOrOwned(const ViewOrOwned&) = delete;
    ViewOrOwned& operator=(const ViewOrOwned&) = delete;

    ViewOrOwned(ViewOrOwned&& o) noexcept : view_(o.view_), owned_(std::move(o.owned_)) {
        o.view_ = view_type();
    }
    ViewOrOwned& operator=(ViewOrOwned&& o) noexcept {
        if (this != &o) {
            view_ = o.view_;
            owned_ = std::move(o.owned_);  // releases what this handle owned
            o.view_ = view_type();
        }
        return *this;
    }

    // Becomes a view of `view`; an empty span leaves the handle empty. Releases a private
    // allocation held before, exactly once. A replaced view is never freed.
    [[nodiscard]] bool set_view(view_type view) noexcept {
        T* const before = get();
        owned_.reset();
        view_ = view ? view : view_type();
        return moved_(before);
    }

    // Becomes the owner of `owned`; an empty owner (failed allocation) leaves the handle empty.
    // Releases a private allocation held before, exactly once. A replaced view is never freed.
    [[nodiscard]] bool set_owned(DeviceOwned<T>&& owned) noexcept {
        T* const before = get();
        view_ = view_type();
        owned_ = std::move(owned);
        return moved_(before);
    }

    // Empties the handle: releases a private allocation, drops a view. Idempotent.
    [[nodiscard]] bool reset() noexcept {
        T* const before = get();
        view_ = view_type();
        owned_.reset();
        return moved_(before);
    }

    T* get() const noexcept { return owned_ ? owned_.get() : static_cast<T*>(view_.data()); }
    size_t bytes() const noexcept { return owned_ ? owned_.bytes() : view_.size_bytes(); }
    explicit operator bool() const noexcept { return get() != nullptr; }

    [[nodiscard]] bool is_view() const noexcept { return static_cast<bool>(view_); }
    [[nodiscard]] bool owns() const noexcept { return static_cast<bool>(owned_); }

    // The stability proof, for a launch wrapper that takes a StableSpan. Empty unless is_view().
    view_type view() const noexcept { return view_; }

private:
    [[nodiscard]] bool moved_(const T* before) const noexcept { return before != nullptr && get() != before; }

    view_type view_;
    DeviceOwned<T> owned_;
};

}  // namespace imp

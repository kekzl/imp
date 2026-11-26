#pragma once
// Owner of the Inkling2-Loop per-slot recurrent slab (B-111 S3): one VRAMAllocator allocation under the tag
// inkling_slab, slot s at s * slot_bytes. The engine holds it (Engine::inkling_slab_) like SSMState; each
// InklingForward views its slot, the slot table binds it to a request id.

#include "exec/inkling_slab_layout.h"
#include "memory/vram_owned.h"

#include <cstddef>
#include <utility>

namespace imp {

class InklingSlab {
public:
    // Up to want slots of layout l in one allocation, fewer when the card does not take them all.
    // Returns the slots allocated, 0 = not one. Frees a slab held before.
    int allocate(VRAMAllocator& va, const InklingSlabLayout& l, int want) {
        mem_.reset();
        slots_ = 0;
        layout_ = l;
        for (int n = want; n > 0 && l.slot_bytes > 0; --n) {
            VramOwned<std::byte> m(va, l.total_bytes(n), "inkling_slab");
            if (m) {
                mem_ = std::move(m);
                slots_ = n;
                break;
            }
        }
        return slots_;
    }
    [[nodiscard]] int slots() const { return slots_; }
    [[nodiscard]] const InklingSlabLayout& layout() const { return layout_; }
    [[nodiscard]] size_t bytes() const { return mem_.bytes(); }
    // Base of slot s, null when s holds no slot.
    [[nodiscard]] std::byte* slot(int s) const {
        const size_t off = layout_.slot_offset(s, slots_);
        return off == InklingSlabLayout::kNone ? nullptr : mem_.get() + off;
    }

private:
    InklingSlabLayout layout_;
    VramOwned<std::byte> mem_;
    int slots_ = 0;
};

}  // namespace imp

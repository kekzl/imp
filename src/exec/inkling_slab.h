#pragma once
// Owner of the Inkling2-Loop per-slot recurrent slab (B-111 S3): one VRAMAllocator allocation under the tag
// inkling_slab, slot s at s * slot_bytes, then lookahead region s (S3b). The engine holds it
// (Engine::inkling_slab_) like SSMState; each InklingForward views its slot, the slot table binds it to a
// request id, the draft contexts view the lookahead regions.

#include "exec/inkling_slab_layout.h"
#include "memory/vram_owned.h"

#include <cstddef>
#include <utility>

namespace imp {

class InklingSlab {
public:
    // Up to want slots of layout l in one allocation, each of the first look_slots with a lookahead region;
    // without the regions before fewer slots. Returns the slots allocated, 0 = not one. Frees a slab held
    // before.
    int allocate(VRAMAllocator& va, const InklingSlabLayout& l, int want) {
        mem_.reset();
        slots_ = looks_ = 0;
        layout_ = l;
        for (int n = want; n > 0 && l.slot_bytes > 0; --n)
            for (int looks = l.look_slots(n);; looks = 0) {
                VramOwned<std::byte> m(va, l.total_bytes(n, looks), "inkling_slab");
                if (m) {
                    mem_ = std::move(m);
                    slots_ = n;
                    looks_ = looks;
                    return slots_;
                }
                if (looks == 0)
                    break;
            }
        return slots_;
    }
    [[nodiscard]] int slots() const { return slots_; }
    [[nodiscard]] int look_slots() const { return looks_; }
    [[nodiscard]] const InklingSlabLayout& layout() const { return layout_; }
    [[nodiscard]] size_t bytes() const { return mem_.bytes(); }
    // Base of slot s, null when s holds no slot.
    [[nodiscard]] std::byte* slot(int s) const {
        const size_t off = layout_.slot_offset(s, slots_);
        return off == InklingSlabLayout::kNone ? nullptr : mem_.get() + off;
    }
    // Lookahead region of slot s, empty when s holds none.
    [[nodiscard]] InklingLookRegion look(int s) const {
        const size_t off = layout_.look_offset(s, slots_, looks_);
        return off == InklingSlabLayout::kNone ? InklingLookRegion{}
                                               : InklingLookRegion{&layout_.look, mem_.get() + off};
    }

private:
    InklingSlabLayout layout_;
    VramOwned<std::byte> mem_;
    int slots_ = 0, looks_ = 0;
};

}  // namespace imp

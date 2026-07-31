#pragma once

// Move a loaded Qwen3-VL vision tower from the host mapping onto the device.
//
// The checkpoint is BF16; the encoder runs in FP16. Conversion happens here,
// once, so the forward never sees a source dtype. Every tensor slot is rewritten
// in place to point at device memory, so the tower is either fully resident or
// unchanged — a half-uploaded tower would dereference a host pointer on the
// device, which does not fault, it just reads garbage.

#include "vision/vision_model.h"

#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

// `alloc` may be null, in which case the tower is left alone and this returns
// false — the encoder has no fallback for host-resident weights.
//
// The device blocks are appended to `out_allocs` and belong to the CALLER, not
// to the VisionModel. That is deliberate: a tower holding pointers into an
// allocator it does not own is a use-after-free the moment a teardown order
// puts the allocator first. The caller frees them and calls
// `qwen3vl_release_vision_tower` to invalidate the tower in the same breath.
bool qwen3vl_upload_vision_tower(VisionModel& model, VRAMAllocator* alloc, std::vector<void*>& out_allocs,
                                 size_t& bytes_out, std::string& err);

// Point every tensor slot back at nothing. Call this when the device blocks the
// upload handed out are released, so a tower that outlives them cannot be
// mistaken for a usable one.
void qwen3vl_release_vision_tower(VisionModel& model);

}  // namespace imp

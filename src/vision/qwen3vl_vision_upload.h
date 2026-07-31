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

namespace imp {

class VRAMAllocator;

// `alloc` may be null, in which case the tower is left alone and this returns
// false — the encoder has no fallback for host-resident weights.
bool qwen3vl_upload_vision_tower(VisionModel& model, VRAMAllocator* alloc, size_t& bytes_out,
                                 std::string& err);

}  // namespace imp

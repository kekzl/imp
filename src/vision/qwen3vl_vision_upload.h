#pragma once

// Move a loaded Qwen3-VL vision tower from the host mapping onto the device.
//
// The checkpoint is BF16; the encoder runs in FP16. Conversion happens here,
// once, so the forward never sees a source dtype. Every tensor slot is rewritten
// in place to point at device memory, so the tower is either fully resident or
// unchanged — a half-uploaded tower would dereference a host pointer on the
// device, which does not fault, it just reads garbage.

#include "vision/vision_model.h"

#include <expected>
#include <string>
#include <vector>

namespace imp {

// The tower is engine-lifetime, so its blocks come from the T2 engine arena
// (docs/internals/MEMORY.md). The arena is sized for exactly this in
// Engine::init, from qwen3vl_vision_tower_device_bytes() — the tower uploads long
// after the arena opens, so the two numbers have to be derived from the same
// tensor list to stay in step.
//
// There is no longer a per-block free, and with it goes the hazard the previous
// caller-owned scheme documented: a tower holding pointers into an allocator it
// did not own was a use-after-free the moment teardown ordered the allocator
// first. The arena releases wholesale on close, after which nothing dereferences
// the slots — callers still invoke `qwen3vl_release_vision_tower` so a tower that
// outlives the arena cannot be read.
//
// Returns the bytes uploaded, or the error text when the arena cannot serve the
// tower, which means the plan under-reserved: the encoder has no fallback for
// host-resident weights.
[[nodiscard]] std::expected<size_t, std::string> qwen3vl_upload_vision_tower(VisionModel& model);

// Point every tensor slot back at nothing. Call this when the device blocks the
// upload handed out are released, so a tower that outlives them cannot be
// mistaken for a usable one.
void qwen3vl_release_vision_tower(VisionModel& model);

}  // namespace imp

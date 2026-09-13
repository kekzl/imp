#pragma once

// Moves a loaded Qwen3-VL vision tower from host mapping to device. Checkpoint is BF16,
// encoder runs FP16; conversion happens here once so forward never sees the source dtype.
// Every tensor slot is rewritten in place to device memory, so the tower is either fully
// resident or unchanged - a half-uploaded tower would read garbage from a host pointer on
// device (no fault).

#include "vision/vision_model.h"

#include <expected>
#include <string>
#include <vector>

namespace imp {

// Tower is engine-lifetime, its blocks come from the T2 engine arena (MEMORY.md), sized in
// Engine::init from qwen3vl_vision_tower_device_bytes() (derived from the same tensor list
// to stay in step, since upload happens long after the arena opens). No per-block free: the
// arena releases wholesale on close; callers still call qwen3vl_release_vision_tower so a
// tower outliving the arena cannot be read. Returns bytes uploaded, or an error when the
// arena cannot serve the tower (plan under-reserved; no fallback for host-resident weights).
[[nodiscard]] std::expected<size_t, std::string> qwen3vl_upload_vision_tower(VisionModel& model);

// Point every tensor slot back at nothing. Call this when the device blocks the
// upload handed out are released, so a tower that outlives them cannot be
// mistaken for a usable one.
void qwen3vl_release_vision_tower(VisionModel& model);

}  // namespace imp

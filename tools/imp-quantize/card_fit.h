#pragma once

#include <cstddef>

namespace imp::quantize {

// Reports the checkpoint against the target card by ON-DISK size, not a VRAM prediction: what
// the engine resides is weights PLUS a scale-factor cache MINUS what the loader skips (vision
// tower uploaded separately, MTP sidecar maybe unloaded). Measured on Qwen3.8-27B: 18.60 GiB
// disk arrived as 16.08 GiB weights + 1.49 GiB CUTLASS scale cache - right order, wrong decimal.
// mx_widen: bytes the loader adds when it widens MXFP8 projections to BF16 (#2475).
void report_card_fit(size_t bytes_out, size_t mx_widen);

}  // namespace imp::quantize

#pragma once

// Current-device queries, cached once per process: imp drives device 0 only (imp_context_create
// rejects device_id != 0), so one value serves every caller. A failed query stays failed.

namespace imp {

// SM count of the current device; 0 when the query failed (no device, CPU test lane).
[[nodiscard]] int device_sm_count_or_0();

// device_sm_count_or_0(), or kSm120SmCount (170) when the query failed.
[[nodiscard]] int device_sm_count();

// Compute capability as major * 10 + minor (RTX 5090: 120); 0 when the query failed.
[[nodiscard]] int device_cc();

// device_cc() >= 120: false when the query failed and on a cc < 12.0 device.
[[nodiscard]] bool device_is_sm120();

}  // namespace imp

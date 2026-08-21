#pragma once

// One enum, its own header, because kv_cache.h needs it and it lives in
// namespace imp rather than imp::cfg. Split out with the nine config sections
// on 2026-08-21 so a TU that includes only kv_cache.h still compiles.

namespace imp {

enum class SwaSizingMode { Off, On, Auto };

}  // namespace imp

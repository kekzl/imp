#include "memory/memory_manager.h"

// All MemoryManager methods are currently inline in the header — the façade
// is a thin wrapper around already-existing modules. This TU is kept as the
// home for any future non-inline logic (e.g. cross-allocator accounting,
// telemetry, or pre-allocation orchestration) so callers can keep including
// the header without forcing a rebuild when implementation changes land.
//
// Phase 5 Track C of
// docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

namespace imp {
}  // namespace imp

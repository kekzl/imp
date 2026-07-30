#pragma once

// Opens the engine-persistent (T2) arena for a test that exercises a tenant of
// it directly.
//
// Production opens the arena in Engine::init, before the first tenant runs
// (docs/MEMORY_ARCHITECTURE.md A3.3). A GPU test that calls into a T2 tenant
// without an Engine — the CUTLASS grouped GEMM is the case that needed this —
// would otherwise find no arena, take nothing, and fail for a reason that has
// nothing to do with what it is testing.
//
// Idempotent: if something already opened the arena, this leaves it alone and
// does not close it, so nesting two of these is safe.

#include "memory/backend.h"
#include "memory/engine_arena.h"

namespace imp {

class ScopedEngineArena {
public:
    explicit ScopedEngineArena(size_t capacity = 8ull << 20) {
        if (!engine_arena().is_open())
            owned_ = engine_arena_open(cuda_malloc_backend(), capacity) == MemError::Ok;
    }
    ~ScopedEngineArena() {
        if (owned_)
            engine_arena_close();
    }
    ScopedEngineArena(const ScopedEngineArena&) = delete;
    ScopedEngineArena& operator=(const ScopedEngineArena&) = delete;

    bool opened() const { return owned_; }

private:
    bool owned_ = false;
};

}  // namespace imp

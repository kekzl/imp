#include "memory/engine_arena.h"
#include "memory/backend.h"
#include "core/logging.h"

namespace imp {

ArenaAllocator& engine_arena() {
    static ArenaAllocator inst;
    return inst;
}

MemError engine_arena_open(Backend& backend, size_t capacity) {
    ArenaAllocator& a = engine_arena();
    if (a.is_open())
        return MemError::InvalidArgument;
    MemError e = a.open(backend, capacity, RegionTag::EnginePersistent);
    if (e != MemError::Ok) {
        IMP_LOG_WARN("engine arena: open(%.1f MiB) failed (%s) — tenants fall back to their "
                     "own allocations",
                     capacity / (1024.0 * 1024.0), mem_error_name(e));
        return e;
    }
    IMP_LOG_INFO("engine arena: %.1f MiB reserved (engine-persistent tier)",
                 capacity / (1024.0 * 1024.0));
    return MemError::Ok;
}

void engine_arena_close() {
    ArenaAllocator& a = engine_arena();
    if (!a.is_open())
        return;
    IMP_LOG_INFO("engine arena: closing (high-water %.2f MiB of %.1f MiB reserved)",
                 a.high_water() / (1024.0 * 1024.0), a.capacity() / (1024.0 * 1024.0));
    a.close();
}

}  // namespace imp

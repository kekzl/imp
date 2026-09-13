#pragma once

// The process-global engine-persistent (T2) arena (MEMORY.md A2/A3.3): executor
// workspaces, cuBLAS/CUTLASS scratch, graph buffers, per-kernel scratch sized once at
// init. None is individually freed, so a bump arena's failure mode is under-provisioning
// (visible, reported), never a leak.
// Process-global rather than an Engine member because its tenants are file-scope statics
// in compute/ and exec/ with no Engine to reach through. Single-engine-per-process is the
// supported deployment; Engine::init/~Engine own open/close.
// Sizing is provisional: kEngineArenaDefaultBytes covers current tenants with room to
// spare; high_water() reports actual use for the planner (A4) to take over (A7 step 4).

#include "memory/arena.h"

#include <cstddef>

namespace imp {

class Backend;

// Provisional capacity. The tenants migrated so far are KiB-to-MiB scale;
// this is deliberately generous so an under-provisioned arena cannot be the
// thing that breaks a model, and high_water() makes the real number visible.
constexpr size_t kEngineArenaDefaultBytes = 64ull * 1024 * 1024;

// Open/close. Idempotent-safe: opening twice is an error, closing when unopened is a
// no-op. Called from Engine::init and ~Engine. lazy: see ArenaAllocator::open; the
// backend must be growable for it to take effect, a cudaMalloc backend opens fixed.
MemError engine_arena_open(Backend& backend, size_t capacity = kEngineArenaDefaultBytes, bool lazy = false);
void engine_arena_close();

// The arena itself. Always valid to call; take_bytes() returns an empty span
// when it is not open, which every tenant must already handle because that is
// what an allocation failure looked like before.
ArenaAllocator& engine_arena();

}  // namespace imp

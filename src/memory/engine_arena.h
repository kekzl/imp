#pragma once

// The process-global engine-persistent (T2) arena
// (docs/MEMORY_ARCHITECTURE.md §A2/§A3.3).
//
// T2 holds what lives for the process: executor workspaces, the cuBLAS/CUTLASS
// scratch, graph buffers, per-kernel scratch that is sized once at init. None
// of it is individually freed, which is exactly why a bump arena fits: the
// failure mode becomes under-provisioning (visible, reported) instead of a
// leak (invisible).
//
// It is a process global rather than an Engine member because its tenants are
// file-scope statics in compute/ and exec/ that have no Engine to reach
// through — the same reason gemm.cu's cuBLAS workspace is a static today.
// Single-engine-per-process is the supported deployment (see vram_query.h),
// and Engine::init/~Engine own the open/close.
//
// Sizing is provisional: kEngineArenaDefaultBytes covers the current tenants
// with room to spare, and `high_water()` reports what was actually used so
// the planner (A4) can take the number over when A7 step 4 completes.

#include "memory/arena.h"

#include <cstddef>

namespace imp {

class Backend;

// Provisional capacity. The tenants migrated so far are KiB-to-MiB scale;
// this is deliberately generous so an under-provisioned arena cannot be the
// thing that breaks a model, and high_water() makes the real number visible.
constexpr size_t kEngineArenaDefaultBytes = 64ull * 1024 * 1024;

// Open/close. Idempotent-safe: opening twice is an error, closing when
// unopened is a no-op. Called from Engine::init and ~Engine.
MemError engine_arena_open(Backend& backend, size_t capacity = kEngineArenaDefaultBytes);
void engine_arena_close();

// The arena itself. Always valid to call; take_bytes() returns an empty span
// when it is not open, which every tenant must already handle because that is
// what an allocation failure looked like before.
ArenaAllocator& engine_arena();

}  // namespace imp

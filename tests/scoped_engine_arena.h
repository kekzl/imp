#pragma once

// Opens the engine-persistent (T2) arena for tests that exercise a tenant of it
// directly.
//
// Production opens the arena in Engine::init, before the first tenant runs
// (docs/MEMORY_ARCHITECTURE.md A3.3). A GPU test that calls into a T2 tenant
// without an Engine — the CUTLASS grouped GEMM and the IMMA prefill scratches
// are the cases that needed this — would otherwise find no arena, take nothing,
// and fail for a reason that has nothing to do with what it is testing.
//
// Idempotent: if something already opened the arena, this leaves it alone and
// does not close it, so nesting two of these is safe.

#include "memory/backend.h"
#include "memory/engine_arena.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

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

// Keeps the arena open for a whole test binary AND rewinds it between tests.
//
// The rewind is not tidiness, it is required. A bump arena never reclaims, so
// without it every test that grows a file-scope scratch permanently consumes
// capacity and a later test in the same binary finds 0 MiB free — which is
// exactly how the first version of this failed: the IMMA shape sweep drained
// the arena and the CUTLASS grouped test two files later produced garbage.
// ArenaAllocator::reset() is the phase-boundary operation for this, and it
// bumps generation(), so the tenants notice their cached slice is gone and
// re-take instead of using a rewound address.
class EngineArenaEnvironment : public ::testing::Environment {
public:
    explicit EngineArenaEnvironment(size_t capacity) : capacity_(capacity) {}

    void SetUp() override {
        arena_ = std::make_unique<ScopedEngineArena>(capacity_);
        if (arena_->opened() && !listener_installed_) {
            ::testing::UnitTest::GetInstance()->listeners().Append(new RewindBetweenTests());
            listener_installed_ = true;
        }
    }
    void TearDown() override { arena_.reset(); }

private:
    class RewindBetweenTests : public ::testing::EmptyTestEventListener {
        void OnTestEnd(const ::testing::TestInfo&) override {
            if (engine_arena().is_open())
                engine_arena().reset();
        }
    };

    size_t capacity_;
    bool listener_installed_ = false;
    std::unique_ptr<ScopedEngineArena> arena_;
};

}  // namespace imp

// One line per test file. Registering it twice in the same binary is harmless:
// the second ScopedEngineArena finds the arena open and does nothing.
#define IMP_TEST_ENGINE_ARENA(bytes)                                                           \
    static ::testing::Environment* const imp_arena_env_ = ::testing::AddGlobalTestEnvironment( \
        new ::imp::EngineArenaEnvironment(bytes))

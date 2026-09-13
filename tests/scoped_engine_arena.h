#pragma once

// Opens the engine-persistent (T2) arena (docs/internals/MEMORY.md A3.3) for tests exercising
// a T2 tenant without an Engine - CUTLASS grouped GEMM and IMMA prefill scratch needed this.
// Idempotent: does not close an already-open arena, so nesting is safe.

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

// Rewinds the arena between tests: a bump arena never reclaims, so one test's scratch growth
// would starve capacity for later tests in the same binary.
// ArenaAllocator::reset() bumps generation() so tenants re-take instead of reusing a stale slice.
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

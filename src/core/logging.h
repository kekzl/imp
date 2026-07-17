#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <atomic>

namespace imp {

enum class LogLevel : int {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4,
};

void log_set_level(LogLevel level);

// Inline for zero-overhead log level check in hot paths.
// The atomic is defined in logging.cpp; declared here for inlining.
extern std::atomic<LogLevel> g_log_level;
inline LogLevel log_get_level() { return g_log_level.load(std::memory_order_relaxed); }

void log_message(LogLevel level, const char* file, int line, const char* fmt, ...);

}  // namespace imp

#define IMP_LOG_DEBUG(...)                                                               \
    do {                                                                                 \
        if (::imp::log_get_level() <= ::imp::LogLevel::DEBUG)                            \
            ::imp::log_message(::imp::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define IMP_LOG_INFO(...)                                                               \
    do {                                                                                \
        if (::imp::log_get_level() <= ::imp::LogLevel::INFO)                            \
            ::imp::log_message(::imp::LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define IMP_LOG_WARN(...)                                                               \
    do {                                                                                \
        if (::imp::log_get_level() <= ::imp::LogLevel::WARN)                            \
            ::imp::log_message(::imp::LogLevel::WARN, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define IMP_LOG_ERROR(...)                                                               \
    do {                                                                                 \
        if (::imp::log_get_level() <= ::imp::LogLevel::ERROR)                            \
            ::imp::log_message(::imp::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define IMP_LOG_FATAL(...) ::imp::log_message(::imp::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)

// --- Precondition check ---
// IMP_CHECK is the production-safe replacement for <cassert> assert(). Unlike
// assert(), it does NOT vanish under NDEBUG. On failure it logs at FATAL and
// aborts the process, surfacing internal-invariant violations in Release
// builds the same way they would in Debug.
//
// Use for internal-API preconditions where violation = programmer error.
// Do NOT use for user-input validation — return an ImpError code instead.
#define IMP_CHECK(cond, ...)                       \
    do {                                           \
        if (!(cond)) {                             \
            IMP_LOG_FATAL(__VA_ARGS__);            \
            std::abort();                          \
        }                                          \
    } while (0)

// --- CUDA error checking macros ---
// Log-only: reports CUDA errors without affecting control flow.
// Use in cleanup paths or where failure is non-fatal.
#define IMP_CUDA_CHECK_LOG(call)                                                     \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            IMP_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
        }                                                                            \
    } while (0)

// Check + return false: for bool-returning init/setup functions.
#define IMP_CUDA_CHECK_BOOL(call)                                                    \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            IMP_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
            return false;                                                            \
        }                                                                            \
    } while (0)

// Post-launch check: place immediately after a kernel `<<<>>>` launch. Surfaces
// launch-time failures (invalid configuration, missing kernel image, OOM at
// launch) at the launch site instead of at the next synchronizing call.
// Uses cudaPeekAtLastError() so the sticky error is NOT cleared — existing
// downstream IMP_CUDA_CHECK_* handling still sees and propagates it.
#define IMP_CUDA_CHECK_LAUNCH()                                                      \
    do {                                                                             \
        cudaError_t err_ = cudaPeekAtLastError();                                    \
        if (err_ != cudaSuccess) {                                                   \
            IMP_LOG_ERROR("CUDA kernel launch failed at %s:%d — %s", __FILE__,       \
                          __LINE__, cudaGetErrorString(err_));                       \
        }                                                                            \
    } while (0)

// Check + return void: for void-returning functions.
#define IMP_CUDA_CHECK_VOID(call)                                                    \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            IMP_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
            return;                                                                  \
        }                                                                            \
    } while (0)

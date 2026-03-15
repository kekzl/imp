#pragma once

#include <cstdio>
#include <cstdarg>
#include <atomic>

namespace imp {

enum class LogLevel : int {
    DEBUG   = 0,
    INFO    = 1,
    WARN    = 2,
    ERROR   = 3,
    FATAL   = 4,
};

void log_set_level(LogLevel level);

// Inline for zero-overhead log level check in hot paths.
// The atomic is defined in logging.cpp; declared here for inlining.
extern std::atomic<LogLevel> g_log_level;
inline LogLevel log_get_level() {
    return g_log_level.load(std::memory_order_relaxed);
}

void log_message(LogLevel level, const char* file, int line, const char* fmt, ...);

} // namespace imp

#define IMP_LOG_DEBUG(...) \
    do { if (::imp::log_get_level() <= ::imp::LogLevel::DEBUG) \
        ::imp::log_message(::imp::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__); } while(0)
#define IMP_LOG_INFO(...) \
    do { if (::imp::log_get_level() <= ::imp::LogLevel::INFO) \
        ::imp::log_message(::imp::LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__); } while(0)
#define IMP_LOG_WARN(...) \
    do { if (::imp::log_get_level() <= ::imp::LogLevel::WARN) \
        ::imp::log_message(::imp::LogLevel::WARN, __FILE__, __LINE__, __VA_ARGS__); } while(0)
#define IMP_LOG_ERROR(...) \
    do { if (::imp::log_get_level() <= ::imp::LogLevel::ERROR) \
        ::imp::log_message(::imp::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__); } while(0)
#define IMP_LOG_FATAL(...) ::imp::log_message(::imp::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)

// --- CUDA error checking macros ---
// Log-only: reports CUDA errors without affecting control flow.
// Use in cleanup paths or where failure is non-fatal.
#define IMP_CUDA_CHECK_LOG(call)                                              \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            IMP_LOG_ERROR("CUDA error: %s at %s:%d — %s",                    \
                          #call, __FILE__, __LINE__,                          \
                          cudaGetErrorString(err_));                           \
        }                                                                     \
    } while (0)

// Check + return false: for bool-returning init/setup functions.
#define IMP_CUDA_CHECK_BOOL(call)                                             \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            IMP_LOG_ERROR("CUDA error: %s at %s:%d — %s",                    \
                          #call, __FILE__, __LINE__,                          \
                          cudaGetErrorString(err_));                           \
            return false;                                                     \
        }                                                                     \
    } while (0)

#include "core/logging.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <atomic>

namespace imp {

std::atomic<LogLevel> g_log_level{LogLevel::INFO};

void log_set_level(LogLevel level) {
    g_log_level.store(level, std::memory_order_relaxed);
}


static const char* level_str(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
    }
    return "?";
}

void log_message(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (static_cast<int>(level) < static_cast<int>(g_log_level.load(std::memory_order_relaxed))) {
        return;
    }

    // Timestamp
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm_buf);

    // Extract filename from path
    const char* basename = file;
    for (const char* p = file; *p; ++p) {
        if (*p == '/') basename = p + 1;
    }

    FILE* out = (level >= LogLevel::WARN) ? stderr : stdout;

    fprintf(out, "[%s][%s] %s:%d: ", time_str, level_str(level), basename, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fprintf(out, "\n");
    fflush(out);
}

} // namespace imp

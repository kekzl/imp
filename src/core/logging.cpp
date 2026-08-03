#include "core/logging.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <atomic>
#include <utility>
#include <cstring>
#include <cctype>

namespace imp {

std::atomic<LogLevel> g_log_level{LogLevel::INFO};

void log_set_level(LogLevel level) { g_log_level.store(level, std::memory_order_relaxed); }

bool log_level_from_string(const char* s, LogLevel& out) {
    if (!s || !*s)
        return false;
    // Lowercase in place into a small buffer; the longest accepted word is
    // "fatal", so anything longer cannot match and is rejected below.
    char buf[8] = {};
    size_t n = 0;
    for (; s[n] && n < sizeof(buf) - 1; ++n)
        buf[n] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[n])));
    if (s[n])
        return false;  // longer than any accepted word
    struct Entry {
        const char* name;
        LogLevel level;
    };
    static constexpr Entry kTable[] = {
        {"debug", LogLevel::DEBUG}, {"info", LogLevel::INFO},   {"warn", LogLevel::WARN},
        {"error", LogLevel::ERROR}, {"fatal", LogLevel::FATAL},
    };
    for (const auto& e : kTable) {
        if (std::strcmp(buf, e.name) == 0) {
            out = e.level;
            return true;
        }
    }
    return false;
}

static const char* level_str(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARN:
            return "WARN";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::FATAL:
            return "FATAL";
    }
    return "?";
}

void log_message(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (std::to_underlying(level) < std::to_underlying(g_log_level.load(std::memory_order_relaxed))) {
        return;
    }

    // Timestamp
    time_t now = time(nullptr);
    struct tm tm_buf {};
    localtime_r(&now, &tm_buf);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm_buf);

    // Extract filename from path
    const char* basename = file;
    for (const char* p = file; *p; ++p) {
        if (*p == '/')
            basename = p + 1;
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

}  // namespace imp

#pragma once

#include <cstdio>
#include <cstdarg>

namespace imp {

enum class LogLevel : int {
    DEBUG   = 0,
    INFO    = 1,
    WARN    = 2,
    ERROR   = 3,
    FATAL   = 4,
};

void log_set_level(LogLevel level);
LogLevel log_get_level();

void log_message(LogLevel level, const char* file, int line, const char* fmt, ...);

} // namespace imp

#define IMP_LOG_DEBUG(...) ::imp::log_message(::imp::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define IMP_LOG_INFO(...)  ::imp::log_message(::imp::LogLevel::INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define IMP_LOG_WARN(...)  ::imp::log_message(::imp::LogLevel::WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define IMP_LOG_ERROR(...) ::imp::log_message(::imp::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define IMP_LOG_FATAL(...) ::imp::log_message(::imp::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)

#pragma once

#include "utils.h"

#include <string>
#include <string_view>

// client_error_log_line (#2279): the one server log line per 4xx, "HTTP <status> <method> <path>:
// <error.message of the body sent>". Control bytes become ' ', message capped at 512 bytes.
inline std::string client_error_log_line(int status, std::string_view method, std::string_view path,
                                         const std::string& body) {
    std::string reason = "(no error message in body)";
    const json j = json::parse(body, nullptr, false);
    if (!j.is_discarded() && j.is_object() && j.contains("error") && j["error"].is_object() &&
        j["error"].contains("message") && j["error"]["message"].is_string())
        reason = j["error"]["message"].get<std::string>();
    constexpr size_t kMaxReason = 512;
    if (reason.size() > kMaxReason)
        reason = reason.substr(0, kMaxReason) + "...";
    for (char& c : reason)
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f)
            c = ' ';
    return "HTTP " + std::to_string(status) + " " + sanitize_for_echo(method, 16) + " " +
           sanitize_for_echo(path, 128) + ": " + reason;
}

#pragma once

// --cors-origins policy (#2402), header-only so the CPU lane tests it without the server TU.

#include <algorithm>
#include <string>
#include <vector>

// Comma list, whitespace trimmed, empty entries dropped.
inline std::vector<std::string> parse_cors_origins(const std::string& list) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos)
            comma = list.size();
        const std::string one = list.substr(pos, comma - pos);
        const size_t b = one.find_first_not_of(" \t");
        if (b != std::string::npos)
            out.push_back(one.substr(b, one.find_last_not_of(" \t") - b + 1));
        pos = comma + 1;
    }
    return out;
}

// Access-Control-Allow-Origin for a request's Origin: "" = send no CORS headers (default, empty
// list), "*" when the list holds "*", else `origin` on an exact, case-sensitive match.
inline std::string cors_allow_origin(const std::vector<std::string>& allowed, const std::string& origin) {
    if (std::find(allowed.begin(), allowed.end(), "*") != allowed.end())
        return "*";
    if (!origin.empty() && std::find(allowed.begin(), allowed.end(), origin) != allowed.end())
        return origin;
    return {};
}

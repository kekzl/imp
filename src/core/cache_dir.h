#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace imp {

// $XDG_CACHE_HOME/imp/<leaf>, else $HOME/.cache/imp/<leaf>, else "" (empty counts as unset).
// Each caller keeps its own fallback for "": warm cache /tmp/imp-warm-cache, library reserve none.
inline std::string imp_cache_dir(std::string_view leaf) {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::string(xdg) + "/imp/" + std::string(leaf);
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::string(home) + "/.cache/imp/" + std::string(leaf);
    return {};
}

}  // namespace imp

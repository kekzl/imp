# Embeds the web UI as a header the server links in: binary carries the page, no runtime
# asset path to get wrong in the container. Usage: cmake -DIN=<html> -DOUT=<header> -P embed_webui.cmake

file(READ "${IN}" IMP_WEBUI_PAYLOAD)

# The delimiter must not occur in the payload, or the raw string ends early.
if(IMP_WEBUI_PAYLOAD MATCHES "\\)IMPUI\"")
    message(FATAL_ERROR "web UI contains the raw-string delimiter )IMPUI\" — change the delimiter in ${CMAKE_CURRENT_LIST_FILE}")
endif()

file(WRITE "${OUT}"
"// Generated from tools/imp-server/webui/index.html — do not edit.
#pragma once

inline constexpr const char* IMP_WEBUI_HTML = R\"IMPUI(${IMP_WEBUI_PAYLOAD})IMPUI\";
")

#pragma once

#include <string>

namespace imp {

// Tool-call tag pair the constraint preamble gate watches for (has_tools). tokens=true also
// matches open/close as single vocab tokens; false = char prefix only (dynamic <function=NAME>).
// Default = ChatML. The server fills it from its tool-call dialect registry.
struct ToolCallGate {
    std::string open = "<tool_call>";
    std::string close = "</tool_call>";
    bool tokens = true;
};

}  // namespace imp

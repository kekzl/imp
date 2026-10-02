#include "tool_call_dialect.h"

#include <algorithm>
#include <cstring>

// Descriptors live next to their parsers (tool_call_<dialect>.cpp).
extern const ToolCallDialect kChatmlToolDialect;
extern const ToolCallDialect kChatmlHintToolDialect;
extern const ToolCallDialect kLlama3ToolDialect;
extern const ToolCallDialect kGemmaToolDialect;
extern const ToolCallDialect kHarmonyToolDialect;

namespace {

struct RegistryRow {
    imp::ChatTemplateFamily family;
    const ToolCallDialect* dialect;
};

// Family -> dialect. Every family not listed renders the ChatML wire format without enforcement.
constexpr RegistryRow kRegistry[] = {
    {imp::ChatTemplateFamily::CHATML, &kChatmlToolDialect},
    {imp::ChatTemplateFamily::LLAMA3, &kLlama3ToolDialect},
    {imp::ChatTemplateFamily::GEMMA, &kGemmaToolDialect},
    {imp::ChatTemplateFamily::HARMONY, &kHarmonyToolDialect},
};

}  // namespace

const ToolCallDialect& tool_call_dialect(imp::ChatTemplateFamily family) {
    for (const auto& row : kRegistry)
        if (row.family == family)
            return *row.dialect;
    return kChatmlHintToolDialect;
}

imp::ToolCallGate tool_call_gate(imp::ChatTemplateFamily family) {
    const ToolCallDialect& d = tool_call_dialect(family);
    return imp::ToolCallGate{d.gate_open, d.gate_close, d.gate_tokens};
}

bool suffix_is_marker_prefix(const std::string& buf, const char* m, size_t mlen) {
    size_t maxk = std::min(buf.size(), mlen - 1);
    for (size_t k = maxk; k >= 1; --k) {
        if (std::memcmp(buf.data() + buf.size() - k, m, k) == 0)
            return true;
    }
    return false;
}

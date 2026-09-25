#pragma once
// "\n</parameter>" tracker shared by the schema FSM (schema_constrain.cu) and its mask build
// (schema_constrain_mask.cpp).
#include <string>

namespace imp {

inline const std::string kXmlParamDelim = "\n</parameter>";

// Delimiter tracker for the raw-value phase: one KMP step over the delimiter. Chars are
// never rejected (any text is a legal value); a dead partial match falls back and retries.
// Relies on the delimiter's first char '\n' appearing nowhere else in it, making every
// KMP border 0 - a mismatch can only ask "did this char restart a match".
inline int xml_delim_step(const std::string& target, int len, char c) {
    if (len < static_cast<int>(target.size()) && c == target[len])
        return len + 1;
    return c == target[0] ? 1 : 0;
}

}  // namespace imp

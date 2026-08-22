// GBNF grammar parser (#1620). It had a repetition bound from the start and no
// nesting bound until #1609.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "compute/gbnf_grammar.h"

#include <new>
#include <string>
#include <vector>

extern "C" int imp_fuzz_gbnf(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > 65536)
        return 0;
    const std::string src(reinterpret_cast<const char*>(data), size);
    try {
        std::vector<imp::GbnfRule> rules;
        int32_t root = -1;
        std::string err;
        if (imp::parse_gbnf(src, rules, root, &err))
            (void)rules.size();
    } catch (const std::bad_alloc&) {}
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { return imp_fuzz_gbnf(data, size); }
#endif

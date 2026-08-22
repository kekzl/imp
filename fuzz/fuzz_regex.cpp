// Regex -> NFA compiler (#1620). Shipped defects: `{n,m}` cloned the atom with
// no bound and the digit scan overflowed a long (#1608), and the mutual
// recursion had no depth cap (#1609).

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "compute/json_schema.h"

#include <cstdint>
#include <new>
#include <string>
#include <vector>

extern "C" int imp_fuzz_regex(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    // A pattern is a request field; anything past a few KB is not a pattern.
    if (size > 8192)
        return 0;
    const std::string pattern(reinterpret_cast<const char*>(data), size);
    try {
        imp::RegexNfa nfa;
        if (!nfa.compile(pattern))
            return 0;
        // Drive the matcher too: compile() succeeding on a pattern that then
        // faults on the first byte is the more interesting failure.
        std::vector<int> states = nfa.start_set();
        for (unsigned char c : {uint8_t{'a'}, uint8_t{'0'}, uint8_t{'"'}, uint8_t{0xC3}, uint8_t{0x28}}) {
            states = nfa.step(states, c);
            if (states.empty())
                break;
        }
        (void)nfa.accepts(states);
    } catch (const std::bad_alloc&) {}
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { return imp_fuzz_regex(data, size); }
#endif

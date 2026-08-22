// JSON Schema parser (#1620). The defects this surface shipped: a value that
// desynced the key loop and truncated the rest of the document (#1564), and
// unbounded recursion (#1609).

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "compute/json_schema.h"

#include <new>
#include <string>

extern "C" int imp_fuzz_json_schema(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    const std::string s(reinterpret_cast<const char*>(data), size);
    try {
        auto node = imp::parse_json_schema(s);
        // Touch the result so the tree is not optimised away and a corrupt
        // node is dereferenced rather than merely allocated.
        if (node)
            (void)node->properties.size();
    } catch (const std::bad_alloc&) {
        // A resource limit, not a defect. Every other exception escapes on
        // purpose: this parser reports failure by returning nullptr, so a throw
        // out of it is the finding.
    }
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_json_schema(data, size);
}
#endif

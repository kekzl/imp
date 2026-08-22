// tokenizer.json loader against a real file (#1620). #1606: a negative token id
// was an out-of-bounds vector write during load, before any inference.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "model/tokenizer.h"

#include <exception>

extern "C" int imp_fuzz_tokenizer_json(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    imp_fuzz::TempFile f(data, size, ".json");
    if (!f.ok())
        return 0;
    try {
        imp::Tokenizer tok;
        if (tok.load(f.path())) {
            // Exercise the tables the load path filled, so an id that got past
            // the bounds check is dereferenced here rather than sitting unused.
            const int n = tok.vocab_size();
            for (int id : {0, 1, n - 1, n, -1})
                (void)tok.decode_token(id);
            (void)tok.encode("Grüße 😀");
        }
    } catch (const std::exception&) {}
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_tokenizer_json(data, size);
}
#endif

// SafeTensors shard loader against a real file (#1620).
//
// This surface carried four out-of-bounds accesses (#1603-#1606) and had no
// fault-injection battery at all: tests/test_safetensors_loader.cpp drove two
// extracted helpers, never load_safetensors() against a corrupt file.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "model/safetensors_loader.h"

#include <exception>

extern "C" int imp_fuzz_safetensors(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    imp_fuzz::TempFile f(data, size, ".safetensors");
    if (!f.ok())
        return 0;
    try {
        // No config.json next to it, so a Model is never built - the shard scan
        // is what this exercises, and that is where the defects were.
        auto model = imp::load_safetensors(f.path());
        (void)model;
    } catch (const std::exception&) {
        // The loader reports failure by returning nullptr, but the layers under
        // it (mmap, JSON) may throw on input this hostile. Catching keeps the
        // signal on memory errors, which is what ASan reports and what all four
        // shipped defects were.
    }
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_safetensors(data, size);
}
#endif

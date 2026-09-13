// SafeTensors shard loader against a real file (#1620): drives load_safetensors() directly,
// not just the two extracted helpers tests/test_safetensors_loader.cpp covers.

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
        // Loader reports failure via nullptr, but mmap/JSON layers under it may throw on hostile input.
        // Catching keeps the signal on memory errors, which is what ASan reports.
    }
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_safetensors(data, size);
}
#endif

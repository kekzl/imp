// GGUF loader against a real file (AUDIT_arch_2026 F1-12).
//
// The two S0 findings of the 2026 audit (F1-1: `n_dims > 4` stack write,
// F1-2: the mmproj fork of the same parse loop) were both in unfuzzed parsers,
// and the hand-written battery (tests/test_gguf_fault_injection.cpp) never
// patched `n_dims`. Same shape as fuzz_safetensors.cpp: the input becomes a
// file, the loader runs to whatever it refuses on. No GPU: load_gguf() builds
// the host-side Model over the mmap and stops before any upload.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "model/gguf_loader.h"

#include <exception>

extern "C" int imp_fuzz_gguf(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    imp_fuzz::TempFile f(data, size, ".gguf");
    if (!f.ok())
        return 0;
    try {
        auto model = imp::load_gguf(f.path());
        (void)model;
    } catch (const std::exception&) {
        // nullptr is the loader's refusal; a throw is the documented internal
        // error channel. Both are clean. Memory errors are what ASan reports.
    }
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { return imp_fuzz_gguf(data, size); }
#endif

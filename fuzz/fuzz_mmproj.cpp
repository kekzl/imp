// Vision (mmproj) GGUF loader against a real file (AUDIT_arch_2026 F1-12).
//
// Drives vision_gguf_probe(), the dry pass of the SAME load path the real
// upload takes: header, metadata, tensor infos, bounds, config extraction and
// every tensor's slot lookup, with the device uploads replaced by a byte count.
// That is the whole parser and none of the GPU, so it fits the CPU lane. The
// mmproj is the one file an operator usually downloads separately, and until
// F1-2 it was the loader with no bounds check at all.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "vision/vision_loader.h"

#include <exception>

extern "C" int imp_fuzz_mmproj(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    imp_fuzz::TempFile f(data, size, ".gguf");
    if (!f.ok())
        return 0;
    try {
        (void)imp::vision_gguf_probe(f.path());
    } catch (const std::exception&) {
        // 0 is the probe's refusal; a throw is the internal error channel.
    }
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_mmproj(data, size);
}
#endif

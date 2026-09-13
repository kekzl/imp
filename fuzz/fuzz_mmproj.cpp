// Vision (mmproj) GGUF loader (AUDIT_arch_2026 F1-12): drives vision_gguf_probe(), the same
// load path as the real upload (header, metadata, tensor infos, bounds) with uploads replaced
// by a byte count. CPU lane: whole parser, no GPU.

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

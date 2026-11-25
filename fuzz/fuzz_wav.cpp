// WAV (RIFF) decoder (AUDIT_ARCH_2026_11_24 X-5): decode_wav() on the bytes of an audio
// request part. Every chunk size in the buffer is attacker-chosen. CPU lane: whole parser, no GPU.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "audio/audio_wav.h"

#include <cstdint>
#include <new>
#include <span>

extern "C" int imp_fuzz_wav(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size > imp_fuzz::kMaxInput)
        return 0;
    try {
        const auto pcm = imp::decode_wav(std::span<const uint8_t>(data, size));
        // One mono sample needs at least one input byte (8-bit PCM, 1 channel).
        if (pcm && pcm->samples.size() > size)
            return 1;
    } catch (const std::bad_alloc&) {}
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // libFuzzer has no return-value channel for an invariant: abort instead.
    if (imp_fuzz_wav(data, size) != 0)
        __builtin_trap();
    return 0;
}
#endif

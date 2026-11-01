#pragma once
// WAV (RIFF) decoding for audio input: PCM 8/16/24/32-bit integer and IEEE float 32/64,
// WAVE_FORMAT_EXTENSIBLE included; channels averaged to mono. Untrusted input: every chunk
// size is bounds-checked against the buffer.

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace imp {

struct AudioPcm {
    std::vector<float> samples;  // mono, [-1, 1] (int16 / 32768, the soundfile convention)
    int sample_rate = 0;
};

[[nodiscard]] std::expected<AudioPcm, std::string> decode_wav(std::span<const uint8_t> bytes);

}  // namespace imp

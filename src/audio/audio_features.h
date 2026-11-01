#pragma once
// Gemma-4 audio front end, numerically HF Gemma4AudioFeatureExtractor (transformers 5.22.1):
// pad to a multiple of 128 samples, 160 zeros of semicausal left padding, 320-sample frames
// every 160 samples, periodic Hann, |rfft 512|, 128 HTK mel bins over 0..8000 Hz (no norm),
// log(mel + 1e-3); a frame is valid only when its whole window holds real samples, invalid
// frames are zeroed.

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace imp {

struct Gemma4AudioFrontend {
    int sample_rate = 16000;
    int n_mels = 128;
    int frame_length = 320;
    int hop_length = 160;
    int fft_length = 512;
    double min_frequency = 0.0;
    double max_frequency = 8000.0;
    double mel_floor = 1e-3;
    int pad_to_multiple = 128;
    int max_samples = 480000;   // 30 s: HF truncates, imp refuses
    int max_soft_tokens = 750;  // processor audio_seq_length
};

// One clip, CPU side: what the encoder consumes and how many soft tokens it yields.
struct AudioFeatures {
    int n_frames = 0;
    int n_mels = 0;
    std::vector<float> mel;     // [n_frames, n_mels], invalid frames zero
    std::vector<uint8_t> mask;  // [n_frames], 1 = valid frame
    int n_tokens = 0;           // soft tokens after the two stride-2 subsampling convs
    size_t content_hash = 0;    // of the source bytes, salts the prefix cache
};

// Mel filter bank [n_bins = fft/2+1, n_mels], HF mel_filter_bank(norm=None, mel_scale="htk").
std::vector<double> gemma4_mel_filters(const Gemma4AudioFrontend& fe);

// pcm: mono at fe.sample_rate. Refuses empty, too-short (no frame) and over-long clips.
[[nodiscard]] std::expected<AudioFeatures, std::string> gemma4_log_mel(std::span<const float> pcm,
                                                                       const Gemma4AudioFrontend& fe);

// HF Gemma4Processor.replace_audio_token: the frame mask through two stride-2 subsamplings,
// summed, capped at max_soft_tokens. Equals the encoder's valid output rows.
int gemma4_audio_soft_tokens(std::span<const uint8_t> frame_mask, int max_soft_tokens);

// WAV bytes -> features. Refuses a sample rate other than fe.sample_rate (no resampler).
[[nodiscard]] std::expected<AudioFeatures, std::string> gemma4_audio_from_wav(std::span<const uint8_t> wav,
                                                                              const Gemma4AudioFrontend& fe);

}  // namespace imp

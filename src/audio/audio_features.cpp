#include "audio/audio_features.h"

#include "audio/audio_wav.h"
#include "model/image_placeholders.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>

namespace imp {

namespace {

double hz_to_mel(double f) { return 2595.0 * std::log10(1.0 + (f / 700.0)); }
double mel_to_hz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// numpy.linspace: start + i * step, the last point exactly `stop`.
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> v(static_cast<size_t>(num));
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i)
        v[static_cast<size_t>(i)] = start + (i * step);
    v.back() = stop;
    return v;
}

// In-place iterative radix-2 FFT, n a power of two.
void fft_inplace(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> w(std::cos(ang * static_cast<double>(k)),
                                             std::sin(ang * static_cast<double>(k)));
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + (len / 2)] * w;
                a[i + k] = u + v;
                a[i + k + (len / 2)] = u - v;
            }
    }
}

}  // namespace

std::vector<double> gemma4_mel_filters(const Gemma4AudioFrontend& fe) {
    const int n_bins = (fe.fft_length / 2) + 1;
    const int n_mels = fe.n_mels;
    const std::vector<double> mel_pts = linspace(hz_to_mel(fe.min_frequency), hz_to_mel(fe.max_frequency),
                                                 n_mels + 2);
    std::vector<double> filt_hz(mel_pts.size());
    for (size_t i = 0; i < mel_pts.size(); ++i)
        filt_hz[i] = mel_to_hz(mel_pts[i]);
    const double nyquist = std::floor(static_cast<double>(fe.sample_rate) / 2.0);  // HF: sampling_rate // 2
    const std::vector<double> fft_hz = linspace(0.0, nyquist, n_bins);
    std::vector<double> out(static_cast<size_t>(n_bins) * n_mels, 0.0);
    for (int k = 0; k < n_bins; ++k)
        for (int m = 0; m < n_mels; ++m) {
            const double down = -(filt_hz[m] - fft_hz[k]) / (filt_hz[m + 1] - filt_hz[m]);
            const double up = (filt_hz[m + 2] - fft_hz[k]) / (filt_hz[m + 2] - filt_hz[m + 1]);
            out[(static_cast<size_t>(k) * n_mels) + m] = std::max(0.0, std::min(down, up));
        }
    return out;
}

int gemma4_audio_soft_tokens(std::span<const uint8_t> frame_mask, int max_soft_tokens) {
    // mask[::2] twice == mask[::4]
    int n = 0;
    for (size_t i = 0; i < frame_mask.size(); i += 4)
        n += frame_mask[i] ? 1 : 0;
    return std::min(n, max_soft_tokens);
}

std::expected<AudioFeatures, std::string> gemma4_log_mel(std::span<const float> pcm,
                                                         const Gemma4AudioFrontend& fe) {
    const int64_t n = static_cast<int64_t>(pcm.size());
    if (n == 0)
        return std::unexpected(std::string("audio clip is empty"));
    if (n > fe.max_samples)
        return std::unexpected("audio clip is " + std::to_string(n / fe.sample_rate) +
                               " s long; the limit is " + std::to_string(fe.max_samples / fe.sample_rate) +
                               " s per clip");
    const int64_t padded = ((n + fe.pad_to_multiple - 1) / fe.pad_to_multiple) * fe.pad_to_multiple;
    const int pad_left = fe.frame_length / 2;
    const int unfold = fe.frame_length + 1;
    const int64_t total = padded + pad_left;
    if (total < unfold)
        return std::unexpected(std::string("audio clip is too short for one analysis frame"));
    const int n_frames = static_cast<int>(((total - unfold) / fe.hop_length) + 1);

    // Signal with semicausal left padding and the multiple-of-128 right padding.
    std::vector<float> wav(static_cast<size_t>(total), 0.0f);
    std::copy(pcm.begin(), pcm.end(), wav.begin() + pad_left);
    std::vector<float> window(static_cast<size_t>(fe.frame_length));
    for (int i = 0; i < fe.frame_length; ++i)
        window[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - (0.5 * std::cos(2.0 * std::numbers::pi * i / static_cast<double>(fe.frame_length))));
    const std::vector<double> filters = gemma4_mel_filters(fe);
    const int n_bins = (fe.fft_length / 2) + 1;

    AudioFeatures out;
    out.n_frames = n_frames;
    out.n_mels = fe.n_mels;
    out.mel.assign(static_cast<size_t>(n_frames) * fe.n_mels, 0.0f);
    out.mask.assign(static_cast<size_t>(n_frames), 0);
    std::vector<std::complex<double>> buf(static_cast<size_t>(fe.fft_length));
    std::vector<double> mag(static_cast<size_t>(n_bins));
    for (int f = 0; f < n_frames; ++f) {
        const int64_t start = static_cast<int64_t>(f) * fe.hop_length;
        // Valid when the last sample of the 321-sample unfold window is real audio.
        const int64_t last = start + unfold - 1 - pad_left;
        const bool valid = last >= 0 && last < n;
        out.mask[static_cast<size_t>(f)] = valid ? 1 : 0;
        if (!valid)
            continue;  // HF multiplies the frame by its mask: exactly zero
        std::fill(buf.begin(), buf.end(), std::complex<double>(0.0, 0.0));
        for (int i = 0; i < fe.frame_length; ++i)
            buf[static_cast<size_t>(i)] = static_cast<double>(wav[static_cast<size_t>(start + i)] *
                                                              window[i]);
        fft_inplace(buf);
        for (int k = 0; k < n_bins; ++k)
            mag[static_cast<size_t>(k)] = std::abs(buf[static_cast<size_t>(k)]);
        float* row = out.mel.data() + (static_cast<size_t>(f) * fe.n_mels);
        for (int m = 0; m < fe.n_mels; ++m) {
            double acc = 0.0;
            for (int k = 0; k < n_bins; ++k)
                acc += mag[static_cast<size_t>(k)] * filters[(static_cast<size_t>(k) * fe.n_mels) + m];
            row[m] = static_cast<float>(std::log(acc + fe.mel_floor));
        }
    }
    out.n_tokens = gemma4_audio_soft_tokens(out.mask, fe.max_soft_tokens);
    if (out.n_tokens == 0)
        return std::unexpected(std::string("audio clip is too short: it yields no audio token"));
    return out;
}

std::expected<AudioFeatures, std::string> gemma4_audio_from_wav(std::span<const uint8_t> wav,
                                                                const Gemma4AudioFrontend& fe) {
    auto pcm = decode_wav(wav);
    if (!pcm)
        return std::unexpected(pcm.error());
    if (pcm->sample_rate != fe.sample_rate)
        return std::unexpected("WAV sample rate is " + std::to_string(pcm->sample_rate) +
                               " Hz; this model takes " + std::to_string(fe.sample_rate) +
                               " Hz audio (resample before sending)");
    auto feats = gemma4_log_mel(pcm->samples, fe);
    if (feats)
        feats->content_hash = image_content_hash(wav);
    return feats;
}

}  // namespace imp

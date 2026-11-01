// Gemma-4 audio front end vs HF Gemma4AudioFeatureExtractor (tests/refs golden) and the WAV
// decoder on hand-built files. CPU only.

#include "audio/audio_features.h"
#include "audio/audio_wav.h"
#include "model/image_placeholders.h"
#include "refs/gemma4_audio_features_golden.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <vector>

namespace imp {
namespace {

// Same formula as tests/refs/gen_gemma4_audio_golden.py.
std::vector<float> test_signal(int n) {
    std::vector<float> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = i / 16000.0;
        const double pi = std::numbers::pi;
        double v = (0.30 * std::sin(2 * pi * 440.0 * t)) + (0.20 * std::sin((2 * pi * 1234.5 * t) + 0.7));
        v += 0.10 * std::sin(2 * pi * (200.0 + (1500.0 * t)) * t);
        x[static_cast<size_t>(i)] = static_cast<float>(v);
    }
    return x;
}

// HF Gemma4Processor._compute_audio_num_tokens.
int hf_token_count(int64_t n_samples) {
    const int64_t frames = ((n_samples + 160 - 321) / 160) + 1;
    if (frames <= 0)
        return 0;
    int64_t t = frames;
    for (int i = 0; i < 2; ++i)
        t = ((t + 2 - 3) / 2) + 1;
    return static_cast<int>(std::min<int64_t>(t, 750));
}

void put16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>(v >> 8));
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
    put16(b, static_cast<uint16_t>(v & 0xFFFF));
    put16(b, static_cast<uint16_t>(v >> 16));
}

std::vector<uint8_t> make_wav(uint16_t format, uint16_t channels, uint32_t rate, uint16_t bits,
                              const std::vector<uint8_t>& payload, bool extensible = false) {
    std::vector<uint8_t> b;
    const uint32_t fmt_size = extensible ? 40 : 16;
    b.insert(b.end(), {'R', 'I', 'F', 'F'});
    put32(b, 4 + (8 + fmt_size) + 8 + 4 + static_cast<uint32_t>(payload.size()));
    b.insert(b.end(), {'W', 'A', 'V', 'E'});
    b.insert(b.end(), {'L', 'I', 'S', 'T'});  // a chunk the decoder has to skip
    put32(b, 4);
    b.insert(b.end(), {'I', 'N', 'F', 'O'});
    b.insert(b.end(), {'f', 'm', 't', ' '});
    put32(b, fmt_size);
    put16(b, extensible ? 0xFFFE : format);
    put16(b, channels);
    put32(b, rate);
    put32(b, rate * channels * (bits / 8));
    put16(b, static_cast<uint16_t>(channels * (bits / 8)));
    put16(b, bits);
    if (extensible) {
        put16(b, 22);
        put16(b, bits);
        put32(b, 0);
        put16(b, format);  // subformat GUID, first two bytes
        for (int i = 0; i < 14; ++i)
            b.push_back(0);
    }
    b.insert(b.end(), {'d', 'a', 't', 'a'});
    put32(b, static_cast<uint32_t>(payload.size()));
    b.insert(b.end(), payload.begin(), payload.end());
    return b;
}

TEST(Gemma4AudioFeatures, MatchesHfFeatureExtractor) {
    const Gemma4AudioFrontend fe;
    const auto x = test_signal(golden::kGemma4AudioSamples);
    auto f = gemma4_log_mel(x, fe);
    ASSERT_TRUE(f.has_value()) << f.error();
    ASSERT_EQ(f->n_frames, golden::kGemma4AudioFrames);
    ASSERT_EQ(f->n_mels, 128);
    int valid = 0;
    for (uint8_t m : f->mask)
        valid += m;
    EXPECT_EQ(valid, golden::kGemma4AudioValidFrames);
    for (int r = 0; r < f->n_frames; ++r) {
        double sum = 0.0;
        for (int m = 0; m < 128; ++m)
            sum += f->mel[(static_cast<size_t>(r) * 128) + m];
        EXPECT_NEAR(sum, golden::kGemma4AudioRowSums[r], 2e-3) << "frame " << r;
    }
    const float* rows[] = {golden::kGemma4AudioRow0, golden::kGemma4AudioRow1, golden::kGemma4AudioRow2,
                           golden::kGemma4AudioRow3};
    for (int i = 0; i < 4; ++i) {
        const int r = golden::kGemma4AudioRowIdx[i];
        for (int m = 0; m < 128; ++m)
            EXPECT_NEAR(f->mel[(static_cast<size_t>(r) * 128) + m], rows[i][m], 1e-4)
                << "frame " << r << " bin " << m;
    }
    EXPECT_EQ(f->n_tokens, hf_token_count(golden::kGemma4AudioSamples));
    EXPECT_EQ(f->n_tokens, 9);
}

TEST(Gemma4AudioFeatures, SoftTokenCountMatchesProcessorArithmetic) {
    const Gemma4AudioFrontend fe;
    for (int n : {481, 1600, 2048, 5920, 16000, 36877, 48001}) {
        auto f = gemma4_log_mel(test_signal(n), fe);
        ASSERT_TRUE(f.has_value()) << n << ": " << f.error();
        EXPECT_EQ(f->n_tokens, hf_token_count(n)) << n << " samples";
    }
}

TEST(Gemma4AudioFeatures, RefusesEmptyShortAndLongClips) {
    const Gemma4AudioFrontend fe;
    EXPECT_FALSE(gemma4_log_mel({}, fe).has_value());
    EXPECT_FALSE(gemma4_log_mel(test_signal(100), fe).has_value());
    EXPECT_FALSE(gemma4_log_mel(std::vector<float>(480001, 0.0f), fe).has_value());
}

TEST(Gemma4AudioFeatures, MelFiltersAreHtkTriangles) {
    const Gemma4AudioFrontend fe;
    const auto w = gemma4_mel_filters(fe);
    ASSERT_EQ(w.size(), 257u * 128u);
    EXPECT_EQ(w[0], 0.0);  // bin 0 (0 Hz) sits on the first filter's left edge
    // The lowest HTK filters are narrower than one 31.25 Hz bin and stay all-zero, as in HF
    // (it warns about exactly this); the rest peak inside (0, 1].
    int non_zero = 0;
    for (int m = 0; m < 128; ++m) {
        double peak = 0.0;
        for (int k = 0; k < 257; ++k)
            peak = std::max(peak, w[(static_cast<size_t>(k) * 128) + m]);
        EXPECT_LE(peak, 1.0) << "filter " << m;
        non_zero += peak > 0.0 ? 1 : 0;
    }
    EXPECT_GT(non_zero, 100);
    EXPECT_LT(non_zero, 128);
}

TEST(AudioWav, DecodesPcm16MonoWithExtraChunk) {
    std::vector<uint8_t> payload;
    for (int16_t v : {int16_t{0}, int16_t{16384}, int16_t{-32768}, int16_t{32767}})
        put16(payload, static_cast<uint16_t>(v));
    auto pcm = decode_wav(make_wav(1, 1, 16000, 16, payload));
    ASSERT_TRUE(pcm.has_value()) << pcm.error();
    EXPECT_EQ(pcm->sample_rate, 16000);
    ASSERT_EQ(pcm->samples.size(), 4u);
    EXPECT_EQ(pcm->samples[0], 0.0f);
    EXPECT_EQ(pcm->samples[1], 0.5f);
    EXPECT_EQ(pcm->samples[2], -1.0f);
    EXPECT_FLOAT_EQ(pcm->samples[3], 32767.0f / 32768.0f);
}

TEST(AudioWav, AveragesStereoAndReadsExtensibleFloat) {
    std::vector<uint8_t> payload;
    for (int16_t v : {int16_t{16384}, int16_t{0}, int16_t{-16384}, int16_t{-16384}})
        put16(payload, static_cast<uint16_t>(v));
    auto st = decode_wav(make_wav(1, 2, 16000, 16, payload));
    ASSERT_TRUE(st.has_value()) << st.error();
    ASSERT_EQ(st->samples.size(), 2u);
    EXPECT_EQ(st->samples[0], 0.25f);
    EXPECT_EQ(st->samples[1], -0.5f);

    std::vector<uint8_t> fp;
    for (float v : {0.125f, -0.75f}) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        put32(fp, u);
    }
    auto fl = decode_wav(make_wav(3, 1, 22050, 32, fp, /*extensible=*/true));
    ASSERT_TRUE(fl.has_value()) << fl.error();
    EXPECT_EQ(fl->sample_rate, 22050);
    ASSERT_EQ(fl->samples.size(), 2u);
    EXPECT_EQ(fl->samples[0], 0.125f);
    EXPECT_EQ(fl->samples[1], -0.75f);
}

TEST(AudioWav, RefusesMalformedFiles) {
    EXPECT_FALSE(decode_wav(std::vector<uint8_t>{'R', 'I', 'F', 'F'}).has_value());
    std::vector<uint8_t> payload(8, 0);
    auto bad_format = make_wav(2, 1, 16000, 16, payload);  // ADPCM
    EXPECT_FALSE(decode_wav(bad_format).has_value());
    auto wav = make_wav(1, 1, 16000, 16, payload);
    wav.resize(wav.size() - 3);  // data chunk declares 8 bytes, 5 arrive
    auto cut = decode_wav(wav);
    ASSERT_TRUE(cut.has_value()) << cut.error();  // keeps the whole frames it has
    EXPECT_EQ(cut->samples.size(), 2u);
    wav.resize(30);  // inside the fmt chunk
    EXPECT_FALSE(decode_wav(wav).has_value());
}

TEST(Gemma4AudioFeatures, FromWavRefusesOtherSampleRates) {
    const Gemma4AudioFrontend fe;
    std::vector<uint8_t> payload;
    for (float v : test_signal(8000))
        put16(payload, static_cast<uint16_t>(static_cast<int16_t>(std::lround(v * 32767.0f))));
    auto ok = gemma4_audio_from_wav(make_wav(1, 1, 16000, 16, payload), fe);
    ASSERT_TRUE(ok.has_value()) << ok.error();
    EXPECT_EQ(ok->n_tokens, hf_token_count(8000));
    EXPECT_NE(ok->content_hash, 0u);
    auto wrong = gemma4_audio_from_wav(make_wav(1, 1, 8000, 16, payload), fe);
    ASSERT_FALSE(wrong.has_value());
    EXPECT_NE(wrong.error().find("16000"), std::string::npos);
}

// HF Gemma4Processor.replace_audio_token: <|audio|> -> <boa> N x <audio> <eoa>, per clip.
TEST(Gemma4AudioPlaceholders, ExpandsEachClipToItsOwnCount) {
    constexpr int32_t kAudio = 258881, kBoa = 256000, kEoa = 258883;
    std::vector<int32_t> t = {2, 10, kAudio, kAudio, 11, 12};
    ASSERT_TRUE(expand_audio_placeholders(t, kAudio, kBoa, kEoa, {2, 3}).has_value());
    const std::vector<int32_t> want = {2,      10,     kBoa,   kAudio, kAudio, kEoa, kBoa,
                                       kAudio, kAudio, kAudio, kEoa,   11,     12};
    EXPECT_EQ(t, want);
}

TEST(Gemma4AudioPlaceholders, RefusesCountMismatchAndLeavesTokens) {
    constexpr int32_t kAudio = 7;
    const std::vector<int32_t> orig = {1, kAudio, 2, kAudio};  // a literal placeholder typed by the user
    std::vector<int32_t> t = orig;
    EXPECT_FALSE(expand_audio_placeholders(t, kAudio, 5, 6, {4}).has_value());
    EXPECT_EQ(t, orig);
    EXPECT_FALSE(expand_audio_placeholders(t, kAudio, 5, 6, {4, 0}).has_value());
    EXPECT_EQ(t, orig);
}

}  // namespace
}  // namespace imp

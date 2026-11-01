#include "audio/audio_wav.h"

#include <cstring>

namespace imp {

namespace {

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatFloat = 3;
constexpr uint16_t kFormatExtensible = 0xFFFE;

// One sample at p, normalised like soundfile: integer / 2^(bits-1).
float sample_at(const uint8_t* p, uint16_t format, uint16_t bits) {
    if (format == kFormatFloat) {
        if (bits == 32) {
            float f;
            std::memcpy(&f, p, 4);
            return f;
        }
        double d;
        std::memcpy(&d, p, 8);
        return static_cast<float>(d);
    }
    switch (bits) {
        case 8:
            return (static_cast<float>(p[0]) - 128.0f) / 128.0f;
        case 16:
            return static_cast<float>(static_cast<int16_t>(rd16(p))) / 32768.0f;
        case 24: {
            int32_t v = static_cast<int32_t>(p[0] | (p[1] << 8) | (p[2] << 16));
            if (v & 0x800000)
                v -= 0x1000000;
            return static_cast<float>(v) / 8388608.0f;
        }
        default:
            return static_cast<float>(static_cast<double>(static_cast<int32_t>(rd32(p))) / 2147483648.0);
    }
}

struct WavLayout {
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;
    const uint8_t* data = nullptr;
    size_t data_bytes = 0;
};

// fmt chunk body at ck + 8, ck_size bytes (already bounds-checked against the buffer).
std::expected<void, std::string> read_fmt(const uint8_t* ck, size_t ck_size, WavLayout& w) {
    if (ck_size < 16)
        return std::unexpected(std::string("truncated fmt chunk"));
    w.format = rd16(ck + 8);
    w.channels = rd16(ck + 10);
    w.rate = rd32(ck + 12);
    w.bits = rd16(ck + 22);
    if (w.format == kFormatExtensible) {
        if (ck_size < 40)
            return std::unexpected(std::string("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk"));
        w.format = rd16(ck + 8 + 24);  // first two bytes of the subformat GUID
    }
    w.have_fmt = true;
    return {};
}

// Walks the RIFF chunks up to the data chunk.
std::expected<WavLayout, std::string> scan_chunks(std::span<const uint8_t> bytes) {
    const uint8_t* b = bytes.data();
    const size_t n = bytes.size();
    WavLayout w;
    for (size_t pos = 12; pos + 8 <= n;) {
        const uint8_t* ck = b + pos;
        const size_t ck_size = rd32(ck + 4);
        const size_t avail = n - (pos + 8);
        if (std::memcmp(ck, "data", 4) == 0) {
            w.data = ck + 8;
            w.data_bytes = ck_size > avail ? avail : ck_size;  // streamed writers leave 0xFFFFFFFF
            break;
        }
        if (ck_size > avail) {
            if (std::memcmp(ck, "fmt ", 4) == 0)
                return std::unexpected(std::string("truncated fmt chunk"));
            break;
        }
        if (std::memcmp(ck, "fmt ", 4) == 0)
            if (auto r = read_fmt(ck, ck_size, w); !r)
                return std::unexpected(r.error());
        pos += 8 + ck_size + (ck_size & 1);
    }
    return w;
}

std::expected<void, std::string> check_layout(const WavLayout& w) {
    if (!w.have_fmt)
        return std::unexpected(std::string("WAV has no fmt chunk"));
    if (w.data == nullptr)
        return std::unexpected(std::string("WAV has no data chunk"));
    const uint16_t b = w.bits;
    const bool int_ok = w.format == kFormatPcm && (b == 8 || b == 16 || b == 24 || b == 32);
    const bool float_ok = w.format == kFormatFloat && (b == 32 || b == 64);
    if (!int_ok && !float_ok)
        return std::unexpected("unsupported WAV encoding (format " + std::to_string(w.format) + ", " +
                               std::to_string(b) + " bit); send PCM integer or IEEE float");
    if (w.channels == 0 || w.channels > 32)
        return std::unexpected("unsupported WAV channel count " + std::to_string(w.channels));
    if (w.rate == 0)
        return std::unexpected(std::string("WAV sample rate is 0"));
    return {};
}

}  // namespace

std::expected<AudioPcm, std::string> decode_wav(std::span<const uint8_t> bytes) {
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        return std::unexpected(std::string("not a RIFF/WAVE file"));
    auto w = scan_chunks(bytes);
    if (!w)
        return std::unexpected(w.error());
    if (auto ok = check_layout(*w); !ok)
        return std::unexpected(ok.error());
    const size_t sample_bytes = w->bits / 8;
    const size_t frame_bytes = sample_bytes * w->channels;
    const size_t frames = w->data_bytes / frame_bytes;
    AudioPcm out;
    out.sample_rate = static_cast<int>(w->rate);
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        const uint8_t* fp = w->data + (i * frame_bytes);
        float acc = 0.0f;
        for (uint16_t c = 0; c < w->channels; ++c)
            acc += sample_at(fp + (c * sample_bytes), w->format, w->bits);
        out.samples[i] = acc / static_cast<float>(w->channels);
    }
    return out;
}

}  // namespace imp

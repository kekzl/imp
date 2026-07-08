// CPU unit tests for the base64 codec in tools/imp-server/utils.cpp.
// base64_encode was added for the OpenAI `encoding_format: "base64"` embeddings
// response (the little-endian float32 array encoded as bytes) and had no
// coverage; base64_decode backs the image-data path. These assert the RFC 4648
// vectors, padding at every residue, and a bytes round-trip.

#include "utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string enc(const std::string& s) {
    return base64_encode(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

std::string dec(const std::string& s) {
    auto v = base64_decode(s);
    return std::string(v.begin(), v.end());
}

}  // namespace

TEST(Base64, Rfc4648Vectors) {
    // The canonical test vectors from RFC 4648 §10.
    EXPECT_EQ(enc(""), "");
    EXPECT_EQ(enc("f"), "Zg==");
    EXPECT_EQ(enc("fo"), "Zm8=");
    EXPECT_EQ(enc("foo"), "Zm9v");
    EXPECT_EQ(enc("foob"), "Zm9vYg==");
    EXPECT_EQ(enc("fooba"), "Zm9vYmE=");
    EXPECT_EQ(enc("foobar"), "Zm9vYmFy");
}

TEST(Base64, PaddingAtEveryResidue) {
    // len % 3 == 1 → two '=', == 2 → one '=', == 0 → none.
    EXPECT_EQ(enc("f").size() % 4, 0u);
    EXPECT_EQ(enc("f").substr(2), "==");
    EXPECT_EQ(enc("fo").substr(3), "=");
    EXPECT_EQ(enc("foo").find('='), std::string::npos);
}

TEST(Base64, DecodeInvertsEncode) {
    for (const std::string& s : {std::string(""), std::string("a"), std::string("ab"),
                                 std::string("abc"), std::string("hello, world"),
                                 std::string("\x00\x01\x02\xff\xfe", 5)}) {
        EXPECT_EQ(dec(enc(s)), s) << "round-trip failed for len " << s.size();
    }
}

TEST(Base64, EncodesRawFloatBytes) {
    // The embeddings path encodes the little-endian float32 array as bytes;
    // decoding must reproduce the exact float payload.
    std::vector<float> v = {0.0f, 1.0f, -2.5f, 3.14159f};
    std::string b64 = base64_encode(reinterpret_cast<const uint8_t*>(v.data()), v.size() * sizeof(float));
    auto bytes = base64_decode(b64);
    ASSERT_EQ(bytes.size(), v.size() * sizeof(float));
    std::vector<float> back(v.size());
    std::memcpy(back.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_FLOAT_EQ(back[i], v[i]);
}

TEST(Base64, HandlesAllByteValues) {
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i)
        all[i] = static_cast<uint8_t>(i);
    std::string b64 = base64_encode(all.data(), all.size());
    // No stray characters outside the standard alphabet + padding.
    for (char c : b64)
        EXPECT_TRUE((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                    c == '+' || c == '/' || c == '=')
            << "unexpected char in output: " << c;
    EXPECT_EQ(base64_decode(b64), all);
}

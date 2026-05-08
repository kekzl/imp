// Unit tests for the native SentencePiece (.model protobuf) parser.
// Closes audit followups item AU2.
//
// Strategy: synthesize valid protobuf blobs in-memory matching the
// sentencepiece.ModelProto schema, then verify parse_sentencepiece_model
// extracts the expected vocabulary, scores, types, and trainer ids. Plus
// one optional integration test that runs against a real spiece.model
// file from the user's HF cache when available — gracefully skips
// otherwise so the unit suite stays GPU-runner-agnostic.

#include "model/sentencepiece_loader.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace imp {
namespace {

// ---- Protobuf wire-format encoders (test fixtures only) ----

void put_varint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

void put_signed_int32_as_varint(std::vector<uint8_t>& out, int32_t v) {
    // Protobuf int32 negative encodes as 10-byte sign-extended uint64.
    uint64_t u;
    if (v < 0) {
        u = static_cast<uint64_t>(static_cast<int64_t>(v));
    } else {
        u = static_cast<uint64_t>(v);
    }
    put_varint(out, u);
}

void put_tag(std::vector<uint8_t>& out, uint32_t field, uint32_t wire) {
    put_varint(out, (static_cast<uint64_t>(field) << 3) | (wire & 0x7));
}

void put_string_field(std::vector<uint8_t>& out, uint32_t field, const std::string& s) {
    put_tag(out, field, 2);
    put_varint(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

void put_float_field(std::vector<uint8_t>& out, uint32_t field, float f) {
    put_tag(out, field, 5);
    uint8_t b[4];
    std::memcpy(b, &f, 4);
    out.insert(out.end(), b, b + 4);
}

void put_varint_field(std::vector<uint8_t>& out, uint32_t field, uint64_t v) {
    put_tag(out, field, 0);
    put_varint(out, v);
}

void put_signed_int32_field(std::vector<uint8_t>& out, uint32_t field, int32_t v) {
    put_tag(out, field, 0);
    put_signed_int32_as_varint(out, v);
}

void put_submessage(std::vector<uint8_t>& out, uint32_t field, const std::vector<uint8_t>& body) {
    put_tag(out, field, 2);
    put_varint(out, body.size());
    out.insert(out.end(), body.begin(), body.end());
}

// Build one ModelProto.SentencePiece sub-message body bytes.
std::vector<uint8_t> make_piece(const std::string& s, float score, int32_t type) {
    std::vector<uint8_t> body;
    put_string_field(body, 1, s);
    put_float_field(body, 2, score);
    if (type != 1)  // proto omits default
        put_varint_field(body, 3, static_cast<uint64_t>(type));
    return body;
}

// ---- Tests ----

TEST(SentencePieceLoader, ParsesPiecesAndScores) {
    std::vector<uint8_t> blob;
    put_submessage(blob, 1, make_piece("<unk>", 0.0f, 2));
    put_submessage(blob, 1, make_piece("<s>", 0.0f, 3));
    put_submessage(blob, 1, make_piece("</s>", 0.0f, 3));
    put_submessage(blob, 1, make_piece("hello", -1.5f, 1));
    put_submessage(blob, 1, make_piece("world", -2.0f, 1));

    SentencePieceModel m;
    std::string err;
    ASSERT_TRUE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err)) << err;

    ASSERT_EQ(m.pieces.size(), 5u);
    EXPECT_EQ(m.pieces[0], "<unk>");
    EXPECT_EQ(m.pieces[1], "<s>");
    EXPECT_EQ(m.pieces[2], "</s>");
    EXPECT_EQ(m.pieces[3], "hello");
    EXPECT_EQ(m.pieces[4], "world");

    ASSERT_EQ(m.scores.size(), 5u);
    EXPECT_FLOAT_EQ(m.scores[3], -1.5f);
    EXPECT_FLOAT_EQ(m.scores[4], -2.0f);

    ASSERT_EQ(m.types.size(), 5u);
    EXPECT_EQ(m.types[0], 2);  // UNKNOWN
    EXPECT_EQ(m.types[1], 3);  // CONTROL
    EXPECT_EQ(m.types[2], 3);  // CONTROL
    EXPECT_EQ(m.types[3], 1);  // NORMAL (default)
    EXPECT_EQ(m.types[4], 1);  // NORMAL
}

TEST(SentencePieceLoader, ParsesTrainerSpec) {
    // Build a valid ModelProto with trainer_spec carrying non-default ids.
    std::vector<uint8_t> trainer_body;
    put_varint_field(trainer_body, 3, 1);             // model_type = UNIGRAM
    put_signed_int32_field(trainer_body, 40, 0);      // unk_id
    put_signed_int32_field(trainer_body, 41, 1);      // bos_id
    put_signed_int32_field(trainer_body, 42, 2);      // eos_id
    put_signed_int32_field(trainer_body, 43, -1);     // pad_id (default, but explicit)

    std::vector<uint8_t> blob;
    put_submessage(blob, 1, make_piece("<unk>", 0.0f, 2));
    put_submessage(blob, 1, make_piece("<s>", 0.0f, 3));
    put_submessage(blob, 1, make_piece("</s>", 0.0f, 3));
    put_submessage(blob, 2, trainer_body);

    SentencePieceModel m;
    std::string err;
    ASSERT_TRUE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err)) << err;
    EXPECT_EQ(m.bos_id, 1);
    EXPECT_EQ(m.eos_id, 2);
    EXPECT_EQ(m.unk_id, 0);
    EXPECT_EQ(m.pad_id, -1);
    EXPECT_EQ(m.model_type, SentencePieceModel::ModelType::UNIGRAM);
}

TEST(SentencePieceLoader, ModelTypeBPE) {
    std::vector<uint8_t> trainer_body;
    put_varint_field(trainer_body, 3, 2);  // BPE

    std::vector<uint8_t> blob;
    put_submessage(blob, 1, make_piece("a", 0.0f, 1));
    put_submessage(blob, 2, trainer_body);

    SentencePieceModel m;
    ASSERT_TRUE(parse_sentencepiece_model(blob.data(), blob.size(), &m, nullptr));
    EXPECT_EQ(m.model_type, SentencePieceModel::ModelType::BPE);
}

TEST(SentencePieceLoader, RejectsEmptyInput) {
    SentencePieceModel m;
    std::string err;
    EXPECT_FALSE(parse_sentencepiece_model(nullptr, 0, &m, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SentencePieceLoader, RejectsNoPieces) {
    // Valid protobuf wire format but no field 1 entries → rejected.
    std::vector<uint8_t> trainer_body;
    put_varint_field(trainer_body, 3, 1);
    std::vector<uint8_t> blob;
    put_submessage(blob, 2, trainer_body);

    SentencePieceModel m;
    std::string err;
    EXPECT_FALSE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err));
    EXPECT_NE(err.find("no pieces"), std::string::npos) << err;
}

TEST(SentencePieceLoader, RejectsTruncatedVarint) {
    // Tag for field 1, wire 2, then no length byte — truncated.
    std::vector<uint8_t> blob = {0x0A};  // tag = (1<<3) | 2
    SentencePieceModel m;
    std::string err;
    EXPECT_FALSE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SentencePieceLoader, RejectsLengthDelimPastEnd) {
    // Field 1, wire 2, claimed length 100, only 5 bytes follow.
    std::vector<uint8_t> blob;
    put_tag(blob, 1, 2);
    put_varint(blob, 100);
    blob.push_back('a');
    blob.push_back('b');
    blob.push_back('c');
    blob.push_back('d');
    blob.push_back('e');

    SentencePieceModel m;
    std::string err;
    EXPECT_FALSE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err));
    EXPECT_FALSE(err.empty());
}

TEST(SentencePieceLoader, SkipsUnknownFields) {
    // Real .model files have lots of fields we don't parse (NormalizerSpec,
    // SelfTestData). Verify they don't break the parse.
    std::vector<uint8_t> blob;

    // Field 99 (unknown), wire 0 (varint)
    put_varint_field(blob, 99, 12345);

    // Field 100 (unknown), wire 5 (fixed32)
    put_tag(blob, 100, 5);
    uint32_t junk = 0xDEADBEEF;
    blob.insert(blob.end(), reinterpret_cast<uint8_t*>(&junk), reinterpret_cast<uint8_t*>(&junk) + 4);

    // Field 101 (unknown), wire 1 (fixed64)
    put_tag(blob, 101, 1);
    uint64_t junk64 = 0xFEEDC0DEDEADBEEFull;
    blob.insert(blob.end(), reinterpret_cast<uint8_t*>(&junk64), reinterpret_cast<uint8_t*>(&junk64) + 8);

    // Real piece
    put_submessage(blob, 1, make_piece("hello", -1.0f, 1));

    SentencePieceModel m;
    std::string err;
    ASSERT_TRUE(parse_sentencepiece_model(blob.data(), blob.size(), &m, &err)) << err;
    ASSERT_EQ(m.pieces.size(), 1u);
    EXPECT_EQ(m.pieces[0], "hello");
}

TEST(SentencePieceLoader, NegativePadIdRoundtrips) {
    // pad_id default is -1, encoded as a 10-byte sign-extended varint.
    std::vector<uint8_t> trainer_body;
    put_signed_int32_field(trainer_body, 43, -1);

    std::vector<uint8_t> blob;
    put_submessage(blob, 1, make_piece("a", 0.0f, 1));
    put_submessage(blob, 2, trainer_body);

    SentencePieceModel m;
    ASSERT_TRUE(parse_sentencepiece_model(blob.data(), blob.size(), &m, nullptr));
    EXPECT_EQ(m.pad_id, -1);
}

// Optional integration test: parse a real spiece.model from the HF cache.
// Skipped when the file is not present (the test runner box may differ).
TEST(SentencePieceLoader, RealHfCacheSpieceModelLoadsCleanly) {
    namespace fs = std::filesystem;
    const std::vector<std::string> candidates = {
        // Host path (when running outside docker)
        "/home/kekz/.cache/huggingface/hub/models--facebook--musicgen-small/"
        "snapshots/4c8334b02c6ec4e8664a91979669a501ec497792/spiece.model",
        // Container path (when -v /home/kekz/.cache/huggingface:/hf_cache is bound)
        "/hf_cache/hub/models--facebook--musicgen-small/"
        "snapshots/4c8334b02c6ec4e8664a91979669a501ec497792/spiece.model",
    };

    std::string found;
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            found = p;
            break;
        }
    }
    if (found.empty()) {
        GTEST_SKIP() << "no real spiece.model fixture present";
    }

    SentencePieceModel m = load_sentencepiece_model_file(found);
    EXPECT_FALSE(m.empty());
    EXPECT_GT(m.pieces.size(), 100u) << "expected a non-trivial vocabulary";
    // T5 vocab has <pad>, </s>, <unk> as ids 0, 1, 2.
    EXPECT_EQ(m.pieces[0], "<pad>");
    EXPECT_EQ(m.pieces[1], "</s>");
    EXPECT_EQ(m.pieces[2], "<unk>");
    EXPECT_EQ(m.pieces.size(), m.scores.size());
    EXPECT_EQ(m.pieces.size(), m.types.size());
}

}  // namespace
}  // namespace imp

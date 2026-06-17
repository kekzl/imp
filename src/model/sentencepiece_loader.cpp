#include "model/sentencepiece_loader.h"

#include "core/logging.h"

#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace imp {

namespace {

// ---- Protobuf wire-format primitives (proto2/proto3 binary encoding) ----
//
// Each field on the wire is `tag` (varint) followed by typed payload:
//   wire_type 0 = varint   (int32/int64/uint32/uint64/sint32/sint64/bool/enum)
//   wire_type 1 = fixed64  (double, fixed64, sfixed64)
//   wire_type 2 = length-delimited (string, bytes, sub-message, packed repeated)
//   wire_type 5 = fixed32  (float, fixed32, sfixed32)
//
// `tag = (field_number << 3) | wire_type`.
//
// We only need a tiny subset for SentencePiece ModelProto; unknown fields
// are skipped without error.

class ProtoReader {
public:
    ProtoReader(const uint8_t* data, size_t size) : data_(data), end_(data + size) {}

    bool ok() const { return ok_; }
    bool at_end() const { return data_ >= end_; }
    const char* err() const { return err_; }

    // Returns false on truncated varint or overflow.
    bool read_varint(uint64_t* out) {
        uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (data_ >= end_) {
                fail("truncated varint");
                return false;
            }
            uint8_t b = *data_++;
            v |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                *out = v;
                return true;
            }
        }
        fail("varint > 10 bytes");
        return false;
    }

    bool read_fixed32(uint32_t* out) {
        if (data_ + 4 > end_) {
            fail("truncated fixed32");
            return false;
        }
        std::memcpy(out, data_, 4);
        data_ += 4;
        return true;
    }

    bool read_fixed64(uint64_t* out) {
        if (data_ + 8 > end_) {
            fail("truncated fixed64");
            return false;
        }
        std::memcpy(out, data_, 8);
        data_ += 8;
        return true;
    }

    // Returns a view into the underlying buffer (length-delimited block).
    bool read_length_delim(const uint8_t** out_data, size_t* out_size) {
        uint64_t len = 0;
        if (!read_varint(&len))
            return false;
        if (data_ + len > end_) {
            fail("length-delim past end");
            return false;
        }
        *out_data = data_;
        *out_size = static_cast<size_t>(len);
        data_ += len;
        return true;
    }

    // Skip the payload for a given wire_type. Used for unknown fields.
    bool skip_field(uint32_t wire_type) {
        switch (wire_type) {
            case 0: {  // varint
                uint64_t v;
                return read_varint(&v);
            }
            case 1: {  // fixed64
                if (data_ + 8 > end_) {
                    fail("skip fixed64 past end");
                    return false;
                }
                data_ += 8;
                return true;
            }
            case 2: {  // length-delimited
                const uint8_t* p;
                size_t n;
                return read_length_delim(&p, &n);
            }
            case 5: {  // fixed32
                if (data_ + 4 > end_) {
                    fail("skip fixed32 past end");
                    return false;
                }
                data_ += 4;
                return true;
            }
            default:
                fail("unknown wire_type");
                return false;
        }
    }

private:
    void fail(const char* msg) {
        if (ok_) {
            ok_ = false;
            err_ = msg;
        }
    }
    const uint8_t* data_;
    const uint8_t* end_;
    bool ok_ = true;
    const char* err_ = nullptr;
};

// ---- ModelProto.SentencePiece sub-message ----
//
//   message SentencePiece {
//     optional string piece = 1;
//     optional float  score = 2;
//     optional Type   type  = 3;     // enum NORMAL=1, UNKNOWN=2, CONTROL=3, ...
//   }
struct SpPiece {
    std::string piece;
    float score = 0.0f;
    int32_t type = 1;  // default NORMAL
};

bool parse_sentencepiece(const uint8_t* data, size_t size, SpPiece* out) {
    ProtoReader r(data, size);
    while (!r.at_end()) {
        uint64_t tag = 0;
        if (!r.read_varint(&tag))
            return false;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
            case 1: {  // piece
                if (wire != 2) {
                    if (!r.skip_field(wire))
                        return false;
                    break;
                }
                const uint8_t* p;
                size_t n;
                if (!r.read_length_delim(&p, &n))
                    return false;
                out->piece.assign(reinterpret_cast<const char*>(p), n);
                break;
            }
            case 2: {  // score (float, fixed32)
                if (wire != 5) {
                    if (!r.skip_field(wire))
                        return false;
                    break;
                }
                uint32_t bits = 0;
                if (!r.read_fixed32(&bits))
                    return false;
                std::memcpy(&out->score, &bits, 4);
                break;
            }
            case 3: {  // type (enum, varint)
                if (wire != 0) {
                    if (!r.skip_field(wire))
                        return false;
                    break;
                }
                uint64_t v = 0;
                if (!r.read_varint(&v))
                    return false;
                out->type = static_cast<int32_t>(v);
                break;
            }
            default:
                if (!r.skip_field(wire))
                    return false;
        }
    }
    return r.ok();
}

// ---- ModelProto.TrainerSpec sub-message (only the fields we need) ----
//
//   message TrainerSpec {
//     optional ModelType model_type = 3;     // UNIGRAM=1, BPE=2, WORD=3, CHAR=4
//     optional int32 unk_id = 40 [default = 0];
//     optional int32 bos_id = 41 [default = 1];
//     optional int32 eos_id = 42 [default = 2];
//     optional int32 pad_id = 43 [default = -1];
//     // ... lots of other fields, all skipped
//   }
struct SpTrainer {
    int32_t model_type = 0;  // UNKNOWN
    int32_t unk_id = 0;
    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t pad_id = -1;
};

bool parse_trainer_spec(const uint8_t* data, size_t size, SpTrainer* out) {
    ProtoReader r(data, size);
    while (!r.at_end()) {
        uint64_t tag = 0;
        if (!r.read_varint(&tag))
            return false;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        if (wire == 0) {
            uint64_t v = 0;
            if (!r.read_varint(&v))
                return false;
            // Treat varint as int32; protobuf signed varints (sint32) use
            // zigzag, but the SentencePiece schema uses int32 directly so
            // a -1 default appears as a 10-byte all-ones varint already.
            int32_t iv = static_cast<int32_t>(v);
            switch (field) {
                case 3:
                    out->model_type = iv;
                    break;
                case 40:
                    out->unk_id = iv;
                    break;
                case 41:
                    out->bos_id = iv;
                    break;
                case 42:
                    out->eos_id = iv;
                    break;
                case 43:
                    out->pad_id = iv;
                    break;
                default:
                    break;
            }
        } else {
            if (!r.skip_field(wire))
                return false;
        }
    }
    return r.ok();
}

}  // namespace

bool parse_sentencepiece_model(const void* data, size_t size, SentencePieceModel* out, std::string* err) {
    if (!data || size == 0) {
        if (err)
            *err = "empty input";
        return false;
    }
    if (!out) {
        if (err)
            *err = "null output";
        return false;
    }
    *out = SentencePieceModel{};

    ProtoReader r(static_cast<const uint8_t*>(data), size);
    SpTrainer trainer;
    bool got_trainer = false;

    while (!r.at_end()) {
        uint64_t tag = 0;
        if (!r.read_varint(&tag)) {
            if (err)
                *err = r.err() ? r.err() : "varint read failed";
            return false;
        }
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        // ModelProto fields:
        //   1 = repeated SentencePiece pieces
        //   2 = TrainerSpec trainer_spec
        //   3 = NormalizerSpec normalizer_spec
        //   4 = SelfTestData self_test_data
        //   5 = NormalizerSpec denormalizer_spec
        if (field == 1 && wire == 2) {  // SentencePiece pieces (length-delimited)
            const uint8_t* p;
            size_t n;
            if (!r.read_length_delim(&p, &n)) {
                if (err)
                    *err = r.err() ? r.err() : "pieces read failed";
                return false;
            }
            SpPiece sp;
            if (!parse_sentencepiece(p, n, &sp)) {
                if (err)
                    *err = "malformed SentencePiece sub-message";
                return false;
            }
            out->pieces.push_back(std::move(sp.piece));
            out->scores.push_back(sp.score);
            out->types.push_back(sp.type);
            continue;
        }
        if (field == 2 && wire == 2) {  // TrainerSpec
            const uint8_t* p;
            size_t n;
            if (!r.read_length_delim(&p, &n)) {
                if (err)
                    *err = r.err() ? r.err() : "trainer_spec read failed";
                return false;
            }
            if (!parse_trainer_spec(p, n, &trainer)) {
                if (err)
                    *err = "malformed TrainerSpec sub-message";
                return false;
            }
            got_trainer = true;
            continue;
        }
        // Unknown / skipped field.
        if (!r.skip_field(wire)) {
            if (err)
                *err = r.err() ? r.err() : "skip_field failed";
            return false;
        }
    }

    if (out->pieces.empty()) {
        if (err)
            *err = "no pieces in ModelProto";
        return false;
    }

    if (got_trainer) {
        out->bos_id = trainer.bos_id;
        out->eos_id = trainer.eos_id;
        out->unk_id = trainer.unk_id;
        out->pad_id = trainer.pad_id;
        switch (trainer.model_type) {
            case 1:
                out->model_type = SentencePieceModel::ModelType::UNIGRAM;
                break;
            case 2:
                out->model_type = SentencePieceModel::ModelType::BPE;
                break;
            case 3:
                out->model_type = SentencePieceModel::ModelType::WORD;
                break;
            case 4:
                out->model_type = SentencePieceModel::ModelType::CHAR;
                break;
            default:
                out->model_type = SentencePieceModel::ModelType::UNKNOWN;
        }
    }
    return true;
}

SentencePieceModel load_sentencepiece_model_file(const std::string& path) {
    SentencePieceModel result;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("SentencePiece: failed to open %s", path.c_str());
        return result;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        IMP_LOG_ERROR("SentencePiece: fstat failed on %s", path.c_str());
        return result;
    }
    size_t size = static_cast<size_t>(st.st_size);
    if (size == 0) {
        ::close(fd);
        IMP_LOG_ERROR("SentencePiece: empty file %s", path.c_str());
        return result;
    }
    void* mm = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        IMP_LOG_ERROR("SentencePiece: mmap failed on %s", path.c_str());
        return result;
    }

    std::string err;
    if (!parse_sentencepiece_model(mm, size, &result, &err)) {
        IMP_LOG_ERROR("SentencePiece: parse failed for %s — %s", path.c_str(), err.c_str());
        result = SentencePieceModel{};
    } else {
        const char* type_name = "unknown";
        switch (result.model_type) {
            case SentencePieceModel::ModelType::UNIGRAM:
                type_name = "unigram";
                break;
            case SentencePieceModel::ModelType::BPE:
                type_name = "bpe";
                break;
            case SentencePieceModel::ModelType::WORD:
                type_name = "word";
                break;
            case SentencePieceModel::ModelType::CHAR:
                type_name = "char";
                break;
            default:
                break;
        }
        IMP_LOG_INFO("SentencePiece: parsed %s — %zu pieces, model_type=%s, bos=%d eos=%d unk=%d",
                     path.c_str(), result.pieces.size(), type_name, result.bos_id, result.eos_id,
                     result.unk_id);
    }
    ::munmap(mm, size);
    return result;
}

}  // namespace imp

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Native SentencePiece (.model protobuf) parser.
//
// The on-disk format is a serialized `sentencepiece.ModelProto` protobuf
// message. imp's tokenizer already implements SentencePiece-style scoring
// (`encode_spm`, see tokenizer.cpp) for vocabularies loaded from GGUF; this
// loader extracts the same vocabulary + scores from the protobuf so the
// SafeTensors path can fall back to a `tokenizer.model` file when no
// `tokenizer.json` is present (older Llama 1/2, some Mistral variants).
//
// Scope: parses Unigram-style ModelProto. BPE-from-spm checkpoints are
// detected (model_type=2) but their vocabulary still loads; the encoder
// will run score-based merging on it which matches Unigram behaviour for
// most practical text.

struct SentencePieceModel {
    enum class ModelType : int32_t { UNIGRAM = 1, BPE = 2, WORD = 3, CHAR = 4, UNKNOWN = 0 };

    // ModelProto.SentencePiece.Type values (mapped 1:1 to imp's tokenizer
    // token-type convention: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4,
    // UNUSED=5, BYTE=6).
    std::vector<std::string> pieces;  // vocab[id] = string
    std::vector<float> scores;        // score[id] = log-prob (Unigram) or merge rank (BPE proxy)
    std::vector<int32_t> types;       // type[id] from spec; empty when not parsed

    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t unk_id = 0;
    int32_t pad_id = -1;

    ModelType model_type = ModelType::UNKNOWN;

    bool empty() const { return pieces.empty(); }
};

// Parse a SentencePiece .model protobuf blob. `data` is the full file
// contents; `size` is its length. Returns true on a structurally valid
// parse with at least one piece. Sets `*err` on failure (when non-null).
//
// The parser is wire-format-tolerant: unknown fields and unexpected wire
// types are skipped without aborting the parse.
bool parse_sentencepiece_model(const void* data, size_t size, SentencePieceModel* out, std::string* err);

// Convenience wrapper that opens `path`, mmaps it, and calls
// parse_sentencepiece_model. Returns nullptr-equivalent (empty) result on
// I/O or parse failure with the error logged.
SentencePieceModel load_sentencepiece_model_file(const std::string& path);

}  // namespace imp

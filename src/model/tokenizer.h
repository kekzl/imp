#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <memory>

namespace imp {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load from standalone tokenizer file (SentencePiece .model or HuggingFace .json)
    bool load(const std::string& path);

    // Load vocabulary extracted from GGUF metadata
    bool load_vocab(const std::vector<std::string>& tokens,
                    const std::vector<float>& scores,
                    int bos_id, int eos_id);

    // Load BPE merge rules (for GPT2-style tokenizers)
    void load_merges(const std::vector<std::string>& merges);

    // Set tokenizer type: "spm" (SentencePiece) or "gpt2" (byte-level BPE)
    void set_type(const std::string& type) { type_ = type; }
    const std::string& type() const { return type_; }

    // Control BOS token prepending
    void set_add_bos(bool add) { add_bos_ = add; }
    bool add_bos() const { return add_bos_; }

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& tokens) const;
    std::string decode_token(int32_t token) const;

    int vocab_size() const;
    int bos_id() const;
    int eos_id() const;

private:
    // UTF-8 helper: returns byte length of character starting at c
    static int utf8_char_len(uint8_t c);

    // SentencePiece-style BPE (score-based merging, LOWER_ONE_EIGHTH_BLOCK space)
    std::vector<int32_t> encode_spm(const std::string& text) const;

    // GPT2-style byte-level BPE (merge-rank based)
    std::vector<int32_t> encode_gpt2(const std::string& text) const;

    // GPT2 decode (reverse byte encoding)
    std::string decode_gpt2(const std::vector<int32_t>& tokens) const;
    std::string decode_gpt2_token(int32_t token) const;

    // SentencePiece decode
    std::string decode_spm(const std::vector<int32_t>& tokens) const;
    std::string decode_spm_token(int32_t token) const;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    int bos_id_ = 1, eos_id_ = 2;

    std::string type_ = "spm";  // "spm" or "gpt2"
    bool add_bos_ = true;

    // GPT2 BPE merge ranks: "token1 token2" -> rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;
};

} // namespace imp

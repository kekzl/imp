#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <memory>

namespace imp {

struct SplitSequence;  // tokenizer_pretok.cpp

// #1606: largest token id a tokenizer.json may declare. Ids arrive as JSON doubles and
// index vocab_/scores_/token_types_ directly, needing both bounds: below zero is an OOB
// write, near INT_MAX the max_id+1 sizing wraps. 4M is ~16x the largest shipped vocabulary
// (Gemma, ~256k).
constexpr int64_t kMaxTokenId = int64_t{4} * 1024 * 1024;

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load from standalone tokenizer file (SentencePiece .model or HuggingFace .json)
    [[nodiscard]] bool load(const std::string& path);

    // Load vocabulary extracted from GGUF metadata
    [[nodiscard]] bool load_vocab(const std::vector<std::string>& tokens, const std::vector<float>& scores, int bos_id,
                    int eos_id);

    // Load BPE merge rules (for GPT2-style tokenizers)
    void load_merges(const std::vector<std::string>& merges, bool build_pairs = true);

    // Set tokenizer type: "spm" (SentencePiece) or "gpt2" (byte-level BPE)
    void set_type(const std::string& type) { type_ = type; }
    const std::string& type() const { return type_; }
    [[nodiscard]] bool nfc() const { return nfc_; }
    void set_nfc(bool v) { nfc_ = v; }

    // Control BOS token prepending
    void set_add_bos(bool add) { add_bos_ = add; }
    [[nodiscard]] bool add_bos() const { return add_bos_; }

    // Override the BOS token ID (default 1). Called by model loaders when the
    // model's BOS token is not in the hardcoded detection list (e.g. DeepSeek's
    // "<｜begin▁of▁sentence｜>" which maps to ID 100000).
    void set_bos_id(int id) { bos_id_ = id; }

    // Control SentencePiece leading-space prefix (▁)
    void set_add_space_prefix(bool add) { add_space_prefix_ = add; }
    [[nodiscard]] bool add_space_prefix() const { return add_space_prefix_; }

    // Pre-tokenizer type from GGUF metadata (e.g. "default", "llama3", "deepseek-llm").
    // "deepseek-r1-qwen": tokenizer.json regex and NFC equal Qwen2's, stored as "qwen2" (#2270).
    void set_pre_tokenizer(const std::string& pre) {
        pre_tokenizer_ = pre == "deepseek-r1-qwen" ? "qwen2" : pre;
    }
    const std::string& pre_tokenizer() const { return pre_tokenizer_; }

    // Chat template string from GGUF metadata (Jinja2 format, used for detection)
    void set_chat_template_str(const std::string& tpl) { chat_template_str_ = tpl; }
    const std::string& chat_template_str() const { return chat_template_str_; }

    // Author-shipped tokenizer_config.json::use_default_system_prompt. When false, the
    // chat-template apply path must inject an empty system message so the template's
    // no-system-> default-message branch doesn't fire (Mistral-Small-3.2 otherwise injects a
    // 600-token default).
    void set_use_default_system_prompt(bool v) { use_default_system_prompt_ = v; }
    bool use_default_system_prompt() const { return use_default_system_prompt_; }

    // Encode text to token IDs
    // no_prefix=true skips SPM space prefix (for chat template pieces after special tokens)
    std::vector<int32_t> encode(const std::string& text, bool no_prefix = false) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& tokens) const;
    std::string decode_token(int32_t token) const;

    int vocab_size() const;
    int bos_id() const;
    int eos_id() const { return eos_ids_.empty() ? 2 : eos_ids_[0]; }
    const std::vector<int32_t>& eos_ids() const { return eos_ids_; }
    void add_eos_id(int32_t id) {
        if (eos_is_default_) {
            eos_ids_ = {id};
            eos_is_default_ = false;
            return;
        }
        for (int32_t eid : eos_ids_)
            if (eid == id)
                return;
        eos_ids_.push_back(id);
    }
    bool is_eos(int32_t id) const {
        for (int32_t eid : eos_ids_)
            if (eid == id)
                return true;
        return false;
    }

    // Raw token text from vocabulary (for special token scanning)
    const std::string& token_text(int id) const {
        static const std::string empty;
        return (id >= 0 && id < static_cast<int>(vocab_.size())) ? vocab_[id] : empty;
    }

    // Look up a token string in the vocabulary, returns -1 if not found
    int32_t find_token(const std::string& text) const;

    // Token type metadata from GGUF (tokenizer.ggml.token_type).
    // Types: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
    void load_token_types(const std::vector<int32_t>& types) {
        token_types_ = types;
        build_special_pieces();
    }
    bool has_token_types() const { return !token_types_.empty(); }
    // tokenizer.json added-token whitespace stripping: bit 0 lstrip, bit 1 rstrip.
    uint8_t strip_flags(int32_t id) const {
        return id >= 0 && static_cast<size_t>(id) < strip_flags_.size() ? strip_flags_[id] : 0;
    }
    bool is_control_token(int id) const {
        return id >= 0 && id < static_cast<int>(token_types_.size()) && token_types_[id] == 3;
    }
    bool is_special_token(int id) const {
        return id >= 0 && id < static_cast<int>(token_types_.size()) && token_types_[id] != 1;
    }

    // True if `id` was declared in tokenizer.json's added_tokens array, regardless of its
    // `special` flag. Distinguishes a deliberately-added marker (Qwen3's </think>, added,
    // special=false) from a normal BPE piece that happens to spell the same text (Nemotron ID
    // 12, not added). Empty vector => always false.
    bool is_added_token(int id) const {
        return id >= 0 && id < static_cast<int>(added_token_ids_.size()) && added_token_ids_[id];
    }

    // FIM token ids declared in GGUF metadata (tokenizer.ggml.fim_*_token_id), -1 = not declared.
    // Index by FimRole (model/fim.h); fim.cpp falls back to a vocab text lookup.
    void set_fim_meta_id(int role, int32_t id) {
        if (role >= 0 && role < kFimRoles)
            fim_meta_ids_[role] = id;
    }
    int32_t fim_meta_id(int role) const { return role >= 0 && role < kFimRoles ? fim_meta_ids_[role] : -1; }
    static constexpr int kFimRoles = 6;

    // Defensive overlay: marks a token CONTROL even when the source tokenizer didn't tag it.
    // Used to cross-check special_tokens_map.json against tokenizer.json's special-flag column.
    // No-op on an invalid id; allocates the type vector lazily if empty.
    void mark_as_control(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(vocab_.size()))
            return;
        if (token_types_.empty()) {
            token_types_.assign(vocab_.size(), 1);  // default NORMAL=1
        }
        token_types_[id] = 3;  // CONTROL
        build_special_pieces();
    }

private:
    // UTF-8 helper: returns byte length of character starting at c
    static int utf8_char_len(uint8_t c);

    // SentencePiece-style BPE (score-based merging, LOWER_ONE_EIGHTH_BLOCK space)
    std::vector<int32_t> encode_spm(const std::string& text, bool no_prefix = false) const;

    // Gemma-4 SPM-style BPE (▁ escaping + merge ranks, raw UTF-8)
    std::vector<int32_t> encode_gemma4(const std::string& text) const;

    // GPT2-style byte-level BPE (merge-rank based)
    std::vector<int32_t> encode_gpt2(const std::string& text) const;

    // BERT WordPiece (#836, nomic-bert embedder): lowercase basic tokenizer +
    // greedy longest-match with "##" continuation pieces, [UNK] fallback.
    std::vector<int32_t> encode_wordpiece(const std::string& text) const;

    // GPT2 decode (reverse byte encoding)
    std::string decode_gpt2(const std::vector<int32_t>& tokens) const;
    std::string decode_gpt2_token(int32_t token) const;

    // SentencePiece decode
    std::string decode_spm(const std::vector<int32_t>& tokens) const;
    std::string decode_spm_token(int32_t token) const;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    int bos_id_ = 1;
    std::vector<int32_t> eos_ids_ = {2};
    // {2} is a placeholder until a source names the EOS; the first add_eos_id replaces it
    // (DeepSeek-V2 id 2 is "#", #2377).
    bool eos_is_default_ = true;

    std::string type_ = "spm";   // "spm" or "gpt2"
    std::string pre_tokenizer_;  // Pre-tokenizer type from GGUF tokenizer.ggml.pre
    std::shared_ptr<const SplitSequence> split_seq_;  // pre_tokenizer_ == "split-seq"
    bool add_bos_ = true;
    bool add_space_prefix_ = true;           // SentencePiece ▁ prefix (false for Gemma)
    bool use_default_system_prompt_ = true;  // false → skip template's hardcoded default system
    std::string chat_template_str_;          // Raw Jinja2 template from GGUF
    // encode() applies NFC. tokenizer.json sets it from `normalizer` (HF: null / Replace do not
    // compose, so gpt-oss / Nemotron / Phi-4 / Gemma-4 see NFD input as-is); GGUF: Qwen pre only.
    bool nfc_ = true;

    // GPT2 BPE merge ranks: "token1 token2" -> rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;
    // Id form of merge_ranks_ for encode_gpt2: (left id << 32 | right id) -> {rank, merged id}.
    // Valid only when every merge of two vocab tokens yields a vocab token. byte_ids_ -1: byte has
    // no token, a chunk containing it takes the string path.
    std::unordered_map<uint64_t, std::pair<int, int32_t>> merge_pairs_;
    int32_t byte_ids_[256] = {};
    bool merge_pairs_ok_ = false;
    void build_merge_pairs();
    void bpe_gpt2_ids(const std::string& chunk, std::vector<int32_t>& out) const;

    // Per-token type from GGUF (NORMAL=1, CONTROL=3, etc.). Empty if not available.
    std::vector<int32_t> token_types_;

    // Membership flag per id for tokenizer.json `added_tokens` (HF SafeTensors).
    // Empty when the source carries no added_tokens array.
    std::vector<bool> added_token_ids_;

    // Per-id `lstrip` (bit 0) / `rstrip` (bit 1) of tokenizer.json added_tokens: the match
    // swallows adjacent whitespace (Phi-4 <|im_end|>). Empty when none sets them.
    std::vector<uint8_t> strip_flags_;
    int32_t fim_meta_ids_[kFimRoles] = {-1, -1, -1, -1, -1, -1};

    // Cached special-token strings (CONTROL type) sorted by length descending, so encode_*
    // pre-splits input on these literals and multi-character markers (e.g. <|tool_call>)
    // round-trip as their assigned single-token id instead of being BPE'd as raw bytes.
    std::vector<std::pair<std::string, int32_t>> special_pieces_;
    void build_special_pieces();
};

// Regex pre-tokenizer scans (tokenizer_pretok.cpp; #657), one per tokenizer.json regex family.
// qwen2: Qwen2/Qwen3; qwen35: letter runs include \p{M}; cl100k: digit triples (Phi-4, Llama 3);
// o200k: case-aware letter runs (gpt-oss); nemotron: o200k without contractions, single digits.
std::vector<std::string> qwen2_pre_tokenize(const std::string& text);
std::vector<std::string> qwen35_pre_tokenize(const std::string& text);
std::vector<std::string> cl100k_pre_tokenize(const std::string& text);
std::vector<std::string> o200k_pre_tokenize(const std::string& text);
std::vector<std::string> nemotron_pre_tokenize(const std::string& text);

// tokenizer.json pre-tokenizer given as a Sequence of Split(Isolated) regexes plus Digits
// (DeepSeek-V2 / Coder-V2). Steps: "re:<regex>" or "digits:<individual 0|1>". Accepts only the
// regex forms it can match exactly ([\s?]<class>[+], \s+$); nullptr otherwise.
std::shared_ptr<const SplitSequence> compile_split_sequence(const std::vector<std::string>& steps);
std::vector<std::string> split_sequence_pre_tokenize(const SplitSequence& seq, const std::string& text);

// NFC as encode() applies it: table compositions of Latin/Greek/Cyrillic base + combining mark,
// and algorithmic Hangul L+V(+T). Exposed for unit tests.
std::string nfc_normalize(const std::string& text);

}  // namespace imp

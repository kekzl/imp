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
    bool load_vocab(const std::vector<std::string>& tokens, const std::vector<float>& scores, int bos_id,
                    int eos_id);

    // Load BPE merge rules (for GPT2-style tokenizers)
    void load_merges(const std::vector<std::string>& merges);

    // Set tokenizer type: "spm" (SentencePiece) or "gpt2" (byte-level BPE)
    void set_type(const std::string& type) { type_ = type; }
    const std::string& type() const { return type_; }

    // Control BOS token prepending
    void set_add_bos(bool add) { add_bos_ = add; }
    bool add_bos() const { return add_bos_; }

    // Control SentencePiece leading-space prefix (▁)
    void set_add_space_prefix(bool add) { add_space_prefix_ = add; }
    bool add_space_prefix() const { return add_space_prefix_; }

    // Pre-tokenizer type from GGUF metadata (e.g. "default", "llama3", "deepseek-llm")
    void set_pre_tokenizer(const std::string& pre) { pre_tokenizer_ = pre; }
    const std::string& pre_tokenizer() const { return pre_tokenizer_; }

    // Chat template string from GGUF metadata (Jinja2 format, used for detection)
    void set_chat_template_str(const std::string& tpl) { chat_template_str_ = tpl; }
    const std::string& chat_template_str() const { return chat_template_str_; }

    // Author-shipped flag from tokenizer_config.json::use_default_system_prompt.
    // When false, the chat-template apply path must inject an empty system
    // message so the template's "no-system → default_system_message" branch
    // doesn't fire (e.g. Mistral-Small-3.2 ships this flag false but its
    // chat_template.jinja line 158 still injects a 600-token default unless
    // an explicit system message is present).
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
    bool is_control_token(int id) const {
        return id >= 0 && id < static_cast<int>(token_types_.size()) && token_types_[id] == 3;
    }
    bool is_special_token(int id) const {
        return id >= 0 && id < static_cast<int>(token_types_.size()) && token_types_[id] != 1;
    }

    // Defensive overlay: mark a token as CONTROL even when it wasn't tagged
    // by the source tokenizer. Used to cross-check special_tokens_map.json
    // against tokenizer.json's special-flag column for HF model directories.
    // No-op when id is invalid; allocates the type vector lazily if empty.
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

    std::string type_ = "spm";   // "spm" or "gpt2"
    std::string pre_tokenizer_;  // Pre-tokenizer type from GGUF tokenizer.ggml.pre
    bool add_bos_ = true;
    bool add_space_prefix_ = true;           // SentencePiece ▁ prefix (false for Gemma)
    bool use_default_system_prompt_ = true;  // false → skip template's hardcoded default system
    std::string chat_template_str_;          // Raw Jinja2 template from GGUF

    // GPT2 BPE merge ranks: "token1 token2" -> rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;

    // Per-token type from GGUF (NORMAL=1, CONTROL=3, etc.). Empty if not available.
    std::vector<int32_t> token_types_;

    // Cached list of special-token strings (CONTROL type) sorted by length
    // descending. Used by encode_*() to pre-split input on these literals so
    // multi-character markers like `<|tool_call>` round-trip as their assigned
    // single-token id instead of being BPE'd as raw bytes.
    std::vector<std::pair<std::string, int32_t>> special_pieces_;
    void build_special_pieces();
};

}  // namespace imp

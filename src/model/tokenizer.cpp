#include "model/tokenizer.h"
#include "core/logging.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <climits>
#include <cstring>
#include <queue>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace imp {

// ---- Minimal JSON parser (local copy — handles \uXXXX for tokenizer.json) ----

namespace {

enum class JType { NUL, STRING, NUMBER, ARRAY, OBJECT };

struct JValue {
    JType type = JType::NUL;
    std::string str_val;
    double num_val = 0.0;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;
};

class JsonParser {
public:
    explicit JsonParser(const char* data, size_t len)
        : data_(data), len_(len), pos_(0) {}

    JValue parse() {
        skip_ws();
        return parse_value();
    }

    bool ok() const { return !error_; }

private:
    const char* data_;
    size_t len_;
    size_t pos_;
    bool error_ = false;

    char peek() const {
        if (pos_ >= len_) return '\0';
        return data_[pos_];
    }

    char advance() {
        if (pos_ >= len_) { error_ = true; return '\0'; }
        return data_[pos_++];
    }

    void skip_ws() {
        while (pos_ < len_ && (data_[pos_] == ' ' || data_[pos_] == '\t' ||
                                data_[pos_] == '\n' || data_[pos_] == '\r')) {
            pos_++;
        }
    }

    bool expect(char c) {
        skip_ws();
        if (peek() == c) { advance(); return true; }
        error_ = true;
        return false;
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + c - 'a';
        if (c >= 'A' && c <= 'F') return 10 + c - 'A';
        return -1;
    }

    uint32_t parse_u4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            if (pos_ >= len_) { error_ = true; return 0; }
            int d = hex_digit(data_[pos_++]);
            if (d < 0) { error_ = true; return 0; }
            v = (v << 4) | d;
        }
        return v;
    }

    static void append_codepoint_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x110000) {
            s += static_cast<char>(0xF0 | (cp >> 18));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    JValue parse_value() {
        skip_ws();
        if (error_) return {};
        char c = peek();
        if (c == '"') return parse_string_value();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        error_ = true;
        return {};
    }

    JValue parse_string_value() {
        JValue v;
        v.type = JType::STRING;
        v.str_val = parse_string_raw();
        return v;
    }

    std::string parse_string_raw() {
        if (!expect('"')) return "";
        std::string s;
        while (pos_ < len_) {
            char c = advance();
            if (c == '"') return s;
            if (c == '\\') {
                if (pos_ >= len_) { error_ = true; return s; }
                char esc = advance();
                switch (esc) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        uint32_t cp = parse_u4();
                        // Handle UTF-16 surrogate pairs
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (pos_ + 1 < len_ && data_[pos_] == '\\' && data_[pos_+1] == 'u') {
                                pos_ += 2;
                                uint32_t lo = parse_u4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                }
                            }
                        }
                        append_codepoint_utf8(s, cp);
                        break;
                    }
                    default: s += esc; break;
                }
            } else {
                s += c;
            }
        }
        error_ = true;
        return s;
    }

    JValue parse_number() {
        JValue v;
        v.type = JType::NUMBER;
        size_t start = pos_;
        if (peek() == '-') advance();
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        if (pos_ < len_ && data_[pos_] == '.') {
            advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        }
        if (pos_ < len_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
            advance();
            if (pos_ < len_ && (data_[pos_] == '+' || data_[pos_] == '-')) advance();
            while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') advance();
        }
        std::string num_str(data_ + start, pos_ - start);
        v.num_val = std::stod(num_str);
        return v;
    }

    JValue parse_object() {
        JValue v;
        v.type = JType::OBJECT;
        if (!expect('{')) return v;
        skip_ws();
        if (peek() == '}') { advance(); return v; }
        while (!error_) {
            skip_ws();
            std::string key = parse_string_raw();
            if (!expect(':')) break;
            JValue val = parse_value();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect('}');
        return v;
    }

    JValue parse_array() {
        JValue v;
        v.type = JType::ARRAY;
        if (!expect('[')) return v;
        skip_ws();
        if (peek() == ']') { advance(); return v; }
        while (!error_) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        expect(']');
        return v;
    }

    JValue parse_bool() {
        JValue v;
        v.type = JType::NUMBER;
        if (peek() == 't') {
            for (int i = 0; i < 4 && pos_ < len_; i++) advance();
            v.num_val = 1.0;
        } else {
            for (int i = 0; i < 5 && pos_ < len_; i++) advance();
            v.num_val = 0.0;
        }
        return v;
    }

    JValue parse_null() {
        JValue v;
        v.type = JType::NUL;
        for (int i = 0; i < 4 && pos_ < len_; i++) advance();
        return v;
    }
};

const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

bool jobj_get_string(const JValue& obj, const std::string& key, std::string& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::STRING) return false;
    out = v->str_val;
    return true;
}

} // anonymous namespace

// ---- UTF-8 helpers ----

int Tokenizer::utf8_char_len(uint8_t c) {
    if ((c & 0x80) == 0)    return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // invalid byte, treat as single
}

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ---- GPT2 byte-level encoding tables ----
//
// GPT2 maps each byte (0-255) to a unique Unicode codepoint:
// - Printable ASCII (33-126): identity mapping
// - Latin-1 supplement (161-172, 174-255): identity mapping
// - All other bytes (0-32, 127-160, 173): mapped to 256+ range
//
// This ensures every byte has a visible Unicode representation.

static const uint32_t BYTE_TO_CODEPOINT[256] = {
    // 0-32: mapped to 256-288
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
    266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
    276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
    286, 287, 288,
    // 33-126: identity (! to ~)
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126,
    // 127-160: mapped to 289-322
    289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
    299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
    309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
    319, 320, 321, 322,
    // 161-172: identity
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
    // 173: mapped to 323
    323,
    // 174-255: identity
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
    234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
    254, 255,
};

// Reverse mapping: codepoint -> byte value (built once)
static uint8_t CODEPOINT_TO_BYTE[324];
static bool CODEPOINT_TABLE_INIT = false;

static void init_codepoint_table() {
    if (CODEPOINT_TABLE_INIT) return;
    for (int b = 0; b < 256; b++) {
        CODEPOINT_TO_BYTE[BYTE_TO_CODEPOINT[b]] = static_cast<uint8_t>(b);
    }
    CODEPOINT_TABLE_INIT = true;
}

// Convert a single byte to its GPT2 Unicode character (UTF-8 encoded)
static std::string byte_to_gpt2(uint8_t byte) {
    return codepoint_to_utf8(BYTE_TO_CODEPOINT[byte]);
}

// Convert a UTF-8 character (from GPT2 encoding) back to the original byte
// Returns -1 if not a valid GPT2 byte-encoded character
static int gpt2_to_byte(const char* s, int len) {
    init_codepoint_table();
    uint32_t cp = 0;
    if (len == 1) {
        cp = static_cast<uint8_t>(s[0]);
    } else if (len == 2) {
        cp = ((static_cast<uint32_t>(s[0]) & 0x1F) << 6) |
              (static_cast<uint32_t>(s[1]) & 0x3F);
    } else if (len == 3) {
        cp = ((static_cast<uint32_t>(s[0]) & 0x0F) << 12) |
             ((static_cast<uint32_t>(s[1]) & 0x3F) << 6) |
              (static_cast<uint32_t>(s[2]) & 0x3F);
    } else {
        return -1;
    }
    if (cp < 324) {
        return CODEPOINT_TO_BYTE[cp];
    }
    return -1;
}

// ---- GPT2 pre-tokenization ----
//
// Splits input text into chunks before applying BPE to each independently.
// This is a simplified version of the cl100k_base / Qwen2 pre-tokenizer.
// Key rules:
// - Spaces attach to the following word
// - Letter sequences form chunks
// - Digit sequences (up to 3) form chunks
// - Individual punctuation chars form chunks
// - Newlines group together

static std::vector<std::string> gpt2_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty()) return result;

    size_t i = 0;
    while (i < text.size()) {
        std::string chunk;

        // Collect leading spaces/tabs (attach to next non-whitespace chunk)
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) {
            chunk += text[i++];
        }

        if (i >= text.size()) {
            if (!chunk.empty()) result.push_back(chunk);
            break;
        }

        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == '\n' || c == '\r') {
            // Newlines: collect consecutive newlines
            while (i < text.size() && (text[i] == '\n' || text[i] == '\r')) {
                chunk += text[i++];
            }
        } else if (std::isalpha(c) || c >= 128) {
            // Letters (ASCII + multi-byte UTF-8 treated as letters)
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (std::isalpha(cc) || cc >= 128) {
                    int len = 1;
                    if ((cc & 0xE0) == 0xC0) len = 2;
                    else if ((cc & 0xF0) == 0xE0) len = 3;
                    else if ((cc & 0xF8) == 0xF0) len = 4;
                    for (int j = 0; j < len && i < text.size(); j++)
                        chunk += text[i++];
                } else {
                    break;
                }
            }
        } else if (std::isdigit(c)) {
            // Digits: groups of up to 3
            int count = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) && count < 3) {
                chunk += text[i++];
                count++;
            }
        } else {
            // Punctuation/other: single character
            chunk += text[i++];
        }

        if (!chunk.empty()) {
            result.push_back(chunk);
        }
    }

    return result;
}

// ---- Load vocabulary ----

bool Tokenizer::load(const std::string& path) {
    // Read file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    std::string file_data(st.st_size, '\0');
    ssize_t n = ::read(fd, file_data.data(), st.st_size);
    close(fd);
    if (n != st.st_size) return false;

    // Parse JSON
    JsonParser parser(file_data.c_str(), file_data.size());
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_WARN("failed to parse tokenizer.json: %s", path.c_str());
        return false;
    }

    // Extract model object
    const JValue* model = jobj_find(root, "model");
    if (!model || model->type != JType::OBJECT) {
        IMP_LOG_WARN("tokenizer.json missing 'model' object");
        return false;
    }

    // Model type
    std::string model_type;
    jobj_get_string(*model, "type", model_type);

    // Extract vocabulary from model.vocab
    const JValue* vocab = jobj_find(*model, "vocab");
    if (vocab && vocab->type == JType::OBJECT) {
        // Find max id to size the vocab vector
        int max_id = 0;
        for (const auto& [token, val] : vocab->obj) {
            if (val.type == JType::NUMBER) {
                int id = static_cast<int>(val.num_val);
                if (id > max_id) max_id = id;
            }
        }
        vocab_.resize(max_id + 1);
        scores_.resize(max_id + 1, 0.0f);

        token_to_id_.clear();
        token_to_id_.reserve(vocab->obj.size());
        for (const auto& [token, val] : vocab->obj) {
            if (val.type != JType::NUMBER) continue;
            int id = static_cast<int>(val.num_val);
            vocab_[id] = token;
            token_to_id_[token] = id;
        }

        IMP_LOG_INFO("tokenizer.json: loaded %zu vocab entries (type=%s)",
                     vocab->obj.size(), model_type.c_str());
    }

    // Extract merges (BPE only)
    const JValue* merges = jobj_find(*model, "merges");
    if (merges && merges->type == JType::ARRAY) {
        std::vector<std::string> merge_strs;
        merge_strs.reserve(merges->arr.size());
        for (const auto& m : merges->arr) {
            if (m.type == JType::STRING) merge_strs.push_back(m.str_val);
        }
        load_merges(merge_strs);
        IMP_LOG_INFO("tokenizer.json: loaded %zu merges", merge_strs.size());
    }

    // Extract added_tokens — may extend vocab and mark special tokens
    const JValue* added = jobj_find(root, "added_tokens");
    if (added && added->type == JType::ARRAY) {
        token_types_.resize(vocab_.size(), 1);  // default NORMAL=1

        for (const auto& tok : added->arr) {
            if (tok.type != JType::OBJECT) continue;
            const JValue* id_v = jobj_find(tok, "id");
            const JValue* content_v = jobj_find(tok, "content");
            const JValue* special_v = jobj_find(tok, "special");

            if (!id_v || !content_v) continue;
            if (id_v->type != JType::NUMBER || content_v->type != JType::STRING) continue;
            int id = static_cast<int>(id_v->num_val);
            const std::string& content = content_v->str_val;
            bool is_special = special_v && special_v->type == JType::NUMBER &&
                              special_v->num_val != 0.0;

            // Ensure vectors are large enough
            if (id >= static_cast<int>(vocab_.size())) {
                vocab_.resize(id + 1);
                scores_.resize(id + 1, 0.0f);
                token_types_.resize(id + 1, 1);
            }

            vocab_[id] = content;
            token_to_id_[content] = id;
            if (is_special) token_types_[id] = 3;  // CONTROL

            // Detect BOS/EOS tokens
            if (content == "<s>" || content == "<|begin_of_text|>" ||
                content == "<|startoftext|>") {
                bos_id_ = id;
            }
            if (content == "</s>" || content == "<|end_of_text|>" ||
                content == "<|endoftext|>" || content == "<|eot_id|>") {
                if (eos_ids_.size() == 1 && eos_ids_[0] == 2) {
                    // Replace default
                    eos_ids_ = {static_cast<int32_t>(id)};
                } else {
                    add_eos_id(static_cast<int32_t>(id));
                }
            }
        }
    }

    // Detect pre-tokenizer type
    const JValue* pre_tok = jobj_find(root, "pre_tokenizer");
    if (pre_tok && pre_tok->type == JType::OBJECT) {
        std::string pt_type;
        jobj_get_string(*pre_tok, "type", pt_type);

        if (pt_type == "ByteLevel") {
            type_ = "gpt2";
            const JValue* prefix = jobj_find(*pre_tok, "add_prefix_space");
            if (prefix && prefix->type == JType::NUMBER)
                add_space_prefix_ = (prefix->num_val != 0.0);
            else
                add_space_prefix_ = false;
        } else if (pt_type == "Metaspace") {
            type_ = "spm";
            const JValue* prefix = jobj_find(*pre_tok, "add_prefix_space");
            if (prefix && prefix->type == JType::NUMBER)
                add_space_prefix_ = (prefix->num_val != 0.0);
        } else if (pt_type == "Sequence") {
            // Check inner pre-tokenizers for ByteLevel or Metaspace
            const JValue* pretoks = jobj_find(*pre_tok, "pretokenizers");
            if (pretoks && pretoks->type == JType::ARRAY) {
                for (const auto& pt : pretoks->arr) {
                    if (pt.type != JType::OBJECT) continue;
                    std::string inner_type;
                    jobj_get_string(pt, "type", inner_type);
                    if (inner_type == "ByteLevel") {
                        type_ = "gpt2";
                        const JValue* prefix = jobj_find(pt, "add_prefix_space");
                        if (prefix && prefix->type == JType::NUMBER)
                            add_space_prefix_ = (prefix->num_val != 0.0);
                        else
                            add_space_prefix_ = false;
                        break;
                    }
                    if (inner_type == "Metaspace") {
                        type_ = "spm";
                        const JValue* prefix = jobj_find(pt, "add_prefix_space");
                        if (prefix && prefix->type == JType::NUMBER)
                            add_space_prefix_ = (prefix->num_val != 0.0);
                        break;
                    }
                }
            }
        }
    } else if (model_type == "BPE") {
        // No pre_tokenizer specified but model is BPE — default to gpt2
        type_ = "gpt2";
        add_space_prefix_ = false;
    } else if (model_type == "Unigram") {
        type_ = "spm";
    }

    // For Unigram models, populate scores from model.vocab (array of [token, score])
    // Some Unigram tokenizer.json have vocab as array instead of object
    if (model_type == "Unigram") {
        const JValue* uni_vocab = jobj_find(*model, "vocab");
        if (uni_vocab && uni_vocab->type == JType::ARRAY) {
            int max_id = static_cast<int>(uni_vocab->arr.size()) - 1;
            vocab_.resize(max_id + 1);
            scores_.resize(max_id + 1, 0.0f);
            token_to_id_.clear();
            token_to_id_.reserve(uni_vocab->arr.size());
            for (size_t i = 0; i < uni_vocab->arr.size(); i++) {
                const auto& entry = uni_vocab->arr[i];
                if (entry.type == JType::ARRAY && entry.arr.size() >= 2) {
                    vocab_[i] = entry.arr[0].str_val;
                    scores_[i] = static_cast<float>(entry.arr[1].num_val);
                    token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
                }
            }
            IMP_LOG_INFO("tokenizer.json: loaded %zu Unigram vocab entries",
                         uni_vocab->arr.size());
        }
    }

    IMP_LOG_INFO("tokenizer.json: type=%s vocab_size=%d bos=%d eos=%d add_prefix=%s",
                 type_.c_str(), static_cast<int>(vocab_.size()), bos_id_,
                 eos_ids_.empty() ? -1 : eos_ids_[0],
                 add_space_prefix_ ? "true" : "false");
    return true;
}

bool Tokenizer::load_vocab(const std::vector<std::string>& tokens,
                           const std::vector<float>& scores,
                           int bos_id, int eos_id) {
    if (tokens.empty()) return false;

    vocab_ = tokens;
    scores_ = scores;
    scores_.resize(vocab_.size(), 0.0f);
    bos_id_ = bos_id;
    eos_ids_ = {eos_id};

    token_to_id_.clear();
    token_to_id_.reserve(vocab_.size());
    for (size_t i = 0; i < vocab_.size(); i++) {
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }

    return true;
}

void Tokenizer::load_merges(const std::vector<std::string>& merges) {
    merge_ranks_.clear();
    merge_ranks_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); i++) {
        merge_ranks_[merges[i]] = static_cast<int>(i);
    }
}

// ---- BPE Encode (SentencePiece style) ----

static const std::string SPIECE_SPACE = "\xe2\x96\x81";

std::vector<int32_t> Tokenizer::encode_spm(const std::string& text, bool no_prefix) const {
    if (text.empty() || vocab_.empty()) return {};

    // Pre-process: SentencePiece convention - replace spaces with ▁
    // add_space_prefix_: prepend ▁ at start (true for LLaMA/Mistral, false for Gemma)
    // no_prefix: skip the leading ▁ (for chat template pieces after special tokens)
    std::string processed;
    processed.reserve(text.size() + 4);
    if (add_space_prefix_ && !no_prefix) {
        processed += SPIECE_SPACE;
    }

    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            processed += SPIECE_SPACE;
        } else {
            processed += text[i];
        }
    }

    // Split into UTF-8 characters as initial symbols
    std::vector<std::string> symbols;
    symbols.reserve(processed.size());

    for (size_t i = 0; i < processed.size(); ) {
        int len = utf8_char_len(static_cast<uint8_t>(processed[i]));
        if (i + len > processed.size()) len = 1;
        symbols.push_back(processed.substr(i, len));
        i += len;
    }

    // BPE merge loop using priority queue: O(n log n) instead of O(n²).
    // Linked list of symbols with prev/next pointers; deleted nodes are skipped.
    int n = static_cast<int>(symbols.size());
    std::vector<int> prev(n), next(n);
    std::vector<bool> deleted(n, false);
    for (int i = 0; i < n; i++) {
        prev[i] = i - 1;
        next[i] = i + 1;
    }

    // Max-heap: highest score first, then lowest position for tie-breaking
    struct MergeCand {
        float score;
        int pos;       // left symbol index
        int seq;       // left sequence number at insertion (for invalidation)
        int rseq;      // right sequence number at insertion
    };
    auto cmp = [](const MergeCand& a, const MergeCand& b) {
        if (a.score != b.score) return a.score < b.score;
        return a.pos > b.pos;
    };
    std::priority_queue<MergeCand, std::vector<MergeCand>, decltype(cmp)> pq(cmp);

    // Sequence counters per position: incremented on merge to invalidate stale entries
    std::vector<int> seq(n, 0);

    // Seed the queue with all valid adjacent pairs
    for (int i = 0; i < n - 1; i++) {
        std::string merged = symbols[i] + symbols[next[i]];
        auto it = token_to_id_.find(merged);
        if (it != token_to_id_.end()) {
            pq.push({scores_[it->second], i, seq[i], seq[next[i]]});
        }
    }

    while (!pq.empty()) {
        auto [score, pos, s, rs] = pq.top();
        pq.pop();

        // Validate: both symbols still exist and haven't been modified since insertion
        if (deleted[pos] || seq[pos] != s) continue;
        int right = next[pos];
        if (right >= n || deleted[right]) continue;
        if (seq[right] != rs) continue;  // right symbol was modified

        // Merge: symbols[pos] absorbs symbols[right]
        symbols[pos] = symbols[pos] + symbols[right];
        deleted[right] = true;
        seq[pos]++;  // invalidate stale entries for this position

        // Update linked list
        next[pos] = next[right];
        if (next[right] < n) prev[next[right]] = pos;

        // Try new pair with left neighbor
        if (prev[pos] >= 0) {
            int lp = prev[pos];
            std::string m = symbols[lp] + symbols[pos];
            auto it = token_to_id_.find(m);
            if (it != token_to_id_.end()) {
                pq.push({scores_[it->second], lp, seq[lp], seq[pos]});
            }
        }
        // Try new pair with right neighbor
        if (next[pos] < n) {
            std::string m = symbols[pos] + symbols[next[pos]];
            auto it = token_to_id_.find(m);
            if (it != token_to_id_.end()) {
                pq.push({scores_[it->second], pos, seq[pos], seq[next[pos]]});
            }
        }
    }

    // Collect non-deleted symbols → token IDs
    std::vector<int32_t> ids;
    ids.reserve(n);

    for (int i = 0; i < n; i++) {
        if (deleted[i]) continue;
        const auto& sym = symbols[i];
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback
            for (unsigned char byte : sym) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                auto byte_it = token_to_id_.find(buf);
                if (byte_it != token_to_id_.end()) {
                    ids.push_back(byte_it->second);
                }
            }
        }
    }

    return ids;
}

// ---- BPE Encode (GPT2 byte-level style) ----

// ---- Llama3 pre-tokenizer ----
// Key differences from default:
//  - Contractions like 's, 't, 're etc. split separately
//  - Spaces are individual tokens (not attached to next word)
//  - Digits are split individually (not groups of 3)

static std::vector<std::string> llama3_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty()) return result;

    // Common English contractions that get their own tokens
    static const char* contractions[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
        "\xe2\x80\x99s", "\xe2\x80\x99t", "\xe2\x80\x99re",
        "\xe2\x80\x99ve", "\xe2\x80\x99m", "\xe2\x80\x99ll", "\xe2\x80\x99d",
    };

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Check for contractions
        bool found_contraction = false;
        if (c == '\'' || (c == 0xe2 && i + 2 < text.size() &&
            text[i+1] == '\x80' && text[i+2] == '\x99')) {
            for (const char* ctr : contractions) {
                size_t len = std::strlen(ctr);
                if (i + len <= text.size() && text.compare(i, len, ctr) == 0) {
                    result.push_back(text.substr(i, len));
                    i += len;
                    found_contraction = true;
                    break;
                }
            }
        }
        if (found_contraction) continue;

        if (c == ' ' || c == '\t') {
            // Space: attach to following word (like GPT2)
            std::string chunk;
            chunk += text[i++];
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r') break;
                if (std::ispunct(cc) && cc != '\'') break;
                int len = 1;
                if ((cc & 0xE0) == 0xC0) len = 2;
                else if ((cc & 0xF0) == 0xE0) len = 3;
                else if ((cc & 0xF8) == 0xF0) len = 4;
                for (int j = 0; j < len && i < text.size(); j++)
                    chunk += text[i++];
            }
            result.push_back(std::move(chunk));
        } else if (c == '\n' || c == '\r') {
            std::string chunk;
            while (i < text.size() && (text[i] == '\n' || text[i] == '\r'))
                chunk += text[i++];
            result.push_back(std::move(chunk));
        } else if (std::isalpha(c) || c >= 128) {
            std::string chunk;
            while (i < text.size()) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (!std::isalpha(cc) && cc < 128) break;
                int len = 1;
                if ((cc & 0xE0) == 0xC0) len = 2;
                else if ((cc & 0xF0) == 0xE0) len = 3;
                else if ((cc & 0xF8) == 0xF0) len = 4;
                for (int j = 0; j < len && i < text.size(); j++)
                    chunk += text[i++];
            }
            result.push_back(std::move(chunk));
        } else if (std::isdigit(c)) {
            // Digits: one at a time (llama3 splits individual digits)
            result.push_back(std::string(1, text[i++]));
        } else {
            result.push_back(std::string(1, text[i++]));
        }
    }
    return result;
}

std::vector<int32_t> Tokenizer::encode_gpt2(const std::string& text) const {
    if (text.empty() || vocab_.empty()) return {};

    // 1. Pre-tokenize into chunks (dispatch based on pre-tokenizer type)
    std::vector<std::string> chunks;
    if (pre_tokenizer_ == "llama3" || pre_tokenizer_ == "llama-v3" ||
        pre_tokenizer_ == "llama-bpe") {
        chunks = llama3_pre_tokenize(text);
    } else {
        chunks = gpt2_pre_tokenize(text);
    }

    std::vector<int32_t> all_ids;
    all_ids.reserve(text.size());  // rough estimate

    for (const auto& chunk : chunks) {
        // 2. Convert each byte to GPT2 unicode character
        std::vector<std::string> symbols;
        symbols.reserve(chunk.size());
        for (unsigned char byte : chunk) {
            symbols.push_back(byte_to_gpt2(byte));
        }

        // 3. BPE merge loop using priority queue: O(n log n)
        int ns = static_cast<int>(symbols.size());
        std::vector<int> sprev(ns), snext(ns);
        std::vector<bool> sdel(ns, false);
        for (int i = 0; i < ns; i++) {
            sprev[i] = i - 1;
            snext[i] = i + 1;
        }

        // Min-heap: lowest rank first, then lowest position
        struct GPT2Merge {
            int rank;
            int pos;
            int seq;
        };
        auto gcmp = [](const GPT2Merge& a, const GPT2Merge& b) {
            if (a.rank != b.rank) return a.rank > b.rank;
            return a.pos > b.pos;
        };
        std::priority_queue<GPT2Merge, std::vector<GPT2Merge>, decltype(gcmp)> gpq(gcmp);

        std::vector<int> sseq(ns, 0);

        for (int i = 0; i < ns - 1; i++) {
            std::string key = symbols[i] + " " + symbols[snext[i]];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end()) {
                gpq.push({it->second, i, sseq[i]});
            }
        }

        while (!gpq.empty()) {
            auto [rank, pos, s] = gpq.top();
            gpq.pop();

            if (sdel[pos] || sseq[pos] != s) continue;
            int right = snext[pos];
            if (right >= ns || sdel[right]) continue;

            // Re-validate: the pair at this position may have changed since
            // the merge was enqueued (e.g., the right neighbor was merged
            // with ITS right neighbor, changing the symbol). Check that the
            // current pair still maps to the same rank.
            {
                std::string cur_key = symbols[pos] + " " + symbols[right];
                auto vit = merge_ranks_.find(cur_key);
                if (vit == merge_ranks_.end() || vit->second != rank) continue;
            }

            symbols[pos] = symbols[pos] + symbols[right];
            sdel[right] = true;
            sseq[pos]++;

            snext[pos] = snext[right];
            if (snext[right] < ns) sprev[snext[right]] = pos;

            if (sprev[pos] >= 0) {
                int lp = sprev[pos];
                std::string key = symbols[lp] + " " + symbols[pos];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end()) {
                    gpq.push({it->second, lp, sseq[lp]});
                }
            }
            if (snext[pos] < ns) {
                std::string key = symbols[pos] + " " + symbols[snext[pos]];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end()) {
                    gpq.push({it->second, pos, sseq[pos]});
                }
            }
        }

        // 4. Look up token IDs
        for (int i = 0; i < ns; i++) {
            if (sdel[i]) continue;
            const auto& sym = symbols[i];
            auto it = token_to_id_.find(sym);
            if (it != token_to_id_.end()) {
                all_ids.push_back(it->second);
            } else {
                // Fallback: try individual GPT2 byte tokens
                for (size_t ci = 0; ci < sym.size(); ) {
                    int len = utf8_char_len(static_cast<uint8_t>(sym[ci]));
                    if (ci + len > sym.size()) len = 1;
                    std::string ch = sym.substr(ci, len);
                    auto ch_it = token_to_id_.find(ch);
                    if (ch_it != token_to_id_.end()) {
                        all_ids.push_back(ch_it->second);
                    }
                    ci += len;
                }
            }
        }
    }

    return all_ids;
}

// ---- NFC Normalization ----
// Handles the most common combining sequences for Latin scripts.
// Covers: accented Latin characters (é, ñ, ü, etc.) which are the vast
// majority of NFC normalization cases in real-world text.

namespace {

// Composition table: (base_codepoint, combining_codepoint) → composed_codepoint
struct NfcEntry {
    uint32_t base;
    uint32_t combining;
    uint32_t composed;
};

// Most common Latin composition pairs (base + combining mark → precomposed)
// Combining marks: 0x0300 (grave), 0x0301 (acute), 0x0302 (circumflex),
//   0x0303 (tilde), 0x0304 (macron), 0x0308 (diaeresis), 0x030C (caron)
static const NfcEntry kNfcTable[] = {
    // Grave accent (0x0300)
    {0x0041, 0x0300, 0x00C0}, // À
    {0x0045, 0x0300, 0x00C8}, // È
    {0x0049, 0x0300, 0x00CC}, // Ì
    {0x004F, 0x0300, 0x00D2}, // Ò
    {0x0055, 0x0300, 0x00D9}, // Ù
    {0x0061, 0x0300, 0x00E0}, // à
    {0x0065, 0x0300, 0x00E8}, // è
    {0x0069, 0x0300, 0x00EC}, // ì
    {0x006F, 0x0300, 0x00F2}, // ò
    {0x0075, 0x0300, 0x00F9}, // ù

    // Acute accent (0x0301)
    {0x0041, 0x0301, 0x00C1}, // Á
    {0x0043, 0x0301, 0x0106}, // Ć
    {0x0045, 0x0301, 0x00C9}, // É
    {0x0049, 0x0301, 0x00CD}, // Í
    {0x004C, 0x0301, 0x0139}, // Ĺ
    {0x004E, 0x0301, 0x0143}, // Ń
    {0x004F, 0x0301, 0x00D3}, // Ó
    {0x0052, 0x0301, 0x0154}, // Ŕ
    {0x0053, 0x0301, 0x015A}, // Ś
    {0x0055, 0x0301, 0x00DA}, // Ú
    {0x0059, 0x0301, 0x00DD}, // Ý
    {0x005A, 0x0301, 0x0179}, // Ź
    {0x0061, 0x0301, 0x00E1}, // á
    {0x0063, 0x0301, 0x0107}, // ć
    {0x0065, 0x0301, 0x00E9}, // é
    {0x0069, 0x0301, 0x00ED}, // í
    {0x006C, 0x0301, 0x013A}, // ĺ
    {0x006E, 0x0301, 0x0144}, // ń
    {0x006F, 0x0301, 0x00F3}, // ó
    {0x0072, 0x0301, 0x0155}, // ŕ
    {0x0073, 0x0301, 0x015B}, // ś
    {0x0075, 0x0301, 0x00FA}, // ú
    {0x0079, 0x0301, 0x00FD}, // ý
    {0x007A, 0x0301, 0x017A}, // ź

    // Circumflex (0x0302)
    {0x0041, 0x0302, 0x00C2}, // Â
    {0x0043, 0x0302, 0x0108}, // Ĉ
    {0x0045, 0x0302, 0x00CA}, // Ê
    {0x0047, 0x0302, 0x011C}, // Ĝ
    {0x0048, 0x0302, 0x0124}, // Ĥ
    {0x0049, 0x0302, 0x00CE}, // Î
    {0x004A, 0x0302, 0x0134}, // Ĵ
    {0x004F, 0x0302, 0x00D4}, // Ô
    {0x0053, 0x0302, 0x015C}, // Ŝ
    {0x0055, 0x0302, 0x00DB}, // Û
    {0x0057, 0x0302, 0x0174}, // Ŵ
    {0x0059, 0x0302, 0x0176}, // Ŷ
    {0x0061, 0x0302, 0x00E2}, // â
    {0x0063, 0x0302, 0x0109}, // ĉ
    {0x0065, 0x0302, 0x00EA}, // ê
    {0x0067, 0x0302, 0x011D}, // ĝ
    {0x0068, 0x0302, 0x0125}, // ĥ
    {0x0069, 0x0302, 0x00EE}, // î
    {0x006A, 0x0302, 0x0135}, // ĵ
    {0x006F, 0x0302, 0x00F4}, // ô
    {0x0073, 0x0302, 0x015D}, // ŝ
    {0x0075, 0x0302, 0x00FB}, // û
    {0x0077, 0x0302, 0x0175}, // ŵ
    {0x0079, 0x0302, 0x0177}, // ŷ

    // Tilde (0x0303)
    {0x0041, 0x0303, 0x00C3}, // Ã
    {0x004E, 0x0303, 0x00D1}, // Ñ
    {0x004F, 0x0303, 0x00D5}, // Õ
    {0x0061, 0x0303, 0x00E3}, // ã
    {0x006E, 0x0303, 0x00F1}, // ñ
    {0x006F, 0x0303, 0x00F5}, // õ

    // Diaeresis/Umlaut (0x0308)
    {0x0041, 0x0308, 0x00C4}, // Ä
    {0x0045, 0x0308, 0x00CB}, // Ë
    {0x0049, 0x0308, 0x00CF}, // Ï
    {0x004F, 0x0308, 0x00D6}, // Ö
    {0x0055, 0x0308, 0x00DC}, // Ü
    {0x0059, 0x0308, 0x0178}, // Ÿ
    {0x0061, 0x0308, 0x00E4}, // ä
    {0x0065, 0x0308, 0x00EB}, // ë
    {0x0069, 0x0308, 0x00EF}, // ï
    {0x006F, 0x0308, 0x00F6}, // ö
    {0x0075, 0x0308, 0x00FC}, // ü
    {0x0079, 0x0308, 0x00FF}, // ÿ

    // Caron/Háček (0x030C)
    {0x0043, 0x030C, 0x010C}, // Č
    {0x0044, 0x030C, 0x010E}, // Ď
    {0x0045, 0x030C, 0x011A}, // Ě
    {0x004E, 0x030C, 0x0147}, // Ň
    {0x0052, 0x030C, 0x0158}, // Ř
    {0x0053, 0x030C, 0x0160}, // Š
    {0x0054, 0x030C, 0x0164}, // Ť
    {0x005A, 0x030C, 0x017D}, // Ž
    {0x0063, 0x030C, 0x010D}, // č
    {0x0064, 0x030C, 0x010F}, // ď
    {0x0065, 0x030C, 0x011B}, // ě
    {0x006E, 0x030C, 0x0148}, // ň
    {0x0072, 0x030C, 0x0159}, // ř
    {0x0073, 0x030C, 0x0161}, // š
    {0x0074, 0x030C, 0x0165}, // ť
    {0x007A, 0x030C, 0x017E}, // ž

    // Cedilla (0x0327)
    {0x0043, 0x0327, 0x00C7}, // Ç
    {0x0063, 0x0327, 0x00E7}, // ç
    {0x0053, 0x0327, 0x015E}, // Ş
    {0x0073, 0x0327, 0x015F}, // ş

    // Ring above (0x030A)
    {0x0041, 0x030A, 0x00C5}, // Å
    {0x0061, 0x030A, 0x00E5}, // å
    {0x0055, 0x030A, 0x016E}, // Ů
    {0x0075, 0x030A, 0x016F}, // ů

    // Macron (0x0304)
    {0x0041, 0x0304, 0x0100}, // Ā
    {0x0045, 0x0304, 0x0112}, // Ē
    {0x0049, 0x0304, 0x012A}, // Ī
    {0x004F, 0x0304, 0x014C}, // Ō
    {0x0055, 0x0304, 0x016A}, // Ū
    {0x0061, 0x0304, 0x0101}, // ā
    {0x0065, 0x0304, 0x0113}, // ē
    {0x0069, 0x0304, 0x012B}, // ī
    {0x006F, 0x0304, 0x014D}, // ō
    {0x0075, 0x0304, 0x016B}, // ū
};

static constexpr int kNfcTableSize = sizeof(kNfcTable) / sizeof(kNfcTable[0]);

// Decode one UTF-8 codepoint from text at position pos, advance pos
static uint32_t nfc_decode_utf8(const std::string& s, size_t& pos) {
    uint8_t c = static_cast<uint8_t>(s[pos]);
    uint32_t cp;
    int len;
    if ((c & 0x80) == 0) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else { pos++; return 0xFFFD; }
    for (int i = 1; i < len && pos + i < s.size(); i++) {
        cp = (cp << 6) | (static_cast<uint8_t>(s[pos + i]) & 0x3F);
    }
    pos += len;
    return cp;
}

// Encode a Unicode codepoint to UTF-8 and append to result
static void nfc_encode_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Check if a codepoint is a combining mark (Unicode General Category Mn/Mc/Me)
// Simplified: only checks the combining diacritical marks block (0x0300-0x036F)
// which covers the vast majority of combining marks in practice.
static bool is_combining_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F);
}

// Look up composition in table
static uint32_t try_compose(uint32_t base, uint32_t combining) {
    for (int i = 0; i < kNfcTableSize; i++) {
        if (kNfcTable[i].base == base && kNfcTable[i].combining == combining) {
            return kNfcTable[i].composed;
        }
    }
    return 0; // no composition found
}

// Normalize a UTF-8 string to NFC form (basic Latin coverage)
static std::string normalize_nfc(const std::string& text) {
    if (text.empty()) return text;

    // Quick check: if no bytes in the combining mark range (0xCC-0xCD in UTF-8),
    // the text has no combining marks and is already NFC.
    bool has_combining = false;
    for (size_t i = 0; i + 1 < text.size(); i++) {
        uint8_t c = static_cast<uint8_t>(text[i]);
        if (c == 0xCC || c == 0xCD) { has_combining = true; break; }
    }
    if (!has_combining) return text;

    // Decode to codepoints, compose adjacent base+combining pairs
    std::vector<uint32_t> codepoints;
    size_t pos = 0;
    while (pos < text.size()) {
        codepoints.push_back(nfc_decode_utf8(text, pos));
    }

    // Compose: scan for base + combining mark pairs
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < codepoints.size()) {
        uint32_t cp = codepoints[i];

        // Try to compose with following combining marks
        while (i + 1 < codepoints.size() && is_combining_mark(codepoints[i + 1])) {
            uint32_t composed = try_compose(cp, codepoints[i + 1]);
            if (composed != 0) {
                cp = composed;
                i++;
            } else {
                break; // can't compose further
            }
        }

        nfc_encode_utf8(result, cp);
        i++;
    }

    return result;
}

} // anonymous namespace

// ---- Encode dispatch ----

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool no_prefix) const {
    // NFC normalization: compose decomposed Unicode sequences
    std::string normalized = normalize_nfc(text);
    if (type_ == "gpt2") {
        return encode_gpt2(normalized);
    }
    return encode_spm(normalized, no_prefix);
}

// ---- Decode (SentencePiece) ----

std::string Tokenizer::decode_spm(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t tok : tokens) {
        result += decode_spm_token(tok);
    }
    return result;
}

std::string Tokenizer::decode_spm_token(int32_t token) const {
    if (token < 0 || token >= static_cast<int32_t>(vocab_.size())) return "";

    std::string piece = vocab_[token];

    // Replace SentencePiece space marker with actual space
    size_t pos = 0;
    while ((pos = piece.find(SPIECE_SPACE, pos)) != std::string::npos) {
        piece.replace(pos, SPIECE_SPACE.size(), " ");
        pos += 1;
    }

    // Handle byte tokens: <0xHH> -> single byte
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
        piece[2] == 'x' && piece[5] == '>') {
        unsigned int byte_val = 0;
        if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    return piece;
}

// ---- Decode (GPT2 byte-level) ----

std::string Tokenizer::decode_gpt2(const std::vector<int32_t>& tokens) const {
    init_codepoint_table();
    std::string result;
    for (int32_t tok : tokens) {
        result += decode_gpt2_token(tok);
    }
    return result;
}

std::string Tokenizer::decode_gpt2_token(int32_t token) const {
    if (token < 0 || token >= static_cast<int32_t>(vocab_.size())) return "";

    const std::string& piece = vocab_[token];
    std::string decoded;

    // Each UTF-8 character in piece represents a byte via GPT2 encoding
    for (size_t i = 0; i < piece.size(); ) {
        int len = utf8_char_len(static_cast<uint8_t>(piece[i]));
        if (i + len > piece.size()) len = 1;

        int byte_val = gpt2_to_byte(piece.data() + i, len);
        if (byte_val >= 0) {
            decoded += static_cast<char>(byte_val);
        } else {
            // Not a GPT2 byte-encoded char, pass through
            decoded += piece.substr(i, len);
        }
        i += len;
    }

    return decoded;
}

// ---- Decode dispatch ----

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    if (type_ == "gpt2") {
        return decode_gpt2(tokens);
    }
    return decode_spm(tokens);
}

std::string Tokenizer::decode_token(int32_t token) const {
    if (type_ == "gpt2") {
        return decode_gpt2_token(token);
    }
    return decode_spm_token(token);
}

// ---- Accessors ----

int Tokenizer::vocab_size() const {
    return static_cast<int>(vocab_.size());
}

int Tokenizer::bos_id() const {
    return bos_id_;
}

// eos_id() is now inline in tokenizer.h

int32_t Tokenizer::find_token(const std::string& text) const {
    auto it = token_to_id_.find(text);
    if (it != token_to_id_.end()) return it->second;
    return -1;
}

} // namespace imp

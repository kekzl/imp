#include "model/tokenizer.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <climits>
#include <queue>

namespace imp {

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
    (void)path;
    return false;
}

bool Tokenizer::load_vocab(const std::vector<std::string>& tokens,
                           const std::vector<float>& scores,
                           int bos_id, int eos_id) {
    if (tokens.empty()) return false;

    vocab_ = tokens;
    scores_ = scores;
    scores_.resize(vocab_.size(), 0.0f);
    bos_id_ = bos_id;
    eos_id_ = eos_id;

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

std::vector<int32_t> Tokenizer::encode_gpt2(const std::string& text) const {
    if (text.empty() || vocab_.empty()) return {};

    // 1. Pre-tokenize into chunks
    std::vector<std::string> chunks = gpt2_pre_tokenize(text);

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

// ---- Encode dispatch ----

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool no_prefix) const {
    if (type_ == "gpt2") {
        return encode_gpt2(text);
    }
    return encode_spm(text, no_prefix);
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

int Tokenizer::eos_id() const {
    return eos_id_;
}

int32_t Tokenizer::find_token(const std::string& text) const {
    auto it = token_to_id_.find(text);
    if (it != token_to_id_.end()) return it->second;
    return -1;
}

} // namespace imp

// tiktoken checkpoints (moonshotai Kimi family): `tiktoken.model` is one "<base64 bytes> <rank>" line
// per token, rank = id, no merges list. tiktoken BPE merges the adjacent pair whose concatenation has
// the lowest rank; every two-token split of a token is that token's merge pair, ranked by its id.
#include "core/logging.h"
#include "model/json_util.h"
#include "model/tokenizer.h"
#include "model/tokenizer_regex.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace imp {

// TikTokenTokenizer.pat_str of tokenization_kimi.py, alternatives joined in order.
const char* const kKimiTiktokenRegex =
    R"([\p{Han}]+)"
    R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|\p{N}{1,3})"
    R"(| ?[^\s\p{L}\p{N}]+[\r\n]*)"
    R"(|\s*[\r\n]+)"
    R"(|\s+(?!\S))"
    R"(|\s+)";

namespace {

// tokenization_kimi.py: num_reserved_special_tokens 256 + 2 ids above the base vocabulary.
constexpr int64_t kTiktokenSpecialSlots = 258;
// Largest real file is 2.8 MB (Kimi-Linear, 163584 lines).
constexpr size_t kMaxTiktokenFileBytes = size_t{64} << 20;

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

// Standard alphabet, '=' padding; false on any other byte or a dangling 6-bit group.
bool b64_decode(std::string_view in, std::string& out) {
    out.clear();
    uint32_t acc = 0;
    int bits = 0;
    size_t pad = 0;
    for (char c : in) {
        if (c == '=') {
            ++pad;
            continue;
        }
        const int v = b64_value(c);
        if (v < 0 || pad > 0)
            return false;
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += static_cast<char>((acc >> bits) & 0xFF);
        }
    }
    return pad <= 2 && bits < 6 && !out.empty();
}

// Decimal rank in [0, kMaxTokenId], -1 otherwise.
int64_t parse_rank(std::string_view s) {
    if (s.empty())
        return -1;
    int64_t rank = 0;
    for (char c : s) {
        if (c < '0' || c > '9' || rank > kMaxTokenId)
            return -1;
        rank = (rank * 10) + (c - '0');
    }
    return rank <= kMaxTokenId ? rank : -1;
}

struct Ranks {
    std::vector<std::string> raw;                  // id -> token bytes
    std::unordered_map<std::string, int32_t> ids;  // token bytes -> id
};

// One "<base64> <rank>" line; a repeated rank or token is not a tiktoken file.
bool add_rank(std::string_view line, size_t line_no, const std::string& path, Ranks& r) {
    const size_t sp = line.find(' ');
    const int64_t rank = sp == std::string_view::npos ? -1 : parse_rank(line.substr(sp + 1));
    std::string bytes;
    if (rank < 0 || !b64_decode(line.substr(0, sp), bytes)) {
        IMP_LOG_WARN("tiktoken: %s line %zu is not '<base64> <rank>'", path.c_str(), line_no);
        return false;
    }
    if (static_cast<size_t>(rank) >= r.raw.size())
        r.raw.resize(static_cast<size_t>(rank) + 1);
    if (!r.raw[rank].empty() || !r.ids.emplace(bytes, static_cast<int32_t>(rank)).second) {
        IMP_LOG_WARN("tiktoken: %s line %zu repeats rank %lld or its token", path.c_str(), line_no,
                     static_cast<long long>(rank));
        return false;
    }
    r.raw[rank] = std::move(bytes);
    return true;
}

// Ranks dense from 0 (rank = id) and every byte a token, so byte-level BPE always ends in tokens.
bool parse_ranks(const std::string& data, const std::string& path, Ranks& r) {
    size_t line_no = 0;
    for (size_t pos = 0; pos < data.size();) {
        const size_t eol = std::min(data.find('\n', pos), data.size());
        std::string_view line(data.data() + pos, eol - pos);
        pos = eol + 1;
        ++line_no;
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (!line.empty() && !add_rank(line, line_no, path, r))
            return false;
    }
    if (r.raw.empty() || r.ids.size() != r.raw.size()) {
        IMP_LOG_WARN("tiktoken: %s ranks are not dense (%zu tokens, %zu slots)", path.c_str(), r.ids.size(),
                     r.raw.size());
        return false;
    }
    for (int b = 0; b < 256; b++) {
        if (!r.ids.contains(std::string(1, static_cast<char>(b)))) {
            IMP_LOG_WARN("tiktoken: %s has no token for byte 0x%02x", path.c_str(), b);
            return false;
        }
    }
    return true;
}

// tokenizer_config.json added_tokens_decoder entries above the base vocabulary; a missing file is none.
std::unordered_map<int64_t, std::string> named_specials(const std::string& config_path, int64_t n_base) {
    std::unordered_map<int64_t, std::string> named;
    JValue cfg;
    if (!parse_json_file(config_path, cfg))
        return named;
    const JValue* dec = jobj_find(cfg, "added_tokens_decoder");
    if (!dec || dec->type != JType::OBJECT)
        return named;
    for (const auto& [key, tok] : dec->obj) {
        std::string content;
        const int64_t id = parse_rank(key);
        if (tok.type == JType::OBJECT && jobj_get_string(tok, "content", content) && id >= n_base &&
            !content.empty())
            named[id] = content;
    }
    return named;
}

// (left id << 32 | right id) -> {rank, merged id} for every split of a token into two tokens.
std::unordered_map<uint64_t, std::pair<int, int32_t>> merge_pairs(const Ranks& r) {
    std::unordered_map<uint64_t, std::pair<int, int32_t>> pairs;
    for (size_t id = 0; id < r.raw.size(); id++) {
        const std::string& t = r.raw[id];
        for (size_t k = 1; k < t.size(); k++) {
            auto left = r.ids.find(t.substr(0, k));
            auto right = left == r.ids.end() ? r.ids.end() : r.ids.find(t.substr(k));
            if (right == r.ids.end())
                continue;
            const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left->second)) << 32) |
                                 static_cast<uint32_t>(right->second);
            pairs[key] = {static_cast<int>(id), static_cast<int32_t>(id)};
        }
    }
    return pairs;
}

}  // namespace

bool Tokenizer::load_tiktoken(const std::string& model_path, const std::string& config_path) {
    const std::string data = read_file(model_path);
    Ranks r;
    if (data.empty() || data.size() > kMaxTiktokenFileBytes) {
        IMP_LOG_WARN("tiktoken: %s unreadable or over %zu bytes", model_path.c_str(), kMaxTiktokenFileBytes);
        return false;
    }
    if (!parse_ranks(data, model_path, r))
        return false;
    const auto n_base = static_cast<int64_t>(r.raw.size());

    // Specials: added_tokens_decoder names, every other slot <|reserved_token_N|> as tiktoken names it.
    const auto named = named_specials(config_path, n_base);
    int64_t n_total = n_base + kTiktokenSpecialSlots;
    for (const auto& [id, content] : named)
        n_total = std::max(n_total, id + 1);

    vocab_.assign(static_cast<size_t>(n_total), {});
    scores_.assign(static_cast<size_t>(n_total), 0.0f);
    token_types_.assign(static_cast<size_t>(n_total), 1);
    added_token_ids_.assign(static_cast<size_t>(n_total), false);
    token_to_id_.clear();
    token_to_id_.reserve(static_cast<size_t>(n_total));
    for (int64_t id = 0; id < n_base; id++) {
        for (unsigned char c : r.raw[id])
            vocab_[id] += gpt2_byte_piece(c);
        token_to_id_.emplace(vocab_[id], static_cast<int32_t>(id));
    }
    for (int64_t id = n_base; id < n_total; id++) {
        auto it = named.find(id);
        vocab_[id] = it != named.end() ? it->second : "<|reserved_token_" + std::to_string(id) + "|>";
        token_to_id_[vocab_[id]] = static_cast<int32_t>(id);
        token_types_[id] = 3;  // CONTROL: tiktoken allowed_special="all" matches every special atomically
        added_token_ids_[id] = true;
    }
    for (int b = 0; b < 256; b++)
        byte_ids_[b] = r.ids.at(std::string(1, static_cast<char>(b)));
    merge_ranks_.clear();
    merge_pairs_ = merge_pairs(r);
    merge_pairs_ok_ = true;

    type_ = "gpt2";
    nfc_ = false;  // tiktoken encodes the input bytes as given
    add_space_prefix_ = false;
    add_bos_ = false;       // TikTokenTokenizer.encode adds no BOS
    ignore_merges_ = true;  // tiktoken: a pre-token found whole in the ranks is one token
    std::string why;
    regex_pretok_ = RegexPretokenizer::from_regexes({kKimiTiktokenRegex}, &why);
    if (!regex_pretok_) {
        IMP_LOG_WARN("tiktoken: pre-tokenizer regex refused: %s", why.c_str());
        return false;
    }
    build_special_pieces();
    IMP_LOG_INFO("tiktoken: %s: %lld ranks, %lld specials (%zu named), %zu merge pairs", model_path.c_str(),
                 static_cast<long long>(n_base), static_cast<long long>(n_total - n_base), named.size(),
                 merge_pairs_.size());
    return true;
}

}  // namespace imp

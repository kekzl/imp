#pragma once

// Banned phrases (roadmap row 118): a text match on the decoded output rewinds the request to
// the phrase's first token and bans that token at that position. Pure helpers and the per-request
// state here; the engine side is runtime/engine_phrase_ban.cpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp {

class PhraseRegex;

inline constexpr int kMaxBannedPhrases = 64;
inline constexpr size_t kMaxBannedPhraseBytes = 256;
inline constexpr int kMaxPhraseRewinds = 64;  // per request; past it a phrase is let through
// Hybrids: recurrent-state slabs per request, slot 0 = after prefill, slots 1.. = every
// kPhraseSnapEvery output tokens; a rewind replays at most the tokens after its slab.
inline constexpr int kPhraseSnapEvery = 32;
inline constexpr int kPhraseSnapSlots = 3;

struct PhraseBanState {
    std::vector<std::string> phrases;
    std::vector<std::shared_ptr<const PhraseRegex>> regex;  // runtime/phrase_regex.h
    size_t max_len = 0;
    std::string text;                                  // decoded output_tokens
    std::vector<uint32_t> tok_end;                     // byte end of output token i in text
    std::vector<std::pair<int, int32_t>> bans;         // (output position, banned token id)
    std::vector<std::pair<int32_t, float>> user_bias;  // the request's own logit_bias
    std::vector<float> first_logits;                   // raw logits of output position 0 (host)
    size_t safe = 0;                                   // output tokens no rewind can reach
    int rewinds = 0;
    // Think state at add_request: a rewind replays track_think_state from here.
    bool in_think0 = false;
    int think_exit0 = -1;
    bool content_after0 = false;
    std::string think_tail0;
    // Hybrids: output tokens folded into each recurrent slab (-1 = empty); kept tokens a rewind
    // re-forwards before the banned position, emitted from replay_at on.
    std::array<int, kPhraseSnapSlots> snap_consumed = [] {
        std::array<int, kPhraseSnapSlots> a{};
        a.fill(-1);
        return a;
    }();
    std::vector<int32_t> replay;
    size_t replay_at = 0;
    [[nodiscard]] bool replaying() const { return replay_at < replay.size(); }
    // The token a decode step emits: the kept one while replaying, else the sampled one.
    [[nodiscard]] int32_t emit(int32_t sampled) const { return replaying() ? replay[replay_at] : sampled; }
};

// Slab to restore for a rewind to output position pos > 0: the largest consumed <= pos - 1
// (the next forward then reads output[consumed]). -1 = none.
inline int phrase_pick_snapshot(const std::array<int, kPhraseSnapSlots>& consumed, size_t pos) {
    int best = -1;
    for (int i = 0; i < kPhraseSnapSlots; i++)
        if (consumed[i] >= 0 && static_cast<size_t>(consumed[i]) + 1 <= pos &&
            (best < 0 || consumed[i] > consumed[best]))
            best = i;
    return best;
}

// Slot a periodic snapshot overwrites: never slot 0, an empty one first, else the oldest.
inline int phrase_snapshot_slot(const std::array<int, kPhraseSnapSlots>& consumed) {
    int slot = 1;
    for (int i = 1; i < kPhraseSnapSlots; i++)
        if (consumed[i] < consumed[slot])
            slot = i;
    return slot;
}

// Earliest start byte of a phrase occurrence that ends after byte prev_len (only new text matches).
inline std::optional<size_t> phrase_match_start(std::string_view text, size_t prev_len,
                                                const std::vector<std::string>& phrases) {
    std::optional<size_t> best;
    for (const auto& p : phrases) {
        if (p.empty() || p.size() > text.size())
            continue;
        const size_t from = prev_len + 1 > p.size() ? prev_len + 1 - p.size() : 0;
        const size_t at = text.find(p, from);
        if (at != std::string_view::npos && (!best || at < *best))
            best = at;
    }
    return best;
}

// Output token holding byte b: the first i with tok_end[i] > b.
inline size_t phrase_token_at(const std::vector<uint32_t>& tok_end, size_t b) {
    return static_cast<size_t>(std::upper_bound(tok_end.begin(), tok_end.end(), static_cast<uint32_t>(b)) -
                               tok_end.begin());
}

// Tokens whose text ends at or before text_len - (max_len - 1): no later match can start in them.
inline size_t phrase_safe_tokens(const std::vector<uint32_t>& tok_end, size_t text_len, size_t max_len) {
    const size_t keep = max_len > 0 ? max_len - 1 : 0;
    if (text_len < keep)
        return 0;
    const size_t limit = text_len - keep;
    return static_cast<size_t>(
        std::upper_bound(tok_end.begin(), tok_end.end(), static_cast<uint32_t>(limit)) - tok_end.begin());
}

}  // namespace imp

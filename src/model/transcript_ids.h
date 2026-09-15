#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace imp {

class Tokenizer;

// Finished transcripts (prompt + reply) as the model forwarded them, keyed by their text.
// A follow-up turn that resends the reply verbatim tokenizes it with the model's own BPE
// split instead of the canonical one, so the prefix-cache hash chain survives a
// non-canonical merge inside the reply (` pre`+`pref` re-encodes as ` prep`+`ref`).
// FIFO, max_entries transcripts; a subsumed shorter transcript stays for branches.
class TranscriptIdStore {
public:
    explicit TranscriptIdStore(size_t max_entries = 256) : max_entries_(max_entries) {}

    // The text `ids` decode to, control tokens as their literal (what a template renders).
    static std::string text_of(const Tokenizer& tok, std::span<const int32_t> ids);

    void remember(std::string text, std::vector<int32_t> ids);
    // remember() of a request's forwarded span with trailing EOS / stop ids dropped: the
    // template renders its own end marker after the reply.
    void remember_forwarded(const Tokenizer& tok, std::span<const int32_t> stop_ids,
                            std::span<const int32_t> forwarded);

    // Longest stored text that is a prefix of `rendered`: appends its ids to `out`, returns
    // the matched byte length (0 = no entry matches, `out` untouched).
    size_t lookup(std::string_view rendered, std::vector<int32_t>& out) const;

    // ChatTemplate::tokenize_rendered: `result` holds a pushed BOS or nothing; on a match
    // it becomes the stored ids and the byte offset to tokenize `rendered` from is returned
    // (0 = no match, `result` untouched). A pushed BOS is probed as its literal text.
    size_t splice(const Tokenizer& tok, int32_t bos_id, const std::string& rendered,
                  std::vector<int32_t>& result) const;

    size_t size() const;

private:
    struct Entry {
        std::string text;
        std::vector<int32_t> ids;
    };
    size_t max_entries_;
    mutable std::mutex mu_;
    std::deque<Entry> entries_;  // insertion order, oldest first
};

}  // namespace imp

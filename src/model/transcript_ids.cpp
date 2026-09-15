#include "model/transcript_ids.h"

#include "model/tokenizer.h"

#include <algorithm>

namespace imp {

std::string TranscriptIdStore::text_of(const Tokenizer& tok, std::span<const int32_t> ids) {
    std::string text;
    for (int32_t id : ids)
        text += tok.decode_token(id);
    return text;
}

void TranscriptIdStore::remember(std::string text, std::vector<int32_t> ids) {
    if (text.empty() || ids.empty() || max_entries_ == 0)
        return;
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.text == text) {
            e.ids = std::move(ids);
            return;
        }
    }
    while (entries_.size() >= max_entries_)
        entries_.pop_front();
    entries_.push_back({std::move(text), std::move(ids)});
}

void TranscriptIdStore::remember_forwarded(const Tokenizer& tok, std::span<const int32_t> stop_ids,
                                           std::span<const int32_t> forwarded) {
    const auto& eos = tok.eos_ids();
    auto is_stop = [&](int32_t id) {
        return std::find(eos.begin(), eos.end(), id) != eos.end() ||
               std::find(stop_ids.begin(), stop_ids.end(), id) != stop_ids.end();
    };
    while (!forwarded.empty() && is_stop(forwarded.back()))
        forwarded = forwarded.first(forwarded.size() - 1);
    if (forwarded.empty())
        return;
    remember(text_of(tok, forwarded), std::vector<int32_t>(forwarded.begin(), forwarded.end()));
}

size_t TranscriptIdStore::lookup(std::string_view rendered, std::vector<int32_t>& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    const Entry* best = nullptr;
    for (const auto& e : entries_) {
        if (e.text.size() > rendered.size() || (best && e.text.size() <= best->text.size()))
            continue;
        if (rendered.compare(0, e.text.size(), e.text) == 0)
            best = &e;
    }
    if (!best)
        return 0;
    out.insert(out.end(), best->ids.begin(), best->ids.end());
    return best->text.size();
}

size_t TranscriptIdStore::splice(const Tokenizer& tok, int32_t bos_id, const std::string& rendered,
                                 std::vector<int32_t>& result) const {
    // The stored text carries the BOS literal when the ids do, so a BOS the caller pushed
    // is probed as text too.
    std::string with_bos;
    if (!result.empty() && bos_id >= 0)
        with_bos = tok.token_text(bos_id) + rendered;
    const size_t bos_len = with_bos.empty() ? 0 : with_bos.size() - rendered.size();
    std::vector<int32_t> ids;
    const size_t matched = lookup(bos_len ? std::string_view(with_bos) : std::string_view(rendered), ids);
    if (matched <= bos_len)
        return 0;
    result = std::move(ids);
    return matched - bos_len;
}

size_t TranscriptIdStore::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

}  // namespace imp

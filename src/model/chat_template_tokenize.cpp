// ChatTemplate: rendered text -> token ids (control-token split, transcript id splice).
#include "model/chat_template.h"
#include "core/logging.h"

#include <algorithm>

namespace imp {

void ChatTemplate::build_control_token_map(const Tokenizer& tok) {
    control_tokens_.clear();
    if (!tok.has_token_types())
        return;

    int vs = tok.vocab_size();
    for (int i = 0; i < vs; i++) {
        if (tok.is_control_token(i)) {
            const auto& text = tok.token_text(i);
            if (!text.empty()) {
                control_tokens_.emplace_back(text, static_cast<int32_t>(i));
            }
        }
    }
    // Sort longest first for greedy matching
    std::sort(control_tokens_.begin(), control_tokens_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

// Split rendered Jinja2 output on control tokens and encode the segments between them.
std::vector<int32_t> ChatTemplate::tokenize_rendered(const Tokenizer& tok,
                                                     const std::string& rendered) const {
    // Control tokens appear as literal text in the rendered output (e.g. "<|im_start|>").
    // We identify them via the control_tokens_ map (sorted longest-first).
    std::vector<int32_t> result;
    // Skip BOS if the rendered string already contains the BOS token text: the
    // control token splitter below will add it. Adding it here would duplicate.
    bool rendered_has_bos = false;
    if (bos_id_ >= 0) {
        const std::string& bos_text = tok.token_text(bos_id_);
        if (!bos_text.empty() && rendered.find(bos_text) != std::string::npos)
            rendered_has_bos = true;
    }
    if (tok.add_bos() && !rendered_has_bos) {
        result.push_back(bos_id_);
    }

    // A finished transcript's ids cover a prefix of this render: keep the model's own split
    // and tokenize only the tail (roadmap row 15).
    size_t pos = 0;
    if (transcript_store_) {
        pos = transcript_store_->splice(tok, bos_id_, rendered, result);
        if (pos > 0)
            IMP_LOG_DEBUG("TranscriptIds: kept %zu forwarded ids over %zu bytes, tokenizing %zu tail bytes",
                          result.size(), pos, rendered.size() - pos);
    }
    while (pos < rendered.size()) {
        // Try to match a control token at current position
        bool matched = false;
        for (const auto& [text, id] : control_tokens_) {
            if (rendered.compare(pos, text.size(), text) == 0) {
                result.push_back(id);
                pos += text.size();
                matched = true;
                break;
            }
        }
        if (matched)
            continue;

        // Collect text until the next control token
        size_t next = rendered.size();
        for (const auto& [text, id] : control_tokens_) {
            size_t found = rendered.find(text, pos);
            if (found != std::string::npos && found < next) {
                next = found;
            }
        }

        // Encode the text segment
        std::string segment = rendered.substr(pos, next - pos);
        if (!segment.empty()) {
            auto ids = tok.encode(segment, true);  // no_prefix=true for template segments
            result.insert(result.end(), ids.begin(), ids.end());
        }
        pos = next;
    }

    return result;
}

}  // namespace imp

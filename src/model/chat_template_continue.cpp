// Assistant prefill (HF continue_final_message): the model continues a trailing assistant turn.

#include "model/chat_template.h"

namespace imp {

namespace {

constexpr const char* kSpace = " \t\n\r\f\v";

std::string lstrip(const std::string& s) {
    const size_t b = s.find_first_not_of(kSpace);
    return b == std::string::npos ? std::string() : s.substr(b);
}

std::string strip(const std::string& s) {
    const std::string l = lstrip(s);
    const size_t e = l.find_last_not_of(kSpace);
    return e == std::string::npos ? std::string() : l.substr(0, e + 1);
}

}  // namespace

std::string ChatTemplate::cut_after_final_content(const std::string& rendered,
                                                  const std::string& final_content) {
    const std::string stripped = strip(final_content);
    if (stripped.empty())
        return {};
    const size_t pos = rendered.rfind(stripped);
    if (pos == std::string::npos)
        return {};
    // Verbatim (trailing whitespace kept by the template): keep it; trimmed: stop at the text.
    const std::string lstripped = lstrip(final_content);
    if (rendered.compare(pos, lstripped.size(), lstripped) == 0)
        return rendered.substr(0, pos + lstripped.size());
    return rendered.substr(0, pos + stripped.size());
}

bool ChatTemplate::continuing_(const std::vector<ChatMessage>& msgs) const {
    return continue_final_ && !msgs.empty() && msgs.back().role == "assistant" &&
           msgs.back().tool_calls_json.empty() && !strip(msgs.back().content).empty();
}

std::vector<int32_t> ChatTemplate::continue_after_history_(
    const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
    const std::function<std::vector<int32_t>(const ChatTemplate&, const std::vector<ChatMessage>&)>& render)
    const {
    ChatTemplate plain = *this;
    plain.continue_final_ = false;
    const std::vector<ChatMessage> history(msgs.begin(), msgs.end() - 1);
    std::vector<int32_t> tokens = render(plain, history);
    if (tokens.empty())
        return tokens;
    const std::vector<int32_t> text = tok.encode(msgs.back().content);
    tokens.insert(tokens.end(), text.begin(), text.end());
    return tokens;
}

}  // namespace imp

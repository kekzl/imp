#include "model/chat_template.h"
#include "core/logging.h"

#include <algorithm>

namespace imp {

const char* chat_template_family_name(ChatTemplateFamily family) {
    switch (family) {
        case ChatTemplateFamily::RAW:      return "raw";
        case ChatTemplateFamily::CHATML:   return "chatml";
        case ChatTemplateFamily::LLAMA2:   return "llama2";
        case ChatTemplateFamily::LLAMA3:   return "llama3";
        case ChatTemplateFamily::NEMOTRON: return "nemotron";
    }
    return "unknown";
}

ChatTemplateFamily ChatTemplate::detect_family(const std::string& jinja2_str) {
    if (jinja2_str.empty()) return ChatTemplateFamily::RAW;

    // Order matters: check more specific patterns first
    if (jinja2_str.find("<|im_start|>") != std::string::npos)
        return ChatTemplateFamily::CHATML;
    if (jinja2_str.find("<|start_header_id|>") != std::string::npos)
        return ChatTemplateFamily::LLAMA3;
    if (jinja2_str.find("[INST]") != std::string::npos)
        return ChatTemplateFamily::LLAMA2;
    if (jinja2_str.find("<extra_id_0>") != std::string::npos)
        return ChatTemplateFamily::NEMOTRON;

    return ChatTemplateFamily::RAW;
}

ChatTemplateFamily ChatTemplate::parse_family(const std::string& name) {
    if (name == "auto")     return ChatTemplateFamily::RAW;  // caller handles auto
    if (name == "none")     return ChatTemplateFamily::RAW;
    if (name == "chatml")   return ChatTemplateFamily::CHATML;
    if (name == "llama2")   return ChatTemplateFamily::LLAMA2;
    if (name == "llama3")   return ChatTemplateFamily::LLAMA3;
    if (name == "nemotron") return ChatTemplateFamily::NEMOTRON;
    return ChatTemplateFamily::RAW;
}

bool ChatTemplate::init(ChatTemplateFamily family, const Tokenizer& tokenizer) {
    family_ = family;
    stop_token_ids_.clear();

    if (family_ == ChatTemplateFamily::RAW) {
        return true;
    }

    bos_id_ = static_cast<int32_t>(tokenizer.bos_id());

    switch (family_) {
        case ChatTemplateFamily::CHATML: {
            im_start_id_ = tokenizer.find_token("<|im_start|>");
            im_end_id_   = tokenizer.find_token("<|im_end|>");
            if (im_start_id_ < 0 || im_end_id_ < 0) {
                IMP_LOG_WARN("ChatML template: missing special tokens "
                             "(im_start=%d, im_end=%d), falling back to raw",
                             im_start_id_, im_end_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(im_end_id_);
            break;
        }
        case ChatTemplateFamily::LLAMA3: {
            start_header_id_ = tokenizer.find_token("<|start_header_id|>");
            end_header_id_   = tokenizer.find_token("<|end_header_id|>");
            eot_id_          = tokenizer.find_token("<|eot_id|>");
            if (start_header_id_ < 0 || end_header_id_ < 0 || eot_id_ < 0) {
                IMP_LOG_WARN("Llama3 template: missing special tokens "
                             "(start_header=%d, end_header=%d, eot=%d), falling back to raw",
                             start_header_id_, end_header_id_, eot_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(eot_id_);
            break;
        }
        case ChatTemplateFamily::LLAMA2: {
            inst_start_id_ = tokenizer.find_token("[INST]");
            inst_end_id_   = tokenizer.find_token("[/INST]");
            if (inst_start_id_ < 0 || inst_end_id_ < 0) {
                IMP_LOG_WARN("Llama2 template: missing special tokens "
                             "(inst_start=%d, inst_end=%d), falling back to raw",
                             inst_start_id_, inst_end_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(static_cast<int32_t>(tokenizer.eos_id()));
            break;
        }
        case ChatTemplateFamily::NEMOTRON: {
            extra_id_0_ = tokenizer.find_token("<extra_id_0>");
            extra_id_1_ = tokenizer.find_token("<extra_id_1>");
            if (extra_id_0_ < 0 || extra_id_1_ < 0) {
                IMP_LOG_WARN("Nemotron template: missing special tokens "
                             "(extra_id_0=%d, extra_id_1=%d), falling back to raw",
                             extra_id_0_, extra_id_1_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(extra_id_1_);
            break;
        }
        default:
            break;
    }

    IMP_LOG_INFO("Chat template: %s", chat_template_family_name(family_));
    return true;
}

std::vector<int32_t> ChatTemplate::apply(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& messages) const
{
    switch (family_) {
        case ChatTemplateFamily::CHATML:   return apply_chatml(tok, messages);
        case ChatTemplateFamily::LLAMA3:   return apply_llama3(tok, messages);
        case ChatTemplateFamily::LLAMA2:   return apply_llama2(tok, messages);
        case ChatTemplateFamily::NEMOTRON: return apply_nemotron(tok, messages);
        default: break;
    }
    // RAW: should not be called, but handle gracefully
    return {};
}

// ChatML: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
std::vector<int32_t> ChatTemplate::apply_chatml(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    for (const auto& msg : msgs) {
        tokens.push_back(im_start_id_);
        auto role_ids = tok.encode(msg.role + "\n");
        tokens.insert(tokens.end(), role_ids.begin(), role_ids.end());
        auto content_ids = tok.encode(msg.content);
        tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        tokens.push_back(im_end_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Assistant generation prefix
    tokens.push_back(im_start_id_);
    auto asst_ids = tok.encode("assistant\n");
    tokens.insert(tokens.end(), asst_ids.begin(), asst_ids.end());

    return tokens;
}

// Llama3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|> ...
std::vector<int32_t> ChatTemplate::apply_llama3(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    for (const auto& msg : msgs) {
        tokens.push_back(start_header_id_);
        auto role_ids = tok.encode(msg.role);
        tokens.insert(tokens.end(), role_ids.begin(), role_ids.end());
        tokens.push_back(end_header_id_);
        auto content_ids = tok.encode("\n\n" + msg.content);
        tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        tokens.push_back(eot_id_);
    }

    // Assistant generation prefix
    tokens.push_back(start_header_id_);
    auto asst_ids = tok.encode("assistant");
    tokens.insert(tokens.end(), asst_ids.begin(), asst_ids.end());
    tokens.push_back(end_header_id_);
    auto nl_ids = tok.encode("\n\n");
    tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());

    return tokens;
}

// Llama2: <s>[INST] content [/INST]
std::vector<int32_t> ChatTemplate::apply_llama2(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    // Find system message if present
    std::string system_text;
    for (const auto& msg : msgs) {
        if (msg.role == "system") {
            system_text = msg.content;
            break;
        }
    }

    bool first_user = true;
    for (const auto& msg : msgs) {
        if (msg.role == "system") continue;

        if (msg.role == "user") {
            tokens.push_back(inst_start_id_);
            if (first_user && !system_text.empty()) {
                auto sys_ids = tok.encode("<<SYS>>\n" + system_text + "\n<</SYS>>\n\n");
                tokens.insert(tokens.end(), sys_ids.begin(), sys_ids.end());
            }
            auto content_ids = tok.encode(" " + msg.content + " ");
            tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
            tokens.push_back(inst_end_id_);
            first_user = false;
        } else if (msg.role == "assistant") {
            auto content_ids = tok.encode(" " + msg.content + " ");
            tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
            tokens.push_back(static_cast<int32_t>(tok.eos_id()));
        }
    }

    return tokens;
}

// Nemotron: <extra_id_0>System\ncontent\n<extra_id_1>\n<extra_id_0>User\ncontent\n<extra_id_1>\n<extra_id_0>Assistant\n
std::vector<int32_t> ChatTemplate::apply_nemotron(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    // Capitalize role names for Nemotron format
    auto capitalize = [](const std::string& s) -> std::string {
        if (s.empty()) return s;
        std::string result = s;
        result[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(result[0])));
        return result;
    };

    for (const auto& msg : msgs) {
        tokens.push_back(extra_id_0_);
        auto header_ids = tok.encode(capitalize(msg.role) + "\n" + msg.content + "\n");
        tokens.insert(tokens.end(), header_ids.begin(), header_ids.end());
        tokens.push_back(extra_id_1_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Assistant generation prefix
    tokens.push_back(extra_id_0_);
    auto asst_ids = tok.encode("Assistant\n");
    tokens.insert(tokens.end(), asst_ids.begin(), asst_ids.end());

    return tokens;
}

} // namespace imp

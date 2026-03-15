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
        case ChatTemplateFamily::NEMOTRON:    return "nemotron";
        case ChatTemplateFamily::GEMMA:       return "gemma";
        case ChatTemplateFamily::DEEPSEEK_R1: return "deepseek_r1";
        case ChatTemplateFamily::PHI:          return "phi";
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
    if (jinja2_str.find("<start_of_turn>") != std::string::npos)
        return ChatTemplateFamily::GEMMA;
    if (jinja2_str.find("[INST]") != std::string::npos)
        return ChatTemplateFamily::LLAMA2;
    if (jinja2_str.find("<extra_id_0>") != std::string::npos)
        return ChatTemplateFamily::NEMOTRON;
    // DeepSeek R1: fullwidth vertical bars ｜ (U+FF5C = \xef\xbd\x9c)
    if (jinja2_str.find("\xef\xbd\x9c" "User" "\xef\xbd\x9c") != std::string::npos)
        return ChatTemplateFamily::DEEPSEEK_R1;
    // Phi: <|end|> is literal in the Jinja2 template (role tags are dynamic)
    if (jinja2_str.find("<|end|>") != std::string::npos)
        return ChatTemplateFamily::PHI;

    return ChatTemplateFamily::RAW;
}

ChatTemplateFamily ChatTemplate::default_family_for_arch(ModelArch arch) {
    switch (arch) {
        case ModelArch::LLAMA:          return ChatTemplateFamily::LLAMA3;
        case ModelArch::MISTRAL:        return ChatTemplateFamily::LLAMA2;
        case ModelArch::MIXTRAL:        return ChatTemplateFamily::LLAMA2;
        case ModelArch::DEEPSEEK:       return ChatTemplateFamily::DEEPSEEK_R1;
        case ModelArch::NEMOTRON_H_MOE: return ChatTemplateFamily::NEMOTRON;
        case ModelArch::QWEN3:          return ChatTemplateFamily::CHATML;
        case ModelArch::QWEN3_MOE:      return ChatTemplateFamily::CHATML;
        case ModelArch::GEMMA3:         return ChatTemplateFamily::GEMMA;
        case ModelArch::LLAMA4:         return ChatTemplateFamily::LLAMA3;
        default:                        return ChatTemplateFamily::RAW;
    }
}

ChatTemplateFamily ChatTemplate::parse_family(const std::string& name) {
    if (name == "auto")     return ChatTemplateFamily::RAW;  // caller handles auto
    if (name == "none")     return ChatTemplateFamily::RAW;
    if (name == "chatml")   return ChatTemplateFamily::CHATML;
    if (name == "llama2")   return ChatTemplateFamily::LLAMA2;
    if (name == "llama3")   return ChatTemplateFamily::LLAMA3;
    if (name == "nemotron") return ChatTemplateFamily::NEMOTRON;
    if (name == "gemma")   return ChatTemplateFamily::GEMMA;
    if (name == "deepseek_r1" || name == "deepseek-r1") return ChatTemplateFamily::DEEPSEEK_R1;
    if (name == "phi") return ChatTemplateFamily::PHI;
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
        case ChatTemplateFamily::GEMMA: {
            start_of_turn_id_ = tokenizer.find_token("<start_of_turn>");
            end_of_turn_id_   = tokenizer.find_token("<end_of_turn>");
            if (start_of_turn_id_ < 0 || end_of_turn_id_ < 0) {
                IMP_LOG_WARN("Gemma template: missing special tokens "
                             "(start_of_turn=%d, end_of_turn=%d), falling back to raw",
                             start_of_turn_id_, end_of_turn_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(end_of_turn_id_);
            stop_token_ids_.push_back(static_cast<int32_t>(tokenizer.eos_id()));

            // Vision tokens (optional — only present in Gemma-3 multimodal)
            boi_id_ = tokenizer.find_token("<start_of_image>");
            eoi_id_ = tokenizer.find_token("<end_of_image>");
            img_soft_token_id_ = tokenizer.find_token("<image_soft_token>");
            // Fall back to well-known Gemma-3 IDs
            if (boi_id_ < 0) boi_id_ = 255999;
            if (eoi_id_ < 0) eoi_id_ = 256000;
            if (img_soft_token_id_ < 0) img_soft_token_id_ = 262144;
            break;
        }
        case ChatTemplateFamily::DEEPSEEK_R1: {
            ds_user_id_      = tokenizer.find_token("<\xef\xbd\x9c" "User\xef\xbd\x9c>");
            ds_assistant_id_ = tokenizer.find_token("<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>");
            ds_eos_id_       = tokenizer.find_token("<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>");
            if (ds_user_id_ < 0 || ds_assistant_id_ < 0 || ds_eos_id_ < 0) {
                IMP_LOG_WARN("DeepSeek R1 template: missing tokens "
                             "(user=%d, asst=%d, eos=%d), falling back to raw",
                             ds_user_id_, ds_assistant_id_, ds_eos_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(ds_eos_id_);
            break;
        }
        case ChatTemplateFamily::PHI: {
            phi_user_id_      = tokenizer.find_token("<|user|>");
            phi_assistant_id_ = tokenizer.find_token("<|assistant|>");
            phi_end_id_       = tokenizer.find_token("<|end|>");
            if (phi_user_id_ < 0 || phi_assistant_id_ < 0 || phi_end_id_ < 0) {
                IMP_LOG_WARN("Phi template: missing tokens "
                             "(user=%d, asst=%d, end=%d), falling back to raw",
                             phi_user_id_, phi_assistant_id_, phi_end_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(phi_end_id_);
            break;
        }
        default:
            break;
    }

    // Extract default system message from Jinja template (if any).
    // Many models embed a default system prompt that's injected when the user
    // doesn't provide one (e.g. Nanbeige, Qwen).
    const std::string& jinja = tokenizer.chat_template_str();
    if (!jinja.empty()) {
        const std::string sys_prefix = "<|im_start|>system\n";
        size_t pos = 0;
        while ((pos = jinja.find(sys_prefix, pos)) != std::string::npos) {
            size_t content_start = pos + sys_prefix.size();
            size_t content_end = jinja.find("<|im_end|>", content_start);
            if (content_end == std::string::npos) break;

            std::string candidate = jinja.substr(content_start, content_end - content_start);
            // Skip entries that reference Jinja variables (user-provided messages)
            if (candidate.find("messages") == std::string::npos &&
                candidate.find("{{") == std::string::npos &&
                candidate.find("content") == std::string::npos &&
                !candidate.empty()) {
                default_system_message_ = candidate;
                IMP_LOG_INFO("Default system message: %.40s%s",
                             default_system_message_.c_str(),
                             default_system_message_.size() > 40 ? "..." : "");
                break;
            }
            pos = content_end;
        }
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
        case ChatTemplateFamily::NEMOTRON:    return apply_nemotron(tok, messages);
        case ChatTemplateFamily::GEMMA:       return apply_gemma(tok, messages);
        case ChatTemplateFamily::DEEPSEEK_R1: return apply_deepseek_r1(tok, messages);
        case ChatTemplateFamily::PHI:          return apply_phi(tok, messages);
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

    // Skip BOS if it's the same token as im_start (e.g. Nanbeige: bos = <|im_start|>)
    if (tok.add_bos() && bos_id_ != im_start_id_) {
        tokens.push_back(bos_id_);
    }

    // Inject default system message if the model has one and the user didn't provide one
    bool has_system = false;
    for (const auto& m : msgs) {
        if (m.role == "system") { has_system = true; break; }
    }
    if (!has_system && !default_system_message_.empty()) {
        tokens.push_back(im_start_id_);
        // Encode role+content as one piece to match reference tokenization
        auto sys_ids = tok.encode("system\n" + default_system_message_);
        tokens.insert(tokens.end(), sys_ids.begin(), sys_ids.end());
        tokens.push_back(im_end_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    for (const auto& msg : msgs) {
        tokens.push_back(im_start_id_);
        auto role_ids = tok.encode(msg.role + "\n");
        tokens.insert(tokens.end(), role_ids.begin(), role_ids.end());
        // Content follows role+newline within the same text piece — skip SPM ▁ prefix
        auto content_ids = tok.encode(msg.content, /*no_prefix=*/true);
        tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        tokens.push_back(im_end_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Assistant generation prefix
    tokens.push_back(im_start_id_);
    auto asst_ids = tok.encode("assistant\n");
    tokens.insert(tokens.end(), asst_ids.begin(), asst_ids.end());

    if (static const bool dbg_tpl = (getenv("IMP_DEBUG_TEMPLATE") != nullptr); dbg_tpl) {
        fprintf(stderr, "[DEBUG_TPL] chatml %zu tokens:", tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
    }

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

// Gemma: <start_of_turn>user\ncontent<end_of_turn>\n<start_of_turn>model\n
// Note: Gemma uses "model" instead of "assistant" for the AI role.
std::vector<int32_t> ChatTemplate::apply_gemma(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    for (const auto& msg : msgs) {
        tokens.push_back(start_of_turn_id_);
        // Gemma uses "model" for the assistant role
        std::string role = (msg.role == "assistant") ? "model" : msg.role;
        auto content_ids = tok.encode(role + "\n" + msg.content);
        tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        tokens.push_back(end_of_turn_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Model generation prefix
    tokens.push_back(start_of_turn_id_);
    auto model_ids = tok.encode("model\n");
    tokens.insert(tokens.end(), model_ids.begin(), model_ids.end());

    // Debug: print template token IDs
    if (static const bool dbg_tpl = (getenv("IMP_DEBUG_TEMPLATE") != nullptr); dbg_tpl) {
        fprintf(stderr, "[DEBUG_TPL] %zu tokens:", tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
    }

    return tokens;
}

// DeepSeek R1: {bos}{system}<｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
std::vector<int32_t> ChatTemplate::apply_deepseek_r1(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;
    tokens.push_back(bos_id_);

    // System message (if any) goes right after BOS as plain text
    for (const auto& msg : msgs) {
        if (msg.role == "system") {
            auto sys_ids = tok.encode(msg.content);
            tokens.insert(tokens.end(), sys_ids.begin(), sys_ids.end());
        }
    }

    // User/assistant turns
    for (const auto& msg : msgs) {
        if (msg.role == "user") {
            tokens.push_back(ds_user_id_);
            auto content_ids = tok.encode(msg.content);
            tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        } else if (msg.role == "assistant") {
            tokens.push_back(ds_assistant_id_);
            auto content_ids = tok.encode(msg.content);
            tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
            tokens.push_back(ds_eos_id_);
        }
    }

    // Assistant generation prefix
    tokens.push_back(ds_assistant_id_);

    return tokens;
}

// Phi: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>\n ... <|assistant|>\n
std::vector<int32_t> ChatTemplate::apply_phi(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& msgs) const
{
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    for (const auto& msg : msgs) {
        if (msg.role == "user") {
            tokens.push_back(phi_user_id_);
        } else if (msg.role == "assistant") {
            tokens.push_back(phi_assistant_id_);
        } else if (msg.role == "system") {
            tokens.push_back(phi_user_id_);
        }
        // Newline after role token, then content
        auto content_ids = tok.encode("\n" + msg.content);
        tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        tokens.push_back(phi_end_id_);
        // Newline after <|end|>
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Assistant generation prefix with trailing newline
    tokens.push_back(phi_assistant_id_);
    auto nl_ids = tok.encode("\n");
    tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());

    // Debug: print template token IDs
    if (static const bool dbg_tpl = (getenv("IMP_DEBUG_TEMPLATE") != nullptr); dbg_tpl) {
        fprintf(stderr, "[DEBUG_TPL] phi %zu tokens:", tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
        // Also decode back to text for verification
        fprintf(stderr, "[DEBUG_TPL] decoded: ");
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = tok.decode_token(tokens[i]);
            // Escape control chars for readability
            for (char c : piece) {
                if (c == '\n') fprintf(stderr, "\\n");
                else if (c == '\r') fprintf(stderr, "\\r");
                else fputc(c, stderr);
            }
            fprintf(stderr, "|");
        }
        fprintf(stderr, "\n");
    }

    return tokens;
}

std::vector<int32_t> ChatTemplate::apply_with_image(
    const Tokenizer& tok,
    const std::vector<ChatMessage>& messages,
    int n_image_tokens) const
{
    // Currently only Gemma family supports vision tokens.
    // For other families, fall back to text-only apply.
    if (family_ != ChatTemplateFamily::GEMMA ||
        boi_id_ < 0 || eoi_id_ < 0 || img_soft_token_id_ < 0) {
        return apply(tok, messages);
    }

    // Gemma vision format:
    // <bos><start_of_turn>user\n<boi><img_soft>*N<eoi>\n{text}<end_of_turn>\n<start_of_turn>model\n
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    for (size_t mi = 0; mi < messages.size(); mi++) {
        const auto& msg = messages[mi];
        tokens.push_back(start_of_turn_id_);

        std::string role = (msg.role == "assistant") ? "model" : msg.role;

        if (mi == 0 && msg.role == "user") {
            // First user message: inject image tokens before text
            auto role_ids = tok.encode(role + "\n");
            tokens.insert(tokens.end(), role_ids.begin(), role_ids.end());

            // Image token block: <boi> <img_soft>*N <eoi> \n
            tokens.push_back(boi_id_);
            for (int i = 0; i < n_image_tokens; i++)
                tokens.push_back(img_soft_token_id_);
            tokens.push_back(eoi_id_);

            auto text_ids = tok.encode("\n" + msg.content);
            tokens.insert(tokens.end(), text_ids.begin(), text_ids.end());
        } else {
            auto content_ids = tok.encode(role + "\n" + msg.content);
            tokens.insert(tokens.end(), content_ids.begin(), content_ids.end());
        }

        tokens.push_back(end_of_turn_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    // Model generation prefix
    tokens.push_back(start_of_turn_id_);
    auto model_ids = tok.encode("model\n");
    tokens.insert(tokens.end(), model_ids.begin(), model_ids.end());

    return tokens;
}

} // namespace imp

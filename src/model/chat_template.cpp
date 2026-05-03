#include "model/chat_template.h"
#include "core/logging.h"
#include "runtime/config.h"

#include <algorithm>
#include <functional>

namespace imp {

const char* chat_template_family_name(ChatTemplateFamily family) {
    switch (family) {
        case ChatTemplateFamily::RAW:
            return "raw";
        case ChatTemplateFamily::CHATML:
            return "chatml";
        case ChatTemplateFamily::LLAMA2:
            return "llama2";
        case ChatTemplateFamily::MISTRAL_V3:
            return "mistral_v3";
        case ChatTemplateFamily::LLAMA3:
            return "llama3";
        case ChatTemplateFamily::NEMOTRON:
            return "nemotron";
        case ChatTemplateFamily::GEMMA:
            return "gemma";
        case ChatTemplateFamily::DEEPSEEK_R1:
            return "deepseek_r1";
        case ChatTemplateFamily::PHI:
            return "phi";
    }
    return "unknown";
}

ChatTemplateFamily ChatTemplate::detect_family(const std::string& jinja2_str) {
    if (jinja2_str.empty())
        return ChatTemplateFamily::RAW;

    // Order matters: check more specific patterns first
    if (jinja2_str.find("<|im_start|>") != std::string::npos)
        return ChatTemplateFamily::CHATML;
    if (jinja2_str.find("<|start_header_id|>") != std::string::npos)
        return ChatTemplateFamily::LLAMA3;
    if (jinja2_str.find("<start_of_turn>") != std::string::npos)
        return ChatTemplateFamily::GEMMA;
    // Gemma-4 uses <|turn> instead of <start_of_turn>
    if (jinja2_str.find("<|turn>") != std::string::npos)
        return ChatTemplateFamily::GEMMA;
    // Mistral V3-Tekken (Mistral-Small-3.x, Mistral-Nemo, Mixtral-8x22B):
    // adds [TOOL_CALLS] / [AVAILABLE_TOOLS] / [TOOL_RESULTS] markers on top of
    // the V1/V2 [INST] core. Check BEFORE the LLAMA2 [INST] fallback so newer
    // Mistrals don't get misclassified as the older family.
    if (jinja2_str.find("[TOOL_CALLS]") != std::string::npos ||
        jinja2_str.find("[AVAILABLE_TOOLS]") != std::string::npos)
        return ChatTemplateFamily::MISTRAL_V3;
    if (jinja2_str.find("[INST]") != std::string::npos)
        return ChatTemplateFamily::LLAMA2;
    if (jinja2_str.find("<extra_id_0>") != std::string::npos)
        return ChatTemplateFamily::NEMOTRON;
    // DeepSeek R1: fullwidth vertical bars ｜ (U+FF5C = \xef\xbd\x9c)
    if (jinja2_str.find("\xef\xbd\x9c"
                        "User"
                        "\xef\xbd\x9c") != std::string::npos)
        return ChatTemplateFamily::DEEPSEEK_R1;
    // Phi: <|end|> is literal in the Jinja2 template (role tags are dynamic)
    if (jinja2_str.find("<|end|>") != std::string::npos)
        return ChatTemplateFamily::PHI;

    return ChatTemplateFamily::RAW;
}

ChatTemplateFamily ChatTemplate::default_family_for_arch(ModelArch arch) {
    switch (arch) {
        case ModelArch::LLAMA:
            return ChatTemplateFamily::LLAMA3;
        case ModelArch::MISTRAL:
            return ChatTemplateFamily::LLAMA2;
        case ModelArch::MIXTRAL:
            return ChatTemplateFamily::LLAMA2;
        case ModelArch::DEEPSEEK:
            return ChatTemplateFamily::DEEPSEEK_R1;
        case ModelArch::NEMOTRON_H_MOE:
            return ChatTemplateFamily::NEMOTRON;
        case ModelArch::QWEN3:
            return ChatTemplateFamily::CHATML;
        case ModelArch::QWEN3_MOE:
            return ChatTemplateFamily::CHATML;
        case ModelArch::QWEN35:
            return ChatTemplateFamily::CHATML;
        case ModelArch::QWEN35_MOE:
            return ChatTemplateFamily::CHATML;
        case ModelArch::QWEN36_MOE:
            return ChatTemplateFamily::CHATML;
        case ModelArch::GEMMA3:
            return ChatTemplateFamily::GEMMA;
        case ModelArch::GEMMA4:
            return ChatTemplateFamily::GEMMA;
        case ModelArch::LLAMA4:
            return ChatTemplateFamily::LLAMA3;
        default:
            return ChatTemplateFamily::RAW;
    }
}

ChatTemplateFamily ChatTemplate::parse_family(const std::string& name) {
    if (name == "auto")
        return ChatTemplateFamily::RAW;  // caller handles auto
    if (name == "none")
        return ChatTemplateFamily::RAW;
    if (name == "chatml")
        return ChatTemplateFamily::CHATML;
    if (name == "llama2")
        return ChatTemplateFamily::LLAMA2;
    if (name == "mistral_v3" || name == "mistral-v3" || name == "mistralv3")
        return ChatTemplateFamily::MISTRAL_V3;
    if (name == "llama3")
        return ChatTemplateFamily::LLAMA3;
    if (name == "nemotron")
        return ChatTemplateFamily::NEMOTRON;
    if (name == "gemma")
        return ChatTemplateFamily::GEMMA;
    if (name == "deepseek_r1" || name == "deepseek-r1")
        return ChatTemplateFamily::DEEPSEEK_R1;
    if (name == "phi")
        return ChatTemplateFamily::PHI;
    return ChatTemplateFamily::RAW;
}

bool ChatTemplate::init(ChatTemplateFamily family, const Tokenizer& tokenizer, const std::string& jinja_str) {
    family_ = family;
    stop_token_ids_.clear();
    use_jinja_ = false;

    // Try Jinja2 rendering if template string provided
    if (!jinja_str.empty()) {
        auto tpl = std::make_shared<jinja::Template>();
        if (tpl->parse(jinja_str)) {
            jinja_tpl_ = std::move(tpl);
            use_jinja_ = true;
            build_control_token_map(tokenizer);
            IMP_LOG_INFO("Chat template: using Jinja2 engine");
        } else {
            IMP_LOG_WARN("Jinja2 parse failed (%s), falling back to hardcoded template",
                         tpl->error().c_str());
        }
    }

    if (family_ == ChatTemplateFamily::RAW && !use_jinja_) {
        return true;
    }

    bos_id_ = static_cast<int32_t>(tokenizer.bos_id());

    switch (family_) {
        case ChatTemplateFamily::CHATML: {
            im_start_id_ = tokenizer.find_token("<|im_start|>");
            im_end_id_ = tokenizer.find_token("<|im_end|>");
            if (im_start_id_ < 0 || im_end_id_ < 0) {
                IMP_LOG_WARN(
                    "ChatML template: missing special tokens "
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
            end_header_id_ = tokenizer.find_token("<|end_header_id|>");
            eot_id_ = tokenizer.find_token("<|eot_id|>");
            if (start_header_id_ < 0 || end_header_id_ < 0 || eot_id_ < 0) {
                IMP_LOG_WARN(
                    "Llama3 template: missing special tokens "
                    "(start_header=%d, end_header=%d, eot=%d), falling back to raw",
                    start_header_id_, end_header_id_, eot_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(eot_id_);
            break;
        }
        case ChatTemplateFamily::LLAMA2:
        case ChatTemplateFamily::MISTRAL_V3: {
            // Both share the [INST]/[/INST] core. V3 adds tool-call markers
            // ([TOOL_CALLS], [AVAILABLE_TOOLS], [TOOL_RESULTS]) which are
            // emitted via the Jinja2 path when present in chat_template.jinja
            // — the hardcoded apply method only handles the message-frame.
            inst_start_id_ = tokenizer.find_token("[INST]");
            inst_end_id_ = tokenizer.find_token("[/INST]");
            if (inst_start_id_ < 0 || inst_end_id_ < 0) {
                IMP_LOG_WARN(
                    "Mistral/Llama2 template: missing special tokens "
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
                IMP_LOG_WARN(
                    "Nemotron template: missing special tokens "
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
            end_of_turn_id_ = tokenizer.find_token("<end_of_turn>");
            // Gemma-4 uses different token names: <|turn> / <turn|>
            if (start_of_turn_id_ < 0)
                start_of_turn_id_ = tokenizer.find_token("<|turn>");
            if (end_of_turn_id_ < 0)
                end_of_turn_id_ = tokenizer.find_token("<turn|>");
            if (start_of_turn_id_ < 0 || end_of_turn_id_ < 0) {
                IMP_LOG_WARN(
                    "Gemma template: missing special tokens "
                    "(start_of_turn=%d, end_of_turn=%d), falling back to raw",
                    start_of_turn_id_, end_of_turn_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(end_of_turn_id_);
            stop_token_ids_.push_back(static_cast<int32_t>(tokenizer.eos_id()));

            // Vision tokens (optional — only present in Gemma-3 multimodal)
            // Resolved from vocabulary; stays -1 if not found (disables vision).
            boi_id_ = tokenizer.find_token("<start_of_image>");
            eoi_id_ = tokenizer.find_token("<end_of_image>");
            img_soft_token_id_ = tokenizer.find_token("<image_soft_token>");
            break;
        }
        case ChatTemplateFamily::DEEPSEEK_R1: {
            ds_user_id_ = tokenizer.find_token(
                "<\xef\xbd\x9c"
                "User\xef\xbd\x9c>");
            ds_assistant_id_ = tokenizer.find_token(
                "<\xef\xbd\x9c"
                "Assistant\xef\xbd\x9c>");
            ds_eos_id_ = tokenizer.find_token(
                "<\xef\xbd\x9c"
                "end\xe2\x96\x81"
                "of\xe2\x96\x81"
                "sentence\xef\xbd\x9c>");
            if (ds_user_id_ < 0 || ds_assistant_id_ < 0 || ds_eos_id_ < 0) {
                IMP_LOG_WARN(
                    "DeepSeek R1 template: missing tokens "
                    "(user=%d, asst=%d, eos=%d), falling back to raw",
                    ds_user_id_, ds_assistant_id_, ds_eos_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            stop_token_ids_.push_back(ds_eos_id_);
            break;
        }
        case ChatTemplateFamily::PHI: {
            phi_user_id_ = tokenizer.find_token("<|user|>");
            phi_assistant_id_ = tokenizer.find_token("<|assistant|>");
            phi_end_id_ = tokenizer.find_token("<|end|>");
            if (phi_user_id_ < 0 || phi_assistant_id_ < 0 || phi_end_id_ < 0) {
                IMP_LOG_WARN(
                    "Phi template: missing tokens "
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
            if (content_end == std::string::npos)
                break;

            std::string candidate = jinja.substr(content_start, content_end - content_start);
            // Skip entries that reference Jinja variables (user-provided messages)
            if (candidate.find("messages") == std::string::npos &&
                candidate.find("{{") == std::string::npos && candidate.find("content") == std::string::npos &&
                !candidate.empty()) {
                default_system_message_ = candidate;
                IMP_LOG_INFO("Default system message: %.40s%s", default_system_message_.c_str(),
                             default_system_message_.size() > 40 ? "..." : "");
                break;
            }
            pos = content_end;
        }
    }

    IMP_LOG_INFO("Chat template: %s", chat_template_family_name(family_));
    return true;
}

// Honor `tokenizer_config.json::use_default_system_prompt: false`. When the
// model author opts out and the caller didn't provide an explicit system
// message, prepend an empty one so the Jinja template's "if no system →
// inject default_system_message" branch doesn't fire (Mistral-Small-3.2
// otherwise auto-injects ~600 tokens of boilerplate).
static std::vector<ChatMessage> maybe_suppress_default_system(const Tokenizer& tok,
                                                              const std::vector<ChatMessage>& messages) {
    if (tok.use_default_system_prompt())
        return messages;
    if (!messages.empty() && messages.front().role == "system")
        return messages;
    std::vector<ChatMessage> out;
    out.reserve(messages.size() + 1);
    out.push_back({"system", ""});
    for (const auto& m : messages)
        out.push_back(m);
    return out;
}

std::vector<int32_t> ChatTemplate::apply(const Tokenizer& tok, const std::vector<ChatMessage>& messages,
                                         bool suppress_thinking) const {
    auto eff_msgs = maybe_suppress_default_system(tok, messages);
    // Prefer Jinja2 rendering when available (data-driven from GGUF).
    // Falls back to hardcoded families if Jinja rendering fails.
    if (use_jinja_ && jinja_tpl_) {
        auto tokens = apply_jinja(tok, eff_msgs, true, suppress_thinking);
        if (!tokens.empty())
            return tokens;
        IMP_LOG_WARN("Jinja2 render produced empty result, falling back to hardcoded template");
    }

    switch (family_) {
        case ChatTemplateFamily::CHATML:
            return apply_chatml(tok, eff_msgs, suppress_thinking);
        case ChatTemplateFamily::LLAMA3:
            return apply_llama3(tok, eff_msgs);
        case ChatTemplateFamily::LLAMA2:
        case ChatTemplateFamily::MISTRAL_V3:
            return apply_llama2(tok, eff_msgs);
        case ChatTemplateFamily::NEMOTRON:
            return apply_nemotron(tok, eff_msgs);
        case ChatTemplateFamily::GEMMA:
            return apply_gemma(tok, eff_msgs);
        case ChatTemplateFamily::DEEPSEEK_R1:
            return apply_deepseek_r1(tok, eff_msgs);
        case ChatTemplateFamily::PHI:
            return apply_phi(tok, eff_msgs);
        default:
            break;
    }
    return {};
}

// ChatML: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
std::vector<int32_t> ChatTemplate::apply_chatml(const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                                bool suppress_thinking) const {
    std::vector<int32_t> tokens;

    // Skip BOS if it's the same token as im_start (e.g. Nanbeige: bos = <|im_start|>)
    if (tok.add_bos() && bos_id_ != im_start_id_) {
        tokens.push_back(bos_id_);
    }

    // When suppress_thinking is set, inject /no_think into the system message.
    // Qwen3 models respect this directive to skip the thinking phase entirely.
    // Build effective messages with /no_think appended to system message.
    std::vector<ChatMessage> effective_msgs;
    const std::vector<ChatMessage>* msgs_ptr = &msgs;
    if (suppress_thinking) {
        effective_msgs = msgs;
        bool found_system = false;
        for (auto& m : effective_msgs) {
            if (m.role == "system") {
                m.content += " /no_think";
                found_system = true;
                break;
            }
        }
        if (!found_system) {
            effective_msgs.insert(effective_msgs.begin(), ChatMessage{"system", "/no_think"});
        }
        msgs_ptr = &effective_msgs;
    }
    const auto& messages = *msgs_ptr;

    // Inject default system message if the model has one and the user didn't provide one
    bool has_system = false;
    for (const auto& m : messages) {
        if (m.role == "system") {
            has_system = true;
            break;
        }
    }
    if (!has_system && !default_system_message_.empty()) {
        std::string sys_content = default_system_message_;
        if (suppress_thinking)
            sys_content += " /no_think";
        tokens.push_back(im_start_id_);
        // Encode role+content as one piece to match reference tokenization
        auto sys_ids = tok.encode("system\n" + sys_content);
        tokens.insert(tokens.end(), sys_ids.begin(), sys_ids.end());
        tokens.push_back(im_end_id_);
        auto nl_ids = tok.encode("\n");
        tokens.insert(tokens.end(), nl_ids.begin(), nl_ids.end());
    }

    for (const auto& msg : messages) {
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

    if (RuntimeConfig::current().diagnostics.debug_template) {
        fprintf(stderr, "[DEBUG_TPL] chatml %zu tokens:", tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
    }

    return tokens;
}

// Llama3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|> ...
std::vector<int32_t> ChatTemplate::apply_llama3(const Tokenizer& tok,
                                                const std::vector<ChatMessage>& msgs) const {
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
std::vector<int32_t> ChatTemplate::apply_llama2(const Tokenizer& tok,
                                                const std::vector<ChatMessage>& msgs) const {
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
        if (msg.role == "system")
            continue;

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

// Nemotron:
// <extra_id_0>System\ncontent\n<extra_id_1>\n<extra_id_0>User\ncontent\n<extra_id_1>\n<extra_id_0>Assistant\n
std::vector<int32_t> ChatTemplate::apply_nemotron(const Tokenizer& tok,
                                                  const std::vector<ChatMessage>& msgs) const {
    std::vector<int32_t> tokens;

    if (tok.add_bos()) {
        tokens.push_back(bos_id_);
    }

    // Capitalize role names for Nemotron format
    auto capitalize = [](const std::string& s) -> std::string {
        if (s.empty())
            return s;
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
std::vector<int32_t> ChatTemplate::apply_gemma(const Tokenizer& tok,
                                               const std::vector<ChatMessage>& msgs) const {
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
    if (RuntimeConfig::current().diagnostics.debug_template) {
        fprintf(stderr, "[DEBUG_TPL] %zu tokens:", tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            fprintf(stderr, " %d", tokens[i]);
        fprintf(stderr, "\n");
    }

    return tokens;
}

// DeepSeek R1: {bos}{system}<｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
std::vector<int32_t> ChatTemplate::apply_deepseek_r1(const Tokenizer& tok,
                                                     const std::vector<ChatMessage>& msgs) const {
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
std::vector<int32_t> ChatTemplate::apply_phi(const Tokenizer& tok,
                                             const std::vector<ChatMessage>& msgs) const {
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
    if (RuntimeConfig::current().diagnostics.debug_template) {
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
                if (c == '\n')
                    fprintf(stderr, "\\n");
                else if (c == '\r')
                    fprintf(stderr, "\\r");
                else
                    fputc(c, stderr);
            }
            fprintf(stderr, "|");
        }
        fprintf(stderr, "\n");
    }

    return tokens;
}

std::vector<int32_t> ChatTemplate::apply_with_image(const Tokenizer& tok,
                                                    const std::vector<ChatMessage>& messages,
                                                    int n_image_tokens, bool suppress_thinking) const {
    // Currently only Gemma family supports vision tokens.
    // For other families, fall back to text-only apply.
    if (family_ != ChatTemplateFamily::GEMMA || boi_id_ < 0 || eoi_id_ < 0 || img_soft_token_id_ < 0) {
        return apply(tok, messages, suppress_thinking);
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

// ---------------------------------------------------------------------------
// Jinja2-based rendering
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared helper: split rendered Jinja2 output on control tokens and encode
// ---------------------------------------------------------------------------

std::vector<int32_t> ChatTemplate::tokenize_rendered(const Tokenizer& tok,
                                                     const std::string& rendered) const {
    // Split rendered string into tokens by finding control token boundaries.
    // Control tokens appear as literal text in the rendered output (e.g. "<|im_start|>").
    // We identify them via the control_tokens_ map (sorted longest-first).
    std::vector<int32_t> result;
    // Skip BOS if the rendered string already contains the BOS token text —
    // the control token splitter below will add it. Adding it here would duplicate.
    bool rendered_has_bos = false;
    if (bos_id_ >= 0) {
        const std::string& bos_text = tok.token_text(bos_id_);
        if (!bos_text.empty() && rendered.find(bos_text) != std::string::npos)
            rendered_has_bos = true;
    }
    if (tok.add_bos() && !rendered_has_bos) {
        result.push_back(bos_id_);
    }

    size_t pos = 0;
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

// ---------------------------------------------------------------------------
// Build Jinja2 messages array from ChatMessages
// ---------------------------------------------------------------------------

static jinja::Value::Array build_jinja_messages(const std::vector<ChatMessage>& msgs,
                                                bool suppress_thinking) {
    jinja::Value::Array msg_arr;
    for (const auto& m : msgs) {
        std::string content = m.content;
        if (suppress_thinking && m.role == "system") {
            content += " /no_think";
        }
        msg_arr.push_back(jinja::Value::object({
            {"role", jinja::Value(m.role)},
            {"content", jinja::Value(content)},
        }));
    }
    return msg_arr;
}

// ---------------------------------------------------------------------------
// Parse a JSON string into a jinja::Value (recursive)
// ---------------------------------------------------------------------------

static jinja::Value json_string_to_value(const std::string& json_str) {
    // Minimal JSON parser for tool parameter schemas.
    // Handles: objects, arrays, strings, numbers, booleans, null.
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' ||
                                         json_str[pos] == '\n' || json_str[pos] == '\r'))
            pos++;
    };

    std::function<jinja::Value()> parse_value;

    auto parse_string = [&]() -> std::string {
        if (pos >= json_str.size() || json_str[pos] != '"')
            return "";
        pos++;  // skip opening "
        std::string result;
        while (pos < json_str.size() && json_str[pos] != '"') {
            if (json_str[pos] == '\\' && pos + 1 < json_str.size()) {
                pos++;
                switch (json_str[pos]) {
                    case '"':
                        result += '"';
                        break;
                    case '\\':
                        result += '\\';
                        break;
                    case '/':
                        result += '/';
                        break;
                    case 'n':
                        result += '\n';
                        break;
                    case 't':
                        result += '\t';
                        break;
                    case 'r':
                        result += '\r';
                        break;
                    default:
                        result += json_str[pos];
                        break;
                }
            } else {
                result += json_str[pos];
            }
            pos++;
        }
        if (pos < json_str.size())
            pos++;  // skip closing "
        return result;
    };

    auto parse_object = [&]() -> jinja::Value {
        pos++;  // skip {
        auto obj = jinja::Value::make_object();
        skip_ws();
        if (pos < json_str.size() && json_str[pos] == '}') {
            pos++;
            return obj;
        }
        while (pos < json_str.size()) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (pos < json_str.size() && json_str[pos] == ':')
                pos++;
            skip_ws();
            obj.set(key, parse_value());
            skip_ws();
            if (pos < json_str.size() && json_str[pos] == ',') {
                pos++;
                continue;
            }
            if (pos < json_str.size() && json_str[pos] == '}') {
                pos++;
                break;
            }
            break;
        }
        return obj;
    };

    auto parse_array = [&]() -> jinja::Value {
        pos++;  // skip [
        jinja::Value::Array arr;
        skip_ws();
        if (pos < json_str.size() && json_str[pos] == ']') {
            pos++;
            return jinja::Value(std::move(arr));
        }
        while (pos < json_str.size()) {
            skip_ws();
            arr.push_back(parse_value());
            skip_ws();
            if (pos < json_str.size() && json_str[pos] == ',') {
                pos++;
                continue;
            }
            if (pos < json_str.size() && json_str[pos] == ']') {
                pos++;
                break;
            }
            break;
        }
        return jinja::Value(std::move(arr));
    };

    parse_value = [&]() -> jinja::Value {
        skip_ws();
        if (pos >= json_str.size())
            return jinja::Value();
        char c = json_str[pos];
        if (c == '"')
            return jinja::Value(parse_string());
        if (c == '{')
            return parse_object();
        if (c == '[')
            return parse_array();
        if (c == 't' && json_str.compare(pos, 4, "true") == 0) {
            pos += 4;
            return jinja::Value(true);
        }
        if (c == 'f' && json_str.compare(pos, 5, "false") == 0) {
            pos += 5;
            return jinja::Value(false);
        }
        if (c == 'n' && json_str.compare(pos, 4, "null") == 0) {
            pos += 4;
            return jinja::Value();
        }
        // Number
        size_t start = pos;
        bool is_float = false;
        if (c == '-')
            pos++;
        while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9')
            pos++;
        if (pos < json_str.size() && json_str[pos] == '.') {
            is_float = true;
            pos++;
        }
        while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9')
            pos++;
        if (pos < json_str.size() && (json_str[pos] == 'e' || json_str[pos] == 'E')) {
            is_float = true;
            pos++;
            if (pos < json_str.size() && (json_str[pos] == '+' || json_str[pos] == '-'))
                pos++;
            while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9')
                pos++;
        }
        std::string num_str = json_str.substr(start, pos - start);
        if (num_str.empty())
            return jinja::Value();
        if (is_float)
            return jinja::Value(std::stod(num_str));
        return jinja::Value(static_cast<int64_t>(std::stoll(num_str)));
    };

    return parse_value();
}

// ---------------------------------------------------------------------------
// Auto-detect stop tokens from Jinja2 rendering
// ---------------------------------------------------------------------------

void ChatTemplate::auto_detect_stop_tokens(const jinja::Context& ctx) const {
    if (!stop_token_ids_.empty() || control_tokens_.empty())
        return;

    jinja::Context ctx_no_gen = ctx;
    ctx_no_gen["add_generation_prompt"] = jinja::Value(false);
    std::string rendered_no_gen = jinja_tpl_->render(ctx_no_gen);

    for (const auto& [text, id] : control_tokens_) {
        size_t last = rendered_no_gen.rfind(text);
        if (last != std::string::npos && last > rendered_no_gen.size() / 2) {
            const_cast<ChatTemplate*>(this)->stop_token_ids_.push_back(id);
            IMP_LOG_INFO("Jinja2: auto-detected stop token '%s' (id=%d)", text.c_str(), id);
            break;
        }
    }
}

std::vector<int32_t> ChatTemplate::apply_jinja(const Tokenizer& tok, const std::vector<ChatMessage>& msgs,
                                               bool add_generation_prompt, bool suppress_thinking) const {
    if (!jinja_tpl_)
        return {};

    // Build Jinja2 context
    jinja::Context ctx;
    ctx["messages"] = jinja::Value(build_jinja_messages(msgs, suppress_thinking));
    ctx["add_generation_prompt"] = jinja::Value(add_generation_prompt);
    // Only stamp `enable_thinking` when the caller is explicitly suppressing
    // thinking. Different model families pick OPPOSITE defaults when the
    // variable is undefined: Qwen3 / Qwen3.6 inject an open `<think>\n` and
    // expect the model to write reasoning; Gemma-4's it template injects a
    // pre-closed `<|channel>thought\n<channel|>` block (model writes the
    // answer directly). Forcing `enable_thinking=true` on every render
    // overrode Gemma-4's template default and put NVFP4 quants into a
    // verbose-think loop with no exit ("* Wait, I should...", "* Let's try
    // a simpler one:", repeating). Leaving the variable undefined for the
    // default case lets each template author's default win.
    if (suppress_thinking) {
        ctx["enable_thinking"] = jinja::Value(false);
    }
    ctx["bos_token"] = (bos_id_ >= 0) ? jinja::Value(tok.token_text(bos_id_)) : jinja::Value(std::string(""));
    ctx["eos_token"] = jinja::Value(tok.token_text(tok.eos_id()));

    // Render
    std::string rendered = jinja_tpl_->render(ctx);
    if (rendered.empty()) {
        IMP_LOG_WARN("Jinja2 render returned empty string (error: %s)", jinja_tpl_->error().c_str());
        return {};
    }
    IMP_LOG_DEBUG("Jinja2 rendered (%zu chars)", rendered.size());
    if (RuntimeConfig::current().diagnostics.debug_template) {
        std::string escaped;
        for (char c : rendered) {
            if (c == '\n')
                escaped += "\\n";
            else
                escaped += c;
        }
        fprintf(stderr, "[DEBUG_TPL_JINJA] rendered: \"%s\"\n", escaped.c_str());
    }

    auto result = tokenize_rendered(tok, rendered);

    // Auto-detect stop tokens if needed
    auto_detect_stop_tokens(ctx);

    return result;
}

// ---------------------------------------------------------------------------
// Jinja2 rendering with tool definitions in context
// ---------------------------------------------------------------------------

std::vector<int32_t> ChatTemplate::apply_jinja_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& msgs, const std::vector<ToolFunction>& tools,
    const std::string& tool_choice, bool add_generation_prompt, bool suppress_thinking) const {
    if (!jinja_tpl_)
        return {};

    // Build tools array as Jinja2 values (OpenAI format: {type, function: {name, description, parameters}})
    jinja::Value::Array tools_arr;
    for (const auto& t : tools) {
        // Parse parameters JSON into a proper Jinja2 object so templates can
        // traverse properties, use tojson, etc.
        jinja::Value params = t.parameters_json.empty() ? jinja::Value::make_object()
                                                        : json_string_to_value(t.parameters_json);

        tools_arr.push_back(jinja::Value::object({
            {"type", jinja::Value(std::string("function"))},
            {"function", jinja::Value::object({
                             {"name", jinja::Value(t.name)},
                             {"description", jinja::Value(t.description)},
                             {"parameters", std::move(params)},
                         })},
        }));
    }

    // Build context
    jinja::Context ctx;
    ctx["messages"] = jinja::Value(build_jinja_messages(msgs, suppress_thinking));
    ctx["tools"] = jinja::Value(std::move(tools_arr));
    ctx["tool_choice"] = jinja::Value(tool_choice);
    ctx["add_generation_prompt"] = jinja::Value(add_generation_prompt);
    // See apply_jinja for the defaults rationale; same logic for the
    // tools-aware path.
    if (suppress_thinking) {
        ctx["enable_thinking"] = jinja::Value(false);
    }
    ctx["bos_token"] = (bos_id_ >= 0) ? jinja::Value(tok.token_text(bos_id_)) : jinja::Value(std::string(""));
    ctx["eos_token"] = jinja::Value(tok.token_text(tok.eos_id()));

    // Render
    std::string rendered = jinja_tpl_->render(ctx);
    if (rendered.empty()) {
        IMP_LOG_WARN("Jinja2 tools render returned empty string (error: %s)", jinja_tpl_->error().c_str());
        return {};
    }
    IMP_LOG_DEBUG("Jinja2 tools rendered (%zu chars): %.200s", rendered.size(), rendered.c_str());

    auto result = tokenize_rendered(tok, rendered);

    // Auto-detect stop tokens if needed
    auto_detect_stop_tokens(ctx);

    return result;
}

// ---------------------------------------------------------------------------
// Public: apply_with_tools — try Jinja2 tools path, fallback to standard apply
// ---------------------------------------------------------------------------

std::vector<int32_t> ChatTemplate::apply_with_tools(const Tokenizer& tok,
                                                    const std::vector<ChatMessage>& messages,
                                                    const std::vector<ToolFunction>& tools,
                                                    const std::string& tool_choice,
                                                    bool suppress_thinking) const {
    // Try Jinja2 tools-aware path. Returns empty if Jinja2 is unavailable or
    // rendering fails, signaling the caller to fall back to text-based tool injection.
    if (use_jinja_ && jinja_tpl_ && !tools.empty()) {
        auto eff_msgs = maybe_suppress_default_system(tok, messages);
        auto tokens = apply_jinja_with_tools(tok, eff_msgs, tools, tool_choice, true, suppress_thinking);
        if (!tokens.empty())
            return tokens;
        IMP_LOG_WARN("Jinja2 tools render failed, caller should inject text-based tool prompt");
    }

    return {};
}

bool ChatTemplate::supports_tools() const { return use_jinja_ && jinja_tpl_ != nullptr; }

}  // namespace imp

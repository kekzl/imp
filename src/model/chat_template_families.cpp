#include "model/chat_template.h"
#include "core/logging.h"
#include "runtime/process_diag.h"

namespace {
// These debug dumps used to be a prefix fprintf, a loop of partial writes and a
// trailing newline — three calls the logging framework cannot express, which is
// exactly why they bypassed it. Build the line, then log it once.
std::string join_token_ids(const std::vector<int32_t>& t) {
    std::string s;
    for (int32_t id : t) {
        s += ' ';
        s += std::to_string(id);
    }
    return s;
}
}  // namespace

#include <algorithm>
#include <functional>

namespace imp {

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

    if (imp::process_diag_debug_template()) {
        IMP_LOG_DEBUG("[DEBUG_TPL] chatml %zu tokens:%s", tokens.size(), join_token_ids(tokens).c_str());
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
    if (imp::process_diag_debug_template()) {
        IMP_LOG_DEBUG("[DEBUG_TPL] %zu tokens:%s", tokens.size(), join_token_ids(tokens).c_str());
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
    if (imp::process_diag_debug_template()) {
        IMP_LOG_DEBUG("[DEBUG_TPL] phi %zu tokens:%s", tokens.size(), join_token_ids(tokens).c_str());
        // Also decode back to text for verification
        std::string decoded;
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = tok.decode_token(tokens[i]);
            // Escape control chars for readability
            for (char c : piece) {
                if (c == '\n')
                    decoded += "\\n";
                else if (c == '\r')
                    decoded += "\\r";
                else
                    decoded += c;
            }
            decoded += '|';
        }
        IMP_LOG_DEBUG("[DEBUG_TPL] decoded: %s", decoded.c_str());
    }

    return tokens;
}

// Harmony (gpt-oss): <|start|>role<|message|>content<|end|> blocks.
// System turn carries the channel declaration the model was trained on;
// a user-provided "system" message maps to the developer role per the
// Harmony spec. Assistant history is rendered as the final channel.
std::vector<int32_t> ChatTemplate::apply_harmony(const Tokenizer& tok,
                                                 const std::vector<ChatMessage>& msgs) const {
    std::vector<int32_t> tokens;

    auto push_text = [&](const std::string& text) {
        auto ids = tok.encode(text);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
    };
    auto open_role = [&](const char* role) {
        tokens.push_back(hm_start_id_);
        push_text(role);
    };

    // Fixed system turn (Harmony spec): identity + reasoning level + the
    // channel contract. The model relies on this to route analysis vs final.
    open_role("system");
    tokens.push_back(hm_message_id_);
    push_text(
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n\n"
        "Reasoning: medium\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.");
    tokens.push_back(hm_end_id_);

    for (const auto& msg : msgs) {
        if (msg.role == "system") {
            // User-supplied instructions → developer role (Harmony spec).
            open_role("developer");
            tokens.push_back(hm_message_id_);
            push_text("# Instructions\n\n" + msg.content);
            tokens.push_back(hm_end_id_);
        } else if (msg.role == "assistant") {
            open_role("assistant");
            tokens.push_back(hm_channel_id_);
            push_text("final");
            tokens.push_back(hm_message_id_);
            push_text(msg.content);
            tokens.push_back(hm_end_id_);
        } else {  // user (and any unknown role)
            open_role(msg.role == "user" ? "user" : msg.role.c_str());
            tokens.push_back(hm_message_id_);
            push_text(msg.content);
            tokens.push_back(hm_end_id_);
        }
    }

    // Generation prompt: the model emits "<|channel|>analysis<|message|>..."
    // itself, then "<|start|>assistant<|channel|>final<|message|>...<|return|>".
    open_role("assistant");

    if (imp::process_diag_debug_template()) {
        IMP_LOG_DEBUG("[DEBUG_TPL] harmony %zu tokens:%s", tokens.size(), join_token_ids(tokens).c_str());
    }

    return tokens;
}

std::vector<int32_t> ChatTemplate::apply_with_image(const Tokenizer& tok,
                                                    const std::vector<ChatMessage>& messages,
                                                    int n_image_tokens, bool suppress_thinking,
                                                    bool force_thinking,
                                                    const std::string& reasoning_effort) const {
    // Currently only Gemma family supports vision tokens.
    // For other families, fall back to text-only apply.
    if (family_ != ChatTemplateFamily::GEMMA || boi_id_ < 0 || eoi_id_ < 0 || img_soft_token_id_ < 0) {
        return apply(tok, messages, suppress_thinking, force_thinking, reasoning_effort);
    }

    // The image rides the FIRST USER turn — not message index 0. Keying it on
    // index 0 meant any request opening with a system prompt (the normal shape
    // for a pipeline) rendered text-only: the picture was still decoded and
    // encoded, but the prompt held no soft tokens for the embeddings to replace,
    // so the model answered fluently that it could not see an image (#1246).
    //
    // "First user turn" matches what the Qwen3-VL path does with its placeholder
    // blocks: the request parser keeps pictures in order but not which message
    // they came from, so this is the position that is known rather than guessed.
    size_t image_turn = messages.size();  // == none
    for (size_t mi = 0; mi < messages.size(); mi++) {
        if (messages[mi].role == "user") {
            image_turn = mi;
            break;
        }
    }
    // Nowhere to put it: fall back to the text-only render rather than drop the
    // block on an unrelated turn.
    if (image_turn == messages.size())
        return apply(tok, messages, suppress_thinking, force_thinking);

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

        if (mi == image_turn) {
            // The user turn that carries the picture.
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

}  // namespace imp

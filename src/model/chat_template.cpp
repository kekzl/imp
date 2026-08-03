#include "model/chat_template.h"
#include "core/logging.h"
#include "runtime/process_diag.h"

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
        case ChatTemplateFamily::HARMONY:
            return "harmony";
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
    // Harmony (gpt-oss): <|channel|> is unique to the format. Must be checked
    // BEFORE Phi — the Harmony template also contains the literal <|end|>.
    if (jinja2_str.find("<|channel|>") != std::string::npos)
        return ChatTemplateFamily::HARMONY;
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
        case ModelArch::GPT_OSS:
            return ChatTemplateFamily::HARMONY;
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
    if (name == "harmony")
        return ChatTemplateFamily::HARMONY;
    return ChatTemplateFamily::RAW;
}

// imp's minimal Jinja2 engine has no {% generation %}/{% endgeneration %} block
// tags — a HuggingFace chat-template extension that marks the assistant-response
// span for training-time loss masking (render-neutral). Phi-4-reasoning places
// them inside the assistant branch; the unknown opening tag derails block nesting
// so the trailing {% if add_generation_prompt %} is dropped → the assistant
// generation prompt is never appended → the model emits role markers as text.
// Strip the (render-neutral) tags before parsing.
static std::string strip_generation_tags(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '{' && i + 1 < s.size() && s[i + 1] == '%') {
            size_t close = s.find("%}", i + 2);
            if (close != std::string::npos) {
                std::string inner = s.substr(i + 2, close - (i + 2));
                size_t a = inner.find_first_not_of(" \t\r\n-");
                size_t b = inner.find_last_not_of(" \t\r\n-");
                std::string kw = (a == std::string::npos) ? std::string() : inner.substr(a, b - a + 1);
                if (kw == "generation" || kw == "endgeneration") {
                    i = close + 2;  // drop the whole tag, render its inner body as-is
                    continue;
                }
            }
        }
        out.push_back(s[i++]);
    }
    return out;
}

bool ChatTemplate::init(ChatTemplateFamily family, const Tokenizer& tokenizer, const std::string& jinja_str) {
    family_ = family;
    stop_token_ids_.clear();
    use_jinja_ = false;
    mentions_thinking_ = false;  // resolved below once the Jinja engine is up
    tool_xml_dialect_ = false;

    // Try Jinja2 rendering if template string provided
    if (!jinja_str.empty()) {
        std::string cleaned = strip_generation_tags(jinja_str);
        auto tpl = std::make_shared<jinja::Template>();
        if (tpl->parse(cleaned)) {
            jinja_tpl_ = std::move(tpl);
            use_jinja_ = true;
            build_control_token_map(tokenizer);
            IMP_LOG_INFO("Chat template: using Jinja2 engine");
            // Think evidence (drives the server-side thinking DEFAULT):
            //  a) the template exposes the `enable_thinking` switch
            //     (Qwen3 hybrid, Qwen3.6, Nemotron), or
            //  b) a fresh-conversation render emits "<think>" (Phi-4
            //     reasoning system prompt, DeepSeek-R1 generation prefix).
            // A raw substring match is NOT enough: Qwen3-*-Instruct-2507
            // mentions <think> only in branches that re-render PAST
            // assistant turns — for a fresh turn nothing thinks, and
            // defaulting it to thinking traps the whole answer in
            // reasoning_content.
            mentions_thinking_ = jinja_str.find("enable_thinking") != std::string::npos;
            if (!mentions_thinking_ && jinja_str.find("<think>") != std::string::npos)
                mentions_thinking_ = probe_render_mentions_think(tokenizer);
            // Qwen-Coder / Qwen3.6 XML tool-call dialect: the template teaches
            // <function=NAME><parameter=KEY> bodies inside <tool_call> (raw-text
            // values, not JSON). Constrained tool enforcement must use the XML
            // grammar on these templates — the JSON body FSM masks raw newlines
            // and mangles multi-line arguments. The source-substring hit is only
            // a prefilter; the probe render proves the RENDERED prompt actually
            // teaches the dialect (see probe_render_teaches_xml_tools).
            tool_xml_dialect_ = jinja_str.find("<parameter=") != std::string::npos &&
                                jinja_str.find("<function=") != std::string::npos &&
                                probe_render_teaches_xml_tools(tokenizer);
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

            // Vision tokens (optional — Gemma-3 and Gemma-4 multimodal).
            // Resolved from vocabulary; stays -1 if not found (disables vision).
            // Gemma-3: <start_of_image>/<image_soft_token>/<end_of_image>.
            // Gemma-4: <|image>/<|image|>/<image|> (begin / repeated soft / end).
            boi_id_ = tokenizer.find_token("<start_of_image>");
            if (boi_id_ < 0)
                boi_id_ = tokenizer.find_token("<|image>");
            eoi_id_ = tokenizer.find_token("<end_of_image>");
            if (eoi_id_ < 0)
                eoi_id_ = tokenizer.find_token("<image|>");
            img_soft_token_id_ = tokenizer.find_token("<image_soft_token>");
            if (img_soft_token_id_ < 0)
                img_soft_token_id_ = tokenizer.find_token("<|image|>");
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
        case ChatTemplateFamily::HARMONY: {
            hm_start_id_ = tokenizer.find_token("<|start|>");
            hm_end_id_ = tokenizer.find_token("<|end|>");
            hm_message_id_ = tokenizer.find_token("<|message|>");
            hm_channel_id_ = tokenizer.find_token("<|channel|>");
            hm_return_id_ = tokenizer.find_token("<|return|>");
            hm_call_id_ = tokenizer.find_token("<|call|>");
            if (hm_start_id_ < 0 || hm_end_id_ < 0 || hm_message_id_ < 0 || hm_channel_id_ < 0 ||
                hm_return_id_ < 0) {
                IMP_LOG_WARN(
                    "Harmony template: missing tokens "
                    "(start=%d, end=%d, message=%d, channel=%d, return=%d), falling back to raw",
                    hm_start_id_, hm_end_id_, hm_message_id_, hm_channel_id_, hm_return_id_);
                family_ = ChatTemplateFamily::RAW;
                return false;
            }
            // The final-channel answer ends with <|return|>; tool calls end
            // with <|call|>. <|end|> is deliberately NOT a stop token — it
            // separates the analysis message from the final message inside
            // one assistant turn (stopping there would truncate after the
            // reasoning block).
            stop_token_ids_.push_back(hm_return_id_);
            if (hm_call_id_ >= 0)
                stop_token_ids_.push_back(hm_call_id_);
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
                                         bool suppress_thinking, bool force_thinking) const {
    auto eff_msgs = maybe_suppress_default_system(tok, messages);
    // Prefer Jinja2 rendering when available (data-driven from GGUF).
    // Falls back to hardcoded families if Jinja rendering fails.
    if (use_jinja_ && jinja_tpl_) {
        auto tokens = apply_jinja(tok, eff_msgs, true, suppress_thinking, force_thinking);
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
        case ChatTemplateFamily::HARMONY:
            return apply_harmony(tok, eff_msgs);
        default:
            break;
    }
    return {};
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
                                               bool add_generation_prompt, bool suppress_thinking,
                                               bool force_thinking) const {
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
    // force_thinking stamps enable_thinking=true so a template that defaults the
    // variable to a pre-CLOSED block (Qwen3.5-4B) opens the block for an explicit
    // caller request — without it, `enable_thinking:true` was silently a no-op on
    // such templates. suppress wins if both are set.
    if (suppress_thinking) {
        ctx["enable_thinking"] = jinja::Value(false);
    } else if (force_thinking) {
        ctx["enable_thinking"] = jinja::Value(true);
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
    if (imp::process_diag_debug_template()) {
        std::string escaped;
        for (char c : rendered) {
            if (c == '\n')
                escaped += "\\n";
            else
                escaped += c;
        }
        IMP_LOG_DEBUG("[DEBUG_TPL_JINJA] rendered: \"%s\"", escaped.c_str());
    }

    auto result = tokenize_rendered(tok, rendered);

    // Auto-detect stop tokens if needed
    auto_detect_stop_tokens(ctx);

    return result;
}

// Render a minimal fresh conversation and report whether the generation
// prompt itself contains "<think>" — i.e. the template (not the model)
// enters reasoning mode on a new turn. See init() for why this exists.
bool ChatTemplate::probe_render_mentions_think(const Tokenizer& tok) const {
    if (!jinja_tpl_)
        return false;
    jinja::Context ctx;
    std::vector<ChatMessage> probe{{"user", "hi"}};
    ctx["messages"] = jinja::Value(build_jinja_messages(probe, /*suppress_thinking=*/false));
    ctx["add_generation_prompt"] = jinja::Value(true);
    ctx["bos_token"] = (bos_id_ >= 0) ? jinja::Value(tok.token_text(bos_id_)) : jinja::Value(std::string(""));
    ctx["eos_token"] = jinja::Value(tok.token_text(tok.eos_id()));
    std::string rendered = jinja_tpl_->render(ctx);
    return rendered.find("<think>") != std::string::npos;
}

// Render a dummy tools conversation and report whether the RENDERED prompt
// teaches the Qwen-Coder XML calling convention inside the ChatML
// <tool_call> envelope. A raw source-substring match is not evidence: a
// template may mention the markers in a comment, an example, or an untaken
// branch while actually prompting JSON bodies — and Seed-OSS-style templates
// pair the XML body with a non-<tool_call> envelope the enforcement's gate
// and forced literals would then contradict.
bool ChatTemplate::probe_render_teaches_xml_tools(const Tokenizer& tok) const {
    if (!jinja_tpl_)
        return false;
    jinja::Value::Array tools_arr;
    tools_arr.push_back(jinja::Value::object({
        {"type", jinja::Value(std::string("function"))},
        {"function",
         jinja::Value::object({
             {"name", jinja::Value(std::string("probe_fn"))},
             {"description", jinja::Value(std::string("probe"))},
             {"parameters",
              json_string_to_value(R"({"type":"object","properties":{"p":{"type":"string"}}})")},
         })},
    }));
    jinja::Context ctx;
    std::vector<ChatMessage> probe{{"user", "hi"}};
    ctx["messages"] = jinja::Value(build_jinja_messages(probe, /*suppress_thinking=*/false));
    ctx["tools"] = jinja::Value(std::move(tools_arr));
    ctx["tool_choice"] = jinja::Value(std::string("auto"));
    ctx["add_generation_prompt"] = jinja::Value(true);
    ctx["bos_token"] = (bos_id_ >= 0) ? jinja::Value(tok.token_text(bos_id_)) : jinja::Value(std::string(""));
    ctx["eos_token"] = jinja::Value(tok.token_text(tok.eos_id()));
    std::string rendered = jinja_tpl_->render(ctx);
    return rendered.find("<function=") != std::string::npos &&
           rendered.find("<parameter=") != std::string::npos &&
           rendered.find("<tool_call>") != std::string::npos;
}

// ---------------------------------------------------------------------------
// Jinja2 rendering with tool definitions in context
// ---------------------------------------------------------------------------

std::vector<int32_t> ChatTemplate::apply_jinja_with_tools(
    const Tokenizer& tok, const std::vector<ChatMessage>& msgs, const std::vector<ToolFunction>& tools,
    const std::string& tool_choice, bool add_generation_prompt, bool suppress_thinking,
    bool force_thinking) const {
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
    // tools-aware path. When tools are present, default to enable_thinking=true:
    // models like Gemma-4 emit tool_calls via a thought-channel-driven decision
    // (template auto-closes the channel with empty content when thinking is off,
    // which trains the model to skip tool selection and answer in plain text).
    if (suppress_thinking) {
        ctx["enable_thinking"] = jinja::Value(false);
    } else if (force_thinking || family_ == ChatTemplateFamily::GEMMA) {
        ctx["enable_thinking"] = jinja::Value(true);
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
                                                    bool suppress_thinking,
                                                    bool force_thinking) const {
    // Try Jinja2 tools-aware path. Returns empty if Jinja2 is unavailable or
    // rendering fails, signaling the caller to fall back to text-based tool injection.
    if (use_jinja_ && jinja_tpl_ && !tools.empty()) {
        auto eff_msgs = maybe_suppress_default_system(tok, messages);
        auto tokens =
            apply_jinja_with_tools(tok, eff_msgs, tools, tool_choice, true, suppress_thinking, force_thinking);
        if (!tokens.empty())
            return tokens;
        IMP_LOG_WARN("Jinja2 tools render failed, caller should inject text-based tool prompt");
    }

    return {};
}

bool ChatTemplate::supports_tools() const { return use_jinja_ && jinja_tpl_ != nullptr; }

}  // namespace imp

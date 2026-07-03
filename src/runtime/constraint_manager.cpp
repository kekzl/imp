#include "runtime/constraint_manager.h"

namespace imp {

namespace {

int32_t detect_think_close(Tokenizer* tokenizer) {
    if (!tokenizer)
        return -1;
    int32_t close = tokenizer->find_token("</think>");
    if (close < 0)
        return -1;
    if (tokenizer->find_token("<think>") < 0)
        return -1;
    return close;
}

struct ToolDialect {
    std::vector<int32_t> open_tokens;
    std::vector<int32_t> close_tokens;
    std::string open_prefix;
    std::string close_suffix;

    bool empty() const {
        return open_tokens.empty() && close_tokens.empty() && open_prefix.empty() &&
               close_suffix.empty();
    }
};

// Resolves dialect-specific tool tags into token IDs (where the vocab has them
// as single special tokens) plus char-level prefix/suffix fallbacks.
//
// ChatML/Hermes/Mistral: <tool_call>...</tool_call>  (single special tokens)
// Gemma:                 <|tool_call>...<tool_call|> (single special tokens)
// Llama3:                <function=...></function>   (multi-token, char fallback)
// Other families fall through to ChatML defaults.
ToolDialect resolve_tool_dialect(Tokenizer* tokenizer, ChatTemplateFamily family) {
    ToolDialect d;
    if (!tokenizer)
        return d;

    auto add_token_if_present = [&](const std::string& s, std::vector<int32_t>& out) {
        int32_t id = tokenizer->find_token(s);
        if (id >= 0)
            out.push_back(id);
    };

    switch (family) {
        case ChatTemplateFamily::LLAMA3:
            // <function=NAME> has dynamic NAME — char-prefix is the only path.
            d.open_prefix = "<function=";
            d.close_suffix = "</function>";
            return d;

        case ChatTemplateFamily::GEMMA:
            d.open_prefix = "<|tool_call>";
            d.close_suffix = "<tool_call|>";
            add_token_if_present("<|tool_call>", d.open_tokens);
            add_token_if_present("<tool_call|>", d.close_tokens);
            return d;

        case ChatTemplateFamily::CHATML:
        case ChatTemplateFamily::MISTRAL_V3:
        case ChatTemplateFamily::DEEPSEEK_R1:
        case ChatTemplateFamily::PHI:
        case ChatTemplateFamily::NEMOTRON:
        case ChatTemplateFamily::LLAMA2:
        case ChatTemplateFamily::RAW:
        default:
            d.open_prefix = "<tool_call>";
            d.close_suffix = "</tool_call>";
            add_token_if_present("<tool_call>", d.open_tokens);
            add_token_if_present("</tool_call>", d.close_tokens);
            return d;
    }
}

}  // namespace

void ConstraintManager::prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer,
                                bool has_tools, ChatTemplateFamily tpl_family, bool thinking_open) {
    active_json_ = false;
    active_schema_ = false;

    const int32_t think_close = detect_think_close(tokenizer);

    // The large think-close budget applies only when THIS REQUEST is actually
    // reasoning (thinking_open). Keying it off tokenizer capability alone gave
    // every request on a think-capable tokenizer an 8192-token unmasked
    // preamble — a non-thinking json_mode request never emits </think>, so the
    // grammar never engaged and the output was unconstrained garbage.
    //   - has_tools: 512-token slack. Think-capable models with thinking
    //     suppressed (json/tools requests) deliberate in PLAIN TEXT before
    //     opening the tool tag — Qwen3-8B spends 60-130 tokens of prose
    //     deciding to call get_weather. The old 64-token slack (sized for
    //     "Sure! "-style preambles) expired mid-sentence, and the schema
    //     mask then forced a contentless `{"answer":""}` INSTEAD of the
    //     tool call the model was about to make (#840). 512 covers real
    //     deliberation while still bounding a model that never produces
    //     structure.
    //   - no tools: 0 — the old 8-token "markdown fence" slack let the model
    //     open ```json fences that then leaked into content around otherwise
    //     perfect JSON. With the mask active from token 1 the model starts
    //     at '{' directly (whitespace is still legal in the START state).
    int preamble_budget;
    if (thinking_open && think_close >= 0) {
        preamble_budget = 8192;
    } else if (has_tools) {
        preamble_budget = 512;
    } else {
        preamble_budget = 0;
    }

    ToolDialect dialect;
    if (has_tools) {
        dialect = resolve_tool_dialect(tokenizer, tpl_family);
        if (dialect.empty()) {
            // Tokenizer surfaced none of the dialect tags AND the family had
            // no char fallback — degrade to current "drop schema" behaviour.
            IMP_LOG_INFO(
                "ConstraintManager: no tool-tag dialect for family %d, dropping schema/json_mode",
                static_cast<int>(tpl_family));
            return;
        }
    }

    auto configure_gate = [&](auto* constrainer) {
        if (has_tools) {
            constrainer->set_preamble_with_tools(think_close, preamble_budget,
                                                 dialect.open_tokens, dialect.close_tokens,
                                                 dialect.open_prefix, dialect.close_suffix, thinking_open);
        } else {
            constrainer->set_preamble(think_close, preamble_budget, thinking_open);
        }
    };

    // Free-form object schema ({"type":"object"} without properties/enum):
    // the schema FSM knows no legal key, rejects every token in the key
    // phase, and the empty-allow guard force-finishes right after "{".
    // Semantically this IS json_object — route to the any-JSON constrainer,
    // which is whole-token validated.
    bool use_schema = !json_schema.empty();
    if (use_schema) {
        auto probe = parse_json_schema(json_schema);
        const SchemaNode* probe_res = probe ? resolve_schema_ref(probe.get(), probe.get()) : nullptr;
        if (probe_res && probe_res->type == SchemaType::OBJECT && probe_res->properties.empty() &&
            probe_res->enum_values.empty()) {
            IMP_LOG_INFO("ConstraintManager: free-form object schema → any-JSON constrainer");
            use_schema = false;
            json_mode = true;
        }
    }

    if (use_schema) {
        if (schema_constrainer_ && schema_constrainer_->is_initialized() &&
            json_schema == cached_schema_string_) {
            configure_gate(schema_constrainer_.get());
            schema_constrainer_->reset();
            active_schema_ = true;
        } else {
            auto schema = parse_json_schema(json_schema);
            if (schema) {
                schema_constrainer_ = std::make_unique<SchemaConstrainer>();
                if (tokenizer && schema_constrainer_->init(*tokenizer, std::move(schema))) {
                    cached_schema_string_ = json_schema;
                    configure_gate(schema_constrainer_.get());
                    schema_constrainer_->reset();
                    active_schema_ = true;
                } else {
                    IMP_LOG_ERROR("Failed to initialize schema constrainer");
                    schema_constrainer_.reset();
                    cached_schema_string_.clear();
                }
            } else {
                IMP_LOG_ERROR("Failed to parse JSON schema");
            }
        }
        return;
    }

    if (json_mode) {
        if (!json_constrainer_) {
            json_constrainer_ = std::make_unique<JsonConstrainer>();
            if (!tokenizer || !json_constrainer_->init(*tokenizer)) {
                IMP_LOG_ERROR("Failed to initialize JSON constrainer");
                json_constrainer_.reset();
                return;
            }
        }
        configure_gate(json_constrainer_.get());
        json_constrainer_->reset();
        active_json_ = true;
    }
}

void ConstraintManager::update(int32_t token) {
    if (active_schema_ && schema_constrainer_) {
        schema_constrainer_->update(token);
    } else if (active_json_ && json_constrainer_) {
        json_constrainer_->update(token);
    }
}

int ConstraintManager::forced_text(std::string& out, int max_chars) const {
    if (active_schema_ && schema_constrainer_)
        return schema_constrainer_->forced_text(out, max_chars);
    out.clear();
    return 0;
}

void ConstraintManager::reset() {
    if (active_schema_ && schema_constrainer_) {
        schema_constrainer_->reset();
    } else if (active_json_ && json_constrainer_) {
        json_constrainer_->reset();
    }
    active_json_ = false;
    active_schema_ = false;
}

}  // namespace imp

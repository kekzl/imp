#include "runtime/constraint_manager.h"

namespace imp {

namespace {

// Look up the </think> token id (or -1 if the tokenizer has no thinking
// markers). Reasoning models — Qwen3.6, DeepSeek-R1, Gemma-4-thinking — all
// emit `<think>...</think>` before the actual answer; without a preamble
// gate, strict JSON enforcement would mask the very first token they want
// to sample.
int32_t detect_think_close(Tokenizer* tokenizer) {
    if (!tokenizer)
        return -1;
    int32_t close = tokenizer->find_token("</think>");
    if (close < 0)
        return -1;
    // Only enable preamble if the OPEN token also exists — otherwise the
    // tokenizer doesn't actually treat thinking as a structural element and
    // we'd risk swallowing legitimate output.
    if (tokenizer->find_token("<think>") < 0)
        return -1;
    return close;
}

}  // namespace

void ConstraintManager::prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer) {
    active_json_ = false;
    active_schema_ = false;

    const int32_t think_close = detect_think_close(tokenizer);

    // Schema mode takes priority over plain JSON mode
    if (!json_schema.empty()) {
        if (schema_constrainer_ && schema_constrainer_->is_initialized() &&
            json_schema == cached_schema_string_) {
            // Reuse cached constrainer — just reset FSM state.
            schema_constrainer_->set_preamble(think_close);
            schema_constrainer_->reset();
            active_schema_ = true;
        } else {
            auto schema = parse_json_schema(json_schema);
            if (schema) {
                schema_constrainer_ = std::make_unique<SchemaConstrainer>();
                if (tokenizer && schema_constrainer_->init(*tokenizer, std::move(schema))) {
                    cached_schema_string_ = json_schema;
                    schema_constrainer_->set_preamble(think_close);
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
        json_constrainer_->set_preamble(think_close);
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

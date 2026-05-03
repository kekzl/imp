#include "runtime/constraint_manager.h"

namespace imp {

void ConstraintManager::prepare(bool json_mode, const std::string& json_schema, Tokenizer* tokenizer) {
    active_json_ = false;
    active_schema_ = false;

    // Schema mode takes priority over plain JSON mode
    if (!json_schema.empty()) {
        if (schema_constrainer_ && schema_constrainer_->is_initialized() &&
            json_schema == cached_schema_string_) {
            // Reuse cached constrainer — just reset FSM state.
            schema_constrainer_->reset();
            active_schema_ = true;
        } else {
            auto schema = parse_json_schema(json_schema);
            if (schema) {
                schema_constrainer_ = std::make_unique<SchemaConstrainer>();
                if (tokenizer && schema_constrainer_->init(*tokenizer, std::move(schema))) {
                    cached_schema_string_ = json_schema;
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

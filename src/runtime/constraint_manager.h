#pragma once

#include "compute/json_constrain.h"
#include "compute/schema_constrain.h"
#include "compute/json_schema.h"
#include "model/tokenizer.h"
#include "core/logging.h"
#include <memory>
#include <string>
#include <cstdint>

namespace imp {

// Manages JSON and JSON-schema constrained decoding.
// Caches the schema constrainer across requests with identical schemas
// to avoid re-parsing and re-classifying ~151K tokens per request.
class ConstraintManager {
public:
    ConstraintManager() = default;

    // Prepare constraints for a request. Call before building InferenceState.
    // json_mode: enforce valid JSON syntax
    // json_schema: enforce JSON matching this schema string (empty = disabled)
    // tokenizer: needed for lazy init
    void prepare(bool json_mode, const std::string& json_schema,
                 Tokenizer* tokenizer);

    // Get current constrainer pointers (nullptr if not active).
    JsonConstrainer* json_constrainer() const noexcept {
        return active_json_ ? json_constrainer_.get() : nullptr;
    }
    SchemaConstrainer* schema_constrainer() const noexcept {
        return active_schema_ ? schema_constrainer_.get() : nullptr;
    }

    // Update FSM state after sampling a token.
    void update(int32_t token);

    // Reset FSM state (call when request finishes).
    void reset();

    // Check if any constraint is active.
    bool is_active() const noexcept { return active_json_ || active_schema_; }
    bool has_json() const noexcept { return active_json_; }
    bool has_schema() const noexcept { return active_schema_; }

private:
    std::unique_ptr<JsonConstrainer> json_constrainer_;
    std::unique_ptr<SchemaConstrainer> schema_constrainer_;
    std::string cached_schema_string_;
    bool active_json_ = false;
    bool active_schema_ = false;
};

} // namespace imp

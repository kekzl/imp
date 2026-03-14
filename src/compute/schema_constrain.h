#pragma once

#include "compute/json_schema.h"
#include "model/tokenizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <string>
#include <cstdint>
#include <memory>

namespace imp {

// Schema-aware JSON generation constrainer.
// Extends the basic JSON FSM with schema position tracking to ensure
// generated JSON matches a specific JSON Schema.
//
// Design: category bitmask for structural tokens (reused from JsonConstrainer)
// + per-token allow mask for property names and enum values.

enum class SchemaPhase : uint8_t {
    VALUE_START,           // Expecting value matching current schema node
    OBJECT_OPEN,           // After {, expecting first key or }
    OBJECT_KEY,            // Inside a key string (constraining property names)
    OBJECT_AFTER_KEY,      // After closing " of key, expecting :
    OBJECT_COLON,          // After :, expecting value
    OBJECT_AFTER_VALUE,    // After value, expecting , or }
    ARRAY_OPEN,            // After [, expecting first item or ]
    ARRAY_AFTER_ITEM,      // After item, expecting , or ]
    STRING_VALUE,          // Inside a free string value
    STRING_ESCAPE,         // After \ inside string
    NUMBER_VALUE,          // Inside a number
    LITERAL_VALUE,         // Generating true/false/null
    ENUM_VALUE,            // Inside an enum string (constrained to exact matches)
    DONE
};

struct SchemaFrame {
    const SchemaNode* node = nullptr;
    SchemaPhase phase = SchemaPhase::VALUE_START;

    // Object tracking
    std::set<std::string> emitted_keys;
    std::string current_key;
    std::string key_buffer;         // accumulated key characters

    // Enum tracking
    std::string enum_buffer;        // accumulated enum value characters

    // Literal tracking
    std::string literal_target;     // "true", "false", "null"
    int literal_pos = 0;

    // Array item count
    int item_count = 0;
};

class SchemaConstrainer {
public:
    SchemaConstrainer() = default;
    ~SchemaConstrainer();

    // Initialize with tokenizer (classifies all tokens) and schema.
    bool init(const Tokenizer& tok, std::unique_ptr<SchemaNode> schema);

    // Apply logit mask before sampling.
    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);

    // Update state with sampled token.
    void update(int32_t token);

    // Reset for a new generation with the same schema.
    void reset();

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    int vocab_size_ = 0;

    // Schema tree (owned)
    std::unique_ptr<SchemaNode> schema_;

    // Per-token classification (shared pattern with JsonConstrainer)
    std::vector<uint16_t> token_categories_;
    uint16_t* d_token_categories_ = nullptr;
    std::vector<std::string> token_texts_;

    // Per-token allow mask for fine-grained control (key names, enum values)
    std::vector<uint8_t> token_allow_;
    uint8_t* d_token_allow_ = nullptr;
    uint16_t* d_allowed_mask_ = nullptr;
    bool need_token_allow_ = false;

    // Schema FSM state
    std::vector<SchemaFrame> stack_;

    // Helpers
    SchemaFrame& top() { return stack_.back(); }
    const SchemaFrame& top() const { return stack_.back(); }

    void push_value_frame(const SchemaNode* node);

    uint16_t compute_category_mask() const;
    void compute_token_allow_mask();

    void advance_char(char c);

    // Find property schema by key name
    const SchemaNode* find_property(const SchemaNode* obj, const std::string& key) const;

    // Check if a string is a valid prefix of any remaining property name
    bool is_valid_key_prefix(const SchemaNode* obj, const std::string& prefix,
                             const std::set<std::string>& emitted) const;

    // Check if prefix matches any enum value
    bool is_valid_enum_prefix(const std::vector<std::string>& values,
                              const std::string& prefix) const;
};

} // namespace imp

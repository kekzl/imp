#pragma once

#include "model/tokenizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <string>

namespace imp {

// JSON state machine states
enum class JsonState : uint8_t {
    START,              // Expecting { or [
    OBJECT_START,       // After {: expecting " (key) or }
    OBJECT_KEY,         // Inside key string
    AFTER_KEY,          // After key string: expecting :
    AFTER_COLON,        // After :: expecting value
    AFTER_VALUE,        // After value in object: expecting , or }
    ARRAY_START,        // After [: expecting value or ]
    ARRAY_AFTER_VALUE,  // After value in array: expecting , or ]
    IN_STRING,          // Inside a string value
    IN_STRING_ESCAPE,   // After \ in string
    IN_NUMBER,          // Inside a number
    IN_LITERAL,         // Partial literal (tru, fal, nul)
    DONE                // Finished parsing
};

// Token category bitfield — each token gets a bitmask of which JSON categories
// it belongs to. At decode time, the FSM produces an allowed_mask of categories.
enum JsonTokenCat : uint16_t {
    CAT_OPEN_BRACE   = 1 << 0,   // {
    CAT_CLOSE_BRACE  = 1 << 1,   // }
    CAT_OPEN_BRACKET = 1 << 2,   // [
    CAT_CLOSE_BRACKET= 1 << 3,   // ]
    CAT_COLON        = 1 << 4,   // :
    CAT_COMMA        = 1 << 5,   // ,
    CAT_QUOTE        = 1 << 6,   // " (starts/ends string)
    CAT_STRING_CHAR  = 1 << 7,   // any char valid inside a string (including escaped)
    CAT_NUMBER_START = 1 << 8,   // 0-9, -
    CAT_TRUE_START   = 1 << 9,   // t (starts "true")
    CAT_FALSE_START  = 1 << 10,  // f (starts "false")
    CAT_NULL_START   = 1 << 11,  // n (starts "null")
    CAT_WHITESPACE   = 1 << 12,  // space, tab, newline
    CAT_LITERAL_CONT = 1 << 13,  // continuation of a partial literal (r, u, e, a, l, s)
    CAT_NUMBER_CONT  = 1 << 14,  // 0-9, ., e, E, +, - (continuation of number)
};

// Mask for tokens that can start a JSON value
static constexpr uint16_t CAT_VALUE_START =
    CAT_OPEN_BRACE | CAT_OPEN_BRACKET | CAT_QUOTE |
    CAT_NUMBER_START | CAT_TRUE_START | CAT_FALSE_START | CAT_NULL_START;

class JsonConstrainer {
public:
    JsonConstrainer() = default;
    ~JsonConstrainer();

    // Initialize: classify all tokens in the vocabulary.
    // Must be called once before use.
    bool init(const Tokenizer& tok);

    // Apply logit mask: set logits of invalid tokens to -inf.
    // Called after penalties, before sampling.
    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);

    // Update FSM state with the text of the sampled token.
    void update(int32_t token);

    // Reset FSM for a new generation.
    void reset();

    // Check if initialized
    bool is_initialized() const { return initialized_; }

    // Get max tokens to finish (force-close open structures near limit)
    int closing_tokens_needed() const { return static_cast<int>(state_stack_.size()); }

private:
    bool initialized_ = false;
    int vocab_size_ = 0;

    // Per-token category bitmask (host, copied to device at init)
    std::vector<uint16_t> token_categories_;
    uint16_t* d_token_categories_ = nullptr;

    // Per-token decoded text (for FSM update)
    std::vector<std::string> token_texts_;

    // FSM state
    std::vector<JsonState> state_stack_;
    JsonState current_state_ = JsonState::START;
    std::string partial_literal_;   // for tracking partial "true"/"false"/"null"
    std::string target_literal_;    // full expected literal

    // Device buffer for allowed mask (1 uint16_t, stable address)
    uint16_t* d_allowed_mask_ = nullptr;

    // Compute allowed category mask from current FSM state
    uint16_t compute_allowed_mask() const;

    // Advance FSM by one character
    void advance_char(char c);
};

} // namespace imp

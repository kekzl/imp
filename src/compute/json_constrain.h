#pragma once

#include "compute/preamble_gate.h"
#include "compute/json_schema.h"  // SchemaNode / RegexNfa
#include "model/tokenizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace imp {

// JSON state machine states.
//
// Note the two *_NEED_* states: after a comma a value/key is MANDATORY. They
// exist because the comma used to return to the matching *_START state, and
// those legally accept the closer — an empty [] / {} is valid JSON — so `[1,]`
// and `{"a":1,}` passed the mask and the reply did not parse (#1096).
enum class JsonState : uint8_t {
    START,              // Expecting { or [
    OBJECT_START,       // After {: expecting " (key) or }
    OBJECT_KEY,         // Inside key string
    AFTER_KEY,          // After key string: expecting :
    AFTER_COLON,        // After :: expecting value
    AFTER_VALUE,        // After value in object: expecting , or }
    ARRAY_START,        // After [: expecting value or ]
    ARRAY_AFTER_VALUE,  // After value in array: expecting , or ]
    ARRAY_NEED_VALUE,   // After , in array: expecting a value, NOT ]
    OBJECT_NEED_KEY,    // After , in object: expecting a key, NOT }
    IN_STRING,          // Inside a string value
    IN_STRING_ESCAPE,   // After \ in string
    IN_NUMBER,          // Inside a number
    IN_LITERAL,         // Partial literal (tru, fal, nul)
    DONE                // Finished parsing
};

// Token category bitfield — each token gets a bitmask of which JSON categories
// it belongs to. At decode time, the FSM produces an allowed_mask of categories.
enum JsonTokenCat : uint16_t {
    CAT_OPEN_BRACE = 1 << 0,     // {
    CAT_CLOSE_BRACE = 1 << 1,    // }
    CAT_OPEN_BRACKET = 1 << 2,   // [
    CAT_CLOSE_BRACKET = 1 << 3,  // ]
    CAT_COLON = 1 << 4,          // :
    CAT_COMMA = 1 << 5,          // ,
    CAT_QUOTE = 1 << 6,          // " (starts/ends string)
    CAT_STRING_CHAR = 1 << 7,    // any char valid inside a string (including escaped)
    CAT_NUMBER_START = 1 << 8,   // 0-9, -
    CAT_TRUE_START = 1 << 9,     // t (starts "true")
    CAT_FALSE_START = 1 << 10,   // f (starts "false")
    CAT_NULL_START = 1 << 11,    // n (starts "null")
    CAT_WHITESPACE = 1 << 12,    // space, tab, newline
    CAT_LITERAL_CONT = 1 << 13,  // continuation of a partial literal (r, u, e, a, l, s)
    CAT_NUMBER_CONT = 1 << 14,   // 0-9, ., e, E, +, - (continuation of number)
    CAT_EOS = 1 << 15,           // EOS token ids — allowed only in the DONE state
};

// Mask for tokens that can start a JSON value
static constexpr uint16_t CAT_VALUE_START = CAT_OPEN_BRACE | CAT_OPEN_BRACKET | CAT_QUOTE | CAT_NUMBER_START |
                                            CAT_TRUE_START | CAT_FALSE_START | CAT_NULL_START;

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

    // Output tokens still available to the request. Once only just enough
    // remain to close every open structure, the mask narrows to the closers so
    // the document always ends well-formed instead of being truncated (#1104).
    // -1 (default) disables the narrowing entirely.
    void set_remaining_budget(int n) { remaining_budget_ = n; }

    // Check if initialized
    bool is_initialized() const { return initialized_; }

    // Get max tokens to finish (force-close open structures near limit)
    int closing_tokens_needed() const { return static_cast<int>(state_stack_.size()); }

    // Strict-simulate `text` from the current state without mutating it —
    // true iff every char is a legal grammar continuation. Public for the
    // FSM unit tests; apply_mask uses it for whole-token validation.
    bool sim_token_valid(const std::string& text);

    // Category mask the decode path would apply right now. Public for the FSM
    // unit tests: apply_mask() needs an initialised GPU vocabulary, but the
    // force-close narrowing (#1104) lives in this mask, not in the grammar
    // simulator, so sim_token_valid() cannot observe it.
    uint16_t allowed_categories_for_test() const { return compute_allowed_mask(); }

    // Advance the FSM over raw text (tests use this to reach mid-document
    // states; the decode path goes through update()).
    void advance_text(const std::string& text) {
        for (char c : text) advance_char(c);
    }

    // Allow the model to emit a free-form preamble before strict JSON
    // enforcement starts. close_token>=0 enables close-token mode (reasoning
    // models with </think>); close_token<0 + max_tokens>0 enables budget-only
    // mode (markdown-fence preambles). Both modes also exit on the first
    // `{` / `[` seen. Pass close_token=-1 with max_tokens<=0 to fully disable.
    void set_preamble(int32_t close_token, int max_tokens = 8192, bool thinking_open = true) {
        preamble_.configure(close_token, max_tokens, thinking_open);
    }

    // Tool-aware preamble: when configured, the gate stays "no-mask" through
    // a tool-call body (delimited by open_tokens/close_tokens or the
    // open_prefix/close_suffix char fallback) and never re-enables the mask
    // after the tool closes. See PreambleGate::configure_with_tools.
    void set_preamble_with_tools(int32_t close_token, int max_tokens,
                                 std::vector<int32_t> open_tokens,
                                 std::vector<int32_t> close_tokens,
                                 std::string open_prefix,
                                 std::string close_suffix,
                                 bool thinking_open = true) {
        preamble_.configure_with_tools(close_token, max_tokens,
                                       std::move(open_tokens),
                                       std::move(close_tokens),
                                       std::move(open_prefix),
                                       std::move(close_suffix), thinking_open);
    }

    // Is the per-token allow list resident? True after a successful
    // initialize() and never false again — the point of issue #1104, where it
    // was allocated lazily inside apply_mask() and a failure there produced a
    // silently UNCONSTRAINED reply instead of a refused one.
    bool has_device_allow_list() const { return d_token_allow_ != nullptr; }

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
    // Consecutive whitespace chars in non-string states (escape-hatch cap,
    // see advance_char/compute_allowed_mask).
    int ws_run_ = 0;
    // JSON number sub-state (RFC 8259: [minus] int [frac] [exp]). Without it
    // IN_NUMBER accepted '.', 'e', 'E', '+', '-' an unlimited number of times,
    // so "3.5.5.5.5…" was a legal continuation and a model that wandered into
    // a number could never be forced out of it — the request then ran to
    // max_tokens and returned truncated, unparseable JSON (#1104).
    bool num_seen_frac_ = false;    // a '.' has been consumed
    bool num_seen_exp_ = false;     // an 'e'/'E' has been consumed
    bool num_exp_sign_ok_ = false;  // '+'/'-' legal only right after 'e'/'E'
    bool num_need_digit_ = false;   // a digit is required next (after '-', '.', 'e', sign)
    int remaining_budget_ = -1;     // see set_remaining_budget
    mutable bool force_close_active_ = false;  // last compute_allowed_mask() narrowed
    std::string partial_literal_;  // for tracking partial "true"/"false"/"null"
    std::string target_literal_;   // full expected literal

    // Device buffer for allowed mask (1 uint16_t, stable address)
    uint16_t* d_allowed_mask_ = nullptr;
    // Per-token whole-token-validated allow list (host + device)
    std::vector<uint8_t> token_allow_;
    uint8_t* d_token_allow_ = nullptr;
    std::vector<int32_t> eos_ids_;

    // Preamble pass-through (reasoning models emit <think>...</think> first)
    PreambleGate preamble_;

    // Compute allowed category mask from current FSM state
    uint16_t compute_allowed_mask() const;

    // Advance FSM by one character
    bool advance_char(char c);
    void enter_number_(char c);  // seed the RFC 8259 number sub-state (#1104)
};

}  // namespace imp

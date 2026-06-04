#pragma once

#include "compute/preamble_gate.h"
#include "compute/json_schema.h"  // RegexNfa (for GrammarConstrainer)
#include "model/tokenizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

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

    // Check if initialized
    bool is_initialized() const { return initialized_; }

    // Get max tokens to finish (force-close open structures near limit)
    int closing_tokens_needed() const { return static_cast<int>(state_stack_.size()); }

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

    // Whole-token strict validation: snapshot the FSM, advance over the
    // token text, restore. True iff every char is a legal continuation.
    bool sim_token_valid(const std::string& text);

    // Advance FSM by one character
    bool advance_char(char c);
};

// ---------------------------------------------------------------------------
// GrammarConstrainer (Part B, non-recursive GBNF subset) — drives a compiled
// RegexNfa as a token-mask FSM. Allows a candidate token only if its bytes
// keep the grammar NFA alive; EOS is allowed once the NFA is in an accepting
// state. Uses the per-token allow-mask kernel from constrain_common.h.
//
// Wiring into the executor/sampling path is left to the owning agent (this
// class mirrors SchemaConstrainer's apply_mask/update/reset surface so it can
// drop in identically). compile_gbnf_grammar() lives in json_schema.h.
// ---------------------------------------------------------------------------
class GrammarConstrainer {
public:
    GrammarConstrainer() = default;
    ~GrammarConstrainer();

    // Classify tokens and attach a compiled grammar NFA. Returns false if the
    // NFA is null/uncompiled.
    bool init(const Tokenizer& tok, std::shared_ptr<RegexNfa> grammar);

    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);
    void update(int32_t token);
    void reset();
    bool is_initialized() const { return initialized_; }

    // True once the grammar NFA is in an accepting state (generation may stop).
    bool accepts_now() const;

private:
    bool initialized_ = false;
    int vocab_size_ = 0;

    std::shared_ptr<RegexNfa> grammar_;
    std::vector<int> active_states_;  // current NFA state set

    std::vector<std::string> token_texts_;
    std::vector<uint8_t> token_allow_;
    uint8_t* d_token_allow_ = nullptr;

    void compute_token_allow_mask();
};

}  // namespace imp

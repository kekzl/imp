#pragma once

#include "compute/constrain_device_buffers.h"

#include "compute/json_schema.h"
#include "compute/preamble_gate.h"
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
    VALUE_START,         // Expecting value matching current schema node
    OBJECT_OPEN,         // After {, expecting first key or }
    OBJECT_KEY,          // Inside a key string (constraining property names)
    OBJECT_AFTER_KEY,    // After closing " of key, expecting :
    OBJECT_COLON,        // After :, expecting value
    OBJECT_AFTER_VALUE,  // After value, expecting , or }
    ARRAY_OPEN,          // After [, expecting first item or ]
    ARRAY_AFTER_ITEM,    // After item, expecting , or ]
    STRING_VALUE,        // Inside a free string value
    STRING_PATTERN,      // Inside a string value constrained by a regex pattern
    STRING_ESCAPE,       // After \ inside string
    NUMBER_VALUE,        // Inside a number
    LITERAL_VALUE,       // Generating true/false/null
    ENUM_VALUE,          // Inside an enum string (constrained to exact matches)
    ENVELOPE_OPEN,       // Forcing the tool-call open literal (e.g. "<tool_call>\n")
    ENVELOPE_CLOSE,      // Forcing the tool-call close literal (e.g. "\n</tool_call>")
    // Qwen-Coder XML tool-call body (SchemaType::XML_TOOL_CALL). One frame
    // carries the whole body; parameter values are raw text, not JSON.
    XML_FN_OPEN,    // Forcing the "<function=" literal (literal_target/pos)
    XML_FN_NAME,    // Unquoted tool-name enum in the tag, closed by '>'
    XML_PARAMS,     // Matching "\n<parameter=" vs "\n</function>" (key_buffer)
    XML_PARAM_KEY,  // Unquoted parameter-key enum in the tag, closed by '>'
    XML_RAW_VALUE,  // Raw value text until the "\n</parameter>" delimiter
    DONE
};

struct SchemaFrame {
    const SchemaNode* node = nullptr;
    SchemaPhase phase = SchemaPhase::VALUE_START;

    // Object tracking
    std::set<std::string> emitted_keys;
    std::string current_key;
    std::string key_buffer;  // accumulated key characters

    // Enum tracking
    std::string enum_buffer;  // accumulated enum value characters

    // String pattern / length tracking (for JSON-Schema "pattern"/min/maxLength).
    // Active NFA state set for the regex over chars emitted so far in the
    // current string value; refreshed as characters are consumed.
    std::vector<int> regex_states;
    int string_len = 0;  // number of content chars emitted in the current string

    // Literal tracking
    std::string literal_target;  // "true", "false", "null"
    int literal_pos = 0;

    // Array item count
    int item_count = 0;

    // TOOL_CALL frames: the tool name chosen by the completed "name" enum —
    // "arguments" resolves against the root defs entry of this name.
    // XML_TOOL_CALL frames: the name from the completed <function=NAME> tag —
    // parameter keys/required resolve against the same defs entry.
    std::string chosen_tool;

    // XML_TOOL_CALL frames only. Dedicated state instead of overloading
    // literal_pos/string_len: xml_tool caches the chosen tool's resolved
    // parameter schema (bound once when the name tag closes — the lookup is
    // per-char otherwise), xml_delim_match tracks the "\n</parameter>" match
    // inside a raw value, xml_value_open flags the forced value-opening
    // newline as consumed.
    const SchemaNode* xml_tool = nullptr;
    int xml_delim_match = 0;
    bool xml_value_open = false;

    // True right after a ',' inside an object: a key is now mandatory, so the
    // object may not close (`}`) until another key/value is emitted — prevents
    // trailing commas (`{"a":1,}`).
    bool after_comma = false;

    // True while the current number's integer part is the single digit '0'
    // (JSON forbids leading zeros: `0` is fine, `09` is not). Cleared once a
    // '.'/'e' is seen. Guards integer/number degeneration like `0999...`.
    bool num_leading_zero = false;

    // RFC 8259 number sub-state. Without it the NUMBER_VALUE phase accepted
    // '.', 'e', 'E', '+', '-' unconditionally, so "3.5.5.5…" was legal and a
    // degenerating model could not be forced to close the number (#1104 —
    // same defect as JsonConstrainer's IN_NUMBER).
    bool num_frac = false;        // a '.' has been consumed
    bool num_exp = false;         // an 'e'/'E' has been consumed
    bool num_sign_ok = false;     // '+'/'-' legal only right after 'e'/'E'
    bool num_need_digit = false;  // a digit is required next
};

class SchemaConstrainer {
public:
    SchemaConstrainer() = default;
    ~SchemaConstrainer();

    // Initialize with tokenizer (classifies all tokens) and schema.
    [[nodiscard]] bool init(const Tokenizer& tok, std::unique_ptr<SchemaNode> schema);

    // Grammar-only init for the CPU FSM tests: installs the schema and the
    // frame stack, skipping the tokenizer classification and the device
    // buffers that only apply_mask needs. Lets the generative battery run in
    // the `unit` lane — the grammar bugs this surface has shipped (#761, #850,
    // #1014) all escaped CI, which has no GPU runner. Not for engine use:
    // apply_mask/update by token id require the full init above.
    bool init_grammar_for_test(std::unique_ptr<SchemaNode> schema);

    // Apply logit mask before sampling.
    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);

    // Update state with sampled token.
    void update(int32_t token);

    // Jump-ahead (#844): appends to `out` the characters every schema-legal
    // continuation must spell next (the schema skeleton — braces, quotes,
    // single-candidate keys, colons, literals, unambiguous enum prefixes).
    // Pure probe: never advances the FSM. Returns the char count.
    //
    // CHAR level, not token level: on a real BPE vocab a "forced" state
    // almost always admits several tokens spelling the same forced text
    // (':' vs ':"' vs ':{"'), so exactly-one-legal-token forcing never
    // fires. The caller drafts the canonical tokenization of this text and
    // verifies by sampling (see the constrained-pipeline jump-ahead).
    int forced_text(std::string& out, int max_chars) const;

    // True iff emitting the whole token keeps the schema satisfiable: every
    // char is a legal transition and nothing trails past the root close. This
    // catches multi-char tokens that span phase transitions (`{}`, `":"`,
    // `"Why`, integer `0.98`) which the first-char category mask misses.
    // Public for the FSM unit tests; apply_mask uses it for whole-token
    // validation.
    bool token_legal(const std::string& text) const;

    // Reset for a new generation with the same schema.
    void reset();

    // Tool-call envelope (#1002): when set, generation is framed by forced
    // literals (open before the root value, close after it) — e.g.
    // "<tool_call>\n" ... "\n</tool_call>". Configure BEFORE reset().
    void set_envelope(std::string open, std::string close) {
        envelope_open_ = std::move(open);
        envelope_close_ = std::move(close);
    }

    // Strict OPTIONAL tool call (#1002, OpenAI `strict: true` with a model-chosen
    // call): the envelope is emitted freely by the model — the preamble gate
    // detects the opener and hands off to the TOOL_CALL body FSM (which enforces
    // the arguments, then forces the close literal + EOS). Unlike set_envelope's
    // forced mode, no tool call is forced: if the model answers in text, the
    // constraint never engages. Requires set_envelope (for the close literal)
    // and a tool-aware preamble configured with strict_tool=true. Configure
    // BEFORE reset().
    void set_strict_optional_envelope(bool v) { strict_optional_envelope_ = v; }

    // parallel_tool_calls (#1002, strict optional mode only): when true, the gate
    // re-arms after each tool-call body completes instead of forcing EOS, so the
    // model may emit several tool calls (each body FSM-enforced) or stop. When
    // false, EOS is forced after the first call (at most one). Configure BEFORE
    // reset().
    void set_allow_parallel(bool v) { allow_parallel_ = v; }

    bool is_initialized() const { return initialized_; }

    // See JsonConstrainer::set_preamble for semantics — close-token mode for
    // reasoning models (</think>) or budget-only mode for markdown fences.
    void set_preamble(int32_t close_token, int max_tokens = 8192, bool thinking_open = true) {
        preamble_.configure(close_token, max_tokens, thinking_open);
    }

    // Tool-aware preamble: when configured, the gate stays "no-mask" through
    // a tool-call body (delimited by open_tokens/close_tokens or the
    // open_prefix/close_suffix char fallback) and never re-enables the mask
    // after the tool closes. See PreambleGate::configure_with_tools.
    void set_preamble_with_tools(int32_t close_token, int max_tokens, std::vector<int32_t> open_tokens,
                                 std::vector<int32_t> close_tokens, std::string open_prefix,
                                 std::string close_suffix, bool thinking_open = true,
                                 bool strict_tool = false) {
        preamble_.configure_with_tools(close_token, max_tokens, std::move(open_tokens),
                                       std::move(close_tokens), std::move(open_prefix),
                                       std::move(close_suffix), thinking_open, strict_tool);
    }

private:
    bool initialized_ = false;
    int vocab_size_ = 0;

    // Schema tree (owned)
    std::unique_ptr<SchemaNode> schema_;

    // Per-token classification (shared pattern with JsonConstrainer)
    std::vector<uint16_t> token_categories_;
    std::vector<std::string> token_texts_;

    // Per-token allow mask for fine-grained control (key names, enum values)
    std::vector<uint8_t> token_allow_;
    // categories + allowed-mask + token-allow, one lifetime (F-18)
    ConstrainDeviceBuffers dev_;
    bool need_token_allow_ = false;

    // Schema FSM state
    std::vector<SchemaFrame> stack_;

    // EOS token ids — forced (everything else masked) once the root value is
    // complete, so generation stops cleanly instead of trailing free text.
    std::vector<int32_t> eos_tokens_;

    // Preamble pass-through (reasoning models emit <think>...</think> first)
    PreambleGate preamble_;

    // Tool-call envelope literals (empty = no envelope).
    std::string envelope_open_;
    std::string envelope_close_;

    // Strict OPTIONAL tool call: the open literal is NOT forced (the model emits
    // it freely); the preamble gate hands off to the body FSM on the opener.
    bool strict_optional_envelope_ = false;

    // parallel_tool_calls: re-arm the gate after each strict tool-call body so
    // the model may emit several calls (strict mode only).
    bool allow_parallel_ = false;

    // Helpers
    // C++23 deducing this: one overload serves const and non-const callers.
    template <typename Self>
    auto&& top(this Self&& self) {
        return self.stack_.back();
    }

    void push_value_frame(const SchemaNode* node);

    // Strict optional tool call (#1002): the preamble gate has just seen the
    // opener, so install the TOOL_CALL body frame with the close literal armed
    // (the post-ENVELOPE_OPEN state) — the FSM now enforces the body, then the
    // forced close literal, then EOS.
    void engage_tool_body();

    uint16_t compute_category_mask() const;
    // cat_mask: the current category mask — tokens failing it are masked by
    // the kernel anyway, so their (expensive) per-token simulation is skipped.
    void compute_token_allow_mask(uint16_t cat_mask);

    // Single-char transition over a frame stack. Returns false when c is not a
    // legal transition for the current phase. Drives both the real update path
    // (on stack_) and per-token mask simulation (on a cloned stack), so there
    // is one source of truth for the schema grammar.
    bool sim_advance(std::vector<SchemaFrame>& stk, char c) const;

    // Find property schema by key name
    const SchemaNode* find_property(const SchemaNode* obj, const std::string& key) const;

    // Check if a string is a valid prefix of any remaining property name
    bool is_valid_key_prefix(const SchemaNode* obj, const std::string& prefix,
                             const std::set<std::string>& emitted) const;

    // Check if prefix matches any enum value
    bool is_valid_enum_prefix(const std::vector<std::string>& values, const std::string& prefix) const;

    // True if the current string can legally close: regex accepts and length
    // constraints satisfied.
    bool can_close_string(const SchemaFrame& f, const std::vector<int>& states, int len) const;
};

}  // namespace imp

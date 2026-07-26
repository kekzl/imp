#pragma once

#include "compute/json_schema.h"  // RegexNfa
#include "compute/preamble_gate.h"
#include "model/tokenizer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace imp {

// Constrain generation to a regular expression.
//
// imp could pin JSON (schema or free-form) and the tool dialects, but nothing
// else — an agent needing a diff header, an SQL statement, an ID format or a
// small DSL had to prompt and hope (docs/roadmap.md gap 4). A regex is the
// smallest surface covering those, and it is what vLLM and SGLang expose too.
//
// The engine is NOT new: `RegexNfa` (compute/json_schema.h) already backs
// JSON-Schema `pattern`. This class is the decode-time wrapper around it, with
// the same apply_mask contract as JsonConstrainer / SchemaConstrainer.
//
// The pattern is implicitly anchored: the whole output must match, which is
// what "constrain the output to this format" means. EOS is allowed only from an
// accepting state, so generation cannot stop half-way through the format.

class RegexConstrainer {
public:
    RegexConstrainer() = default;
    ~RegexConstrainer();
    RegexConstrainer(const RegexConstrainer&) = delete;
    RegexConstrainer& operator=(const RegexConstrainer&) = delete;

    // Compiles `pattern` and classifies the tokenizer vocabulary. Returns false
    // on an unsupported/malformed pattern — the caller then declines
    // constrained decoding rather than enforcing a wrong grammar.
    bool init(const std::string& pattern, Tokenizer* tokenizer, int vocab_size);

    // Pattern-only init for unit tests: no tokenizer, no device buffers. Only
    // update_text/is_done/would_accept work afterwards.
    bool init_pattern_only(const std::string& pattern);

    bool is_initialized() const { return initialized_; }

    // Back to the start state, keeping the compiled pattern and the classified
    // vocabulary (both are the expensive parts).
    void reset();

    // Mask logits in place: every token that would leave the language is
    // driven to -inf. No-op while the preamble gate is open.
    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);

    // Advance the FSM with the token the sampler actually chose.
    // Returns false if that token was not in the language (should not happen
    // when apply_mask ran, and is logged by the caller if it does).
    bool update(int32_t token_id);

    // Advance by raw text — the path unit tests use.
    bool update_text(const std::string& text);

    // True when the active state set accepts, i.e. stopping here is legal.
    bool is_done() const;

    // Would `text` keep the output inside the language? Does not advance.
    bool would_accept(const std::string& text) const;

    // See JsonConstrainer::set_preamble — lets a reasoning model emit its
    // <think> block before the constraint engages.
    void set_preamble(int32_t close_token, int max_tokens = 8192, bool thinking_open = true) {
        preamble_.configure(close_token, max_tokens, thinking_open);
    }
    PreambleGate& preamble() { return preamble_; }

    const std::string& pattern() const { return pattern_; }

private:
    // Per-token allow list for the current state set, cached: recomputing it
    // walks the whole vocabulary, which is far too slow to do every step.
    const std::vector<uint8_t>& allow_for_current_state(int vocab_size);

    // False when no input can ever reach an accepting state — such a pattern is
    // refused at init rather than silently producing an empty completion.
    bool language_non_empty() const;

    bool initialized_ = false;
    int vocab_size_ = 0;
    std::string pattern_;

    RegexNfa nfa_;
    std::vector<int> states_;  // active NFA state set

    std::vector<std::string> token_texts_;
    std::vector<int32_t> eos_ids_;

    std::map<std::vector<int>, std::vector<uint8_t>> mask_cache_;

    uint8_t* d_token_allow_ = nullptr;
    PreambleGate preamble_;
};

}  // namespace imp

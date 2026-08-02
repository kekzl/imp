#pragma once

#include "compute/gbnf_grammar.h"
#include "compute/preamble_gate.h"
#include "model/tokenizer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace imp {

// Constrain generation to a context-free grammar written in GBNF
// (docs/roadmap.md gap 8).
//
// This is the decode-time wrapper around GbnfMatcher, with the same apply_mask
// contract as JsonConstrainer / SchemaConstrainer / RegexConstrainer. What the
// grammar adds over the regex wrapper is a STACK — the language is context-free,
// so the live state is a set of parse continuations, not a set of NFA states —
// and UTF-8 assembly, since a BPE token can end mid-character.
//
// The grammar is implicitly anchored: the whole output must be a derivation of
// `root`, and EOS is allowed only once that derivation is complete.

class GrammarConstrainer {
public:
    GrammarConstrainer() = default;
    ~GrammarConstrainer();
    GrammarConstrainer(const GrammarConstrainer&) = delete;
    GrammarConstrainer& operator=(const GrammarConstrainer&) = delete;

    // Compiles `gbnf` and classifies the tokenizer vocabulary. Returns false on
    // a grammar that does not compile — the caller then declines constrained
    // decoding rather than enforcing something nobody wrote.
    [[nodiscard]] bool init(const std::string& gbnf, Tokenizer* tokenizer, int vocab_size);

    // Grammar-only init for unit tests: no tokenizer, no device buffers. Only
    // update_text/is_done/would_accept work afterwards.
    bool init_grammar_only(const std::string& gbnf);

    bool is_initialized() const { return initialized_; }

    // Back to the start of `root`, keeping the compiled grammar and the
    // classified vocabulary (both are the expensive parts).
    void reset();

    // Mask logits in place: every token that would leave the language is driven
    // to -inf. No-op while the preamble gate is open.
    void apply_mask(float* d_logits, int vocab_size, cudaStream_t stream);

    // Advance with the token the sampler actually chose. Returns false if that
    // token was not in the language (should not happen when apply_mask ran).
    bool update(int32_t token_id);

    // Advance by raw text — the path unit tests use.
    bool update_text(const std::string& text) { return matcher_.update_text(text); }

    // True when the derivation is complete, i.e. stopping here is legal.
    bool is_done() const { return initialized_ && matcher_.is_done(); }

    // Would `text` keep the output inside the language? Does not advance.
    bool would_accept(const std::string& text) const { return !initialized_ || matcher_.would_accept(text); }

    // See JsonConstrainer::set_preamble — lets a reasoning model emit its
    // <think> block before the constraint engages.
    void set_preamble(int32_t close_token, int max_tokens = 8192, bool thinking_open = true) {
        preamble_.configure(close_token, max_tokens, thinking_open);
    }
    PreambleGate& preamble() { return preamble_; }

    const std::string& source() const { return source_; }
    const std::string& error() const { return error_; }

private:
    // Per-token allow list for the current state, cached: recomputing it walks
    // the whole vocabulary, which is far too slow to do every step.
    const std::vector<uint8_t>& allow_for_current_state(int vocab_size);

    bool initialized_ = false;
    int vocab_size_ = 0;
    std::string source_;
    std::string error_;

    GbnfMatcher matcher_;

    std::vector<std::string> token_texts_;
    std::vector<int32_t> eos_ids_;

    std::map<std::vector<int32_t>, std::vector<uint8_t>> mask_cache_;

    uint8_t* d_token_allow_ = nullptr;
    PreambleGate preamble_;
};

}  // namespace imp

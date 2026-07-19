#include "compute/schema_constrain.h"
#include "compute/json_constrain.h"  // reuse token category definitions
#include "compute/constrain_common.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <utility>

namespace imp {

// Max simulated frame-stack depth. Each '{'/'[' nesting level holds ~2 frames
// (container frame + value frame), so 192 frames ~= 96 nesting levels. Only
// reachable via recursive $ref schemas; hitting the cap forces closure (still
// schema-valid — any finite nesting satisfies a recursive schema).
static constexpr size_t kMaxSchemaStackDepth = 192;

// Effective item ceiling for an array frame: explicit "maxItems" wins; an
// enum-items array without one is capped at the enum's cardinality — a list
// that repeats an enum member more often than the enum has members carries no
// information, and an unbounded enum array is the observed degeneration loop
// of a reasoning model whose think block was budget-force-closed (#1014:
// `["tech","tech","tech",...` until max_tokens). Same anti-runaway class as
// the number-digit cap (#751). INT_MAX = unbounded.
static int effective_max_items(const SchemaNode* root, const SchemaNode* array_node) {
    if (!array_node)
        return INT_MAX;
    if (array_node->max_items >= 0)
        return array_node->max_items;
    const SchemaNode* items = resolve_schema_ref(root, array_node->items.get());
    if (items && items->type == SchemaType::ENUM && !items->enum_values.empty())
        return static_cast<int>(items->enum_values.size());
    return INT_MAX;
}

// ---------------------------------------------------------------------------
// Qwen-Coder XML tool-call body (SchemaType::XML_TOOL_CALL):
//   <function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>
// Tags are literal, the name/keys are unquoted enums, and VALUES are raw text
// (multi-line, unescaped) ending at the "\n</parameter>" delimiter.
// ---------------------------------------------------------------------------

static const char* const kXmlFnOpen = "<function=";
static const std::string kXmlParamOpen = "\n<parameter=";
static const std::string kXmlFnClose = "\n</function>";
static const std::string kXmlParamDelim = "\n</parameter>";

// Delimiter tracker for the raw-value phase: one KMP step over the delimiter.
// Chars are never rejected (any text is a legal value) — a dead partial match
// falls back and re-tries, so values containing partial delimiters
// ("\n</param" + more text, "\n\n</parameter>") track correctly. The closed
// form below relies on the delimiter's first char '\n' appearing nowhere else
// in it, which makes every KMP border 0 — a mismatch can only fall back to
// "did this char restart a match".
static int xml_delim_step(const std::string& target, int len, char c) {
    if (len < static_cast<int>(target.size()) && c == target[len])
        return len + 1;
    return c == target[0] ? 1 : 0;
}

// The chosen tool's parameter schema for an XML body frame (root defs entry,
// same layout as TOOL_CALL's dynamic "arguments" binding). REVERSE scan: a
// tool's hoisted "<tool>/<def>" entries precede the tool entries, so a tool
// whose (request-supplied, unvalidated) name collides with a hoisted key —
// e.g. a tool literally named "a/B" next to tool "a" hoisting $def "B" —
// must still bind its own entry, which always comes later.
static const SchemaNode* xml_tool_params(const SchemaNode* root, const std::string& chosen) {
    if (!root)
        return nullptr;
    for (auto it = root->defs.rbegin(); it != root->defs.rend(); ++it) {
        if (it->first == chosen)
            return resolve_schema_ref(root, it->second.get());
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

SchemaConstrainer::~SchemaConstrainer() {
    if (d_token_categories_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_categories_));
    if (d_token_allow_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_allow_));
    if (d_allowed_mask_)
        IMP_CUDA_CHECK_LOG(cudaFree(d_allowed_mask_));
}

bool SchemaConstrainer::init(const Tokenizer& tok, std::unique_ptr<SchemaNode> schema) {
    schema_ = std::move(schema);
    if (!schema_)
        return false;

    vocab_size_ = tok.vocab_size();
    eos_tokens_ = tok.eos_ids();

    // Classify all tokens (same logic as JsonConstrainer)
    token_categories_.resize(vocab_size_, 0);
    token_texts_.resize(vocab_size_);
    token_allow_.resize(vocab_size_, 1);

    for (int i = 0; i < vocab_size_; i++) {
        std::string text = tok.decode_token(i);
        token_texts_[i] = text;
        token_categories_[i] = classify_token(text);
        // XML tool-call schemas ONLY: classify_token gives category 0 to every
        // token with a byte outside printable ASCII — mixed text+control code
        // tokens ("):\n    def") AND all multi-byte UTF-8/CJK tokens (~40% of
        // the Qwen vocab) — and the mask kernel requires category AND allow,
        // so they could never appear in a raw XML value whose whole point is
        // arbitrary multi-line text. Retag them CAT_STRING_CHAR so the
        // per-token simulation decides. Scoped to XML schemas: for plain JSON
        // constrainers the category-0 prefilter is a load-bearing cost cutoff
        // (see compute_token_allow_mask) and its behavior must not change.
        if (schema_->type == SchemaType::XML_TOOL_CALL && token_categories_[i] == 0 && !text.empty())
            token_categories_[i] = CAT_STRING_CHAR;
    }

    // Upload to GPU
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_token_categories_, vocab_size_ * sizeof(uint16_t)));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_token_categories_, token_categories_.data(),
                                  vocab_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice));

    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_token_allow_, vocab_size_ * sizeof(uint8_t)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_allowed_mask_, sizeof(uint16_t)));

    reset();
    initialized_ = true;

    IMP_LOG_INFO("SchemaConstrainer: initialized with %d tokens, schema type=%d", vocab_size_,
                 std::to_underlying(schema_->type));
    return true;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void SchemaConstrainer::reset() {
    stack_.clear();
    if (strict_optional_envelope_) {
        // Strict OPTIONAL tool call: the model may or may not call. Leave the
        // stack EMPTY — the tool-aware preamble gate is ACTIVE (mask bypassed),
        // so free text / a plain answer passes through unconstrained. update()
        // installs the body frame (engage_tool_body) only if the gate detects
        // the opener; until then apply_mask/forced_text no-op on the empty stack
        // behind the active gate.
    } else if (!envelope_open_.empty()) {
        // Envelope wrapper frame: forces the open literal, then hosts the root
        // value; when the value pops it flips to ENVELOPE_CLOSE (#1002).
        SchemaFrame env;
        env.node = schema_.get();
        env.phase = SchemaPhase::ENVELOPE_OPEN;
        env.literal_target = envelope_open_;
        env.literal_pos = 0;
        stack_.push_back(std::move(env));
    } else {
        push_value_frame(schema_.get());
    }
    need_token_allow_ = false;
    std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)1);
    preamble_.reset();
}

void SchemaConstrainer::engage_tool_body() {
    // Install the post-ENVELOPE_OPEN state: an envelope frame armed to force the
    // close literal after the body pops (mirrors ENVELOPE_OPEN's completion at
    // sim_advance), plus the TOOL_CALL body value-frame on top. The model has
    // already emitted the open tag (absorbed by the gate); the body FSM enforces
    // the arguments, then the close literal is forced, then EOS.
    SchemaFrame env;
    env.node = schema_.get();
    env.phase = SchemaPhase::ENVELOPE_CLOSE;
    env.literal_target = envelope_close_;
    env.literal_pos = 0;
    stack_.push_back(std::move(env));
    push_value_frame(schema_.get());
}

void SchemaConstrainer::push_value_frame(const SchemaNode* node) {
    SchemaFrame frame;
    // Frames always hold RESOLVED nodes: $ref indirection ends here, so the
    // phase machine below never sees SchemaType::REF.
    frame.node = resolve_schema_ref(schema_.get(), node);
    frame.phase = SchemaPhase::VALUE_START;
    stack_.push_back(std::move(frame));
}

// ---------------------------------------------------------------------------
// Property / enum helpers
// ---------------------------------------------------------------------------

const SchemaNode* SchemaConstrainer::find_property(const SchemaNode* obj, const std::string& key) const {
    for (auto& [name, schema] : obj->properties) {
        if (name == key)
            return resolve_schema_ref(schema_.get(), schema.get());
    }
    return nullptr;
}

// TOOL_CALL keys are ORDERED: "name" must be emitted before "arguments" so the
// argument schema is bound before any argument content is generated (#1002).
static bool tool_call_key_available(const std::string& key, const std::set<std::string>& emitted) {
    if (!emitted.count("name"))
        return key == "name";
    return key == "arguments" && !emitted.count("arguments");
}

bool SchemaConstrainer::is_valid_key_prefix(const SchemaNode* obj, const std::string& prefix,
                                            const std::set<std::string>& emitted) const {
    for (auto& [name, _] : obj->properties) {
        if (emitted.count(name))
            continue;
        if (obj->type == SchemaType::TOOL_CALL && !tool_call_key_available(name, emitted))
            continue;
        if (name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0)
            return true;
    }
    return false;
}

bool SchemaConstrainer::is_valid_enum_prefix(const std::vector<std::string>& values,
                                             const std::string& prefix) const {
    for (auto& v : values) {
        if (v.size() >= prefix.size() && v.compare(0, prefix.size(), prefix) == 0)
            return true;
    }
    return false;
}

bool SchemaConstrainer::token_keeps_pattern_alive(const SchemaFrame& f, const std::string& content,
                                                  std::vector<int>& out_states, int& out_len) const {
    out_states = f.regex_states;
    out_len = f.string_len;

    bool has_nfa = f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled();
    int max_len = f.node ? f.node->max_length : -1;

    for (char ch : content) {
        // Reject control chars / quote / backslash inside a constrained string.
        unsigned char uc = static_cast<unsigned char>(ch);
        if (uc < 32 || ch == '"' || ch == '\\')
            return false;
        if (max_len >= 0 && out_len + 1 > max_len)
            return false;
        if (has_nfa) {
            out_states = f.node->pattern_nfa->step(out_states, uc);
            if (out_states.empty())
                return false;  // prefix died -> no string can complete
        }
        out_len++;
    }
    return true;
}

bool SchemaConstrainer::can_close_string(const SchemaFrame& f, const std::vector<int>& states,
                                         int len) const {
    bool has_nfa = f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled();
    if (has_nfa && !f.node->pattern_nfa->accepts(states))
        return false;
    if (f.node && f.node->min_length >= 0 && len < f.node->min_length)
        return false;
    if (f.node && f.node->max_length >= 0 && len > f.node->max_length)
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Compute category mask from schema FSM state
// ---------------------------------------------------------------------------

uint16_t SchemaConstrainer::compute_category_mask() const {
    if (stack_.empty())
        return CAT_WHITESPACE;

    const auto& f = top();
    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            // No insignificant whitespace: a reasoning model whose top token is
            // a newline would otherwise stall forever (whitespace is always
            // re-allowed, never forcing structural progress). Compact JSON only
            // — whitespace *inside* string values is CAT_STRING_CHAR, untouched.
            uint16_t mask = 0;
            if (!f.node)
                return mask | CAT_VALUE_START;
            switch (f.node->type) {
                case SchemaType::OBJECT:
                case SchemaType::TOOL_CALL:
                    mask |= CAT_OPEN_BRACE;
                    break;
                case SchemaType::ARRAY:
                    mask |= CAT_OPEN_BRACKET;
                    break;
                case SchemaType::STRING:
                    mask |= CAT_QUOTE;
                    break;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER:
                    mask |= CAT_NUMBER_START;
                    break;
                case SchemaType::BOOLEAN:
                    mask |= CAT_TRUE_START | CAT_FALSE_START;
                    break;
                case SchemaType::NULL_TYPE:
                    mask |= CAT_NULL_START;
                    break;
                case SchemaType::ENUM:
                    mask |= CAT_QUOTE;
                    break;  // enum values are strings
                case SchemaType::XML_TOOL_CALL:
                    // '<' (and tolerated leading whitespace) span categories —
                    // per-token simulation decides, like the XML phases below.
                    return 0xFFFF;
                case SchemaType::ANY_OF:
                    mask |= CAT_VALUE_START;
                    break;
                default:
                    mask |= CAT_VALUE_START;
                    break;
            }
            return mask;
        }

        case SchemaPhase::OBJECT_OPEN: {
            uint16_t mask = CAT_QUOTE;  // " for key (compact JSON, no whitespace)
            // Allow } only if all required keys are emitted
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) {
                        all_required = false;
                        break;
                    }
                }
            }
            if (all_required)
                mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::OBJECT_KEY:
            return CAT_STRING_CHAR | CAT_QUOTE;  // token_allow handles key constraining

        case SchemaPhase::OBJECT_AFTER_KEY:
            return CAT_COLON;

        case SchemaPhase::OBJECT_COLON:
            return CAT_COLON;

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            uint16_t mask = CAT_COMMA;
            bool all_required = true;
            if (f.node) {
                for (auto& req : f.node->required) {
                    if (!f.emitted_keys.count(req)) {
                        all_required = false;
                        break;
                    }
                }
            }
            if (all_required)
                mask |= CAT_CLOSE_BRACE;
            return mask;
        }

        case SchemaPhase::ARRAY_OPEN: {
            // minItems > 0: the empty array may not close yet.
            uint16_t mask = (f.node && f.node->min_items > 0) ? 0 : CAT_CLOSE_BRACKET;
            // Allow value start for first item ($ref items resolve first)
            const SchemaNode* items = f.node ? resolve_schema_ref(schema_.get(), f.node->items.get())
                                             : nullptr;
            if (items) {
                switch (items->type) {
                    case SchemaType::STRING:
                        mask |= CAT_QUOTE;
                        break;
                    case SchemaType::NUMBER:
                    case SchemaType::INTEGER:
                        mask |= CAT_NUMBER_START;
                        break;
                    case SchemaType::BOOLEAN:
                        mask |= CAT_TRUE_START | CAT_FALSE_START;
                        break;
                    case SchemaType::OBJECT:
                        mask |= CAT_OPEN_BRACE;
                        break;
                    case SchemaType::ARRAY:
                        mask |= CAT_OPEN_BRACKET;
                        break;
                    default:
                        mask |= CAT_VALUE_START;
                        break;
                }
            } else {
                mask |= CAT_VALUE_START;
            }
            return mask;
        }

        case SchemaPhase::ARRAY_AFTER_ITEM: {
            uint16_t mask = 0;
            if (f.item_count < effective_max_items(schema_.get(), f.node))
                mask |= CAT_COMMA;
            if (!f.node || f.node->min_items < 0 || f.item_count >= f.node->min_items)
                mask |= CAT_CLOSE_BRACKET;
            // Contradictory bounds (maxItems < minItems): let the array close
            // rather than deadlock the mask.
            if (mask == 0)
                mask = CAT_CLOSE_BRACKET;
            return mask;
        }

        case SchemaPhase::STRING_VALUE:
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::STRING_PATTERN:
            // token_allow enforces the regex / length; category just gates to
            // string content + closing quote.
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::STRING_ESCAPE:
            return 0xFFFF;  // any char valid after backslash

        case SchemaPhase::NUMBER_VALUE: {
            uint16_t m = CAT_COMMA | CAT_CLOSE_BRACE | CAT_CLOSE_BRACKET;
            // Cap the digit run so a model that degenerates into a digit loop
            // (e.g. "age":42000000…) can't run an unbounded number to max_tokens
            // and leave the JSON unterminated (#751). f.string_len counts digits;
            // once it hits the cap, drop continue-number so the number must close
            // — the result is a valid (if large) JSON number, not a runaway. The
            // cap (40) is far beyond any real int64 (≤19) / double (~17 sig) value.
            constexpr int kMaxNumberDigits = 40;
            if (f.string_len < kMaxNumberDigits)
                m |= CAT_NUMBER_CONT;
            return m;
        }

        case SchemaPhase::LITERAL_VALUE:
            return CAT_LITERAL_CONT;

        case SchemaPhase::ENUM_VALUE:
            return CAT_STRING_CHAR | CAT_QUOTE;

        case SchemaPhase::ENVELOPE_OPEN:
        case SchemaPhase::ENVELOPE_CLOSE:
            // Envelope chars ('<', '/', letters, newline) span categories —
            // legality is fully decided by the per-token simulation.
            return 0xFFFF;

        case SchemaPhase::XML_FN_OPEN:
        case SchemaPhase::XML_FN_NAME:
        case SchemaPhase::XML_PARAMS:
        case SchemaPhase::XML_PARAM_KEY:
        case SchemaPhase::XML_RAW_VALUE:
            // XML tags and raw values span categories — same delegation.
            return 0xFFFF;

        case SchemaPhase::DONE:
            return CAT_WHITESPACE;
    }
    return 0xFFFF;
}

// ---------------------------------------------------------------------------
// Compute per-token allow mask (for key names and enum values)
// ---------------------------------------------------------------------------

void SchemaConstrainer::compute_token_allow_mask(uint16_t cat_mask) {
    need_token_allow_ = false;

    if (stack_.empty())
        return;

    // Full per-token legality: a candidate token is allowed only if simulating
    // its entire text from the current FSM state stays legal at every char.
    // This is what the first-char category mask cannot do — it catches
    // multi-char tokens that span phase transitions (`{}` closing an object
    // with unmet required keys, `":"` as a bogus enum value, `"Why` opening a
    // non-existent key, `0.98` for an integer). The category mask still runs
    // alongside (it governs EOS / whitespace / structural first-char), so a
    // token must pass BOTH. token_legal() handles empty (EOS/special) tokens by
    // deferring to the category mask.
    //
    // Cost control (this loop runs per decode step over the whole vocab, and
    // token_legal deep-copies the frame stack per candidate — it dominated
    // json_schema decode at 151k tokens):
    //  - category prefilter: the kernel ANDs category and allow, so a token
    //    that fails the category mask is masked regardless of allow — skip
    //    its simulation entirely. In structural phases the category mask is
    //    narrow and this eliminates almost all simulations.
    //  - in-string O(1) shortcut (mirrors JsonConstrainer): in a free string
    //    value, any token without '"', '\\' or a raw control char stays
    //    inside the string by construction — sim_advance would accept every
    //    char without touching the stack, so skip the simulation. Pattern /
    //    enum / key strings still simulate (prefix & regex constraints).
    need_token_allow_ = true;
    const SchemaPhase phase = top().phase;
    const bool free_string = (phase == SchemaPhase::STRING_VALUE);
    // XML cost control — the XML phases delegate the whole vocab to the
    // per-token simulation (category 0xFFFF), which deep-copies the frame
    // stack per candidate; unchecked that is ~150k stack copies per decode
    // step (the exact regime the comment above exists for). Two shortcuts:
    //  - tag phases have a tiny legal-first-char set: probe all 256 first
    //    chars ONCE on a cloned stack and reject by first byte (a token whose
    //    first char is an illegal transition can never be legal).
    //  - open raw values accept ANY text; a token is only ever illegal if the
    //    "\n</parameter>" delimiter COMPLETES inside it (its tail then falls
    //    into the tag grammar). Run the int-only delimiter automaton over the
    //    token first; only completing tokens pay the full simulation.
    const bool xml_raw = (phase == SchemaPhase::XML_RAW_VALUE);
    const bool xml_raw_open = xml_raw && top().xml_value_open;
    const bool xml_tag_phase =
        phase == SchemaPhase::XML_FN_OPEN || phase == SchemaPhase::XML_FN_NAME ||
        phase == SchemaPhase::XML_PARAMS || phase == SchemaPhase::XML_PARAM_KEY ||
        (xml_raw && !top().xml_value_open);
    bool first_ok[256];
    if (xml_tag_phase) {
        for (int c = 0; c < 256; c++) {
            std::vector<SchemaFrame> probe = stack_;
            first_ok[c] = sim_advance(probe, static_cast<char>(c));
        }
    }
    for (int i = 0; i < vocab_size_; i++) {
        if ((token_categories_[i] & cat_mask) == 0) {
            token_allow_[i] = 0;  // masked by category — simulation irrelevant
            continue;
        }
        const std::string& text = token_texts_[i];
        if (xml_tag_phase && !text.empty() &&
            !first_ok[static_cast<unsigned char>(text[0])]) {
            token_allow_[i] = 0;
            continue;
        }
        if (xml_raw_open && !text.empty()) {
            int m = top().xml_delim_match;
            bool completes = false;
            for (char ch : text) {
                m = xml_delim_step(kXmlParamDelim, m, ch);
                if (m == static_cast<int>(kXmlParamDelim.size())) {
                    completes = true;
                    break;
                }
            }
            token_allow_[i] = completes ? (token_legal(text) ? 1 : 0) : 1;
            continue;
        }
        if (free_string && !text.empty()) {
            bool plain = true;
            for (char c : text) {
                if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20) {
                    plain = false;
                    break;
                }
            }
            if (plain) {
                token_allow_[i] = 1;
                continue;
            }
        }
        token_allow_[i] = token_legal(text) ? 1 : 0;
    }
    // EOS must not stop generation mid-value: its rendered text ("<|im_end|>")
    // would pass the anything-goes value scan above. Post-loop so no shortcut
    // can re-allow it.
    if (xml_raw) {
        for (int32_t e : eos_tokens_)
            if (e >= 0 && e < vocab_size_)
                token_allow_[e] = 0;
    }
}

// ---------------------------------------------------------------------------
// Apply mask to logits
// ---------------------------------------------------------------------------

void SchemaConstrainer::apply_mask(float* d_logits, int vocab_size, cudaStream_t stream) {
    if (!initialized_)
        return;

    if (preamble_.active())
        return;

    // Root value complete (stack drained): force EOS so generation stops cleanly
    // instead of trailing free text after the closing brace/bracket. Mask
    // everything except the EOS token(s) via the allow path (category = all).
    if (stack_.empty()) {
        if (eos_tokens_.empty())
            return;  // no EOS to force — leave the model unconstrained
        std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)0);
        for (int32_t e : eos_tokens_)
            if (e >= 0 && e < vocab_size_)
                token_allow_[e] = 1;
        uint16_t all_cats = 0xFFFF;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_allowed_mask_, &all_cats, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, token_allow_.data(), vocab_size_ * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream));
        int t = 256, b = (vocab_size + t - 1) / t;
        constrain_mask_allow_kernel<<<b, t, 0, stream>>>(d_logits, d_token_categories_, d_token_allow_,
                                                         d_allowed_mask_, vocab_size,
                                                         /*n_classified=*/vocab_size_, /*use_allow=*/true);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }

    // Compute masks
    uint16_t cat_mask = compute_category_mask();
    compute_token_allow_mask(cat_mask);

    // Empty-allow guard: an over-tight schema/state combination (e.g.
    // {"type":"object"} without properties — the key phase knows no legal
    // key) can reject every token. All logits then go to -FLT_MAX and greedy
    // argmax degenerates to token id 0 ("!!!!" spam on byte-level BPE
    // vocabs). Force a clean EOS finish instead.
    if (need_token_allow_) {
        size_t n_allowed = 0;
        for (int i = 0; i < vocab_size_ && n_allowed == 0; i++) {
            if (token_allow_[i] && (token_categories_[i] & cat_mask) != 0)
                n_allowed++;
        }
        if (n_allowed == 0) {
            std::fill(token_allow_.begin(), token_allow_.end(), (uint8_t)0);
            for (int32_t e : eos_tokens_)
                if (e >= 0 && e < vocab_size_)
                    token_allow_[e] = 1;
            cat_mask = 0xFFFF;
            IMP_LOG_WARN("SchemaConstrainer: no token satisfies the schema in phase %d — allowing EOS",
                         std::to_underlying(top().phase));
        }
    }

    IMP_LOG_DEBUG("SchemaConstrainer::apply_mask phase=%d cat_mask=0x%04x need_allow=%d stack=%zu",
                  std::to_underlying(top().phase), cat_mask, need_token_allow_, stack_.size());

    // Upload category mask
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_allowed_mask_, &cat_mask, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    // Upload token allow mask if needed
    if (need_token_allow_) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_allow_, token_allow_.data(), vocab_size_ * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream));
    }

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    // vocab_size is the LOGITS width (model vocab); the category/allow buffers
    // only cover the tokenizer vocab — padding logits are masked via
    // n_classified (SafeTensors lm_head padding, see json_constrain.cu).
    constrain_mask_allow_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_token_categories_, d_token_allow_,
                                                                d_allowed_mask_, vocab_size,
                                                                /*n_classified=*/vocab_size_,
                                                                need_token_allow_);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Update FSM with sampled token
// ---------------------------------------------------------------------------

void SchemaConstrainer::update(int32_t token) {
    if (token < 0 || token >= vocab_size_)
        return;

    const auto& text = token_texts_[token];
    // Feed the gate BEFORE the empty-stack guard: in strict optional mode the
    // stack is empty during free generation, and the gate must still see every
    // token to detect the tool-call opener.
    if (preamble_.absorb(token, text)) {
        // Opener just seen in strict mode → install the body FSM (once: the
        // push makes the stack non-empty so this cannot re-fire).
        if (strict_optional_envelope_ && preamble_.in_tool_args() && stack_.empty())
            engage_tool_body();
        return;
    }
    if (stack_.empty())
        return;
    SchemaPhase before = top().phase;
    for (char c : text) {
        if (!sim_advance(stack_, c))
            break;  // illegal char (mask should have prevented this) — stop early
    }
    // parallel_tool_calls (#1002): a strict tool-call body just drained the
    // stack (envelope close consumed). Instead of forcing EOS, re-arm the gate
    // so the model may open another `<tool_call>` (fresh body FSM) or stop.
    if (strict_optional_envelope_ && allow_parallel_ && stack_.empty() && preamble_.in_tool_args()) {
        preamble_.rearm_after_call();
        return;
    }
    if (!stack_.empty()) {
        IMP_LOG_DEBUG("SchemaConstrainer::update token=%d [%s] phase %d->%d stack=%zu", token, text.c_str(),
                      std::to_underlying(before), std::to_underlying(top().phase), stack_.size());
    }
}

// ---------------------------------------------------------------------------
// Jump-ahead (#844): forced-continuation text probe
// ---------------------------------------------------------------------------

int SchemaConstrainer::forced_text(std::string& out, int max_chars) const {
    out.clear();
    // No forcing during the preamble (the mask is inactive there) or once
    // the root value completed (the final EOS stays a normal masked step).
    if (!initialized_ || preamble_.active() || stack_.empty())
        return 0;

    // Walk a cloned frame stack. Per state, derive a SUPERSET of the legal
    // next chars from the phase (small: structural chars, matching property
    // / enum next-chars, the literal target) and test each candidate via
    // sim_advance — the grammar's single source of truth, so required-key
    // gating, trailing-comma bans and enum/key prefix narrowing all apply.
    // Exactly one legal candidate → the char is forced; append and advance
    // the clone. Phases whose legal-char set is open-ended (free string
    // content, escapes, numbers, patterns) stop the walk — never force
    // there.
    std::vector<SchemaFrame> stk = stack_;
    while (static_cast<int>(out.size()) < max_chars && !stk.empty()) {
        const SchemaFrame& f = stk.back();
        std::string cands;
        switch (f.phase) {
            case SchemaPhase::VALUE_START: {
                if (!f.node)
                    return static_cast<int>(out.size());
                switch (f.node->type) {
                    case SchemaType::OBJECT:
                        cands = "{";
                        break;
                    case SchemaType::TOOL_CALL:
                        cands = "{";
                        break;
                    case SchemaType::XML_TOOL_CALL:
                        cands = "<";
                        break;
                    case SchemaType::ARRAY:
                        cands = "[";
                        break;
                    case SchemaType::STRING:
                        cands = "\"";
                        break;  // opening quote; interior is free
                    case SchemaType::ENUM:
                        cands = "\"";
                        break;
                    case SchemaType::NULL_TYPE:
                        cands = "n";
                        break;
                    case SchemaType::BOOLEAN:
                        cands = "tf";
                        break;  // two legal starts — stops below
                    default:
                        // number/integer/anyOf/unknown: first char is a real choice
                        return static_cast<int>(out.size());
                }
                break;
            }
            case SchemaPhase::OBJECT_OPEN:
                cands = "\"}";  // sim_advance rejects '}' while required keys are unmet
                break;
            case SchemaPhase::OBJECT_KEY: {
                // Next chars of unemitted properties matching the buffer;
                // '"' closes iff the buffer is a complete key (sim decides).
                // Without a node the key is unconstrained (any string char
                // is legal) — the candidate set below would not be a
                // superset, so never force there.
                if (!f.node)
                    return static_cast<int>(out.size());
                cands = "\"";
                for (const auto& [name, prop] : f.node->properties) {
                    (void)prop;
                    if (f.emitted_keys.count(name))
                        continue;
                    if (f.node->type == SchemaType::TOOL_CALL &&
                        !tool_call_key_available(name, f.emitted_keys))
                        continue;
                    if (name.size() > f.key_buffer.size() &&
                        name.compare(0, f.key_buffer.size(), f.key_buffer) == 0)
                        cands += name[f.key_buffer.size()];
                }
                break;
            }
            case SchemaPhase::OBJECT_AFTER_KEY:
            case SchemaPhase::OBJECT_COLON:
                cands = ":";
                break;
            case SchemaPhase::OBJECT_AFTER_VALUE:
                cands = ",}";  // ',' illegal when no key can follow; '}' while required unmet
                break;
            case SchemaPhase::ARRAY_AFTER_ITEM:
                cands = ",]";  // min/maxItems can force either
                break;
            case SchemaPhase::ENUM_VALUE: {
                // Next chars of enum values matching the buffer; '"' iff a
                // matching value is already complete.
                if (!f.node)
                    return static_cast<int>(out.size());
                for (const auto& v : f.node->enum_values) {
                    if (v.size() < f.enum_buffer.size() ||
                        v.compare(0, f.enum_buffer.size(), f.enum_buffer) != 0)
                        continue;
                    cands += v.size() == f.enum_buffer.size() ? '"' : v[f.enum_buffer.size()];
                }
                break;
            }
            case SchemaPhase::LITERAL_VALUE:
                if (f.literal_pos < static_cast<int>(f.literal_target.size()))
                    cands = f.literal_target[f.literal_pos];
                break;
            case SchemaPhase::ENVELOPE_OPEN:
            case SchemaPhase::ENVELOPE_CLOSE:
                // Envelope literals are fully forced (jump-ahead drafts them).
                if (f.literal_pos < static_cast<int>(f.literal_target.size()))
                    cands = f.literal_target[f.literal_pos];
                break;
            case SchemaPhase::XML_FN_OPEN:
                if (f.literal_pos < static_cast<int>(f.literal_target.size()))
                    cands = f.literal_target[f.literal_pos];
                break;
            case SchemaPhase::XML_FN_NAME: {
                // Next chars of tool names matching the buffer; '>' iff a
                // matching name is already complete.
                if (!f.node)
                    return static_cast<int>(out.size());
                for (const auto& v : f.node->enum_values) {
                    if (v.size() < f.enum_buffer.size() ||
                        v.compare(0, f.enum_buffer.size(), f.enum_buffer) != 0)
                        continue;
                    cands += v.size() == f.enum_buffer.size() ? '>' : v[f.enum_buffer.size()];
                }
                break;
            }
            case SchemaPhase::XML_PARAMS: {
                // Superset of both tag targets' next char — sim_advance
                // filters by required/param availability.
                if (f.key_buffer.size() < kXmlParamOpen.size())
                    cands += kXmlParamOpen[f.key_buffer.size()];
                if (f.key_buffer.size() < kXmlFnClose.size())
                    cands += kXmlFnClose[f.key_buffer.size()];
                break;
            }
            case SchemaPhase::XML_PARAM_KEY: {
                const SchemaNode* tool = f.xml_tool;
                if (!tool)
                    return static_cast<int>(out.size());
                cands = ">";
                for (const auto& [name, prop] : tool->properties) {
                    (void)prop;
                    if (f.emitted_keys.count(name))
                        continue;
                    if (name.size() > f.key_buffer.size() &&
                        name.compare(0, f.key_buffer.size(), f.key_buffer) == 0)
                        cands += name[f.key_buffer.size()];
                }
                break;
            }
            case SchemaPhase::XML_RAW_VALUE:
                if (f.xml_value_open)
                    return static_cast<int>(out.size());  // free text — never force
                cands = "\n";  // the forced value-opening newline
                break;
            default:
                // ARRAY_OPEN (item-start vs ']' choice), STRING_VALUE /
                // STRING_PATTERN / STRING_ESCAPE (open-ended), NUMBER_VALUE
                // (digit choices), DONE.
                return static_cast<int>(out.size());
        }

        char forced = 0;
        int legal = 0;
        std::string seen;
        for (char c : cands) {
            if (seen.find(c) != std::string::npos)
                continue;
            seen += c;
            std::vector<SchemaFrame> probe = stk;
            if (sim_advance(probe, c)) {
                forced = c;
                if (++legal > 1)
                    break;
            }
        }
        if (legal != 1 || !sim_advance(stk, forced))
            break;
        out.push_back(forced);
    }
    return static_cast<int>(out.size());
}

// After a value frame pops, advance the parent frame's phase. Shared by the
// transition simulator below.
static void sim_fixup_parent(std::vector<SchemaFrame>& stk) {
    if (stk.empty())
        return;
    SchemaFrame& parent = stk.back();
    if (parent.phase == SchemaPhase::OBJECT_COLON)
        parent.phase = SchemaPhase::OBJECT_AFTER_VALUE;
    else if (parent.phase == SchemaPhase::ARRAY_OPEN || parent.phase == SchemaPhase::ARRAY_AFTER_ITEM) {
        parent.phase = SchemaPhase::ARRAY_AFTER_ITEM;
        parent.item_count++;  // one completed item per child-frame pop
    }
}

// ---------------------------------------------------------------------------
// Transition simulator — the single source of truth for the schema grammar.
// Drives the real update path (on stack_) and per-token mask legality (on a
// cloned stack). Returns false on any illegal transition, so a multi-char
// token that spans phase transitions (`{}`, `":"`, `"Why`, integer `0.98`,
// trailing comma) is rejected as a whole rather than slipping past the
// first-char category mask.
// ---------------------------------------------------------------------------
bool SchemaConstrainer::sim_advance(std::vector<SchemaFrame>& stk, char c) const {
    if (stk.empty())
        return false;  // trailing content after the root value completed

    SchemaFrame& f = stk.back();
    auto push_value = [&](const SchemaNode* node) {
        SchemaFrame nf;
        nf.node = resolve_schema_ref(schema_.get(), node);
        nf.phase = SchemaPhase::VALUE_START;
        stk.push_back(std::move(nf));
    };
    auto required_satisfied = [](const SchemaFrame& fr) {
        if (!fr.node)
            return true;
        for (auto& req : fr.node->required)
            if (!fr.emitted_keys.count(req))
                return false;
        return true;
    };
    auto has_unemitted_property = [](const SchemaFrame& fr) {
        if (!fr.node)
            return true;  // unknown object — can't tell, allow another key
        for (auto& [name, _] : fr.node->properties)
            if (!fr.emitted_keys.count(name))
                return true;
        return false;
    };
    const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;

    switch (f.phase) {
        case SchemaPhase::VALUE_START: {
            if (space)
                return true;
            if (!f.node) {
                stk.pop_back();  // unconstrained value — accept opaquely
                return true;
            }
            switch (f.node->type) {
                case SchemaType::OBJECT:
                    // Depth cap: recursive $ref schemas allow unbounded nesting;
                    // refuse to open deeper than ~96 levels (any finite nesting
                    // still satisfies the schema, so this only forces closure).
                    if (c == '{') {
                        if (stk.size() >= kMaxSchemaStackDepth)
                            return false;
                        f.phase = SchemaPhase::OBJECT_OPEN;
                        return true;
                    }
                    return false;
                case SchemaType::ARRAY:
                    if (c == '[') {
                        if (stk.size() >= kMaxSchemaStackDepth)
                            return false;
                        f.phase = SchemaPhase::ARRAY_OPEN;
                        return true;
                    }
                    return false;
                case SchemaType::STRING:
                    if (c == '"') {
                        if ((f.node->pattern_nfa && f.node->pattern_nfa->compiled()) ||
                            f.node->min_length >= 0 || f.node->max_length >= 0) {
                            f.phase = SchemaPhase::STRING_PATTERN;
                            f.string_len = 0;
                            if (f.node->pattern_nfa && f.node->pattern_nfa->compiled())
                                f.regex_states = f.node->pattern_nfa->start_set();
                            else
                                f.regex_states.clear();
                        } else {
                            f.phase = SchemaPhase::STRING_VALUE;
                        }
                        return true;
                    }
                    return false;
                case SchemaType::NUMBER:
                case SchemaType::INTEGER:
                    if (c == '-' || (c >= '0' && c <= '9')) {
                        f.phase = SchemaPhase::NUMBER_VALUE;
                        f.string_len = (c >= '0' && c <= '9') ? 1 : 0;  // digit count
                        f.num_leading_zero = (c == '0');                // "0..." forbids more int digits
                        return true;
                    }
                    return false;
                case SchemaType::BOOLEAN:
                    if (c == 't') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "true";
                        f.literal_pos = 1;
                        return true;
                    }
                    if (c == 'f') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "false";
                        f.literal_pos = 1;
                        return true;
                    }
                    return false;
                case SchemaType::NULL_TYPE:
                    if (c == 'n') {
                        f.phase = SchemaPhase::LITERAL_VALUE;
                        f.literal_target = "null";
                        f.literal_pos = 1;
                        return true;
                    }
                    return false;
                case SchemaType::ENUM:
                    if (c == '"') {
                        f.phase = SchemaPhase::ENUM_VALUE;
                        f.enum_buffer.clear();
                        return true;
                    }
                    return false;
                case SchemaType::ANY_OF:
                    // anyOf is hard to constrain precisely — accept as free string.
                    f.phase = SchemaPhase::STRING_VALUE;
                    return true;
                case SchemaType::TOOL_CALL:
                    // Tool-call body — an object with ordered keys (#1002).
                    if (c == '{') {
                        if (stk.size() >= kMaxSchemaStackDepth)
                            return false;
                        f.phase = SchemaPhase::OBJECT_OPEN;
                        return true;
                    }
                    return false;
                case SchemaType::XML_TOOL_CALL:
                    // Qwen-Coder XML body — the <function= tag opens it (any
                    // leading whitespace was consumed by `space` above).
                    if (c == '<') {
                        f.phase = SchemaPhase::XML_FN_OPEN;
                        f.literal_target = kXmlFnOpen;
                        f.literal_pos = 1;
                        return true;
                    }
                    return false;
                default:
                    return false;
            }
        }

        case SchemaPhase::OBJECT_OPEN: {
            if (space)
                return true;
            if (c == '}') {
                // After a comma a key is mandatory — closing here is a trailing
                // comma. Otherwise close only once required keys are present.
                if (f.after_comma || !required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (c == '"') {
                f.phase = SchemaPhase::OBJECT_KEY;
                f.key_buffer.clear();
                f.after_comma = false;
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_KEY: {
            if (c == '"') {
                // Close the key only if it is a complete, not-yet-emitted property.
                bool complete = false;
                if (f.node) {
                    for (auto& [name, _] : f.node->properties) {
                        if (f.node->type == SchemaType::TOOL_CALL &&
                            !tool_call_key_available(name, f.emitted_keys))
                            continue;
                        if (!f.emitted_keys.count(name) && name == f.key_buffer) {
                            complete = true;
                            break;
                        }
                    }
                }
                if (!complete)
                    return false;
                f.current_key = f.key_buffer;
                f.emitted_keys.insert(f.current_key);
                f.phase = SchemaPhase::OBJECT_AFTER_KEY;
                return true;
            }
            // No escapes in keys (#850): accepting '\' and dropping it let
            // the NEXT char match the property prefix while the emitted
            // text carried the escape — `{"\number_x":5}` passed the mask.
            // Property names are matched on raw chars (escapes were never
            // decoded), so no legal key needs one.
            if (c == '\\')
                return false;
            if (!f.node || !is_valid_key_prefix(f.node, f.key_buffer + c, f.emitted_keys))
                return false;
            f.key_buffer += c;
            return true;
        }

        case SchemaPhase::OBJECT_AFTER_KEY: {
            if (space)
                return true;
            if (c == ':') {
                f.phase = SchemaPhase::OBJECT_COLON;
                const SchemaNode* prop = f.node ? find_property(f.node, f.current_key) : nullptr;
                // TOOL_CALL "arguments" resolves to the parameter schema of
                // the tool chosen by the completed "name" enum (#1002).
                if (f.node && f.node->type == SchemaType::TOOL_CALL && f.current_key == "arguments") {
                    // Reverse scan: tool entries follow their hoisted
                    // "<tool>/<def>" entries, so a tool name colliding with a
                    // hoisted key still binds its own entry (see
                    // xml_tool_params).
                    prop = nullptr;
                    for (auto it = schema_->defs.rbegin(); it != schema_->defs.rend(); ++it) {
                        if (it->first == f.chosen_tool) {
                            prop = it->second.get();
                            break;
                        }
                    }
                }
                push_value(prop);
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_COLON: {
            // The value is normally handled by the pushed sub-frame; this branch
            // only fires for robustness if a comma/brace reaches the colon frame.
            if (space)
                return true;
            if (c == ',') {
                if (!has_unemitted_property(f))
                    return false;  // no more keys possible — comma would dangle
                f.phase = SchemaPhase::OBJECT_OPEN;
                f.after_comma = true;
                return true;
            }
            if (c == '}') {
                if (!required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::OBJECT_AFTER_VALUE: {
            if (space)
                return true;
            if (c == ',') {
                if (!has_unemitted_property(f))
                    return false;  // every property emitted — comma would dangle
                f.phase = SchemaPhase::OBJECT_OPEN;
                f.after_comma = true;
                return true;
            }
            if (c == '}') {
                if (!required_satisfied(f))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::ARRAY_OPEN: {
            if (space)
                return true;
            if (c == ']') {
                if (f.node && f.node->min_items > 0)
                    return false;  // empty array below minItems
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (f.node && f.node->items) {
                push_value(f.node->items.get());
                return sim_advance(stk, c);  // process first-item char in new frame
            }
            return true;  // array without an items schema — accept opaquely
        }

        case SchemaPhase::ARRAY_AFTER_ITEM: {
            if (space)
                return true;
            const int max_items = effective_max_items(schema_.get(), f.node);
            const bool can_close = !f.node || f.node->min_items < 0 || f.item_count >= f.node->min_items;
            if (c == ',') {
                // At the item ceiling the array must close (unless closing is
                // itself illegal via contradictory minItems — then allow the
                // comma rather than deadlock; the mask mirrors this).
                if (f.item_count >= max_items && can_close)
                    return false;
                if (f.node && f.node->items)
                    push_value(f.node->items.get());
                return true;
            }
            if (c == ']') {
                if (!can_close && f.item_count < max_items)
                    return false;  // below minItems and comma is still legal
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            return false;
        }

        case SchemaPhase::STRING_VALUE: {
            if (c == '\\') {
                f.phase = SchemaPhase::STRING_ESCAPE;
                return true;
            }
            if (c == '"') {
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            // JSON forbids raw control chars (U+0000–U+001F) inside strings —
            // they must arrive escaped (\n, \uXXXX). Multi-char tokens whose
            // first char passes the category mask (e.g. `"<newline>`) used to
            // smuggle them through, producing unparseable output.
            if (static_cast<unsigned char>(c) < 0x20)
                return false;
            return true;  // any content char
        }

        case SchemaPhase::STRING_PATTERN: {
            if (c == '"') {
                if (!can_close_string(f, f.regex_states, f.string_len))
                    return false;
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (static_cast<unsigned char>(c) < 0x20)
                return false;  // raw control char — see STRING_VALUE
            if (f.node && f.node->max_length >= 0 && f.string_len + 1 > f.node->max_length)
                return false;
            if (f.node && f.node->pattern_nfa && f.node->pattern_nfa->compiled()) {
                std::vector<int> next = f.node->pattern_nfa->step(f.regex_states,
                                                                  static_cast<unsigned char>(c));
                if (next.empty())
                    return false;  // pattern prefix died
                f.regex_states = std::move(next);
            }
            f.string_len++;
            return true;
        }

        case SchemaPhase::STRING_ESCAPE: {
            if (static_cast<unsigned char>(c) < 0x20)
                return false;  // `\` + raw control char is not a legal escape
            f.phase = SchemaPhase::STRING_VALUE;
            return true;
        }

        case SchemaPhase::NUMBER_VALUE: {
            const bool is_int = f.node && f.node->type == SchemaType::INTEGER;
            if (c >= '0' && c <= '9') {
                if (f.string_len == 0) {  // first digit (came after a leading '-')
                    f.num_leading_zero = (c == '0');
                    f.string_len = 1;
                    return true;
                }
                if (f.num_leading_zero)
                    return false;  // JSON forbids leading zeros: `0` then digit
                f.string_len++;
                return true;
            }
            if (!is_int && (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) {
                f.num_leading_zero = false;  // fractional/exponent part — int rule done
                return true;
            }
            // Any other char ends the number — legal only if >=1 digit was seen.
            if (f.string_len == 0)
                return false;
            stk.pop_back();
            sim_fixup_parent(stk);
            return sim_advance(stk, c);  // reprocess the terminator in the parent
        }

        case SchemaPhase::LITERAL_VALUE: {
            if (f.literal_pos >= static_cast<int>(f.literal_target.size()))
                return false;
            if (c != f.literal_target[f.literal_pos])
                return false;  // wrong char for true/false/null
            f.literal_pos++;
            if (f.literal_pos >= static_cast<int>(f.literal_target.size())) {
                stk.pop_back();
                sim_fixup_parent(stk);
            }
            return true;
        }

        case SchemaPhase::ENUM_VALUE: {
            if (c == '"') {
                bool exact = false;
                if (f.node) {
                    for (auto& v : f.node->enum_values)
                        if (v == f.enum_buffer) {
                            exact = true;
                            break;
                        }
                }
                if (!exact)
                    return false;  // close only on an exact enum value
                // TOOL_CALL name binding (#1002): the completed "name" enum
                // selects which parameter schema "arguments" resolves to.
                if (stk.size() >= 2) {
                    SchemaFrame& parent = stk[stk.size() - 2];
                    if (parent.node && parent.node->type == SchemaType::TOOL_CALL &&
                        parent.current_key == "name")
                        parent.chosen_tool = f.enum_buffer;
                }
                stk.pop_back();
                sim_fixup_parent(stk);
                return true;
            }
            if (!f.node || !is_valid_enum_prefix(f.node->enum_values, f.enum_buffer + c))
                return false;
            f.enum_buffer += c;
            return true;
        }

        case SchemaPhase::ENVELOPE_OPEN: {
            // Optional whitespace before the open literal (models emit "\n\n"
            // after </think>); inside the literal every char is forced.
            if (f.literal_pos == 0 && space)
                return true;
            if (f.literal_pos < static_cast<int>(f.literal_target.size()) &&
                c == f.literal_target[f.literal_pos]) {
                f.literal_pos++;
                if (f.literal_pos == static_cast<int>(f.literal_target.size())) {
                    // Open literal complete: arm the close literal on this
                    // frame and host the root value on top of it.
                    f.phase = SchemaPhase::ENVELOPE_CLOSE;
                    f.literal_target = envelope_close_;
                    f.literal_pos = 0;
                    push_value(f.node);
                }
                return true;
            }
            return false;
        }

        case SchemaPhase::ENVELOPE_CLOSE: {
            if (f.literal_pos < static_cast<int>(f.literal_target.size()) &&
                c == f.literal_target[f.literal_pos]) {
                f.literal_pos++;
                if (f.literal_pos == static_cast<int>(f.literal_target.size())) {
                    stk.pop_back();  // envelope done — stack drains, EOS is forced
                    sim_fixup_parent(stk);
                }
                return true;
            }
            return false;
        }

        case SchemaPhase::XML_FN_OPEN: {
            if (f.literal_pos < static_cast<int>(f.literal_target.size()) &&
                c == f.literal_target[f.literal_pos]) {
                f.literal_pos++;
                if (f.literal_pos == static_cast<int>(f.literal_target.size())) {
                    f.phase = SchemaPhase::XML_FN_NAME;
                    f.enum_buffer.clear();
                }
                return true;
            }
            return false;
        }

        case SchemaPhase::XML_FN_NAME: {
            if (c == '>') {
                bool exact = false;
                if (f.node) {
                    for (auto& v : f.node->enum_values)
                        if (v == f.enum_buffer) {
                            exact = true;
                            break;
                        }
                }
                if (!exact)
                    return false;  // the tag closes only on a complete tool name
                f.chosen_tool = f.enum_buffer;
                // Bind the parameter schema ONCE — the phases below run per
                // simulated char over the whole vocab, a per-char defs scan
                // there is hot-path cost for nothing.
                f.xml_tool = xml_tool_params(schema_.get(), f.chosen_tool);
                if (!f.xml_tool)
                    return false;
                f.phase = SchemaPhase::XML_PARAMS;
                f.key_buffer.clear();
                return true;
            }
            if (!f.node || !is_valid_enum_prefix(f.node->enum_values, f.enum_buffer + c))
                return false;
            f.enum_buffer += c;
            return true;
        }

        case SchemaPhase::XML_PARAMS: {
            // On a fresh line: "\n<parameter=" while unemitted params remain,
            // "\n</function>" once the required set is emitted.
            const SchemaNode* tool = f.xml_tool;
            if (!tool)
                return false;
            bool can_param = false;
            for (auto& [pname, _] : tool->properties) {
                if (!f.emitted_keys.count(pname)) {
                    can_param = true;
                    break;
                }
            }
            bool can_close = true;
            for (auto& req : tool->required) {
                if (!f.emitted_keys.count(req)) {
                    can_close = false;
                    break;
                }
            }
            const std::string next = f.key_buffer + c;
            const bool param_pfx = can_param && kXmlParamOpen.compare(0, next.size(), next) == 0;
            const bool close_pfx = can_close && kXmlFnClose.compare(0, next.size(), next) == 0;
            if (!param_pfx && !close_pfx)
                return false;
            f.key_buffer = next;
            if (param_pfx && next.size() == kXmlParamOpen.size()) {
                f.phase = SchemaPhase::XML_PARAM_KEY;
                f.key_buffer.clear();
            } else if (close_pfx && next.size() == kXmlFnClose.size()) {
                stk.pop_back();  // body complete — the envelope frame takes over
                sim_fixup_parent(stk);
            }
            return true;
        }

        case SchemaPhase::XML_PARAM_KEY: {
            const SchemaNode* tool = f.xml_tool;
            if (!tool)
                return false;
            if (c == '>') {
                bool complete = false;
                for (auto& [pname, _] : tool->properties) {
                    if (!f.emitted_keys.count(pname) && pname == f.key_buffer) {
                        complete = true;
                        break;
                    }
                }
                if (!complete)
                    return false;
                f.emitted_keys.insert(f.key_buffer);
                f.key_buffer.clear();
                f.phase = SchemaPhase::XML_RAW_VALUE;
                f.xml_value_open = false;  // forced opening newline not yet seen
                f.xml_delim_match = 0;
                return true;
            }
            if (!is_valid_key_prefix(tool, f.key_buffer + c, f.emitted_keys))
                return false;
            f.key_buffer += c;
            return true;
        }

        case SchemaPhase::XML_RAW_VALUE: {
            if (!f.xml_value_open) {
                // The value region opens with a forced newline: <parameter=KEY>\n
                if (c != '\n')
                    return false;
                f.xml_value_open = true;
                // The opener doubles as the delimiter's first char: a model
                // closing an empty value as '>\n</parameter>' (one newline,
                // not the template's canonical '>\n\n</parameter>') must not
                // have its close tag swallowed as value text — the value
                // would then never close and EOS stays masked to max_tokens.
                f.xml_delim_match = 1;
                return true;
            }
            // Any char is legal value text; only the delimiter tracker moves.
            f.xml_delim_match = xml_delim_step(kXmlParamDelim, f.xml_delim_match, c);
            if (f.xml_delim_match == static_cast<int>(kXmlParamDelim.size())) {
                // "\n</parameter>" complete — back to the tag choice.
                f.phase = SchemaPhase::XML_PARAMS;
                f.key_buffer.clear();
                f.xml_delim_match = 0;
            }
            return true;
        }

        case SchemaPhase::DONE:
            return false;
    }
    return false;
}

bool SchemaConstrainer::token_legal(const std::string& text) const {
    if (text.empty())
        return true;                        // EOS / special tokens — governed by the category mask
    std::vector<SchemaFrame> sim = stack_;  // deep copy of the frame stack
    for (char c : text) {
        if (!sim_advance(sim, c))
            return false;
    }
    return true;
}

}  // namespace imp

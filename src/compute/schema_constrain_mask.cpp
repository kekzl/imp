// SchemaConstrainer per-step token mask (host): compute_token_allow_mask and its CPU test hook.
#include "compute/json_constrain.h"  // token categories
#include "compute/constrain_common.h"
#include "compute/schema_constrain.h"
#include "compute/schema_xml_delim.h"
#include <algorithm>
#include <array>

namespace imp {

std::vector<uint8_t> SchemaConstrainer::mask_for_test(const std::vector<std::string>& texts) {
    vocab_size_ = static_cast<int>(texts.size());
    token_texts_ = texts;
    token_categories_.resize(texts.size());
    for (size_t i = 0; i < texts.size(); i++)
        token_categories_[i] = classify_token(texts[i]);
    token_allow_.assign(texts.size(), 1);
    token_plain_.clear();
    const uint16_t cat = compute_category_mask();
    compute_token_allow_mask(cat);
    std::vector<uint8_t> out(texts.size());
    for (size_t i = 0; i < texts.size(); i++)
        out[i] = (token_categories_[i] & cat) != 0 && (!need_token_allow_ || token_allow_[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Compute per-token allow mask (for key names and enum values)
// ---------------------------------------------------------------------------

void SchemaConstrainer::compute_token_allow_mask(uint16_t cat_mask) {
    need_token_allow_ = false;

    if (stack_.empty())
        return;

    // Full per-token legality: a candidate is allowed only if simulating its whole text stays
    // legal at every char - catches multi-char tokens spanning phase transitions (`{}` on
    // unmet required keys, `":"` as a bogus enum value, `0.98` for an integer) that the
    // first-char category mask misses. Category mask still runs alongside (governs
    // EOS/whitespace/structural first-char); a token must pass BOTH.
    // Cost control (runs per decode step over the whole vocab; token_legal deep-copies the
    // frame stack per candidate): category prefilter skips simulation for tokens the category
    // mask already fails; in a free string value, any token without '"','\\' or a raw control
    // char stays inside the string by construction, so simulation is skipped (pattern/enum/key
    // strings still simulate).
    need_token_allow_ = true;
    const SchemaPhase phase = top().phase;
    // A free value whose grammar sits inside a string is the same regime as a
    // schema-described free string, and string content is most of what a free
    // value costs, so give it the same O(1) shortcut (#1729).
    const bool free_value_phase = (phase == SchemaPhase::FREE_VALUE);
    const bool free_string = (phase == SchemaPhase::STRING_VALUE) ||
                             (free_value_phase && top().free_grammar.current_state == JsonState::IN_STRING);
    // XML cost control: XML phases delegate the whole vocab to per-token simulation
    // (category 0xFFFF), which deep-copies the frame stack per candidate (unchecked, near the
    // full vocab per step). Two shortcuts: tag phases probe all 256 first chars once on a
    // cloned stack and reject by first byte; open raw values accept any text and are illegal
    // only if the "\n</parameter>" delimiter completes inside them (run the int-only delimiter
    // automaton first; only completing tokens pay the full simulation).
    const bool xml_raw = (phase == SchemaPhase::XML_RAW_VALUE);
    // An enum-typed value is not free text: it takes the per-token simulation below.
    const bool xml_raw_open = xml_raw && top().xml_value_open && !top().xml_enum;
    // The envelope phases and the XML body's VALUE_START are the same
    // full-vocab-delegation regime (category 0xFFFF) with a 1-2 char legal
    // set — include them in the first-byte prefilter.
    const bool xml_tag_phase = phase == SchemaPhase::XML_FN_OPEN || phase == SchemaPhase::XML_FN_NAME ||
                               phase == SchemaPhase::XML_PARAMS || phase == SchemaPhase::XML_PARAM_KEY ||
                               (xml_raw && (!top().xml_value_open || top().xml_enum)) ||
                               phase == SchemaPhase::ENVELOPE_OPEN || phase == SchemaPhase::ENVELOPE_CLOSE ||
                               (phase == SchemaPhase::VALUE_START && top().node &&
                                top().node->type == SchemaType::XML_TOOL_CALL);
    // FREE_VALUE delegates the whole vocabulary to the simulation exactly as the
    // XML phases do, so it needs the same first-byte prefilter or it pays ~150k
    // stack copies per decode step. Every phase whose verdict is token_legal gets it too
    // (a first byte that fails fails the token): keys, enums, patterns, escapes cost 20-36 ms
    // per step without it (Qwen3.8 vocab).
    const bool first_byte_prefilter = xml_tag_phase || free_value_phase ||
                                      phase == SchemaPhase::ENUM_LITERAL || (!xml_raw_open && !free_string);
    if (token_plain_.size() != token_texts_.size()) {
        token_plain_.resize(token_texts_.size());
        for (size_t i = 0; i < token_texts_.size(); i++)
            token_plain_[i] = std::none_of(token_texts_[i].begin(), token_texts_[i].end(), [](char c) {
                return c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20;
            });
    }
    // first_ok[c]: the stack accepts byte c. When at most 16 first bytes pass, second_ok holds
    // the same probe one byte deeper for each of them (exact, like first_ok: token_legal needs
    // every prefix legal). Both gate only the token_legal calls below.
    bool first_ok[256];
    int n_first = 0;
    for (int c = 0; c < 256; c++) {
        std::vector<SchemaFrame> probe = stack_;
        first_ok[c] = sim_advance(probe, static_cast<char>(c));
        n_first += first_ok[c];
    }
    int16_t second_row[256];
    std::vector<std::array<bool, 256>> second_ok;
    std::fill(std::begin(second_row), std::end(second_row), int16_t(-1));
    if (n_first <= 16) {
        for (int c = 0; c < 256; c++) {
            if (!first_ok[c])
                continue;
            second_row[c] = static_cast<int16_t>(second_ok.size());
            auto& row = second_ok.emplace_back();
            std::vector<SchemaFrame> base = stack_;
            sim_advance(base, static_cast<char>(c));
            for (int d = 0; d < 256; d++) {
                std::vector<SchemaFrame> probe = base;
                row[d] = sim_advance(probe, static_cast<char>(d));
            }
        }
    }
    auto prefix_ok = [&](const std::string& text) {
        if (text.empty())
            return true;
        const auto c0 = static_cast<unsigned char>(text[0]);
        if (!first_ok[c0])
            return false;
        return text.size() < 2 || second_row[c0] < 0 ||
               second_ok[second_row[c0]][static_cast<unsigned char>(text[1])];
    };
    auto legal = [&](const std::string& text) { return prefix_ok(text) && token_legal(text); };
    for (int i = 0; i < vocab_size_; i++) {
        if ((token_categories_[i] & cat_mask) == 0) {
            token_allow_[i] = 0;  // masked by category — simulation irrelevant
            continue;
        }
        const std::string& text = token_texts_[i];
        if (first_byte_prefilter && !prefix_ok(text)) {
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
            token_allow_[i] = completes ? (legal(text) ? 1 : 0) : 1;
            continue;
        }
        if (free_string && !text.empty() && token_plain_[i]) {
            token_allow_[i] = 1;
            continue;
        }
        token_allow_[i] = legal(text) ? 1 : 0;
    }
    // EOS must not stop generation mid-value: its rendered text ("<|im_end|>") would pass the
    // anything-goes value scan above, so this check runs post-loop. The hazard is not XML-only
    // (#1199): "<|im_end|>" is plain ASCII, so classify_token tags it CAT_STRING_CHAR and the
    // category mask does not govern it inside a string. A model reaching EOS mid-string ended
    // the request with the document open, returning a 200 with invalid JSON.
    // Unconditional is correct: this function only runs on a non-empty stack; apply_mask
    // returns earlier when the root value is complete, the one place EOS is legal.
    for (int32_t e : eos_tokens_)
        if (e >= 0 && e < vocab_size_)
            token_allow_[e] = 0;
}

}  // namespace imp

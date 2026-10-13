// SchemaConstrainer phase steps that sim_advance (schema_constrain.cu) delegates to: anyOf
// branches, numbers with minimum/maximum, array items with prefixItems, property values with an
// additionalProperties schema, and the object close / next-key rules.
#include "compute/schema_constrain.h"

#include "compute/json_number_bounds.h"

#include <climits>

namespace imp {

namespace {

// The value a schema leaves undescribed ("items" absent, additionalProperties: true, `{}`).
const SchemaNode* any_node() {
    static const SchemaNode node = [] {
        SchemaNode n;
        n.type = SchemaType::ANY;
        return n;
    }();
    return &node;
}

int digit_cap(bool is_int) { return is_int ? kMaxIntegerDigits : kMaxNumberDigits; }

// '.', 'e'/'E', '+'/'-' inside a number: RFC 8259 sub-states (#1104). A bounded number stays a
// plain decimal (no exponent); `dot_ok` is the bounds' verdict on the '.'.
bool number_punct_step(SchemaFrame& f, char c, bool bounded, bool dot_ok) {
    if (c == '.') {
        if (f.num_frac || f.num_exp || f.num_need_digit || !dot_ok)
            return false;
        f.num_frac = true;
        f.num_need_digit = true;
    } else if (c == 'e' || c == 'E') {
        if (bounded || f.num_exp || f.num_need_digit)
            return false;
        f.num_exp = true;
        f.num_sign_ok = true;
        f.num_need_digit = true;
    } else {  // '+' / '-'
        if (!f.num_sign_ok)
            return false;
        f.num_sign_ok = false;
        f.num_need_digit = true;
    }
    f.num_leading_zero = false;  // fractional/exponent part — int rule done
    f.num_text += c;
    return true;
}

}  // namespace

void push_schema_value(std::vector<SchemaFrame>& stk, const SchemaNode* resolved) {
    SchemaFrame nf;
    nf.node = resolved;
    nf.phase = SchemaPhase::VALUE_START;
    stk.push_back(std::move(nf));
}

void sim_fixup_parent(std::vector<SchemaFrame>& stk) {
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

// additionalProperties:true (and bare {"type":"object"}, JSON Schema treats them the
// same): undeclared keys are legal. TOOL_CALL is excluded: its two keys are ORDERED and
// arguments bind a chosen tool's schema, so an extra key there is a desync (#1729).
bool object_allows_additional(const SchemaNode* n) {
    return n && n->type == SchemaType::OBJECT && n->additional_properties;
}

void push_free_value(std::vector<SchemaFrame>& stk) {
    SchemaFrame nf;
    nf.phase = SchemaPhase::FREE_VALUE;
    // Seeded in AFTER_COLON, not START: START accepts only '{' and '[', while an undescribed
    // value may be any JSON value. AFTER_COLON pushes AFTER_VALUE for every value form, so
    // that single stack entry is this frame's completion marker.
    nf.free_grammar.reset();
    nf.free_grammar.current_state = JsonState::AFTER_COLON;
    stk.push_back(std::move(nf));
}

bool SchemaConstrainer::any_of_start(std::vector<SchemaFrame>& stk, char c) const {
    SchemaFrame& f = stk.back();
    const size_t base = schema_stack_depth(stk);
    f.branches.clear();
    for (const auto& alt : f.node->any_of) {
        std::vector<SchemaFrame> branch(1);
        branch[0].phase = SchemaPhase::BRANCH_END;
        branch[0].depth_base = base;
        push_schema_value(branch, resolve_schema_ref(schema_.get(), alt.get()));
        f.branches.push_back(std::move(branch));
    }
    f.phase = SchemaPhase::ANY_OF_BRANCHES;
    return any_of_step(stk, c);
}

bool SchemaConstrainer::any_of_step(std::vector<SchemaFrame>& stk, char c) const {
    std::vector<std::vector<SchemaFrame>> live;
    bool done_with = false;    // a branch's value ended ON c
    bool done_before = false;  // a branch's value ended BEFORE c: c belongs to the parent
    for (const auto& branch : stk.back().branches) {
        std::vector<SchemaFrame> next = branch;
        const bool ok = sim_advance(next, c);
        const bool finished = next.size() == 1;  // only BRANCH_END left
        if (ok && finished)
            done_with = true;
        else if (ok)
            live.push_back(std::move(next));
        else if (finished)
            done_before = true;
    }
    if (done_with || (live.empty() && done_before)) {
        stk.pop_back();
        sim_fixup_parent(stk);
        return done_with || sim_advance(stk, c);
    }
    if (live.size() == 1) {  // one alternative left: back on the main stack, no branch cost
        stk.pop_back();
        for (size_t i = 1; i < live[0].size(); ++i)
            stk.push_back(std::move(live[0][i]));
        return true;
    }
    if (live.empty())
        return false;
    stk.back().branches = std::move(live);
    return true;
}

bool SchemaConstrainer::number_start(SchemaFrame& f, char c) {
    if (c != '-' && (c < '0' || c > '9'))
        return false;
    const bool is_int = f.node->type == SchemaType::INTEGER;
    if (f.node->bounds.active() &&
        !number_prefix_feasible(f.node->bounds, std::string(1, c), is_int, digit_cap(is_int)))
        return false;
    f.phase = SchemaPhase::NUMBER_VALUE;
    f.string_len = (c >= '0' && c <= '9') ? 1 : 0;  // digit count
    f.num_leading_zero = (c == '0');                // "0..." forbids more int digits
    f.num_frac = f.num_exp = f.num_sign_ok = false;
    f.num_need_digit = (c == '-');  // a bare '-' still owes a digit
    f.num_text.assign(1, c);
    return true;
}

bool SchemaConstrainer::number_step(std::vector<SchemaFrame>& stk, char c) const {
    SchemaFrame& f = stk.back();
    const bool is_int = f.node && f.node->type == SchemaType::INTEGER;
    const NumberBounds* bounds = f.node && f.node->bounds.active() ? &f.node->bounds : nullptr;
    auto feasible = [&](char next) {
        return !bounds || number_prefix_feasible(*bounds, f.num_text + next, is_int, digit_cap(is_int));
    };
    if (c >= '0' && c <= '9') {
        // JSON forbids leading zeros (`0` then digit). #1540: an `integer` stops at 19 digits
        // (int64); the next legal tokens are the ones that close it.
        if (f.string_len > 0 && (f.num_leading_zero || (is_int && f.string_len >= kMaxIntegerDigits)))
            return false;
        if (!feasible(c))
            return false;
        if (f.string_len == 0)  // first digit (came after a leading '-')
            f.num_leading_zero = (c == '0');
        f.string_len++;
        f.num_need_digit = false;
        f.num_sign_ok = false;
        f.num_text += c;
        return true;
    }
    if (!is_int && (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'))
        return number_punct_step(f, c, bounds != nullptr, c != '.' || feasible(c));
    // Any other char ends the number — legal only if >=1 digit was seen, the number does not
    // still owe one ("3.", "1e", "-") and it is in bounds.
    if (f.string_len == 0 || f.num_need_digit || (bounds && !number_in_bounds(*bounds, f.num_text)))
        return false;
    stk.pop_back();
    sim_fixup_parent(stk);
    return sim_advance(stk, c);  // reprocess the terminator in the parent
}

const SchemaNode* SchemaConstrainer::item_schema(const SchemaNode* arr, int index) const {
    if (!arr)
        return any_node();
    const auto i = static_cast<size_t>(index);
    if (i < arr->prefix_items.size())
        return resolve_schema_ref(schema_.get(), arr->prefix_items[i].get());
    if (arr->items_false)
        return nullptr;
    return arr->items ? resolve_schema_ref(schema_.get(), arr->items.get()) : any_node();
}

bool SchemaConstrainer::push_array_item(std::vector<SchemaFrame>& stk, int index) const {
    const SchemaNode* item = item_schema(stk.back().node, index);
    if (!item)
        return false;
    push_schema_value(stk, item);
    return true;
}

void SchemaConstrainer::push_property_value(std::vector<SchemaFrame>& stk, const SchemaNode* prop) const {
    const SchemaNode* obj = stk.back().node;
    if (!prop && object_allows_additional(obj))
        prop = obj->additional_schema ? obj->additional_schema.get() : any_node();
    push_schema_value(stk, resolve_schema_ref(schema_.get(), prop));
}

uint16_t SchemaConstrainer::value_start_categories(const SchemaNode* n) {
    if (!n)
        return CAT_VALUE_START;
    switch (n->type) {
        case SchemaType::OBJECT:
        case SchemaType::TOOL_CALL:
            return CAT_OPEN_BRACE;
        case SchemaType::ARRAY:
            return CAT_OPEN_BRACKET;
        case SchemaType::STRING:
            return CAT_QUOTE;
        case SchemaType::NUMBER:
        case SchemaType::INTEGER:
            return CAT_NUMBER_START;
        case SchemaType::BOOLEAN:
            return CAT_TRUE_START | CAT_FALSE_START;
        case SchemaType::NULL_TYPE:
            return CAT_NULL_START;
        case SchemaType::ENUM: {
            uint16_t mask = n->enum_values.empty() ? 0 : CAT_QUOTE;
            for (const auto& v : n->enum_literals)
                mask |= v[0] == 't'   ? CAT_TRUE_START
                        : v[0] == 'f' ? CAT_FALSE_START
                        : v[0] == 'n' ? CAT_NULL_START
                                      : CAT_NUMBER_START;
            return mask;
        }
        case SchemaType::XML_TOOL_CALL:
            // '<' (and tolerated leading whitespace) span categories — per-token simulation decides.
            return 0xFFFF;
        default:  // anyOf, any value
            return CAT_VALUE_START;
    }
}

bool SchemaConstrainer::object_may_close(const SchemaFrame& f) {
    if (!f.node)
        return true;
    for (const auto& req : f.node->required)
        if (!f.emitted_keys.count(req))
            return false;
    return f.node->min_properties < 0 || static_cast<int>(f.emitted_keys.size()) >= f.node->min_properties;
}

bool SchemaConstrainer::object_may_add_key(const SchemaFrame& f) {
    if (!f.node)
        return true;  // unknown object — can't tell, allow another key
    if (f.node->max_properties >= 0 && static_cast<int>(f.emitted_keys.size()) >= f.node->max_properties)
        return false;
    if (object_allows_additional(f.node))
        return true;  // an undeclared key can always follow
    for (const auto& [name, _] : f.node->properties)
        if (!f.emitted_keys.count(name))
            return true;
    return false;
}

bool SchemaConstrainer::completes_for_test(const std::string& text) const {
    std::vector<SchemaFrame> sim = stack_;
    for (char c : text)
        if (!sim_advance(sim, c))
            return false;
    return sim.empty();
}

}  // namespace imp

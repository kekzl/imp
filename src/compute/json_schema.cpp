#include "compute/json_schema.h"
#include "core/logging.h"
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace imp {

// Minimal JSON parser for schema documents (no external deps): objects, arrays, strings,
// numbers, booleans, null. No comments, no trailing commas.

namespace {

class SchemaParser {
public:
    SchemaParser(const char* data, size_t len) : data_(data), len_(len), pos_(0) {}

    std::unique_ptr<SchemaNode> parse() {
        skip_ws();
        auto node = parse_schema_object();
        // #1564: an object loop can desync and silently return a truncated tree.
        // Trailing input after the top-level object is the cheapest place to catch that.
        skip_ws();
        if (!eof())
            fail("trailing input after the schema object");
        return node;
    }

    // Definitions encountered anywhere in the document ($defs/definitions).
    // parse_json_schema() attaches them to the root node after parsing.
    std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>> collected_defs_;
    // Set when an unsupported $ref form is seen — the whole parse must fail
    // (constraining with a silently-dropped $ref would enforce a WRONG grammar).
    bool ref_error_ = false;
    bool ref_error() const { return ref_error_; }

    // Set when the document uses a keyword this build cannot enforce. Same
    // contract as ref_error_: parse fails, caller returns 400 rather than
    // silently answering a bounded request with an unbounded grammar.
    bool parse_error() const { return parse_error_; }
    const std::string& error_reason() const { return error_reason_; }

private:
    bool parse_error_ = false;
    std::string error_reason_;

    // Recursion depth of parse_schema_object(). A schema is request-supplied
    // text and the parser is recursive descent, so nesting depth maps 1:1 onto
    // stack frames (#1609). Real schemas nest a handful deep.
    int depth_ = 0;
    static constexpr int kMaxSchemaDepth = 64;

    // Scope-bound depth counter: every early return out of
    // parse_schema_object() has to decrement, and there are several.
    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& depth) : d(depth) { d++; }
        ~DepthGuard() { d--; }
    };

    bool fail(const std::string& why) {
        if (!parse_error_) {
            parse_error_ = true;
            error_reason_ = why;
        }
        return false;
    }

    const char* data_;
    size_t len_;
    size_t pos_;

    char peek() const { return pos_ < len_ ? data_[pos_] : '\0'; }
    char next() { return pos_ < len_ ? data_[pos_++] : '\0'; }
    bool eof() const { return pos_ >= len_; }

    void skip_ws() {
        while (pos_ < len_ && std::isspace(static_cast<unsigned char>(data_[pos_])))
            pos_++;
    }

    bool expect(char c) {
        skip_ws();
        if (peek() == c) {
            pos_++;
            return true;
        }
        return false;
    }

    // Four hex digits into `out`. False when fewer than four remain or one is
    // not hex - a truncated escape is malformed input, not a character.
    bool read_hex4(uint32_t& out) {
        if (pos_ + 4 > len_)
            return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            const char c = data_[pos_ + i];
            uint32_t d;
            if (c >= '0' && c <= '9')
                d = static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                d = static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                d = static_cast<uint32_t>(c - 'A' + 10);
            else
                return false;
            v = (v << 4) | d;
        }
        pos_ += 4;
        out = v;
        return true;
    }

    // A lone surrogate is not a legal codepoint; JSON allows one to appear and
    // the standard replacement character is what every other decoder emits.
    static void append_utf8(std::string& s, uint32_t cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF)
            cp = 0xFFFD;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            s += static_cast<char>(0xF0 | (cp >> 18));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parse_string() {
        skip_ws();
        if (peek() != '"')
            return {};
        pos_++;
        std::string s;
        while (!eof() && peek() != '"') {
            if (peek() == '\\') {
                pos_++;
                char esc = next();
                switch (esc) {
                    case '"':
                        s += '"';
                        break;
                    case '\\':
                        s += '\\';
                        break;
                    case '/':
                        s += '/';
                        break;
                    case 'n':
                        s += '\n';
                        break;
                    case 't':
                        s += '\t';
                        break;
                    case 'r':
                        s += '\r';
                        break;
                    case 'b':
                        s += '\b';
                        break;
                    case 'f':
                        s += '\f';
                        break;
                    case 'u': {
                        // \uXXXX -> UTF-8. json.dumps's ensure_ascii=True means schemas commonly
                        // arrive with every non-ASCII char escaped; parse_string() feeds enum
                        // values, property names, `required` and `pattern` (#1563).
                        uint32_t cp = 0;
                        if (!read_hex4(cp))
                            return s;  // truncated escape: stop, do not invent
                        // A high surrogate must be followed by \uDC00-\uDFFF;
                        // the pair encodes one codepoint above the BMP.
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (pos_ + 1 < len_ && data_[pos_] == '\\' && data_[pos_ + 1] == 'u') {
                                const size_t save = pos_;
                                pos_ += 2;
                                uint32_t lo = 0;
                                if (read_hex4(lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    pos_ = save;  // not a pair; emit the lone surrogate below
                                }
                            }
                        }
                        append_utf8(s, cp);
                        break;
                    }
                    default:
                        s += esc;
                        break;
                }
            } else {
                s += next();
            }
        }
        if (peek() == '"')
            pos_++;
        return s;
    }

    // Parse a (possibly negative) integer literal. Returns false if none.
    bool parse_int(long& out) {
        skip_ws();
        size_t start = pos_;
        if (peek() == '-')
            pos_++;
        bool any = false;
        while (!eof() && peek() >= '0' && peek() <= '9') {
            pos_++;
            any = true;
        }
        if (!any) {
            pos_ = start;
            return false;
        }
        out = std::strtol(data_ + start, nullptr, 10);
        return true;
    }

    // Raw text of a scalar non-string literal: true, false, null or an RFC 8259 number.
    // Returns false (position unchanged) for anything else, including objects and arrays.
    bool parse_scalar_literal(std::string& out) {
        skip_ws();
        const size_t start = pos_;
        for (const char* word : {"true", "false", "null"}) {
            const size_t n = strlen(word);
            if (pos_ + n <= len_ && strncmp(data_ + pos_, word, n) == 0) {
                pos_ += n;
                out.assign(word);
                return true;
            }
        }
        auto digits = [&] {
            const size_t d0 = pos_;
            while (!eof() && peek() >= '0' && peek() <= '9')
                pos_++;
            return pos_ - d0;
        };
        if (peek() == '-')
            pos_++;
        const size_t int0 = pos_;
        const size_t n_int = digits();
        bool ok = n_int > 0 && !(n_int > 1 && data_[int0] == '0');
        if (ok && peek() == '.') {
            pos_++;
            ok = digits() > 0;
        }
        if (ok && (peek() == 'e' || peek() == 'E')) {
            pos_++;
            if (peek() == '+' || peek() == '-')
                pos_++;
            ok = digits() > 0;
        }
        if (!ok) {
            pos_ = start;
            return false;
        }
        out.assign(data_ + start, pos_ - start);
        return true;
    }

    bool parse_bool() {
        skip_ws();
        if (pos_ + 4 <= len_ && strncmp(data_ + pos_, "true", 4) == 0) {
            pos_ += 4;
            return true;
        }
        if (pos_ + 5 <= len_ && strncmp(data_ + pos_, "false", 5) == 0) {
            pos_ += 5;
            return false;
        }
        return false;
    }

    void skip_value() {
        skip_ws();
        char c = peek();
        if (c == '"') {
            parse_string();
            return;
        }
        if (c == '{') {
            pos_++;
            int depth = 1;
            while (!eof() && depth > 0) {
                char ch = next();
                if (ch == '{')
                    depth++;
                else if (ch == '}')
                    depth--;
                else if (ch == '"') {  // skip string content
                    while (!eof() && peek() != '"') {
                        if (peek() == '\\')
                            pos_++;
                        pos_++;
                    }
                    if (peek() == '"')
                        pos_++;
                }
            }
            return;
        }
        if (c == '[') {
            pos_++;
            int depth = 1;
            while (!eof() && depth > 0) {
                char ch = next();
                if (ch == '[')
                    depth++;
                else if (ch == ']')
                    depth--;
                else if (ch == '"') {
                    while (!eof() && peek() != '"') {
                        if (peek() == '\\')
                            pos_++;
                        pos_++;
                    }
                    if (peek() == '"')
                        pos_++;
                }
            }
            return;
        }
        // number, bool, null — skip non-delimiter chars
        while (!eof() && peek() != ',' && peek() != '}' && peek() != ']' &&
               !std::isspace(static_cast<unsigned char>(peek())))
            pos_++;
    }

    std::vector<std::string> parse_string_array() {
        std::vector<std::string> result;
        skip_ws();
        if (!expect('['))
            return result;
        skip_ws();
        if (peek() == ']') {
            pos_++;
            return result;
        }
        while (!eof()) {
            result.push_back(parse_string());
            skip_ws();
            if (peek() == ',') {
                pos_++;
                continue;
            }
            if (peek() == ']') {
                pos_++;
                break;
            }
            break;
        }
        return result;
    }

    // Assertion keywords this build parses past but cannot enforce. Distinct from
    // annotations (`format`, `title`, `description`, etc.) which the spec
    // defines as silently ignored.
    static bool is_unenforceable_keyword(const std::string& k) {
        static const char* kUnenforceable[] = {
            "multipleOf",
            "not",
            "uniqueItems",
            "patternProperties",
            "propertyNames",
            "contains",
            "minContains",
            "maxContains",
            "dependentRequired",
            "dependentSchemas",
            "if",
            "then",
            "else",
        };
        for (const char* u : kUnenforceable) {
            if (k == u)
                return true;
        }
        return false;
    }

    static SchemaType type_from_string(const std::string& s) {
        if (s == "string")
            return SchemaType::STRING;
        if (s == "number")
            return SchemaType::NUMBER;
        if (s == "integer")
            return SchemaType::INTEGER;
        if (s == "boolean")
            return SchemaType::BOOLEAN;
        if (s == "null")
            return SchemaType::NULL_TYPE;
        if (s == "object")
            return SchemaType::OBJECT;
        if (s == "array")
            return SchemaType::ARRAY;
        return SchemaType::STRING;  // default fallback
    }

    // What one schema object said besides its node fields; finish_node() reads it.
    struct ObjectState {
        bool has_type = false;
        // Distinguishes "additionalProperties: false" from the keyword being
        // absent, which JSON Schema defines as true.
        bool saw_additional_properties = false;
        bool string_keyword = false;  // pattern / minLength / maxLength
        bool nullable = false;        // OpenAPI "nullable": true
        bool draft4_excl_lo = false;  // Draft 4 "exclusiveMinimum": true
        bool draft4_excl_hi = false;
        std::vector<SchemaType> types;  // "type": [...] with more than one member
        std::vector<std::unique_ptr<SchemaNode>> all_of;
    };

    // '[' schema, ... ']'
    void parse_schema_list(std::vector<std::unique_ptr<SchemaNode>>& out) {
        skip_ws();
        if (!expect('[')) {
            fail("expected an array of schemas");
            return;
        }
        skip_ws();
        while (!eof() && peek() != ']') {
            auto sub = parse_schema_object();
            if (sub)
                out.push_back(std::move(sub));
            skip_ws();
            if (peek() != ',')
                break;
            pos_++;
            skip_ws();
        }
        if (!expect(']'))
            fail("schema array is not closed");
    }

    // '{' "name": schema, ... '}' (properties, $defs)
    void parse_schema_map(std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>>& out) {
        skip_ws();
        if (!expect('{')) {
            fail("expected an object of schemas");
            return;
        }
        skip_ws();
        while (!eof() && peek() != '}') {
            std::string name = parse_string();
            skip_ws();
            if (!expect(':')) {
                fail("expected ':' after a schema name");
                return;
            }
            auto sub = parse_schema_object();
            if (sub)
                out.emplace_back(std::move(name), std::move(sub));
            skip_ws();
            if (peek() != ',')
                break;
            pos_++;
            skip_ws();
        }
        if (!expect('}'))
            fail("schema object is not closed");
    }

    // enum members: strings go to enum_values, scalars to enum_literals (emitted verbatim); an
    // object or array member has no FSM representation (#1564).
    void parse_enum_members(SchemaNode& node) {
        skip_ws();
        if (!expect('[')) {
            fail("enum is not an array");
            return;
        }
        skip_ws();
        while (!eof() && peek() != ']') {
            std::string lit;
            if (peek() == '"') {
                node.enum_values.push_back(parse_string());
            } else if (parse_scalar_literal(lit)) {
                node.enum_literals.push_back(std::move(lit));
            } else {
                fail("enum members must be strings, numbers, booleans or null");
                return;
            }
            skip_ws();
            if (peek() != ',')
                break;
            pos_++;
            skip_ws();
        }
        if (!expect(']'))
            fail("enum array is not closed");
    }

    // "type": "x" or ["x", "y"]; the node takes the first, ObjectState keeps a longer list.
    void parse_type(SchemaNode& node, ObjectState& st) {
        skip_ws();
        if (peek() != '[') {
            node.type = type_from_string(parse_string());
            return;
        }
        for (const std::string& t : parse_string_array())
            st.types.push_back(type_from_string(t));
        if (st.types.empty()) {
            fail("type array is empty");
            return;
        }
        node.type = st.types.front();
        if (st.types.size() == 1)
            st.types.clear();
    }

    // minLength / maxLength / minItems / maxItems / minProperties / maxProperties.
    bool parse_count_keyword(const std::string& key, SchemaNode& node, ObjectState& st) {
        int* slot = key == "minLength"       ? &node.min_length
                    : key == "maxLength"     ? &node.max_length
                    : key == "minItems"      ? &node.min_items
                    : key == "maxItems"      ? &node.max_items
                    : key == "minProperties" ? &node.min_properties
                    : key == "maxProperties" ? &node.max_properties
                                             : nullptr;
        if (!slot)
            return false;
        long v = 0;
        if (parse_int(v))
            *slot = static_cast<int>(v);
        st.string_keyword |= slot == &node.min_length || slot == &node.max_length;
        return true;
    }

    // minimum / maximum / exclusiveMinimum / exclusiveMaximum (Draft 4 boolean form included).
    bool parse_bound_keyword(const std::string& key, SchemaNode& node, ObjectState& st) {
        const bool lo = key == "minimum" || key == "exclusiveMinimum";
        const bool hi = key == "maximum" || key == "exclusiveMaximum";
        if (!lo && !hi)
            return false;
        const bool excl = key[0] == 'e';
        skip_ws();
        if (excl && (peek() == 't' || peek() == 'f')) {
            (lo ? st.draft4_excl_lo : st.draft4_excl_hi) = parse_bool();
            return true;
        }
        std::string lit;
        Decimal d;
        if (!parse_scalar_literal(lit) || !parse_decimal(lit, d)) {
            fail("'" + key + "' is not a number");
            skip_value();
            return true;
        }
        if (lo)
            node.bounds.tighten_lo(d, excl);
        else
            node.bounds.tighten_hi(d, excl);
        return true;
    }

    // "items": a schema, or true (any item) / false (none past prefixItems).
    void parse_items(SchemaNode& node) {
        skip_ws();
        if (peek() == 't' || peek() == 'f')
            node.items_false = !parse_bool();
        else
            node.items = parse_schema_object();
    }

    // "additionalProperties": a boolean, or the value schema of undeclared keys.
    void parse_additional(SchemaNode& node) {
        skip_ws();
        if (peek() == 't' || peek() == 'f') {
            node.additional_properties = parse_bool();
            return;
        }
        node.additional_properties = true;
        node.additional_schema = parse_schema_object();
    }

    void parse_ref(SchemaNode& node, ObjectState& st) {
        std::string ref = parse_string();
        std::string name;
        if (ref == "#") {
            name = "#";
        } else if (ref.rfind("#/$defs/", 0) == 0) {
            name = ref.substr(8);
        } else if (ref.rfind("#/definitions/", 0) == 0) {
            name = ref.substr(14);
        }
        if (name.empty() || name.find('/') != std::string::npos) {
            // External refs / deep JSON pointers — unsupported; fail the
            // parse so the caller declines constrained decoding.
            IMP_LOG_WARN("JSON schema: unsupported $ref '%s'", ref.c_str());
            ref_error_ = true;
            return;
        }
        node.type = SchemaType::REF;
        node.ref_name = name;
        st.has_type = true;
    }

    // #1567: const is enum with one member, and the FSM already has that path. Same member types
    // as enum.
    void parse_const(SchemaNode& node) {
        skip_ws();
        std::string lit;
        if (peek() == '"') {
            node.enum_values.push_back(parse_string());
        } else if (parse_scalar_literal(lit)) {
            node.enum_literals.push_back(std::move(lit));
        } else {
            fail("const must be a string, number, boolean or null");
            return;
        }
        node.type = SchemaType::ENUM;
    }

    // Combinator and structure keywords; false = not one of them.
    bool parse_structure_keyword(const std::string& key, SchemaNode& node, ObjectState& st) {
        if (key == "properties") {
            parse_schema_map(node.properties);
            if (!st.has_type)
                node.type = SchemaType::OBJECT;
        } else if (key == "items" || key == "prefixItems") {
            if (key == "items")
                parse_items(node);
            else
                parse_schema_list(node.prefix_items);
            if (!st.has_type)
                node.type = SchemaType::ARRAY;
        } else if (key == "additionalProperties") {
            st.saw_additional_properties = true;
            parse_additional(node);
        } else if (key == "anyOf" || key == "oneOf") {
            // oneOf reads as anyOf: exclusivity is not enforced (disjoint unions are the common case).
            parse_schema_list(node.any_of);
        } else if (key == "allOf") {
            parse_schema_list(st.all_of);
        } else if (key == "$defs" || key == "definitions") {
            parse_schema_map(collected_defs_);
        } else {
            return false;
        }
        return true;
    }

    // allOf members fold into the node: one $ref, `{}`, a bare required list, or object schemas
    // whose properties merge. Anything else is a 400 (an intersection the FSM cannot express).
    void merge_all_of(SchemaNode& node, ObjectState& st) {
        const bool lone_ref = st.all_of.size() == 1 && st.all_of[0]->type == SchemaType::REF;
        if (lone_ref && !st.has_type && node.properties.empty()) {
            node.type = SchemaType::REF;  // Pydantic {"allOf": [{"$ref": ...}], "description": ...}
            node.ref_name = st.all_of[0]->ref_name;
            st.has_type = true;
            return;
        }
        // Declared properties close an object unless additionalProperties opens it (imp default).
        bool additional = st.saw_additional_properties ? node.additional_properties : node.properties.empty();
        for (auto& sub : st.all_of) {
            for (auto& req : sub->required)
                node.required.push_back(req);
            if (sub->type == SchemaType::ANY)
                continue;
            if (sub->type != SchemaType::OBJECT || (st.has_type && node.type != SchemaType::OBJECT)) {
                fail("allOf: only object members, a required list or one $ref are enforceable");
                return;
            }
            for (auto& [name, prop] : sub->properties) {
                for (const auto& [mine, _] : node.properties)
                    if (mine == name) {
                        fail("allOf: property '" + name + "' declared twice");
                        return;
                    }
                node.properties.emplace_back(name, std::move(prop));
            }
            additional = additional && sub->additional_properties;
            if (sub->additional_schema && !node.additional_schema)
                node.additional_schema = std::move(sub->additional_schema);
            node.min_properties = std::max(node.min_properties, sub->min_properties);
            if (sub->max_properties >= 0)
                node.max_properties = node.max_properties < 0
                                          ? sub->max_properties
                                          : std::min(node.max_properties, sub->max_properties);
            node.type = SchemaType::OBJECT;
            st.has_type = true;
        }
        node.additional_properties = additional;
        st.saw_additional_properties = true;
    }

    // A type array, or OpenAPI nullable: the node becomes anyOf over one copy per type.
    void expand_types(SchemaNode& node, const ObjectState& st) {
        std::vector<std::unique_ptr<SchemaNode>> alts;
        for (const SchemaType t : st.types) {
            auto alt = node.clone();
            alt->type = t;
            if (t == SchemaType::OBJECT && alt->properties.empty() && !st.saw_additional_properties)
                alt->additional_properties = true;
            alts.push_back(std::move(alt));
        }
        if (st.nullable) {
            if (alts.empty())
                alts.push_back(node.clone());
            alts.push_back(std::make_unique<SchemaNode>());
            alts.back()->type = SchemaType::NULL_TYPE;
        }
        node.any_of = std::move(alts);
        node.type = SchemaType::ANY_OF;
    }

    // anyOf beside a type or properties: an alternative that only adds `required` inherits that
    // base (anyOf [A, B] next to base reads as anyOf [base + A, base + B]).
    static void finish_any_of(SchemaNode& node, const ObjectState& st) {
        const bool has_base = st.has_type || !node.properties.empty();
        for (auto& alt : node.any_of) {
            if (!has_base || alt->type != SchemaType::ANY)
                continue;
            auto merged = node.clone();
            merged->any_of.clear();
            for (const auto& req : alt->required)
                merged->required.push_back(req);
            if (merged->type == SchemaType::OBJECT && merged->properties.empty() &&
                !st.saw_additional_properties)
                merged->additional_properties = true;  // #1729, as finish_node applies it
            alt = std::move(merged);
        }
        node.type = SchemaType::ANY_OF;
    }

    void finish_node(SchemaNode& node, ObjectState& st) {
        if (st.draft4_excl_lo && node.bounds.has_lo)
            node.bounds.lo_excl = true;
        if (st.draft4_excl_hi && node.bounds.has_hi)
            node.bounds.hi_excl = true;
        if (!st.all_of.empty())
            merge_all_of(node, st);
        // Enum takes precedence over a co-declared "type" (JSON key order is not
        // significant; {"type":"string","enum":[...]} must not demote back to a
        // free string, which the constrainer would then accept any value for).
        if (node.has_enum() && node.type != SchemaType::REF) {
            node.type = SchemaType::ENUM;
            st.types.clear();
        } else if (!node.any_of.empty() && node.type != SchemaType::REF) {
            finish_any_of(node, st);
        } else if (!st.has_type && node.type == SchemaType::STRING) {
            // No type and no structure: the keywords present imply one, else any value.
            node.type = st.string_keyword                                          ? SchemaType::STRING
                        : node.bounds.active()                                     ? SchemaType::NUMBER
                        : st.saw_additional_properties || node.min_properties >= 0 ? SchemaType::OBJECT
                                                                                   : SchemaType::ANY;
        }
        // #1729: an object with no properties/additionalProperties defaults to
        // free-form (`true`), else the FSM's only legal doc is {}. Not applied
        // when properties ARE declared - that would loosen tool-schema strictness.
        if (node.type == SchemaType::OBJECT && node.properties.empty() && !st.saw_additional_properties)
            node.additional_properties = true;
        if (st.types.size() > 1 || st.nullable)
            expand_types(node, st);
    }

    std::unique_ptr<SchemaNode> parse_schema_object() {
        // #1609: one frame per nesting level, from an unauthenticated request
        // body. 10^5 nested "items" objects overflow the worker thread's stack.
        if (depth_ >= kMaxSchemaDepth) {
            fail("schema nests deeper than " + std::to_string(kMaxSchemaDepth) + " levels");
            return nullptr;
        }
        DepthGuard guard(depth_);

        skip_ws();
        if (!expect('{')) {
            fail("expected '{' at the start of a schema object");
            return nullptr;
        }

        auto node = std::make_unique<SchemaNode>();
        ObjectState st;
        skip_ws();
        while (!eof() && peek() != '}') {
            std::string key = parse_string();
            skip_ws();
            if (!expect(':'))
                break;

            if (key == "type") {
                parse_type(*node, st);
                st.has_type = true;
            } else if (key == "pattern") {
                node->pattern = parse_string();
                st.string_keyword = true;
            } else if (key == "required") {
                node->required = parse_string_array();
            } else if (key == "enum") {
                parse_enum_members(*node);
                node->type = SchemaType::ENUM;
            } else if (key == "const") {
                parse_const(*node);
            } else if (key == "$ref") {
                parse_ref(*node, st);
            } else if (key == "nullable") {
                st.nullable = parse_bool();
            } else if (!parse_structure_keyword(key, *node, st) && !parse_count_keyword(key, *node, st) &&
                       !parse_bound_keyword(key, *node, st)) {
                // #1567: an assertion keyword is not an annotation - dropping one answers a
                // bounded request with an unbounded grammar. docs/API.md: "A constraint imp
                // cannot compile is a 400, not an unconstrained answer." Unknown fields and pure
                // annotations ($schema, title, description, examples, default, format) are skipped.
                if (is_unenforceable_keyword(key))
                    fail("schema keyword '" + key + "' is not enforceable by this build");
                skip_value();
            }

            skip_ws();
            if (peek() != ',')
                break;
            pos_++;
            skip_ws();
        }
        if (!expect('}'))
            fail("schema object is not closed; a value before this point was not consumed");
        finish_node(*node, st);
        return node;
    }
};
}  // namespace

// Calls fn on every child schema of `node` (properties, items, prefixItems, anyOf,
// additionalProperties, defs) until fn returns false; returns false then.
template <typename Node, typename Fn>
static bool for_each_child(Node* node, Fn&& fn) {
    for (auto& [name, prop] : node->properties)
        if (!fn(prop.get()))
            return false;
    for (SchemaNode* single : {node->items.get(), node->additional_schema.get()})
        if (single && !fn(single))
            return false;
    for (auto* list : {&node->prefix_items, &node->any_of})
        for (auto& sub : *list)
            if (!fn(sub.get()))
                return false;
    for (auto& [name, def] : node->defs)
        if (!fn(def.get()))
            return false;
    return true;
}

// Recursively compile any `pattern` fields into NFAs. Unsupported patterns
// leave pattern_nfa null (enforcement is then skipped for that node).
static void compile_patterns(SchemaNode* node) {
    if (!node)
        return;
    if (!node->pattern.empty() && !node->pattern_nfa) {
        auto nfa = std::make_shared<RegexNfa>();
        if (nfa->compile(node->pattern)) {
            node->pattern_nfa = std::move(nfa);
        } else {
            IMP_LOG_WARN("JSON schema: unsupported regex pattern '%s' — pattern not enforced",
                         node->pattern.c_str());
        }
    }
    for_each_child(node, [](SchemaNode* child) {
        compile_patterns(child);
        return true;
    });
}

// Resolve a REF chain against the root defs table. Defensive guard against
// pure ref->ref cycles; validate_refs() rejects those at parse time.
const SchemaNode* resolve_schema_ref(const SchemaNode* root, const SchemaNode* node) {
    int guard = 0;
    while (node && node->type == SchemaType::REF) {
        if (++guard > 64 || !root)
            return nullptr;
        if (node->ref_name == "#") {
            node = root;
            continue;
        }
        const SchemaNode* found = nullptr;
        for (auto& [name, def] : root->defs) {
            if (name == node->ref_name) {
                found = def.get();
                break;
            }
        }
        node = found;
    }
    return node;
}

// Validate every $ref in the tree: the target must exist and a chain of pure
// refs must terminate in a non-REF node. Returns false (with a log) on the
// first unresolvable ref so the caller declines constrained decoding.
static bool validate_refs(const SchemaNode* root, const SchemaNode* node) {
    if (!node)
        return true;
    if (node->type == SchemaType::REF && resolve_schema_ref(root, node) == nullptr) {
        IMP_LOG_WARN("JSON schema: unresolvable $ref '%s'", node->ref_name.c_str());
        return false;
    }
    return for_each_child(node, [root](const SchemaNode* child) { return validate_refs(root, child); });
}

std::unique_ptr<SchemaNode> parse_json_schema(const std::string& json) {
    SchemaParser parser(json.c_str(), json.size());
    auto root = parser.parse();
    if (!root) {
        IMP_LOG_ERROR("Failed to parse JSON schema");
        return nullptr;
    }
    if (parser.ref_error()) {
        IMP_LOG_ERROR("Failed to parse JSON schema: unsupported $ref form");
        return nullptr;
    }
    if (parser.parse_error()) {
        IMP_LOG_ERROR("Failed to parse JSON schema: %s", parser.error_reason().c_str());
        return nullptr;
    }
    // Attach definitions collected anywhere in the document to the root —
    // resolve_schema_ref() searches only the root table.
    for (auto& [name, def] : parser.collected_defs_)
        root->defs.emplace_back(std::move(name), std::move(def));
    if (!validate_refs(root.get(), root.get())) {
        IMP_LOG_ERROR("Failed to parse JSON schema: unresolvable $ref");
        return nullptr;
    }
    compile_patterns(root.get());
    return root;
}

std::unique_ptr<SchemaNode> SchemaNode::clone() const {
    auto c = std::make_unique<SchemaNode>();
    c->type = type;
    c->additional_properties = additional_properties;
    c->required = required;
    c->enum_values = enum_values;
    c->enum_literals = enum_literals;
    c->pattern = pattern;
    c->min_length = min_length;
    c->max_length = max_length;
    c->pattern_nfa = pattern_nfa;  // shared; compiled NFA is immutable
    c->min_items = min_items;
    c->max_items = max_items;
    c->items_false = items_false;
    c->min_properties = min_properties;
    c->max_properties = max_properties;
    c->bounds = bounds;
    c->ref_name = ref_name;
    for (auto& [name, def] : defs)
        c->defs.emplace_back(name, def->clone());
    for (auto& [name, prop] : properties)
        c->properties.emplace_back(name, prop->clone());
    if (items)
        c->items = items->clone();
    if (additional_schema)
        c->additional_schema = additional_schema->clone();
    for (auto& sub : prefix_items)
        c->prefix_items.push_back(sub->clone());
    for (auto& sub : any_of)
        c->any_of.push_back(sub->clone());
    return c;
}

// Rewrite every REF in the subtree whose ref_name is a key in `rename` to the
// mapped name. Used to namespace a tool's hoisted $defs (#1002 stage 2).
static void rewrite_refs(SchemaNode* node, const std::map<std::string, std::string>& rename) {
    if (!node)
        return;
    if (node->type == SchemaType::REF) {
        auto it = rename.find(node->ref_name);
        if (it != rename.end())
            node->ref_name = it->second;
    }
    for_each_child(node, [&rename](SchemaNode* child) {
        rewrite_refs(child, rename);
        return true;
    });
}

// Shared per-tool loop: parses each tool's schema, hoists its $defs into
// root->defs under "<tool>/<def>", returns false if any tool is unenforceable.
// xml: names/keys are UNQUOTED in tags, so enum params or '<','>','\n' have no XML form.
static bool xml_tag_name_ok(const std::string& s) {
    return s.find_first_of("<>\n") == std::string::npos;
}

static bool collect_tool_defs(const std::vector<std::pair<std::string, std::string>>& tools,
                              SchemaNode* root, std::vector<std::string>& names, bool xml) {
    for (auto& [name, params_json] : tools) {
        if (name.empty())
            return false;
        if (xml && !xml_tag_name_ok(name))
            return false;
        auto params = parse_json_schema(params_json);
        const SchemaNode* res = params ? resolve_schema_ref(params.get(), params.get()) : nullptr;
        // #1729: a free-form object (free keys, undescribed values) is
        // representable only in the JSON dialect - the XML dialect renders
        // parameter keys as tags, and a schema with none has no tag to render.
        const bool free_form_object = res && res->type == SchemaType::OBJECT &&
                                      res->properties.empty() && res->additional_properties;
        const bool enforceable = res && ((res->type == SchemaType::OBJECT && !res->properties.empty()) ||
                                         (!xml && free_form_object) ||
                                         (!xml && res->type == SchemaType::ENUM && res->has_enum()));
        if (!enforceable)
            return false;
        if (xml) {
            for (auto& [key, _] : res->properties)
                if (!xml_tag_name_ok(key))
                    return false;
        }
        // #1002 stage 2: hoist per-tool $defs into TOOL_CALL root (schema_constrain.cu
        // resolves REFs only there). "<tool>/<def>" keys contain '/', which no other
        // name can, so they never collide; "#" self-refs rewrite to the tool name.
        if (!params->defs.empty()) {
            std::map<std::string, std::string> rename;
            rename["#"] = name;
            for (auto& [def_name, def_schema] : params->defs)
                rename[def_name] = name + "/" + def_name;
            rewrite_refs(params.get(), rename);  // recurses into params->defs too
            for (auto& [def_name, def_schema] : params->defs)
                root->defs.emplace_back(name + "/" + def_name, std::move(def_schema));
            params->defs.clear();
        }
        names.push_back(name);
        root->defs.emplace_back(name, std::move(params));
    }
    return true;
}

std::unique_ptr<SchemaNode> build_tool_call_schema(
    const std::vector<std::pair<std::string, std::string>>& tools) {
    if (tools.empty())
        return nullptr;

    auto root = std::make_unique<SchemaNode>();
    root->type = SchemaType::TOOL_CALL;
    root->required = {"name", "arguments"};
    root->additional_properties = false;

    std::vector<std::string> names;
    if (!collect_tool_defs(tools, root.get(), names, /*xml=*/false))
        return nullptr;

    auto name_enum = std::make_unique<SchemaNode>();
    name_enum->type = SchemaType::ENUM;
    name_enum->enum_values = std::move(names);

    root->properties.emplace_back("name", std::move(name_enum));
    // "arguments" placeholder — resolved dynamically against defs via the
    // frame's chosen_tool (schema_constrain.cu, SchemaType::TOOL_CALL).
    auto args_placeholder = std::make_unique<SchemaNode>();
    args_placeholder->type = SchemaType::OBJECT;
    root->properties.emplace_back("arguments", std::move(args_placeholder));
    return root;
}

std::unique_ptr<SchemaNode> build_xml_tool_call_schema(
    const std::vector<std::pair<std::string, std::string>>& tools) {
    if (tools.empty())
        return nullptr;

    auto root = std::make_unique<SchemaNode>();
    root->type = SchemaType::XML_TOOL_CALL;
    root->additional_properties = false;

    std::vector<std::string> names;
    if (!collect_tool_defs(tools, root.get(), names, /*xml=*/true))
        return nullptr;

    // The tool name lives in the <function=NAME> tag — an unquoted enum on the
    // root itself; parameter keys/required come from defs[chosen_tool].
    root->enum_values = std::move(names);
    return root;
}

}  // namespace imp

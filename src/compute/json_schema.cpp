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

// ---------------------------------------------------------------------------
// Minimal JSON parser for schema documents (no external dependencies).
// Only handles the subset needed for JSON Schema: objects, arrays, strings,
// numbers, booleans, null. No comments, no trailing commas.
// ---------------------------------------------------------------------------

class SchemaParser {
public:
    SchemaParser(const char* data, size_t len) : data_(data), len_(len), pos_(0) {}

    std::unique_ptr<SchemaNode> parse() {
        skip_ws();
        auto node = parse_schema_object();
        // #1564: the object loop breaks on anything that is not ',' and the
        // closing expect() was discarded, so a desync inside the document
        // returned a truncated tree that still looked like a parse. Trailing
        // input is the outermost symptom of that and the cheapest place to
        // catch a shape nothing else noticed.
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

    // Set when the document is structurally unparseable or uses a keyword this
    // build cannot enforce. Same contract as ref_error_: the whole parse fails,
    // and the caller turns that into a 400. Silently dropping the keyword would
    // answer a bounded request with an unbounded grammar.
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
                        // Skip unicode escape (4 hex digits)
                        for (int i = 0; i < 4 && !eof(); i++)
                            pos_++;
                        s += '?';
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

    // Assertion keywords this build parses past but cannot enforce. Split from
    // annotations on purpose: `format`, `title`, `description`, `examples`,
    // `default`, `deprecated`, `readOnly`, `writeOnly` change no legal value
    // and stay silently ignored, which is what the spec says they do.
    static bool is_unenforceable_keyword(const std::string& k) {
        static const char* kUnenforceable[] = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "allOf",
            "not",
            "uniqueItems",
            "patternProperties",
            "propertyNames",
            "prefixItems",
            "contains",
            "minContains",
            "maxContains",
            "minProperties",
            "maxProperties",
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
        bool has_type = false;

        skip_ws();
        if (peek() == '}') {
            pos_++;
            return node;
        }

        while (!eof()) {
            std::string key = parse_string();
            skip_ws();
            if (!expect(':'))
                break;

            if (key == "type") {
                std::string type_str = parse_string();
                node->type = type_from_string(type_str);
                has_type = true;
            } else if (key == "properties") {
                skip_ws();
                if (expect('{')) {
                    skip_ws();
                    while (!eof() && peek() != '}') {
                        std::string prop_name = parse_string();
                        skip_ws();
                        if (!expect(':')) {
                            fail("expected ':' after a property name");
                            break;
                        }
                        auto prop_schema = parse_schema_object();
                        if (prop_schema) {
                            node->properties.emplace_back(std::move(prop_name), std::move(prop_schema));
                        }
                        skip_ws();
                        if (peek() == ',') {
                            pos_++;
                            skip_ws();
                            continue;
                        }
                        break;
                    }
                    expect('}');
                }
                if (!has_type)
                    node->type = SchemaType::OBJECT;
            } else if (key == "pattern") {
                node->pattern = parse_string();
            } else if (key == "minLength") {
                long v = 0;
                if (parse_int(v))
                    node->min_length = static_cast<int>(v);
            } else if (key == "maxLength") {
                long v = 0;
                if (parse_int(v))
                    node->max_length = static_cast<int>(v);
            } else if (key == "required") {
                node->required = parse_string_array();
            } else if (key == "additionalProperties") {
                // #1564: parse_bool() returns false WITHOUT consuming when the
                // value is not true/false, so the schema-object form left pos_
                // on '{'. The key loop then saw '{' instead of ',', broke, and
                // every key after this one was dropped - including
                // `properties`, which downgrades the request to json_object at
                // constraint_manager.cpp:149. Consume the value either way.
                skip_ws();
                if (peek() == 't' || peek() == 'f') {
                    node->additional_properties = parse_bool();
                } else {
                    // The schema form ({"type": "..."}) is legal and common
                    // (Pydantic emits it for Dict[str, T]). imp does not
                    // enforce a schema on extra keys, so this reads as the
                    // permissive `true` - a weaker constraint, not a wrong one.
                    skip_value();
                    node->additional_properties = true;
                }
            } else if (key == "items") {
                node->items = parse_schema_object();
                if (!has_type)
                    node->type = SchemaType::ARRAY;
            } else if (key == "minItems") {
                long v = 0;
                if (parse_int(v))
                    node->min_items = static_cast<int>(v);
            } else if (key == "maxItems") {
                long v = 0;
                if (parse_int(v))
                    node->max_items = static_cast<int>(v);
            } else if (key == "enum") {
                skip_ws();
                if (!expect('[')) {
                    fail("enum is not an array");
                } else {
                    skip_ws();
                    while (!eof() && peek() != ']') {
                        // #1564: parse_string() also returns "" without
                        // consuming. A non-string member therefore produced
                        // enum_values == {""} and left pos_ on the member, so
                        // the rest of the schema was dropped AND the only legal
                        // output became the empty string. The FSM emits an enum
                        // as quoted string content (schema_constrain.cu:790),
                        // so a numeric or boolean member has no representation:
                        // refuse rather than constrain to something else.
                        if (peek() != '"') {
                            fail(
                                "enum members must be strings; a number, boolean or null "
                                "enum cannot be enforced by this build");
                            break;
                        }
                        node->enum_values.push_back(parse_string());
                        skip_ws();
                        if (peek() == ',') {
                            pos_++;
                            skip_ws();
                            continue;
                        }
                        break;
                    }
                    if (!parse_error_ && !expect(']'))
                        fail("enum array is not closed");
                }
                node->type = SchemaType::ENUM;
            } else if (key == "$ref") {
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
                } else {
                    node->type = SchemaType::REF;
                    node->ref_name = name;
                    has_type = true;
                }
            } else if (key == "$defs" || key == "definitions") {
                skip_ws();
                if (expect('{')) {
                    skip_ws();
                    while (!eof() && peek() != '}') {
                        std::string def_name = parse_string();
                        skip_ws();
                        if (!expect(':'))
                            break;
                        auto def_schema = parse_schema_object();
                        if (def_schema)
                            collected_defs_.emplace_back(std::move(def_name), std::move(def_schema));
                        skip_ws();
                        if (peek() == ',') {
                            pos_++;
                            skip_ws();
                            continue;
                        }
                        break;
                    }
                    expect('}');
                }
            } else if (key == "anyOf" || key == "oneOf") {
                skip_ws();
                if (expect('[')) {
                    skip_ws();
                    while (!eof() && peek() != ']') {
                        auto sub = parse_schema_object();
                        if (sub)
                            node->any_of.push_back(std::move(sub));
                        skip_ws();
                        if (peek() == ',') {
                            pos_++;
                            skip_ws();
                            continue;
                        }
                        break;
                    }
                    expect(']');
                }
                node->type = SchemaType::ANY_OF;
            } else if (key == "const") {
                // #1567: const is enum with one member, and the FSM already
                // has that path. Same string-only limit as enum above.
                skip_ws();
                if (peek() != '"') {
                    fail(
                        "const must be a string; a number, boolean or null const "
                        "cannot be enforced by this build");
                } else {
                    node->enum_values.push_back(parse_string());
                    node->type = SchemaType::ENUM;
                }
            } else if (is_unenforceable_keyword(key)) {
                // #1567: these are assertions, not annotations. Dropping one
                // answers a request that bounded its output with a grammar that
                // does not - the exact failure #1540/#751 describes, reached by
                // a caller who did bound the field. docs/API.md: "A constraint
                // imp cannot compile is a 400, not an unconstrained answer."
                fail("schema keyword '" + key + "' is not enforceable by this build");
                skip_value();
            } else {
                // Skip unknown fields and pure annotations ($schema, title,
                // description, examples, default, format - which is an
                // annotation in Draft 2020-12 unless the format-assertion
                // vocabulary is in use, and imp does not claim it).
                skip_value();
            }

            skip_ws();
            if (peek() == ',') {
                pos_++;
                skip_ws();
                continue;
            }
            break;
        }
        if (!expect('}'))
            fail("schema object is not closed; a value before this point was not consumed");

        // Enum takes precedence over a co-declared "type". Key order in the
        // object is not significant in JSON, and clients commonly emit
        // {"type":"string","enum":[...]} — a later "type":"string" must NOT
        // demote the node back to a free string (the constrainer would then
        // accept any value). Resolve this order-independently.
        if (!node->enum_values.empty() && node->type != SchemaType::REF)
            node->type = SchemaType::ENUM;

        return node;
    }
};

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
    for (auto& [name, prop] : node->properties)
        compile_patterns(prop.get());
    if (node->items)
        compile_patterns(node->items.get());
    for (auto& sub : node->any_of)
        compile_patterns(sub.get());
    for (auto& [name, def] : node->defs)
        compile_patterns(def.get());
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
    for (auto& [name, prop] : node->properties)
        if (!validate_refs(root, prop.get()))
            return false;
    if (node->items && !validate_refs(root, node->items.get()))
        return false;
    for (auto& sub : node->any_of)
        if (!validate_refs(root, sub.get()))
            return false;
    for (auto& [name, def] : node->defs)
        if (!validate_refs(root, def.get()))
            return false;
    return true;
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
    c->pattern = pattern;
    c->min_length = min_length;
    c->max_length = max_length;
    c->pattern_nfa = pattern_nfa;  // shared; compiled NFA is immutable
    c->min_items = min_items;
    c->max_items = max_items;
    c->ref_name = ref_name;
    for (auto& [name, def] : defs)
        c->defs.emplace_back(name, def->clone());
    for (auto& [name, prop] : properties)
        c->properties.emplace_back(name, prop->clone());
    if (items)
        c->items = items->clone();
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
    for (auto& [n, prop] : node->properties)
        rewrite_refs(prop.get(), rename);
    if (node->items)
        rewrite_refs(node->items.get(), rename);
    for (auto& sub : node->any_of)
        rewrite_refs(sub.get(), rename);
    for (auto& [n, def] : node->defs)
        rewrite_refs(def.get(), rename);
}

// Shared per-tool loop for both tool-call roots: parses every parameter
// schema, applies the enforceability gates, hoists per-tool $defs into the
// root under the "<tool>/<def>" namespace, and records (tool name, parameter
// schema) in root->defs. `names` receives the tool names in order. Returns
// false when any tool is unenforceable (caller declines the whole set).
// xml: the Qwen-Coder XML dialect writes names/keys UNQUOTED inside
// <function=NAME>/<parameter=KEY> tags and can only express object
// properties — an ENUM params schema (legal for the JSON dialect, where
// "arguments" IS the enum string) has no XML representation, and a name/key
// containing '<', '>' or a newline can never complete its tag.
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
        // Enforceable structure only: a free-form object dead-ends the key
        // phase (see the free_form route in ConstraintManager::prepare).
        const bool enforceable =
            res && ((res->type == SchemaType::OBJECT && !res->properties.empty()) ||
                    (!xml && res->type == SchemaType::ENUM && !res->enum_values.empty()));
        if (!enforceable)
            return false;
        if (xml) {
            for (auto& [key, _] : res->properties)
                if (!xml_tag_name_ok(key))
                    return false;
        }
        // Hoist any per-tool $defs into the TOOL_CALL root (#1002 stage 2).
        // REF resolution in schema_constrain.cu always searches the TOOL_CALL
        // root's defs, so a tool's nested models (pydantic/zod emit $defs+$ref
        // for every nested model) must live there. The namespace key
        // "<tool>/<def>" carries a '/', which parse_json_schema forbids in any
        // $ref-derived name (and no function name contains one), so a hoisted
        // key can never collide with a tool name or another tool's hoisted def.
        // "#" self-refs (recursive root schema) rewrite to the tool name, whose
        // root->defs entry IS this param schema — so `arguments` chases it back.
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

// ===========================================================================
// RegexNfa — Thompson-construction NFA over bytes for the supported subset.
// All host-side; never compiled into device code.
// ===========================================================================

int RegexNfa::new_state() {
    // #1608: the {n,m} builder allocates per clone and the nested form
    // multiplies, so the state count is the resource an attacker actually
    // spends. Once the budget is gone the parse is an error and every caller
    // unwinds through the `error_` checks; returning the last valid index keeps
    // the add_edge()/add_epsilon() calls already in flight in bounds.
    if (states_.size() >= kMaxStates) {
        error_ = true;
        return states_.empty() ? 0 : static_cast<int>(states_.size()) - 1;
    }
    states_.emplace_back();
    return static_cast<int>(states_.size()) - 1;
}

void RegexNfa::add_epsilon(int from, int to) {
    NfaEdge e;
    e.to = to;
    e.is_epsilon = true;
    states_[from].edges.push_back(std::move(e));
}

void RegexNfa::add_edge(int from, int to, const std::vector<uint8_t>& cls) {
    NfaEdge e;
    e.to = to;
    e.is_epsilon = false;
    e.char_class = cls;
    states_[from].edges.push_back(std::move(e));
}

bool RegexNfa::make_shorthand(char esc, std::vector<uint8_t>& cls) {
    cls.assign(256, 0);
    auto set_digit = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool d = (c >= '0' && c <= '9');
            cls[c] = (d != neg) ? 1 : 0;
        }
    };
    auto set_word = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool w = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
            cls[c] = (w != neg) ? 1 : 0;
        }
    };
    auto set_space = [&](bool neg) {
        for (int c = 0; c < 256; c++) {
            bool s = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
            cls[c] = (s != neg) ? 1 : 0;
        }
    };
    switch (esc) {
        case 'd':
            set_digit(false);
            return true;
        case 'D':
            set_digit(true);
            return true;
        case 'w':
            set_word(false);
            return true;
        case 'W':
            set_word(true);
            return true;
        case 's':
            set_space(false);
            return true;
        case 'S':
            set_space(true);
            return true;
        default:
            return false;
    }
}

// atom := '(' alt ')' | '[' class ']' | '.' | escape | literal
bool RegexNfa::parse_atom(Frag& out) {
    if (pos_ >= src_->size()) {
        error_ = true;
        return false;
    }
    char c = (*src_)[pos_];

    if (c == '(') {
        pos_++;  // consume '('
        // `(?:…)` is a non-capturing group. Nothing here captures and
        // backreferences are refused upstream, so the marker carries no
        // matching semantics — skip it and parse the body as an ordinary group.
        // Without this, `?` was read as a quantifier with no atom and `:` as a
        // literal, so `(?:a|b)c` compiled to `(:a|b)c`: it matched "bc", not
        // "ac", while reporting a successful compile. A wrong pattern enforced
        // silently is the one failure mode this parser must not have.
        if (pos_ + 1 < src_->size() && (*src_)[pos_] == '?' && (*src_)[pos_ + 1] == ':')
            pos_ += 2;
        if (!parse_alt(out))
            return false;
        if (pos_ >= src_->size() || (*src_)[pos_] != ')') {
            error_ = true;
            return false;
        }
        pos_++;  // consume ')'
        return true;
    }

    if (c == '[') {
        return parse_class(out);
    }

    if (c == ')' || c == '|') {
        // empty atom — caller handles
        error_ = true;
        return false;
    }

    std::vector<uint8_t> cls(256, 0);

    if (c == '.') {
        pos_++;
        for (int i = 0; i < 256; i++)
            cls[i] = (i == '\n') ? 0 : 1;  // any except newline
    } else if (c == '\\') {
        pos_++;
        if (pos_ >= src_->size()) {
            error_ = true;
            return false;
        }
        char esc = (*src_)[pos_++];
        std::vector<uint8_t> sc;
        if (make_shorthand(esc, sc)) {
            cls = sc;
        } else {
            // escaped literal (\. \\ \+ \{ etc.) — also map common control escapes
            unsigned char lit;
            switch (esc) {
                case 'n':
                    lit = '\n';
                    break;
                case 't':
                    lit = '\t';
                    break;
                case 'r':
                    lit = '\r';
                    break;
                case 'f':
                    lit = '\f';
                    break;
                case 'v':
                    lit = '\v';
                    break;
                default:
                    lit = static_cast<unsigned char>(esc);
                    break;
            }
            cls[lit] = 1;
        }
    } else if (c == '^' || c == '$') {
        // Anchors: JSON-Schema pattern matching is treated as whole-string here
        // (token masking cannot enforce sub-string matches), so a leading ^ /
        // trailing $ are no-ops. Emit an epsilon fragment.
        pos_++;
        int s = new_state();
        int a = new_state();
        add_epsilon(s, a);
        out.start = s;
        out.accept = a;
        return true;
    } else {
        // literal byte
        pos_++;
        cls[static_cast<unsigned char>(c)] = 1;
    }

    int s = new_state();
    int a = new_state();
    add_edge(s, a, cls);
    out.start = s;
    out.accept = a;
    return true;
}

// class := '[' '^'? ( range | shorthand | char )+ ']'
bool RegexNfa::parse_class(Frag& out) {
    pos_++;  // consume '['
    bool negate = false;
    if (pos_ < src_->size() && (*src_)[pos_] == '^') {
        negate = true;
        pos_++;
    }
    std::vector<uint8_t> cls(256, 0);
    bool any = false;

    while (pos_ < src_->size() && (*src_)[pos_] != ']') {
        char c = (*src_)[pos_];
        if (c == '\\') {
            pos_++;
            if (pos_ >= src_->size()) {
                error_ = true;
                return false;
            }
            char esc = (*src_)[pos_++];
            std::vector<uint8_t> sc;
            if (make_shorthand(esc, sc)) {
                for (int i = 0; i < 256; i++)
                    if (sc[i])
                        cls[i] = 1;
                any = true;
                continue;
            }
            unsigned char lit;
            switch (esc) {
                case 'n':
                    lit = '\n';
                    break;
                case 't':
                    lit = '\t';
                    break;
                case 'r':
                    lit = '\r';
                    break;
                default:
                    lit = static_cast<unsigned char>(esc);
                    break;
            }
            cls[lit] = 1;
            any = true;
            continue;
        }
        // range a-z ?
        if (pos_ + 2 < src_->size() && (*src_)[pos_ + 1] == '-' && (*src_)[pos_ + 2] != ']') {
            unsigned char lo = static_cast<unsigned char>(c);
            unsigned char hi = static_cast<unsigned char>((*src_)[pos_ + 2]);
            pos_ += 3;
            if (lo > hi) {
                error_ = true;
                return false;
            }
            for (int i = lo; i <= hi; i++)
                cls[i] = 1;
            any = true;
        } else {
            cls[static_cast<unsigned char>(c)] = 1;
            pos_++;
            any = true;
        }
    }

    if (pos_ >= src_->size() || (*src_)[pos_] != ']' || !any) {
        error_ = true;
        return false;
    }
    pos_++;  // consume ']'

    if (negate) {
        for (int i = 0; i < 256; i++)
            cls[i] = cls[i] ? 0 : 1;
    }

    int s = new_state();
    int a = new_state();
    add_edge(s, a, cls);
    out.start = s;
    out.accept = a;
    return true;
}

// repeat := atom ('*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}')?
bool RegexNfa::parse_repeat(Frag& out) {
    // Snapshot the atom's source start here so {n,m} can re-parse the exact
    // source span of *this* atom (parse_atom may advance pos_ arbitrarily for
    // groups / classes).
    size_t atom_begin = pos_;
    Frag atom{};
    if (!parse_atom(atom))
        return false;

    if (pos_ >= src_->size())
        return (out = atom, true);

    char q = (*src_)[pos_];

    if (q == '*' || q == '+' || q == '?') {
        pos_++;
        int s = new_state();
        int a = new_state();
        if (q == '*') {
            add_epsilon(s, atom.start);
            add_epsilon(s, a);
            add_epsilon(atom.accept, atom.start);
            add_epsilon(atom.accept, a);
        } else if (q == '+') {
            add_epsilon(s, atom.start);
            add_epsilon(atom.accept, atom.start);
            add_epsilon(atom.accept, a);
        } else {  // '?'
            add_epsilon(s, atom.start);
            add_epsilon(s, a);
            add_epsilon(atom.accept, a);
        }
        out.start = s;
        out.accept = a;
        return true;
    }

    if (q == '{') {
        // {n} {n,} {n,m}
        size_t save = pos_;
        pos_++;  // consume '{'
        long n = 0, m = -1;
        auto scan_int = [&](long& out_val) -> bool {
            long v = 0;
            bool any = false;
            while (pos_ < src_->size() && (*src_)[pos_] >= '0' && (*src_)[pos_] <= '9') {
                // #1608: `v = v*10 + d` over an unbounded digit run is signed
                // overflow, i.e. UB, before it is ever compared to a limit.
                // Saturate at the repeat cap; the caller rejects from there.
                if (v <= kMaxRepeat)
                    v = v * 10 + ((*src_)[pos_] - '0');
                pos_++;
                any = true;
            }
            if (any)
                out_val = v;
            return any;
        };
        bool has_n = scan_int(n);
        bool comma = false;
        if (pos_ < src_->size() && (*src_)[pos_] == ',') {
            comma = true;
            pos_++;
            long mm = 0;
            if (scan_int(mm))
                m = mm;  // {n,m}; else {n,} -> m stays -1 (unbounded)
        }
        if (!has_n || pos_ >= src_->size() || (*src_)[pos_] != '}') {
            // Not a valid quantifier — treat '{' as literal: rewind.
            pos_ = save;
            return (out = atom, true);
        }
        pos_++;  // consume '}'
        if (!comma)
            m = n;  // {n}

        // #1608: n is a clone count. `a{2000000000}` ran a two-billion
        // iteration loop, each iteration re-parsing the atom and allocating
        // states, on an HTTP worker thread at admission time. The GBNF parser
        // has had this same bound since it was written.
        if (n > kMaxRepeat || m > kMaxRepeat) {
            error_ = true;
            return false;
        }

        // Build: n mandatory copies, then either unbounded (*) or (m-n) optional.
        int s = new_state();
        int cur = s;
        // We need fresh copies of the atom sub-pattern. Re-parse its exact
        // source span [atom_begin, save) for each additional copy. `save` is
        // the position right after the atom (the '{').
        std::string atom_src = src_->substr(atom_begin, save - atom_begin);

        auto clone_atom = [&](Frag& f) -> bool {
            const std::string* prev_src = src_;
            size_t prev_pos = pos_;
            bool prev_err = error_;
            // Parse the captured atom text in isolation.
            std::string local = atom_src;
            src_ = &local;
            pos_ = 0;
            error_ = false;
            bool ok = parse_repeat(f);  // atom may itself be a repeat-free atom
            // The clone's own error has to survive the restore. It did not:
            // error_ was overwritten with prev_err and THEN read, so `!error_`
            // reported the state before the clone. That also swallowed the
            // state-budget signal new_state() raises (#1608), which is what
            // stops the nested `(((a{100}){100}){100})` form.
            bool inner_err = error_;
            src_ = prev_src;
            pos_ = prev_pos;
            error_ = prev_err || inner_err;
            return ok && !inner_err;
        };

        // first mandatory copy is the already-built `atom`
        long built = 0;
        if (n >= 1) {
            add_epsilon(cur, atom.start);
            cur = atom.accept;
            built = 1;
        } else {
            // n == 0: atom is optional/unbounded from the start
        }

        for (long i = built; i < n; i++) {
            Frag f{};
            if (!clone_atom(f))
                return false;
            add_epsilon(cur, f.start);
            cur = f.accept;
        }

        int a = new_state();
        if (m < 0) {
            // {n,} -> after n mandatory, a Kleene star of the atom
            Frag f{};
            if (!clone_atom(f))
                return false;
            int ls = new_state();
            add_epsilon(cur, ls);
            add_epsilon(ls, f.start);
            add_epsilon(ls, a);
            add_epsilon(f.accept, f.start);
            add_epsilon(f.accept, a);
        } else {
            // {n,m} -> (m-n) optional copies
            for (long i = n; i < m; i++) {
                Frag f{};
                if (!clone_atom(f))
                    return false;
                add_epsilon(cur, f.start);
                add_epsilon(cur, a);  // optional: skip the rest
                cur = f.accept;
            }
            add_epsilon(cur, a);
        }
        out.start = s;
        out.accept = a;
        return true;
    }

    out = atom;
    return true;
}

// concat := repeat*
bool RegexNfa::parse_concat(Frag& out) {
    int s = new_state();
    int cur = s;
    bool any = false;
    while (pos_ < src_->size() && (*src_)[pos_] != '|' && (*src_)[pos_] != ')') {
        Frag f{};
        if (!parse_repeat(f))
            return false;
        add_epsilon(cur, f.start);
        cur = f.accept;
        any = true;
    }
    if (!any) {
        // empty concatenation matches empty string
        int a = new_state();
        add_epsilon(s, a);
        out.start = s;
        out.accept = a;
        return true;
    }
    out.start = s;
    out.accept = cur;
    return true;
}

// alt := concat ('|' concat)*
bool RegexNfa::parse_alt(Frag& out) {
    // #1609: parse_atom() recurses back into parse_alt() for a group, so a
    // pattern of '(' costs one frame per byte - the cheapest stack overflow in
    // the request surface. parse_alt is the single point that closes the
    // mutual recursion, so one guard here covers all four functions.
    if (depth_ >= kMaxDepth) {
        error_ = true;
        return false;
    }
    depth_++;
    struct Pop {
        int& d;
        ~Pop() { d--; }
    } pop{depth_};

    Frag first{};
    if (!parse_concat(first))
        return false;
    if (pos_ >= src_->size() || (*src_)[pos_] != '|') {
        out = first;
        return true;
    }
    int s = new_state();
    int a = new_state();
    add_epsilon(s, first.start);
    add_epsilon(first.accept, a);
    while (pos_ < src_->size() && (*src_)[pos_] == '|') {
        pos_++;  // consume '|'
        Frag next{};
        if (!parse_concat(next))
            return false;
        add_epsilon(s, next.start);
        add_epsilon(next.accept, a);
    }
    out.start = s;
    out.accept = a;
    return true;
}

bool RegexNfa::compile(const std::string& pattern) {
    states_.clear();
    compiled_ = false;
    error_ = false;
    depth_ = 0;
    src_ = &pattern;
    pos_ = 0;

    Frag root{};
    if (!parse_alt(root)) {
        src_ = nullptr;
        return false;
    }
    // Must have consumed the whole pattern.
    if (pos_ != pattern.size() || error_) {
        src_ = nullptr;
        return false;
    }
    src_ = nullptr;

    start_ = root.start;
    accept_ = root.accept;
    states_[accept_].accepting = true;
    compiled_ = true;
    return true;
}

void RegexNfa::epsilon_closure(std::vector<int>& set) const {
    std::vector<int> stack(set.begin(), set.end());
    std::vector<uint8_t> in(states_.size(), 0);
    for (int s : set)
        in[s] = 1;
    while (!stack.empty()) {
        int s = stack.back();
        stack.pop_back();
        for (const auto& e : states_[s].edges) {
            if (e.is_epsilon && !in[e.to]) {
                in[e.to] = 1;
                set.push_back(e.to);
                stack.push_back(e.to);
            }
        }
    }
    std::sort(set.begin(), set.end());
}

std::vector<int> RegexNfa::start_set() const {
    std::vector<int> set;
    if (!compiled_)
        return set;
    set.push_back(start_);
    epsilon_closure(set);
    return set;
}

std::vector<int> RegexNfa::step(const std::vector<int>& states, unsigned char c) const {
    std::vector<int> next;
    if (!compiled_)
        return next;
    std::vector<uint8_t> in(states_.size(), 0);
    for (int s : states) {
        for (const auto& e : states_[s].edges) {
            if (!e.is_epsilon && e.char_class[c] && !in[e.to]) {
                in[e.to] = 1;
                next.push_back(e.to);
            }
        }
    }
    epsilon_closure(next);
    return next;
}

bool RegexNfa::accepts(const std::vector<int>& states) const {
    for (int s : states)
        if (states_[s].accepting)
            return true;
    return false;
}

}  // namespace imp

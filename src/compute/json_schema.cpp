#include "compute/json_schema.h"
#include "core/logging.h"
#include <cstring>
#include <cctype>

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
        return node;
    }

private:
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
        if (peek() == c) { pos_++; return true; }
        return false;
    }

    std::string parse_string() {
        skip_ws();
        if (peek() != '"') return {};
        pos_++;
        std::string s;
        while (!eof() && peek() != '"') {
            if (peek() == '\\') {
                pos_++;
                char esc = next();
                switch (esc) {
                    case '"': s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/': s += '/'; break;
                    case 'n': s += '\n'; break;
                    case 't': s += '\t'; break;
                    case 'r': s += '\r'; break;
                    case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'u': {
                        // Skip unicode escape (4 hex digits)
                        for (int i = 0; i < 4 && !eof(); i++) pos_++;
                        s += '?';
                        break;
                    }
                    default: s += esc; break;
                }
            } else {
                s += next();
            }
        }
        if (peek() == '"') pos_++;
        return s;
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
        if (c == '"') { parse_string(); return; }
        if (c == '{') {
            pos_++;
            int depth = 1;
            while (!eof() && depth > 0) {
                char ch = next();
                if (ch == '{') depth++;
                else if (ch == '}') depth--;
                else if (ch == '"') { // skip string content
                    while (!eof() && peek() != '"') {
                        if (peek() == '\\') pos_++;
                        pos_++;
                    }
                    if (peek() == '"') pos_++;
                }
            }
            return;
        }
        if (c == '[') {
            pos_++;
            int depth = 1;
            while (!eof() && depth > 0) {
                char ch = next();
                if (ch == '[') depth++;
                else if (ch == ']') depth--;
                else if (ch == '"') {
                    while (!eof() && peek() != '"') {
                        if (peek() == '\\') pos_++;
                        pos_++;
                    }
                    if (peek() == '"') pos_++;
                }
            }
            return;
        }
        // number, bool, null — skip non-delimiter chars
        while (!eof() && peek() != ',' && peek() != '}' && peek() != ']'
               && !std::isspace(static_cast<unsigned char>(peek())))
            pos_++;
    }

    std::vector<std::string> parse_string_array() {
        std::vector<std::string> result;
        skip_ws();
        if (!expect('[')) return result;
        skip_ws();
        if (peek() == ']') { pos_++; return result; }
        while (!eof()) {
            result.push_back(parse_string());
            skip_ws();
            if (peek() == ',') { pos_++; continue; }
            if (peek() == ']') { pos_++; break; }
            break;
        }
        return result;
    }

    SchemaType type_from_string(const std::string& s) {
        if (s == "string")  return SchemaType::STRING;
        if (s == "number")  return SchemaType::NUMBER;
        if (s == "integer") return SchemaType::INTEGER;
        if (s == "boolean") return SchemaType::BOOLEAN;
        if (s == "null")    return SchemaType::NULL_TYPE;
        if (s == "object")  return SchemaType::OBJECT;
        if (s == "array")   return SchemaType::ARRAY;
        return SchemaType::STRING;  // default fallback
    }

    std::unique_ptr<SchemaNode> parse_schema_object() {
        skip_ws();
        if (!expect('{')) return nullptr;

        auto node = std::make_unique<SchemaNode>();
        bool has_type = false;
        bool has_enum = false;
        bool has_any_of = false;

        skip_ws();
        if (peek() == '}') { pos_++; return node; }

        while (!eof()) {
            std::string key = parse_string();
            skip_ws();
            if (!expect(':')) break;

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
                        if (!expect(':')) break;
                        auto prop_schema = parse_schema_object();
                        if (prop_schema) {
                            node->properties.emplace_back(std::move(prop_name),
                                                           std::move(prop_schema));
                        }
                        skip_ws();
                        if (peek() == ',') { pos_++; skip_ws(); continue; }
                        break;
                    }
                    expect('}');
                }
                if (!has_type) node->type = SchemaType::OBJECT;
            } else if (key == "required") {
                node->required = parse_string_array();
            } else if (key == "additionalProperties") {
                node->additional_properties = parse_bool();
            } else if (key == "items") {
                node->items = parse_schema_object();
                if (!has_type) node->type = SchemaType::ARRAY;
            } else if (key == "enum") {
                skip_ws();
                if (expect('[')) {
                    skip_ws();
                    while (!eof() && peek() != ']') {
                        // Store enum values as raw strings
                        node->enum_values.push_back(parse_string());
                        skip_ws();
                        if (peek() == ',') { pos_++; skip_ws(); continue; }
                        break;
                    }
                    expect(']');
                }
                has_enum = true;
                node->type = SchemaType::ENUM;
            } else if (key == "anyOf" || key == "oneOf") {
                skip_ws();
                if (expect('[')) {
                    skip_ws();
                    while (!eof() && peek() != ']') {
                        auto sub = parse_schema_object();
                        if (sub) node->any_of.push_back(std::move(sub));
                        skip_ws();
                        if (peek() == ',') { pos_++; skip_ws(); continue; }
                        break;
                    }
                    expect(']');
                }
                has_any_of = true;
                node->type = SchemaType::ANY_OF;
            } else {
                // Skip unknown fields ($schema, title, description, etc.)
                skip_value();
            }

            skip_ws();
            if (peek() == ',') { pos_++; skip_ws(); continue; }
            break;
        }
        expect('}');

        return node;
    }
};

std::unique_ptr<SchemaNode> parse_json_schema(const std::string& json) {
    SchemaParser parser(json.c_str(), json.size());
    auto root = parser.parse();
    if (!root) {
        IMP_LOG_ERROR("Failed to parse JSON schema");
        return nullptr;
    }
    return root;
}

std::unique_ptr<SchemaNode> SchemaNode::clone() const {
    auto c = std::make_unique<SchemaNode>();
    c->type = type;
    c->additional_properties = additional_properties;
    c->required = required;
    c->enum_values = enum_values;
    for (auto& [name, prop] : properties)
        c->properties.emplace_back(name, prop->clone());
    if (items) c->items = items->clone();
    for (auto& sub : any_of)
        c->any_of.push_back(sub->clone());
    return c;
}

} // namespace imp

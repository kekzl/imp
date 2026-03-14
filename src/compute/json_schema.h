#pragma once

#include <string>
#include <vector>
#include <memory>

namespace imp {

enum class SchemaType {
    STRING,
    NUMBER,
    INTEGER,
    BOOLEAN,
    NULL_TYPE,
    OBJECT,
    ARRAY,
    ENUM,
    ANY_OF,
};

struct SchemaNode {
    SchemaType type = SchemaType::STRING;

    // OBJECT
    std::vector<std::pair<std::string, std::unique_ptr<SchemaNode>>> properties;
    std::vector<std::string> required;
    bool additional_properties = false;

    // ARRAY
    std::unique_ptr<SchemaNode> items;

    // ENUM
    std::vector<std::string> enum_values;

    // ANY_OF
    std::vector<std::unique_ptr<SchemaNode>> any_of;

    // Deep copy
    std::unique_ptr<SchemaNode> clone() const;
};

// Parse a JSON Schema string into a SchemaNode tree.
// Returns nullptr on parse failure.
std::unique_ptr<SchemaNode> parse_json_schema(const std::string& json);

} // namespace imp

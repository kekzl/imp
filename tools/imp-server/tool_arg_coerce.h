#pragma once
// Schema typing of text tool-call arguments as vLLM 0.33 does it (tool_parsers/utils.py
// extract_types_from_schema + coerce_to_schema_type, parser_engine.py _fix_arg_types): every value
// arrives as the emitted text and only its property schema may turn it into another JSON type.

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace imp {

// Types a property schema allows: "type" (string or list), enum value kinds, anyOf / oneOf / allOf
// recursively; lowercased, aliases folded ("int" -> "integer", "list" -> "array"). None: {"string"}.
std::vector<std::string> schema_arg_types(const nlohmann::ordered_json& schema);

// First type in null > integer > number > boolean > object > array > string that the text converts
// to, else the text as JSON, else the text. Non-finite numbers never leave as numbers.
nlohmann::ordered_json coerce_arg_text(const std::string& text, const std::vector<std::string>& types);

}  // namespace imp

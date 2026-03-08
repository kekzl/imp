#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace imp {

// Minimal TOML parser — supports only the subset needed for presets:
// sections, strings, integers, floats, booleans, string arrays.
// No nested tables, inline tables, or multiline strings.

using TomlValue = std::variant<std::string, int64_t, double, bool, std::vector<std::string>>;
using TomlTable = std::unordered_map<std::string, TomlValue>;

// Ordered list of (section_name, table) pairs. Order matches file order.
using TomlDocument = std::vector<std::pair<std::string, TomlTable>>;

// Parse a TOML file. Returns empty document on error.
TomlDocument toml_parse_file(const std::string& path);

// Parse TOML from a string.
TomlDocument toml_parse_string(const std::string& content);

// Typed accessors with defaults (no exceptions).
std::string toml_get_string(const TomlTable& t, const std::string& key, const std::string& def = "");
int64_t toml_get_int(const TomlTable& t, const std::string& key, int64_t def = 0);
double toml_get_float(const TomlTable& t, const std::string& key, double def = 0.0);
bool toml_get_bool(const TomlTable& t, const std::string& key, bool def = false);
std::vector<std::string> toml_get_string_array(const TomlTable& t, const std::string& key);

// Check if key exists.
bool toml_has(const TomlTable& t, const std::string& key);

// Find a section by name. Returns nullptr if not found.
const TomlTable* toml_find_section(const TomlDocument& doc, const std::string& name);

} // namespace imp

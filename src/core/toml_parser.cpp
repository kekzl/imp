#include "core/toml_parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace imp {

// Trim whitespace from both ends
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Parse a string array: ["a", "b", "c"]
static std::vector<std::string> parse_string_array(const std::string& val) {
    std::vector<std::string> result;
    size_t pos = val.find('[');
    if (pos == std::string::npos) return result;
    size_t end = val.rfind(']');
    if (end == std::string::npos || end <= pos) return result;

    std::string inner = val.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while (i < inner.size()) {
        size_t q1 = inner.find('"', i);
        if (q1 == std::string::npos) break;
        size_t q2 = inner.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        result.push_back(inner.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return result;
}

// Parse a single value
static TomlValue parse_value(const std::string& raw) {
    std::string val = trim(raw);
    if (val.empty()) return std::string{};

    // String: "..."
    if (val.front() == '"' && val.back() == '"' && val.size() >= 2)
        return val.substr(1, val.size() - 2);

    // String array: [...]
    if (val.front() == '[')
        return parse_string_array(val);

    // Boolean
    if (val == "true") return true;
    if (val == "false") return false;

    // Number: try int first, then float
    bool has_dot = val.find('.') != std::string::npos;
    if (has_dot) {
        try { return std::stod(val); } catch (...) {}
    } else {
        try { return static_cast<int64_t>(std::stoll(val)); } catch (...) {}
    }

    // Fallback: treat as string
    return val;
}

TomlDocument toml_parse_string(const std::string& content) {
    TomlDocument doc;
    std::string current_section;
    TomlTable current_table;

    auto flush_section = [&]() {
        if (!current_section.empty() || !current_table.empty()) {
            doc.emplace_back(current_section, std::move(current_table));
            current_table = {};
        }
    };

    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        // Section header: [name]
        if (trimmed.front() == '[' && trimmed.back() == ']') {
            flush_section();
            current_section = trim(trimmed.substr(1, trimmed.size() - 2));
            continue;
        }

        // Key = value
        size_t eq = trimmed.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(trimmed.substr(0, eq));
        std::string val_str = trimmed.substr(eq + 1);

        // Strip inline comment (not inside strings)
        bool in_str = false;
        bool in_arr = false;
        for (size_t i = 0; i < val_str.size(); ++i) {
            if (val_str[i] == '"') in_str = !in_str;
            if (val_str[i] == '[') in_arr = true;
            if (val_str[i] == ']') in_arr = false;
            if (val_str[i] == '#' && !in_str && !in_arr) {
                val_str = val_str.substr(0, i);
                break;
            }
        }

        current_table[key] = parse_value(val_str);
    }

    flush_section();
    return doc;
}

TomlDocument toml_parse_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};

    std::ostringstream buf;
    buf << file.rdbuf();
    return toml_parse_string(buf.str());
}

// --- Typed accessors ---

std::string toml_get_string(const TomlTable& t, const std::string& key, const std::string& def) {
    auto it = t.find(key);
    if (it == t.end()) return def;
    if (auto* s = std::get_if<std::string>(&it->second)) return *s;
    return def;
}

int64_t toml_get_int(const TomlTable& t, const std::string& key, int64_t def) {
    auto it = t.find(key);
    if (it == t.end()) return def;
    if (auto* v = std::get_if<int64_t>(&it->second)) return *v;
    if (auto* v = std::get_if<double>(&it->second))  return static_cast<int64_t>(*v);
    if (auto* v = std::get_if<bool>(&it->second))    return *v ? 1 : 0;
    return def;
}

double toml_get_float(const TomlTable& t, const std::string& key, double def) {
    auto it = t.find(key);
    if (it == t.end()) return def;
    if (auto* v = std::get_if<double>(&it->second))  return *v;
    if (auto* v = std::get_if<int64_t>(&it->second)) return static_cast<double>(*v);
    return def;
}

bool toml_get_bool(const TomlTable& t, const std::string& key, bool def) {
    auto it = t.find(key);
    if (it == t.end()) return def;
    if (auto* v = std::get_if<bool>(&it->second))    return *v;
    if (auto* v = std::get_if<int64_t>(&it->second)) return *v != 0;
    return def;
}

std::vector<std::string> toml_get_string_array(const TomlTable& t, const std::string& key) {
    auto it = t.find(key);
    if (it == t.end()) return {};
    if (auto* v = std::get_if<std::vector<std::string>>(&it->second)) return *v;
    return {};
}

bool toml_has(const TomlTable& t, const std::string& key) {
    return t.find(key) != t.end();
}

const TomlTable* toml_find_section(const TomlDocument& doc, const std::string& name) {
    for (auto& [sec, table] : doc) {
        if (sec == name) return &table;
    }
    return nullptr;
}

} // namespace imp

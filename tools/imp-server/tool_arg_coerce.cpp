// Port of vLLM 0.33 tool_parsers/utils.py extract_types_from_schema / coerce_to_schema_type
// (tool_arg_coerce.h). Python int() / float() / str.strip() / str.lower() semantics on ASCII.
#include "tool_arg_coerce.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_map>

using json = nlohmann::json;

namespace imp {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string_view strip(std::string_view s) {
    const auto ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (!s.empty() && ws(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && ws(s.back()))
        s.remove_suffix(1);
    return s;
}

// Python digit run: digits, single underscores only between digits. Appends the digits to out.
bool digit_run(std::string_view s, std::string& out) {
    if (s.empty() || !std::isdigit(static_cast<unsigned char>(s.front())) ||
        !std::isdigit(static_cast<unsigned char>(s.back())))
        return false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '_') {
            if (s[i - 1] == '_')
                return false;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(s[i])))
            return false;
        out += s[i];
    }
    return true;
}

// int(text): optional sign, one digit run. Values past int64 / uint64 are refused (vLLM: big int).
bool py_int(std::string_view text, json& out) {
    std::string_view s = strip(text);
    std::string num;
    if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
        if (s.front() == '-')
            num += '-';
        s.remove_prefix(1);
    }
    if (!digit_run(s, num))
        return false;
    errno = 0;
    char* end = nullptr;
    if (num.front() == '-') {
        const long long v = std::strtoll(num.c_str(), &end, 10);
        if (errno == ERANGE)
            return false;
        out = static_cast<int64_t>(v);
        return true;
    }
    const unsigned long long v = std::strtoull(num.c_str(), &end, 10);
    if (errno == ERANGE)
        return false;
    if (v <= static_cast<unsigned long long>(std::numeric_limits<int64_t>::max()))
        out = static_cast<int64_t>(v);
    else
        out = static_cast<uint64_t>(v);
    return true;
}

// float(text) for finite values: [sign] (digits[.digits] | .digits) [(e|E)[sign]digits].
bool py_float(std::string_view text, double& out) {
    std::string_view s = strip(text);
    std::string num;
    if (!s.empty() && (s.front() == '+' || s.front() == '-')) {
        num += s.front();
        s.remove_prefix(1);
    }
    std::string_view mant = s;
    std::string_view expo;
    const size_t e = s.find_first_of("eE");
    if (e != std::string_view::npos) {
        mant = s.substr(0, e);
        expo = s.substr(e + 1);
    }
    const size_t dot = mant.find('.');
    const std::string_view ip = dot == std::string_view::npos ? mant : mant.substr(0, dot);
    const std::string_view fp = dot == std::string_view::npos ? std::string_view() : mant.substr(dot + 1);
    if (ip.empty() && fp.empty())
        return false;
    if (!ip.empty() && !digit_run(ip, num))
        return false;
    if (dot != std::string_view::npos) {
        num += '.';
        if (!fp.empty() && !digit_run(fp, num))
            return false;
    }
    if (e != std::string_view::npos) {
        num += 'e';
        if (!expo.empty() && (expo.front() == '+' || expo.front() == '-')) {
            num += expo.front();
            expo.remove_prefix(1);
        }
        if (!digit_run(expo, num))
            return false;
    }
    const double v = std::strtod(num.c_str(), nullptr);
    if (!std::isfinite(v))
        return false;
    out = v;
    return true;
}

bool finite_json(const json& j) {
    if (j.is_number_float())
        return std::isfinite(j.get<double>());
    if (j.is_array() || j.is_object())
        return std::all_of(j.begin(), j.end(), [](const json& c) { return finite_json(c); });
    return true;
}

bool parse_json(const std::string& text, json& out) {
    json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !finite_json(j))
        return false;
    out = std::move(j);
    return true;
}

const std::unordered_map<std::string, std::string>& aliases() {
    static const std::unordered_map<std::string, std::string> m{
        {"str", "string"},     {"text", "string"},      {"varchar", "string"}, {"char", "string"},
        {"enum", "string"},    {"int", "integer"},      {"int32", "integer"},  {"int64", "integer"},
        {"uint", "integer"},   {"uint32", "integer"},   {"uint64", "integer"}, {"long", "integer"},
        {"short", "integer"},  {"unsigned", "integer"}, {"float", "number"},   {"float32", "number"},
        {"float64", "number"}, {"double", "number"},    {"bool", "boolean"},   {"dict", "object"},
        {"arr", "array"},      {"list", "array"},       {"sequence", "array"}};
    return m;
}

void collect_types(const json& schema, std::set<std::string>& types) {
    if (!schema.is_object())
        return;
    if (schema.contains("type")) {
        const json& t = schema["type"];
        if (t.is_string())
            types.insert(t.get<std::string>());
        else if (t.is_array())
            for (const auto& x : t)
                if (x.is_string())
                    types.insert(x.get<std::string>());
    }
    if (schema.contains("enum") && schema["enum"].is_array())
        for (const auto& v : schema["enum"]) {
            if (v.is_null())
                types.insert("null");
            else if (v.is_boolean())
                types.insert("boolean");
            else if (v.is_number_integer() || v.is_number_unsigned())
                types.insert("integer");
            else if (v.is_number_float())
                types.insert("number");
            else if (v.is_string())
                types.insert("string");
            else if (v.is_array())
                types.insert("array");
            else if (v.is_object())
                types.insert("object");
        }
    for (const char* f : {"anyOf", "oneOf", "allOf"})
        if (schema.contains(f) && schema[f].is_array())
            for (const auto& c : schema[f])
                collect_types(c, types);
}

}  // namespace

std::vector<std::string> schema_arg_types(const json& schema) {
    std::set<std::string> raw;
    collect_types(schema, raw);
    if (raw.empty())
        return {"string"};
    std::set<std::string> norm;
    for (const auto& t : raw) {
        std::string k = lower(std::string(strip(t)));
        const auto it = aliases().find(k);
        norm.insert(it == aliases().end() ? k : it->second);
    }
    return {norm.begin(), norm.end()};
}

json coerce_arg_text(const std::string& text, const std::vector<std::string>& types) {
    const auto has = [&](const char* t) { return std::find(types.begin(), types.end(), t) != types.end(); };
    json out;
    if (has("null") && lower(text) == "null")
        return nullptr;
    if (has("integer") && py_int(text, out))
        return out;
    if (has("number")) {
        if (py_int(text, out))
            return out;
        double v = 0.0;
        if (py_float(text, v)) {
            const double t = std::trunc(v);
            if (v == t && std::fabs(t) < 9.2e18)
                return static_cast<int64_t>(t);
            return v;
        }
    }
    if (has("boolean")) {
        const std::string b = lower(std::string(strip(text)));
        if (b == "true" || b == "1")
            return true;
        if (b == "false" || b == "0")
            return false;
    }
    if ((has("object") || has("array")) && parse_json(text, out))
        return out;
    if (has("string"))
        return text;
    return parse_json(text, out) ? out : json(text);
}

}  // namespace imp

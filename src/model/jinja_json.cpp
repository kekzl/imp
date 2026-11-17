// Jinja2 tojson and Python float repr (jinja_builtins.h).

#include "model/jinja_builtins.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace imp::jinja::detail {

namespace {

void json_escape_into(std::string& out, const std::string& s, bool ensure_ascii) {
    out += '"';
    for (size_t i = 0; i < s.size(); i++) {
        const auto c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"':
                out += "\\\"";
                continue;
            case '\\':
                out += "\\\\";
                continue;
            case '\n':
                out += "\\n";
                continue;
            case '\r':
                out += "\\r";
                continue;
            case '\t':
                out += "\\t";
                continue;
            case '\b':
                out += "\\b";
                continue;
            case '\f':
                out += "\\f";
                continue;
            default:
                break;
        }
        char hex[16];
        if (c < 0x20) {
            std::snprintf(hex, sizeof(hex), "\\u%04x", c);
            out += hex;
        } else if (c < 0x80 || !ensure_ascii) {
            out += static_cast<char>(c);
        } else {
            // Decode one UTF-8 sequence, emit \uXXXX (surrogate pair above the BMP).
            const int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
            uint32_t cp = c & (0xFF >> (len + 1));
            for (int k = 1; k < len && i + 1 < s.size(); k++)
                cp = (cp << 6) | (static_cast<unsigned char>(s[++i]) & 0x3F);
            if (cp >= 0x10000) {
                cp -= 0x10000;
                std::snprintf(hex, sizeof(hex), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
            } else {
                std::snprintf(hex, sizeof(hex), "\\u%04x", cp);
            }
            out += hex;
        }
    }
    out += '"';
}

void to_json_into(std::string& out, const Value& v, const JsonOpts& o, int depth);

void json_newline(std::string& out, const JsonOpts& o, int depth) {
    if (o.indent >= 0) {
        out += '\n';
        out.append(static_cast<size_t>(o.indent) * static_cast<size_t>(depth), ' ');
    }
}

void json_array_into(std::string& out, const Value::Array& arr, const JsonOpts& o, int depth) {
    if (arr.empty()) {
        out += "[]";
        return;
    }
    out += '[';
    for (size_t i = 0; i < arr.size(); i++) {
        if (i > 0)
            out += o.item_sep;
        json_newline(out, o, depth + 1);
        to_json_into(out, arr[i], o, depth + 1);
    }
    json_newline(out, o, depth);
    out += ']';
}

void json_object_into(std::string& out, const ObjectMap& obj, const JsonOpts& o, int depth) {
    if (obj.empty()) {
        out += "{}";
        return;
    }
    std::vector<const ObjectMap::value_type*> items;
    for (const auto& kv : obj)
        items.push_back(&kv);
    if (o.sort_keys)
        std::ranges::sort(items, [](auto* a, auto* b) { return a->first < b->first; });
    out += '{';
    for (size_t i = 0; i < items.size(); i++) {
        if (i > 0)
            out += o.item_sep;
        json_newline(out, o, depth + 1);
        json_escape_into(out, items[i]->first, o.ensure_ascii);
        out += o.key_sep;
        to_json_into(out, items[i]->second, o, depth + 1);
    }
    json_newline(out, o, depth);
    out += '}';
}

void to_json_into(std::string& out, const Value& v, const JsonOpts& o, int depth) {
    constexpr int kMaxJsonDepth = 128;
    if (depth >= kMaxJsonDepth || v.is_none())
        out += "null";
    else if (v.is_bool())
        out += v.as_bool() ? "true" : "false";
    else if (v.is_int())
        out += std::to_string(v.as_int());
    else if (v.is_double())
        out += std::isnan(v.as_double())   ? "NaN"
               : std::isinf(v.as_double()) ? (v.as_double() > 0 ? "Infinity" : "-Infinity")
                                           : py_float_repr(v.as_double());
    else if (v.is_string())
        json_escape_into(out, v.as_string(), o.ensure_ascii);
    else if (v.is_array())
        json_array_into(out, v.as_array(), o, depth);
    else if (v.is_object() && v.as_object())
        json_object_into(out, *v.as_object(), o, depth);
    else
        out += "null";
}
}  // namespace

std::string py_float_repr(double v) {
    if (std::isnan(v))
        return "nan";
    if (std::isinf(v))
        return v > 0 ? "inf" : "-inf";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string s(buf, res.ptr);
    const bool neg = s.front() == '-';
    if (neg)
        s.erase(0, 1);
    const size_t e = s.find('e');
    const int exp = std::atoi(s.c_str() + e + 1);
    std::string digits;
    for (char c : s.substr(0, e))
        if (c != '.')
            digits += c;
    const int decpt = exp + 1;  // digits d1d2.. scaled so the point sits after decpt digits
    std::string out;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0)
            out = "0." + std::string(static_cast<size_t>(-decpt), '0') + digits;
        else if (decpt >= static_cast<int>(digits.size()))
            out = digits + std::string(static_cast<size_t>(decpt) - digits.size(), '0') + ".0";
        else
            out = digits.substr(0, static_cast<size_t>(decpt)) + "." +
                  digits.substr(static_cast<size_t>(decpt));
    } else {
        out = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    }
    return neg ? "-" + out : out;
}

std::string to_json(const Value& v, const JsonOpts& opts) {
    std::string out;
    to_json_into(out, v, opts, 0);
    return out;
}

}  // namespace imp::jinja::detail

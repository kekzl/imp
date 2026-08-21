#include "model/json_util.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace imp {

JsonParser::JsonParser(std::string_view data) : data_(data.data()), len_(data.size()), pos_(0) {}

char JsonParser::peek() const {
    if (pos_ >= len_)
        return '\0';
    return data_[pos_];
}

char JsonParser::advance() {
    if (pos_ >= len_) {
        error_ = true;
        return '\0';
    }
    return data_[pos_++];
}

void JsonParser::skip_ws() {
    while (pos_ < len_ &&
           (data_[pos_] == ' ' || data_[pos_] == '\t' || data_[pos_] == '\n' || data_[pos_] == '\r')) {
        pos_++;
    }
}

bool JsonParser::expect(char c) {
    skip_ws();
    if (peek() == c) {
        advance();
        return true;
    }
    error_ = true;
    return false;
}

int JsonParser::hex_digit(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return 10 + c - 'a';
    if (c >= 'A' && c <= 'F')
        return 10 + c - 'A';
    return -1;
}

uint32_t JsonParser::parse_u4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        if (pos_ >= len_) {
            error_ = true;
            return 0;
        }
        int d = hex_digit(data_[pos_++]);
        if (d < 0) {
            error_ = true;
            return 0;
        }
        v = (v << 4) | d;
    }
    return v;
}

void JsonParser::append_codepoint_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

JValue JsonParser::parse() {
    skip_ws();
    return parse_value();
}

JValue JsonParser::parse_value() {
    skip_ws();
    if (error_)
        return {};
    char c = peek();
    if (c == '"')
        return parse_string_value();
    if (c == '{')
        return parse_object();
    if (c == '[')
        return parse_array();
    if (c == 't' || c == 'f')
        return parse_bool();
    if (c == 'n')
        return parse_null();
    if (c == '-' || (c >= '0' && c <= '9'))
        return parse_number();
    error_ = true;
    return {};
}

JValue JsonParser::parse_string_value() {
    JValue v;
    v.type = JType::STRING;
    v.str_val = parse_string_raw();
    return v;
}

std::string JsonParser::parse_string_raw() {
    if (!expect('"'))
        return "";
    std::string s;
    while (pos_ < len_) {
        char c = advance();
        if (c == '"')
            return s;
        if (c == '\\') {
            if (pos_ >= len_) {
                error_ = true;
                return s;
            }
            char esc = advance();
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
                case 'b':
                    s += '\b';
                    break;
                case 'f':
                    s += '\f';
                    break;
                case 'n':
                    s += '\n';
                    break;
                case 'r':
                    s += '\r';
                    break;
                case 't':
                    s += '\t';
                    break;
                case 'u': {
                    uint32_t cp = parse_u4();
                    // UTF-16 surrogate pair handling
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 1 < len_ && data_[pos_] == '\\' && data_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            uint32_t lo = parse_u4();
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                        }
                    }
                    append_codepoint_utf8(s, cp);
                    break;
                }
                default:
                    s += esc;
                    break;
            }
        } else {
            s += c;
        }
    }
    error_ = true;
    return s;
}

JValue JsonParser::parse_number() {
    JValue v;
    v.type = JType::NUMBER;
    size_t start = pos_;
    if (peek() == '-')
        advance();
    while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
        advance();
    if (pos_ < len_ && data_[pos_] == '.') {
        advance();
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
            advance();
    }
    if (pos_ < len_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
        advance();
        if (pos_ < len_ && (data_[pos_] == '+' || data_[pos_] == '-'))
            advance();
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9')
            advance();
    }
    std::string num_str(data_ + start, pos_ - start);
    v.num_val = std::stod(num_str);
    return v;
}

JValue JsonParser::parse_object() {
    JValue v;
    v.type = JType::OBJECT;
    if (!expect('{'))
        return v;
    skip_ws();
    if (peek() == '}') {
        advance();
        return v;
    }
    while (!error_) {
        skip_ws();
        std::string key = parse_string_raw();
        if (!expect(':'))
            break;
        JValue val = parse_value();
        v.obj.emplace_back(std::move(key), std::move(val));
        skip_ws();
        if (peek() == ',') {
            advance();
            continue;
        }
        break;
    }
    expect('}');
    return v;
}

JValue JsonParser::parse_array() {
    JValue v;
    v.type = JType::ARRAY;
    if (!expect('['))
        return v;
    skip_ws();
    if (peek() == ']') {
        advance();
        return v;
    }
    while (!error_) {
        v.arr.push_back(parse_value());
        skip_ws();
        if (peek() == ',') {
            advance();
            continue;
        }
        break;
    }
    expect(']');
    return v;
}

JValue JsonParser::parse_bool() {
    JValue v;
    v.type = JType::NUMBER;
    if (peek() == 't') {
        for (int i = 0; i < 4 && pos_ < len_; i++)
            advance();
        v.num_val = 1.0;
    } else {
        for (int i = 0; i < 5 && pos_ < len_; i++)
            advance();
        v.num_val = 0.0;
    }
    return v;
}

JValue JsonParser::parse_null() {
    JValue v;
    v.type = JType::NUL;
    for (int i = 0; i < 4 && pos_ < len_; i++)
        advance();
    return v;
}

const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) {
        if (kv.first == key)
            return &kv.second;
    }
    return nullptr;
}

bool jobj_get_float(const JValue& obj, const std::string& key, float& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::NUMBER)
        return false;
    out = static_cast<float>(v->num_val);
    return true;
}

bool jobj_get_string(const JValue& obj, const std::string& key, std::string& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::STRING)
        return false;
    out = v->str_val;
    return true;
}

std::string read_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return "";
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        return "";
    }
    std::string data(st.st_size, '\0');
    if (::read(fd, data.data(), st.st_size) != st.st_size) {
        close(fd);
        return "";
    }
    close(fd);
    return data;
}

bool parse_json_file(const std::string& path, JValue& out) {
    std::string data = read_file(path);
    if (data.empty())
        return false;

    JsonParser parser(data);
    out = parser.parse();
    if (!parser.ok() || out.type != JType::OBJECT) {
        IMP_LOG_WARN("failed to parse JSON: %s", path.c_str());
        return false;
    }
    return true;
}

}  // namespace imp

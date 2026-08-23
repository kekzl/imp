#pragma once

// Machine-readable output for the shipped binaries (#1583).
//
// Two pieces:
//
//  - `JsonOut`, a 60-line object writer. imp-cli does not link nlohmann/json
//    and the documents here are flat maps of numbers and short strings, so a
//    dependency would buy nothing.
//  - `json_stdout_reserve()` / `json_emit()`, which make the "stdout carries
//    EXACTLY one JSON document" promise structural. Rather than auditing every
//    print site in a 1000-line main() - and every one added later - stdout is
//    pointed at stderr for the whole run and the real stdout is kept on a
//    private fd. A print site added tomorrow cannot break the contract.

#include <cstdio>
#include <string>
#include <unistd.h>

namespace imp_tools {

class JsonOut {
public:
    JsonOut& key(const std::string& k) {
        sep_();
        buf_ += '"';
        escape_(k);
        buf_ += "\":";
        return *this;
    }
    JsonOut& str(const std::string& k, const std::string& v) {
        key(k);
        buf_ += '"';
        escape_(v);
        buf_ += '"';
        return *this;
    }
    JsonOut& num(const std::string& k, double v, int prec = 4) {
        key(k);
        char tmp[64];
        std::snprintf(tmp, sizeof(tmp), "%.*f", prec, v);
        buf_ += tmp;
        return *this;
    }
    JsonOut& intg(const std::string& k, long long v) {
        key(k);
        buf_ += std::to_string(v);
        return *this;
    }
    JsonOut& boolean(const std::string& k, bool v) {
        key(k);
        buf_ += v ? "true" : "false";
        return *this;
    }
    // Nested object, already rendered by another JsonOut.
    JsonOut& obj(const std::string& k, const JsonOut& v) {
        key(k);
        buf_ += v.str();
        return *this;
    }
    std::string str() const { return "{" + buf_ + "}"; }

private:
    void sep_() {
        if (!buf_.empty())
            buf_ += ',';
    }
    void escape_(const std::string& s) {
        for (unsigned char c : s) {
            switch (c) {
                case '"':
                    buf_ += "\\\"";
                    break;
                case '\\':
                    buf_ += "\\\\";
                    break;
                case '\n':
                    buf_ += "\\n";
                    break;
                case '\r':
                    buf_ += "\\r";
                    break;
                case '\t':
                    buf_ += "\\t";
                    break;
                default:
                    if (c < 0x20) {
                        char tmp[8];
                        std::snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                        buf_ += tmp;
                    } else {
                        buf_ += static_cast<char>(c);
                    }
            }
        }
    }
    std::string buf_;
};

// Point stdout at stderr and keep the real stdout for json_emit(). No-op if
// called twice.
inline int& json_reserved_fd() {
    static int fd = -1;
    return fd;
}

inline void json_stdout_reserve() {
    if (json_reserved_fd() >= 0)
        return;
    std::fflush(stdout);
    const int saved = ::dup(STDOUT_FILENO);
    if (saved < 0)
        return;  // no fd to spare: leave stdout alone, json_emit() stays silent
    json_reserved_fd() = saved;
    ::dup2(STDERR_FILENO, STDOUT_FILENO);
}

// Write the document (plus a newline) to the reserved fd and close it. Safe to
// call when reserve() was never called or already emitted: it does nothing.
inline void json_emit(const std::string& doc) {
    int& fd = json_reserved_fd();
    if (fd < 0)
        return;
    std::fflush(stdout);
    const std::string line = doc + "\n";
    size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::write(fd, line.data() + off, line.size() - off);
        if (n <= 0)
            break;
        off += static_cast<size_t>(n);
    }
    ::close(fd);
    fd = -1;
}

}  // namespace imp_tools

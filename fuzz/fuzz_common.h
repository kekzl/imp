#pragma once

// Shared plumbing for the fuzz targets (#1620).

#include "core/logging.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace imp_fuzz {

// A parser under fuzz logs on every malformed input, which at a few hundred
// thousand executions per second is the only thing the run would produce.
// Called once per process.
inline void quiet_logs() {
    static bool done = false;
    if (!done) {
        done = true;
        imp::log_set_level(imp::LogLevel::FATAL);
    }
}

// The two loader targets take a PATH, not a buffer, so the input has to become
// a file. Named with the extension the loader dispatches on.
class TempFile {
public:
    TempFile(const uint8_t* data, size_t size, const char* suffix) {
        char tmpl[] = "/tmp/imp_fuzz_XXXXXX";
        int fd = ::mkstemp(tmpl);
        if (fd < 0)
            return;
        ::close(fd);
        path_ = std::string(tmpl) + suffix;
        ::rename(tmpl, path_.c_str());
        FILE* f = ::fopen(path_.c_str(), "wb");
        if (!f) {
            path_.clear();
            return;
        }
        if (size > 0)
            ::fwrite(data, 1, size, f);
        ::fclose(f);
    }
    ~TempFile() {
        if (!path_.empty())
            ::remove(path_.c_str());
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    bool ok() const { return !path_.empty(); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

// Full UTF-8 validation. imp::stream::utf8_complete_len only inspects the TAIL
// of a buffer (it answers "does this end mid-character"), so it says nothing
// about a byte flipped in the middle - which is exactly what a mutator
// produces. Using it as an input filter let the tool-stream target report its
// own mutated garbage as findings.
inline bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = 0;
        unsigned cp = 0;
        if (c < 0x80) {
            i++;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            n = 1;
            cp = c & 0x1Fu;
        } else if ((c & 0xF0) == 0xE0) {
            n = 2;
            cp = c & 0x0Fu;
        } else if ((c & 0xF8) == 0xF0) {
            n = 3;
            cp = c & 0x07u;
        } else {
            return false;  // continuation byte or 5+ byte lead
        }
        if (i + n >= s.size() + 0 && i + n > s.size() - 1)
            return false;
        for (size_t k = 1; k <= n; k++) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80)
                return false;
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        // Overlong forms, surrogates and out-of-range are ill-formed too.
        if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000))
            return false;
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        i += n + 1;
    }
    return true;
}

// An input the fuzzer grew past anything a real file would be tells us nothing
// new and costs the whole time budget. The loaders themselves cap at 128 MiB
// (kMaxHeaderBytes); this is about keeping one execution short.
constexpr size_t kMaxInput = 1u << 20;  // 1 MiB

}  // namespace imp_fuzz

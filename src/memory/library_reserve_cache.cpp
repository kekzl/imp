#include "memory/library_reserve_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace imp {

std::string LibraryReserveKey::str() const {
    // Fixed-width hex so the file stays greppable and a key can never contain a
    // tab or newline, which is the whole parsing contract.
    char buf[80];
    std::snprintf(buf, sizeof(buf), "%016llx-nv%d-fp8%d-cuda%d",
                  static_cast<unsigned long long>(model_fingerprint), nvfp4_decode_mode,
                  fp8_prefill ? 1 : 0, cuda_runtime_version);
    return buf;
}

std::string library_reserve_cache_default_path() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::string(xdg) + "/imp/library_reserve";
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::string(home) + "/.cache/imp/library_reserve";
    return {};
}

namespace {

// key -> bytes, in file order. A duplicate key keeps the LAST value, so a
// re-measurement always wins over a stale one.
std::vector<std::pair<std::string, size_t>> read_all(const std::string& path) {
    std::vector<std::pair<std::string, size_t>> out;
    std::ifstream in(path);
    if (!in)
        return out;
    std::string line;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos || tab == 0)
            continue;  // malformed line: skip it, do not fail the load
        const std::string key = line.substr(0, tab);
        errno = 0;
        char* end = nullptr;
        const unsigned long long v = std::strtoull(line.c_str() + tab + 1, &end, 10);
        if (errno != 0 || end == line.c_str() + tab + 1)
            continue;
        bool replaced = false;
        for (auto& kv : out) {
            if (kv.first == key) {
                kv.second = static_cast<size_t>(v);
                replaced = true;
                break;
            }
        }
        if (!replaced)
            out.emplace_back(key, static_cast<size_t>(v));
    }
    return out;
}

}  // namespace

size_t library_reserve_cache_load(const std::string& path, const LibraryReserveKey& key,
                                  bool* found) {
    if (found)
        *found = false;
    if (path.empty())
        return 0;
    const std::string k = key.str();
    for (const auto& [name, bytes] : read_all(path)) {
        if (name == k) {
            if (found)
                *found = true;  // a recorded 0 is an answer, not a miss
            return bytes;
        }
    }
    return 0;
}

bool library_reserve_cache_store(const std::string& path, const LibraryReserveKey& key,
                                 size_t bytes) {
    if (path.empty())
        return false;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    // ec is ignored on purpose: the directory may already exist, and if it truly
    // cannot be created the ofstream below fails and we report that instead.

    auto entries = read_all(path);
    const std::string k = key.str();
    bool replaced = false;
    for (auto& kv : entries) {
        if (kv.first == k) {
            kv.second = bytes;
            replaced = true;
            break;
        }
    }
    if (!replaced)
        entries.emplace_back(k, bytes);

    // Write to a sibling temp then rename: two processes starting together must
    // not leave a half-written cache behind, and a torn file would be read as
    // "no entry" on the next boot — silently losing the measurement rather than
    // noisily failing.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out)
            return false;
        out << "# imp library-reserve cache — what the first forward pass actually\n"
               "# claimed, per (model, quant path, CUDA runtime). Safe to delete;\n"
               "# imp re-measures and rewrites it. See docs/MEMORY_ARCHITECTURE.md A1.5.\n";
        for (const auto& [name, v] : entries)
            out << name << '\t' << v << '\n';
        if (!out)
            return false;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

}  // namespace imp

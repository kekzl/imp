#include "quant/calibration_stats.h"

#include <cstdio>
#include <cstring>

namespace imp {

namespace {

constexpr char kMagic[8] = {'I', 'M', 'P', 'C', 'A', 'L', '0', '1'};
// A calibration file is small (K floats per weight), so an entry claiming a
// K in the hundreds of millions is a corrupt read, not a large model.
constexpr int64_t kMaxK = 1 << 22;
constexpr uint32_t kMaxEntries = 1u << 20;
constexpr uint32_t kMaxStrLen = 1024;

template <typename T>
bool put(FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
bool get(FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

bool put_str(FILE* f, const std::string& s) {
    uint32_t n = static_cast<uint32_t>(s.size());
    return put(f, n) && (n == 0 || std::fwrite(s.data(), 1, n, f) == n);
}

bool get_str(FILE* f, std::string& s) {
    uint32_t n = 0;
    if (!get(f, n) || n > kMaxStrLen)
        return false;
    s.assign(n, '\0');
    return n == 0 || std::fread(s.data(), 1, n, f) == n;
}

}  // namespace

const CalibrationEntry* CalibrationStats::find(int layer, const std::string& kind) const {
    for (const auto& e : entries)
        if (e.layer == layer && e.kind == kind)
            return &e;
    return nullptr;
}

std::string write_calibration_stats(const std::string& path, const CalibrationStats& stats) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return "cannot open " + path + " for writing";
    bool ok = std::fwrite(kMagic, 1, sizeof(kMagic), f) == sizeof(kMagic);
    ok = ok && put_str(f, stats.model_id);
    ok = ok && put(f, static_cast<uint32_t>(stats.entries.size()));
    for (const auto& e : stats.entries) {
        ok = ok && put(f, static_cast<int32_t>(e.layer));
        ok = ok && put_str(f, e.kind);
        ok = ok && put(f, static_cast<uint64_t>(e.rows));
        ok = ok && put(f, static_cast<int64_t>(e.mean_abs.size()));
        ok = ok && (e.mean_abs.empty() ||
                    std::fwrite(e.mean_abs.data(), sizeof(float), e.mean_abs.size(), f) == e.mean_abs.size());
        if (!ok)
            break;
    }
    // A short write here silently truncates the file the quantizer will trust,
    // so the close result is part of the contract.
    ok = (std::fclose(f) == 0) && ok;
    if (!ok)
        return "failed writing " + path;
    return {};
}

std::string read_calibration_stats(const std::string& path, CalibrationStats& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "cannot open " + path;
    char magic[sizeof(kMagic)] = {};
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        std::fclose(f);
        return path + " is not an imp calibration file";
    }
    uint32_t n = 0;
    if (!get_str(f, out.model_id) || !get(f, n) || n > kMaxEntries) {
        std::fclose(f);
        return "corrupt header in " + path;
    }
    out.entries.clear();
    out.entries.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        CalibrationEntry e;
        int32_t layer = 0;
        uint64_t rows = 0;
        int64_t k = 0;
        if (!get(f, layer) || !get_str(f, e.kind) || !get(f, rows) || !get(f, k) || k < 0 || k > kMaxK) {
            std::fclose(f);
            return "corrupt entry " + std::to_string(i) + " in " + path;
        }
        e.layer = layer;
        e.rows = rows;
        e.mean_abs.resize(static_cast<size_t>(k));
        if (k > 0 && std::fread(e.mean_abs.data(), sizeof(float), static_cast<size_t>(k), f) !=
                         static_cast<size_t>(k)) {
            std::fclose(f);
            return "truncated entry " + std::to_string(i) + " in " + path;
        }
        out.entries.push_back(std::move(e));
    }
    std::fclose(f);
    return {};
}

}  // namespace imp

#include "quant/calibration_hessian.h"

#include <cstring>

namespace imp {

namespace {

constexpr char kMagic[8] = {'I', 'M', 'P', 'H', 'E', 'S', '0', '1'};
constexpr char kMagic2[8] = {'I', 'M', 'P', 'H', 'E', 'S', '0', '2'};
constexpr int64_t kMaxK = 1 << 16;
constexpr uint32_t kMaxEntries = 1u << 16;
constexpr uint32_t kMaxStrLen = 64;

template <typename T>
bool put(FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}
template <typename T>
bool get(FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

}  // namespace

HessianWriter::~HessianWriter() {
    if (f_)
        std::fclose(f_);
}

std::string HessianWriter::open(const std::string& path, uint32_t count, uint32_t mats) {
    if (mats != 1 && mats != 2)
        return "unsupported matrix count " + std::to_string(mats);
    mats_ = mats;
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_)
        return "cannot open " + path + " for writing";
    const char* magic = mats == 1 ? kMagic : kMagic2;
    if (std::fwrite(magic, 1, sizeof(kMagic), f_) != sizeof(kMagic) || !put(f_, count) ||
        (mats == 2 && !put(f_, mats)))
        return "write failed: " + path;
    return "";
}

std::string HessianWriter::add(int layer, const std::string& kind, int64_t K, uint64_t rows, const float* m0,
                               const float* m1) {
    const auto len = static_cast<uint32_t>(kind.size());
    const size_t n = static_cast<size_t>(K) * static_cast<size_t>(K);
    if (mats_ == 2 && !m1)
        return "second matrix missing at layer " + std::to_string(layer) + " " + kind;
    if (!f_ || !put(f_, static_cast<int32_t>(layer)) || !put(f_, len) ||
        std::fwrite(kind.data(), 1, len, f_) != len || !put(f_, K) || !put(f_, rows) ||
        std::fwrite(m0, sizeof(float), n, f_) != n ||
        (mats_ == 2 && std::fwrite(m1, sizeof(float), n, f_) != n))
        return "write failed at layer " + std::to_string(layer) + " " + kind;
    return "";
}

std::string HessianWriter::close() {
    if (!f_)
        return "";
    const bool ok = std::fclose(f_) == 0;
    f_ = nullptr;
    return ok ? "" : "close failed";
}

std::string read_hessian_index(const std::string& path, std::vector<HessianIndexEntry>& out) {
    out.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "cannot open " + path;
    // fseek past the end succeeds, so the data skip is checked against the file size.
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string err;
    char magic[8];
    uint32_t count = 0, mats = 1;
    const bool read_magic = std::fread(magic, 1, sizeof(magic), f) == sizeof(magic);
    const bool v2 = read_magic && std::memcmp(magic, kMagic2, sizeof(magic)) == 0;
    if (!read_magic || (!v2 && std::memcmp(magic, kMagic, sizeof(magic)) != 0) || !get(f, count) ||
        count > kMaxEntries || (v2 && (!get(f, mats) || mats != 2))) {
        err = path + ": not an IMPHES01/IMPHES02 file";
    }
    for (uint32_t i = 0; err.empty() && i < count; i++) {
        HessianIndexEntry e;
        int32_t layer = 0;
        uint32_t len = 0;
        if (!get(f, layer) || !get(f, len) || len > kMaxStrLen) {
            err = path + ": truncated entry " + std::to_string(i);
            break;
        }
        e.kind.assign(len, '\0');
        if (std::fread(e.kind.data(), 1, len, f) != len || !get(f, e.K) || !get(f, e.rows) || e.K <= 0 ||
            e.K > kMaxK) {
            err = path + ": bad entry " + std::to_string(i);
            break;
        }
        e.layer = layer;
        e.mats = mats;
        e.offset = static_cast<uint64_t>(std::ftell(f));
        const auto bytes = static_cast<long>(static_cast<uint64_t>(e.K) * static_cast<uint64_t>(e.K) *
                                             sizeof(float) * mats);
        if (std::fseek(f, bytes, SEEK_CUR) != 0 || std::ftell(f) > size) {
            err = path + ": truncated data " + std::to_string(i);
            break;
        }
        out.push_back(std::move(e));
    }
    std::fclose(f);
    return err;
}

std::string read_hessian(const std::string& path, const HessianIndexEntry& e, std::vector<float>& out,
                         uint32_t mat) {
    if (mat >= e.mats)
        return path + ": entry has no matrix " + std::to_string(mat);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "cannot open " + path;
    const size_t n = static_cast<size_t>(e.K) * static_cast<size_t>(e.K);
    out.resize(n);
    const auto at = static_cast<long>(e.offset + (static_cast<uint64_t>(mat) * n * sizeof(float)));
    const bool ok = std::fseek(f, at, SEEK_SET) == 0 && std::fread(out.data(), sizeof(float), n, f) == n;
    std::fclose(f);
    return ok ? "" : path + ": short read at layer " + std::to_string(e.layer) + " " + e.kind;
}

const HessianIndexEntry* find_hessian(const std::vector<HessianIndexEntry>& idx, int layer,
                                      const std::string& kind) {
    auto find = [&](const std::string& k) -> const HessianIndexEntry* {
        for (const auto& e : idx)
            if (e.layer == layer && e.kind == k)
                return &e;
        return nullptr;
    };
    if (const auto* e = find(kind))
        return e;
    if (kind == "WK" || kind == "WV")
        return find("WQ");
    if (kind == "W_UP")
        return find("W_GATE");
    return nullptr;
}

}  // namespace imp

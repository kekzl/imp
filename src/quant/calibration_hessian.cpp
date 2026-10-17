#include "quant/calibration_hessian.h"

#include <cstring>

namespace imp {

namespace {

constexpr char kMagic[8] = {'I', 'M', 'P', 'H', 'E', 'S', '0', '1'};
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

std::string HessianWriter::open(const std::string& path, uint32_t count) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_)
        return "cannot open " + path + " for writing";
    if (std::fwrite(kMagic, 1, sizeof(kMagic), f_) != sizeof(kMagic) || !put(f_, count))
        return "write failed: " + path;
    return "";
}

std::string HessianWriter::add(int layer, const std::string& kind, int64_t K, uint64_t rows, const float* h) {
    const auto len = static_cast<uint32_t>(kind.size());
    const size_t n = static_cast<size_t>(K) * static_cast<size_t>(K);
    if (!f_ || !put(f_, static_cast<int32_t>(layer)) || !put(f_, len) ||
        std::fwrite(kind.data(), 1, len, f_) != len || !put(f_, K) || !put(f_, rows) ||
        std::fwrite(h, sizeof(float), n, f_) != n)
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
    std::string err;
    char magic[8];
    uint32_t count = 0;
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::memcmp(magic, kMagic, sizeof(magic)) != 0 || !get(f, count) || count > kMaxEntries) {
        err = path + ": not an IMPHES01 file";
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
        e.offset = static_cast<uint64_t>(std::ftell(f));
        const auto bytes = static_cast<long>(static_cast<uint64_t>(e.K) * static_cast<uint64_t>(e.K) *
                                             sizeof(float));
        if (std::fseek(f, bytes, SEEK_CUR) != 0) {
            err = path + ": truncated data " + std::to_string(i);
            break;
        }
        out.push_back(std::move(e));
    }
    std::fclose(f);
    return err;
}

std::string read_hessian(const std::string& path, const HessianIndexEntry& e, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return "cannot open " + path;
    const size_t n = static_cast<size_t>(e.K) * static_cast<size_t>(e.K);
    out.resize(n);
    const bool ok = std::fseek(f, static_cast<long>(e.offset), SEEK_SET) == 0 &&
                    std::fread(out.data(), sizeof(float), n, f) == n;
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

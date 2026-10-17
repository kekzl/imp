#pragma once

// Second-moment calibration for GPTQ (roadmap row 101): per (layer, kind) the mean of x^T x over
// the corpus rows, FP32 [K, K]. File IMPHES01: magic, u32 count, then per entry i32 layer,
// u32 len + kind, i64 K, u64 rows, K*K floats. Qwen3-1.7B: about 5.6 GB, so entries are read
// one at a time through an index.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {

struct HessianIndexEntry {
    int layer = -1;
    std::string kind;  // tensor_kind_name(), e.g. "WQ"
    int64_t K = 0;
    uint64_t rows = 0;
    uint64_t offset = 0;  // byte offset of the first float
};

// Streams entries into one file; "" on success, else the reason (the file is then incomplete).
class HessianWriter {
public:
    ~HessianWriter();
    std::string open(const std::string& path, uint32_t count);
    std::string add(int layer, const std::string& kind, int64_t K, uint64_t rows, const float* mean_xxt);
    std::string close();

private:
    FILE* f_ = nullptr;
};

std::string read_hessian_index(const std::string& path, std::vector<HessianIndexEntry>& out);
std::string read_hessian(const std::string& path, const HessianIndexEntry& e, std::vector<float>& out);

// The entry GPTQ uses for a weight of `kind` at `layer`: the kind itself, else the kind whose
// input it shares (WK, WV -> WQ; W_UP -> W_GATE). nullptr when none.
const HessianIndexEntry* find_hessian(const std::vector<HessianIndexEntry>& idx, int layer,
                                      const std::string& kind);

}  // namespace imp

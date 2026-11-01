#pragma once

// Second-moment calibration for GPTQ (roadmap row 101): per (layer, kind) the mean of x^T x over
// the corpus rows, FP32 [K, K]. File IMPHES01: magic, u32 count, then per entry i32 layer,
// u32 len + kind, i64 K, u64 rows, K*K floats. Qwen3-1.7B: about 5.6 GB, so entries are read
// one at a time through an index.
// IMPHES02 (GPTAQ): magic, u32 count, u32 mats, then per entry the same header and `mats` K*K
// matrices: 0 = mean xq^T xq of the quantized export's inputs, 1 = C, C[i*K+j] = mean x_i xq_j
// with x the BF16 source's input on the same token.

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
    uint64_t offset = 0;  // byte offset of the first float of matrix 0
    uint32_t mats = 1;    // matrices per entry: 1 (IMPHES01) or 2 (IMPHES02)
};

// Streams entries into one file; "" on success, else the reason (the file is then incomplete).
class HessianWriter {
public:
    ~HessianWriter();
    // mats 1 writes IMPHES01, 2 writes IMPHES02.
    std::string open(const std::string& path, uint32_t count, uint32_t mats = 1);
    // m1 is required when the file holds 2 matrices per entry, ignored otherwise.
    std::string add(int layer, const std::string& kind, int64_t K, uint64_t rows, const float* m0,
                    const float* m1 = nullptr);
    std::string close();

private:
    FILE* f_ = nullptr;
    uint32_t mats_ = 1;
};

std::string read_hessian_index(const std::string& path, std::vector<HessianIndexEntry>& out);
// Matrix `mat` (< e.mats) of entry e.
std::string read_hessian(const std::string& path, const HessianIndexEntry& e, std::vector<float>& out,
                         uint32_t mat = 0);

// The entry GPTQ uses for a weight of `kind` at `layer`: the kind itself, else the kind whose
// input it shares (WK, WV -> WQ; W_UP -> W_GATE). nullptr when none.
const HessianIndexEntry* find_hessian(const std::vector<HessianIndexEntry>& idx, int layer,
                                      const std::string& kind);

}  // namespace imp

#include "quant/dequant_gptq.h"

#include <algorithm>
#include <cctype>

namespace imp::gptq {

bool parse_zero_format(const std::string& checkpoint_format, ZeroFormat* out) {
    std::string f = checkpoint_format;
    std::transform(f.begin(), f.end(), f.begin(), [](unsigned char c) { return std::tolower(c); });
    if (f.empty() || f == "gptq") {
        *out = ZeroFormat::V1;
        return true;
    }
    if (f == "gptq_v2") {
        *out = ZeroFormat::V2;
        return true;
    }
    return false;
}

bool check_shapes(const int64_t* qweight_shape, const int64_t* qzeros_shape, const int64_t* scales_shape,
                  int group_size, Dims* d, std::string* err) {
    const int64_t k = qweight_shape[0] * 8;
    const int64_t n = qweight_shape[1];
    auto fail = [&](const char* what) {
        if (err)
            *err = std::string(what) + " (qweight [" + std::to_string(qweight_shape[0]) + ", " +
                   std::to_string(qweight_shape[1]) + "], group_size " + std::to_string(group_size) + ")";
        return false;
    };
    if (k <= 0 || n <= 0)
        return fail("empty qweight");
    if (group_size == 0 || group_size < -1)
        return fail("group_size must be > 0 or -1");
    if (n % 8 != 0)
        return fail("N is not a multiple of 8");
    if (k > INT32_MAX || n > INT32_MAX)
        return fail("K or N exceeds INT32_MAX");
    const int64_t gs = group_size == -1 ? k : group_size;
    const int64_t groups = (k + gs - 1) / gs;
    if (qzeros_shape[0] != groups || qzeros_shape[1] != n / 8)
        return fail("qzeros is not [ceil(K/group_size), N/8]");
    if (scales_shape[0] != groups || scales_shape[1] != n)
        return fail("scales is not [ceil(K/group_size), N]");
    d->K = static_cast<int>(k);
    d->N = static_cast<int>(n);
    d->groups = static_cast<int>(groups);
    d->group_size = static_cast<int>(gs);
    return true;
}

bool check_g_idx(const int32_t* g_idx, int64_t len, const Dims& d, std::string* err) {
    if (len != d.K) {
        if (err)
            *err = "g_idx has " + std::to_string(len) + " entries, K is " + std::to_string(d.K);
        return false;
    }
    for (int64_t k = 0; k < len; ++k)
        if (g_idx[k] < 0 || g_idx[k] >= d.groups) {
            if (err)
                *err = "g_idx[" + std::to_string(k) + "] = " + std::to_string(g_idx[k]) + " outside [0, " +
                       std::to_string(d.groups) + ")";
            return false;
        }
    return true;
}

void dequant4_host(__half* out, const int32_t* qweight, const int32_t* qzeros, const __half* scales,
                   const int32_t* g_idx, int N, int K, int group_size, ZeroFormat fmt) {
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            out[static_cast<int64_t>(n) * K + k] = dequant_elem(qweight, qzeros, scales, g_idx, N, group_size,
                                                                fmt, k, n);
}

}  // namespace imp::gptq

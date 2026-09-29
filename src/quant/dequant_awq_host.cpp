#include "quant/dequant_awq.h"

namespace imp::awq {

bool check_shapes(const int64_t* qweight_shape, const int64_t* qzeros_shape, const int64_t* scales_shape,
                  int group_size, int* K, int* N, std::string* err) {
    const int64_t k = qweight_shape[0];
    const int64_t n = qweight_shape[1] * 8;
    auto fail = [&](const char* what) {
        if (err)
            *err = std::string(what) + " (qweight [" + std::to_string(qweight_shape[0]) + ", " +
                   std::to_string(qweight_shape[1]) + "], group_size " + std::to_string(group_size) + ")";
        return false;
    };
    if (group_size <= 0 || k <= 0 || n <= 0)
        return fail("empty qweight or group_size <= 0");
    if (k % group_size != 0)
        return fail("K is not a multiple of group_size");
    if (k > INT32_MAX || n > INT32_MAX)
        return fail("K or N exceeds INT32_MAX");
    const int64_t groups = k / group_size;
    if (qzeros_shape[0] != groups || qzeros_shape[1] != n / 8)
        return fail("qzeros is not [K/group_size, N/8]");
    if (scales_shape[0] != groups || scales_shape[1] != n)
        return fail("scales is not [K/group_size, N]");
    *K = static_cast<int>(k);
    *N = static_cast<int>(n);
    return true;
}

void dequant4_host(__half* out, const int32_t* qweight, const int32_t* qzeros, const __half* scales, int N,
                   int K, int group_size) {
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            out[static_cast<int64_t>(n) * K + k] = dequant_elem(qweight, qzeros, scales, N, group_size, k, n);
}

}  // namespace imp::awq

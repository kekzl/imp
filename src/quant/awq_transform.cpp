#include "quant/awq_transform.h"

#include "core/fp_bits.h"

#include <cuda_fp16.h>

#include <cstring>

namespace imp {

namespace {

// __half is a class with a protected member, so memcpy over it is UB the
// compiler warns about. __half_raw is the documented bit-level view.
float fp16_at(const std::vector<uint16_t>& v, size_t i) {
    __half_raw raw;
    raw.x = v[i];
    return __half2float(__half(raw));
}

void fp16_set(std::vector<uint16_t>& v, size_t i, float f) {
    const __half_raw raw(__float2half(f));
    v[i] = raw.x;
}

}  // namespace

void awq_apply_matrix(std::vector<uint16_t>& fp16, int64_t N, int64_t K, const std::vector<float>& row_div,
                      const std::vector<float>& col_scale) {
    if (N <= 0 || K <= 0 || static_cast<int64_t>(fp16.size()) != N * K)
        return;
    const bool do_rows = static_cast<int64_t>(row_div.size()) == N;
    const bool do_cols = static_cast<int64_t>(col_scale.size()) == K;
    if (!do_rows && !do_cols)
        return;
    for (int64_t i = 0; i < N; i++) {
        const float d = do_rows ? row_div[static_cast<size_t>(i)] : 1.0f;
        const float inv_d = (d == 0.0f) ? 1.0f : 1.0f / d;
        for (int64_t j = 0; j < K; j++) {
            const float c = do_cols ? col_scale[static_cast<size_t>(j)] : 1.0f;
            const float f = inv_d * c;
            if (f == 1.0f)
                continue;
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(j);
            fp16_set(fp16, idx, fp16_at(fp16, idx) * f);
        }
    }
}

bool awq_apply_vector_div(unsigned char* bytes, size_t n_elems, const std::string& dtype,
                          const std::vector<float>& div) {
    if (!bytes || div.size() != n_elems)
        return false;
    if (dtype != "F32" && dtype != "F16" && dtype != "BF16")
        return false;
    for (size_t i = 0; i < n_elems; i++) {
        const float d = div[i];
        if (d == 0.0f || d == 1.0f)
            continue;
        if (dtype == "F32") {
            float v;
            std::memcpy(&v, bytes + i * 4, 4);
            v /= d;
            std::memcpy(bytes + i * 4, &v, 4);
        } else if (dtype == "F16") {
            __half_raw raw;
            std::memcpy(&raw.x, bytes + i * 2, 2);
            const __half_raw nraw(__float2half(__half2float(__half(raw)) / d));
            std::memcpy(bytes + i * 2, &nraw.x, 2);
        } else {
            uint16_t raw;
            std::memcpy(&raw, bytes + i * 2, 2);
            raw = float_to_bf16(bf16_to_float(raw) / d);
            std::memcpy(bytes + i * 2, &raw, 2);
        }
    }
    return true;
}

}  // namespace imp

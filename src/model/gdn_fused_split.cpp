#include "model/gdn_fused_split.h"

#include "core/logging.h"
#include "model/model_config.h"

#include <cstdlib>
#include <cstring>

namespace imp {

namespace {

bool dense_2d(const Tensor& t) {
    const bool dense = t.qtype == QType::F32 || t.qtype == QType::F16 || t.qtype == QType::BF16;
    return dense && !t.scales && t.ndim == 2 && t.shape[0] > 0 && t.shape[1] > 0 && t.data;
}

// One output tensor of `rows` rows, same dtype and K as `src`; its bytes are owned by `owned`.
bool make_rows(const Tensor& src, int64_t rows, std::vector<void*>& owned, Tensor& out) {
    const size_t row_bytes = src.nbytes() / static_cast<size_t>(src.shape[0]);
    void* buf = std::malloc(row_bytes * static_cast<size_t>(rows));
    if (!buf)
        return false;
    owned.push_back(buf);
    out = src;
    out.data = buf;
    out.shape[0] = rows;
    out.compute_strides();
    return true;
}

void copy_rows(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row, int64_t n) {
    const size_t row_bytes = src.nbytes() / static_cast<size_t>(src.shape[0]);
    std::memcpy(static_cast<char*>(dst.data) + static_cast<size_t>(dst_row) * row_bytes,
                static_cast<const char*>(src.data) + static_cast<size_t>(src_row) * row_bytes,
                static_cast<size_t>(n) * row_bytes);
}

bool dims_ok(const GdnFusedDims& d) {
    return d.n_k > 0 && d.dk > 0 && d.n_v > 0 && d.dv > 0 && d.n_v % d.n_k == 0;
}

}  // namespace

GdnFusedDims gdn_fused_dims(const ModelConfig& cfg) {
    GdnFusedDims d;
    d.n_k = cfg.ssm_group_count;
    d.dk = cfg.ssm_state_size;
    d.n_v = cfg.ssm_dt_rank;
    d.dv = d.n_v > 0 ? cfg.ssm_inner_size / d.n_v : 0;
    return d;
}

bool split_gdn_qkvz(const Tensor& t, const GdnFusedDims& d, std::vector<void*>& owned, Tensor& qkv,
                    Tensor& z) {
    if (!dims_ok(d) || !dense_2d(t))
        return false;
    const int64_t r = d.n_v / d.n_k;
    const int64_t group = 2 * d.dk + 2 * r * d.dv;
    if (t.shape[0] != d.n_k * group)
        return false;
    const int64_t key_rows = d.n_k * d.dk;
    if (!make_rows(t, 2 * key_rows + d.n_v * d.dv, owned, qkv) || !make_rows(t, d.n_v * d.dv, owned, z))
        return false;
    for (int64_t g = 0; g < d.n_k; ++g) {
        const int64_t s = g * group;
        copy_rows(t, s, qkv, g * d.dk, d.dk);                                    // q
        copy_rows(t, s + d.dk, qkv, key_rows + g * d.dk, d.dk);                  // k
        copy_rows(t, s + 2 * d.dk, qkv, 2 * key_rows + g * r * d.dv, r * d.dv);  // v
        copy_rows(t, s + 2 * d.dk + r * d.dv, z, g * r * d.dv, r * d.dv);        // z
    }
    return true;
}

bool split_gdn_ba(const Tensor& t, const GdnFusedDims& d, std::vector<void*>& owned, Tensor& b, Tensor& a) {
    if (!dims_ok(d) || !dense_2d(t) || t.shape[0] != 2 * d.n_v)
        return false;
    const int64_t r = d.n_v / d.n_k;
    if (!make_rows(t, d.n_v, owned, b) || !make_rows(t, d.n_v, owned, a))
        return false;
    for (int64_t g = 0; g < d.n_k; ++g) {
        copy_rows(t, g * 2 * r, b, g * r, r);
        copy_rows(t, g * 2 * r + r, a, g * r, r);
    }
    return true;
}

bool assign_gdn_fused_input(const std::string& proj, const Tensor& t, const ModelConfig& cfg,
                            std::vector<void*>& owned, TransformerLayer& layer) {
    const GdnFusedDims d = gdn_fused_dims(cfg);
    const bool ok = proj == "in_proj_qkvz" ? split_gdn_qkvz(t, d, owned, layer.ssm_in, layer.gdn_gate)
                                           : split_gdn_ba(t, d, owned, layer.gdn_beta, layer.gdn_alpha);
    if (!ok)
        IMP_LOG_ERROR(
            "WeightMap: linear_attn.%s [%lld x %lld] %s does not split as n_k=%d dk=%d n_v=%d dv=%d "
            "(dense F32/F16/BF16 only); refusing the checkpoint",
            proj.c_str(), static_cast<long long>(t.shape[0]), static_cast<long long>(t.shape[1]),
            qtype_name(t.qtype), static_cast<int>(d.n_k), static_cast<int>(d.dk), static_cast<int>(d.n_v),
            static_cast<int>(d.dv));
    return ok;
}

}  // namespace imp

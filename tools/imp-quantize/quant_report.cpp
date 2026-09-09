#include "quant_report.h"

#include "core/fp_bits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace imp::quantize {

namespace {

// OCP E2M1: bit 3 is the sign, bits 2:0 index the magnitude.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float e2m1_to_float(unsigned char nibble) {
    const float mag = kE2M1[nibble & 0x07];
    return (nibble & 0x08) ? -mag : mag;
}

// FP8 E4M3 (finite, bias 7), decoded from the spec rather than from imp's
// device helper: a shared mistake between writer and reader cancels.
float fp8_e4m3_to_float(unsigned char bits) {
    const uint32_t sign = (bits >> 7) & 1u;
    const uint32_t exp = (bits >> 3) & 0x0Fu;
    const uint32_t man = bits & 0x07u;
    const float mag = exp == 0 ? static_cast<float>(man) / 512.0f
                               : (1.0f + static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, int(exp) - 7);
    return sign ? -mag : mag;
}

std::string fixed(double v, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
    return buf;
}

std::string sci(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3e", v);
    return buf;
}

}  // namespace

TensorError nvfp4_tensor_error(const std::string& name, const uint16_t* src_fp16, const unsigned char* packed,
                               const unsigned char* micro, float tensor_scale, int64_t N, int64_t K) {
    TensorError out;
    out.name = name;
    if (!src_fp16 || !packed || !micro || N <= 0 || K <= 0 || K % 16 != 0)
        return out;

    double sq = 0.0, max_abs_err = 0.0, absmax = 0.0;
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < K; j++) {
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(j);
            const double src = half_to_float(src_fp16[idx]);
            const unsigned char byte = packed[idx / 2];
            const unsigned char nib = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            const size_t ms_idx = static_cast<size_t>(i) * static_cast<size_t>(K / 16) +
                                  static_cast<size_t>(j / 16);
            const double dq = double(e2m1_to_float(nib)) * double(fp8_e4m3_to_float(micro[ms_idx])) *
                              double(tensor_scale);
            const double err = std::fabs(dq - src);
            sq += err * err;
            max_abs_err = std::max(max_abs_err, err);
            absmax = std::max(absmax, std::fabs(src));
        }
    }
    const double n = double(N) * double(K);
    out.mse = sq / n;
    // Normalised by the tensor's absmax, not per element: a 4-bit grid reports
    // hundreds of percent on a weight near zero, which says nothing about the
    // tensor and buries the tensors that are actually damaged.
    out.max_rel = absmax > 0.0 ? max_abs_err / absmax : 0.0;
    return out;
}

std::vector<TensorError> worst_tensors(std::vector<TensorError> rows, size_t limit) {
    std::sort(rows.begin(), rows.end(),
              [](const TensorError& a, const TensorError& b) { return a.max_rel > b.max_rel; });
    if (rows.size() > limit)
        rows.resize(limit);
    return rows;
}

std::string format_error_summary(const std::vector<TensorError>& rows, size_t limit) {
    if (rows.empty())
        return "quantization error: no tensor was measured";
    double sum_rel = 0.0, sum_mse = 0.0, worst = 0.0;
    for (const auto& r : rows) {
        sum_rel += r.max_rel;
        sum_mse += r.mse;
        worst = std::max(worst, r.max_rel);
    }
    std::string s = "quantization error over " + std::to_string(rows.size()) + " tensor(s): worst " +
                    fixed(worst, 4) + " max-rel, mean " + fixed(sum_rel / double(rows.size()), 4) +
                    ", mean MSE " + sci(sum_mse / double(rows.size()));
    for (const auto& r : worst_tensors(rows, limit))
        s += "\n      " + fixed(r.max_rel, 4) + "  MSE " + sci(r.mse) + "  " + r.name;
    return s;
}

std::string error_report_json(const std::vector<TensorError>& tensors, const std::vector<GroupError>& groups,
                              bool calibrated) {
    std::string j = "{\n";
    j += "  \"producer\": \"imp-quantize\",\n";
    j += std::string("  \"calibrated\": ") + (calibrated ? "true" : "false") + ",\n";
    j += "  \"tensors\": [";
    for (size_t i = 0; i < tensors.size(); i++) {
        j += std::string(i ? "," : "") + "\n    {\"name\": \"" + tensors[i].name +
             "\", \"max_rel\": " + fixed(tensors[i].max_rel, 6) + ", \"mse\": " + sci(tensors[i].mse) + "}";
    }
    j += tensors.empty() ? "]" : "\n  ]";
    j += ",\n  \"groups\": [";
    for (size_t i = 0; i < groups.size(); i++) {
        j += std::string(i ? "," : "") + "\n    {\"layer\": " + std::to_string(groups[i].layer) +
             ", \"kind\": \"" + groups[i].kind + "\", \"err_rtn\": " + sci(groups[i].err_rtn) +
             ", \"err_best\": " + sci(groups[i].err_best) + "}";
    }
    j += groups.empty() ? "]" : "\n  ]";
    j += "\n}\n";
    return j;
}

std::expected<void, std::string> write_error_report(const std::string& out_dir,
                                                    const std::vector<TensorError>& tensors,
                                                    const std::vector<GroupError>& groups, bool calibrated) {
    const std::string path = out_dir + "/quant_report.json";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return std::unexpected("cannot write " + path);
    const std::string body = error_report_json(tensors, groups, calibrated);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (!f)
        return std::unexpected("short write on " + path);
    return {};
}

std::string experimental_banner(bool calibrated, uint64_t samples) {
    if (!calibrated)
        return "experimental: RTN export, no calibration";
    return "experimental: AWQ calibrated on " + std::to_string(samples) +
           " samples, validated against no vendor export for this family";
}

}  // namespace imp::quantize

#include "fp8_head.h"

#include "core/config/lm_head_mode.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "model/weight_upload_traits.h"
#include "quant/fp8_quant.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace imp::quantize {

bool parse_lm_head_export(const std::string& s, LmHeadExport& out) {
    if (s == "fp8")
        out = LmHeadExport::Fp8;
    else if (s == "nvfp4")
        out = LmHeadExport::Nvfp4;
    else if (s == "source")
        out = LmHeadExport::Source;
    else
        return false;
    return true;
}

const char* lm_head_export_name(LmHeadExport h) {
    switch (h) {
        case LmHeadExport::Fp8:
            return "fp8";
        case LmHeadExport::Nvfp4:
            return "nvfp4";
        case LmHeadExport::Source:
            return "source";
    }
    return "?";
}

LmHeadExport default_lm_head_export(OutputFormat fmt) {
    return fmt == OutputFormat::Modelopt ? LmHeadExport::Fp8 : LmHeadExport::Source;
}

bool fp8_head_writable(const RawTensor& t, std::string& why_not) {
    if (t.name != "lm_head.weight") {
        why_not = "not lm_head.weight";
        return false;
    }
    if (t.shape.size() != 2 || (t.dtype != "BF16" && t.dtype != "F16" && t.dtype != "F32")) {
        why_not = "lm_head is not a 2-D BF16/F16/F32 tensor (dtype " + t.dtype + ")";
        return false;
    }
    if (t.shape[1] % 256 != 0) {
        why_not = "lm_head cols " + std::to_string(t.shape[1]) + " not a multiple of 256 (FP8 head GEMV)";
        return false;
    }
    return true;
}

std::vector<uint16_t> head_fp16_as_loaded(const RawTensor& t) {
    const size_t n = static_cast<size_t>(t.numel());
    if (t.dtype == "F16") {
        std::vector<uint16_t> out(n);
        std::memcpy(out.data(), t.data, n * sizeof(uint16_t));
        return out;
    }
    // The loader's own staging, so the bits cannot drift from what upload_weight produces.
    const int64_t shape[2] = {t.shape[0], t.shape[1]};
    const Tensor w(const_cast<void*>(t.data), t.dtype == "BF16" ? QType::BF16 : QType::F32, 2, shape, false);
    wupload::StagePlan p;
    if (!wupload::F32Fmt::stage(w, w.qtype, false, 0.0f, p))
        return {};
    return std::move(p.h16);
}

size_t fp8_head_bytes(int64_t rows, int64_t cols) {
    return (static_cast<size_t>(rows) * static_cast<size_t>(cols)) +
           (static_cast<size_t>(rows) * sizeof(float));
}

std::array<SafeTensorsOut, 2> fp8_head_outputs(int64_t rows, int64_t cols, const uint8_t* codes,
                                               const float* row_scales) {
    return {SafeTensorsOut{"lm_head.weight",
                           "F8_E4M3",
                           {rows, cols},
                           codes,
                           static_cast<size_t>(rows) * static_cast<size_t>(cols)},
            SafeTensorsOut{
                kLmHeadRowScaleTensor, "F32", {rows}, row_scales, static_cast<size_t>(rows) * sizeof(float)}};
}

std::expected<Fp8Head, std::string> quantize_head_fp8(const std::vector<uint16_t>& h_fp16, int64_t rows,
                                                      int64_t cols) {
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    void *d_in = nullptr, *d_codes = nullptr, *d_scales = nullptr;
    Fp8Head out;
    bool ok = h_fp16.size() == n && cudaMalloc(&d_in, n * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc(&d_codes, n) == cudaSuccess &&
              cudaMalloc(&d_scales, static_cast<size_t>(rows) * sizeof(float)) == cudaSuccess &&
              cudaMemcpy(d_in, h_fp16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        quantize_fp8_rows_async(d_in, d_codes, static_cast<int>(rows), static_cast<int>(cols),
                                static_cast<float*>(d_scales), nullptr);
        out.codes.resize(n);
        out.scales.resize(static_cast<size_t>(rows));
        ok = cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(out.codes.data(), d_codes, n, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.scales.data(), d_scales, out.scales.size() * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_in);
    cudaFree(d_codes);
    cudaFree(d_scales);
    if (!ok)
        return std::unexpected("FP8 head quantization failed on the device");
    return out;
}

int emit_fp8_head(const RawTensor& t, bool dry_run, std::vector<Fp8Head>& store,
                  std::vector<SafeTensorsOut>& out, size_t& bytes_out) {
    if (t.name != "lm_head.weight")
        return 0;
    std::string why_not;
    if (!fp8_head_writable(t, why_not)) {
        printf("  note: lm_head kept at source precision: %s\n", why_not.c_str());
        return 0;
    }
    const int64_t rows = t.shape[0], cols = t.shape[1];
    const size_t written = fp8_head_bytes(rows, cols);
    printf("  FP8   %-58s [%lld,%lld] per-row E4M3, %.1f MiB (source %.1f MiB)\n", t.name.c_str(),
           static_cast<long long>(rows), static_cast<long long>(cols), written / 1048576.0,
           t.nbytes / 1048576.0);
    bytes_out += written;
    if (dry_run)
        return 1;
    auto head = quantize_head_fp8(head_fp16_as_loaded(t), rows, cols);
    if (!head) {
        fprintf(stderr, "  %s: %s\n", t.name.c_str(), head.error().c_str());
        return -1;
    }
    store.push_back(std::move(*head));
    for (const auto& o : fp8_head_outputs(rows, cols, store.back().codes.data(), store.back().scales.data()))
        out.push_back(o);
    return 1;
}

}  // namespace imp::quantize

#include "quantize_weight.h"

#include "checkpoint_out.h"
#include "gptq.h"

#include "core/logging.h"
#include "core/tensor.h"
#include "core/tensor_kind.h"
#include "model/tensor_kind_matcher.h"
#include "quant/nvfp4_quant.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace imp::quantize {

std::expected<Quantized, std::string> quantize_one(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K,
                                                   float forced_scale) {
    Quantized out;
    void* d_in = nullptr;
    const size_t in_bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * 2;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess)
        return std::unexpected("cudaMalloc failed for " + std::to_string(in_bytes) + " bytes");
    if (cudaMemcpy(d_in, h_fp16.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_in));
        return std::unexpected("H2D copy failed");
    }

    int64_t shape[2] = {N, K};
    Tensor in(d_in, QType::F16, 2, shape, /*on_device=*/true);

    NvFP4QuantResult q;
    const float scale = forced_scale > 0.0f ? forced_scale
                                            : export_tensor_scale(fp16_absmax(h_fp16.data(), h_fp16.size()));
    quantize_fp16_to_nvfp4_with_scale(in, scale, q);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_in));
        free_nvfp4_result(q);
        return std::unexpected("quantization kernel failed");
    }

    out.packed.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 2));
    out.micro.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 16));
    out.tensor_scale = q.tensor_scale;
    bool ok = cudaMemcpy(out.packed.data(), q.packed_data, out.packed.size(), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              cudaMemcpy(out.micro.data(), q.micro_scales, out.micro.size(), cudaMemcpyDeviceToHost) ==
                  cudaSuccess;
    IMP_CUDA_CHECK_LOG(cudaFree(d_in));
    free_nvfp4_result(q);
    if (!ok)
        return std::unexpected("D2H copy failed");
    return out;
}

std::string WeightQuantizer::open_gptq(const std::string& hess_path, float damp) {
    path_ = hess_path;
    damp_ = damp;
    if (std::string err = read_hessian_index(path_, index_); !err.empty())
        return err;
    printf("GPTQ: %zu Hessians from %s\n", index_.size(), path_.c_str());
    return "";
}

std::expected<Quantized, std::string> WeightQuantizer::quantize(
    const std::string& name, const std::vector<uint16_t>& h, int64_t N, int64_t K, float forced_scale,
    const std::map<std::string, std::vector<float>>& col_scale) {
    const HessianIndexEntry* he = index_.empty() ? nullptr
                                                 : find_hessian(index_, gptq::layer_of(name),
                                                                tensor_kind_name(match_tensor_kind(name)));
    if (!he || he->K != K)
        return quantize_one(h, N, K, forced_scale);
    std::vector<float> H;
    if (std::string err = read_hessian(path_, *he, H); !err.empty())
        return std::unexpected(err);
    if (const auto cs = col_scale.find(name);
        cs != col_scale.end() && cs->second.size() == static_cast<size_t>(K))
        for (int64_t i = 0; i < K; i++)
            for (int64_t j = 0; j < K; j++)
                H[static_cast<size_t>(i) * K + j] /= cs->second[i] * cs->second[j];
    const float scale = forced_scale > 0.0f ? forced_scale
                                            : export_tensor_scale(fp16_absmax(h.data(), h.size()));
    auto g = gptq::quantize_nvfp4(h.data(), N, K, std::move(H), scale, damp_);
    if (!g)
        return std::unexpected(g.error());
    loss_rtn_ += g->loss_rtn;
    loss_gptq_ += g->loss_gptq;
    n_++;
    n_worse_ += g->loss_gptq > g->loss_rtn ? 1 : 0;
    return Quantized{std::move(g->packed), std::move(g->micro), scale};
}

std::string WeightQuantizer::summary() const {
    if (n_ == 0)
        return "";
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "GPTQ: %zu tensors (damp %g), output loss sum(dW H dW^T) %.6g vs round-to-nearest %.6g "
                  "(%.1f%%), %zu tensors worse",
                  n_, static_cast<double>(damp_), loss_gptq_, loss_rtn_,
                  loss_rtn_ > 0.0 ? 100.0 * loss_gptq_ / loss_rtn_ : 0.0, n_worse_);
    return buf;
}

}  // namespace imp::quantize

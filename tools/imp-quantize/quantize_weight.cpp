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

std::string WeightQuantizer::open_gptq(const std::string& hess_paths, float damp) {
    damp_ = damp;
    if (std::string err = read_hessian_set(hess_paths, files_, index_); !err.empty())
        return err;
    printf("GPTQ: %zu Hessians from %zu file(s)\n", index_.size(), files_.size());
    return "";
}

std::string WeightQuantizer::open(const Options& opt) {
    if (!opt.gptq_file.empty() && !opt.gptaq_file.empty())
        return "--gptq and --gptaq exclude each other";
    if (!opt.gptq_file.empty())
        return open_gptq(opt.gptq_file, opt.gptq_damp);
    return opt.gptaq_file.empty() ? "" : open_gptaq(opt.gptaq_file, opt.gptq_damp, opt.gptaq_alpha);
}

std::string WeightQuantizer::open_gptaq(const std::string& path, float damp, float alpha) {
    if (std::string err = open_gptq(path, damp); !err.empty())
        return err;
    if (!index_.empty() && index_.front().mats != 2)
        return path + ": IMPHES01 holds no cross term; GPTAQ needs IMPHES02 (calibration.pair_read)";
    gptaq_ = true;
    alpha_ = alpha;
    printf("GPTAQ: alpha %g\n", static_cast<double>(alpha));
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
    if (std::string err = read_hessian(files_[he->file], *he, H); !err.empty())
        return std::unexpected(err);
    const auto cs = col_scale.find(name);
    const std::vector<float>* s = cs != col_scale.end() && cs->second.size() == static_cast<size_t>(K)
                                      ? &cs->second
                                      : nullptr;
    std::vector<float> dxx;
    if (gptaq_ && alpha_ != 0.0f) {
        // Row-major C[i*K+j] = E[x_i xq_j] -> column-major dXX(i, j) = E[(x_i / s_i - xq_i) xq_j].
        std::vector<float> C;
        if (std::string err = read_hessian(files_[he->file], *he, C, 1); !err.empty())
            return std::unexpected(err);
        dxx.resize(C.size());
        for (int64_t i = 0; i < K; i++)
            for (int64_t j = 0; j < K; j++) {
                const size_t ij = (static_cast<size_t>(i) * K) + j;
                dxx[(static_cast<size_t>(j) * K) + i] = (C[ij] / (s ? (*s)[i] : 1.0f)) - H[ij];
            }
    } else if (s && !gptaq_) {
        for (int64_t i = 0; i < K; i++)
            for (int64_t j = 0; j < K; j++)
                H[(static_cast<size_t>(i) * K) + j] /= (*s)[i] * (*s)[j];
    }
    const float scale = forced_scale > 0.0f ? forced_scale
                                            : export_tensor_scale(fp16_absmax(h.data(), h.size()));
    auto g = gptq::quantize_nvfp4(h.data(), N, K, std::move(H), scale, damp_, std::move(dxx), alpha_);
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
                  "%s: %zu tensors (damp %g), output loss sum(dW H dW^T) %.6g vs round-to-nearest %.6g "
                  "(%.1f%%), %zu tensors worse",
                  gptaq_ ? "GPTAQ" : "GPTQ", n_, static_cast<double>(damp_), loss_gptq_, loss_rtn_,
                  loss_rtn_ > 0.0 ? 100.0 * loss_gptq_ / loss_rtn_ : 0.0, n_worse_);
    return buf;
}

}  // namespace imp::quantize

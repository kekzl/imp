#include "fp8_keep_run.h"

#include "checkpoint_out.h"
#include "fp8_head.h"
#include "fp8_keep.h"
#include "gptq.h"
#include "quantize_weight.h"
#include "tensor_policy.h"

#include "core/fp_bits.h"
#include "core/logging.h"
#include "core/tensor_kind.h"
#include "model/tensor_kind_matcher.h"
#include "quant/calibration_hessian.h"
#include "quant/fp8_quant.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace imp::quantize {

namespace {

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// src - decode(NVFP4 codes), row-major.
std::vector<float> nvfp4_error(const std::vector<uint16_t>& h, const Quantized& q, int64_t N, int64_t K) {
    std::vector<float> e(h.size());
    for (int64_t i = 0; i < N; i++)
        for (int64_t j = 0; j < K; j++) {
            const size_t idx = (static_cast<size_t>(i) * K) + j;
            const unsigned char byte = q.packed[idx / 2];
            const unsigned char nib = (j % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            const float mag = kE2M1[nib & 0x07] *
                              e4m3_to_float(q.micro[(static_cast<size_t>(i) * (K / 16)) + (j / 16)]) *
                              q.tensor_scale;
            e[idx] = half_to_float(h[idx]) - ((nib & 0x08) ? -mag : mag);
        }
    return e;
}

std::vector<float> fp8_error(const std::vector<uint16_t>& h, const Fp8Weight& f) {
    std::vector<float> e(h.size());
    for (size_t i = 0; i < h.size(); i++)
        e[i] = half_to_float(h[i]) - (e4m3_to_float(f.codes[i]) * f.scale);
    return e;
}

bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

}  // namespace

std::expected<Fp8Weight, std::string> quantize_fp8_tensor(const std::vector<uint16_t>& h_fp16, int64_t N,
                                                          int64_t K) {
    const size_t n = static_cast<size_t>(N) * static_cast<size_t>(K);
    if (h_fp16.size() != n || n > static_cast<size_t>(INT_MAX))
        return std::unexpected("FP8 weight of " + std::to_string(n) +
                               " elements: size mismatch or above INT_MAX");
    Fp8Weight out;
    const float amax = fp16_absmax(h_fp16.data(), n);
    out.scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    void *d_in = nullptr, *d_out = nullptr;
    bool ok = cudaMalloc(&d_in, n * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc(&d_out, n) == cudaSuccess &&
              cudaMemcpy(d_in, h_fp16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        quantize_fp16_to_fp8_e4m3_scaled(d_in, d_out, static_cast<int>(n), out.scale);
        out.codes.resize(n);
        ok = cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(out.codes.data(), d_out, n, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    IMP_CUDA_CHECK_LOG(cudaFree(d_in));
    IMP_CUDA_CHECK_LOG(cudaFree(d_out));
    if (!ok)
        return std::unexpected("FP8 quantization failed on the device");
    return out;
}

std::string Fp8Keep::score_units(const Options& opt,
                                 const std::map<std::string, std::vector<const RawTensor*>>& members,
                                 const ForcedOf& forced_of,
                                 const std::map<std::string, std::vector<float>>& col_scale,
                                 std::map<std::string, double>& scores) {
    char magic[8] = {};
    std::ifstream probe(opt.fp8_score, std::ios::binary);
    if (!probe)
        return "--fp8-score " + opt.fp8_score + ": cannot open";
    probe.read(magic, sizeof(magic));
    if (std::memcmp(magic, "IMPHES01", 8) != 0) {
        std::string text, err;
        if (!read_text(opt.fp8_score, text) || !parse_fp8_rank_text(text, scores, err))
            return "--fp8-score " + opt.fp8_score + ": " + err;
        printf("FP8 keep: %zu unit scores from %s\n", scores.size(), opt.fp8_score.c_str());
        return "";
    }
    if (opt.dry_run)
        return "--fp8-score with a Hessian file needs the weights: not with --dry-run";
    std::vector<std::string> files;
    std::vector<HessianIndexEntry> index;
    if (std::string err = read_hessian_set(opt.fp8_score, files, index); !err.empty())
        return err;
    // Proxy: per tensor tr(dW H dW^T) of the NVFP4 codes the writer would emit minus that of FP8,
    // in the AWQ-scaled space (H / (s_i s_j)), summed over the unit.
    for (const auto& [unit, ts] : members) {
        double gain = 0.0;
        for (const RawTensor* t : ts) {
            const int64_t N = t->shape[0], K = t->shape[1];
            const HessianIndexEntry* he = find_hessian(index, gptq::layer_of(t->name),
                                                       tensor_kind_name(match_tensor_kind(t->name)));
            if (!he || he->K != K)
                return t->name + ": no Hessian of width " + std::to_string(K) + " in " + opt.fp8_score;
            std::vector<float> H;
            if (std::string err = read_hessian(files[he->file], *he, H); !err.empty())
                return err;
            if (const auto cs = col_scale.find(t->name);
                cs != col_scale.end() && cs->second.size() == size_t(K))
                for (int64_t i = 0; i < K; i++)
                    for (int64_t j = 0; j < K; j++)
                        H[(static_cast<size_t>(i) * K) + j] /= cs->second[i] * cs->second[j];
            const auto h = fp16_of_(*t);
            if (!h)
                return t->name + ": " + h.error();
            const auto q = quantize_one(*h, N, K, forced_of(t->name));
            const auto f = quantize_fp8_tensor(*h, N, K);
            if (!q || !f)
                return t->name + ": " + (q ? f.error() : q.error());
            const auto l_nv = gptq::output_loss(nvfp4_error(*h, *q, N, K), H, N, K);
            const auto l_f8 = gptq::output_loss(fp8_error(*h, *f), H, N, K);
            if (!l_nv || !l_f8)
                return t->name + ": " + (l_nv ? l_f8.error() : l_nv.error());
            printf("  fp8-score %-48s nvfp4 %.6g fp8 %.6g\n", t->name.c_str(), *l_nv, *l_f8);
            gain += *l_nv - *l_f8;
        }
        scores[unit] = gain;
    }
    return "";
}

std::string Fp8Keep::plan(const Options& opt, const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                          const Pred& nvfp4_target, Fp16Of fp16_of, const ForcedOf& forced_of,
                          const std::map<std::string, std::vector<float>>& col_scale) {
    if (opt.fp8_keep.empty() && opt.fp8_budget < 0.0)
        return opt.fp8_score.empty() ? "" : "--fp8-score needs --fp8-budget";
    if (opt.format != OutputFormat::Modelopt)
        return "--fp8-keep / --fp8-budget declare Modelopt MIXED_PRECISION: not with --format vllm";
    if (opt.fp8_budget >= 0.0 && opt.fp8_score.empty())
        return "--fp8-budget needs --fp8-score (F.hess or a rank file)";
    fp16_of_ = std::move(fp16_of);
    col_scale_ = &col_scale;
    if (!opt.dry_run && !opt.calib_file.empty())
        if (std::string err = read_calibration_stats(opt.calib_file, calib_); !err.empty())
            return err;

    // Units and the pure NVFP4 export size the budget is a percent of.
    std::map<std::string, std::vector<const RawTensor*>> members;
    std::map<std::string, size_t> extra;
    size_t pure = 0;
    for (const auto& src : opened)
        for (const auto& t : src->tensors()) {
            std::string why;
            const bool nv = nvfp4_target(t);
            if (t.name == "lm_head.weight" && opt.lm_head == LmHeadExport::Fp8 && fp8_head_writable(t, why))
                pure += fp8_head_bytes(t.shape[0], t.shape[1]);
            else
                pure += nv ? nvfp4_output_bytes(t.shape[0], t.shape[1]) : t.nbytes;
            const std::string u = nv ? fp8_unit_of(t.name) : "";
            if (u.empty())
                continue;
            members[u].push_back(&t);
            extra[u] += fp8_extra_bytes(t.shape[0], t.shape[1]);
        }
    if (members.empty())
        return "--fp8-keep / --fp8-budget: no dense q/k/v/o/gate/up/down projection would be NVFP4 here";

    if (!opt.fp8_keep.empty()) {
        std::vector<std::string> list;
        std::string err;
        if (!parse_fp8_keep_list(opt.fp8_keep, list, err))
            return "--fp8-keep " + err;
        for (const auto& [u, ts] : members)
            if (fp8_keep_list_matches(list, u))
                units_.insert(u);
    }
    if (opt.fp8_budget >= 0.0) {
        std::map<std::string, double> scores;
        if (std::string err = score_units(opt, members, forced_of, col_scale, scores); !err.empty())
            return err;
        std::vector<Fp8Unit> ranked;
        for (const auto& [u, ts] : members) {
            const auto s = scores.find(u);
            ranked.push_back({u, s == scores.end() ? 0.0 : s->second, extra[u]});
        }
        std::ranges::stable_sort(ranked, [](const Fp8Unit& a, const Fp8Unit& b) {
            return a.score / double(a.extra_bytes) > b.score / double(b.extra_bytes);
        });
        for (size_t i = 0; i < ranked.size(); i++)
            printf("  fp8-rank %3zu %-12s score %.6g extra %.2f MiB\n", i + 1, ranked[i].unit.c_str(),
                   ranked[i].score, ranked[i].extra_bytes / 1048576.0);
        const auto budget = static_cast<size_t>(static_cast<double>(pure) * opt.fp8_budget / 100.0);
        for (const auto& u : select_fp8_units(ranked, budget))
            units_.insert(u);
    }
    size_t added = 0, n_tensors = 0;
    std::string list;
    for (const auto& u : units_) {
        added += extra[u];
        n_tensors += members[u].size();
        list += (list.empty() ? "" : ",") + u;
    }
    printf(
        "FP8 keep: %zu unit(s), %zu tensor(s), +%.2f MiB = +%.2f %% over the pure NVFP4 export (%.2f MiB)\n"
        "  units: %s\n",
        units_.size(), n_tensors, added / 1048576.0, pure ? 100.0 * double(added) / double(pure) : 0.0,
        pure / 1048576.0, list.empty() ? "(none)" : list.c_str());
    return "";
}

bool Fp8Keep::selected(const std::string& weight_name) const {
    return !units_.empty() && units_.contains(fp8_unit_of(weight_name));
}

bool Fp8Keep::emit(const RawTensor& t, bool dry_run, std::vector<SafeTensorsOut>& out, size_t& bytes_out) {
    const int64_t N = t.shape[0], K = t.shape[1];
    bytes_out += fp8_output_bytes(N, K);
    fp8_modules_.push_back(t.name.substr(0, t.name.size() - std::string(".weight").size()));
    if (dry_run) {
        printf("  FP8   %-58s [%lld,%lld] per-tensor E4M3 (--fp8-keep)\n", t.name.c_str(),
               static_cast<long long>(N), static_cast<long long>(K));
        return true;
    }
    const auto h = fp16_of_(t);
    if (!h) {
        fprintf(stderr, "  %s: %s\n", t.name.c_str(), h.error().c_str());
        return false;
    }
    auto f = quantize_fp8_tensor(*h, N, K);
    if (!f) {
        fprintf(stderr, "  %s: %s\n", t.name.c_str(), f.error().c_str());
        return false;
    }
    f->input_scale = static_input_scale(t.name, K);
    if (f->input_scale <= 0.0f && n_dynamic_++ == 0)
        printf(
            "  note: %s has no max|x| in --calib (IMPCAL03): written without input_scale, the runtime\n"
            "        scales each FP8 call by its batch absmax, so rows depend on their batch neighbours\n",
            t.name.c_str());
    store_.push_back(std::move(*f));
    const Fp8Weight& w = store_.back();
    for (const auto& o : fp8_weight_outputs(t.name, N, K, w.codes.data(), &w.scale,
                                            w.input_scale > 0.0f ? &w.input_scale : nullptr))
        out.push_back(o);
    return true;
}

float Fp8Keep::static_input_scale(const std::string& weight_name, int64_t K) const {
    // The calibrator keys the shared input under each consumer's kind; max over the unit's kinds.
    const std::string unit = fp8_unit_of(weight_name);
    const std::string group = unit.substr(unit.find('.') + 1);
    const std::vector<const char*> kinds = group == "qkv"       ? std::vector<const char*>{"WQ", "WK", "WV"}
                                           : group == "gate_up" ? std::vector<const char*>{"W_GATE", "W_UP"}
                                           : group == "o"       ? std::vector<const char*>{"WO"}
                                                                : std::vector<const char*>{"W_DOWN"};
    std::vector<float> amax;
    for (const char* k : kinds) {
        const CalibrationEntry* e = calib_.find(gptq::layer_of(weight_name), k);
        if (!e || e->max_abs.size() != static_cast<size_t>(K))
            continue;
        if (amax.empty())
            amax = e->max_abs;
        else
            for (size_t c = 0; c < amax.size(); c++)
                amax[c] = std::max(amax[c], e->max_abs[c]);
    }
    static const std::vector<float> kNone;
    const std::vector<float>* cs = &kNone;
    if (col_scale_)
        if (const auto it = col_scale_->find(weight_name); it != col_scale_->end())
            cs = &it->second;
    return fp8_input_scale(amax, *cs, K);
}

std::string Fp8Keep::quantized_layers_json(const std::string& indent) const {
    return fp8_modules_.empty() ? "" : mixed_precision_layers_json(nvfp4_modules_, fp8_modules_, indent);
}

}  // namespace imp::quantize

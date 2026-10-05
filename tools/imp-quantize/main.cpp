// imp-quantize: turn a BF16/FP16 SafeTensors checkpoint into NVFP4. STATUS: EXPERIMENTAL:
// verified end to end, --calib recovers some loss, but still below a published Modelopt
// export; for NVFP4 coverage/eval/perf work, not checkpoints to rely on.
// Two output layouts (--format): modelopt (default: weight/weight_scale/weight_scale_2 +
// hf_quant_config.json) or compressed-tensors (--format vllm: weight_packed/weight_scale/
// weight_global_scale=1/scale + quantization_config in config.json). Both read by imp; only
// the second by vLLM.
// Scope: dense models and MoE with per-expert 2-D tensors; 3-D expert stacks (gpt-oss, Gemma-4)
// are split into per-expert matrices by a per-model layout descriptor (expert_destack.h) and
// refused for a model_type without one. MLA latent projections and the MoE router are
// excluded from quantization even though 2-D/K-aligned (should_quantize).
// Quality (ppl_corpus_45k.txt, calibrated on separate prose): Qwen3-0.6B BF16 24.08 -> RTN
// 29.42 -> AWQ 27.60; Qwen3-1.7B BF16 17.22 -> RTN 20.39 -> AWQ 18.71. A short (199-token)
// corpus gives unrelated numbers; always judge with a corpus this size.
// Coherence: degen_suite.py reads 45/45 on these checkpoints; --calibrate forces
// runtime.deterministic_gemm since a non-deterministic calib file flips probes randomly.
// A published export isn't automatically better: on Qwen3-14B (same weights/corpus) this tool
// without --calib read PPL 9.9252 vs a Modelopt export's 10.0301 (one model, not a general
// claim; docs/quantization.md).

#include "common/exit_codes.h"
#include "awq.h"
#include "checkpoint_out.h"
#include "expert_destack.h"
#include "fp8_head.h"
#include "options.h"
#include "fp8_source.h"
#include "quant_report.h"
#include "recipe.h"
#include "tensor_policy.h"
#include "usage.h"

#include "core/tensor.h"
#include "memory/plan.h"
#include "model/mxfp8_widen.h"
#include "model/safetensors_raw.h"
#include "model/safetensors_writer.h"
#include "quant/awq_transform.h"
#include "quant/calibration_stats.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace imp;
using imp::quantize::Options;

namespace {

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Look up a plan vector by tensor name, or an empty one when the plan has
// nothing to say about it. awq_apply_* treat empty (and wrong-length) vectors
// as no-ops, so an untouched tensor takes the same path as a transformed one.
const std::vector<float>& plan_vec(const std::map<std::string, std::vector<float>>& m,
                                   const std::string& name) {
    static const std::vector<float> kNone;
    auto it = m.find(name);
    return (it == m.end()) ? kNone : it->second;
}

// Fresh copy of a 1-D producer with 1/s folded in, in the tensor's ORIGINAL dtype: the loader
// reads by dtype, so widening would be a format change, and would be WRONG on a unit-offset
// norm (weight_upload.cpp adds +1 on BF16-source paths only). `offset` comes from the plan, not
// a guess: folding a norm as plain produces a checkpoint that loads and is a different model.
std::vector<unsigned char> folded_copy(const RawTensor& t, const std::vector<float>& div, NormOffset offset,
                                       bool& ok) {
    std::vector<unsigned char> out(t.nbytes);
    std::memcpy(out.data(), t.data, t.nbytes);
    ok = awq_apply_vector_div(out.data(), static_cast<size_t>(t.numel()), t.dtype, div, offset);
    return out;
}

// Quantize one host FP16 matrix, returning host-side packed data + scales.
struct Quantized {
    std::vector<unsigned char> packed;  // [N, K/2]
    std::vector<unsigned char> micro;   // [N, K/16] FP8 E4M3
    float tensor_scale = 1.0f;
};

// `forced_scale` > 0 quantizes against a scale decided elsewhere — the shared
// scale of a fused group. At 0 the tensor's own absmax decides.
std::expected<Quantized, std::string> quantize_one(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K,
                                                   float forced_scale) {
    Quantized out;
    void* d_in = nullptr;
    const size_t in_bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * 2;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess)
        return std::unexpected("cudaMalloc failed for " + std::to_string(in_bytes) + " bytes");
    if (cudaMemcpy(d_in, h_fp16.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_in);
        return std::unexpected("H2D copy failed");
    }

    int64_t shape[2] = {N, K};
    Tensor in(d_in, QType::F16, 2, shape, /*on_device=*/true);

    NvFP4QuantResult q;
    const float scale = forced_scale > 0.0f ? forced_scale
                                            : quantize::export_tensor_scale(
                                                  quantize::fp16_absmax(h_fp16.data(), h_fp16.size()));
    quantize_fp16_to_nvfp4_with_scale(in, scale, q);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_in);
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
    cudaFree(d_in);
    free_nvfp4_result(q);
    if (!ok)
        return std::unexpected("D2H copy failed");
    return out;
}

// The FP16 form a tensor is quantized from: an FP8 source widened against its block-scale
// grid, otherwise raw BF16/F16 bits, then the AWQ transform. One function for two callers (scale
// planner, writer) so they never measure/quantize a different tensor from each other.
std::expected<std::vector<uint16_t>, std::string> tensor_as_fp16(
    const RawTensor& t, const std::map<std::string, const RawTensor*>& fp8_scale_of, const awq::Plan& plan) {
    std::vector<uint16_t> out;
    if (const auto it = fp8_scale_of.find(t.name); it != fp8_scale_of.end()) {
        auto widened = quantize::fp8_scaled_to_fp16(t, *it->second);
        if (!widened)
            return std::unexpected(widened.error());
        out = std::move(*widened);
    } else {
        out = awq::raw_to_fp16(t);
    }
    awq_apply_matrix(out, t.shape[0], t.shape[1], plan_vec(plan.row_div, t.name),
                     plan_vec(plan.col_scale, t.name));
    return out;
}

// 3-D expert stacks (gpt-oss, Gemma-4) are split into the per-expert 2-D matrices the loader
// reads, by the model's layout descriptor (expert_destack.h). Without a descriptor the checkpoint
// is refused before anything is written: the experts are the bulk of the bytes, a guessed layout
// quantizes a transposed or mis-paired matrix that still loads, and copying them through would
// label a mostly-BF16 checkpoint NVFP4 (#1188). false = refused, already printed.
bool resolve_expert_stacks(const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                           const std::string& in_dir, std::set<std::string>& stacked_names,
                           quantize::StackedExpertLayout& stack_layout) {
    std::vector<const RawTensor*> stacked;
    size_t stacked_bytes = 0, total_bytes = 0;
    for (const auto& src : opened) {
        for (const auto& t : src->tensors())
            total_bytes += t.nbytes;
        auto found = quantize::find_stacked_expert_tensors(src->tensors());
        for (const RawTensor* t : found)
            stacked_bytes += t->nbytes;
        stacked.insert(stacked.end(), found.begin(), found.end());
    }
    if (stacked.empty())
        return true;
    const std::string model_type = quantize::model_type_from_config(
        (fs::path(in_dir) / "config.json").string());
    const auto layout = quantize::stacked_expert_layout(model_type);
    const double share = total_bytes ? 100.0 * double(stacked_bytes) / double(total_bytes) : 0.0;
    if (!layout) {
        fprintf(stderr,
                "refusing: %zu tensor(s) store MoE experts as a 3-D stack (%.1f%% of this\n"
                "checkpoint) and this tool has no stack layout for model_type '%s'\n"
                "(known: gpt_oss, gemma4). A guessed layout quantizes a transposed or\n"
                "mis-paired matrix that loads; add the descriptor to expert_destack.cpp.\n",
                stacked.size(), share, model_type.c_str());
        for (size_t i = 0; i < stacked.size() && i < 3; ++i)
            fprintf(stderr, "  %s\n", stacked[i]->name.c_str());
        if (stacked.size() > 3)
            fprintf(stderr, "  ... and %zu more\n", stacked.size() - 3);
        return false;
    }
    stack_layout = *layout;
    for (const RawTensor* t : stacked)
        stacked_names.insert(t->name);
    printf(
        "  SPLITTING %zu expert stack(s), %.1f%% of the checkpoint, into per-expert matrices "
        "(model_type %s: %s, gate/up %s)\n",
        stacked.size(), share, model_type.c_str(), layout->transposed ? "[ne,K,N]" : "[ne,N,K]",
        layout->gate_up == quantize::GateUpOrder::Interleaved ? "interleaved" : "concatenated");
    return true;
}

// Plans every stack in one shard: each becomes ne x (2|1) matrices, and the shard's stores hand
// out pointers into themselves, so their reserve must count them (`n_destacked`).
bool plan_shard_stacks(const RawSafeTensors& src, const std::set<std::string>& stacked_names,
                       const quantize::StackedExpertLayout& layout,
                       std::map<std::string, std::vector<quantize::DestackedMatrix>>& plans,
                       size_t& n_destacked) {
    for (const auto& t : src.tensors()) {
        if (!stacked_names.count(t.name))
            continue;
        auto planned = quantize::destack_plan(t, layout);
        if (!planned) {
            fprintf(stderr, "  %s: %s\n", t.name.c_str(), planned.error().c_str());
            return false;
        }
        n_destacked += planned->size();
        plans[t.name] = std::move(*planned);
    }
    return true;
}

// The per-shard sinks one quantized matrix lands in, bundled so the expert-stack step appends
// exactly as main()'s own loop does.
struct ShardSinks {
    std::vector<Quantized>& quant_store;
    std::vector<float>& scale_store;
    std::vector<SafeTensorsOut>& out;
    std::vector<quantize::TensorError>& tensor_errors;
    size_t& bytes_out;
    size_t& n_quantized;
};
using EmitFn = std::function<void(std::vector<SafeTensorsOut>&, std::vector<float>&, const std::string&,
                                  int64_t, int64_t, const Quantized&)>;

// One expert stack: its per-expert matrices quantized and emitted, the stack itself never
// written. Gate and up of one expert share a tensor scale, the fused-layer rule of
// checkpoint_out.h (the loader merges them as vLLM merges w13). Dry run: forecast only.
bool quantize_expert_stack(const RawTensor& t, const std::vector<quantize::DestackedMatrix>& ms,
                           const quantize::StackedExpertLayout& layout, const awq::Plan& plan, bool dry_run,
                           ShardSinks& s, const EmitFn& emit) {
    if (dry_run) {
        printf("  QUANT %-58s %zu per-expert matrices [%lld,%lld]\n", t.name.c_str(), ms.size(),
               (long long)ms[0].N, (long long)ms[0].K);
        for (const auto& m : ms)
            s.bytes_out += quantize::nvfp4_output_bytes(m.N, m.K);
        s.n_quantized += ms.size();
        return true;
    }
    auto one = [&](const quantize::DestackedMatrix& m, const std::vector<uint16_t>& h, float forced) -> bool {
        auto quantized = quantize_one(h, m.N, m.K, forced);
        if (!quantized) {
            fprintf(stderr, "  %s: %s\n", m.name.c_str(), quantized.error().c_str());
            return false;
        }
        s.quant_store.push_back(std::move(*quantized));
        const Quantized& q = s.quant_store.back();
        emit(s.out, s.scale_store, m.name, m.N, m.K, q);
        s.tensor_errors.push_back(quantize::nvfp4_tensor_error(m.name, h.data(), q.packed.data(),
                                                               q.micro.data(), q.tensor_scale, m.N, m.K));
        s.bytes_out += q.packed.size() + q.micro.size() + sizeof(float);
        s.n_quantized++;
        return true;
    };
    for (size_t i = 0; i < ms.size(); ++i) {
        std::vector<uint16_t> h = quantize::destack_read(t, layout, ms[i]);
        awq_apply_matrix(h, ms[i].N, ms[i].K, plan_vec(plan.row_div, ms[i].name),
                         plan_vec(plan.col_scale, ms[i].name));
        if (ms[i].part == quantize::ExpertPart::Down) {
            if (!one(ms[i], h, 0.0f))
                return false;
            continue;
        }
        // gate at i, its up at i + 1 (destack_plan orders them so)
        std::vector<uint16_t> up = quantize::destack_read(t, layout, ms[i + 1]);
        awq_apply_matrix(up, ms[i + 1].N, ms[i + 1].K, plan_vec(plan.row_div, ms[i + 1].name),
                         plan_vec(plan.col_scale, ms[i + 1].name));
        const float shared = quantize::export_tensor_scale(
            std::max(quantize::fp16_absmax(h.data(), h.size()), quantize::fp16_absmax(up.data(), up.size())));
        if (!one(ms[i], h, shared) || !one(ms[i + 1], up, shared))
            return false;
        ++i;
    }
    return true;
}

// What is gone from the card before imp allocates a single weight: kMeasuredLibraryReserveBytes
// (src/memory/plan.h, re-measure after a CUDA/driver bump) and the CUDA primary context size
// for this WSL2/WDDM box. Both are measurements, not headroom guesses.
constexpr size_t kContextBytes = 1680ull * 1024 * 1024;

// Where the bytes that did NOT shrink went, largest first: the compression ratio answers
// "did it work", not "why is it still this big". Every line is a role deliberately left at
// source precision, so the table doubles as what could be traded (e.g. the embedding pair is a
// quarter of the output on a modern vocabulary, which the ratio alone never reveals).
void report_copied_breakdown(const std::map<std::string, size_t>& by_reason, size_t bytes_out) {
    if (by_reason.empty() || bytes_out == 0)
        return;
    std::vector<std::pair<std::string, size_t>> rows(by_reason.begin(), by_reason.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    size_t total = 0;
    for (const auto& [_, b] : rows)
        total += b;
    printf("\nkept at source precision: %.2f GiB, %.0f%% of the output", total / 1073741824.0,
           100.0 * double(total) / double(bytes_out));
    const double mib = 1024.0 * 1024.0;
    for (size_t i = 0; i < rows.size() && i < 5; i++)
        printf("\n      %8.0f MiB  %s", double(rows[i].second) / mib, rows[i].first.c_str());
    if (rows.size() > 5)
        printf("\n      %8s   (%zu smaller reasons)", "", rows.size() - 5);
}

// --gdn-proj-mxfp8 (#2475): a kept GDN / Mamba projection whose K splits into 32-blocks.
bool mxfp8_target(const Options& opt, const RawTensor& t) {
    return opt.gdn_proj_mxfp8 && quantize::keep_gdn_projection(t.name, opt.keep_gdn_proj) &&
           t.shape.size() == 2 && t.shape[1] % imp::kMxfp8Block == 0;
}

size_t mxfp8_bytes(const RawTensor& t) {
    return static_cast<size_t>(t.numel() + t.numel() / imp::kMxfp8Block);
}

// E4M3 `<m>.weight` + E8M0 `<m>.weight_scale` from FP16 `h`; `store` owns the bytes.
void push_mxfp8(const RawTensor& t, const std::vector<uint16_t>& h, bool dry_run,
                std::vector<SafeTensorsOut>& out, std::vector<std::vector<uint8_t>>& store) {
    const int64_t N = t.shape[0], K = t.shape[1];
    if (dry_run) {
        printf("  MXFP8 %-58s [%lld,%lld]\n", t.name.c_str(), (long long)N, (long long)K);
        return;
    }
    store.emplace_back(static_cast<size_t>(N * K));
    store.emplace_back(static_cast<size_t>(N * (K / imp::kMxfp8Block)));
    std::vector<uint8_t>& q = store[store.size() - 2];
    std::vector<uint8_t>& s = store.back();
    imp::mxfp8_quantize(h.data(), N, K, q.data(), s.data());
    out.push_back({t.name, "F8_E4M3", {N, K}, q.data(), q.size()});
    out.push_back({t.name + "_scale", "U8", {N, K / imp::kMxfp8Block}, s.data(), s.size()});
}

// Reports the checkpoint against the target card by ON-DISK size, not a VRAM prediction: what
// the engine resides is weights PLUS a scale-factor cache MINUS what the loader skips (vision
// tower uploaded separately, MTP sidecar maybe unloaded). Measured on Qwen3.8-27B: 18.60 GiB
// disk arrived as 16.08 GiB weights + 1.49 GiB CUTLASS scale cache - right order, wrong decimal.
// mx_widen: bytes the loader adds when it widens MXFP8 projections to BF16 (#2475).
void report_card_fit(size_t bytes_out, size_t mx_widen) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0)
        return;  // no device visible: the size line above still stands on its own
    const double mib = 1024.0 * 1024.0;
    const size_t overhead = kContextBytes + kMeasuredLibraryReserveBytes;
    if (total_b <= overhead)
        return;
    const double budget = double(total_b - overhead) / mib;
    const double ckpt = double(bytes_out + mx_widen) / mib;
    const char* where = mx_widen > 0 ? "once widened" : "on disk";
    if (mx_widen > 0)
        printf("\nMXFP8 projections are widened to BF16 at load: +%.0f MiB counted below",
               double(mx_widen) / mib);
    printf("\ncard: %.0f MiB total, less %.0f context and %.0f library reserve leaves %.0f MiB",
           double(total_b) / mib, kContextBytes / mib, kMeasuredLibraryReserveBytes / mib, budget);
    if (ckpt >= budget)
        printf(
            "\n      this checkpoint is %.0f MiB %s and does NOT fit: %.0f MiB short before"
            "\n      any KV cache or workspace",
            ckpt, where, ckpt - budget);
    else
        printf(
            "\n      this checkpoint is %.0f MiB %s, leaving %.0f MiB for the KV cache and"
            "\n      workspaces. On-disk size is not the resident figure: a scale-factor cache is"
            "\n      built on top, and a vision tower or MTP sidecar may load separately or not at all",
            ckpt, where, budget - ckpt);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (const auto exit_code = quantize::parse_args(argc, argv, opt))
        return *exit_code;
    // States what this export IS before it starts: EXPERIMENTAL, previously said only in a
    // source comment and a doc heading where no operator would see it. The calibrated arm prints
    // the same line plus its sample count once the calibration file is read.
    if (opt.calib_file.empty())
        printf("%s\n", quantize::experimental_banner(/*calibrated=*/false, 0).c_str());

    std::vector<fs::path> shards;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(opt.in_dir, ec))
        if (e.path().extension() == ".safetensors")
            shards.push_back(e.path());
    if (shards.empty()) {
        fprintf(stderr, "no .safetensors files in %s\n", opt.in_dir.c_str());
        return 1;
    }
    std::sort(shards.begin(), shards.end());

    if (!opt.dry_run) {
        // Before any work: a source that cannot carry the chosen format's
        // declaration must cost a second, not a full conversion that then
        // refuses to finish.
        if (const auto declarable = quantize::can_declare_quantization(opt.in_dir, opt.format); !declarable) {
            fprintf(stderr, "%s\n", declarable.error().c_str());
            return 1;
        }
        // Same reason: a combination that will not load where the caller is
        // aiming should be said now, not after the conversion.
        if (const char* warn = quantize::portability_warning(opt.format,
                                                             opt.lm_head != quantize::LmHeadExport::Source))
            fprintf(stderr, "note: %s\n\n", warn);
        fs::create_directories(opt.out_dir, ec);
        if (ec) {
            fprintf(stderr, "cannot create %s: %s\n", opt.out_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    // Every shard is opened before anything is written: the AWQ transform is
    // cross-tensor (a scale decided on down_proj is folded into up_proj) and a
    // sharded checkpoint gives no guarantee those two live in the same file.
    std::vector<std::unique_ptr<RawSafeTensors>> opened;
    for (const auto& shard : shards) {
        auto src = std::make_unique<RawSafeTensors>();
        std::string err = src->open(shard.string());
        if (!err.empty()) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        opened.push_back(std::move(src));
    }

    // Expert stacks: split per expert by the model's layout, or refused (resolve_expert_stacks).
    std::set<std::string> stacked_names;
    quantize::StackedExpertLayout stack_layout;
    if (!resolve_expert_stacks(opened, opt.in_dir, stacked_names, stack_layout))
        return 1;

    // Fused Q+gate projections are quantized like anything else; reported because the gate half
    // is where #1273's divergence is created (+0.0169 injected per attention block on a rounded
    // twin vs +0.0156 for the real NVFP4 checkpoint; the Q half sits below the noise floor).
    // That divergence is real, but excluding the tensor measured ~1.5% WORSE perplexity end to end
    // on Qwen3.5-4B (gate quantized 14.6665/14.6476/14.6716 vs gate excluded 14.8672/14.9339/14.8672,
    // BF16 reference 12.6735), for a 1-4% size cost. --keep-attn-gate keeps the option: the one
    // measured checkpoint has a lower gate share than the worst #1273 offender, so the trade may
    // still turn on a model with more of them.
    std::set<std::string> gated_q_proj;
    {
        std::vector<const RawTensor*> gated;
        for (const auto& src : opened) {
            auto found = quantize::find_fused_gate_q_projections(src->tensors());
            gated.insert(gated.end(), found.begin(), found.end());
        }
        if (!gated.empty()) {
            // --keep-attn-gate + --calib is mathematically unsound and would be SILENT: a fused Q+gate
            // q_proj is copied through without the group's column scale, but the planner unconditionally
            // builds group A from {q,k,v} and folds its 1/s into input_layernorm regardless, producing a
            // wrong checkpoint that loads and generates. Refused outright, same call #1188 made for
            // stacked experts.
            if (opt.keep_attn_gate && !opt.calib_file.empty()) {
                fprintf(stderr,
                        "Error: --keep-attn-gate cannot be combined with --calib. The gate half is\n"
                        "excluded from the group scale, but the calibration plan still folds that\n"
                        "group's 1/s into input_layernorm — the checkpoint would be silently wrong.\n"
                        "Drop one of the two flags (%zu fused Q+gate projection(s) found).\n",
                        gated.size());
                return 1;
            }
            size_t extra_bytes = 0;
            for (const RawTensor* t : gated) {
                // What keeping it full precision costs over quantizing it.
                const size_t nvfp4 = t->numel() / 2 + t->numel() / 16;
                extra_bytes += t->nbytes > nvfp4 ? t->nbytes - nvfp4 : 0;
                if (opt.keep_attn_gate)
                    gated_q_proj.insert(t->name);
            }
            printf("%s %zu fused Q+gate projection(s), %.0f MiB %s (#1273)\n",
                   opt.keep_attn_gate ? "  KEEPING" : "  QUANTIZING", gated.size(),
                   double(extra_bytes) / (1024.0 * 1024.0),
                   opt.keep_attn_gate ? "larger (--keep-attn-gate)" : "saved");
            if (opt.keep_attn_gate)
                fprintf(stderr,
                        "note: --keep-attn-gate excludes the gate half. On the one checkpoint this\n"
                        "could be measured on it cost ~1.5%% perplexity rather than gaining any.\n");
        }
    }

    awq::Plan plan;
    quantize::Recipe recipe;  // #2481: written into the checkpoint
    // A dry run reports what would be quantized; the scale search costs a GPU
    // pass per layer and changes nothing it would report.
    if (!opt.calib_file.empty() && !opt.dry_run) {
        CalibrationStats stats;
        std::string err = read_calibration_stats(opt.calib_file, stats);
        if (!err.empty()) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::map<std::string, const RawTensor*> index;
        for (const auto& src : opened)
            for (const auto& t : src->tensors())
                index[t.name] = &t;
        uint64_t samples = 0;
        for (const auto& e : stats.entries)
            samples = std::max(samples, e.rows);
        printf("%s\n", quantize::experimental_banner(/*calibrated=*/true, samples).c_str());
        recipe.calib_model_id = stats.model_id;
        recipe.calib_samples = samples;
        recipe.calib_entries = stats.entries.size();
        printf("AWQ calibration: %zu entries from %s\n", stats.entries.size(),
               stats.model_id.empty() ? opt.calib_file.c_str() : stats.model_id.c_str());
        // Expert stacks enter the plan under their per-expert export names (groups X, Y).
        awq::StackedIndex stacked;
        for (const auto& src : opened) {
            std::map<std::string, std::vector<quantize::DestackedMatrix>> shard_plans;
            size_t n_unused = 0;
            if (!plan_shard_stacks(*src, stacked_names, stack_layout, shard_plans, n_unused))
                return 1;
            for (const auto& [stack_name, ms] : shard_plans)
                for (const auto& m : ms)
                    stacked[m.name] = {index.at(stack_name), stack_layout, m};
        }
        auto built = awq::build_plan(index, stats, (fs::path(opt.in_dir) / "config.json").string(),
                                     opt.calib_groups, opt.calib_weight_sq, stacked);
        if (!built) {
            fprintf(stderr, "%s\n", built.error().c_str());
            return 1;
        }
        plan = std::move(*built);
        recipe.calib_groups = plan.groups;
        recipe.n_rep = plan.n_rep;
        recipe.hybrid = plan.hybrid;
        printf("AWQ: %d groups scaled, %d kept round-to-nearest, %d skipped", plan.groups_scaled,
               plan.groups_rtn, plan.groups_skipped);
        if (plan.groups_disabled > 0)
            printf(", %d disabled (groups %s)", plan.groups_disabled, plan.groups.c_str());
        if (plan.channels_clamped > 0)
            printf(", %d norm channel(s) clamped to what the dtype can store", plan.channels_clamped);
        printf("\n");
        for (const auto& n : plan.notes)
            printf("  note: %s\n", n.c_str());
        if (plan.groups_scaled == 0) {
            fprintf(stderr,
                    "AWQ found no group worth scaling — refusing to write a checkpoint that\n"
                    "would be labelled calibrated but is byte-identical to round-to-nearest.\n");
            return 1;
        }
    }

    // FP8 sources store an E4M3 weight beside a weight_scale_inv block grid (DeepSeek-V3, Qwen3.8 FP8)
    // or a scalar weight_scale (Modelopt, #2473); paired up front across shards since the two aren't
    // guaranteed to share a file and should_quantize sees one tensor at a time. Scale tensors are then
    // CONSUMED (once the weight is NVFP4 they describe nothing).
    std::map<std::string, const RawTensor*> fp8_scale_of;
    std::set<std::string> fp8_scale_names;
    {
        std::map<std::string, const RawTensor*> by_name;
        for (const auto& src : opened)
            for (const auto& t : src->tensors())
                by_name[t.name] = &t;
        quantize::Fp8Pairing pairing = quantize::pair_fp8_scales(by_name);
        for (const auto& name : pairing.unpaired)
            fprintf(stderr,
                    "note: %s is E4M3 with neither .weight_scale_inv nor a scalar .weight_scale beside it,\n"
                    "      so it is copied through unquantized.\n",
                    name.c_str());
        if (!pairing.scale_of.empty())
            printf(
                "FP8 source: %zu block-scaled + %zu per-tensor-scaled E4M3 tensor(s) will be widened before "
                "quantizing\n",
                pairing.n_block, pairing.n_tensor);
        fp8_scale_of = std::move(pairing.scale_of);
        fp8_scale_names = std::move(pairing.consumed);
    }

    // Fused layers share one tensor scale (checkpoint_out.h): an engine merging q/k/v into one
    // linear keeps one scale for the merged weight, so three independently calibrated scales leave
    // two matrices dequantized wrong. Decided in its own pass over the source since the scale must
    // be known before the first member is quantized and members aren't guaranteed to share a shard.
    std::map<std::string, float> forced_scale;
    if (!opt.dry_run) {
        std::map<std::string, std::vector<const RawTensor*>> groups;
        for (const auto& src : opened) {
            for (const auto& t : src->tensors()) {
                if (fp8_scale_names.count(t.name) || gated_q_proj.count(t.name))
                    continue;
                // Asks the policy exactly as the writer does, so a tensor the writer keeps at full precision
                // isn't given a scale here, and doesn't drag a fused sibling's scale up with an absmax that
                // was
                // never quantized.
                std::string why;
                const bool refused = quantize::keep_gdn_projection(t.name, opt.keep_gdn_proj) ||
                                     (fp8_scale_of.count(t.name)
                                          ? quantize::fp8_source_action(t, false, opt.quantize_lm_head,
                                                                        why) !=
                                                quantize::Fp8SourceAction::Quantize
                                          : !quantize::should_quantize(t, opt.quantize_lm_head, why));
                if (refused)
                    continue;
                const std::string key = quantize::fusion_group_key(t.name);
                if (!key.empty())
                    groups[key].push_back(&t);
            }
        }
        size_t n_shared = 0, n_groups = 0;
        for (auto& [key, members] : groups) {
            if (members.size() < 2)
                continue;  // a group of one shares with nobody
            float amax = 0.0f;
            for (const RawTensor* t : members) {
                const auto h = tensor_as_fp16(*t, fp8_scale_of, plan);
                if (!h) {
                    fprintf(stderr, "  %s: %s\n", t->name.c_str(), h.error().c_str());
                    return 1;
                }
                amax = std::max(amax, quantize::fp16_absmax(h->data(), h->size()));
            }
            const float s = quantize::export_tensor_scale(amax);
            for (const RawTensor* t : members)
                forced_scale[t->name] = s;
            n_shared += members.size();
            n_groups++;
        }
        if (n_groups)
            printf("fused layers: %zu tensors in %zu groups share a tensor scale\n", n_shared, n_groups);
    }

    auto forced_of = [&](const std::string& name) {
        const auto it = forced_scale.find(name);
        return it == forced_scale.end() ? 0.0f : it->second;
    };
    // The three tensors one quantized module becomes, named and scaled for the
    // output format. `scale_store` owns the F32 until the shard is written, so
    // it must not reallocate — the callers reserve it per shard.
    auto emit_quantized = [&](std::vector<SafeTensorsOut>& out, std::vector<float>& scale_store,
                              const std::string& weight_name, int64_t N, int64_t K, const Quantized& q) {
        scale_store.push_back(quantize::global_scale_value(q.tensor_scale, opt.format));
        const std::string base = weight_name.substr(0, weight_name.size() - strlen(".weight"));
        const quantize::QuantTensorNames names = quantize::quant_tensor_names(base, opt.format);
        out.push_back({names.packed, "U8", {N, K / 2}, q.packed.data(), q.packed.size()});
        out.push_back({names.micro_scale, "F8_E4M3", {N, K / 16}, q.micro.data(), q.micro.size()});
        out.push_back({names.global_scale, "F32", {1}, &scale_store.back(), sizeof(float)});
    };

    size_t n_quantized = 0, n_copied = 0, n_stacks_split = 0;
    size_t bytes_in = 0, bytes_out = 0, mx_widen_bytes = 0;
    // Where the bytes that did NOT shrink went: a checkpoint missing the card by a gigabyte is a
    // question about this table, not the ratio (e.g. the embedding pair alone is a quarter of the
    // output on a modern vocabulary, which the ratio never reveals).
    std::map<std::string, size_t> copied_bytes_by_reason;
    std::vector<std::string> excluded_modules;
    // --gdn-proj-mxfp8 (#2475): 0 = not this tensor, 1 = written (store owns the bytes until the
    // shard is written), -1 = error.
    auto emit_mxfp8 = [&](const RawTensor& t, std::vector<SafeTensorsOut>& out,
                          std::vector<std::vector<uint8_t>>& store) -> int {
        if (!mxfp8_target(opt, t))
            return 0;
        const size_t bytes = mxfp8_bytes(t);
        bytes_out += bytes;
        copied_bytes_by_reason["GDN projection (--gdn-proj-mxfp8)"] += bytes;
        mx_widen_bytes += static_cast<size_t>(t.numel()) * 2 - bytes;
        n_copied++;
        excluded_modules.push_back(t.name.substr(0, t.name.size() - strlen(".weight")));
        const auto h = opt.dry_run ? std::vector<uint16_t>{} : tensor_as_fp16(t, fp8_scale_of, plan);
        if (!h)
            fprintf(stderr, "  %s: %s\n", t.name.c_str(), h.error().c_str());
        else
            push_mxfp8(t, *h, opt.dry_run, out, store);
        return h ? 1 : -1;
    };
    // What every quantized tensor cost, measured on the bytes that were
    // written. Costs no GPU: the packed nibbles and micro-scales are already
    // on the host at this point.
    std::vector<quantize::TensorError> tensor_errors;
    std::vector<std::pair<std::string, std::string>> tensor_to_shard;

    for (size_t shard_idx = 0; shard_idx < shards.size(); shard_idx++) {
        const fs::path& shard = shards[shard_idx];
        const RawSafeTensors& src = *opened[shard_idx];
        std::string err;
        printf("[%s] %zu tensors\n", shard.filename().string().c_str(), src.tensors().size());

        // Owns the buffers the output descriptors point at until the shard is written.
        std::vector<Quantized> quant_store;
        std::vector<float> scale_store;
        std::vector<std::vector<unsigned char>> folded_store;
        // FP8 weights this tool refuses: widened here, so the buffer must outlive
        // the descriptor that points at it.
        std::vector<std::vector<uint16_t>> widened_store;
        std::vector<std::vector<uint8_t>> mx_store;  // --gdn-proj-mxfp8 bytes, same lifetime
        std::vector<quantize::Fp8Head> head_store;  // the FP8 LM head; moves keep the buffers in place
        std::map<std::string, std::vector<quantize::DestackedMatrix>> stack_plans;
        size_t n_destacked = 0;
        if (!plan_shard_stacks(src, stacked_names, stack_layout, stack_plans, n_destacked))
            return 1;
        quant_store.reserve(src.tensors().size() + n_destacked);
        scale_store.reserve(src.tensors().size() + n_destacked);
        folded_store.reserve(src.tensors().size());
        widened_store.reserve(src.tensors().size());
        std::vector<SafeTensorsOut> out;

        for (const auto& t : src.tensors()) {
            bytes_in += t.nbytes;
            // A consumed FP8 scale grid: its weight is about to become NVFP4
            // with scales of its own, so this tensor must not reach the output.
            if (fp8_scale_names.count(t.name)) {
                if (opt.dry_run)
                    printf("  DROP  %-58s FP8 scale, consumed by its weight\n", t.name.c_str());
                continue;
            }
            const auto fp8_it = fp8_scale_of.find(t.name);
            if (fp8_it != fp8_scale_of.end()) {
                const int64_t N = t.shape.size() == 2 ? t.shape[0] : 0;
                const int64_t K = t.shape.size() == 2 ? t.shape[1] : 0;
                // The same roles that must stay full precision in a BF16 source must stay full precision
                // here; only the dtype gate differs, so the policy is asked about the widened form rather
                // than
                // duplicating the rule. An FP8 tensor it refuses is copied through as-is.
                std::string why_fp8;
                const bool kept_gdn_fp8 = quantize::keep_gdn_projection(t.name, opt.keep_gdn_proj);
                if (kept_gdn_fp8)
                    why_fp8 = "GDN projection (--keep-gdn-proj)";
                if (const int mx = emit_mxfp8(t, out, mx_store); mx != 0) {
                    if (mx < 0)
                        return 1;
                    continue;
                }
                if (kept_gdn_fp8 ||
                    quantize::fp8_source_action(t, gated_q_proj.count(t.name) != 0, opt.quantize_lm_head,
                                                why_fp8) != quantize::Fp8SourceAction::Quantize) {
                    // Its block grid was dropped above, so the E4M3 bytes cannot
                    // travel as they are. Widen and write full precision.
                    const size_t widened_bytes = static_cast<size_t>(t.numel()) * sizeof(uint16_t);
                    copied_bytes_by_reason[why_fp8] += widened_bytes;
                    bytes_out += widened_bytes;
                    n_copied++;
                    if (ends_with(t.name, ".weight") && t.shape.size() >= 2)
                        excluded_modules.push_back(t.name.substr(0, t.name.size() - strlen(".weight")));
                    if (opt.dry_run) {
                        printf("  COPY  %-58s widened from FP8: %s\n", t.name.c_str(), why_fp8.c_str());
                        continue;
                    }
                    auto h = tensor_as_fp16(t, fp8_scale_of, plan);
                    if (!h) {
                        fprintf(stderr, "  %s: %s\n", t.name.c_str(), h.error().c_str());
                        return 1;
                    }
                    widened_store.push_back(std::move(*h));
                    const std::vector<uint16_t>& w = widened_store.back();
                    out.push_back({t.name, "F16", t.shape, w.data(), w.size() * sizeof(uint16_t)});
                    continue;
                }
                if (opt.dry_run) {
                    printf("  QUANT %-58s [%lld,%lld] (from FP8)\n", t.name.c_str(), (long long)N,
                           (long long)K);
                    bytes_out += quantize::nvfp4_output_bytes(N, K);
                    n_quantized++;
                    continue;
                }
                const auto widened = tensor_as_fp16(t, fp8_scale_of, plan);
                if (!widened) {
                    fprintf(stderr, "  %s: %s\n", t.name.c_str(), widened.error().c_str());
                    return 1;
                }
                const std::vector<uint16_t>& h = *widened;
                auto quantized = quantize_one(h, N, K, forced_of(t.name));
                if (!quantized) {
                    fprintf(stderr, "  %s: %s\n", t.name.c_str(), quantized.error().c_str());
                    return 1;
                }
                quant_store.push_back(std::move(*quantized));
                const Quantized& q = quant_store.back();
                emit_quantized(out, scale_store, t.name, N, K, q);
                tensor_errors.push_back(quantize::nvfp4_tensor_error(t.name, h.data(), q.packed.data(),
                                                                     q.micro.data(), q.tensor_scale, N, K));
                bytes_out += q.packed.size() + q.micro.size() + sizeof(float);
                n_quantized++;
                continue;
            }
            // An expert stack: quantized as its per-expert matrices (quantize_expert_stack).
            if (const auto sp = stack_plans.find(t.name); sp != stack_plans.end()) {
                ShardSinks sinks{quant_store, scale_store, out, tensor_errors, bytes_out, n_quantized};
                if (!quantize_expert_stack(t, sp->second, stack_layout, plan, opt.dry_run, sinks,
                                           emit_quantized))
                    return 1;
                n_stacks_split++;
                continue;
            }
            if (opt.lm_head == quantize::LmHeadExport::Fp8) {
                const int head = quantize::emit_fp8_head(t, opt.dry_run, head_store, out, bytes_out);
                if (head < 0)
                    return 1;
                if (head > 0) {
                    excluded_modules.emplace_back("lm_head");  // not NVFP4: no scales expected
                    continue;
                }
            }
            std::string why;
            // Checked before should_quantize, which sees one tensor and cannot
            // know a q_proj is gated — that needs the layer's o_proj too.
            const bool gated = gated_q_proj.count(t.name) != 0;
            const bool kept_gdn = quantize::keep_gdn_projection(t.name, opt.keep_gdn_proj);
            if (gated)
                why = "fused Q+gate projection (--keep-attn-gate)";
            else if (kept_gdn)
                why = "GDN projection (--keep-gdn-proj)";
            if (gated || kept_gdn || !quantize::should_quantize(t, opt.quantize_lm_head, why)) {
                if (const int mx = emit_mxfp8(t, out, mx_store); mx != 0) {
                    if (mx < 0)
                        return 1;
                    continue;
                }
                if (ends_with(t.name, ".weight") && t.shape.size() >= 2) {
                    // Record real matrices we left alone so the runtime does not
                    // expect scales for them.
                    std::string mod = t.name.substr(0, t.name.size() - strlen(".weight"));
                    excluded_modules.push_back(mod);
                }
                // Copied through — unless it is the producer AWQ folded 1/s
                // into (an RMSNorm weight, or a bias on a scaled output).
                const void* data = t.data;
                auto vd = plan.vec_div.find(t.name);
                if (vd != plan.vec_div.end() && !opt.dry_run) {
                    bool folded = false;
                    const auto off = plan.vec_offset.find(t.name);
                    folded_store.push_back(folded_copy(
                        t, vd->second, off == plan.vec_offset.end() ? NormOffset::Plain : off->second,
                        folded));
                    if (!folded) {
                        fprintf(stderr,
                                "  %s: cannot fold the AWQ scale into a %s tensor of %lld elements "
                                "— refusing to write a half-transformed checkpoint\n",
                                t.name.c_str(), t.dtype.c_str(), (long long)t.numel());
                        return 1;
                    }
                    data = folded_store.back().data();
                }
                out.push_back({t.name, t.dtype, t.shape, data, t.nbytes});
                bytes_out += t.nbytes;
                copied_bytes_by_reason[why] += t.nbytes;
                n_copied++;
                continue;
            }

            const int64_t N = t.shape[0], K = t.shape[1];
            if (opt.dry_run) {
                printf("  QUANT %-58s [%lld,%lld]\n", t.name.c_str(), (long long)N, (long long)K);
                bytes_out += quantize::nvfp4_output_bytes(N, K);
                n_quantized++;
                continue;
            }

            const auto widened = tensor_as_fp16(t, fp8_scale_of, plan);
            if (!widened) {
                fprintf(stderr, "  %s: %s\n", t.name.c_str(), widened.error().c_str());
                return 1;
            }
            const std::vector<uint16_t>& h = *widened;
            auto quantized = quantize_one(h, N, K, forced_of(t.name));
            if (!quantized) {
                fprintf(stderr, "  %s: %s\n", t.name.c_str(), quantized.error().c_str());
                return 1;
            }
            quant_store.push_back(std::move(*quantized));
            const Quantized& q = quant_store.back();
            emit_quantized(out, scale_store, t.name, N, K, q);
            tensor_errors.push_back(quantize::nvfp4_tensor_error(t.name, h.data(), q.packed.data(),
                                                                 q.micro.data(), q.tensor_scale, N, K));
            const size_t written = q.packed.size() + q.micro.size() + sizeof(float);
            // The forecast --dry-run printed is this same arithmetic. If the two
            // ever disagree the forecast has quietly become a guess, so say so
            // here rather than let a wrong size be published as a measurement.
            if (written != quantize::nvfp4_output_bytes(N, K)) {
                fprintf(stderr,
                        "  %s: quantized to %zu bytes but the size forecast says %zu. The NVFP4\n"
                        "output layout changed without nvfp4_output_bytes() following it\n",
                        t.name.c_str(), written, quantize::nvfp4_output_bytes(N, K));
                return 1;
            }
            bytes_out += written;
            n_quantized++;
        }

        if (opt.dry_run)
            continue;

        const fs::path dst = fs::path(opt.out_dir) / shard.filename();
        err = write_safetensors(dst.string(), out, {{"format", "pt"}, {"producer", "imp-quantize"}});
        if (!err.empty()) {
            fprintf(stderr, "writing %s: %s\n", dst.string().c_str(), err.c_str());
            return 1;
        }
        for (const auto& o : out)
            tensor_to_shard.emplace_back(o.name, shard.filename().string());
        printf("  -> %s\n", dst.string().c_str());
    }

    if (n_quantized == 0) {
        fprintf(stderr, "nothing was quantized — is this already an NVFP4 checkpoint?\n");
        return 1;
    }

    if (!opt.dry_run) {
        const bool calibrated = !opt.calib_file.empty();
        if (const auto copied = quantize::copy_aux_files(opt.in_dir, opt.out_dir, opt.format,
                                                         excluded_modules, calibrated);
            !copied) {
            fprintf(stderr, "%s\n", copied.error().c_str());
            return 1;
        }
        // Modelopt declares itself in a file of its own; compressed-tensors put
        // its declaration into the config.json copy_aux_files just patched, and
        // repeats it in recipe.yaml for readers that look only there.
        const std::string mt = quantize::model_type_from_config(
            (fs::path(opt.in_dir) / "config.json").string());
        const bool kv_fp8 = quantize::kv_fp8_for_export(opt.kv_hint, opt.format, mt);  // #2480
        printf("kv_cache_quant_algo: %s (model_type %s, --kv-hint %s, --format %s)\n",
               kv_fp8 ? "FP8" : "null", mt.c_str(), quantize::kv_hint_name(opt.kv_hint),
               quantize::format_name(opt.format));
        quantize::fill_recipe(recipe, opt, kv_fp8);
        auto declared = opt.format == quantize::OutputFormat::Modelopt
                            ? quantize::write_modelopt_quant_config(opt.out_dir, excluded_modules, calibrated,
                                                                    kv_fp8,
                                                                    quantize::recipe_json(recipe, "    "))
                            : quantize::write_recipe_yaml(opt.out_dir, excluded_modules);
        if (declared && opt.format != quantize::OutputFormat::Modelopt)
            declared = quantize::write_recipe_json(opt.out_dir, quantize::recipe_json(recipe, ""));
        if (!declared) {
            fprintf(stderr, "%s\n", declared.error().c_str());
            return 1;
        }
        // Single-shard checkpoints load off model.safetensors directly; sharded
        // ones need the index, and it must describe the tensors we wrote.
        if (shards.size() > 1) {
            const auto indexed = quantize::write_shard_index(opt.out_dir, tensor_to_shard, bytes_out);
            if (!indexed) {
                fprintf(stderr, "%s\n", indexed.error().c_str());
                return 1;
            }
        }
    }

    printf("\n%s: %zu tensors quantized, %zu copied, %s layout", opt.dry_run ? "dry run" : "done",
           n_quantized, n_copied, quantize::format_name(opt.format));
    if (!opt.dry_run && opt.calib_file.empty())
        printf(
            "\n\nUNCALIBRATED OUTPUT: round-to-nearest, no activation calibration.\n"
            "      Measured cost on the dense Qwen3 pair: PPL +25%% (0.6B) / +19%% (1.7B).\n"
            "      Pass --calib to spend a calibration pass and recover most of that.");
    else if (!opt.dry_run)
        // The scale search minimises a per-group weight-reconstruction error, a local proxy: it can
        // improve on every group and still leave the model worse. On Qwen3-14B, two independently
        // produced calibration files gave PPL 9.93 (round-to-nearest) vs 12.60/12.29 (calibrated).
        // Attributed via --calib-groups: the harm is the ATTENTION groups on wide GQA, mostly their
        // interaction (AxC +1.36); BD alone GAINS 0.13. The default drops A and C there; the note
        // fires only when an explicit selector puts them back.
        printf(
            "\n\nAWQ-calibrated: %d groups scaled, %d left at round-to-nearest (groups %s).%s"
            "\n      Score this checkpoint with --perplexity against the uncalibrated one"
            "\n      before using it; see docs/quantization.md.",
            plan.groups_scaled, plan.groups_rtn, plan.groups.c_str(),
            awq::attention_groups_on_wide_gqa(plan.groups, plan.n_rep, plan.hybrid)
                ? "\n      NOTE: attention groups A/C on wide GQA (n_rep >= 5) measured HARMFUL:"
                  "\n      Qwen3-14B ABCD 12.2634 vs BD 9.9068 vs round-to-nearest 9.9849 PPL."
                  "\n      Omit --calib-groups to get the wide-GQA default."
                : "");
    if (n_stacks_split)
        printf(", %zu expert stack(s) split into per-expert matrices", n_stacks_split);
    printf("\nsize: %.2f GiB -> %.2f GiB (%.2fx)%s", bytes_in / 1073741824.0, bytes_out / 1073741824.0,
           bytes_out ? double(bytes_in) / double(bytes_out) : 0.0, opt.dry_run ? " (forecast)" : "");
    // What it cost, per tensor. Until this line existed the only number an
    // operator could get out of an export was its size.
    if (!opt.dry_run) {
        printf("\n%s", quantize::format_error_summary(tensor_errors, 5).c_str());
        if (const auto wrote = quantize::write_error_report(opt.out_dir, tensor_errors, plan.group_errors,
                                                            !opt.calib_file.empty());
            !wrote)
            fprintf(stderr, "\n%s\n", wrote.error().c_str());
    }
    report_copied_breakdown(copied_bytes_by_reason, bytes_out);
    report_card_fit(bytes_out, mx_widen_bytes);
    printf("\n");
    return 0;
}

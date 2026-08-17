// imp-quantize — turn a BF16/FP16 SafeTensors checkpoint into an NVFP4 one.
//
// STATUS: EXPERIMENTAL. The pipeline is verified end to end and `--calib` now
// recovers a measurable part of the quantization loss, but the result still
// sits below a published Modelopt export. Intended for getting a model onto
// the NVFP4 path for evaluation or performance work — not for producing
// checkpoints anyone should rely on.
//
// Why this exists: imp could only ever CONSUME NVFP4 checkpoints, so both model
// coverage and quantization quality were gated on somebody else publishing a
// Modelopt / llm-compressor export (docs/roadmap.md, gap 1). A model without an
// export falls back to the GGUF path, whose prefill ceiling is architectural.
//
// The quantization itself is not new code: src/quant/nvfp4_quant.h already has
// the two-level (FP8 E4M3 micro-scale per 16 values + FP32 tensor scale) kernel
// used to build the decode cache at load time. This tool applies it offline and
// writes the result in one of two layouts (--format, see checkpoint_out.h):
//
//   modelopt (default)             compressed-tensors (--format vllm)
//   <p>.weight         U8          <p>.weight_packed        U8
//   <p>.weight_scale   F8_E4M3     <p>.weight_scale         F8_E4M3
//   <p>.weight_scale_2 F32         <p>.weight_global_scale  F32 (1/scale)
//   + hf_quant_config.json         + quantization_config in config.json
//
// Both are read by imp. Only the second is read by vLLM.
//
// Scope of this version: dense models AND MoE checkpoints that store experts
// the HF-standard way — one 2-D tensor per expert
// (`...mlp.experts.<e>.gate_proj.weight`). Those need no special handling and
// are quantized like any other matrix; measured on DeepSeek-V2-Lite, 4992
// expert tensors quantize and the result generates correctly. Expert weights
// stored as a single 3-D [n_experts, N, K] STACK (gpt-oss-style) are still
// unsupported, and such a checkpoint is now REFUSED outright rather than
// written with its experts copied through — see tensor_policy.h.
//
// Two roles are deliberately excluded even though they are 2-D and K-aligned:
// the MLA latent projections and the MoE router (see should_quantize for the
// measurements). Quantizing either produces a checkpoint that loads and then
// emits garbage — which is why they are refused rather than trusted to shape.
//
// QUALITY, MEASURED — read before using this on anything you care about.
// Without --calib the scales are plain round-to-nearest over the weights
// (absmax per micro-block, absmax per tensor): nothing protects the channels
// that matter most, because nothing has looked at an activation. With --calib
// they are AWQ scales searched against a calibration pass (awq.h, awq_plan.cpp).
// Measured over tools/analysis/ppl_corpus_45k.txt (13 537 tokens), calibrated
// on general prose that is NOT the scoring corpus:
//
//   Qwen3-0.6B  BF16 24.08 -> RTN 29.42 (+22%) -> AWQ 27.60 (+15%)
//   Qwen3-1.7B  BF16 17.22 -> RTN 20.39 (+18%) -> AWQ 18.71 (+9%)
//
// Both arms improved when fused layers started sharing a tensor scale
// (2026-08-17). The figures this file carried before that change were
// 30.10 / 28.48 and 20.43 / 19.21.
//
// Use a corpus of that size to judge this. The same model over the 199-token
// ppl_corpus.txt reads wildly different numbers and inverts the size trend —
// an artifact of too few tokens, not a property of the quantizer.
//
// Coherence: tools/analysis/degen_suite.py reads 45/45 on every checkpoint
// above (the AWQ ones re-run). Checkpoints built from a NON-deterministic
// calibration file each flipped one probe, a different one each time — which
// is why --calibrate forces runtime.deterministic_gemm. See
// docs/quantization.md.
//
// A published export is NOT automatically better: on Qwen3-14B — same weights
// (bit-identical untouched tensors), same 280 quantized tensors, same corpus —
// this tool without --calib read PPL 9.9252 against a Modelopt export's 10.0301.
// One model, one corpus, so not a general claim; see docs/quantization.md.
// Useful today for: getting a model onto the NVFP4 path at all, and for
// performance work where the weights only need to be the right shape.

#include "awq.h"
#include "checkpoint_out.h"
#include "fp8_source.h"
#include "tensor_policy.h"

#include "core/tensor.h"
#include "memory/plan.h"
#include "model/safetensors_raw.h"
#include "model/safetensors_writer.h"
#include "quant/awq_transform.h"
#include "quant/calibration_stats.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace imp;

namespace {

struct Options {
    std::string in_dir, out_dir;
    std::string calib_file;  // --calib: activation statistics for AWQ scaling
    // --calib-groups: which AWQ scale groups run, for attributing a bad result.
    std::string calib_groups = awq::kAwqAllGroups;
    bool quantize_lm_head = false;  // imp has its own lm_head NVFP4 policy (#982)
    // Keep a fused Q+gate q_proj out of NVFP4. OFF by default: the gate half
    // demonstrably carries the #1273 divergence, but excluding it did NOT
    // improve perplexity when measured end to end — it cost ~1.5% (see the
    // header). Kept as an opt-in so the trade stays available on models where
    // the gate share is higher than the one checkpoint that could be measured.
    bool keep_attn_gate = false;
    bool dry_run = false;
    quantize::OutputFormat format = quantize::OutputFormat::Modelopt;
};

void usage() {
    printf(
        "usage: imp-quantize --model <safetensors-dir> --out <dir> [--calib <file>]\n"
        "                    [--format modelopt|vllm] [--lm-head] [--dry-run]\n"
        "\n"
        "  --model DIR   source checkpoint (BF16/FP16 SafeTensors + config.json)\n"
        "  --out DIR     destination; created if missing\n"
        "  --format F    output layout (default modelopt):\n"
        "                  modelopt            .weight / .weight_scale_2, declared in\n"
        "                                      hf_quant_config.json. imp reads it.\n"
        "                  vllm                compressed-tensors nvfp4-pack-quantized:\n"
        "                  (= compressed-tensors)  .weight_packed / .weight_global_scale,\n"
        "                                      declared in config.json. imp AND vLLM read\n"
        "                                      it, and fused layers (q/k/v, gate/up) share\n"
        "                                      one tensor scale, which vLLM requires.\n"
        "  --calib FILE  activation calibration (AWQ scaling). Produce it with:\n"
        "                  imp-cli --model DIR --perplexity <corpus> --calibrate FILE\n"
        "                Without it the quantization is plain round-to-nearest,\n"
        "                which costs measurably more quality (see the header).\n"
        "  --calib-groups ABCD\n"
        "                Which AWQ scale groups run (default ABCD).\n"
        "                  A q,k,v  <- input_layernorm        C o_proj   <- v_proj\n"
        "                  B gate,up<- post_attention_norm    D down_proj<- up_proj\n"
        "                The ATTENTION groups (A, C) are what breaks on wide GQA:\n"
        "                on Qwen3-14B (n_rep=5) ABCD costs +2.68 PPL while BD --\n"
        "                the two FFN groups -- GAINS 0.13 over round-to-nearest.\n"
        "                Use 'BD' on wide-GQA models, the default on narrow ones,\n"
        "                or any subset to attribute a bad result. See\n"
        "                docs/quantization.md.\n"
        "  --lm-head     also quantize lm_head (default: excluded, imp applies its\n"
        "                own measured lm_head policy at runtime)\n"
        "  --keep-attn-gate\n"
        "                keep a fused Q+gate q_proj (Qwen3.5 / Qwen3-Next\n"
        "                `attn_output_gate`) out of NVFP4. OFF by default: the gate\n"
        "                half carries #1273's divergence, but excluding it measured\n"
        "                ~1.5%% WORSE on perplexity, not better, and costs 1-4% of\n"
        "                the checkpoint. Kept for models with a higher gate share.\n"
        "  --dry-run     report what would be quantized and how large the result\n"
        "                will be, against what the card has left. Writes nothing.\n");
}

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

// Look up a plan vector by tensor name, or an empty one when the plan has
// nothing to say about it. awq_apply_* treat empty (and wrong-length) vectors
// as no-ops, so an untouched tensor takes the same path as a transformed one.
const std::vector<float>& plan_vec(const std::map<std::string, std::vector<float>>& m,
                                   const std::string& name) {
    static const std::vector<float> kNone;
    auto it = m.find(name);
    return (it == m.end()) ? kNone : it->second;
}

// A fresh copy of a 1-D producer with 1/s folded in, in the tensor's ORIGINAL
// dtype — the loader reads these by dtype, so widening here would be a format
// change rather than a fix.
std::vector<unsigned char> folded_copy(const RawTensor& t, const std::vector<float>& div, bool& ok) {
    std::vector<unsigned char> out(t.nbytes);
    std::memcpy(out.data(), t.data, t.nbytes);
    ok = awq_apply_vector_div(out.data(), static_cast<size_t>(t.numel()), t.dtype, div);
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
bool quantize_one(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K, float forced_scale,
                  Quantized& out, std::string& err) {
    void* d_in = nullptr;
    const size_t in_bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * 2;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        err = "cudaMalloc failed for " + std::to_string(in_bytes) + " bytes";
        return false;
    }
    if (cudaMemcpy(d_in, h_fp16.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_in);
        err = "H2D copy failed";
        return false;
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
        err = "quantization kernel failed";
        return false;
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
    if (!ok) {
        err = "D2H copy failed";
        return false;
    }
    return true;
}

// The FP16 form a tensor is quantized from: an FP8 source widened against its
// block-scale grid, otherwise the raw BF16/F16 bits, then the AWQ transform.
//
// One function because two callers need it — the scale planner and the writer —
// and a planner that measured a different tensor than the writer quantized
// would produce scales that are wrong by exactly the transform.
bool tensor_as_fp16(const RawTensor& t, const std::map<std::string, const RawTensor*>& fp8_scale_of,
                    const awq::Plan& plan, std::vector<uint16_t>& out, std::string& err) {
    const auto it = fp8_scale_of.find(t.name);
    if (it != fp8_scale_of.end()) {
        if (!quantize::fp8_block_scaled_to_fp16(t, *it->second, out, err))
            return false;
    } else {
        out = awq::raw_to_fp16(t);
    }
    awq_apply_matrix(out, t.shape[0], t.shape[1], plan_vec(plan.row_div, t.name),
                     plan_vec(plan.col_scale, t.name));
    return true;
}

// What is gone from the card before imp allocates a single weight. Both are
// measurements on this target, not headroom guesses: the library reserve is
// kMeasuredLibraryReserveBytes (src/memory/plan.h, re-measure after a CUDA or
// driver bump), and the CUDA primary context is the figure the same header
// records for this WSL2/WDDM box.
constexpr size_t kContextBytes = 1680ull * 1024 * 1024;

// Where the bytes that did not shrink went, largest first.
//
// The compression ratio answers "did it work" and not "why is it still this
// big", and those have different fixes. Every line here is a role deliberately
// left in source precision, so the table doubles as the list of what could be
// traded if a checkpoint misses the card: on a modern vocabulary the embedding
// pair is a quarter of the output, which no ratio reveals.
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

// Report the checkpoint against the card it is meant to run on. Deliberately
// states the ON-DISK size against the budget rather than predicting VRAM:
// what the engine actually resides is the weights PLUS a scale-factor cache and
// MINUS whatever the loader skips (a vision tower is uploaded separately, an MTP
// sidecar may not be loaded at all). Measured on Qwen3.8-27B: 18.60 GiB on disk
// arrived as 16.08 GiB of weights plus a 1.49 GiB CUTLASS scale cache. So the
// disk figure is the right order and the wrong decimal, and this prints the
// budget the reader needs rather than a number that looks exact and is not.
void report_card_fit(size_t bytes_out) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0)
        return;  // no device visible: the size line above still stands on its own
    const double mib = 1024.0 * 1024.0;
    const size_t overhead = kContextBytes + kMeasuredLibraryReserveBytes;
    if (total_b <= overhead)
        return;
    const double budget = double(total_b - overhead) / mib;
    const double ckpt = double(bytes_out) / mib;
    printf("\ncard: %.0f MiB total, less %.0f context and %.0f library reserve leaves %.0f MiB",
           double(total_b) / mib, kContextBytes / mib, kMeasuredLibraryReserveBytes / mib, budget);
    if (ckpt >= budget)
        printf(
            "\n      this checkpoint is %.0f MiB on disk and does NOT fit: %.0f MiB short before"
            "\n      any KV cache or workspace",
            ckpt, ckpt - budget);
    else
        printf(
            "\n      this checkpoint is %.0f MiB on disk, leaving %.0f MiB for the KV cache and"
            "\n      workspaces. On-disk size is not the resident figure: a scale-factor cache is"
            "\n      built on top, and a vision tower or MTP sidecar may load separately or not at all",
            ckpt, budget - ckpt);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model")
            opt.in_dir = next();
        else if (a == "--out")
            opt.out_dir = next();
        else if (a == "--lm-head")
            opt.quantize_lm_head = true;
        else if (a == "--keep-attn-gate")
            opt.keep_attn_gate = true;
        else if (a == "--calib")
            opt.calib_file = next();
        else if (a == "--format") {
            const std::string f = next();
            if (!quantize::parse_output_format(f, opt.format)) {
                fprintf(stderr, "imp-quantize: unknown --format '%s' (modelopt | vllm)\n", f.c_str());
                return 2;
            }
        } else if (a == "--calib-groups") {
            opt.calib_groups = next();
            for (char& c : opt.calib_groups)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            // An unknown letter here would silently select fewer groups than the
            // caller meant and quietly change what the checkpoint is — the same
            // shape as the `--set` unknown-key hole closed in #1186.
            if (opt.calib_groups.find_first_not_of(awq::kAwqAllGroups) != std::string::npos) {
                fprintf(stderr, "imp-quantize: --calib-groups '%s' has a letter outside %s\n",
                        opt.calib_groups.c_str(), awq::kAwqAllGroups);
                return 2;
            }
        } else if (a == "--dry-run")
            opt.dry_run = true;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (opt.in_dir.empty() || (opt.out_dir.empty() && !opt.dry_run)) {
        usage();
        return 2;
    }

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
        std::string derr;
        if (!quantize::can_declare_quantization(opt.in_dir, opt.format, derr)) {
            fprintf(stderr, "%s\n", derr.c_str());
            return 1;
        }
        // Same reason: a combination that will not load where the caller is
        // aiming should be said now, not after the conversion.
        if (const char* warn = quantize::portability_warning(opt.format, opt.quantize_lm_head))
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

    // Refuse a stacked-expert checkpoint before writing anything. The experts
    // are the bulk of the bytes and there is no NVFP4 layout the loader can
    // read them back from, so "quantizing" one produced a checkpoint that was
    // mostly still BF16 and said NVFP4 in its config — the failure being that
    // it loads and runs, just without the size or bandwidth win it claims.
    {
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
        if (!stacked.empty()) {
            fprintf(stderr,
                    "refusing: %zu tensor(s) store MoE experts as a 3-D stack — %.1f%% of this\n"
                    "checkpoint — and imp has no NVFP4 layout to read stacked experts back from.\n"
                    "Quantizing would copy them through unchanged and still label the result\n"
                    "NVFP4, so the output would claim a size and bandwidth win it does not have.\n",
                    stacked.size(), total_bytes ? 100.0 * double(stacked_bytes) / double(total_bytes) : 0.0);
            for (size_t i = 0; i < stacked.size() && i < 3; ++i)
                fprintf(stderr, "  %s\n", stacked[i]->name.c_str());
            if (stacked.size() > 3)
                fprintf(stderr, "  ... and %zu more\n", stacked.size() - 3);
            fprintf(stderr,
                    "Supported layout: one 2-D tensor per expert\n"
                    "(`...mlp.experts.<e>.gate_proj.weight`), which is the HF standard.\n");
            return 1;
        }
    }

    // Fused Q+gate projections are quantized like anything else. They are
    // reported because the gate half is where #1273's divergence is created:
    // rounding ONLY that half on a healthy GGUF twin reproduces the real defect
    // (+0.0169 injected per attention block vs +0.0156 for the actual NVFP4
    // checkpoint), while the Q half sits below the noise floor.
    //
    // That divergence is real. What does NOT follow from it is a quality win
    // for excluding the tensor, and this shipped briefly asserting one. Measured
    // end to end on Qwen3.5-4B (8 gated layers of 32), perplexity over
    // ppl_corpus_45k, three runs each:
    //
    //   gate quantized  14.6665 / 14.6476 / 14.6716   (spread 0.16%)
    //   gate excluded   14.8672 / 14.9339 / 14.8672   (spread 0.45%)
    //   BF16 reference  12.6735
    //
    // Excluding it is ~1.5% WORSE, consistently, with non-overlapping spreads —
    // so it also costs 1-4% of the checkpoint for nothing. Divergence against a
    // twin is not the same measurement as quality, and only the latter decides
    // this. `--keep-attn-gate` retains the option: the one checkpoint that could
    // be measured has a lower gate share than the worst #1273 offender (8/32
    // against 16/64), so the trade may still turn on a model with more of them.
    std::set<std::string> gated_q_proj;
    {
        std::vector<const RawTensor*> gated;
        for (const auto& src : opened) {
            auto found = quantize::find_fused_gate_q_projections(src->tensors());
            gated.insert(gated.end(), found.begin(), found.end());
        }
        if (!gated.empty()) {
            // --keep-attn-gate + --calib is mathematically unsound and would be
            // SILENT. A fused Q+gate q_proj is copied through below without the
            // group's column scale, but the planner has no idea this set exists
            // — it builds group A from {q,k,v} unconditionally and folds that
            // group's 1/s into input_layernorm. The result is an input divided
            // by s whose columns were never multiplied by it: a wrong
            // checkpoint that loads and generates.
            //
            // Unreachable today only because every arch with a fused attention
            // gate (qwen3_5, qwen3_next) fails arch_uses_plain_rmsnorm() — i.e.
            // it is armed for whoever widens that allowlist. Refuse rather than
            // mangle, the same call #1188 made for stacked experts.
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
        printf("AWQ calibration: %zu entries from %s\n", stats.entries.size(),
               stats.model_id.empty() ? opt.calib_file.c_str() : stats.model_id.c_str());
        if (!awq::build_plan(index, stats, (fs::path(opt.in_dir) / "config.json").string(), opt.calib_groups,
                             plan, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        printf("AWQ: %d groups scaled, %d kept round-to-nearest, %d skipped", plan.groups_scaled,
               plan.groups_rtn, plan.groups_skipped);
        if (plan.groups_disabled > 0)
            printf(", %d disabled (--calib-groups %s)", plan.groups_disabled, opt.calib_groups.c_str());
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

    // FP8 sources. A checkpoint published only in FP8 (DeepSeek-V3, Qwen3.8's
    // FP8 line) stores an E4M3 weight beside a `weight_scale_inv` block grid.
    // Pairing them up front, across shards, because the two are not guaranteed
    // to share a file and should_quantize sees one tensor at a time.
    //
    // The scale tensors are then CONSUMED: once the weight is NVFP4 they
    // describe nothing, and copying them through would leave a checkpoint whose
    // scales contradict its weights.
    std::map<std::string, const RawTensor*> fp8_scale_of;
    std::set<std::string> fp8_scale_names;
    {
        std::map<std::string, const RawTensor*> by_name;
        for (const auto& src : opened)
            for (const auto& t : src->tensors())
                by_name[t.name] = &t;
        static const std::string kW = ".weight";
        for (const auto& [name, t] : by_name) {
            if (!quantize::is_fp8_e4m3_dtype(t->dtype) || !ends_with(name, kW))
                continue;
            const auto it = by_name.find(name.substr(0, name.size() - kW.size()) + ".weight_scale_inv");
            if (it == by_name.end()) {
                // Without its grid the tensor is unreadable, and the generic
                // dtype exclusion below would report it as merely "unsupported".
                // Say which scale is missing: a scalar-scale FP8 export
                // (Modelopt style) lands here and needs different handling.
                fprintf(stderr,
                        "note: %s is E4M3 but has no .weight_scale_inv beside it, so it is copied\n"
                        "      through unquantized. Block-scaled FP8 is what this reads.\n",
                        name.c_str());
                continue;
            }
            fp8_scale_of[name] = it->second;
            fp8_scale_names.insert(it->first);
        }
        if (!fp8_scale_of.empty())
            printf("FP8 source: %zu block-scaled E4M3 tensor(s) will be widened before quantizing\n",
                   fp8_scale_of.size());
    }

    // Fused layers share one tensor scale (see checkpoint_out.h): an engine that
    // merges q/k/v into a single linear keeps one scale for the merged weight,
    // so three independently calibrated scales leave two matrices dequantized
    // against the wrong one. The scale must be known before the first member is
    // quantized and members are not guaranteed to share a shard, so it is
    // decided here, in its own pass over the source.
    std::map<std::string, float> forced_scale;
    if (!opt.dry_run) {
        std::map<std::string, std::vector<const RawTensor*>> groups;
        for (const auto& src : opened) {
            for (const auto& t : src->tensors()) {
                if (fp8_scale_names.count(t.name) || gated_q_proj.count(t.name))
                    continue;
                // Ask the policy about the widened form, exactly as the writer
                // does, so a tensor the writer copies through is not given a
                // scale here — and, worse, does not drag a fused sibling's
                // scale up with an absmax that was never quantized.
                RawTensor probe = t;
                if (fp8_scale_of.count(t.name))
                    probe.dtype = "BF16";
                std::string why;
                if (!quantize::should_quantize(probe, opt.quantize_lm_head, why))
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
                std::vector<uint16_t> h;
                std::string err;
                if (!tensor_as_fp16(*t, fp8_scale_of, plan, h, err)) {
                    fprintf(stderr, "  %s: %s\n", t->name.c_str(), err.c_str());
                    return 1;
                }
                amax = std::max(amax, quantize::fp16_absmax(h.data(), h.size()));
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

    size_t n_quantized = 0, n_copied = 0, n_moe_skipped = 0;
    size_t bytes_in = 0, bytes_out = 0;
    // Where the bytes that did NOT shrink went. A checkpoint that misses the
    // card by a gigabyte is a question about this table, not about the ratio:
    // on a modern vocabulary the embedding pair alone is a quarter of the
    // output, and the ratio never says so.
    std::map<std::string, size_t> copied_bytes_by_reason;
    std::vector<std::string> excluded_modules;
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
        quant_store.reserve(src.tensors().size());
        scale_store.reserve(src.tensors().size());
        folded_store.reserve(src.tensors().size());
        std::vector<SafeTensorsOut> out;

        for (const auto& t : src.tensors()) {
            bytes_in += t.nbytes;
            // A consumed FP8 scale grid: its weight is about to become NVFP4
            // with scales of its own, so this tensor must not reach the output.
            if (fp8_scale_names.count(t.name)) {
                if (opt.dry_run)
                    printf("  DROP  %-58s FP8 block scale, consumed by its weight\n", t.name.c_str());
                continue;
            }
            const auto fp8_it = fp8_scale_of.find(t.name);
            if (fp8_it != fp8_scale_of.end()) {
                const int64_t N = t.shape.size() == 2 ? t.shape[0] : 0;
                const int64_t K = t.shape.size() == 2 ? t.shape[1] : 0;
                // The same roles that must stay full precision in a BF16 source
                // must stay full precision here; only the dtype gate differs, so
                // ask the policy about the widened form rather than duplicating
                // the rule. An FP8 tensor it refuses is copied through as-is.
                RawTensor as_bf16 = t;
                as_bf16.dtype = "BF16";
                std::string why_fp8;
                const bool gated_fp8 = gated_q_proj.count(t.name) != 0;
                if (gated_fp8 || !quantize::should_quantize(as_bf16, opt.quantize_lm_head, why_fp8)) {
                    out.push_back({t.name, t.dtype, t.shape, t.data, t.nbytes});
                    bytes_out += t.nbytes;
                    copied_bytes_by_reason[gated_fp8 ? "fused Q+gate projection" : why_fp8] += t.nbytes;
                    n_copied++;
                    continue;
                }
                if (opt.dry_run) {
                    printf("  QUANT %-58s [%lld,%lld] (from FP8)\n", t.name.c_str(), (long long)N,
                           (long long)K);
                    bytes_out += quantize::nvfp4_output_bytes(N, K);
                    n_quantized++;
                    continue;
                }
                std::vector<uint16_t> h;
                std::string ferr;
                if (!tensor_as_fp16(t, fp8_scale_of, plan, h, ferr)) {
                    fprintf(stderr, "  %s: %s\n", t.name.c_str(), ferr.c_str());
                    return 1;
                }
                quant_store.emplace_back();
                Quantized& q = quant_store.back();
                if (!quantize_one(h, N, K, forced_of(t.name), q, err)) {
                    fprintf(stderr, "  %s: %s\n", t.name.c_str(), err.c_str());
                    return 1;
                }
                emit_quantized(out, scale_store, t.name, N, K, q);
                bytes_out += q.packed.size() + q.micro.size() + sizeof(float);
                n_quantized++;
                continue;
            }
            std::string why;
            // Checked before should_quantize, which sees one tensor and cannot
            // know a q_proj is gated — that needs the layer's o_proj too.
            const bool gated = gated_q_proj.count(t.name) != 0;
            if (gated)
                why = "fused Q+gate projection (--keep-attn-gate)";
            if (gated || !quantize::should_quantize(t, opt.quantize_lm_head, why)) {
                if (contains(why, "3-D stacked")) {
                    n_moe_skipped++;
                    printf("  SKIP  %-58s %s\n", t.name.c_str(), why.c_str());
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
                    folded_store.push_back(folded_copy(t, vd->second, folded));
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

            std::vector<uint16_t> h;
            if (!tensor_as_fp16(t, fp8_scale_of, plan, h, err)) {
                fprintf(stderr, "  %s: %s\n", t.name.c_str(), err.c_str());
                return 1;
            }
            quant_store.emplace_back();
            Quantized& q = quant_store.back();
            if (!quantize_one(h, N, K, forced_of(t.name), q, err)) {
                fprintf(stderr, "  %s: %s\n", t.name.c_str(), err.c_str());
                return 1;
            }
            emit_quantized(out, scale_store, t.name, N, K, q);
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
        std::string err;
        const bool calibrated = !opt.calib_file.empty();
        if (!quantize::copy_aux_files(opt.in_dir, opt.out_dir, opt.format, excluded_modules, calibrated,
                                      err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        // Modelopt declares itself in a file of its own; compressed-tensors put
        // its declaration into the config.json copy_aux_files just patched, and
        // repeats it in recipe.yaml for readers that look only there.
        const bool declared = opt.format == quantize::OutputFormat::Modelopt
                                  ? quantize::write_modelopt_quant_config(opt.out_dir, excluded_modules,
                                                                          calibrated, err)
                                  : quantize::write_recipe_yaml(opt.out_dir, excluded_modules, err);
        if (!declared) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        // Single-shard checkpoints load off model.safetensors directly; sharded
        // ones need the index, and it must describe the tensors we wrote.
        if (shards.size() > 1 && !quantize::write_shard_index(opt.out_dir, tensor_to_shard, bytes_out, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
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
        // The scale search minimises a per-group weight-reconstruction error,
        // which is a local proxy: it improved on every group here and the model
        // can still come out worse. Measured 2026-08-01 on Qwen3-14B, from two
        // independently produced calibration files (imp's own round-to-nearest
        // checkpoint and NVIDIA's Modelopt export): PPL 9.93 round-to-nearest
        // vs 12.60 / 12.29 calibrated. Attributed 2026-08-05 with --calib-groups:
        // the harm is the ATTENTION groups on wide GQA, and mostly their
        // interaction (A x C +1.36). BD alone GAINS 0.13 there, so the warning
        // names the way out instead of just the hazard.
        printf(
            "\n\nAWQ-calibrated: %d groups scaled, %d left at round-to-nearest.%s"
            "\n      Score this checkpoint with --perplexity against the uncalibrated one"
            "\n      before using it; see docs/quantization.md.",
            plan.groups_scaled, plan.groups_rtn,
            opt.calib_groups == std::string(awq::kAwqAllGroups)
                ? "\n      NOTE: the full set is validated on Qwen3-0.6B/1.7B but measured HARMFUL"
                  "\n      on Qwen3-14B (PPL 9.93 -> 12.3-12.6). The attention groups are the cause:"
                  "\n      on wide-GQA models prefer --calib-groups BD (14B: 9.79, better than RTN)."
                : "");
    if (n_moe_skipped)
        printf(", %zu MoE expert stacks left unquantized (not supported yet)", n_moe_skipped);
    printf("\nsize: %.2f GiB -> %.2f GiB (%.2fx)%s", bytes_in / 1073741824.0, bytes_out / 1073741824.0,
           bytes_out ? double(bytes_in) / double(bytes_out) : 0.0, opt.dry_run ? " (forecast)" : "");
    report_copied_breakdown(copied_bytes_by_reason, bytes_out);
    report_card_fit(bytes_out);
    printf("\n");
    return 0;
}

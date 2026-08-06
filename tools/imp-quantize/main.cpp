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
// writes the result in the layout hf_config_loader already recognises:
//
//   <prefix>.weight          U8       [N, K/2]   packed FP4 nibbles
//   <prefix>.weight_scale    F8_E4M3  [N, K/16]  micro-scales
//   <prefix>.weight_scale_2  F32      [1]        tensor scale
//   + hf_quant_config.json   {"quantization": {"quant_algo": "NVFP4", ...}}
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
//   Qwen3-0.6B  BF16 24.06 -> RTN 30.10 (+25%) -> AWQ 28.48 (+18%)
//   Qwen3-1.7B  BF16 17.22 -> RTN 20.43 (+19%) -> AWQ 19.21 (+12%)
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
#include "tensor_policy.h"

#include "core/tensor.h"
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
    std::string calib_file;         // --calib: activation statistics for AWQ scaling
    // --calib-groups: which AWQ scale groups run, for attributing a bad result.
    std::string calib_groups = awq::kAwqAllGroups;
    bool quantize_lm_head = false;  // imp has its own lm_head NVFP4 policy (#982)
    // A fused Q+gate q_proj is excluded by default: NVFP4 in the gate half is
    // the measured root cause of #1273. Costs 230 MiB on Qwen3.6-35B-A3B (10
    // gated layers of 40) and 552 MiB on the 27B (16 of 64) — ~1% and ~4% of the
    // checkpoint, against a 2.08x-6x perplexity penalty. This opts back in.
    bool quantize_attn_gate = false;
    bool dry_run = false;
};

void usage() {
    printf(
        "usage: imp-quantize --model <safetensors-dir> --out <dir> [--calib <file>]\n"
        "                    [--lm-head] [--dry-run]\n"
        "\n"
        "  --model DIR   source checkpoint (BF16/FP16 SafeTensors + config.json)\n"
        "  --out DIR     destination; created if missing\n"
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
        "  --quantize-attn-gate\n"
        "                also quantize a fused Q+gate q_proj (Qwen3.5 / Qwen3-Next\n"
        "                `attn_output_gate`). Excluded by default: NVFP4 in the gate\n"
        "                half is the measured root cause of #1273 and costs 2.08x-6x\n"
        "                perplexity. Keeping it full precision costs ~1-4% of the\n"
        "                checkpoint (230 MiB on Qwen3.6-35B-A3B, 552 on the 27B).\n"
        "  --dry-run     report what would be quantized, write nothing\n");
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

bool quantize_one(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K, Quantized& out,
                  std::string& err) {
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
    quantize_fp16_to_nvfp4(in, q);
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

bool copy_aux_files(const Options& opt, std::string& err) {
    // Everything the runtime needs beside the weights. config.json is copied
    // verbatim: the quantization is declared in hf_quant_config.json, which is
    // where hf_config_loader looks for it.
    static const char* kNames[] = {"config.json",
                                   "generation_config.json",
                                   "tokenizer.json",
                                   "tokenizer_config.json",
                                   "special_tokens_map.json",
                                   "vocab.json",
                                   "merges.txt",
                                   "tokenizer.model",
                                   "chat_template.jinja"};
    std::error_code ec;
    for (const char* n : kNames) {
        const fs::path src = fs::path(opt.in_dir) / n;
        if (!fs::exists(src))
            continue;
        fs::copy_file(src, fs::path(opt.out_dir) / n, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            err = std::string("failed to copy ") + n + ": " + ec.message();
            return false;
        }
    }
    return true;
}

// A sharded checkpoint is only loadable with an index, and the index cannot be
// copied from the source: quantizing one weight turns it into three tensors, so
// the name->shard map has to be rebuilt from what was actually written.
bool write_shard_index(const Options& opt,
                       const std::vector<std::pair<std::string, std::string>>& tensor_to_shard,
                       size_t total_bytes, std::string& err) {
    std::ofstream f(fs::path(opt.out_dir) / "model.safetensors.index.json");
    if (!f) {
        err = "cannot write model.safetensors.index.json";
        return false;
    }
    f << "{\n  \"metadata\": { \"total_size\": " << total_bytes << " },\n  \"weight_map\": {\n";
    for (size_t i = 0; i < tensor_to_shard.size(); i++)
        f << "    \"" << tensor_to_shard[i].first << "\": \"" << tensor_to_shard[i].second << '"'
          << (i + 1 < tensor_to_shard.size() ? ",\n" : "\n");
    f << "  }\n}\n";
    return static_cast<bool>(f);
}

bool write_quant_config(const Options& opt, const std::vector<std::string>& excluded, bool calibrated,
                        std::string& err) {
    std::ofstream f(fs::path(opt.out_dir) / "hf_quant_config.json");
    if (!f) {
        err = "cannot write hf_quant_config.json";
        return false;
    }
    f << "{\n"
      << "  \"producer\": { \"name\": \"imp-quantize\", \"version\": \"2\", \"calibration\": \""
      << (calibrated ? "awq" : "none") << "\" },\n"
      << "  \"quantization\": {\n"
      << "    \"quant_algo\": \"NVFP4\",\n"
      << "    \"kv_cache_quant_algo\": null,\n"
      << "    \"group_size\": 16,\n"
      << "    \"exclude_modules\": [";
    for (size_t i = 0; i < excluded.size(); i++)
        f << (i ? ", " : "") << '"' << excluded[i] << '"';
    f << "]\n  }\n}\n";
    return static_cast<bool>(f);
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
        else if (a == "--quantize-attn-gate")
            opt.quantize_attn_gate = true;
        else if (a == "--calib")
            opt.calib_file = next();
        else if (a == "--calib-groups") {
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

    // Fused Q+gate projections stay full precision by default. NVFP4 in the gate
    // half is the measured root cause of #1273: rounding ONLY that half on a
    // healthy GGUF twin reproduces the real defect (+0.0169 divergence injected
    // per attention block against the +0.0156 the actual NVFP4 checkpoint
    // injects), while the Q half sits below the noise floor. Not a 4-bit
    // problem — the same half in Q4_K is healthy; E2M1 is coarsest near zero,
    // where a sigmoid is most sensitive.
    //
    // imp cannot exclude half a tensor, so the whole q_proj is kept. That costs
    // 230 MiB on Qwen3.6-35B-A3B and 552 MiB on the 27B (~1% and ~4%) against a
    // 2.08x-6x perplexity penalty, which is why it is the default rather than a
    // flag the user has to know about.
    std::set<std::string> gated_q_proj;
    {
        std::vector<const RawTensor*> gated;
        for (const auto& src : opened) {
            auto found = quantize::find_fused_gate_q_projections(src->tensors());
            gated.insert(gated.end(), found.begin(), found.end());
        }
        if (!gated.empty()) {
            size_t extra_bytes = 0;
            for (const RawTensor* t : gated) {
                // What keeping it full precision costs over quantizing it.
                const size_t nvfp4 = t->numel() / 2 + t->numel() / 16;
                extra_bytes += t->nbytes > nvfp4 ? t->nbytes - nvfp4 : 0;
                if (!opt.quantize_attn_gate)
                    gated_q_proj.insert(t->name);
            }
            printf("%s %zu fused Q+gate projection(s), %.0f MiB %s (#1273)\n",
                   opt.quantize_attn_gate ? "  QUANTIZING" : "  KEEPING", gated.size(),
                   double(extra_bytes) / (1024.0 * 1024.0),
                   opt.quantize_attn_gate ? "saved, at a 2.08x-6x perplexity penalty" : "larger");
            if (opt.quantize_attn_gate)
                fprintf(stderr,
                        "warning: --quantize-attn-gate puts the attention gate half in NVFP4, which\n"
                        "is the measured root cause of #1273. Expect 2.08x-6x worse perplexity.\n");
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

    size_t n_quantized = 0, n_copied = 0, n_moe_skipped = 0;
    size_t bytes_in = 0, bytes_out = 0;
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
            std::string why;
            // Checked before should_quantize, which sees one tensor and cannot
            // know a q_proj is gated — that needs the layer's o_proj too.
            const bool gated = gated_q_proj.count(t.name) != 0;
            if (gated)
                why = "fused Q+gate projection — NVFP4 in the gate half is #1273 "
                      "(use --quantize-attn-gate to include it anyway)";
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
                n_copied++;
                continue;
            }

            const int64_t N = t.shape[0], K = t.shape[1];
            if (opt.dry_run) {
                printf("  QUANT %-58s [%lld,%lld]\n", t.name.c_str(), (long long)N, (long long)K);
                n_quantized++;
                continue;
            }

            std::vector<uint16_t> h = awq::raw_to_fp16(t);
            awq_apply_matrix(h, N, K, plan_vec(plan.row_div, t.name), plan_vec(plan.col_scale, t.name));
            quant_store.emplace_back();
            Quantized& q = quant_store.back();
            if (!quantize_one(h, N, K, q, err)) {
                fprintf(stderr, "  %s: %s\n", t.name.c_str(), err.c_str());
                return 1;
            }
            scale_store.push_back(q.tensor_scale);

            const std::string base = t.name.substr(0, t.name.size() - strlen(".weight"));
            out.push_back({t.name, "U8", {N, K / 2}, q.packed.data(), q.packed.size()});
            out.push_back({base + ".weight_scale", "F8_E4M3", {N, K / 16}, q.micro.data(), q.micro.size()});
            out.push_back({base + ".weight_scale_2", "F32", {1}, &scale_store.back(), sizeof(float)});
            bytes_out += q.packed.size() + q.micro.size() + sizeof(float);
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
        if (!copy_aux_files(opt, err) ||
            !write_quant_config(opt, excluded_modules, !opt.calib_file.empty(), err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        // Single-shard checkpoints load off model.safetensors directly; sharded
        // ones need the index, and it must describe the tensors we wrote.
        if (shards.size() > 1 && !write_shard_index(opt, tensor_to_shard, bytes_out, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    }

    printf("\n%s: %zu tensors quantized, %zu copied", opt.dry_run ? "dry run" : "done", n_quantized,
           n_copied);
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
        printf("\n\nAWQ-calibrated: %d groups scaled, %d left at round-to-nearest.%s"
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
    if (!opt.dry_run)
        printf("\nsize: %.2f GiB -> %.2f GiB (%.2fx)", bytes_in / 1073741824.0, bytes_out / 1073741824.0,
               bytes_in ? double(bytes_in) / double(bytes_out) : 0.0);
    printf("\n");
    return 0;
}

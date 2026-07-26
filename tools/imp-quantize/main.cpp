// imp-quantize — turn a BF16/FP16 SafeTensors checkpoint into an NVFP4 one.
//
// STATUS: EXPERIMENTAL. The pipeline is verified end to end, but the
// quantization is uncalibrated and costs measurably more quality than a
// published export (numbers below). Intended for getting a model onto the
// NVFP4 path for evaluation or performance work — not for producing
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
// Scope of this version: DENSE models. MoE expert stacks are 3-D
// [n_experts, N, K] and need the per-expert variant (quantize_moe_to_nvfp4);
// they are reported and left unquantized rather than silently mangled.
//
// QUALITY, MEASURED — read before using this on anything you care about.
// The scales here are plain round-to-nearest over the weights (absmax per
// micro-block, absmax per tensor). There is NO activation calibration, so
// nothing protects the channels that matter most, and it costs:
//
//   Qwen3-0.6B  PPL 24.06 -> 30.10  (+25%)
//   Qwen3-1.7B  PPL 17.22 -> 20.43  (+19%)
//   (imp-cli --perplexity over tools/analysis/ppl_corpus_45k.txt = 13536
//    tokens, deterministic_gemm, 2026-07-26)
//
// Use a corpus of that size to judge this. The same pair measured over the
// 199-token ppl_corpus.txt reads +42% / +57% and appears to get WORSE with
// model size — both artifacts of too few tokens. On the real corpus the loss
// shrinks with size, which is the expected shape.
//
// Coherence is intact beyond perplexity: tools/analysis/degen_suite.py passes
// 41/41 against a server running the quantized 0.6B, including constrained
// json_schema decoding, forced tool calls and thinking channels.
//
// Still, a calibrated export (AWQ / SmoothQuant class, what Modelopt does)
// will beat this — prefer a published checkpoint when one exists. Useful
// today for: getting a model onto the NVFP4 path at all, and for performance
// work where the weights only need to be the right shape.

#include "core/tensor.h"
#include "model/safetensors_raw.h"
#include "model/safetensors_writer.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace imp;

namespace {

struct Options {
    std::string in_dir, out_dir;
    bool quantize_lm_head = false;  // imp has its own lm_head NVFP4 policy (#982)
    bool dry_run = false;
};

void usage() {
    printf(
        "usage: imp-quantize --model <safetensors-dir> --out <dir> [--lm-head] [--dry-run]\n"
        "\n"
        "EXPERIMENTAL: uncalibrated quantization, measurably below a published\n"
        "export. For evaluation and performance work, not for shipping weights.\n"
        "\n"
        "  --model DIR   source checkpoint (BF16/FP16 SafeTensors + config.json)\n"
        "  --out DIR     destination; created if missing\n"
        "  --lm-head     also quantize lm_head (default: excluded, imp applies its\n"
        "                own measured lm_head policy at runtime)\n"
        "  --dry-run     report what would be quantized, write nothing\n");
}

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

// BF16/F16 -> FP16 half bits on the host. Mirrors the upload path's conversion
// (weight_upload.cu): BF16 widens by zero-filling the low mantissa bits.
std::vector<uint16_t> to_fp16(const RawTensor& t) {
    const int64_t n = t.numel();
    std::vector<uint16_t> out(static_cast<size_t>(n));
    const auto* src = static_cast<const uint16_t*>(t.data);
    if (t.dtype == "F16") {
        std::memcpy(out.data(), src, static_cast<size_t>(n) * 2);
        return out;
    }
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        // float -> half, round-to-nearest-even via the compiler's own conversion.
        __half h = __float2half(f);
        std::memcpy(&out[static_cast<size_t>(i)], &h, 2);
    }
    return out;
}

// A weight qualifies when it is a 2-D linear matrix the runtime reads through
// the NVFP4 GEMM path. Everything else is copied through untouched: norms and
// biases are 1-D, embeddings stay full precision (quantizing them costs quality
// for no bandwidth win on the decode hot path), and K must be a multiple of 16
// because that is the micro-block size.
bool should_quantize(const RawTensor& t, const Options& opt, std::string& why_not) {
    if (!ends_with(t.name, ".weight")) {
        why_not = "not a .weight tensor";
        return false;
    }
    if (t.dtype != "BF16" && t.dtype != "F16") {
        why_not = "dtype " + t.dtype + " (already quantized or unsupported)";
        return false;
    }
    if (t.shape.size() == 3) {
        why_not = "3-D MoE expert stack — needs the per-expert path, not supported yet";
        return false;
    }
    if (t.shape.size() != 2) {
        why_not = std::to_string(t.shape.size()) + "-D";
        return false;
    }
    if (contains(t.name, "embed_tokens") || contains(t.name, "embed_positions")) {
        why_not = "embedding";
        return false;
    }
    if (contains(t.name, "norm")) {
        why_not = "norm";
        return false;
    }
    if (contains(t.name, "lm_head") && !opt.quantize_lm_head) {
        why_not = "lm_head (use --lm-head to include)";
        return false;
    }
    if (t.shape[1] % 16 != 0) {
        why_not = "K=" + std::to_string(t.shape[1]) + " is not a multiple of 16";
        return false;
    }
    return true;
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

bool write_quant_config(const Options& opt, const std::vector<std::string>& excluded, std::string& err) {
    std::ofstream f(fs::path(opt.out_dir) / "hf_quant_config.json");
    if (!f) {
        err = "cannot write hf_quant_config.json";
        return false;
    }
    f << "{\n"
      << "  \"producer\": { \"name\": \"imp-quantize\", \"version\": \"1\" },\n"
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
        else if (a == "--dry-run")
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

    size_t n_quantized = 0, n_copied = 0, n_moe_skipped = 0;
    size_t bytes_in = 0, bytes_out = 0;
    std::vector<std::string> excluded_modules;
    std::vector<std::pair<std::string, std::string>> tensor_to_shard;

    for (const auto& shard : shards) {
        RawSafeTensors src;
        std::string err = src.open(shard.string());
        if (!err.empty()) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        printf("[%s] %zu tensors\n", shard.filename().string().c_str(), src.tensors().size());

        // Owns the buffers the output descriptors point at until the shard is written.
        std::vector<Quantized> quant_store;
        std::vector<float> scale_store;
        quant_store.reserve(src.tensors().size());
        scale_store.reserve(src.tensors().size());
        std::vector<SafeTensorsOut> out;

        for (const auto& t : src.tensors()) {
            bytes_in += t.nbytes;
            std::string why;
            if (!should_quantize(t, opt, why)) {
                if (contains(why, "3-D MoE")) {
                    n_moe_skipped++;
                    printf("  SKIP  %-58s %s\n", t.name.c_str(), why.c_str());
                }
                if (ends_with(t.name, ".weight") && t.shape.size() >= 2) {
                    // Record real matrices we left alone so the runtime does not
                    // expect scales for them.
                    std::string mod = t.name.substr(0, t.name.size() - strlen(".weight"));
                    excluded_modules.push_back(mod);
                }
                out.push_back({t.name, t.dtype, t.shape, t.data, t.nbytes});
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

            std::vector<uint16_t> h = to_fp16(t);
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
        if (!copy_aux_files(opt, err) || !write_quant_config(opt, excluded_modules, err)) {
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
    if (!opt.dry_run)
        printf(
            "\n\nEXPERIMENTAL OUTPUT: round-to-nearest, no activation calibration.\n"
            "      Measured cost on the dense Qwen3 pair: PPL +25%% (0.6B) / +19%% (1.7B).\n"
            "      Fine for evaluation and performance work; prefer a published\n"
            "      calibrated checkpoint for anything you rely on.");
    if (n_moe_skipped)
        printf(", %zu MoE expert stacks left unquantized (not supported yet)", n_moe_skipped);
    if (!opt.dry_run)
        printf("\nsize: %.2f GiB -> %.2f GiB (%.2fx)", bytes_in / 1073741824.0, bytes_out / 1073741824.0,
               bytes_in ? double(bytes_in) / double(bytes_out) : 0.0);
    printf("\n");
    return 0;
}

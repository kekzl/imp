// ============================================================================
// GGUF loader.
//
// STATUS: LEGACY / MAINTENANCE MODE (2026-05-24).
//   GGUF is supported for compatibility but is no longer the active dev surface.
//   Priority is NVFP4 + SafeTensors (Qwen3.6-35B-A3B-NVFP4, Qwen3-8B-NVFP4-cortecs,
//   Gemma-4-26B-A4B-it-NVFP4, etc.) — that's where the hero-model perf work lives.
//
//   For GGUF bugs, ship the cleanup fix (load errors, missing pointer replaces,
//   resource cleanup cascades) and move on. Don't sink session time chasing
//   residual quality issues — especially on community MXFP4 quants, which have
//   a track record of being subtly broken (see qwen35_27b_mxfp4_ima_2026_04_25
//   and qwen35_4b_mxfp4_load_partial_fix_2026_05_24 memory notes). If the next
//   debug step requires comparing against an external reference engine
//   (llama.cpp / HF Transformers), declare done and defer.
// ============================================================================

#include "model/gguf_loader.h"
#include "model/gguf_loader_internal.h"
#include "model/gguf_half.h"
#include "model/loader_assign.h"
#include "model/model_arch.h"
#include "model/tensor_kind_matcher.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace imp {

// Host half/bf16 <-> float helpers for the gpt-oss 2^-4 residual rescale moved to
// model/gguf_half.h so they can be unit-tested on the CPU (see test_gguf_half.cpp).

// Format tables (gguf_blck_size / gguf_type_size / gguf_row_size /
// gguf_type_to_qtype / gguf_type_name), the BinaryReader / GGUFValue plumbing,
// metadata-value decoding, tensor-info parsing, and tensor bounds checks live
// in gguf_parse.cpp. Tensor → weight-slot assignment lives in
// gguf_tensor_assign.cpp. Shared declarations are in gguf_loader_internal.h.
// This TU keeps the top-level load orchestration only.

// ---- Main GGUF loader ----

std::unique_ptr<Model> load_gguf(const std::string& path) {
    // 1. Open and mmap the file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Failed to open GGUF file: %s", path.c_str());
        return nullptr;
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        IMP_LOG_ERROR("Failed to stat GGUF file: %s", path.c_str());
        close(fd);
        return nullptr;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (mmap_base == MAP_FAILED) {
        // Retry without MAP_POPULATE (some FS / mount-options reject it).
        mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);

    if (mmap_base == MAP_FAILED) {
        IMP_LOG_ERROR("Failed to mmap GGUF file: %s (size=%zu)", path.c_str(), file_size);
        return nullptr;
    }

    // Hint kernel: sequential read pattern + load pages now.
    madvise(mmap_base, file_size, MADV_WILLNEED);
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    BinaryReader reader(data, file_size);

    // 2. Parse header
    uint32_t magic = reader.read_u32();
    if (magic != GGUF_MAGIC) {
        IMP_LOG_ERROR("Invalid GGUF magic: 0x%08x", magic);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint32_t version = reader.read_u32();
    if (version < 2 || version > 3) {
        IMP_LOG_ERROR("Unsupported GGUF version: %u (expected 2 or 3)", version);
        munmap(mmap_base, file_size);
        return nullptr;
    }

    uint64_t tensor_count = reader.read_u64();
    uint64_t kv_count = reader.read_u64();

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF header truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    IMP_LOG_INFO("GGUF v%u: %lu tensors, %lu metadata KVs", version, (unsigned long)tensor_count,
                 (unsigned long)kv_count);

    // 3. Parse metadata key-value pairs
    std::unordered_map<std::string, GGUFValue> metadata;
    // Clamp the reserve to what the file could physically hold (each KV pair is
    // at least a 8-byte string-length prefix + 4-byte type tag = 12 bytes).
    // Without this, a corrupt kv_count=2^60 would reserve petabytes and OOM
    // before a single read ever fails.
    metadata.reserve(std::min<uint64_t>(kv_count, reader.remaining() / 12));

    for (uint64_t i = 0; i < kv_count && !reader.failed(); i++) {
        std::string key = reader.read_string();
        auto vtype = static_cast<GGUFValueType>(reader.read_u32());
        GGUFValue value = read_gguf_value(reader, vtype);
        metadata.emplace(std::move(key), std::move(value));
    }

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF metadata truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // 4. Parse tensor info entries
    std::vector<GGUFTensorInfo> tensor_infos;
    // Clamp like the metadata reserve above: a tensor-info entry is at least
    // name-len(8) + n_dims(4) + type(4) + offset(8) = 24 bytes, so a corrupt
    // tensor_count cannot drive an unbounded allocation.
    tensor_infos.reserve(std::min<uint64_t>(tensor_count, reader.remaining() / 24));
    parse_tensor_infos(reader, tensor_count, tensor_infos);

    if (reader.failed()) {
        IMP_LOG_ERROR("GGUF tensor info truncated: %s", path.c_str());
        munmap(mmap_base, file_size);
        return nullptr;
    }

    // 5. Compute tensor data start offset (aligned)
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    auto it_align = metadata.find("general.alignment");
    if (it_align != metadata.end()) {
        alignment = static_cast<size_t>(val_uint(it_align->second));
        if (alignment == 0)
            alignment = GGUF_DEFAULT_ALIGNMENT;
    }

    reader.align(alignment);
    size_t tensor_data_start = reader.pos();

    IMP_LOG_DEBUG("Tensor data starts at offset %zu (alignment=%zu)", tensor_data_start, alignment);

    // Set data_base for primary shard tensors. data_limit = bytes of mapped
    // file available past the tensor-data section, used to bounds-check each
    // tensor's [offset, offset+size) window before we hand out a raw pointer.
    size_t primary_data_avail = (tensor_data_start <= file_size) ? file_size - tensor_data_start : 0;
    for (auto& info : tensor_infos) {
        info.data_base = data + tensor_data_start;
        info.data_limit = primary_data_avail;
    }

    // 5b. Handle split GGUF files (multiple shards)
    auto it_split = metadata.find("split.count");
    int split_count = (it_split != metadata.end()) ? static_cast<int>(val_uint(it_split->second)) : 1;

    // Store extra shard mmaps for cleanup (primary shard stored separately in model)
    std::vector<std::pair<void*, size_t>> extra_mmaps;

    if (split_count > 1) {
        IMP_LOG_INFO("Split GGUF: %d shards", split_count);

        // Derive shard filenames: path ends with -00001-of-NNNNN.gguf
        // Replace the shard number for each additional shard
        const std::string& base_path = path;
        auto dash_pos = base_path.rfind("-00001-of-");
        if (dash_pos == std::string::npos) {
            // Try without the -00001 suffix (user passed the base name)
            IMP_LOG_WARN("Split GGUF: cannot derive shard paths from '%s'", path.c_str());
        } else {
            for (int shard = 2; shard <= split_count; shard++) {
                char shard_path[4096];
                snprintf(shard_path, sizeof(shard_path), "%.*s-%05d-of-%05d.gguf", static_cast<int>(dash_pos),
                         base_path.c_str(), shard, split_count);

                int sfd = open(shard_path, O_RDONLY);
                if (sfd < 0) {
                    IMP_LOG_ERROR("Failed to open shard %d: %s", shard, shard_path);
                    for (auto& [p, s] : extra_mmaps)
                        munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }

                struct stat sst {};
                fstat(sfd, &sst);
                size_t shard_size = static_cast<size_t>(sst.st_size);
                void* shard_mmap = mmap(nullptr, shard_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, sfd, 0);
                if (shard_mmap == MAP_FAILED) {
                    shard_mmap = mmap(nullptr, shard_size, PROT_READ, MAP_PRIVATE, sfd, 0);
                }
                close(sfd);

                if (shard_mmap == MAP_FAILED) {
                    IMP_LOG_ERROR("Failed to mmap shard %d: %s", shard, shard_path);
                    for (auto& [p, s] : extra_mmaps)
                        munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }
                madvise(shard_mmap, shard_size, MADV_WILLNEED);
                madvise(shard_mmap, shard_size, MADV_SEQUENTIAL);
                extra_mmaps.emplace_back(shard_mmap, shard_size);

                // Parse shard header to get tensor infos
                auto* sdata = reinterpret_cast<const uint8_t*>(shard_mmap);
                BinaryReader sreader(sdata, shard_size);
                uint32_t smagic = sreader.read_u32();
                sreader.read_u32();  // sversion (unused)
                uint64_t stensor_count = sreader.read_u64();
                uint64_t skv_count = sreader.read_u64();

                if (smagic != GGUF_MAGIC || sreader.failed()) {
                    IMP_LOG_ERROR("Invalid shard %d header", shard);
                    for (auto& [p, s] : extra_mmaps)
                        munmap(p, s);
                    munmap(mmap_base, file_size);
                    return nullptr;
                }

                // Skip shard metadata (we already have it from shard 0)
                for (uint64_t i = 0; i < skv_count && !sreader.failed(); i++) {
                    sreader.read_string();
                    auto vtype = static_cast<GGUFValueType>(sreader.read_u32());
                    read_gguf_value(sreader, vtype);
                }

                // Parse shard tensor infos
                parse_tensor_infos(sreader, stensor_count, tensor_infos);

                // Compute shard tensor data start
                size_t salign = alignment;  // use same alignment as primary
                sreader.align(salign);
                size_t shard_data_start = sreader.pos();

                // Set data_base for this shard's tensors
                size_t shard_tensor_start = tensor_infos.size() - static_cast<size_t>(stensor_count);
                size_t shard_data_avail = (shard_data_start <= shard_size) ? shard_size - shard_data_start : 0;
                for (size_t ti = shard_tensor_start; ti < tensor_infos.size(); ti++) {
                    tensor_infos[ti].data_base = sdata + shard_data_start;
                    tensor_infos[ti].data_limit = shard_data_avail;
                }

                IMP_LOG_INFO("  Shard %d: %lu tensors, %.1f MiB", shard, (unsigned long)stensor_count,
                             shard_size / (1024.0 * 1024.0));
            }
        }
    }

    // 6. Extract model config from metadata
    auto model = std::make_unique<Model>();
    model->source_path_ = path;
    model->mmap_base_ = mmap_base;
    model->mmap_size_ = file_size;
    model->split_mmaps_ = std::move(extra_mmaps);

    ModelConfig& cfg = model->config_;

    auto it_arch = metadata.find("general.architecture");
    std::string arch_str = (it_arch != metadata.end()) ? it_arch->second.str_val : "llama";
    cfg.arch = parse_model_arch(arch_str);

    // #818: encoder-only models (BERT-family embedders like nomic-bert) would
    // fall through to the generic-decoder path, load "successfully", report
    // healthy, and then hit a CUDA illegal memory access on the first request
    // (causal-LM prefill + sampling on a model with no LM head), poisoning the
    // CUDA context for the whole process. Fail loudly at load instead.
    // nomic-bert has a dedicated encoder path (#836): rotary positions,
    // post-LN, mean pooling — served by the encoder forward. Other encoder
    // archs (classic BERT/bge/e5: learned absolute positions + token-type
    // sequences + CLS pooling) stay rejected until implemented.
    if (is_encoder_only_arch(arch_str) && cfg.arch != ModelArch::NOMIC_BERT) {
        throw std::runtime_error("encoder-only architecture '" + arch_str +
                                 "' is not supported (imp runs causal decoder LMs; "
                                 "embedding encoders need pooling support — only "
                                 "nomic-bert is wired, #836)");
    }

    IMP_LOG_INFO("Architecture: %s -> %s", arch_str.c_str(), model_arch_name(cfg.arch));

    // Helper lambdas for metadata lookup with arch prefix
    auto get_uint = [&](const std::string& key, uint64_t def = 0) -> uint64_t {
        auto it = metadata.find(arch_str + "." + key);
        if (it != metadata.end())
            return val_uint(it->second);
        it = metadata.find(key);
        if (it != metadata.end())
            return val_uint(it->second);
        return def;
    };

    auto get_float = [&](const std::string& key, double def = 0.0) -> double {
        auto it = metadata.find(arch_str + "." + key);
        if (it != metadata.end())
            return val_float(it->second);
        it = metadata.find(key);
        if (it != metadata.end())
            return val_float(it->second);
        return def;
    };

    cfg.n_layers = static_cast<int>(get_uint("block_count"));
    cfg.d_model = static_cast<int>(get_uint("embedding_length"));
    cfg.d_ff = static_cast<int>(get_uint("feed_forward_length"));
    cfg.n_heads = static_cast<int>(get_uint("attention.head_count"));
    cfg.n_kv_heads = static_cast<int>(get_uint("attention.head_count_kv", cfg.n_heads));
    cfg.head_dim = static_cast<int>(get_uint("attention.key_length", 0));
    if (cfg.head_dim == 0 && cfg.n_heads > 0) {
        cfg.head_dim = cfg.d_model / cfg.n_heads;
    }
    cfg.max_seq_len = static_cast<int>(get_uint("context_length", 4096));
    cfg.vocab_size = static_cast<int>(get_uint("vocab_size", 0));
    cfg.rope_theta = static_cast<float>(get_float("rope.freq_base", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(get_float("attention.layer_norm_rms_epsilon", 1e-5));

    // RoPE frequency scaling (linear: divide frequencies by factor)
    cfg.rope_freq_scale = static_cast<float>(get_float("rope.scaling.factor", 1.0));
    // Fallback: try legacy key
    if (cfg.rope_freq_scale == 1.0f) {
        float legacy_scale = static_cast<float>(get_float("rope.scale_linear", 0.0));
        if (legacy_scale > 0.0f)
            cfg.rope_freq_scale = legacy_scale;
    }

    // YaRN / Dynamic NTK RoPE scaling
    {
        std::string rope_type_str;
        auto it = metadata.find(arch_str + ".rope.scaling.type");
        if (it == metadata.end())
            it = metadata.find("rope.scaling.type");
        if (it != metadata.end() && it->second.type == GGUFValueType::STRING)
            rope_type_str = it->second.str_val;

        cfg.rope_n_ctx_orig = static_cast<int>(get_uint("rope.scaling.original_context_length", 0));
        cfg.yarn_beta_fast = static_cast<float>(get_float("rope.scaling.yarn_beta_fast", 32.0));
        cfg.yarn_beta_slow = static_cast<float>(get_float("rope.scaling.yarn_beta_slow", 1.0));
        cfg.yarn_attn_factor = static_cast<float>(get_float("rope.scaling.yarn_attn_factor", 1.0));
        // Also try the generic attn_factor key
        if (cfg.yarn_attn_factor == 1.0f)
            cfg.yarn_attn_factor = static_cast<float>(get_float("rope.scaling.attn_factor", 1.0));

        float yarn_ext = static_cast<float>(get_float("rope.scaling.yarn_ext_factor", -1.0));
        if (rope_type_str == "yarn") {
            cfg.yarn_ext_factor = (yarn_ext < 0.0f) ? 1.0f : yarn_ext;
        } else {
            cfg.yarn_ext_factor = (yarn_ext < 0.0f) ? 0.0f : yarn_ext;
        }

        // LongRoPE per-dimension frequency scaling (Phi-4)
        if (rope_type_str == "longrope") {
            auto sf = metadata.find(arch_str + ".rope.scaling.short_factor");
            if (sf == metadata.end())
                sf = metadata.find("rope.scaling.short_factor");
            if (sf != metadata.end())
                cfg.rope_short_factor = sf->second.float_array;

            auto lf = metadata.find(arch_str + ".rope.scaling.long_factor");
            if (lf == metadata.end())
                lf = metadata.find("rope.scaling.long_factor");
            if (lf != metadata.end())
                cfg.rope_long_factor = lf->second.float_array;

            cfg.rope_scaling_orig_max_pos = static_cast<int>(
                get_uint("rope.scaling.original_max_position_embeddings", 0));

            IMP_LOG_INFO("LongRoPE: short_factor[%zu], long_factor[%zu], orig_max_pos=%d",
                         cfg.rope_short_factor.size(), cfg.rope_long_factor.size(),
                         cfg.rope_scaling_orig_max_pos);
        }

        // Compute mscale compensation (same as llama.cpp)
        if (cfg.yarn_ext_factor != 0.0f && cfg.rope_freq_scale > 1.0f) {
            float factor = cfg.rope_freq_scale;  // scaling factor
            float mscale = 1.0f + 0.1f * logf(factor);
            // Pre-compensate for the internal mscale that rope_yarn() also applies
            cfg.yarn_attn_factor *= mscale / (1.0f + 0.1f * logf(factor));
        }
    }

    // Gemma-specific: per-layer sliding window and local RoPE (metadata-dependent)
    // Note: embed_scale, ffn_activation, norm_placement are set by apply_arch_defaults().
    if (arch_str == "gemma" || arch_str == "gemma2" || arch_str == "gemma3") {
        // Per-layer sliding window pattern: every Nth layer is global (no window)
        // Gemma-3 uses pattern=6 (5 local + 1 global)
        cfg.sliding_window_pattern = static_cast<int>(get_uint("attention.sliding_window_pattern", 0));
        if (cfg.sliding_window_pattern == 0 && arch_str == "gemma3") {
            cfg.sliding_window_pattern = 6;  // Gemma-3 default: 5 local + 1 global
        }

        // Local RoPE theta (used for sliding window layers; global layers use rope_theta)
        cfg.rope_local_theta = static_cast<float>(get_float("rope.local.freq_base", 0.0));
        if (cfg.rope_local_theta == 0.0f && cfg.sliding_window_pattern > 0) {
            cfg.rope_local_theta = 10000.0f;  // Gemma-3 default local theta
        }
    }

    // Gemma 4: per-layer SWA pattern (array), SWA-specific head dims, RoPE base.
    if (arch_str == "gemma4") {
        // Per-layer SWA bool array: 1 = sliding-window attention, 0 = full/global attention.
        {
            auto it = metadata.find("gemma4.attention.sliding_window_pattern");
            if (it == metadata.end())
                it = metadata.find("attention.sliding_window_pattern");
            if (it != metadata.end() && !it->second.int_array.empty()) {
                cfg.swa_layers.reserve(it->second.int_array.size());
                for (auto v : it->second.int_array)
                    cfg.swa_layers.push_back(v ? 1 : 0);
            }
        }
        // Default: 5:1 SWA:full pattern (matches google/gemma-4-26B-A4B-it)
        if (cfg.swa_layers.empty()) {
            cfg.swa_layers.resize(cfg.n_layers, 0);
            for (int i = 0; i < cfg.n_layers; i++)
                cfg.swa_layers[i] = ((i % 6) == 5) ? 0 : 1;  // every 6th is full
        }

        // SWA-specific attention dims (full attention uses key_length/value_length)
        int key_len = static_cast<int>(get_uint("attention.key_length", 0));
        int val_len = static_cast<int>(get_uint("attention.value_length", 0));
        int key_len_swa = static_cast<int>(get_uint("attention.key_length_swa", key_len));
        int val_len_swa = static_cast<int>(get_uint("attention.value_length_swa", val_len));
        (void)val_len;
        (void)val_len_swa;  // V head_dim assumed == K head_dim

        cfg.sliding_window = static_cast<int>(get_uint("attention.sliding_window", 0));
        cfg.rope_local_theta = static_cast<float>(get_float("rope.freq_base_swa", 0.0));
        if (cfg.rope_local_theta == 0.0f)
            cfg.rope_local_theta = 10000.0f;
        cfg.rope_theta_swa = cfg.rope_local_theta;

        // Build per-layer head_dim and n_kv_heads from swa_layers.
        // The GGUF may already supply per-layer arrays for head_count_kv; if not,
        // we derive from swa_layers using key_length / key_length_swa.
        if (cfg.head_dim_per_layer.empty() && key_len > 0 && key_len_swa > 0) {
            cfg.head_dim_per_layer.resize(cfg.n_layers);
            for (int i = 0; i < cfg.n_layers; i++)
                cfg.head_dim_per_layer[i] = cfg.swa_layers[i] ? key_len_swa : key_len;
        }
        // scalar head_dim = max for buffer sizing
        if (!cfg.head_dim_per_layer.empty()) {
            int max_hd = 0;
            for (int v : cfg.head_dim_per_layer)
                max_hd = std::max(max_hd, v);
            cfg.head_dim = max_hd;
            IMP_LOG_INFO("Gemma 4 per-layer head_dim: max=%d", max_hd);
        }

        IMP_LOG_INFO("Gemma 4: SWA layers=%zu (of %d), rope_theta_swa=%.0f, key_len=%d, key_len_swa=%d",
                     std::count(cfg.swa_layers.begin(), cfg.swa_layers.end(), uint8_t(1)), cfg.n_layers,
                     cfg.rope_theta_swa, key_len, key_len_swa);
        // Per-layer head_dim/n_kv_heads detection happens at runtime in run_attention
        // by reading wq.shape[0] / hd and wk.shape[0] / hd. Authoritative source =
        // the loaded tensor shapes, not GGUF metadata.
    }

    // gpt-oss: alternating attention — even layers use sliding-window (128),
    // odd layers use full attention. HF encodes this via layer_types[]; the
    // GGUF omits the per-layer array (only a scalar attention.sliding_window),
    // so derive the documented gpt-oss pattern here. Without swa_layers the
    // ModelProfile resolves attn=standard and every layer runs full attention
    // → wrong output (PPL ~3000 instead of ~4.7).
    if (cfg.arch == ModelArch::GPT_OSS) {
        if (cfg.sliding_window <= 0)
            cfg.sliding_window = static_cast<int>(get_uint("attention.sliding_window", 128));
        if (cfg.sliding_window <= 0)
            cfg.sliding_window = 128;
        if (cfg.swa_layers.empty()) {
            cfg.swa_layers.resize(cfg.n_layers, 0);
            for (int i = 0; i < cfg.n_layers; i++)
                cfg.swa_layers[i] = ((i % 2) == 0) ? 1 : 0;  // even = sliding_attention
        }
        IMP_LOG_INFO("gpt-oss: SWA pattern derived — %zu/%d sliding (even layers, window=%d), rest full",
                     std::count(cfg.swa_layers.begin(), cfg.swa_layers.end(), uint8_t(1)), cfg.n_layers,
                     cfg.sliding_window);
    }

    // Attention logit softcapping (Gemma-2/3: tanh(score/cap)*cap)
    cfg.attn_logit_softcap = static_cast<float>(get_float("attn_logit_softcapping", 0.0));
    if (cfg.attn_logit_softcap == 0.0f)
        cfg.attn_logit_softcap = static_cast<float>(get_float("attention.logit_softcapping", 0.0));
    cfg.final_logit_softcap = static_cast<float>(get_float("final_logit_softcapping", 0.0));

    // MXFP4 Hadamard rotation metadata
    cfg.mxfp4_hadamard_attn = static_cast<int>(get_uint("mxfp4.hadamard_block_size_attn", 0));
    cfg.mxfp4_hadamard_ffn = static_cast<int>(get_uint("mxfp4.hadamard_block_size_ffn", 0));
    if (cfg.mxfp4_hadamard_attn > 0 || cfg.mxfp4_hadamard_ffn > 0)
        IMP_LOG_INFO("MXFP4 Hadamard: attn_bs=%d ffn_bs=%d", cfg.mxfp4_hadamard_attn, cfg.mxfp4_hadamard_ffn);

    cfg.sliding_window = static_cast<int>(get_uint("attention.sliding_window", 0));

    cfg.n_experts = static_cast<int>(get_uint("expert_count", 0));
    cfg.n_experts_active = static_cast<int>(get_uint("expert_used_count", 0));
    cfg.expert_d_ff = static_cast<int>(get_uint("expert_feed_forward_length", cfg.d_ff));

    // Per-layer arrays (Nemotron hybrid: head_count_kv and feed_forward_length are arrays)
    {
        auto get_int_array = [&](const std::string& key) -> std::vector<int> {
            auto it = metadata.find(arch_str + "." + key);
            if (it == metadata.end())
                it = metadata.find(key);
            if (it == metadata.end() || it->second.int_array.empty())
                return {};
            std::vector<int> result;
            result.reserve(it->second.int_array.size());
            for (auto v : it->second.int_array)
                result.push_back(static_cast<int>(v));
            return result;
        };

        cfg.n_kv_heads_per_layer = get_int_array("attention.head_count_kv");
        cfg.d_ff_per_layer = get_int_array("feed_forward_length");

        // If we got per-layer arrays, set the scalar config to max values (for buffer sizing)
        if (!cfg.n_kv_heads_per_layer.empty()) {
            int max_kv = 0;
            for (int v : cfg.n_kv_heads_per_layer)
                max_kv = std::max(max_kv, v);
            cfg.n_kv_heads = max_kv;
            IMP_LOG_INFO("Per-layer KV heads: %zu layers, max=%d", cfg.n_kv_heads_per_layer.size(), max_kv);
        }
        if (!cfg.d_ff_per_layer.empty()) {
            int max_ff = 0;
            for (int v : cfg.d_ff_per_layer)
                max_ff = std::max(max_ff, v);
            cfg.d_ff = max_ff;
            IMP_LOG_INFO("Per-layer d_ff: %zu layers, max=%d", cfg.d_ff_per_layer.size(), max_ff);
        }
    }

    // Mamba2 SSM config
    cfg.ssm_conv_kernel = static_cast<int>(get_uint("ssm.conv_kernel", 0));
    cfg.ssm_state_size = static_cast<int>(get_uint("ssm.state_size", 0));
    cfg.ssm_group_count = static_cast<int>(get_uint("ssm.group_count", 0));
    cfg.ssm_inner_size = static_cast<int>(get_uint("ssm.inner_size", 0));
    cfg.ssm_dt_rank = static_cast<int>(get_uint("ssm.time_step_rank", 0));

    // Partial RoPE
    cfg.rope_dim = static_cast<int>(get_uint("rope.dimension_count", 0));

    // Extended MoE config
    cfg.n_experts_shared = static_cast<int>(get_uint("expert_shared_count", 0));
    cfg.expert_shared_d_ff = static_cast<int>(get_uint("expert_shared_feed_forward_length", 0));
    cfg.expert_weights_scale = static_cast<float>(get_float("expert_weights_scale", 1.0));
    cfg.expert_weights_norm = (get_uint("expert_weights_norm", 0) != 0);
    // Apply arch-specific config defaults (e.g. sigmoid gating for Nemotron)
    apply_arch_defaults(cfg);

    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d head_dim=%d vocab=%d ctx=%d",
                 cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                 cfg.vocab_size, cfg.max_seq_len);
    IMP_LOG_INFO("RoPE: theta=%.1f, rope_dim=%d, neox=%d, freq_scale=%.1f, eps=%.2e", cfg.rope_theta,
                 cfg.rope_dim, cfg.rope_neox ? 1 : 0, cfg.rope_freq_scale, cfg.rms_norm_eps);
    if (cfg.yarn_ext_factor > 0.0f)
        IMP_LOG_INFO("YaRN: ext_factor=%.1f, attn_factor=%.3f, beta_fast=%.1f, beta_slow=%.1f, n_ctx_orig=%d",
                     cfg.yarn_ext_factor, cfg.yarn_attn_factor, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                     cfg.rope_n_ctx_orig);
    if (cfg.embed_scale > 0.0f)
        IMP_LOG_INFO("Embedding scale: %.2f (sqrt(d_model))", cfg.embed_scale);

    if (cfg.sliding_window > 0) {
        IMP_LOG_INFO("Sliding window attention: %d tokens", cfg.sliding_window);
        if (cfg.sliding_window_pattern > 0) {
            IMP_LOG_INFO("Sliding window pattern: every %dth layer is global, local_theta=%.1f",
                         cfg.sliding_window_pattern, cfg.rope_local_theta);
        }
    }
    if (cfg.ffn_activation != FFNActivation::SWIGLU) {
        const char* act_name = (cfg.ffn_activation == FFNActivation::GEGLU) ? "GeGLU" : "ReLU²";
        IMP_LOG_INFO("FFN activation: %s", act_name);
    }
    if (cfg.norm_placement == NormPlacement::POST_NORM)
        IMP_LOG_INFO("Norm placement: post-norm (residual after norm)");
    if (cfg.attn_logit_softcap > 0.0f)
        IMP_LOG_INFO("Attention logit softcap: %.1f", cfg.attn_logit_softcap);
    if (cfg.final_logit_softcap > 0.0f)
        IMP_LOG_INFO("Final logit softcap: %.1f", cfg.final_logit_softcap);

    if (cfg.n_experts > 0) {
        IMP_LOG_INFO(
            "MoE: %d experts, %d active, expert_d_ff=%d, shared=%d (shared_d_ff=%d), "
            "norm_weights=%d",
            cfg.n_experts, cfg.n_experts_active, cfg.expert_d_ff, cfg.n_experts_shared,
            cfg.expert_shared_d_ff, cfg.expert_weights_norm ? 1 : 0);
    }

    if (cfg.ssm_inner_size > 0) {
        IMP_LOG_INFO("SSM: conv_kernel=%d state_size=%d groups=%d inner=%d dt_rank=%d", cfg.ssm_conv_kernel,
                     cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_inner_size, cfg.ssm_dt_rank);
    }

    if (cfg.rope_dim > 0) {
        IMP_LOG_INFO("Partial RoPE: rope_dim=%d (full head_dim=%d)", cfg.rope_dim, cfg.head_dim);
    }

    // 7. Allocate layers and assign weights
    model->layers_.resize(cfg.n_layers);

    if (cfg.n_experts > 0) {
        for (auto& layer : model->layers_) {
            layer.expert_w_gate.resize(cfg.n_experts);
            layer.expert_w_up.resize(cfg.n_experts);
            layer.expert_w_down.resize(cfg.n_experts);
        }
    }

    int assigned = 0, skipped = 0;

    for (const auto& info : tensor_infos) {
        // Reject tensors whose [offset, offset+size) window escapes the mapped
        // file before we ever form a pointer into it. A corrupt offset or a
        // dim product that overflows would otherwise yield a wild pointer that
        // weight_upload later reads — an out-of-bounds read / crash on a
        // malformed file. Skipping leaves the slot null; downstream load fails
        // cleanly (missing-weight path) instead of faulting.
        if (!gguf_tensor_in_bounds(info)) {
            IMP_LOG_ERROR(
                "GGUF tensor '%s' out of bounds (offset=%lu, limit=%zu) — skipping; file is corrupt",
                info.name.c_str(), (unsigned long)info.offset, info.data_limit);
            skipped++;
            continue;
        }

        // Compute pointer into mmap'd data (supports split GGUF via per-tensor data_base)
        auto* tensor_data = const_cast<void*>(static_cast<const void*>(info.data_base + info.offset));

        // Build tensor descriptor
        // GGUF stores dims as ne[0]=innermost. We reverse for shape[0]=outermost.
        int ndim = static_cast<int>(info.n_dims);
        int64_t shape[4] = {1, 1, 1, 1};
        for (int d = 0; d < ndim; d++) {
            shape[d] = info.dims[ndim - 1 - d];
        }

        Tensor t(tensor_data, gguf_type_to_qtype(info.type), ndim, shape, /*on_device=*/false);
        t.kind = match_tensor_kind(info.name);
        if (info.type == GgufWireType::MXFP4_V2) {
            t.mxfp4_layout_v2 = true;
        }

        if (assign_tensor(*model, info.name, t, info.type)) {
            assigned++;
        } else {
            IMP_LOG_DEBUG("Unassigned tensor: %s [%s] shape=[%ld,%ld,%ld,%ld]", info.name.c_str(),
                          gguf_type_name(info.type), (long)info.dims[0], (long)info.dims[1],
                          (long)info.dims[2], (long)info.dims[3]);
            skipped++;
        }
    }

    // Infer vocab_size from token_embd if not in metadata
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
        IMP_LOG_INFO("Inferred vocab_size=%d from token_embd.weight", cfg.vocab_size);
    }

    // Weight tying: if no output.weight, share token_embd
    if (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr) {
        model->out_proj_ = model->tok_emb_;
        model->out_proj_.qtype = model->tok_emb_.qtype;
        IMP_LOG_INFO("Weight tying: output projection shares token embedding");
    }

    // Split fused gate+up FFN (Phi-4/phi3): ffn_up contains gate||up concatenated
    // Detected when: w_gate is null, w_up exists, and w_up.shape[0] == 2 * d_ff
    if (cfg.d_ff > 0) {
        int fused_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ly = model->layers_[i];
            if (ly.w_gate.data == nullptr && ly.w_up.data != nullptr && ly.w_up.shape[0] == static_cast<int64_t>(2) * cfg.d_ff) {
                int64_t d_model = ly.w_up.shape[1];
                int64_t d_ff = cfg.d_ff;
                size_t row_bytes = qtype_row_bytes(ly.w_up.qtype, d_model);

                uint8_t* base = static_cast<uint8_t*>(ly.w_up.data);
                int64_t half_shape[4] = {d_ff, d_model, 1, 1};

                ly.w_gate = Tensor(base, ly.w_up.qtype, 2, half_shape, ly.w_up.on_device);
                ly.w_gate.qtype = ly.w_up.qtype;
                ly.w_up = Tensor(base + static_cast<size_t>(d_ff) * row_bytes, ly.w_up.qtype, 2, half_shape,
                                 ly.w_up.on_device);
                // w_up_qtype unchanged
                fused_count++;
            }
        }
        if (fused_count > 0) {
            IMP_LOG_INFO("Split fused gate+up FFN in %d layers (d_ff=%d)", fused_count, cfg.d_ff);
        }
    }

    IMP_LOG_INFO("Weights: %d assigned, %d skipped", assigned, skipped);

    // 7b. Tensor validation and shared expert detection
    //     Inspect actual loaded tensors to detect capabilities, remap shared
    //     experts stored as regular FFN tensors, and warn about mismatches.
    {
        int n_attn = 0, n_moe = 0, n_dense_ffn = 0, n_shared_exp = 0;
        int n_qk_norm = 0, n_ssm = 0, n_gdn = 0, n_remapped = 0;

        for (int i = 0; i < cfg.n_layers; i++) {
            auto& ly = model->layers_[i];
            bool has_moe = (ly.moe_gate.data != nullptr);
            bool has_dense = (ly.w_up.data != nullptr);
            bool has_shared = (ly.w_up_shared.data != nullptr);

            if (ly.wq.data != nullptr)
                n_attn++;
            if (has_moe)
                n_moe++;
            if (ly.attn_q_norm.data != nullptr)
                n_qk_norm++;
            if (ly.ssm_in.data != nullptr)
                n_ssm++;
            if (ly.gdn_gate.data != nullptr)
                n_gdn++;

            // Detect shared expert: MoE layer with dense FFN tensors loaded
            // alongside expert tensors → remap dense FFN to shared expert.
            // Some GGUF converters output shared experts as ffn_gate/ffn_up/ffn_down.
            if (has_moe && has_dense && !has_shared) {
                ly.w_gate_shared = ly.w_gate;
                ly.w_gate_shared.qtype = ly.w_gate.qtype;
                ly.w_up_shared = ly.w_up;
                ly.w_up_shared.qtype = ly.w_up.qtype;
                ly.w_down_shared = ly.w_down;
                ly.w_down_shared.qtype = ly.w_down.qtype;
                ly.w_gate = Tensor();
                ly.w_gate.qtype = QType::NONE;
                ly.w_up = Tensor();
                ly.w_up.qtype = QType::NONE;
                ly.w_down = Tensor();
                ly.w_down.qtype = QType::NONE;
                n_remapped++;
                has_shared = true;
            }

            if (has_shared)
                n_shared_exp++;
            if (has_dense && !has_moe)
                n_dense_ffn++;
        }

        IMP_LOG_INFO(
            "Layer census: %d attn, %d GDN, %d MoE, %d dense FFN, %d shared expert, "
            "%d QK-norm, %d SSM  (of %d layers)",
            n_attn, n_gdn, n_moe, n_dense_ffn, n_shared_exp, n_qk_norm, n_ssm, cfg.n_layers);

        if (n_remapped > 0) {
            IMP_LOG_INFO("Remapped %d layers: dense FFN tensors -> shared expert", n_remapped);
        }
        // Gemma 4: verify MoE-specific norms and router scale loaded
        if (cfg.arch == ModelArch::GEMMA4) {
            int n_pre2 = 0, n_post1 = 0, n_post2 = 0, n_gscale = 0, n_dscale = 0;
            for (int i = 0; i < cfg.n_layers; ++i) {
                if (model->layers_[i].ffn_pre_norm_2.data)
                    n_pre2++;
                if (model->layers_[i].ffn_post_norm_1.data)
                    n_post1++;
                if (model->layers_[i].ffn_post_norm_2.data)
                    n_post2++;
                if (model->layers_[i].ffn_gate_inp_scale.data)
                    n_gscale++;
                if (model->layers_[i].expert_down_scale.data)
                    n_dscale++;
            }
            IMP_LOG_INFO(
                "Gemma 4 MoE norms: pre_ffw_norm_2=%d, post_ffw_norm_1=%d, "
                "post_ffw_norm_2=%d, gate_inp_scale=%d, down_exps_scale=%d (of %d layers)",
                n_pre2, n_post1, n_post2, n_gscale, n_dscale, cfg.n_layers);
        }

        // Qwen3.5: GGUF converter adds +1 to non-GDN norm weights.
        // imp's RMSNorm expects raw weights (w, not w+1). Subtract 1 back.
        // Qwen3.5: GGUF converter adds +1 to norm weights (same as Gemma).
        // imp's rmsnorm uses weight directly, which is correct since the +1 is
        // already baked into the stored weights. No adjustment needed.

        // Update config from actual tensor presence
        if (n_shared_exp > 0 && cfg.n_experts_shared == 0) {
            cfg.n_experts_shared = 1;
            for (int i = 0; i < cfg.n_layers; i++) {
                if (model->layers_[i].w_up_shared.data != nullptr) {
                    cfg.expert_shared_d_ff = static_cast<int>(model->layers_[i].w_up_shared.shape[0]);
                    break;
                }
            }
            IMP_LOG_INFO("Inferred shared expert config: n_shared=%d, shared_d_ff=%d", cfg.n_experts_shared,
                         cfg.expert_shared_d_ff);
        }

        // Gemma 4: convert top-level rope_freqs (a freq DIVISOR table for global
        // layers) into pre-computed effective per-pair frequencies, then fan out
        // to every global layer. The kernel's `longrope_inv_freqs` parameter
        // expects ready-to-use freq values, so do the math on the host.
        if (cfg.arch == ModelArch::GEMMA4 && !cfg.swa_layers.empty() &&
            model->layers_[0].rope_freqs.data != nullptr &&
            model->layers_[0].rope_freqs.qtype == QType::F32) {
            const Tensor& src = model->layers_[0].rope_freqs;
            int n_pairs = static_cast<int>(src.shape[0]);  // hd/2 for global layer
            int hd_global = n_pairs * 2;
            const float* divisors = static_cast<const float*>(src.data);
            float theta_global = cfg.rope_theta;  // 1e6 for Gemma 4

            // Pre-compute effective per-pair frequencies = theta^(-2*pair/hd)/divisor[pair]
            // and present them via the layer's rope_freqs slot. The kernel reads
            // these directly as the freq value (longrope_inv_freqs path), no further
            // theta math. Memory is leaked deliberately (4 KB total, model-lifetime).
            float* effective = new float[n_pairs];
            for (int p = 0; p < n_pairs; ++p) {
                float exp_p = -2.0f * static_cast<float>(p) / static_cast<float>(hd_global);
                float base_freq = std::pow(theta_global, exp_p);
                effective[p] = base_freq / divisors[p];
            }
            int64_t shape[4] = {n_pairs, 0, 0, 0};
            Tensor eff_tensor(effective, QType::F32, 1, shape, /*on_device=*/false);
            int n_global = 0;
            for (int i = 0; i < cfg.n_layers; ++i) {
                bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i]);
                if (!is_swa) {
                    model->layers_[i].rope_freqs = eff_tensor;
                    n_global++;
                }
            }
            if (cfg.swa_layers[0]) {
                model->layers_[0].rope_freqs = Tensor();
            }
            IMP_LOG_INFO("Gemma 4: rope_freqs → %d effective freqs, %d global layers", n_pairs, n_global);
        }

        // Warn about config/tensor mismatches
        if (cfg.n_experts_shared > 0 && n_shared_exp == 0) {
            IMP_LOG_WARN(
                "Config declares %d shared expert(s) but no shared expert "
                "tensors found — GGUF may be incomplete",
                cfg.n_experts_shared);
        }

        if (cfg.n_experts > 0 && n_moe == 0) {
            IMP_LOG_WARN("Config declares %d experts but no MoE gate tensors found", cfg.n_experts);
        }

        if (n_moe > 0 && n_moe < cfg.n_layers && n_dense_ffn == 0 && n_ssm == 0) {
            IMP_LOG_WARN("Only %d/%d layers have MoE, remaining layers have no FFN", n_moe, cfg.n_layers);
        }
    }

    // gpt-oss residual-stream 2^-4 rescale (#547, GGUF parity with the
    // SafeTensors loader). gpt-oss's huge activations overflow imp's FP16
    // hidden state (hidden L2 reaches ±inf by ~L23 → NaN logits / garbage
    // decode). Scaling every contributor to the residual stream by 2^-4 is
    // exact for the model output: the FP16/RMSNorm path is scale-invariant and
    // the lm_head reads only normed values. Contributors handled elsewhere:
    //   - embeddings: cfg.embed_scale = 2^-4 (arch registry)
    //   - expert down weights: tensor_scales in the MXFP4→NVFP4 converter
    //     (pre_dequant_phase3_nvfp4_decode.cu)
    // Handled here, host-side (GGUF is mmap'd read-only, so scale into fresh
    // host_owned_buffers_): attention output Wo + o_bias, and expert down bias.
    if (cfg.arch == ModelArch::GPT_OSS) {
        // The GGUF tensor `blk.N.post_attention_norm.weight` is gpt-oss's
        // PRE-FFN norm (llama convention: post_attention_layernorm gates the
        // FFN/MoE input — see weight_map.cpp's SafeTensors mapping → ffn_norm).
        // The generic 4-part handler routed it to post_attn_norm (the Gemma-3
        // sandwich-norm slot), so the MoE ran on the UN-normalized residual →
        // router logits ~10x too large → wrong expert selection → garbage.
        // Move it to ffn_norm (gpt-oss has no sandwich norm).
        for (auto& ly : model->layers_) {
            if (!ly.ffn_norm.data && ly.post_attn_norm.data) {
                ly.ffn_norm = ly.post_attn_norm;
                ly.post_attn_norm = Tensor();
            }
        }

        // ×2^-4 helpers for each dtype a gpt-oss residual contributor (Wo, o_bias,
        // expert down bias) can appear in across GGUF quants. All scale in the float
        // domain (gguf_*_to_float * 0.0625, then back) — exact for normals, correct
        // for denormals/underflow. (The earlier exponent-bit-subtract was wrong for
        // small fp16 block scales — see the helper comment near the top of the file.)
        // For Q8_0 only the per-block fp16 d is scaled; the int8 quants are untouched.
        // All write into fresh host_owned_buffers_ (the GGUF mmap is read-only).
        auto scale_f32 = [&](Tensor& t) -> bool {
            int64_t n = t.numel();
            float* dst = static_cast<float*>(std::malloc(sizeof(float) * n));
            if (!dst)
                return false;
            const float* src = static_cast<const float*>(t.data);
            for (int64_t i = 0; i < n; i++)
                dst[i] = src[i] * 0.0625f;
            model->host_owned_buffers_.push_back(dst);
            t.data = dst;
            return true;
        };
        auto scale_f16 = [&](Tensor& t) -> bool {
            int64_t n = t.numel();
            uint16_t* dst = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * n));
            if (!dst)
                return false;
            const uint16_t* src = static_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; i++)
                dst[i] = gguf_float_to_half(gguf_half_to_float(src[i]) * 0.0625f);
            model->host_owned_buffers_.push_back(dst);
            t.data = dst;
            return true;
        };
        auto scale_bf16 = [&](Tensor& t) -> bool {
            int64_t n = t.numel();
            uint16_t* dst = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * n));
            if (!dst)
                return false;
            const uint16_t* src = static_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; i++)
                dst[i] = gguf_float_to_bf16(gguf_bf16_to_float(src[i]) * 0.0625f);
            model->host_owned_buffers_.push_back(dst);
            t.data = dst;
            return true;
        };
        auto scale_q8_0 = [&](Tensor& t) -> bool {
            int64_t n = t.numel();
            if (n % 32 != 0) {
                IMP_LOG_ERROR("gpt-oss GGUF rescale: Q8_0 numel %lld not 32-block-aligned", (long long)n);
                return false;
            }
            int64_t nblocks = n / 32;
            size_t bytes = static_cast<size_t>(nblocks) * 34;  // [fp16 d | 32 int8]
            uint8_t* dst = static_cast<uint8_t*>(std::malloc(bytes));
            if (!dst)
                return false;
            std::memcpy(dst, t.data, bytes);
            for (int64_t b = 0; b < nblocks; b++) {
                uint16_t* d = reinterpret_cast<uint16_t*>(dst + static_cast<size_t>(b) * 34);
                *d = gguf_float_to_half(gguf_half_to_float(*d) * 0.0625f);
            }
            model->host_owned_buffers_.push_back(dst);
            t.data = dst;
            return true;
        };
        auto scale_any = [&](Tensor& t) -> bool {
            if (!t.data || t.on_device)
                return true;
            switch (t.qtype) {
                case QType::F32:
                    return scale_f32(t);
                case QType::F16:
                    return scale_f16(t);
                case QType::BF16:
                    return scale_bf16(t);
                case QType::Q8_0:
                    return scale_q8_0(t);
                default:
                    IMP_LOG_ERROR("gpt-oss GGUF rescale: unsupported qtype %d", std::to_underlying(t.qtype));
                    return false;
            }
        };
        bool ok = true;
        for (auto& ly : model->layers_)
            ok = ok && scale_any(ly.wo) && scale_any(ly.o_bias) && scale_any(ly.expert_down_bias);
        if (!ok) {
            IMP_LOG_ERROR("gpt-oss GGUF: residual-stream rescale failed");
            return nullptr;
        }
        IMP_LOG_INFO("gpt-oss GGUF: residual stream rescaled by 2^-4 (Wo + o_bias + expert down bias)");
    }

    // 8. Extract tokenizer from GGUF metadata
    auto tokenizer = std::make_unique<Tokenizer>();

    // Detect tokenizer type (default: SentencePiece)
    auto it_tok_model = metadata.find("tokenizer.ggml.model");
    std::string tok_type = "spm";
    if (it_tok_model != metadata.end()) {
        const std::string& tm = it_tok_model->second.str_val;
        if (tm == "gpt2")
            tok_type = "gpt2";
        // Gemma-4 uses SPM-style BPE: ▁ for spaces + BPE merge ranks.
        else if (tm == "gemma4")
            tok_type = "gemma4";
        // BERT WordPiece (#836, nomic-bert embedder).
        else if (tm == "bert")
            tok_type = "bert";
    }
    tokenizer->set_type(tok_type);

    // Pre-tokenizer type (e.g. "default", "llama3", "deepseek-llm", "qwen2")
    auto it_pre = metadata.find("tokenizer.ggml.pre");
    if (it_pre != metadata.end() && !it_pre->second.str_val.empty()) {
        tokenizer->set_pre_tokenizer(it_pre->second.str_val);
        IMP_LOG_INFO("Tokenizer pre-tokenizer: %s", it_pre->second.str_val.c_str());
    }

    // add_bos_token flag (Qwen3: 0, LLaMA: 1)
    auto it_add_bos = metadata.find("tokenizer.ggml.add_bos_token");
    if (it_add_bos != metadata.end()) {
        tokenizer->set_add_bos(val_uint(it_add_bos->second) != 0);
    } else if (tok_type == "gpt2") {
        // GPT2/BPE tokenizers (Qwen, etc.) typically don't use BOS.
        // Default to false when metadata is absent.
        tokenizer->set_add_bos(false);
    }

    // Gemma-4: always add BOS regardless of GGUF metadata.
    // Some GGUF converters (ggml-org) set add_bos=false incorrectly.
    // llama.cpp forces add_bos=true for Gemma-4 (see llama-vocab.cpp "override").
    if (tok_type == "gemma4") {
        tokenizer->set_add_bos(true);
    }

    // add_space_prefix flag (Gemma: false, LLaMA: true/default)
    auto it_add_sp = metadata.find("tokenizer.ggml.add_space_prefix");
    if (it_add_sp != metadata.end()) {
        tokenizer->set_add_space_prefix(val_uint(it_add_sp->second) != 0);
    }

    auto it_tokens = metadata.find("tokenizer.ggml.tokens");
    if (it_tokens != metadata.end() && !it_tokens->second.str_array.empty()) {
        const auto& tokens = it_tokens->second.str_array;

        // Scores (optional, used for SentencePiece BPE merge priority)
        std::vector<float> scores;
        auto it_scores = metadata.find("tokenizer.ggml.scores");
        if (it_scores != metadata.end()) {
            scores = it_scores->second.float_array;
        }
        scores.resize(tokens.size(), 0.0f);

        // Special token IDs
        int bos_id = 1, eos_id = 2;
        auto it_bos = metadata.find("tokenizer.ggml.bos_token_id");
        if (it_bos != metadata.end())
            bos_id = static_cast<int>(val_uint(it_bos->second));
        auto it_eos = metadata.find("tokenizer.ggml.eos_token_id");
        if (it_eos != metadata.end())
            eos_id = static_cast<int>(val_uint(it_eos->second));

        tokenizer->load_vocab(tokens, scores, bos_id, eos_id);

        // Load BPE merge rules (for GPT2-style tokenizers and gemma4)
        if (tok_type == "gpt2" || tok_type == "gemma4") {
            auto it_merges = metadata.find("tokenizer.ggml.merges");
            if (it_merges != metadata.end() && !it_merges->second.str_array.empty()) {
                tokenizer->load_merges(it_merges->second.str_array);
                IMP_LOG_INFO("Tokenizer: loaded %zu BPE merge rules", it_merges->second.str_array.size());
            }
        }

        // Load per-token type metadata (NORMAL=1, CONTROL=3, etc.)
        auto it_types = metadata.find("tokenizer.ggml.token_type");
        if (it_types != metadata.end() && !it_types->second.int_array.empty()) {
            tokenizer->load_token_types(it_types->second.int_array);
        }

        // Extract chat template string (Jinja2) for template family detection
        auto it_tpl = metadata.find("tokenizer.chat_template");
        if (it_tpl != metadata.end() && !it_tpl->second.str_val.empty()) {
            tokenizer->set_chat_template_str(it_tpl->second.str_val);
            IMP_LOG_INFO("Chat template: %zu chars", it_tpl->second.str_val.size());
        }

        // Load additional EOS-like token IDs (EOT, end-of-generation, etc.)
        // Some models define multiple stop tokens beyond the primary eos_token_id.
        for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eog_token_id"}) {
            auto it_extra = metadata.find(key);
            if (it_extra != metadata.end()) {
                int32_t extra_id = static_cast<int32_t>(val_uint(it_extra->second));
                if (extra_id >= 0) {
                    tokenizer->add_eos_id(extra_id);
                    IMP_LOG_INFO("Tokenizer: additional EOS from %s: %d", key, extra_id);
                }
            }
        }

        IMP_LOG_INFO("Tokenizer: type=%s, %d tokens, bos=%d, eos=%d (%zu total), add_bos=%d",
                     tok_type.c_str(), tokenizer->vocab_size(), bos_id, eos_id, tokenizer->eos_ids().size(),
                     tokenizer->add_bos() ? 1 : 0);
    } else {
        IMP_LOG_WARN("No tokenizer data found in GGUF metadata");
    }

    model->set_tokenizer(std::move(tokenizer));

    IMP_LOG_INFO("GGUF model loaded successfully from %s", path.c_str());
    return model;
}

}  // namespace imp

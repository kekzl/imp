#include "model/safetensors_loader.h"
#include "model/model_arch.h"
#include "model/weight_map.h"
#include "model/hf_config_loader.h"
#include "model/llm_compressor_loader.h"
#include "model/sentencepiece_loader.h"
#include "model/tokenizer.h"
#include "model/json_util.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <utility>

namespace imp {

namespace safetensors_internal {

bool validate_header_size(uint64_t file_size, uint64_t declared_header_size, std::string* err) {
    if (file_size < 8) {
        if (err)
            *err = "file truncated below the 8-byte header_size prefix";
        return false;
    }
    // Overflow-safe form: declared_header_size > (file_size - 8) cannot
    // overflow because file_size - 8 is a valid subtraction (file_size >= 8).
    // The naive 8 + declared_header_size > file_size would wrap to a small
    // value when declared_header_size approaches UINT64_MAX, bypassing the
    // check.
    if (declared_header_size > file_size - 8) {
        if (err)
            *err = "declared header_size exceeds file size (file may be truncated or corrupt)";
        return false;
    }
    if (declared_header_size > kMaxHeaderBytes) {
        if (err)
            *err = "declared header_size exceeds 128 MiB cap (suspicious file or pathological input)";
        return false;
    }
    return true;
}

bool validate_tensor_offsets(uint64_t offset_start, uint64_t offset_end, uint64_t expected_nbytes,
                             uint64_t tensor_data_offset, uint64_t file_size, std::string* err) {
    if (offset_start > offset_end) {
        if (err)
            *err = "offset_start > offset_end (data_offsets swap or corrupt)";
        return false;
    }
    // Overflow-safe: tensor_data_offset + offset_end > file_size becomes
    // offset_end > file_size - tensor_data_offset. file_size >= tensor_data_offset
    // is guaranteed by the upstream validate_header_size check
    // (tensor_data_offset = 8 + header_size, header_size <= file_size - 8).
    if (tensor_data_offset > file_size) {
        if (err)
            *err = "tensor_data_offset > file_size (header_size validation invariant violated)";
        return false;
    }
    if (offset_end > file_size - tensor_data_offset) {
        if (err)
            *err = "tensor offset_end past end of file (file truncated or corrupt)";
        return false;
    }
    if (offset_end - offset_start != expected_nbytes) {
        if (err)
            *err = "tensor byte count does not match shape × dtype width";
        return false;
    }
    return true;
}

}  // namespace safetensors_internal

// ---- SafeTensors wire dtype string -> bytes-per-element ----
//
// Used by validate_tensor_offsets to compute expected_nbytes from shape ×
// dtype width. Returns 0 for unknown dtype strings (validation is then
// skipped for that tensor — the existing safetensors_dtype() WARN covers
// the unknown-type case).
static size_t safetensors_wire_dtype_bytes(const std::string& s) {
    if (s == "F64")
        return 8;
    if (s == "F32" || s == "I32" || s == "U32")
        return 4;
    if (s == "F16" || s == "BF16" || s == "I16" || s == "U16")
        return 2;
    if (s == "F8_E4M3" || s == "F8_E5M2" || s == "I8" || s == "U8" || s == "BOOL")
        return 1;
    if (s == "I64" || s == "U64")
        return 8;
    return 0;
}

// ---- SafeTensors dtype string to QType ----

static QType safetensors_dtype(const std::string& s) {
    if (s == "F32")
        return QType::F32;
    if (s == "F16")
        return QType::F16;
    if (s == "BF16")
        return QType::BF16;
    if (s == "F64")
        return QType::F32;  // closest proxy
    if (s == "I8")
        return QType::INT8;
    if (s == "U8")
        return QType::INT8;  // treat unsigned byte as INT8
    if (s == "I16")
        return QType::INT32;  // closest proxy
    if (s == "I32")
        return QType::INT32;
    if (s == "I64")
        return QType::INT32;  // closest proxy
    if (s == "BOOL")
        return QType::INT8;
    if (s == "F8_E4M3")
        return QType::FP8_E4M3;
    if (s == "F8_E5M2") {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN(
                "SafeTensors F8_E5M2 tensors found; mapping to FP8_E4M3 as a "
                "lossy proxy (no native E5M2 path). Activation-style tensors "
                "may lose precision. (Logged once.)");
        }
        return QType::FP8_E4M3;
    }
    IMP_LOG_WARN("Unknown SafeTensors dtype '%s', defaulting to FP32", s.c_str());
    return QType::F32;
}

// ---- Architecture detection from weight names ----

static ModelArch detect_arch_from_weights(const std::unordered_map<std::string, Tensor>& tensors) {
    bool has_block_sparse_moe = false;
    bool has_mlp_experts = false;
    bool has_ssm = false;
    bool has_layers = false;
    bool has_gptq = false;

    for (const auto& kv : tensors) {
        const auto& name = kv.first;
        if (name.find("model.layers") != std::string::npos)
            has_layers = true;
        if (name.find("block_sparse_moe") != std::string::npos)
            has_block_sparse_moe = true;
        if (name.find("mlp.experts") != std::string::npos)
            has_mlp_experts = true;
        if (name.find("mamba") != std::string::npos || name.find("ssm") != std::string::npos)
            has_ssm = true;
        if (name.find(".qweight") != std::string::npos)
            has_gptq = true;
    }

    if (has_gptq) {
        IMP_LOG_INFO("Detected GPTQ quantized weights");
    }

    if (has_ssm)
        return ModelArch::NEMOTRON_H_MOE;
    if (has_mlp_experts)
        return ModelArch::DEEPSEEK;
    if (has_block_sparse_moe)
        return ModelArch::MIXTRAL;
    if (has_layers)
        return ModelArch::LLAMA;
    return ModelArch::GENERIC;
}

// ---- Extract layer index from a HuggingFace weight name ----
// e.g. "model.layers.5.self_attn.q_proj.weight" -> 5
// Returns -1 if not a layer weight.

static int extract_layer_index(const std::string& name) {
    const char* prefix = "model.layers.";
    size_t plen = std::strlen(prefix);
    if (name.compare(0, plen, prefix) != 0)
        return -1;

    int idx = 0;
    size_t i = plen;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        idx = idx * 10 + (name[i] - '0');
        i++;
    }
    if (i == plen)
        return -1;  // no digits found
    return idx;
}

// ---- Infer max layer index to determine n_layers ----

static int infer_n_layers(const std::unordered_map<std::string, Tensor>& tensors) {
    int max_idx = -1;
    for (const auto& kv : tensors) {
        int idx = extract_layer_index(kv.first);
        if (idx > max_idx)
            max_idx = idx;
    }
    return max_idx + 1;  // 0-indexed, so count = max + 1
}

// ---- Infer model config from weight shapes ----

static void infer_config(ModelConfig& cfg, const std::unordered_map<std::string, Tensor>& tensors) {
    // Only infer fields that are still at their default (zero) values.
    // config.json (via HFConfigLoader) is authoritative when present.

    if (cfg.n_layers == 0)
        cfg.n_layers = infer_n_layers(tensors);

    // token embedding: shape = [vocab_size, d_model]
    auto it = tensors.find("model.embed_tokens.weight");
    if (it != tensors.end() && it->second.ndim == 2) {
        if (cfg.vocab_size == 0)
            cfg.vocab_size = static_cast<int>(it->second.shape[0]);
        if (cfg.d_model == 0)
            cfg.d_model = static_cast<int>(it->second.shape[1]);
    }

    // Only infer heads from weights if config.json didn't set them
    if (cfg.n_heads == 0) {
        auto it_q = tensors.find("model.layers.0.self_attn.q_proj.weight");
        auto it_k = tensors.find("model.layers.0.self_attn.k_proj.weight");
        if (it_q != tensors.end() && it_q->second.ndim == 2 && cfg.d_model > 0) {
            int q_out = static_cast<int>(it_q->second.shape[0]);
            int head_dim = cfg.d_model;
            for (int hd : {128, 64, 96, 80, 256}) {
                if (q_out % hd == 0) {
                    cfg.n_heads = q_out / hd;
                    head_dim = hd;
                    break;
                }
            }
            if (cfg.n_kv_heads == 0 && it_k != tensors.end() && it_k->second.ndim == 2 && head_dim > 0) {
                cfg.n_kv_heads = static_cast<int>(it_k->second.shape[0]) / head_dim;
            }
        }
    }

    if (cfg.d_ff == 0) {
        auto it_gate = tensors.find("model.layers.0.mlp.gate_proj.weight");
        if (it_gate != tensors.end() && it_gate->second.ndim == 2) {
            cfg.d_ff = static_cast<int>(it_gate->second.shape[0]);
        }
    }

    // MoE inference (only if not set by config.json)
    if (cfg.n_experts == 0) {
        auto it_moe = tensors.find("model.layers.0.block_sparse_moe.gate.weight");
        if (it_moe != tensors.end() && it_moe->second.ndim == 2) {
            cfg.n_experts = static_cast<int>(it_moe->second.shape[0]);
            cfg.n_experts_active = std::min(2, cfg.n_experts);
        }
    }

    if (cfg.expert_d_ff == 0 && cfg.n_experts > 0) {
        // Try Mixtral-style (w1) then DeepSeek/Qwen-style (gate_proj)
        for (const char* name : {"model.layers.0.block_sparse_moe.experts.0.w1.weight",
                                 "model.layers.0.mlp.experts.0.gate_proj.weight"}) {
            auto it_expert = tensors.find(name);
            if (it_expert != tensors.end() && it_expert->second.ndim == 2) {
                cfg.expert_d_ff = static_cast<int>(it_expert->second.shape[0]);
                break;
            }
        }
    }

    // Defaults for fields we couldn't infer
    if (cfg.max_seq_len == 0)
        cfg.max_seq_len = 4096;
    if (cfg.n_kv_heads == 0)
        cfg.n_kv_heads = cfg.n_heads;
}

// ---- Per-shard loading helper ----

struct ShardInfo {
    void* mmap_base = nullptr;
    size_t mmap_size = 0;
};

// mtp_out (optional): when non-null, `mtp.*` / `model.mtp.*` tensors — which
// translate_name otherwise SKIPs — are collected there under their raw
// `mtp.*` name (the `model.` prefix stripped). Dense Qwen3.6 checkpoints
// embed the MTP head in the main shard instead of a sidecar.
static bool load_shard(const std::string& path, std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard, bool llm_compressor_format,
                       imp::llm_compressor::TranslationCounters& counters,
                       std::unordered_map<std::string, Tensor>* mtp_out = nullptr) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("Failed to open: %s", path.c_str());
        return false;
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) {
        close(fd);
        return false;
    }

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (mmap_base == MAP_FAILED) {
        // MAP_POPULATE may fail on some filesystems; retry without it.
        int fd2 = open(path.c_str(), O_RDONLY);
        if (fd2 < 0)
            return false;
        mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd2, 0);
        close(fd2);
        if (mmap_base == MAP_FAILED)
            return false;
    }
    madvise(mmap_base, file_size, MADV_WILLNEED);
    madvise(mmap_base, file_size, MADV_SEQUENTIAL);

    shard.mmap_base = mmap_base;
    shard.mmap_size = file_size;

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, sizeof(uint64_t));
    {
        std::string vh_err;
        if (!safetensors_internal::validate_header_size(file_size, header_size, &vh_err)) {
            IMP_LOG_ERROR("SafeTensors %s: %s (file_size=%zu, header_size=%llu)", path.c_str(),
                          vh_err.c_str(), file_size, static_cast<unsigned long long>(header_size));
            munmap(mmap_base, file_size);
            return false;
        }
    }

    const char* json_data = reinterpret_cast<const char*>(data + 8);
    JsonParser parser(json_data, static_cast<size_t>(header_size));
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        munmap(mmap_base, file_size);
        return false;
    }

    size_t tensor_data_offset = 8 + static_cast<size_t>(header_size);
    uint8_t* tensor_data_base = const_cast<uint8_t*>(data + tensor_data_offset);

    // Counters for malformed-entry diagnostics (F5). Each silent skip is
    // counted; a per-shard summary is logged at end of load_shard so users
    // can see which checkpoints have structural issues.
    int n_dropped_no_dtype = 0;
    int n_dropped_no_shape = 0;
    int n_dropped_too_many_dims = 0;
    int n_dropped_no_offsets = 0;
    int n_dropped_offset_validation = 0;
    auto warn_drop = [&](const char* tensor_name, const char* reason) {
        IMP_LOG_WARN("SafeTensors %s: dropping tensor '%s' — %s", path.c_str(), tensor_name, reason);
    };

    for (const auto& kv : root.obj) {
        std::string tensor_name = kv.first;  // copy — may be mutated by translation
        const JValue& tensor_meta = kv.second;

        if (tensor_name == "__metadata__")
            continue;
        if (tensor_meta.type != JType::OBJECT)
            continue;

        // Translate llm-compressor names → modelopt names if applicable.
        bool divert_to_mtp = false;
        if (llm_compressor_format) {
            auto translated = imp::llm_compressor::translate_name(tensor_name, counters);
            if (translated.action == imp::llm_compressor::NameTranslation::SKIP) {
                // Embedded MTP head: keep the tensor, but route it to the
                // separate MTP map under its raw mtp.* name.
                if (mtp_out != nullptr) {
                    if (tensor_name.rfind("model.mtp.", 0) == 0)
                        tensor_name = tensor_name.substr(6);  // strip "model."
                    if (tensor_name.rfind("mtp.", 0) == 0)
                        divert_to_mtp = true;
                }
                if (!divert_to_mtp)
                    continue;
            } else {
                tensor_name = std::move(translated.out_name);
            }
        }

        const JValue* dtype_val = jobj_find(tensor_meta, "dtype");
        if (!dtype_val || dtype_val->type != JType::STRING) {
            n_dropped_no_dtype++;
            warn_drop(tensor_name.c_str(), "missing or non-string 'dtype' field");
            continue;
        }
        QType dtype = safetensors_dtype(dtype_val->str_val);

        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY) {
            n_dropped_no_shape++;
            warn_drop(tensor_name.c_str(), "missing or non-array 'shape' field");
            continue;
        }

        int ndim = static_cast<int>(shape_val->arr.size());
        int64_t shape[kMaxDims] = {};
        if (ndim > kMaxDims) {
            // Was: drop with a WARN. That loses a weight and only says so in a
            // log line — Qwen3-VL's patch embed is [1024, 3, 2, 16, 16] and
            // vanished exactly that way.
            //
            // Flatten the trailing dims into dim 1 instead: [d0, d1..dn] ->
            // [d0, d1*..*dn]. The element order is untouched (row-major), so
            // this is a pure reinterpretation, and it is the only shape imp's
            // GEMM path could consume in any case. It is also the RIGHT shape
            // for the tensors that actually arrive this way: a patch-embed
            // convolution whose kernel equals its stride is a linear projection
            // of flattened patches, i.e. exactly a [out_ch, in_ch*kt*kh*kw]
            // matrix.
            //
            // A genuine sliding-window convolution would lose its window
            // structure here — but it was being DROPPED before, so nothing can
            // regress, and the INFO line makes the reinterpretation visible
            // rather than silent.
            int64_t tail = 1;
            for (int d = 1; d < ndim; d++)
                tail *= shape_val->arr[d].as_int();
            shape[0] = shape_val->arr[0].as_int();
            shape[1] = tail;
            IMP_LOG_WARN("SafeTensors %s: tensor '%s' has %d dims; flattening trailing dims to [%lld, %lld]",
                         path.c_str(), tensor_name.c_str(), ndim, static_cast<long long>(shape[0]),
                         static_cast<long long>(shape[1]));
            ndim = 2;
        } else {
            for (int d = 0; d < ndim; d++) {
                shape[d] = shape_val->arr[d].as_int();
            }
        }

        const JValue* offsets_val = jobj_find(tensor_meta, "data_offsets");
        if (!offsets_val || offsets_val->type != JType::ARRAY || offsets_val->arr.size() != 2) {
            n_dropped_no_offsets++;
            warn_drop(tensor_name.c_str(), "missing or malformed 'data_offsets' field");
            continue;
        }

        uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());

        // Per-tensor offset/size validation (F4): reject swap, OOB end, and
        // shape-vs-byte-count mismatch. expected_nbytes = nelem × wire_dtype_bytes.
        size_t wire_bytes = safetensors_wire_dtype_bytes(dtype_val->str_val);
        if (wire_bytes == 0) {
            // Unknown wire dtype — skip strict validation but keep the soft
            // OOB check below. safetensors_dtype()'s WARN already fires for
            // unknown wire types when the tensor is actually emitted.
            if (tensor_data_offset + offset_end > file_size) {
                n_dropped_offset_validation++;
                warn_drop(tensor_name.c_str(),
                          "offset_end past EOF (unknown wire dtype, lenient check)");
                continue;
            }
        } else {
            int64_t nelem = 1;
            for (int d = 0; d < ndim; d++)
                nelem *= shape[d];
            uint64_t expected_nbytes = static_cast<uint64_t>(nelem) * static_cast<uint64_t>(wire_bytes);
            std::string vt_err;
            if (!safetensors_internal::validate_tensor_offsets(
                    offset_start, offset_end, expected_nbytes, tensor_data_offset, file_size, &vt_err)) {
                n_dropped_offset_validation++;
                warn_drop(tensor_name.c_str(), vt_err.c_str());
                continue;
            }
        }

        void* tensor_ptr = tensor_data_base + offset_start;
        Tensor t(tensor_ptr, dtype, ndim, shape, /*on_device=*/false);
        (divert_to_mtp ? *mtp_out : tensor_map).emplace(tensor_name, t);

        IMP_LOG_DEBUG("Tensor: %s dtype=%s shape=[%ld%s%s%s%s] offsets=[%lu,%lu]", tensor_name.c_str(),
                      dtype_val->str_val.c_str(), (long)shape[0], ndim > 1 ? "," : "",
                      ndim > 1 ? std::to_string(shape[1]).c_str() : "", ndim > 2 ? "," : "",
                      ndim > 2 ? std::to_string(shape[2]).c_str() : "", (unsigned long)offset_start,
                      (unsigned long)offset_end);
    }

    int n_total_dropped = n_dropped_no_dtype + n_dropped_no_shape + n_dropped_too_many_dims +
                          n_dropped_no_offsets + n_dropped_offset_validation;
    if (n_total_dropped > 0) {
        IMP_LOG_WARN("SafeTensors %s: dropped %d malformed tensors (no_dtype=%d no_shape=%d "
                     "too_many_dims=%d no_offsets=%d offset_validation=%d)",
                     path.c_str(), n_total_dropped, n_dropped_no_dtype, n_dropped_no_shape,
                     n_dropped_too_many_dims, n_dropped_no_offsets, n_dropped_offset_validation);
    }

    return true;
}

// ---- Sharded SafeTensors loading ----

static bool load_sharded(const std::string& model_dir, std::unordered_map<std::string, Tensor>& tensor_map,
                         std::vector<ShardInfo>& shards) {
    std::string index_path = model_dir + "/model.safetensors.index.json";

    // Read the index file
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        IMP_LOG_ERROR("Failed to open index: %s", index_path.c_str());
        return false;
    }
    std::string index_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    JsonParser parser(index_json.data(), index_json.size());
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_ERROR("Failed to parse index JSON: %s", index_path.c_str());
        return false;
    }

    const JValue* weight_map = jobj_find(root, "weight_map");
    if (!weight_map || weight_map->type != JType::OBJECT) {
        IMP_LOG_ERROR("No weight_map in index: %s", index_path.c_str());
        return false;
    }

    // Collect tensors per shard (need this to decide whether a shard is
    // entirely skippable — e.g. an MTP-only shard when spec decode is off,
    // or a vision-only shard when no mmproj is configured).
    std::map<std::string, std::vector<std::string>> shard_tensors;
    for (const auto& kv : weight_map->obj) {
        if (kv.second.type == JType::STRING) {
            shard_tensors[kv.second.str_val].push_back(kv.first);
        }
    }

    // Drop shards where every tensor would be skipped by translate_name.
    // Saves the mmap + header parse + page cache pressure for an unused file.
    std::set<std::string> shard_files;
    for (auto& [fname, tensors] : shard_tensors) {
        bool all_skip = !tensors.empty() &&
                        std::all_of(tensors.begin(), tensors.end(), imp::llm_compressor::name_is_skipped);
        if (all_skip) {
            IMP_LOG_INFO("Skipping shard %s (%zu tensors are MTP/vision-only and unused)", fname.c_str(),
                         tensors.size());
            continue;
        }
        shard_files.insert(fname);
    }

    IMP_LOG_INFO("Sharded SafeTensors: %zu shards", shard_files.size());

    // Detect format ONCE so all shards translate consistently.
    imp::HFConfigLoader::NvFP4Config probe_cfg;
    bool probe_ok = imp::HFConfigLoader::load_nvfp4_config(model_dir, probe_cfg);
    bool llm_compressor_format = probe_ok &&
                                 probe_cfg.format == imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    imp::llm_compressor::TranslationCounters tcounters{};

    // Parse shards in parallel (mmap + header decode are independent per file).
    std::vector<std::string> shard_list(shard_files.begin(), shard_files.end());
    std::vector<std::unordered_map<std::string, Tensor>> per_shard_maps(shard_list.size());
    std::vector<ShardInfo> per_shard_info(shard_list.size());
    std::vector<imp::llm_compressor::TranslationCounters> per_shard_counters(shard_list.size());
    std::atomic<bool> any_failure{false};

    std::atomic<size_t> shards_done{0};
    const size_t total_shards = shard_list.size();
    auto worker = [&](size_t i) {
        std::string shard_path = model_dir + "/" + shard_list[i];
        if (!load_shard(shard_path, per_shard_maps[i], per_shard_info[i], llm_compressor_format,
                        per_shard_counters[i])) {
            IMP_LOG_ERROR("Failed to load shard: %s", shard_path.c_str());
            any_failure.store(true);
        }
        const size_t done = shards_done.fetch_add(1) + 1;
        IMP_LOG_INFO("  [%zu/%zu] mmap'd shard: %s (%zu tensors)", done, total_shards,
                     shard_list[i].c_str(), per_shard_maps[i].size());
    };

    {
        std::vector<std::thread> ts;
        ts.reserve(shard_list.size());
        for (size_t i = 0; i < shard_list.size(); ++i)
            ts.emplace_back(worker, i);
        for (auto& t : ts)
            t.join();
    }

    if (any_failure.load())
        return false;

    // Merge per-shard results into the caller-owned aggregates.
    for (size_t i = 0; i < shard_list.size(); ++i) {
        for (auto& kv : per_shard_maps[i])
            tensor_map.emplace(kv.first, kv.second);
        shards.push_back(per_shard_info[i]);
        const auto& c = per_shard_counters[i];
        tcounters.suffix_renames += c.suffix_renames;
        tcounters.prefix_strips += c.prefix_strips;
        tcounters.vision_skipped += c.vision_skipped;
        tcounters.gemma4_extras += c.gemma4_extras;
        tcounters.passed_through += c.passed_through;
        IMP_LOG_INFO("Loaded shard: %s (%zu tensors total)", shard_list[i].c_str(), tensor_map.size());
    }

    if (llm_compressor_format) {
        imp::llm_compressor::log_summary(tcounters);
    }

    return true;
}

// ---- Main SafeTensors loader ----

std::unique_ptr<Model> load_safetensors(const std::string& path, bool load_mtp_head) {
    namespace fs = std::filesystem;

    std::string model_dir;
    std::string single_file;

    if (fs::is_directory(path)) {
        model_dir = path;
    } else if (fs::is_regular_file(path)) {
        single_file = path;
        model_dir = fs::path(path).parent_path().string();
    } else {
        IMP_LOG_ERROR("Path does not exist: %s", path.c_str());
        return nullptr;
    }

    std::unordered_map<std::string, Tensor> tensor_map;
    std::vector<ShardInfo> shards;

    // Detect format ONCE so all shards translate consistently.
    imp::HFConfigLoader::NvFP4Config probe_cfg;
    bool probe_ok = imp::HFConfigLoader::load_nvfp4_config(model_dir, probe_cfg);
    bool llm_compressor_format = probe_ok &&
                                 probe_cfg.format == imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    imp::llm_compressor::TranslationCounters tcounters{};

    // Try loading tensors. embedded_mtp_map collects `mtp.*` tensors that
    // dense Qwen3.6 checkpoints embed in the main shard (no sidecar); only
    // populated when the caller asked for the MTP head.
    bool loaded = false;
    std::unordered_map<std::string, Tensor> embedded_mtp_map;
    auto* mtp_collect = load_mtp_head ? &embedded_mtp_map : nullptr;

    if (!single_file.empty()) {
        // Single file mode
        ShardInfo shard;
        loaded = load_shard(single_file, tensor_map, shard, llm_compressor_format, tcounters,
                            mtp_collect);
        if (loaded)
            shards.push_back(shard);
    } else {
        // Directory mode: try sharded first, then single
        std::string index_path = model_dir + "/model.safetensors.index.json";
        if (fs::exists(index_path)) {
            loaded = load_sharded(model_dir, tensor_map, shards);
        }
        if (!loaded) {
            std::string st_path = model_dir + "/model.safetensors";
            if (fs::exists(st_path)) {
                ShardInfo shard;
                loaded = load_shard(st_path, tensor_map, shard, llm_compressor_format, tcounters,
                                    mtp_collect);
                if (loaded)
                    shards.push_back(shard);
            }
        }
    }

    // For the single-file paths (single_file mode or directory fallback to model.safetensors),
    // emit the summary here. The sharded path (load_sharded) emits its own summary internally
    // with its own counters — tcounters is only populated by the two load_shard calls above.
    if (llm_compressor_format &&
        (tcounters.suffix_renames + tcounters.prefix_strips + tcounters.vision_skipped +
         tcounters.gemma4_extras + tcounters.passed_through) > 0) {
        imp::llm_compressor::log_summary(tcounters);
    }

    if (!loaded || tensor_map.empty()) {
        IMP_LOG_ERROR("Failed to load SafeTensors from %s", path.c_str());
        return nullptr;
    }

    IMP_LOG_INFO("Parsed %zu tensors from SafeTensors", tensor_map.size());

    // Detect + LOAD MTP head sidecar (DeepSeek-V3-family models, e.g. Qwen3.6).
    // Phase 1.B: actual tensor load into Model::mtp_. The main load path
    // continues to skip the `mtp.` prefix; this sidecar load builds a separate
    // host tensor_map keyed by raw `mtp.*` names and dispatches to MtpHead fields.
    // Mmap is retained via Model::split_mmaps_ so Tensor pointers stay valid.
    //
    // Gated on load_mtp_head: the head is ~1.57 GiB BF16 (Qwen3.6) of dead VRAM
    // unless the caller actually enables MTP spec-decode. Default-off skips it
    // entirely so the model behaves as if no sidecar existed.
    std::optional<imp::MtpHead> mtp_local;
    std::unordered_map<std::string, Tensor> mtp_tensor_map;
    ShardInfo mtp_shard{};

    // Dispatch a raw mtp.* tensor map to MtpHead fields. Two MLP variants:
    //   MoE (Qwen3.6-35B): router + packed experts + gated shared expert.
    //   Dense (Qwen3.6-27B embedded head): plain SwiGLU gate/up/down —
    //     mapped onto the shared_expert fields (router/experts stay empty;
    //     mtp_forward runs it as "residual + shared path", no sigmoid gate).
    auto dispatch_mtp = [](std::unordered_map<std::string, Tensor>& tm, const std::string& src,
                           size_t bytes) -> imp::MtpHead {
        imp::MtpHead head;
        head.info.path       = src;
        head.info.file_bytes = bytes;
        head.info.n_tensors  = static_cast<int>(tm.size());
        auto take = [&](const char* key, Tensor& dst) -> bool {
            auto it = tm.find(key);
            if (it == tm.end()) return false;
            dst = it->second;
            return true;
        };
        bool ok = true;
        ok &= take("mtp.pre_fc_norm_embedding.weight", head.pre_fc_norm_embedding);
        ok &= take("mtp.pre_fc_norm_hidden.weight",    head.pre_fc_norm_hidden);
        ok &= take("mtp.fc.weight",                    head.fc);
        ok &= take("mtp.layers.0.input_layernorm.weight",          head.input_layernorm);
        ok &= take("mtp.layers.0.post_attention_layernorm.weight", head.post_attention_layernorm);
        ok &= take("mtp.layers.0.self_attn.q_proj.weight", head.q_proj);
        ok &= take("mtp.layers.0.self_attn.k_proj.weight", head.k_proj);
        ok &= take("mtp.layers.0.self_attn.v_proj.weight", head.v_proj);
        ok &= take("mtp.layers.0.self_attn.o_proj.weight", head.o_proj);
        ok &= take("mtp.layers.0.self_attn.q_norm.weight", head.q_norm);
        ok &= take("mtp.layers.0.self_attn.k_norm.weight", head.k_norm);
        ok &= take("mtp.norm.weight", head.final_norm);

        const bool moe =
            take("mtp.layers.0.mlp.gate.weight", head.router) &&
            take("mtp.layers.0.mlp.experts.gate_up_proj", head.experts_gate_up_packed) &&
            take("mtp.layers.0.mlp.experts.down_proj", head.experts_down_packed) &&
            take("mtp.layers.0.mlp.shared_expert.gate_proj.weight", head.shared_expert_gate_proj) &&
            take("mtp.layers.0.mlp.shared_expert.up_proj.weight", head.shared_expert_up_proj) &&
            take("mtp.layers.0.mlp.shared_expert.down_proj.weight", head.shared_expert_down_proj) &&
            take("mtp.layers.0.mlp.shared_expert_gate.weight", head.shared_expert_gate);
        const bool dense =
            !moe &&
            take("mtp.layers.0.mlp.gate_proj.weight", head.shared_expert_gate_proj) &&
            take("mtp.layers.0.mlp.up_proj.weight", head.shared_expert_up_proj) &&
            take("mtp.layers.0.mlp.down_proj.weight", head.shared_expert_down_proj);
        ok &= (moe || dense);

        head.loaded = ok;
        if (ok) {
            IMP_LOG_INFO("MTP head loaded: %s (%d tensors, %s MLP, BF16)", src.c_str(),
                         head.info.n_tensors, moe ? "MoE" : "dense");
        } else {
            IMP_LOG_WARN("MTP head detected at %s but some expected tensors were missing; "
                         "spec-decode disabled for this model", src.c_str());
        }
        return head;
    };

    if (load_mtp_head && !model_dir.empty()) {
        std::string mtp_path = model_dir + "/model_mtp.safetensors";
        std::error_code ec;
        auto sz = fs::file_size(mtp_path, ec);
        if (!ec && sz > 0) {
            // Sidecar variant. Load the file as a standalone shard.
            // llm_compressor_format=false so translate_name() is NOT applied —
            // MTP tensor names need to stay literal for dispatch.
            imp::llm_compressor::TranslationCounters mtp_counters{};
            bool mtp_loaded = load_shard(mtp_path, mtp_tensor_map, mtp_shard,
                                          /*llm_compressor_format=*/false, mtp_counters);
            if (mtp_loaded) {
                mtp_local = dispatch_mtp(mtp_tensor_map, mtp_path, static_cast<size_t>(sz));
            } else {
                IMP_LOG_WARN("MTP head file %s present but failed to load", mtp_path.c_str());
            }
        } else {
            // Embedded variant. llm-compressor checkpoints had their mtp.*
            // tensors diverted into embedded_mtp_map during load_shard
            // (translate_name SKIPs them); Model-Optimizer checkpoints keep
            // them in the main tensor_map (weight_map skips them later) —
            // harvest from there.
            if (embedded_mtp_map.empty()) {
                for (const auto& kv : tensor_map) {
                    if (kv.first.rfind("mtp.", 0) == 0)
                        embedded_mtp_map.emplace(kv.first, kv.second);
                    else if (kv.first.rfind("model.mtp.", 0) == 0)
                        embedded_mtp_map.emplace(kv.first.substr(6), kv.second);
                }
            }
            if (!embedded_mtp_map.empty()) {
                size_t bytes = 0;
                for (const auto& kv : embedded_mtp_map) bytes += kv.second.nbytes();
                mtp_local = dispatch_mtp(embedded_mtp_map, path + " (embedded mtp.*)", bytes);
            }
        }
    }

    // Create model
    auto model = std::make_unique<Model>();
    model->source_path_ = path;
    if (mtp_local.has_value()) {
        model->mtp_ = std::move(mtp_local);
        // Retain MTP mmap so Tensor data pointers stay valid for the lifetime
        // of the Model. Cleaned up by Model destructor like other shard mmaps.
        if (mtp_shard.mmap_base != nullptr) {
            model->split_mmaps_.emplace_back(mtp_shard.mmap_base, mtp_shard.mmap_size);
        }
    }

    // Store mmap info for cleanup
    model->mmap_base_ = shards[0].mmap_base;
    model->mmap_size_ = shards[0].mmap_size;
    for (size_t i = 1; i < shards.size(); i++) {
        model->split_mmaps_.emplace_back(shards[i].mmap_base, shards[i].mmap_size);
    }

    ModelConfig& cfg = model->config_;

    // 1. Try config.json (authoritative for all hyperparams)
    bool has_config = HFConfigLoader::load_config(model_dir, cfg);

    // 2. Detect architecture from weights if config.json didn't provide it
    if (!has_config || cfg.arch == ModelArch::GENERIC) {
        ModelArch detected = detect_arch_from_weights(tensor_map);
        if (cfg.arch == ModelArch::GENERIC && detected != ModelArch::GENERIC) {
            cfg.arch = detected;
        }
    }

    // 3. Infer remaining config from weight shapes (fills fields still at defaults)
    infer_config(cfg, tensor_map);

    // 4. Apply arch-specific defaults
    apply_arch_defaults(cfg);

    // SafeTensors store Q/K in HuggingFace-native layout, which uses NeoX-style
    // (rotate-half) RoPE: pairs (i, i+rope_dim/2). The arch table defaults
    // LLAMA/MISTRAL/MIXTRAL/LLAMA4 to interleaved RoPE (rope_neox=0) because the
    // GGUF converter pre-permutes Q/K so interleaved reproduces NeoX. That
    // permutation is NOT applied to SafeTensors weights, so the interleaved
    // default scrambles positional encoding (e.g. Phi-3/Phi-4 -> coherent but
    // prompt-blind output with preserved magnitudes). Force NeoX for the
    // SafeTensors path so HF-native Q/K rotate correctly. Arches that already
    // keep the default (rope_neox=true: QWEN3/GEMMA/...) are unaffected.
    if (cfg.arch == ModelArch::LLAMA || cfg.arch == ModelArch::MISTRAL ||
        cfg.arch == ModelArch::MIXTRAL || cfg.arch == ModelArch::LLAMA4) {
        cfg.rope_neox = true;
    }

    IMP_LOG_INFO("Architecture: %s", model_arch_name(cfg.arch));
    IMP_LOG_INFO("Config: layers=%d d_model=%d d_ff=%d heads=%d kv_heads=%d vocab=%d ctx=%d", cfg.n_layers,
                 cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.max_seq_len);
    IMP_LOG_INFO(
        "RoPE: theta=%.0f freq_scale=%.4f head_dim=%d sliding_window=%d "
        "rope_theta_swa=%.0f rope_local_theta=%.0f rope_n_ctx_orig=%d",
        cfg.rope_theta, cfg.rope_freq_scale, cfg.head_dim, cfg.sliding_window, cfg.rope_theta_swa,
        cfg.rope_local_theta, cfg.rope_n_ctx_orig);
    if (cfg.n_experts > 0) {
        IMP_LOG_INFO("MoE: %d experts, %d active, expert_d_ff=%d", cfg.n_experts, cfg.n_experts_active,
                     cfg.expert_d_ff);
    }

    // 5. Allocate layers and expert vectors
    model->layers_.resize(cfg.n_layers);
    if (cfg.n_experts > 0) {
        for (auto& layer : model->layers_) {
            layer.expert_w_gate.resize(cfg.n_experts);
            layer.expert_w_up.resize(cfg.n_experts);
            layer.expert_w_down.resize(cfg.n_experts);
        }
    }

    // 6. Assign tensors via WeightMap
    WeightMap wmap(cfg.arch);
    wmap.apply_weights(*model, tensor_map);

    // 6b. GPTQ config: set bit width and group size on all GPTQ weight structs
    HFConfigLoader::GPTQConfig gptq_cfg;
    bool is_gptq = HFConfigLoader::load_gptq_config(model_dir, gptq_cfg);
    if (is_gptq) {
        for (auto& layer : model->layers_) {
            for (auto* gw : {&layer.gptq_q, &layer.gptq_k, &layer.gptq_v, &layer.gptq_o, &layer.gptq_gate,
                             &layer.gptq_up, &layer.gptq_down}) {
                gw->bits = gptq_cfg.bits;
                gw->group_size = gptq_cfg.group_size;
                gw->desc_act = gptq_cfg.desc_act;
            }
        }
        IMP_LOG_INFO("GPTQ model: %d-bit, group_size=%d, desc_act=%s", gptq_cfg.bits, gptq_cfg.group_size,
                     gptq_cfg.desc_act ? "true" : "false");
    }

    // 6c. NVFP4 config detection. Scale tensors were already routed into
    // model->nvfp4_scratch_ by weight_map.cpp during the layer-pattern pass;
    // executor_pre_dequant.cu's Phase 0 promote() resolves each scratch key
    // back to the main weight tensor and writes the device pointer onto its
    // .scales / .tensor_scale sidecars. No load-side linking needed here.
    // MXFP4 detection. gpt-oss MXFP4 experts ARE decoded natively: they are
    // transcoded MXFP4→NVFP4 at init (pre_dequant_phase3_nvfp4_decode.cu) and
    // run through the CUTLASS NVFP4 grouped GEMM. Other MXFP4 SafeTensors
    // architectures have no decode path yet — only those still need the GGUF
    // conversion warning.
    HFConfigLoader::MxFP4Config mxfp4_cfg;
    if (HFConfigLoader::load_mxfp4_config(model_dir, mxfp4_cfg)) {
        cfg.is_mxfp4_prequant = true;
        cfg.mxfp4_block_size = mxfp4_cfg.block_size;
        if (cfg.arch == ModelArch::GPT_OSS) {
            IMP_LOG_INFO("MXFP4 SafeTensors (block_size=%d): gpt-oss experts will be "
                         "transcoded MXFP4→NVFP4 at init (native decode).",
                         cfg.mxfp4_block_size);
        } else {
            IMP_LOG_WARN(
                "MXFP4 SafeTensors detected (block_size=%d) — imp has no SafeTensors "
                "MXFP4 decode path for this architecture yet. Weights will load as "
                "their wire dtype and inference will likely be incorrect. Convert to "
                "GGUF for actual MXFP4 support.",
                cfg.mxfp4_block_size);
        }
    }

    // AWQ detection. Same posture as MXFP4: the metadata is parsed and
    // surfaced but no native AWQ kernel exists yet. Future work: dequant-
    // to-FP16 fallback or proper AWQ-aware GEMM.
    HFConfigLoader::AWQConfig awq_cfg;
    if (HFConfigLoader::load_awq_config(model_dir, awq_cfg)) {
        cfg.is_awq_prequant = true;
        cfg.awq_group_size = awq_cfg.group_size;
        IMP_LOG_WARN(
            "AWQ SafeTensors detected (bits=%d group_size=%d zero_point=%s "
            "version=%s) — imp does not yet have an AWQ dequant kernel. "
            "Weights load as their wire dtype and inference will be "
            "incorrect. Use a GPTQ or NVFP4 export instead.",
            awq_cfg.bits, awq_cfg.group_size,
            awq_cfg.zero_point ? "true" : "false",
            awq_cfg.version.empty() ? "unspecified" : awq_cfg.version.c_str());
    }

    HFConfigLoader::NvFP4Config nvfp4_cfg;
    bool is_nvfp4 = HFConfigLoader::load_nvfp4_config(model_dir, nvfp4_cfg);
    if (is_nvfp4) {
        cfg.is_nvfp4_prequant = true;
        cfg.nvfp4_group_size = nvfp4_cfg.group_size;
        cfg.is_llm_compressor_nvfp4 = (nvfp4_cfg.format == HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
        cfg.kv_cache_quant_hint = nvfp4_cfg.kv_cache_quant_algo;
        IMP_LOG_INFO("NVFP4 pre-quantized: %zu scratch entries (group_size=%d)", model->nvfp4_scratch_.size(),
                     nvfp4_cfg.group_size);
        if (!cfg.kv_cache_quant_hint.empty()) {
            IMP_LOG_INFO(
                "Model author declared kv_cache_quant_algo=%s; with kv_cache.dtype=auto "
                "(default) imp honors it for arch families verified safe for long-context "
                "FP8 KV, else keeps FP16. Force with --kv-fp8 or opt out with kv_cache.dtype=fp16.",
                cfg.kv_cache_quant_hint.c_str());
        }
    }

    // 7. Tie output projection if not found.
    // Cross-check the author's `tie_word_embeddings` flag (parsed by
    // HFConfigLoader::load_config) against the actual lm_head.weight
    // presence. Mismatch is a real surprise — most models tie, so silent
    // tying when the author said `tie=false` would mask a missing lm_head.
    const bool out_proj_missing =
        (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr);
    if (cfg.tie_word_embeddings == 0 && out_proj_missing) {
        IMP_LOG_WARN(
            "config.json declares tie_word_embeddings=false but lm_head.weight "
            "is absent in the SafeTensors files; tying anyway as a fallback.");
    }
    if (cfg.tie_word_embeddings == 1 && model->out_proj_.data != nullptr &&
        model->tok_emb_.data != nullptr && model->out_proj_.data != model->tok_emb_.data) {
        IMP_LOG_INFO(
            "config.json declares tie_word_embeddings=true but lm_head.weight "
            "was loaded as a separate tensor; honoring the file (no tying).");
    }
    if (out_proj_missing) {
        model->out_proj_ = model->tok_emb_;
        IMP_LOG_INFO("Tied output projection to token embedding");
    }

    // 8. Load chat template from tokenizer_config.json
    if (!model_dir.empty()) {
        std::string chat_tpl = HFConfigLoader::load_chat_template(model_dir);
        if (!chat_tpl.empty()) {
            if (!model->tokenizer_) {
                auto tok = std::make_unique<Tokenizer>();
                tok->set_chat_template_str(chat_tpl);
                model->set_tokenizer(std::move(tok));
            } else {
                model->tokenizer_->set_chat_template_str(chat_tpl);
            }
        }
    }

    // 9. Load tokenizer from tokenizer.json (if available)
    if (!model_dir.empty()) {
        std::string tok_json_path = model_dir + "/tokenizer.json";
        std::string tok_spm_path = model_dir + "/tokenizer.model";
        bool has_json = std::filesystem::exists(tok_json_path);
        bool has_spm = std::filesystem::exists(tok_spm_path);
        if (has_json) {
            auto tok = std::make_unique<Tokenizer>();
            if (tok->load(tok_json_path)) {
                // Preserve chat template if already set
                if (model->tokenizer_ && !model->tokenizer_->chat_template_str().empty()) {
                    tok->set_chat_template_str(model->tokenizer_->chat_template_str());
                }
                model->set_tokenizer(std::move(tok));
                IMP_LOG_INFO("Loaded tokenizer from %s", tok_json_path.c_str());
            }
        } else if (has_spm) {
            // SentencePiece-only checkpoint (older Llama 1/2, Mistral, …).
            // Native protobuf parser populates vocab + scores + token types;
            // the SPM-style encoder in tokenizer.cpp:encode_spm() handles the
            // rest. BPE-from-spm checkpoints get loaded with the same vocab
            // table — the score-based encoder produces equivalent output for
            // most practical text.
            SentencePieceModel spm = load_sentencepiece_model_file(tok_spm_path);
            if (!spm.empty()) {
                auto tok = std::make_unique<Tokenizer>();
                tok->set_type("spm");
                tok->load_vocab(spm.pieces, spm.scores, spm.bos_id, spm.eos_id);
                tok->load_token_types(spm.types);
                if (model->tokenizer_ && !model->tokenizer_->chat_template_str().empty()) {
                    tok->set_chat_template_str(model->tokenizer_->chat_template_str());
                }
                model->set_tokenizer(std::move(tok));
                IMP_LOG_INFO("Loaded SentencePiece tokenizer from %s (no tokenizer.json present)",
                             tok_spm_path.c_str());
            } else {
                IMP_LOG_ERROR(
                    "Found %s but failed to parse — checkpoint may be corrupt or "
                    "use an unsupported SentencePiece variant. Workaround: convert via\n"
                    "  python -c \"from transformers import AutoTokenizer; "
                    "AutoTokenizer.from_pretrained('%s').save_pretrained('%s')\"",
                    tok_spm_path.c_str(), model_dir.c_str(), model_dir.c_str());
            }
        } else {
            IMP_LOG_WARN(
                "No tokenizer.json or tokenizer.model in %s — model will load "
                "without a tokenizer; chat input cannot be encoded.",
                model_dir.c_str());
        }
    }

    // 9b. tokenizer_config.json — tokenizer-side flags (add_bos_token,
    // add_prefix_space). Mirrors gguf_loader.cpp's read of
    // tokenizer.ggml.add_bos_token / add_space_prefix. Without this the
    // SafeTensors path used Tokenizer's hardcoded default `add_bos_=true`
    // — wrong for any model that ships add_bos_token=false in its config
    // (e.g. Qwen3-Coder-30B-A3B-FP4 which auto-prepends <|endoftext|>
    // unwantedly otherwise).
    if (!model_dir.empty() && model->tokenizer_) {
        HFConfigLoader::TokenizerFlags tflags;
        if (HFConfigLoader::load_tokenizer_flags(model_dir, tflags)) {
            if (tflags.add_bos_token >= 0) {
                model->tokenizer_->set_add_bos(tflags.add_bos_token != 0);
            } else if (model->tokenizer_->type() == "gpt2") {
                // Match GGUF default: BPE tokenizers without an explicit
                // flag don't add BOS.
                model->tokenizer_->set_add_bos(false);
            }
            if (tflags.add_prefix_space >= 0) {
                model->tokenizer_->set_add_space_prefix(tflags.add_prefix_space != 0);
            }
            if (tflags.use_default_system_prompt >= 0) {
                model->tokenizer_->set_use_default_system_prompt(tflags.use_default_system_prompt != 0);
            }
            // Resolve BOS/EOS token IDs from the token content strings.
            // This handles models like DeepSeek whose BOS token string
            // ("<｜begin▁of▁sentence｜>") is not in the hardcoded detection
            // list in tokenizer.cpp — they land in added_tokens and are
            // already in the vocabulary; we just need to wire up the ID.
            if (!tflags.bos_token.empty()) {
                int32_t bid = model->tokenizer_->find_token(tflags.bos_token);
                if (bid >= 0) {
                    model->tokenizer_->set_bos_id(bid);
                    IMP_LOG_INFO("tokenizer_config.json: resolved bos_token '%s' -> id %d",
                                 tflags.bos_token.c_str(), bid);
                } else {
                    IMP_LOG_WARN("tokenizer_config.json: bos_token '%s' not found in vocab",
                                 tflags.bos_token.c_str());
                }
            }
            if (!tflags.eos_token.empty()) {
                int32_t eid = model->tokenizer_->find_token(tflags.eos_token);
                if (eid >= 0) {
                    model->tokenizer_->add_eos_id(eid);
                    IMP_LOG_INFO("tokenizer_config.json: resolved eos_token '%s' -> id %d",
                                 tflags.eos_token.c_str(), eid);
                } else {
                    IMP_LOG_WARN("tokenizer_config.json: eos_token '%s' not found in vocab",
                                 tflags.eos_token.c_str());
                }
            }
        }
    }

    // Re-infer vocab_size from token embedding if needed
    if (cfg.vocab_size == 0 && model->tok_emb_.data != nullptr) {
        cfg.vocab_size = static_cast<int>(model->tok_emb_.shape[0]);
    }

    // 10. generation_config.json — sampling/EOS defaults shipped by the model
    // author. Loaded into model->generation_config_ for engine + CLI consumers.
    // EOS IDs additionally pushed onto the tokenizer's eos list so the engine's
    // existing stop-condition path picks them up without further plumbing.
    if (!model_dir.empty()) {
        HFConfigLoader::load_generation_config(model_dir, model->generation_config_);
        if (model->tokenizer_) {
            for (int32_t eid : model->generation_config_.eos_token_ids) {
                model->tokenizer_->add_eos_id(eid);
            }
        }
    }

    // 11. Cross-check special_tokens_map.json against the loaded tokenizer's
    // special-flag column. The model author's list is authoritative; if a
    // string from `additional_special_tokens` exists in vocab but isn't
    // marked CONTROL (token_type=3), patch it. Caught by the engine's
    // banned-token scan in engine.cpp.
    if (!model_dir.empty() && model->tokenizer_) {
        HFConfigLoader::SpecialTokensMap stm;
        if (HFConfigLoader::load_special_tokens_map(model_dir, stm)) {
            int patched = 0, missing = 0;
            for (const auto& s : stm.additional_special_tokens) {
                int32_t id = model->tokenizer_->find_token(s);
                if (id < 0) {
                    missing++;
                    continue;
                }
                if (!model->tokenizer_->is_control_token(id)) {
                    model->tokenizer_->mark_as_control(id);
                    patched++;
                }
            }
            if (patched > 0 || missing > 0) {
                IMP_LOG_INFO(
                    "special_tokens_map: cross-check patched %d, "
                    "missing-from-vocab %d",
                    patched, missing);
            }
        }
    }

    // 12. Validate config-promised biases against actual tensor presence.
    // Some HF configs declare attention_bias/mlp_bias but the SafeTensors
    // export omits the bias tensors. Without this check the loader silently
    // leaves bias slots null and inference proceeds with undefined behaviour
    // depending on which kernels short-circuit on null biases.
    if (cfg.attention_bias == 1) {
        int missing_q = 0, missing_k = 0, missing_v = 0;
        for (const auto& layer : model->layers_) {
            if (layer.q_bias.data == nullptr) missing_q++;
            if (layer.k_bias.data == nullptr) missing_k++;
            if (layer.v_bias.data == nullptr) missing_v++;
        }
        if (missing_q || missing_k || missing_v) {
            IMP_LOG_WARN(
                "config.json says attention_bias=true but %d/%d Q-biases, "
                "%d/%d K-biases, %d/%d V-biases are missing from the SafeTensors "
                "export. Inference will proceed without those biases.",
                missing_q, static_cast<int>(model->layers_.size()),
                missing_k, static_cast<int>(model->layers_.size()),
                missing_v, static_cast<int>(model->layers_.size()));
        }
    }

    if (cfg.arch_inferred_fallback) {
        IMP_LOG_WARN(
            "Architecture detection fell back to GENERIC + tensor-name "
            "heuristics (config.json had no recognized architectures/model_type). "
            "Inference may be incoherent.");
    }

    // gpt-oss post-load pass (#547): de-interleave the fused gate_up expert
    // bias (g0,u0,g1,u1,... along the column dim) into separate gate/up bias
    // tensors so the standard per-projection bias plumbing applies. The
    // packed MXFP4 expert weights stay host-mmap'd; the MXFP4→NVFP4
    // conversion happens at executor pre-dequant (needs the executor's
    // wcache). Host copies live in host_owned_buffers_ (freed in ~Model).
    if (cfg.arch == ModelArch::GPT_OSS) {
        for (auto& layer : model->layers_) {
            const Tensor& fused = layer.expert_gate_up_bias_fused;
            if (!fused.data || fused.ndim != 2)
                continue;
            const int64_t ne = fused.shape[0];
            const int64_t two_ff = fused.shape[1];
            const int64_t ff = two_ff / 2;
            const uint16_t* src = static_cast<const uint16_t*>(fused.data);  // BF16
            auto* g = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * ne * ff));
            auto* u = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * ne * ff));
            if (!g || !u) {
                std::free(g);
                std::free(u);
                IMP_LOG_ERROR("gpt-oss: bias de-interleave alloc failed");
                return nullptr;
            }
            for (int64_t e = 0; e < ne; e++) {
                const uint16_t* row = src + e * two_ff;
                for (int64_t i = 0; i < ff; i++) {
                    g[e * ff + i] = row[2 * i];
                    u[e * ff + i] = row[2 * i + 1];
                }
            }
            model->host_owned_buffers_.push_back(g);
            model->host_owned_buffers_.push_back(u);
            int64_t bshape[4] = {ne, ff, 0, 0};
            layer.expert_gate_bias = Tensor(g, QType::BF16, 2, bshape, /*on_device=*/false);
            layer.expert_up_bias = Tensor(u, QType::BF16, 2, bshape, /*on_device=*/false);
        }
        IMP_LOG_INFO("gpt-oss: expert gate_up biases de-interleaved for %zu layers",
                     model->layers_.size());

        // Residual-stream 2^-4 rescale (#547): gpt-oss's massive BF16
        // activations overflow the FP16 hidden state (hidden L2 jumps ~12x at
        // L6 and reaches ±inf by L23 → NaN logits + garbage decode). Scaling
        // every contributor to h by 2^-4 is exact for the model output: FP16
        // scaling is an exponent shift (lossless), RMSNorm is scale-invariant,
        // and lm_head reads only normed values. Contributors:
        //   - embeddings: cfg.embed_scale = 2^-4 (arch registry)
        //   - Wo + o_bias + expert down bias: scaled here (host BF16)
        //   - expert down weights: tensor_scales inside the MXFP4→NVFP4
        //     converter (pre_dequant_phase3_nvfp4_decode.cu)
        auto scale_bf16_pow2 = [&](Tensor& t, int neg_exp) -> bool {
            if (!t.data || t.on_device)
                return true;
            if (t.qtype != QType::BF16) {
                IMP_LOG_ERROR("gpt-oss rescale: expected BF16, got qtype %d", std::to_underlying(t.qtype));
                return false;
            }
            int64_t n = t.numel();
            auto* dst = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * n));
            if (!dst)
                return false;
            const uint16_t* src = static_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; i++) {
                uint16_t v = src[i];
                int exp = (v >> 7) & 0xFF;
                if (exp == 0xFF || exp == 0)
                    dst[i] = v;  // inf/nan/zero/subnormal: keep as-is
                else if (exp <= neg_exp)
                    dst[i] = static_cast<uint16_t>(v & 0x8000);  // underflow → signed zero
                else
                    dst[i] = static_cast<uint16_t>(v - (neg_exp << 7));
            }
            model->host_owned_buffers_.push_back(dst);
            t.data = dst;
            return true;
        };
        bool rescale_ok = true;
        for (auto& layer : model->layers_) {
            rescale_ok = rescale_ok && scale_bf16_pow2(layer.wo, 4) &&
                         scale_bf16_pow2(layer.o_bias, 4) && scale_bf16_pow2(layer.expert_down_bias, 4);
        }
        if (!rescale_ok) {
            IMP_LOG_ERROR("gpt-oss: residual-stream rescale failed");
            return nullptr;
        }
        IMP_LOG_INFO("gpt-oss: residual stream rescaled by 2^-4 (FP16 range headroom)");
    }

    IMP_LOG_INFO("SafeTensors model loaded successfully from %s", path.c_str());
    return model;
}

}  // namespace imp

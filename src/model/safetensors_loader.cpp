#include "model/safetensors_loader.h"
#include "model/model_limits.h"
#include "model/model_arch.h"
#include "model/weight_map.h"
#include "model/hf_config_loader.h"
#include "model/llm_compressor_loader.h"
#include "model/ngram_table.h"
#include "model/nvfp4_module_policy.h"
#include "model/sentencepiece_loader.h"
#include "model/tokenizer.h"
#include "model/json_util.h"
#include "model/awq_load.h"
#include "quant/dequant_gptq.h"
#include "core/logging.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <string_view>
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
    // Overflow-safe: declared_header_size > (file_size - 8) cannot overflow since file_size-8 is
    // a valid subtraction (file_size >= 8). The naive 8+declared_header_size > file_size wraps to
    // a small value near UINT64_MAX, bypassing the check.
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
    // offset_end > file_size - tensor_data_offset; file_size >= tensor_data_offset is guaranteed
    // by validate_header_size upstream (tensor_data_offset = 8 + header_size <= file_size).
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

// #1604: the width a tensor is validated with and the width its consumer reads it with must
// be the same number (I16 disagreed: 2 bytes on disk vs 4 for the QType::INT32 it mapped to,
// so a passing tensor was read 2x past its window). A dtype with no equal-width engine type
// is refused, not re-typed: a "closest proxy" of a different width reads element i from the
// wrong byte stride.
struct SafeTensorsDtype {
    std::string_view name;
    size_t wire_bytes;  // bytes per element on disk
    QType qtype;        // engine type; only meaningful when servable
    bool servable;
    const char* once_note;  // logged once, the first time this dtype appears
};

static constexpr SafeTensorsDtype kSafeTensorsDtypes[] = {
    {"F32", 4, QType::F32, true, nullptr},
    {"F16", 2, QType::F16, true, nullptr},
    {"BF16", 2, QType::BF16, true, nullptr},
    {"F8_E4M3", 1, QType::FP8_E4M3, true, nullptr},
    // Lossy but equal-width, so the stride is right and only the exponent
    // interpretation is approximate.
    {"F8_E5M2", 1, QType::FP8_E4M3, true,
     "SafeTensors F8_E5M2 tensors found; mapping to FP8_E4M3 as a lossy proxy "
     "(no native E5M2 path). Activation-style tensors may lose precision. (Logged once.)"},
    {"I8", 1, QType::INT8, true, nullptr},
    {"U8", 1, QType::INT8, true, nullptr},
    {"BOOL", 1, QType::INT8, true, nullptr},
    {"I32", 4, QType::INT32, true, nullptr},
    {"U32", 4, QType::INT32, true, nullptr},
    // Not servable: no 8-byte or 16-bit integer engine type exists for an equal-width mapping.
    // These used to map F64/I64->INT32 (8->4, wrong stride) and I16/U16->INT32 or F32 default
    // (2->4, a 100% over-read past the last tensor in a shard).
    {"F64", 8, QType::F32, false, nullptr},
    {"I64", 8, QType::INT32, false, nullptr},
    {"U64", 8, QType::INT32, false, nullptr},
    {"I16", 2, QType::INT32, false, nullptr},
    {"U16", 2, QType::INT32, false, nullptr},
};

const SafeTensorsDtype* safetensors_dtype_entry(const std::string& s) {
    for (const auto& e : kSafeTensorsDtypes) {
        if (e.name == s)
            return &e;
    }
    return nullptr;
}

// Shard loading is multi-threaded, so the once-per-dtype note needs a lock;
// the previous plain `static bool warned` raced (harmlessly, but it raced).
static void log_dtype_note_once(const SafeTensorsDtype& dt) {
    if (!dt.once_note)
        return;
    static std::mutex note_mutex;
    static std::set<std::string> noted;
    std::lock_guard<std::mutex> lock(note_mutex);
    if (noted.insert(std::string(dt.name)).second)
        IMP_LOG_WARN("%s", dt.once_note);
}

namespace safetensors_internal {

size_t dtype_table_size() { return sizeof(kSafeTensorsDtypes) / sizeof(kSafeTensorsDtypes[0]); }

DtypeTableRow dtype_table_row(size_t i) {
    const auto& e = kSafeTensorsDtypes[i];
    return DtypeTableRow{std::string(e.name), e.wire_bytes, e.qtype, e.servable};
}

}  // namespace safetensors_internal

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

    // `idx*10 + digit` on an unchecked digit run is signed overflow (UB), and the result sizes
    // model->layers_ via infer_n_layers when config.json is absent. Stop at the limit: a name
    // past it isn't a layer this build can serve either way.
    int idx = 0;
    size_t i = plen;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        idx = idx * 10 + (name[i] - '0');
        if (idx > kMaxModelLayers)
            return -1;
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

// mtp_out (optional): when non-null, mtp.*/model.mtp.* tensors (which translate_name would
// otherwise SKIP) are collected under their raw mtp.* name. keep_vision: passed through to
// translate_name; not defaulted, since a caller that forgets it silently loses the tower.
// sparse: shard kept only for MTP tensors next to unused ones (Qwen4Exp FP8 shard, 50 GiB of
// PLE table vs 2.5 GiB head): no MAP_POPULATE, unused names dropped, WILLNEED on kept ranges only.
static bool load_shard(const std::string& path, std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard, bool llm_compressor_format, bool keep_vision,
                       imp::llm_compressor::TranslationCounters& counters,
                       std::unordered_map<std::string, Tensor>* mtp_out = nullptr, bool sparse = false) {
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

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | (sparse ? 0 : MAP_POPULATE), fd, 0);
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
    if (!sparse) {
        madvise(mmap_base, file_size, MADV_WILLNEED);
        madvise(mmap_base, file_size, MADV_SEQUENTIAL);
    }
    int n_sparse_unused = 0;
    size_t sparse_willneed_bytes = 0;

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
    JsonParser parser(std::string_view(json_data, static_cast<size_t>(header_size)));
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
    int n_dropped_dtype_unsupported = 0;
    int n_dropped_bad_shape = 0;
    auto warn_drop = [&](const char* tensor_name, const char* reason) {
        IMP_LOG_WARN("SafeTensors %s: dropping tensor '%s' — %s", path.c_str(), tensor_name, reason);
    };

    for (const auto& kv : root.obj) {
        std::string tensor_name = kv.key;  // copy — may be mutated by translation
        const JValue& tensor_meta = kv.value;

        if (tensor_name == "__metadata__")
            continue;
        if (tensor_meta.type != JType::OBJECT)
            continue;
        if (sparse && imp::llm_compressor::name_is_unused(tensor_name, keep_vision, /*keep_mtp=*/true)) {
            n_sparse_unused++;
            continue;
        }

        // Translate llm-compressor names → modelopt names if applicable.
        bool divert_to_mtp = false;
        if (llm_compressor_format) {
            auto translated = imp::llm_compressor::translate_name(tensor_name, counters, keep_vision);
            if (translated.action == imp::llm_compressor::NameTranslation::SKIP) {
                // Embedded MTP head: keep the tensor, but route it to the
                // separate MTP map under its raw mtp.* name.
                if (mtp_out != nullptr && name_is_mtp_tensor(tensor_name)) {
                    if (tensor_name.rfind("model.mtp.", 0) == 0)
                        tensor_name = tensor_name.substr(6);
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
        // #1603/#1604: the dtype decides both the QType and the width the
        // offsets are validated with, so an unknown or unservable one is a
        // drop here and never reaches a lenient validation branch below.
        const SafeTensorsDtype* dt = safetensors_dtype_entry(dtype_val->str_val);
        if (!dt || !dt->servable) {
            n_dropped_dtype_unsupported++;
            std::string reason = dt ? ("dtype '" + dtype_val->str_val +
                                       "' has no equal-width engine type, refusing to re-type it")
                                    : ("unknown SafeTensors dtype '" + dtype_val->str_val + "'");
            warn_drop(tensor_name.c_str(), reason.c_str());
            continue;
        }
        log_dtype_note_once(*dt);
        QType dtype = dt->qtype;

        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY) {
            n_dropped_no_shape++;
            warn_drop(tensor_name.c_str(), "missing or non-array 'shape' field");
            continue;
        }

        int ndim = static_cast<int>(shape_val->arr.size());
        int64_t shape[kMaxDims] = {};

        // #1605: every dim arrives as a JSON double narrowed to int64, so sign and the running
        // product must be checked BEFORE multiplication, or a wrapped product yields a small
        // expected_nbytes that passes the offset check (same guard gguf_tensor_byte_size() has).
        uint64_t nelem = 1;
        bool bad_shape = false;
        for (int d = 0; d < ndim; d++) {
            int64_t dim = shape_val->arr[d].as_int();
            if (dim < 0) {
                bad_shape = true;
                break;
            }
            uint64_t udim = static_cast<uint64_t>(dim);
            if (udim != 0 && nelem > UINT64_MAX / udim) {
                bad_shape = true;
                break;
            }
            nelem *= udim;
        }
        // Tensor::numel() redoes this product in int64_t, so a value above
        // INT64_MAX would be signed overflow there even though it fits here.
        if (bad_shape || nelem > static_cast<uint64_t>(INT64_MAX)) {
            n_dropped_bad_shape++;
            warn_drop(tensor_name.c_str(), "negative or overflowing shape dimension");
            continue;
        }

        if (ndim > kMaxDims) {
            // Flatten trailing dims into dim 1 ([d0,d1..dn]->[d0,d1*..*dn]) instead of dropping the
            // tensor with a WARN (Qwen3-VL's [1024,3,2,16,16] patch embed used to vanish that way).
            // Row-major order is untouched; a same-stride kernel conv is exactly this flattened matrix.
            // shape[0]==0 zeroes the total product while the tail alone can still overflow, so the tail
            // gets its own saturating guard.
            uint64_t tail = 1;
            bool tail_overflow = false;
            for (int d = 1; d < ndim; d++) {
                uint64_t udim = static_cast<uint64_t>(shape_val->arr[d].as_int());
                if (udim != 0 && tail > static_cast<uint64_t>(INT64_MAX) / udim) {
                    tail_overflow = true;
                    break;
                }
                tail *= udim;
            }
            if (tail_overflow) {
                n_dropped_bad_shape++;
                warn_drop(tensor_name.c_str(), "trailing shape dims overflow when flattened");
                continue;
            }
            shape[0] = shape_val->arr[0].as_int();
            shape[1] = static_cast<int64_t>(tail);
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

        // Per-tensor offset/size validation (F4): reject swap, OOB end, shape-vs-byte-count
        // mismatch (expected_nbytes = nelem*wire_bytes). #1603: no lenient branch any more; the old
        // one never checked offset_start and its own check summed two file-controlled uint64s,
        // wrapping for offset_end >= 2^64 - tensor_data_offset (validate_header_size now rejects that).
        if (nelem != 0 && dt->wire_bytes > UINT64_MAX / nelem) {
            n_dropped_bad_shape++;
            warn_drop(tensor_name.c_str(), "shape times dtype width overflows");
            continue;
        }
        uint64_t expected_nbytes = nelem * static_cast<uint64_t>(dt->wire_bytes);
        std::string vt_err;
        if (!safetensors_internal::validate_tensor_offsets(offset_start, offset_end, expected_nbytes,
                                                           tensor_data_offset, file_size, &vt_err)) {
            n_dropped_offset_validation++;
            warn_drop(tensor_name.c_str(), vt_err.c_str());
            continue;
        }

        void* tensor_ptr = tensor_data_base + offset_start;
        if (sparse && expected_nbytes > 0) {
            // Page-aligned WILLNEED on this tensor's bytes only.
            static const uintptr_t kPage = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
            const uintptr_t lo = reinterpret_cast<uintptr_t>(tensor_ptr) & ~(kPage - 1);
            const uintptr_t hi = reinterpret_cast<uintptr_t>(tensor_ptr) + expected_nbytes;
            madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_WILLNEED);
            sparse_willneed_bytes += expected_nbytes;
        }
        Tensor t(tensor_ptr, dtype, ndim, shape, /*on_device=*/false);
        (divert_to_mtp ? *mtp_out : tensor_map).emplace(tensor_name, t);

        IMP_LOG_DEBUG("Tensor: %s dtype=%s shape=[%ld%s%s%s%s] offsets=[%lu,%lu]", tensor_name.c_str(),
                      dtype_val->str_val.c_str(), (long)shape[0], ndim > 1 ? "," : "",
                      ndim > 1 ? std::to_string(shape[1]).c_str() : "", ndim > 2 ? "," : "",
                      ndim > 2 ? std::to_string(shape[2]).c_str() : "", (unsigned long)offset_start,
                      (unsigned long)offset_end);
    }

    int n_total_dropped = n_dropped_no_dtype + n_dropped_no_shape + n_dropped_too_many_dims +
                          n_dropped_no_offsets + n_dropped_offset_validation + n_dropped_dtype_unsupported +
                          n_dropped_bad_shape;
    if (n_total_dropped > 0) {
        IMP_LOG_WARN(
            "SafeTensors %s: dropped %d malformed tensors (no_dtype=%d no_shape=%d "
            "too_many_dims=%d no_offsets=%d offset_validation=%d dtype_unsupported=%d bad_shape=%d)",
            path.c_str(), n_total_dropped, n_dropped_no_dtype, n_dropped_no_shape, n_dropped_too_many_dims,
            n_dropped_no_offsets, n_dropped_offset_validation, n_dropped_dtype_unsupported,
            n_dropped_bad_shape);
    }
    if (sparse) {
        IMP_LOG_INFO("SafeTensors %s: sparse map, %d unused tensors skipped, WILLNEED on %zu of %zu MiB",
                     path.c_str(), n_sparse_unused, sparse_willneed_bytes >> 20, file_size >> 20);
    }

    return true;
}

// A shard name from model.safetensors.index.json is file content, not an operator path, and
// gets concatenated onto the model directory and opened/mmap'd: a separator escapes the
// directory ("../../../etc/hostname", or an absolute path via "dir"+"/"+"/etc/shadow").
// Rule: a bare filename, nothing else. GGUF split shards derive names from the operator's
// own path instead, so this does not apply there.
bool safetensors_shard_name_is_safe(const std::string& name) {
    if (name.empty())
        return false;
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
        return false;
    if (name == "." || name == "..")
        return false;
    return true;
}

// ---- Sharded SafeTensors loading ----

// mtp_out: same contract as load_shard's - non-null means an embedded MTP head is wanted,
// so mtp.* tensors are collected and a shard made only of them is NOT dropped. Passing
// nullptr (the old implicit behavior) silently loses the draft head on every sharded
// llm-compressor checkpoint: translate_name SKIPs the names, so they never reach the map.
static bool load_sharded(const std::string& model_dir, std::unordered_map<std::string, Tensor>& tensor_map,
                         std::vector<ShardInfo>& shards, bool keep_vision,
                         std::unordered_map<std::string, Tensor>* mtp_out) {
    std::string index_path = model_dir + "/model.safetensors.index.json";

    // Read the index file
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        IMP_LOG_ERROR("Failed to open index: %s", index_path.c_str());
        return false;
    }
    std::string index_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    JsonParser parser(index_json);
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
        if (kv.value.type == JType::STRING) {
            if (!safetensors_shard_name_is_safe(kv.value.str_val)) {
                IMP_LOG_ERROR("Shard name escapes the model directory: '%s' (tensor '%s' in %s)",
                              kv.value.str_val.c_str(), kv.key.c_str(), index_path.c_str());
                return false;
            }
            shard_tensors[kv.value.str_val].push_back(kv.key);
        }
    }

    // Drops shards where translate_name would skip every tensor, saving mmap + header parse +
    // page cache pressure for an unused file. keep_vision must reach this loop, not just
    // translate_name: a vision tower usually ships as its own shard, so the drop decides its
    // fate before any tensor is looked at.
    std::set<std::string> shard_files;
    std::set<std::string> sparse_shards;
    const bool keep_mtp = mtp_out != nullptr;
    for (auto& [fname, tensors] : shard_tensors) {
        auto all_unused = [&](bool mtp) {
            return !tensors.empty() && std::all_of(tensors.begin(), tensors.end(), [&](const std::string& n) {
                       return imp::llm_compressor::name_is_unused(n, keep_vision, mtp);
                   });
        };
        if (all_unused(keep_mtp)) {
            IMP_LOG_INFO("Skipping shard %s (%zu tensors are MTP/vision-only and unused)", fname.c_str(),
                         tensors.size());
            continue;
        }
        // Kept only for its MTP tensors, next to unused ones: map sparse (see load_shard).
        const bool any_unused = std::any_of(tensors.begin(), tensors.end(), [&](const std::string& n) {
            return imp::llm_compressor::name_is_unused(n, keep_vision, keep_mtp);
        });
        if (keep_mtp && all_unused(false) && any_unused)
            sparse_shards.insert(fname);
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
    // Per-shard MTP collection, merged below: the head's tensors are not
    // guaranteed to share one shard, and the maps are filled in parallel.
    std::vector<std::unordered_map<std::string, Tensor>> per_shard_mtp(shard_list.size());
    std::atomic<bool> any_failure{false};

    std::atomic<size_t> shards_done{0};
    const size_t total_shards = shard_list.size();
    auto worker = [&](size_t i) {
        std::string shard_path = model_dir + "/" + shard_list[i];
        if (!load_shard(shard_path, per_shard_maps[i], per_shard_info[i], llm_compressor_format, keep_vision,
                        per_shard_counters[i], keep_mtp ? &per_shard_mtp[i] : nullptr,
                        sparse_shards.count(shard_list[i]) > 0)) {
            IMP_LOG_ERROR("Failed to load shard: %s", shard_path.c_str());
            any_failure.store(true);
        }
        const size_t done = shards_done.fetch_add(1) + 1;
        IMP_LOG_INFO("  [%zu/%zu] mmap'd shard: %s (%zu tensors)", done, total_shards, shard_list[i].c_str(),
                     per_shard_maps[i].size());
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
        if (keep_mtp)
            for (auto& kv : per_shard_mtp[i])
                mtp_out->emplace(kv.first, kv.second);
        shards.push_back(per_shard_info[i]);
        const auto& c = per_shard_counters[i];
        tcounters.suffix_renames += c.suffix_renames;
        tcounters.prefix_strips += c.prefix_strips;
        tcounters.vision_skipped += c.vision_skipped;
        tcounters.vision_kept += c.vision_kept;
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

// Name-only MTP presence check across all three layouts a head can carry (sidecar
// model_mtp.safetensors, sharded index.json entry, single-file header), via the same
// name_is_mtp_head_key() dispatch_mtp() keys its shapes on, so "yes" here means enabling
// would actually load. Only names are read, never a weight, so this costs nothing otherwise.
bool probe_mtp_head(const std::string& model_dir) {
    namespace fs = std::filesystem;
    if (model_dir.empty())
        return false;

    auto json_has_head = [](const char* data, size_t len) {
        JsonParser parser(std::string_view(data, len));
        JValue root = parser.parse();
        if (!parser.ok() || root.type != JType::OBJECT)
            return false;
        // The index nests names under weight_map; a shard header lists them at
        // the top level.
        const JValue* map = jobj_find(root, "weight_map");
        const JValue& obj = (map && map->type == JType::OBJECT) ? *map : root;
        for (const auto& kv : obj.obj)
            if (name_is_mtp_head_key(kv.key))
                return true;
        return false;
    };

    // Read a safetensors header without mapping the body.
    auto header_has_head = [&](const std::string& file) {
        std::error_code ec;
        auto total = fs::file_size(file, ec);
        if (ec || total < 8)
            return false;
        std::ifstream f(file, std::ios::binary);
        if (!f.is_open())
            return false;
        uint64_t header_size = 0;
        if (!f.read(reinterpret_cast<char*>(&header_size), sizeof(header_size)))
            return false;
        std::string vh_err;
        if (!safetensors_internal::validate_header_size(total, header_size, &vh_err))
            return false;
        std::string header(static_cast<size_t>(header_size), '\0');
        if (!f.read(header.data(), static_cast<std::streamsize>(header_size)))
            return false;
        return json_has_head(header.data(), header.size());
    };

    if (header_has_head(model_dir + "/model_mtp.safetensors"))
        return true;

    std::ifstream idx(model_dir + "/model.safetensors.index.json");
    if (idx.is_open()) {
        std::string text((std::istreambuf_iterator<char>(idx)), std::istreambuf_iterator<char>());
        return json_has_head(text.data(), text.size());
    }
    return header_has_head(model_dir + "/model.safetensors");
}

// Reconstructs the checkpoint's declared partition from the tensor map and logs it; rules
// live in nvfp4_module_policy.h. Driven off the tensor map rather than imp's slots because
// the ignore list is written in the checkpoint's own namespace, which a slot cannot name.
static bool nvfp4_inventory_refuses(const std::unordered_map<std::string, Tensor>& tensor_map,
                                    const ModelConfig& cfg, std::string* why) {
    namespace pol = imp::nvfp4_policy;
    std::vector<pol::SlotObservation> slots;
    for (const auto& [name, t] : tensor_map) {
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0)
            continue;
        const std::string module = name.substr(0, name.size() - 7);
        pol::SlotObservation s;
        s.name = name;
        s.ndim = t.ndim;
        s.has_micro_scale = tensor_map.count(module + ".weight_scale") > 0;
        s.has_global_scale = tensor_map.count(module + ".weight_scale_2") > 0;
        // A packed NVFP4 weight still carries its on-disk [N, K/2] byte width
        // here; the logical K is what the group-size rule is about.
        s.K = (t.ndim == 2) ? (s.has_micro_scale ? 2 * t.shape[1] : t.shape[1]) : 0;
        slots.push_back(std::move(s));
    }
    const pol::Inventory inv = pol::classify(slots, cfg.nvfp4_exclude_modules);
    // Slots and ignore entries are two different populations and the line keeps
    // them apart: "1 ignored" next to a 170-entry list reads as "169 dropped".
    IMP_LOG_INFO("NVFP4 inventory: %d Linear modules quantized, %d ignored, %d unclassified, "
                 "%d missing global scale; quantization_config.ignore %d entries = %d on a Linear "
                 "slot, %d outside the Linear set (vision tower, conv1d, embeddings), %d with no "
                 "tensor in the map (MTP head, dropped shards)",
                 inv.quantized, inv.ignored, inv.unclassified, inv.missing_global_scale,
                 inv.ignore_entries, inv.ignore_on_linear_slot, inv.ignore_outside_linear_set,
                 inv.ignore_unmatched);
    return pol::refuses(inv, cfg.is_llm_compressor_nvfp4, why);
}

static bool gptq_projection_ok(TransformerLayer::GPTQWeight& gw, int group_size, std::string* why) {
    const bool dtypes_ok = gw.qweight.qtype == QType::INT32 && gw.qweight.ndim == 2 && gw.qzeros.data &&
                           gw.qzeros.qtype == QType::INT32 && gw.qzeros.ndim == 2 && gw.scales.data &&
                           gw.scales.qtype == QType::F16 && gw.scales.ndim == 2;
    if (!dtypes_ok) {
        *why = "needs qweight INT32 2-D, qzeros INT32 2-D, scales F16 2-D";
        return false;
    }
    gptq::Dims d;
    if (!gptq::check_shapes(gw.qweight.shape, gw.qzeros.shape, gw.scales.shape, group_size, &d, why))
        return false;
    gw.group_size = d.group_size;
    if (!gw.g_idx.data)
        return true;
    if (gw.g_idx.qtype != QType::INT32 || gw.g_idx.ndim != 1) {
        *why = "g_idx is not INT32 1-D";
        return false;
    }
    return gptq::check_g_idx(static_cast<const int32_t*>(gw.g_idx.data), gw.g_idx.shape[0], d, why);
}

// GPTQ (#2249): checks every projection and stamps bits/group_size/zero format for upload.
// Refuses unknown checkpoint formats, bits != 4, bad shapes, and a .qweight with no projection slot.
static bool gptq_refuses(Model& model, const std::unordered_map<std::string, Tensor>& tensor_map,
                         const std::string& model_dir) {
    HFConfigLoader::GPTQConfig c;
    if (!HFConfigLoader::load_gptq_config(model_dir, c))
        return false;
    const std::string detected = "bits=" + std::to_string(c.bits) +
                                 " group_size=" + std::to_string(c.group_size) +
                                 " desc_act=" + (c.desc_act ? "true" : "false") + " checkpoint_format=" +
                                 (c.checkpoint_format.empty() ? "unspecified" : c.checkpoint_format);
    gptq::ZeroFormat fmt;
    if (c.bits != 4 || !gptq::parse_zero_format(c.checkpoint_format, &fmt)) {
        IMP_LOG_ERROR(
            "GPTQ SafeTensors detected (%s): variant not supported. Only bits=4 with checkpoint_format "
            "gptq (v1, default) or gptq_v2 dequantizes (#2249).",
            detected.c_str());
        return true;
    }
    static constexpr const char* kProj[] = {"q_proj",    "k_proj",  "v_proj",   "o_proj",
                                            "gate_proj", "up_proj", "down_proj"};
    size_t n_proj = 0;
    for (size_t li = 0; li < model.layers_.size(); ++li) {
        auto& L = model.layers_[li];
        size_t pi = 0;
        for (auto* gw :
             {&L.gptq_q, &L.gptq_k, &L.gptq_v, &L.gptq_o, &L.gptq_gate, &L.gptq_up, &L.gptq_down}) {
            const char* proj = kProj[pi++];
            if (!gw->qweight.data)
                continue;
            std::string why;
            if (!gptq_projection_ok(*gw, c.group_size, &why)) {
                IMP_LOG_ERROR("GPTQ SafeTensors (%s) refused: layer %zu: %s", detected.c_str(), li,
                              why.c_str());
                return true;
            }
            // #2253: desc_act=true without g_idx would dequantize with sequential groups.
            if (c.desc_act && !gw->g_idx.data) {
                IMP_LOG_ERROR(
                    "GPTQ SafeTensors (%s) refused: layer %zu %s: desc_act=true but no g_idx tensor",
                    detected.c_str(), li, proj);
                return true;
            }
            gw->bits = 4;
            gw->desc_act = c.desc_act;
            gw->zero_offset = static_cast<int>(fmt);
            ++n_proj;
        }
    }
    size_t n_qweight = 0;
    for (const auto& kv : tensor_map)
        if (kv.first.size() > 8 && kv.first.compare(kv.first.size() - 8, 8, ".qweight") == 0)
            ++n_qweight;
    if (n_proj != n_qweight) {
        IMP_LOG_ERROR(
            "GPTQ SafeTensors (%s) refused: %zu .qweight tensors, %zu on a q/k/v/o/gate/up/down slot",
            detected.c_str(), n_qweight, n_proj);
        return true;
    }
    IMP_LOG_INFO("GPTQ 4-bit: %zu projections, %s, zero offset +%d, dequantized to FP16 at upload", n_proj,
                 detected.c_str(), static_cast<int>(fmt));
    return false;
}

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

    // Whether to keep model.visual.*. Decided here from config.json alone, since load_config()
    // runs only after shards are mapped and a dropped shard cannot be recovered later. A
    // checkpoint with no vision_config answers false and loads byte-for-byte as before.
    const bool keep_vision = imp::HFConfigLoader::probe_vision_tower(model_dir);

    // Try loading tensors. embedded_mtp_map collects `mtp.*` tensors that
    // dense Qwen3.6 checkpoints embed in the main shard (no sidecar); only
    // populated when the caller asked for the MTP head.
    bool loaded = false;
    std::unordered_map<std::string, Tensor> embedded_mtp_map;
    auto* mtp_collect = load_mtp_head ? &embedded_mtp_map : nullptr;

    if (!single_file.empty()) {
        // Single file mode
        ShardInfo shard;
        loaded = load_shard(single_file, tensor_map, shard, llm_compressor_format, keep_vision, tcounters,
                            mtp_collect);
        if (loaded)
            shards.push_back(shard);
    } else {
        // Directory mode: try sharded first, then single
        std::string index_path = model_dir + "/model.safetensors.index.json";
        if (fs::exists(index_path)) {
            loaded = load_sharded(model_dir, tensor_map, shards, keep_vision, mtp_collect);
        }
        if (!loaded) {
            std::string st_path = model_dir + "/model.safetensors";
            if (fs::exists(st_path)) {
                ShardInfo shard;
                loaded = load_shard(st_path, tensor_map, shard, llm_compressor_format, keep_vision, tcounters,
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
         tcounters.vision_kept + tcounters.gemma4_extras + tcounters.passed_through) > 0) {
        imp::llm_compressor::log_summary(tcounters);
    }

    if (!loaded || tensor_map.empty()) {
        IMP_LOG_ERROR("Failed to load SafeTensors from %s", path.c_str());
        return nullptr;
    }

    IMP_LOG_INFO("Parsed %zu tensors from SafeTensors", tensor_map.size());

    // Detects + loads the MTP head sidecar (DeepSeek-V3-family, e.g. Qwen3.6): a separate host
    // tensor_map keyed by raw mtp.* names, dispatched to MtpHead fields; mmap retained via
    // Model::split_mmaps_ so Tensor pointers stay valid. Gated on load_mtp_head: the head is
    // ~1.57 GiB BF16 (Qwen3.6) of dead VRAM unless MTP spec-decode is actually enabled.
    std::optional<imp::MtpHead> mtp_local;
    std::unordered_map<std::string, Tensor> mtp_tensor_map;
    ShardInfo mtp_shard{};

    // Set when the checkpoint carries an MTP head this load did not take
    // (#1537), so /health can report it instead of only the startup log.
    bool mtp_available_unloaded = false;
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
                                         /*llm_compressor_format=*/false,
                                         /*keep_vision=*/false, mtp_counters);
            if (mtp_loaded) {
                mtp_local = dispatch_mtp_head(mtp_tensor_map, mtp_path, static_cast<size_t>(sz));
            } else {
                IMP_LOG_WARN("MTP head file %s present but failed to load", mtp_path.c_str());
            }
        } else {
            // Embedded variant: llm-compressor checkpoints divert their mtp.* tensors into
            // embedded_mtp_map during load_shard (translate_name SKIPs them); Model-Optimizer
            // checkpoints keep them in the main tensor_map (weight_map skips them later) - harvest
            // from there instead.
            if (embedded_mtp_map.empty()) {
                for (const auto& kv : tensor_map) {
                    if (!name_is_mtp_tensor(kv.first))
                        continue;
                    const bool prefixed = kv.first.rfind("model.mtp.", 0) == 0;
                    embedded_mtp_map.emplace(prefixed ? kv.first.substr(6) : kv.first, kv.second);
                }
            }
            if (!embedded_mtp_map.empty()) {
                size_t bytes = 0;
                for (const auto& kv : embedded_mtp_map)
                    bytes += kv.second.nbytes();
                mtp_local = dispatch_mtp_head(embedded_mtp_map, path + " (embedded mtp.*)", bytes);
            }
        }
    } else if (probe_mtp_head(model_dir)) {
        // The caller asked for no head and this checkpoint has one: say so once, since
        // speculative.mtp_k=auto declines the head outside a single-stream run, and an option
        // nobody is told about isn't a choice the operator gets to make. Costs a name scan, no bytes.
        IMP_LOG_INFO(
            "MTP head present in this checkpoint but not loaded. speculative.mtp_k=auto takes it "
            "only on a single-stream run (max_batch_size=1) with runtime.deterministic off; "
            "force it with --set speculative.mtp_k=2 --set speculative.ngram=false: measured "
            "+17-21 %% plain and +27-30 %% thinking single-stream decode on Qwen3.8-27B-NVFP4 "
            "(2026-08-27), in exchange for the head's VRAM (0.79 GiB there) and eager-equal "
            "greedy trajectories. See docs/LIMITATIONS.md");
        mtp_available_unloaded = true;
    }

    // Create model
    auto model = std::make_unique<Model>();
    model->mtp_head_available_unloaded_ = mtp_available_unloaded;
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
    bool has_config = HFConfigLoader::load_config(model_dir, cfg, &model->vision_tower);

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

    // SafeTensors stores Q/K in HF-native NeoX-style (rotate-half) RoPE: pairs (i, i+rope_dim/2).
    // The arch table defaults LLAMA/MISTRAL/MIXTRAL/LLAMA4 to interleaved because GGUF
    // pre-permutes Q/K so interleaved reproduces NeoX; that permutation is NOT applied to
    // SafeTensors weights, so the interleaved default scrambles positions (coherent but
    // prompt-blind output). Force NeoX on the SafeTensors path; rope_neox=true arches unaffected.
    if (cfg.arch == ModelArch::LLAMA || cfg.arch == ModelArch::MISTRAL || cfg.arch == ModelArch::MIXTRAL ||
        cfg.arch == ModelArch::LLAMA4) {
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

    // Every number below came out of the file (config.json, or inferred from tensor names when
    // absent), so it is checked before it sizes anything: num_hidden_layers alone reaches
    // 18.9 TiB at INT_MAX.
    {
        std::string dim_err;
        if (!validate_declared_dimensions(cfg, &dim_err)) {
            IMP_LOG_ERROR("SafeTensors: %s", dim_err.c_str());
            return nullptr;
        }
    }
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
    // 6a. Qwen4Exp PLE: the layer's projections came through the weight map, its n-gram table
    // (F8 shards + I64 hash buffers) is opened host-side. Missing table = unservable, refuse.
    for (size_t i = 0; i < model->layers_.size(); i++) {
        if (model->layers_[i].ple_key_proj.data == nullptr)
            continue;
        model->ngram_table_ = NGramTable::open(model_dir, static_cast<int>(i), cfg.ple_eos_token_id);
        if (!model->ngram_table_) {
            IMP_LOG_ERROR("SafeTensors: layer %zu has PLE weights but no n-gram table; refusing to load", i);
            return nullptr;
        }
    }

    // 6b. GPTQ: validate projections, stamp bits/group_size/zero format for upload_gptq_weight.
    if (gptq_refuses(*model, tensor_map, model_dir))
        return nullptr;

    // NVFP4: scale tensors were already routed into model->nvfp4_scratch_ by weight_map.cpp;
    // executor_pre_dequant.cpp Phase 0 promote() resolves them back onto the weight's sidecars,
    // no load-side linking needed here. MXFP4: gpt-oss experts decode natively (transcoded
    // MXFP4->NVFP4 at init, run through CUTLASS NVFP4 grouped GEMM); other MXFP4 SafeTensors
    // archs have no decode path yet and still need the GGUF conversion warning.
    HFConfigLoader::MxFP4Config mxfp4_cfg;
    if (HFConfigLoader::load_mxfp4_config(model_dir, mxfp4_cfg)) {
        cfg.is_mxfp4_prequant = true;
        cfg.mxfp4_block_size = mxfp4_cfg.block_size;
        if (cfg.arch == ModelArch::GPT_OSS) {
            IMP_LOG_INFO(
                "MXFP4 SafeTensors (block_size=%d): gpt-oss experts will be "
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

    if (awq_refuses(*model, cfg, tensor_map, model_dir))
        return nullptr;

    HFConfigLoader::NvFP4Config nvfp4_cfg;
    bool is_nvfp4 = HFConfigLoader::load_nvfp4_config(model_dir, nvfp4_cfg);
    if (is_nvfp4) {
        cfg.is_nvfp4_prequant = true;
        cfg.nvfp4_group_size = nvfp4_cfg.group_size;
        cfg.is_llm_compressor_nvfp4 = (nvfp4_cfg.format == HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
        cfg.kv_cache_quant_hint = nvfp4_cfg.kv_cache_quant_algo;
        cfg.nvfp4_exclude_modules = nvfp4_cfg.exclude_modules;
        std::string refusal;
        if (nvfp4_inventory_refuses(tensor_map, cfg, &refusal)) {
            IMP_LOG_ERROR("%s", refusal.c_str());
            return nullptr;
        }
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

    // Cross-checks the author's tie_word_embeddings flag against actual lm_head.weight
    // presence. Mismatch is a real surprise: most models tie, so silently tying when the
    // author declared tie=false would mask a genuinely missing lm_head.
    const bool out_proj_missing = (model->out_proj_.data == nullptr && model->tok_emb_.data != nullptr);
    if (cfg.tie_word_embeddings == 0 && out_proj_missing) {
        IMP_LOG_WARN(
            "config.json declares tie_word_embeddings=false but lm_head.weight "
            "is absent in the SafeTensors files; tying anyway as a fallback.");
    }
    if (cfg.tie_word_embeddings == 1 && model->out_proj_.data != nullptr && model->tok_emb_.data != nullptr &&
        model->out_proj_.data != model->tok_emb_.data) {
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
            // SentencePiece-only checkpoint (older Llama 1/2, Mistral, ...): native protobuf parser
            // populates vocab+scores+token types, encode_spm() handles the rest. BPE-from-spm
            // checkpoints share the same vocab table; the score-based encoder produces equivalent
            // output for most practical text.
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

    // tokenizer_config.json tokenizer-side flags (add_bos_token, add_prefix_space), mirroring
    // gguf_loader.cpp's tokenizer.ggml.add_bos_token/add_space_prefix. Without this the
    // SafeTensors path used the hardcoded add_bos_=true default, wrong for any model shipping
    // add_bos_token=false (e.g. Qwen3-Coder-30B-A3B-FP4 auto-prepending <|endoftext|>).
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
            // Resolves BOS/EOS token IDs from the token content strings. Handles models (e.g.
            // DeepSeek) whose BOS string is not in tokenizer.cpp's hardcoded detection list: they land
            // in added_tokens and are already in the vocabulary, only the ID needs wiring.
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

    // generation_config.json: sampling/EOS defaults shipped by the model author, loaded into
    // model->generation_config_ for engine + CLI consumers. EOS IDs are additionally pushed
    // onto the tokenizer's eos list so the existing stop-condition path picks them up.
    if (!model_dir.empty()) {
        HFConfigLoader::load_generation_config(model_dir, model->generation_config_);
        if (model->tokenizer_) {
            for (int32_t eid : model->generation_config_.eos_token_ids) {
                model->tokenizer_->add_eos_id(eid);
            }
        }
    }

    // Cross-checks special_tokens_map.json against the loaded tokenizer's special-flag column.
    // The model author's list is authoritative: a string in additional_special_tokens that
    // exists in vocab but isn't marked CONTROL (token_type=3) gets patched, caught by the
    // engine's banned-token scan.
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

    // Validates config-promised biases against actual tensor presence: some HF configs declare
    // attention_bias/mlp_bias but the export omits the tensors. Without this the loader
    // silently leaves bias slots null and inference runs with undefined behavior depending on
    // which kernels short-circuit on null biases.
    if (cfg.attention_bias == 1) {
        int missing_q = 0, missing_k = 0, missing_v = 0;
        for (const auto& layer : model->layers_) {
            if (layer.q_bias.data == nullptr)
                missing_q++;
            if (layer.k_bias.data == nullptr)
                missing_k++;
            if (layer.v_bias.data == nullptr)
                missing_v++;
        }
        if (missing_q || missing_k || missing_v) {
            IMP_LOG_WARN(
                "config.json says attention_bias=true but %d/%d Q-biases, "
                "%d/%d K-biases, %d/%d V-biases are missing from the SafeTensors "
                "export. Inference will proceed without those biases.",
                missing_q, static_cast<int>(model->layers_.size()), missing_k,
                static_cast<int>(model->layers_.size()), missing_v, static_cast<int>(model->layers_.size()));
        }
    }

    if (cfg.arch_inferred_fallback) {
        IMP_LOG_WARN(
            "Architecture detection fell back to GENERIC + tensor-name "
            "heuristics (config.json had no recognized architectures/model_type). "
            "Inference may be incoherent.");
    }

    // gpt-oss post-load pass (#547): de-interleaves the fused gate_up expert bias (g0,u0,g1,
    // u1,...) into separate gate/up bias tensors for standard per-projection bias plumbing.
    // Packed MXFP4 expert weights stay host-mmap'd; MXFP4->NVFP4 conversion happens at
    // executor pre-dequant. Host copies live in host_owned_buffers_ (freed in ~Model).
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
        IMP_LOG_INFO("gpt-oss: expert gate_up biases de-interleaved for %zu layers", model->layers_.size());

        // Residual-stream 2^-4 rescale (#547): gpt-oss's BF16 activations overflow FP16 hidden
        // (inf by L23, NaN logits). FP16 scaling is a lossless exponent shift; RMSNorm is
        // scale-invariant and lm_head reads only normed values, so scaling every h-contributor is
        // exact. Contributors: embeddings (cfg.embed_scale), Wo+o_bias+expert down bias (scaled
        // here), expert down weights (tensor_scales in pre_dequant_phase3_nvfp4_decode.cpp).
        auto scale_bf16_pow2 = [&](Tensor& t, int neg_exp) -> bool {
            if (!t.data || t.on_device)
                return true;
            // An NVFP4-packed Wo (U8 nibbles, INT8 wire qtype until Phase 0 promotes it) carries
            // the 2^-4 in its tensor_scale instead: pre_dequant_phase0_nvfp4_loader.cpp.
            if (cfg.is_nvfp4_prequant && t.qtype == QType::INT8)
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
            rescale_ok = rescale_ok && scale_bf16_pow2(layer.wo, 4) && scale_bf16_pow2(layer.o_bias, 4) &&
                         scale_bf16_pow2(layer.expert_down_bias, 4);
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

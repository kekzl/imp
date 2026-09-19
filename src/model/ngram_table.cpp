#include "model/ngram_table.h"

#include "core/logging.h"
#include "model/json_util.h"
#include "model/model_limits.h"
#include "model/safetensors_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>

namespace imp {

namespace {

constexpr int kMaxTableShards = 1024;
constexpr int kMaxNGramHeads = 256;
constexpr int kMaxNGramSize = 8;

// Round-to-nearest-even float -> FP16 bits (host, no CUDA headers here).
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu)  // inf / nan
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f)
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u)))
            half_mant++;
        return static_cast<uint16_t>(sign | half_mant);
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        half++;
    return static_cast<uint16_t>(half);
}

// FP8 E4M3 (bias 7, no inf, 0x7f/0xff NaN) -> float.
float fp8_e4m3_to_f32(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xf;
    const int mant = b & 7;
    float v;
    if (exp == 0)
        v = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    else if (exp == 15 && mant == 7)
        v = std::nanf("");
    else
        v = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
    return sign ? -v : v;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// One SafeTensors file opened for header-driven access: I64 buffers by pread, shards by mmap.
struct StFile {
    int fd = -1;
    uint64_t file_size = 0;
    uint64_t data_offset = 0;  // 8 + header_size
    JValue root;
    ~StFile() {
        if (fd >= 0)
            ::close(fd);
    }
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            IMP_LOG_ERROR("ngram table: cannot open %s", path.c_str());
            return false;
        }
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size < 8) {
            IMP_LOG_ERROR("ngram table: %s is not a SafeTensors file", path.c_str());
            return false;
        }
        file_size = static_cast<uint64_t>(st.st_size);
        uint64_t header_size = 0;
        if (pread(fd, &header_size, 8, 0) != 8)
            return false;
        std::string err;
        if (!safetensors_internal::validate_header_size(file_size, header_size, &err)) {
            IMP_LOG_ERROR("ngram table: %s: %s", path.c_str(), err.c_str());
            return false;
        }
        std::string json(static_cast<size_t>(header_size), '\0');
        if (pread(fd, json.data(), json.size(), 8) != static_cast<ssize_t>(json.size()))
            return false;
        JsonParser parser(json);
        root = parser.parse();
        if (!parser.ok() || root.type != JType::OBJECT) {
            IMP_LOG_ERROR("ngram table: %s: header is not a JSON object", path.c_str());
            return false;
        }
        data_offset = 8 + header_size;
        return true;
    }
    // Locates `name`, checks dtype and byte span, returns the absolute file offset.
    bool locate(const std::string& name, const char* want_dtype, std::vector<int64_t>* shape,
                uint64_t* abs_offset, uint64_t* nbytes) const {
        const JValue* meta = jobj_find(root, name);
        if (!meta || meta->type != JType::OBJECT) {
            IMP_LOG_ERROR("ngram table: tensor %s missing from its shard header", name.c_str());
            return false;
        }
        const JValue* dt = jobj_find(*meta, "dtype");
        if (!dt || dt->type != JType::STRING || dt->str_val != want_dtype) {
            IMP_LOG_ERROR("ngram table: %s: dtype %s, expected %s", name.c_str(),
                          dt && dt->type == JType::STRING ? dt->str_val.c_str() : "?", want_dtype);
            return false;
        }
        const JValue* sh = jobj_find(*meta, "shape");
        const JValue* off = jobj_find(*meta, "data_offsets");
        if (!sh || sh->type != JType::ARRAY || !off || off->type != JType::ARRAY || off->arr.size() != 2)
            return false;
        shape->clear();
        uint64_t numel = 1;
        for (const auto& d : sh->arr) {
            if (d.type != JType::NUMBER || d.num_val < 0)
                return false;
            shape->push_back(d.as_int());
            numel *= static_cast<uint64_t>(d.as_int());
        }
        const size_t width = std::strcmp(want_dtype, "I64") == 0    ? 8
                             : std::strcmp(want_dtype, "BF16") == 0 ? 2
                                                                    : 1;
        const uint64_t start = static_cast<uint64_t>(off->arr[0].as_int());
        const uint64_t end = static_cast<uint64_t>(off->arr[1].as_int());
        std::string err;
        if (!safetensors_internal::validate_tensor_offsets(start, end, numel * width, data_offset, file_size,
                                                           &err)) {
            IMP_LOG_ERROR("ngram table: %s: %s", name.c_str(), err.c_str());
            return false;
        }
        *abs_offset = data_offset + start;
        *nbytes = end - start;
        return true;
    }
    bool read_i64(const std::string& name, std::vector<int64_t>& out, int max_count) const {
        std::vector<int64_t> shape;
        uint64_t off = 0, nbytes = 0;
        if (!locate(name, "I64", &shape, &off, &nbytes))
            return false;
        if (shape.size() != 1 || shape[0] <= 0 || shape[0] > max_count) {
            IMP_LOG_ERROR("ngram table: %s: shape not [1..%d]", name.c_str(), max_count);
            return false;
        }
        out.resize(static_cast<size_t>(shape[0]));
        return pread(fd, out.data(), nbytes, static_cast<off_t>(off)) == static_cast<ssize_t>(nbytes);
    }
};

int64_t py_mod(int64_t a, int64_t m) {
    int64_t r = a % m;
    if (r < 0)
        r += m;
    return r;
}

}  // namespace

void ngram_hash(const NGramHashParams& p, const int32_t* ctx, const int32_t* tokens, int n,
                int64_t* ids_out) {
    const int c = p.ngram_size - 1;
    const int heads_per_ngram = p.n_heads / c;
    // history = ctx ++ tokens; output row t reads history positions t + c - shift.
    auto hist = [&](int i) -> int64_t { return i < c ? ctx[i] : tokens[i - c]; };
    for (int t = 0; t < n; t++) {
        const int pos = t + c;
        int64_t shifted[kMaxNGramSize];
        bool blocked = false;  // an EOS at any position in [pos-s, pos-1] ends the segment
        for (int s = 0; s < p.ngram_size; s++) {
            if (s > 0 && hist(pos - s) == p.eos_token_id)
                blocked = true;
            shifted[s] = blocked ? p.eos_token_id : hist(pos - s);
        }
        uint64_t mixed = static_cast<uint64_t>(shifted[0]) * static_cast<uint64_t>(p.multipliers[0]);
        for (int g = 2; g <= p.ngram_size; g++) {
            mixed ^= static_cast<uint64_t>(shifted[g - 1]) * static_cast<uint64_t>(p.multipliers[g - 1]);
            const int64_t m = static_cast<int64_t>(mixed);
            for (int h = (g - 2) * heads_per_ngram; h < (g - 1) * heads_per_ngram; h++)
                ids_out[static_cast<size_t>(t) * p.n_heads + h] = py_mod(m, p.vocab_sizes[h]) + p.offsets[h];
        }
    }
}

NGramTable::~NGramTable() {
    if (map_ != nullptr)
        munmap(map_, map_size_);
}

std::unique_ptr<NGramTable> NGramTable::open(const std::string& model_dir, int layer, int eos_token_id) {
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream ifs(index_path);
    if (!ifs.is_open()) {
        IMP_LOG_ERROR("ngram table: no %s", index_path.c_str());
        return nullptr;
    }
    std::string index_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    JsonParser parser(index_json);
    JValue root = parser.parse();
    const JValue* wm = parser.ok() ? jobj_find(root, "weight_map") : nullptr;
    if (!wm || wm->type != JType::OBJECT) {
        IMP_LOG_ERROR("ngram table: %s has no weight_map", index_path.c_str());
        return nullptr;
    }
    const std::string key = ".layers." + std::to_string(layer) + ".ple.ple_embedding.";
    std::map<std::string, std::string> file_of;  // suffix after key -> shard file
    std::string prefix;
    for (const auto& kv : wm->obj) {
        const size_t at = kv.key.find(key);
        if (at == std::string::npos || kv.value.type != JType::STRING)
            continue;
        if (!safetensors_shard_name_is_safe(kv.value.str_val)) {
            IMP_LOG_ERROR("ngram table: shard name escapes the model directory: %s",
                          kv.value.str_val.c_str());
            return nullptr;
        }
        prefix = kv.key.substr(0, at + key.size());
        file_of[kv.key.substr(at + key.size())] = kv.value.str_val;
    }
    if (file_of.empty()) {
        IMP_LOG_ERROR("ngram table: no *%s* tensors in %s", key.c_str(), index_path.c_str());
        return nullptr;
    }
    std::map<std::string, std::unique_ptr<StFile>> files;
    auto file = [&](const std::string& suffix) -> StFile* {
        auto it = file_of.find(suffix);
        if (it == file_of.end()) {
            IMP_LOG_ERROR("ngram table: %s%s missing from the index", prefix.c_str(), suffix.c_str());
            return nullptr;
        }
        auto& f = files[it->second];
        if (!f) {
            f = std::make_unique<StFile>();
            if (!f->open(model_dir + "/" + it->second))
                return nullptr;
        }
        return f.get();
    };

    std::unique_ptr<NGramTable> t(new NGramTable());
    t->eos_ = eos_token_id;
    StFile* f;
    if (!(f = file("layer_multipliers")) ||
        !f->read_i64(prefix + "layer_multipliers", t->multipliers_, kMaxNGramSize))
        return nullptr;
    if (!(f = file("ngram_heads_offsets")) ||
        !f->read_i64(prefix + "ngram_heads_offsets", t->offsets_, kMaxNGramHeads))
        return nullptr;
    if (!(f = file("ngram_heads_vocab_sizes")) ||
        !f->read_i64(prefix + "ngram_heads_vocab_sizes", t->vocab_sizes_, kMaxNGramHeads))
        return nullptr;
    t->ngram_size_ = static_cast<int>(t->multipliers_.size());
    t->n_heads_ = static_cast<int>(t->offsets_.size());
    if (t->ngram_size_ < 2 || t->vocab_sizes_.size() != t->offsets_.size() ||
        t->n_heads_ % (t->ngram_size_ - 1) != 0) {
        IMP_LOG_ERROR("ngram table: ngram_size=%d heads=%d/%zu do not form (ngram_size-1) x heads_per_ngram",
                      t->ngram_size_, t->n_heads_, t->vocab_sizes_.size());
        return nullptr;
    }
    for (int h = 0; h < t->n_heads_; h++) {
        if (t->vocab_sizes_[h] <= 0 || t->offsets_[h] < 0) {
            IMP_LOG_ERROR("ngram table: head %d vocab_size=%lld offset=%lld", h,
                          static_cast<long long>(t->vocab_sizes_[h]), static_cast<long long>(t->offsets_[h]));
            return nullptr;
        }
    }

    // The scale and the shards share one file, mapped once without populate.
    if (!(f = file("ngram_embedding.weight_scale")))
        return nullptr;
    {
        std::vector<int64_t> shape;
        uint64_t off = 0, nbytes = 0;
        if (!f->locate(prefix + "ngram_embedding.weight_scale", "BF16", &shape, &off, &nbytes) || nbytes != 2)
            return nullptr;
        uint16_t b = 0;
        if (pread(f->fd, &b, 2, static_cast<off_t>(off)) != 2)
            return nullptr;
        t->scale_ = bf16_to_f32(b);
    }
    int n_shards = 0;
    for (const auto& [suffix, fname] : file_of) {
        if (suffix.rfind("ngram_embedding.shard_", 0) == 0)
            n_shards++;
    }
    if (n_shards == 0 || n_shards > kMaxTableShards) {
        IMP_LOG_ERROR("ngram table: %d shards (limit %d)", n_shards, kMaxTableShards);
        return nullptr;
    }
    void* base = mmap(nullptr, f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (base == MAP_FAILED) {
        IMP_LOG_ERROR("ngram table: mmap of %llu bytes failed",
                      static_cast<unsigned long long>(f->file_size));
        return nullptr;
    }
    madvise(base, f->file_size, MADV_RANDOM);
    t->map_ = base;
    t->map_size_ = f->file_size;
    for (int k = 0; k < n_shards; k++) {
        const std::string suffix = "ngram_embedding.shard_" + std::to_string(k) + ".weight";
        StFile* sf = file(suffix);
        if (sf == nullptr)
            return nullptr;
        if (sf != f) {
            IMP_LOG_ERROR("ngram table: shard_%d lives in another file than the scale; one file expected", k);
            return nullptr;
        }
        std::vector<int64_t> shape;
        uint64_t off = 0, nbytes = 0;
        if (!f->locate(prefix + suffix, "F8_E4M3", &shape, &off, &nbytes))
            return nullptr;
        if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1] > kMaxHeadDim) {
            IMP_LOG_ERROR("ngram table: shard_%d shape is not [rows, head_dim]", k);
            return nullptr;
        }
        if (k == 0)
            t->head_dim_ = static_cast<int>(shape[1]);
        else if (shape[1] != t->head_dim_) {
            IMP_LOG_ERROR("ngram table: shard_%d head_dim %lld != %d", k, static_cast<long long>(shape[1]),
                          t->head_dim_);
            return nullptr;
        }
        t->shards_.push_back({t->total_rows_, shape[0], static_cast<const uint8_t*>(base) + off});
        t->total_rows_ += shape[0];
    }
    for (int h = 0; h < t->n_heads_; h++) {
        if (t->offsets_[h] + t->vocab_sizes_[h] > t->total_rows_) {
            IMP_LOG_ERROR("ngram table: head %d reaches row %lld past %lld", h,
                          static_cast<long long>(t->offsets_[h] + t->vocab_sizes_[h]),
                          static_cast<long long>(t->total_rows_));
            return nullptr;
        }
    }
    IMP_LOG_INFO(
        "ngram table: layer %d, %d shards, %lld rows x %d, %d heads x ngram %d, scale %.5g, eos %d, "
        "%.1f GiB host-mapped (page cache, not resident)",
        layer, n_shards, static_cast<long long>(t->total_rows_), t->head_dim_, t->n_heads_, t->ngram_size_,
        t->scale_, eos_token_id,
        static_cast<double>(t->total_rows_) * t->head_dim_ / (1024.0 * 1024.0 * 1024.0));
    return t;
}

void NGramTable::hash(const int32_t* ctx, const int32_t* tokens, int n, int64_t* ids_out) const {
    NGramHashParams p{multipliers_.data(), offsets_.data(), vocab_sizes_.data(), ngram_size_, n_heads_, eos_};
    ngram_hash(p, ctx, tokens, n, ids_out);
}

void NGramTable::gather(const int64_t* ids, int n, uint16_t* out) const {
    const size_t count = static_cast<size_t>(n) * n_heads_;
    auto row_ptr = [&](int64_t id) -> const uint8_t* {
        if (id < 0 || id >= total_rows_)
            return nullptr;
        auto it = std::upper_bound(shards_.begin(), shards_.end(), id,
                                   [](int64_t v, const Shard& s) { return v < s.row_start; });
        --it;
        return it->data + static_cast<size_t>(id - it->row_start) * head_dim_;
    };
    const long page = sysconf(_SC_PAGESIZE);
    for (size_t i = 0; i < count; i++) {
        const uint8_t* p = row_ptr(ids[i]);
        if (p == nullptr)
            continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~static_cast<uintptr_t>(page - 1);
        const uintptr_t e = reinterpret_cast<uintptr_t>(p) + head_dim_;
        madvise(reinterpret_cast<void*>(a), e - a, MADV_WILLNEED);
    }
    for (size_t i = 0; i < count; i++) {
        uint16_t* o = out + i * head_dim_;
        const uint8_t* p = row_ptr(ids[i]);
        if (p == nullptr) {
            if (n_out_of_range_++ == 0)
                IMP_LOG_ERROR("ngram table: id %lld outside [0, %lld), row zeroed",
                              static_cast<long long>(ids[i]), static_cast<long long>(total_rows_));
            std::memset(o, 0, static_cast<size_t>(head_dim_) * 2);
            continue;
        }
        for (int j = 0; j < head_dim_; j++)
            o[j] = f32_to_f16(fp8_e4m3_to_f32(p[j]) * scale_);
    }
}

}  // namespace imp

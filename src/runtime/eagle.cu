#include "runtime/eagle.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "compute/activation.h"
#include "compute/attention_paged.h"
#include "compute/sampling.h"
#include "compute/rope.h"
#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace imp {

// ---------------------------------------------------------------------------
// Minimal JSON parser (reused from safetensors_loader.cpp pattern)
// ---------------------------------------------------------------------------

namespace {

enum class JType { NUL, STRING, NUMBER, ARRAY, OBJECT };

struct JValue {
    JType type = JType::NUL;
    std::string str_val;
    double num_val = 0.0;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;
    int64_t as_int() const { return static_cast<int64_t>(num_val); }
};

class JsonParser {
public:
    explicit JsonParser(const char* data, size_t len)
        : data_(data), len_(len), pos_(0) {}
    JValue parse() { skip_ws(); return parse_value(); }
    bool ok() const { return !error_; }
private:
    const char* data_; size_t len_; size_t pos_; bool error_ = false;
    char peek() const { return pos_ < len_ ? data_[pos_] : '\0'; }
    char advance() { if (pos_ >= len_) { error_ = true; return '\0'; } return data_[pos_++]; }
    void skip_ws() { while (pos_ < len_ && (data_[pos_]==' '||data_[pos_]=='\t'||data_[pos_]=='\n'||data_[pos_]=='\r')) pos_++; }
    bool expect(char c) { skip_ws(); if (peek()==c) { advance(); return true; } error_=true; return false; }
    JValue parse_value() {
        skip_ws(); if (error_) return {};
        char c = peek();
        if (c=='"') return parse_string_value();
        if (c=='{') return parse_object();
        if (c=='[') return parse_array();
        if (c=='t'||c=='f') return parse_bool();
        if (c=='n') return parse_null();
        if (c=='-'||(c>='0'&&c<='9')) return parse_number();
        error_=true; return {};
    }
    JValue parse_string_value() { JValue v; v.type=JType::STRING; v.str_val=parse_string_raw(); return v; }
    std::string parse_string_raw() {
        if (!expect('"')) return "";
        std::string s;
        while (pos_ < len_) {
            char c = advance();
            if (c=='"') return s;
            if (c=='\\') { if (pos_>=len_) { error_=true; return s; } char e=advance();
                switch(e) { case '"': s+='"'; break; case '\\': s+='\\'; break; case '/': s+='/'; break;
                    case 'n': s+='\n'; break; case 't': s+='\t'; break; case 'r': s+='\r'; break;
                    case 'u': for(int i=0;i<4&&pos_<len_;i++) advance(); s+='?'; break;
                    default: s+=e; break; }
            } else { s+=c; }
        }
        error_=true; return s;
    }
    JValue parse_number() {
        JValue v; v.type=JType::NUMBER; size_t start=pos_;
        if (peek()=='-') advance();
        while (pos_<len_&&data_[pos_]>='0'&&data_[pos_]<='9') advance();
        if (pos_<len_&&data_[pos_]=='.') { advance(); while(pos_<len_&&data_[pos_]>='0'&&data_[pos_]<='9') advance(); }
        if (pos_<len_&&(data_[pos_]=='e'||data_[pos_]=='E')) { advance(); if(pos_<len_&&(data_[pos_]=='+'||data_[pos_]=='-')) advance(); while(pos_<len_&&data_[pos_]>='0'&&data_[pos_]<='9') advance(); }
        v.num_val=std::stod(std::string(data_+start, pos_-start)); return v;
    }
    JValue parse_object() {
        JValue v; v.type=JType::OBJECT; if (!expect('{')) return v; skip_ws();
        if (peek()=='}') { advance(); return v; }
        while (!error_) { skip_ws(); std::string key=parse_string_raw(); if(!expect(':')) break;
            JValue val=parse_value(); v.obj.emplace_back(std::move(key),std::move(val));
            skip_ws(); if(peek()==',') { advance(); continue; } break; }
        expect('}'); return v;
    }
    JValue parse_array() {
        JValue v; v.type=JType::ARRAY; if (!expect('[')) return v; skip_ws();
        if (peek()==']') { advance(); return v; }
        while (!error_) { v.arr.push_back(parse_value()); skip_ws(); if(peek()==',') { advance(); continue; } break; }
        expect(']'); return v;
    }
    JValue parse_bool() { JValue v; v.type=JType::NUMBER;
        if(peek()=='t') { for(int i=0;i<4&&pos_<len_;i++) advance(); v.num_val=1.0; }
        else { for(int i=0;i<5&&pos_<len_;i++) advance(); v.num_val=0.0; } return v; }
    JValue parse_null() { JValue v; for(int i=0;i<4&&pos_<len_;i++) advance(); return v; }
};

static const JValue* jobj_find(const JValue& obj, const std::string& key) {
    for (const auto& kv : obj.obj) if (kv.first == key) return &kv.second;
    return nullptr;
}

static DType safetensors_dtype(const std::string& s) {
    if (s == "F32")  return DType::FP32;
    if (s == "F16")  return DType::FP16;
    if (s == "BF16") return DType::BF16;
    return DType::FP32;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// CUDA kernels for EAGLE forward
// ---------------------------------------------------------------------------

// FC fusion: out[i] = sum_j(weight[i][j] * concat[j]) + bias[i] + embed[i]
// weight: [d_model, 3*d_model], concat: [3*d_model], bias: [d_model], embed: [d_model]
// out: [d_model]
__global__ void eagle_fc_fusion_kernel(const half* __restrict__ weight,
                                        const half* __restrict__ concat,
                                        const half* __restrict__ bias,
                                        const half* __restrict__ embed,
                                        half* __restrict__ out,
                                        int d_model, int k_dim) {
    int row = blockIdx.x;
    if (row >= d_model) return;

    float acc = 0.0f;
    for (int j = threadIdx.x; j < k_dim; j += blockDim.x) {
        acc += __half2float(weight[row * k_dim + j]) * __half2float(concat[j]);
    }

    // Warp reduction
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    // Block reduction via shared memory
    __shared__ float sdata[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) sdata[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        acc = (lane_id < n_warps) ? sdata[lane_id] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, mask);
        if (lane_id == 0) {
            if (bias) acc += __half2float(bias[row]);
            acc += __half2float(embed[row]);
            out[row] = __float2half(acc);
        }
    }
}

// Simple add kernel: c[i] = a[i] + b[i]
__global__ void vec_add_kernel(const half* __restrict__ a,
                                const half* __restrict__ b,
                                half* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// Concat 3 feature vectors: out = [feat_low | feat_mid | feat_high]
__global__ void concat_features_kernel(const half* __restrict__ feat_low,
                                        const half* __restrict__ feat_mid,
                                        const half* __restrict__ feat_high,
                                        half* __restrict__ out,
                                        int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 3 * d_model;
    if (idx < total) {
        if (idx < d_model)
            out[idx] = feat_low[idx];
        else if (idx < 2 * d_model)
            out[idx] = feat_mid[idx - d_model];
        else
            out[idx] = feat_high[idx - 2 * d_model];
    }
}

// ---------------------------------------------------------------------------
// EagleDecoder implementation
// ---------------------------------------------------------------------------

EagleDecoder::~EagleDecoder() {
    free_buffers();
}

void EagleDecoder::free_buffers() {
    for (void* p : gpu_allocs_) {
        if (p) cudaFree(p);
    }
    gpu_allocs_.clear();
    d_concat_ = nullptr;
    d_fused_ = nullptr;
    d_eagle_hidden_ = nullptr;
    d_embed_out_ = nullptr;
    d_norm_out_ = nullptr;
    d_q_ = nullptr; d_k_ = nullptr; d_v_ = nullptr;
    d_attn_out_ = nullptr; d_proj_out_ = nullptr;
    d_gate_out_ = nullptr; d_up_out_ = nullptr;
    d_swiglu_out_ = nullptr; d_ffn_out_ = nullptr;
    d_eagle_logits_fp16_ = nullptr;
    d_eagle_logits_ = nullptr;
    d_token_buf_ = nullptr;
    d_pos_buf_ = nullptr;
    d_bt_buf_ = nullptr;
    d_cl_buf_ = nullptr;
    for (int i = 0; i < 3; i++) d_feat_bufs_[i] = nullptr;
}

void* EagleDecoder::gpu_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        IMP_LOG_ERROR("EAGLE: cudaMalloc(%zu) failed", bytes);
        return nullptr;
    }
    gpu_allocs_.push_back(p);
    return p;
}

bool EagleDecoder::init(GraphExecutor* target_executor, Model* target_model,
                        KVCacheManager* target_kv_manager, KVCache* target_kv_cache,
                        const EagleConfig& config, cudaStream_t stream) {
    if (!target_executor || !target_model || !target_kv_manager || !target_kv_cache) {
        IMP_LOG_ERROR("EAGLE: null target references");
        return false;
    }

    target_executor_ = target_executor;
    target_model_ = target_model;
    target_kv_manager_ = target_kv_manager;
    target_kv_cache_ = target_kv_cache;
    config_ = config;

    // Resolve feature layer indices
    const auto& mcfg = target_model_->config();
    // EAGLE-3 feature layers: (2, n_layers//2, n_layers-3) per reference impl.
    // Features are captured BEFORE the indexed layer (= output of layer i-1).
    feat_low_ = (config_.feat_layer_low >= 0) ? config_.feat_layer_low : 2;
    feat_mid_ = (config_.feat_layer_mid >= 0) ? config_.feat_layer_mid : mcfg.n_layers / 2;
    feat_high_ = (config_.feat_layer_high >= 0) ? config_.feat_layer_high : mcfg.n_layers - 3;

    IMP_LOG_INFO("EAGLE: feature layers: low=%d, mid=%d, high=%d (of %d)",
                 feat_low_, feat_mid_, feat_high_, mcfg.n_layers);

    // Allocate feature snapshot buffers (filled during target forward pass)
    size_t feat_bytes = mcfg.d_model * sizeof(half);
    for (int i = 0; i < 3; i++) {
        d_feat_bufs_[i] = static_cast<half*>(gpu_alloc(feat_bytes));
        if (!d_feat_bufs_[i]) return false;
        eagle_feat_ptrs_[i] = d_feat_bufs_[i];
    }

    return true;
}

bool EagleDecoder::load_head(const std::string& safetensors_path, cudaStream_t stream) {
    // 1. Open and mmap the SafeTensors file
    int fd = open(safetensors_path.c_str(), O_RDONLY);
    if (fd < 0) {
        IMP_LOG_ERROR("EAGLE: failed to open %s", safetensors_path.c_str());
        return false;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) {
        IMP_LOG_ERROR("EAGLE: file too small: %s", safetensors_path.c_str());
        close(fd);
        return false;
    }

    void* mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_base == MAP_FAILED) {
        IMP_LOG_ERROR("EAGLE: mmap failed: %s", safetensors_path.c_str());
        return false;
    }

    auto data = reinterpret_cast<const uint8_t*>(mmap_base);

    // 2. Read header
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, sizeof(uint64_t));
    if (8 + header_size > file_size) {
        IMP_LOG_ERROR("EAGLE: header exceeds file size");
        munmap(mmap_base, file_size);
        return false;
    }

    // 3. Parse JSON header
    const char* json_data = reinterpret_cast<const char*>(data + 8);
    JsonParser parser(json_data, static_cast<size_t>(header_size));
    JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT) {
        IMP_LOG_ERROR("EAGLE: failed to parse JSON header");
        munmap(mmap_base, file_size);
        return false;
    }

    size_t tensor_data_offset = 8 + static_cast<size_t>(header_size);
    const uint8_t* tensor_data_base = data + tensor_data_offset;

    // 4. Extract tensors and upload to GPU
    struct TensorInfo {
        std::string name;
        DType dtype;
        int64_t shape[4];
        int ndim;
        const void* host_ptr;
        size_t bytes;
    };

    std::vector<TensorInfo> tensors;

    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__") continue;
        const JValue& meta = kv.second;
        if (meta.type != JType::OBJECT) continue;

        const JValue* dtype_val = jobj_find(meta, "dtype");
        const JValue* shape_val = jobj_find(meta, "shape");
        const JValue* offsets_val = jobj_find(meta, "data_offsets");
        if (!dtype_val || !shape_val || !offsets_val) continue;

        DType dtype = safetensors_dtype(dtype_val->str_val);
        int ndim = static_cast<int>(shape_val->arr.size());
        if (ndim > 4) continue;

        int64_t shape[4] = {};
        for (int d = 0; d < ndim; d++) shape[d] = shape_val->arr[d].as_int();

        uint64_t off_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t off_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());
        size_t bytes = static_cast<size_t>(off_end - off_start);

        TensorInfo ti;
        ti.name = kv.first;
        ti.dtype = dtype;
        std::memcpy(ti.shape, shape, sizeof(shape));
        ti.ndim = ndim;
        ti.host_ptr = tensor_data_base + off_start;
        ti.bytes = bytes;
        tensors.push_back(std::move(ti));
    }

    IMP_LOG_INFO("EAGLE: parsed %zu tensors from %s", tensors.size(), safetensors_path.c_str());
    for (const auto& ti : tensors) {
        char shape_str[128] = {};
        int off = 0;
        for (int d = 0; d < ti.ndim; d++) {
            off += snprintf(shape_str + off, sizeof(shape_str) - off,
                            "%s%lld", d > 0 ? "x" : "", (long long)ti.shape[d]);
        }
        IMP_LOG_INFO("  %s: [%s] %s", ti.name.c_str(), shape_str,
                     ti.dtype == DType::FP16 ? "FP16" :
                     ti.dtype == DType::FP32 ? "FP32" :
                     ti.dtype == DType::BF16 ? "BF16" : "?");
    }

    // Helper: upload a tensor to GPU (converts BF16→FP16 on host)
    auto upload_tensor = [&](const TensorInfo& ti) -> Tensor {
        if (ti.dtype == DType::BF16) {
            // BF16→FP16 conversion on host via FP32 intermediate
            // BF16 has same exponent range as FP32 (8-bit), so values may exceed
            // FP16 max (65504). We clamp to FP16 range to avoid infinities.
            size_t n_elem = ti.bytes / 2;
            size_t fp16_bytes = n_elem * sizeof(uint16_t);
            std::vector<uint16_t> fp16_data(n_elem);
            const uint16_t* bf16_src = reinterpret_cast<const uint16_t*>(ti.host_ptr);
            for (size_t i = 0; i < n_elem; i++) {
                // BF16 → FP32 (zero-extend mantissa)
                uint32_t fp32_bits = static_cast<uint32_t>(bf16_src[i]) << 16;
                float f;
                memcpy(&f, &fp32_bits, sizeof(float));
                // Clamp to FP16 range to avoid inf/NaN
                if (std::isnan(f) || std::isinf(f)) f = 0.0f;
                else if (f > 65504.0f) f = 65504.0f;
                else if (f < -65504.0f) f = -65504.0f;
                // FP32 → FP16 (manual conversion with proper rounding)
                uint32_t fb;
                memcpy(&fb, &f, sizeof(float));
                uint32_t sign = (fb >> 31) & 1;
                int32_t exp = static_cast<int32_t>((fb >> 23) & 0xFF) - 127;
                uint32_t mant = fb & 0x7FFFFF;
                uint16_t fp16;
                if (exp > 15) fp16 = static_cast<uint16_t>((sign << 15) | (0x1F << 10) | 0x3FF); // max finite
                else if (exp < -24) fp16 = static_cast<uint16_t>(sign << 15); // zero
                else if (exp < -14) {
                    // Denormalized FP16
                    uint32_t shifted = (mant | 0x800000) >> (-(exp + 14) + 13);
                    fp16 = static_cast<uint16_t>((sign << 15) | (shifted & 0x3FF));
                } else {
                    fp16 = static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
                }
                fp16_data[i] = fp16;
            }
            void* d_ptr = gpu_alloc(fp16_bytes);
            if (!d_ptr) return {};
            cudaMemcpyAsync(d_ptr, fp16_data.data(), fp16_bytes,
                            cudaMemcpyHostToDevice, stream);
            return Tensor(d_ptr, DType::FP16, ti.ndim, ti.shape, true);
        }
        void* d_ptr = gpu_alloc(ti.bytes);
        if (!d_ptr) return {};
        cudaMemcpyAsync(d_ptr, ti.host_ptr, ti.bytes, cudaMemcpyHostToDevice, stream);
        return Tensor(d_ptr, ti.dtype, ti.ndim, ti.shape, true);
    };

    // Helper: find tensor by suffix
    auto find_tensor = [&](const std::string& suffix) -> const TensorInfo* {
        for (const auto& ti : tensors) {
            if (ti.name.size() >= suffix.size() &&
                ti.name.compare(ti.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                return &ti;
            }
        }
        return nullptr;
    };

    // 5. Upload EAGLE head weights
    // Naming conventions vary:
    //   v1: eagle_module.embed_tokens.weight, eagle_module.layers.0.*
    //   v3: lm_head.weight, midlayer.*, fc.weight, norm.weight

    // embed_tokens — EAGLE's own embedding (or shared with target)
    auto* ti_embed = find_tensor("embed_tokens.weight");
    if (ti_embed) {
        // Check if we can share with target embedding
        const auto& target_emb = target_model_->token_embedding();
        if (target_emb.data && target_emb.on_device &&
            ti_embed->shape[0] == target_emb.shape[0] &&
            ti_embed->shape[1] == target_emb.shape[1]) {
            head_.embed_tokens = target_emb;
            head_.embed_shared = true;
            head_.embed_qtype = target_model_->tok_emb_qtype_;
            IMP_LOG_INFO("EAGLE: sharing embedding with target (saves %.0f MiB)",
                         ti_embed->bytes / (1024.0 * 1024.0));
        } else {
            head_.embed_tokens = upload_tensor(*ti_embed);
            head_.embed_qtype = GGMLQuantType::F16;  // uploaded as FP16
        }
        head_.vocab_size = static_cast<int>(ti_embed->shape[0]);
        head_.d_model = static_cast<int>(ti_embed->shape[1]);
    } else {
        // No embed_tokens: always share target embedding
        const auto& target_emb = target_model_->token_embedding();
        if (target_emb.data && target_emb.on_device) {
            head_.embed_tokens = target_emb;
            head_.embed_shared = true;
            head_.embed_qtype = target_model_->tok_emb_qtype_;
            head_.vocab_size = target_model_->config().vocab_size;
            head_.d_model = static_cast<int>(target_emb.shape[1]);
            IMP_LOG_INFO("EAGLE: no embed_tokens, sharing target embedding (qtype=%d)",
                         static_cast<int>(head_.embed_qtype));
        }
    }

    // FC weight + bias
    auto* ti_fc_w = find_tensor("fc.weight");
    if (ti_fc_w) {
        head_.fc_weight = upload_tensor(*ti_fc_w);
        // Infer d_model from FC if embed not found
        if (head_.d_model == 0 && ti_fc_w->ndim >= 1)
            head_.d_model = static_cast<int>(ti_fc_w->shape[0]);
    }

    auto* ti_fc_b = find_tensor("fc.bias");
    if (ti_fc_b) head_.fc_bias = upload_tensor(*ti_fc_b);

    // Transformer layer weights: try midlayer.* then eagle_module.layers.0.*
    auto* ti_in = find_tensor("input_layernorm.weight");
    if (!ti_in) ti_in = find_tensor("midlayer.input_layernorm.weight");
    if (ti_in) head_.input_norm = upload_tensor(*ti_in);

    auto* ti_wq = find_tensor("self_attn.q_proj.weight");
    if (!ti_wq) ti_wq = find_tensor("midlayer.self_attn.q_proj.weight");
    if (ti_wq) head_.wq = upload_tensor(*ti_wq);

    auto* ti_wk = find_tensor("self_attn.k_proj.weight");
    if (!ti_wk) ti_wk = find_tensor("midlayer.self_attn.k_proj.weight");
    if (ti_wk) head_.wk = upload_tensor(*ti_wk);

    auto* ti_wv = find_tensor("self_attn.v_proj.weight");
    if (!ti_wv) ti_wv = find_tensor("midlayer.self_attn.v_proj.weight");
    if (ti_wv) head_.wv = upload_tensor(*ti_wv);

    auto* ti_wo = find_tensor("self_attn.o_proj.weight");
    if (!ti_wo) ti_wo = find_tensor("midlayer.self_attn.o_proj.weight");
    if (ti_wo) head_.wo = upload_tensor(*ti_wo);

    // Optional biases (Qwen3)
    auto* ti_qb = find_tensor("self_attn.q_proj.bias");
    if (!ti_qb) ti_qb = find_tensor("midlayer.self_attn.q_proj.bias");
    if (ti_qb) head_.q_bias = upload_tensor(*ti_qb);
    auto* ti_kb = find_tensor("self_attn.k_proj.bias");
    if (!ti_kb) ti_kb = find_tensor("midlayer.self_attn.k_proj.bias");
    if (ti_kb) head_.k_bias = upload_tensor(*ti_kb);
    auto* ti_vb = find_tensor("self_attn.v_proj.bias");
    if (!ti_vb) ti_vb = find_tensor("midlayer.self_attn.v_proj.bias");
    if (ti_vb) head_.v_bias = upload_tensor(*ti_vb);

    auto* ti_pan = find_tensor("post_attention_layernorm.weight");
    if (!ti_pan) ti_pan = find_tensor("midlayer.post_attention_layernorm.weight");
    if (ti_pan) head_.post_attn_norm = upload_tensor(*ti_pan);

    auto* ti_gate = find_tensor("mlp.gate_proj.weight");
    if (!ti_gate) ti_gate = find_tensor("midlayer.mlp.gate_proj.weight");
    if (ti_gate) head_.gate_proj = upload_tensor(*ti_gate);

    auto* ti_up = find_tensor("mlp.up_proj.weight");
    if (!ti_up) ti_up = find_tensor("midlayer.mlp.up_proj.weight");
    if (ti_up) head_.up_proj = upload_tensor(*ti_up);

    auto* ti_down = find_tensor("mlp.down_proj.weight");
    if (!ti_down) ti_down = find_tensor("midlayer.mlp.down_proj.weight");
    if (ti_down) head_.down_proj = upload_tensor(*ti_down);

    // EAGLE-3 v3: hidden_norm, lm_head, norm, vocab mappings
    auto* ti_hn = find_tensor("hidden_norm.weight");
    if (!ti_hn) ti_hn = find_tensor("midlayer.hidden_norm.weight");
    if (ti_hn) head_.hidden_norm = upload_tensor(*ti_hn);

    auto* ti_lm = find_tensor("lm_head.weight");
    if (ti_lm) {
        head_.lm_head = upload_tensor(*ti_lm);
        int lm_vocab = static_cast<int>(ti_lm->shape[0]);
        if (lm_vocab != static_cast<int>(target_model_->config().vocab_size)) {
            head_.has_own_vocab = true;
            head_.target_vocab = target_model_->config().vocab_size;
            // LM head vocab determines EAGLE's output space
            head_.vocab_size = lm_vocab;
            IMP_LOG_INFO("EAGLE: own LM head vocab=%d (target=%d), using d2t/t2d mapping",
                         head_.vocab_size, head_.target_vocab);
        } else if (head_.vocab_size == 0) {
            head_.vocab_size = lm_vocab;
        }
    }

    auto* ti_norm = find_tensor("norm.weight");
    if (ti_norm) head_.lm_norm = upload_tensor(*ti_norm);

    // Vocab mapping tensors — may be INT64, INT32, or FP32 in safetensors.
    // Convert to INT32 on host, upload as INT32 array.
    // Upload d2t mapping: d2t stores OFFSETS, not direct indices.
    // target_id = eagle_id + d2t[eagle_id] (per EAGLE-3 reference implementation).
    // We pre-compute absolute target IDs on host and upload INT32 array.
    auto upload_d2t = [&](const TensorInfo& ti) -> int32_t* {
        size_t elem_size = ti.bytes / ti.shape[0];
        size_t n = static_cast<size_t>(ti.shape[0]);
        std::vector<int32_t> h_map(n);

        // Read raw offsets
        if (elem_size == 8) {
            const int64_t* src = reinterpret_cast<const int64_t*>(ti.host_ptr);
            for (size_t i = 0; i < n; i++)
                h_map[i] = static_cast<int32_t>(static_cast<int64_t>(i) + src[i]);
        } else if (elem_size == 4) {
            const int32_t* src = reinterpret_cast<const int32_t*>(ti.host_ptr);
            for (size_t i = 0; i < n; i++)
                h_map[i] = static_cast<int32_t>(i) + src[i];
        }

        // Keep host copy for fast CPU-side lookup during drafting
        h_d2t_ = h_map;

        int32_t* d_ptr = static_cast<int32_t*>(gpu_alloc(n * sizeof(int32_t)));
        if (d_ptr) {
            cudaMemcpyAsync(d_ptr, h_map.data(), n * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            IMP_LOG_INFO("EAGLE: d2t mapping: %zu entries (offset→absolute), "
                         "first=[%d,%d,%d,%d,%d], last=[%d]",
                         n,
                         n > 0 ? h_map[0] : -1, n > 1 ? h_map[1] : -1,
                         n > 2 ? h_map[2] : -1, n > 3 ? h_map[3] : -1,
                         n > 4 ? h_map[4] : -1,
                         n > 0 ? h_map[n-1] : -1);
        }
        return d_ptr;
    };

    auto* ti_d2t = find_tensor("d2t");
    if (ti_d2t) head_.d_d2t = upload_d2t(*ti_d2t);

    // t2d is a BOOL mask (1 byte/elem), not needed for inference
    auto* ti_t2d = find_tensor("t2d");
    if (ti_t2d) {
        IMP_LOG_INFO("EAGLE: t2d mask: %lld entries (BOOL, not loaded — unused at inference)",
                     (long long)ti_t2d->shape[0]);
    }

    // Unmap the file
    munmap(mmap_base, file_size);

    // 6. Infer dimensions from weight shapes
    if (head_.d_model == 0) {
        IMP_LOG_ERROR("EAGLE: could not determine d_model");
        return false;
    }

    // n_heads, n_kv_heads, and attn_in_dim from Q/K projection shapes
    if (ti_wq && ti_wk) {
        int q_out = static_cast<int>(ti_wq->shape[0]);
        int k_out = static_cast<int>(ti_wk->shape[0]);
        head_.attn_in_dim = static_cast<int>(ti_wq->shape[1]);
        // Try common head dimensions
        for (int hd : {128, 64, 96, 80, 256}) {
            if (q_out % hd == 0) {
                head_.n_heads = q_out / hd;
                head_.head_dim = hd;
                break;
            }
        }
        if (head_.head_dim > 0)
            head_.n_kv_heads = k_out / head_.head_dim;
    } else {
        head_.attn_in_dim = head_.d_model;
    }

    // d_ff from gate_proj shape
    if (ti_gate) {
        head_.d_ff = static_cast<int>(ti_gate->shape[0]);
    }

    // Use target model's rms_norm_eps
    head_.rms_norm_eps = target_model_->config().rms_norm_eps;

    IMP_LOG_INFO("EAGLE head: d_model=%d, attn_in=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, d_ff=%d, vocab=%d%s",
                 head_.d_model, head_.attn_in_dim, head_.n_heads, head_.n_kv_heads,
                 head_.head_dim, head_.d_ff, head_.vocab_size,
                 head_.has_own_vocab ? " (own)" : "");

    // Validate essential weights (fc_bias is optional)
    if (!head_.fc_weight.data ||
        !head_.input_norm.data || !head_.wq.data || !head_.wk.data ||
        !head_.wv.data || !head_.wo.data || !head_.post_attn_norm.data ||
        !head_.gate_proj.data || !head_.up_proj.data || !head_.down_proj.data) {
        IMP_LOG_ERROR("EAGLE: missing essential weights");
        return false;
    }

    // 7. Allocate workspace buffers
    int d = head_.d_model;
    int ain = head_.attn_in_dim;  // may be 2*d_model for EAGLE-3
    // Set target vocab for verify(); EAGLE logit buffers sized for EAGLE's own LM head
    head_.target_vocab = target_model_->config().vocab_size;
    int v = (head_.lm_head.data && head_.has_own_vocab) ? head_.vocab_size : head_.target_vocab;
    int qkv_dim = head_.n_heads * head_.head_dim;
    int kv_dim = head_.n_kv_heads * head_.head_dim;

    d_concat_ = static_cast<half*>(gpu_alloc(3 * d * sizeof(half)));
    d_fused_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_eagle_hidden_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_embed_out_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_norm_out_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_attn_in_ = (ain > d) ? static_cast<half*>(gpu_alloc(ain * sizeof(half))) : nullptr;
    d_hidden_normed_ = head_.hidden_norm.data
        ? static_cast<half*>(gpu_alloc(d * sizeof(half))) : nullptr;
    d_q_ = static_cast<half*>(gpu_alloc(qkv_dim * sizeof(half)));
    d_k_ = static_cast<half*>(gpu_alloc(kv_dim * sizeof(half)));
    d_v_ = static_cast<half*>(gpu_alloc(kv_dim * sizeof(half)));
    d_attn_out_ = static_cast<half*>(gpu_alloc(qkv_dim * sizeof(half)));
    d_proj_out_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_gate_out_ = static_cast<half*>(gpu_alloc(head_.d_ff * sizeof(half)));
    d_up_out_ = static_cast<half*>(gpu_alloc(head_.d_ff * sizeof(half)));
    d_swiglu_out_ = static_cast<half*>(gpu_alloc(head_.d_ff * sizeof(half)));
    d_ffn_out_ = static_cast<half*>(gpu_alloc(d * sizeof(half)));
    d_eagle_logits_fp16_ = static_cast<half*>(gpu_alloc(v * sizeof(half)));
    d_eagle_logits_ = static_cast<float*>(gpu_alloc(v * sizeof(float)));

    // Pre-allocate device buffers for eagle_forward (avoid per-call cudaMalloc)
    d_token_buf_ = static_cast<int32_t*>(gpu_alloc(sizeof(int32_t)));
    d_pos_buf_ = static_cast<int*>(gpu_alloc(sizeof(int)));
    d_cl_buf_ = static_cast<int*>(gpu_alloc(sizeof(int)));
    d_argmax_scratch_ = static_cast<int32_t*>(gpu_alloc(ARGMAX_SCRATCH_BYTES));

    if (!d_concat_ || !d_fused_ || !d_eagle_hidden_ || !d_embed_out_ || !d_norm_out_ ||
        !d_q_ || !d_k_ || !d_v_ || !d_attn_out_ || !d_proj_out_ ||
        !d_gate_out_ || !d_up_out_ || !d_swiglu_out_ || !d_ffn_out_ ||
        !d_eagle_logits_fp16_ || !d_eagle_logits_ ||
        !d_token_buf_ || !d_pos_buf_ || !d_cl_buf_ || !d_argmax_scratch_) {
        IMP_LOG_ERROR("EAGLE: workspace allocation failed");
        return false;
    }

    // 8. Create EAGLE KV cache (1 layer)
    int max_seq = target_kv_cache_->total_blocks();  // reuse target's block count
    int eagle_max_blocks = std::max(max_seq / 4, 64);
    auto eagle_kv = std::make_unique<KVCache>(
        1, head_.n_kv_heads, head_.head_dim,
        DType::FP16, eagle_max_blocks);
    eagle_kv_ = eagle_kv.get();
    eagle_kv_manager_ = std::make_unique<KVCacheManager>(std::move(eagle_kv));

    // Pre-allocate block table buffer for max possible eagle blocks
    max_eagle_blocks_ = (config_.spec_k + eagle_kv_->block_size() - 1) / eagle_kv_->block_size() + 1;
    d_bt_buf_ = static_cast<int*>(gpu_alloc(max_eagle_blocks_ * sizeof(int)));

    cudaStreamSynchronize(stream);

    // Just log the non-shared weight size
    size_t weight_bytes = 0;
    if (!head_.embed_shared && head_.embed_tokens.data)
        weight_bytes += head_.embed_tokens.nbytes();
    if (head_.fc_weight.data) weight_bytes += head_.fc_weight.nbytes();
    if (head_.fc_bias.data) weight_bytes += head_.fc_bias.nbytes();
    if (head_.wq.data) weight_bytes += head_.wq.nbytes();
    if (head_.wk.data) weight_bytes += head_.wk.nbytes();
    if (head_.wv.data) weight_bytes += head_.wv.nbytes();
    if (head_.wo.data) weight_bytes += head_.wo.nbytes();
    if (head_.gate_proj.data) weight_bytes += head_.gate_proj.nbytes();
    if (head_.up_proj.data) weight_bytes += head_.up_proj.nbytes();
    if (head_.down_proj.data) weight_bytes += head_.down_proj.nbytes();

    IMP_LOG_INFO("EAGLE: loaded from %s (weights: %.1f MiB%s)",
                 safetensors_path.c_str(),
                 weight_bytes / (1024.0 * 1024.0),
                 head_.embed_shared ? ", embedding shared" : "");

    initialized_ = true;
    return true;
}

void EagleDecoder::enable_snapshots() {
    if (!target_executor_) return;
    target_executor_->enable_eagle_snapshots(eagle_feat_ptrs_, feat_low_, feat_mid_, feat_high_);
}

void EagleDecoder::disable_snapshots() {
    if (!target_executor_) return;
    target_executor_->disable_eagle_snapshots();
}

// ---------------------------------------------------------------------------
// EAGLE forward: FC fusion + 1 transformer layer
// ---------------------------------------------------------------------------

void EagleDecoder::eagle_forward(int32_t token, int position, cudaStream_t stream) {
    // EAGLE-3 forward pass (per reference implementation):
    //
    // First iteration (first_iter=true):
    //   hidden_states = FC(concat(feat_low, feat_mid, feat_high))  [3*d → d]
    //
    // Subsequent iterations (first_iter=false):
    //   hidden_states = d_eagle_hidden_ (previous EAGLE output, already d_model)
    //   FC is SKIPPED (dimensions already match)
    //
    // Then in both cases (midlayer):
    //   residual = hidden_states
    //   normed_emb = input_layernorm(embed(token))
    //   normed_hidden = hidden_norm(hidden_states)
    //   attn_input = [normed_emb | normed_hidden]  [2*d_model]
    //   attn_out = self_attn(attn_input)  → O proj → [d_model]
    //   hidden = residual + attn_out
    //   normed = post_attn_layernorm(hidden)
    //   mlp_out = MLP(normed)
    //   hidden = hidden + mlp_out  → d_eagle_hidden_

    int d = head_.d_model;

    // d_fused_ holds "hidden_states" — either FC output (first iter) or already set
    // to d_eagle_hidden_ by the caller for subsequent iterations.
    // The residual is taken from d_fused_.

    // 1. Embedding lookup for the token
    {
        cudaMemcpyAsync(d_token_buf_, &token, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        int64_t shape[2] = {1, d};
        Tensor embed_out(d_embed_out_, DType::FP16, 2, shape, true);
        embedding_lookup(head_.embed_tokens, d_token_buf_, 1, embed_out,
                         head_.embed_qtype, stream);
    }

    // 2. Normalize embedding with input_layernorm → d_norm_out_
    {
        int64_t shape[2] = {1, d};
        Tensor emb_t(d_embed_out_, DType::FP16, 2, shape, true);
        Tensor normed_emb_t(d_norm_out_, DType::FP16, 2, shape, true);
        rmsnorm(emb_t, head_.input_norm, normed_emb_t, head_.rms_norm_eps, stream);
    }

    // 3. Normalize hidden_states with hidden_norm → d_hidden_normed_
    //    d_fused_ holds the current hidden_states (FC output or prev EAGLE hidden)
    half* attn_input = d_norm_out_;
    int ain = head_.attn_in_dim;
    if (head_.hidden_norm.data && d_attn_in_ && d_hidden_normed_) {
        int64_t shape[2] = {1, d};
        Tensor hs_t(d_fused_, DType::FP16, 2, shape, true);
        Tensor hn_t(d_hidden_normed_, DType::FP16, 2, shape, true);
        rmsnorm(hs_t, head_.hidden_norm, hn_t, head_.rms_norm_eps, stream);

        // Concat [normed_emb | normed_hidden] → attn_in [2*d_model]
        cudaMemcpyAsync(d_attn_in_, d_norm_out_, d * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_attn_in_ + d, d_hidden_normed_, d * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream);
        attn_input = d_attn_in_;
    } else {
        ain = d;
    }

    // 4. Attention: Q/K/V projections from [2*d_model] input
    {
        int qkv_dim = head_.n_heads * head_.head_dim;
        int kv_dim = head_.n_kv_heads * head_.head_dim;

        int64_t in_shape[1] = {ain};
        int64_t q_shape[1] = {qkv_dim};
        int64_t k_shape_1d[1] = {kv_dim};
        Tensor in_t(attn_input, DType::FP16, 1, in_shape, true);
        Tensor q_t(d_q_, DType::FP16, 1, q_shape, true);
        Tensor k_t(d_k_, DType::FP16, 1, k_shape_1d, true);
        Tensor v_t(d_v_, DType::FP16, 1, k_shape_1d, true);

        gemv(head_.wq, in_t, q_t, stream);
        gemv(head_.wk, in_t, k_t, stream);
        gemv(head_.wv, in_t, v_t, stream);

        // Add biases if present
        if (head_.q_bias.data) {
            int threads = 256;
            int blocks = (qkv_dim + threads - 1) / threads;
            vec_add_kernel<<<blocks, threads, 0, stream>>>(
                d_q_, static_cast<const half*>(head_.q_bias.data), d_q_, qkv_dim);
        }
        if (head_.k_bias.data) {
            int threads = 256;
            int blocks = (kv_dim + threads - 1) / threads;
            vec_add_kernel<<<blocks, threads, 0, stream>>>(
                d_k_, static_cast<const half*>(head_.k_bias.data), d_k_, kv_dim);
        }
        if (head_.v_bias.data) {
            int threads = 256;
            int blocks = (kv_dim + threads - 1) / threads;
            vec_add_kernel<<<blocks, threads, 0, stream>>>(
                d_v_, static_cast<const half*>(head_.v_bias.data), d_v_, kv_dim);
        }

        // RoPE
        {
            float theta = target_model_->config().rope_theta;
            int pos_int = position;
            cudaMemcpyAsync(d_pos_buf_, &pos_int, sizeof(int), cudaMemcpyHostToDevice, stream);

            int64_t q_4d[4] = {1, 1, head_.n_heads, head_.head_dim};
            int64_t k_4d[4] = {1, 1, head_.n_kv_heads, head_.head_dim};
            Tensor q_rope(d_q_, DType::FP16, 4, q_4d, true);
            Tensor k_rope(d_k_, DType::FP16, 4, k_4d, true);
            rope_forward(q_rope, k_rope, d_pos_buf_, head_.head_dim, theta,
                         1.0f, 0, false, 0.0f, 1.0f, nullptr, stream);
        }

        // Write K/V to EAGLE KV cache + paged attention decode
        {
            const auto& bt = eagle_kv_manager_->block_table(0);
            int max_blocks_e = static_cast<int>(bt.size());
            if (max_blocks_e > 0) {
                cudaMemcpyAsync(d_bt_buf_, bt.data(),
                                max_blocks_e * sizeof(int),
                                cudaMemcpyHostToDevice, stream);
                int pos_int = position;
                cudaMemcpyAsync(d_pos_buf_, &pos_int, sizeof(int),
                                cudaMemcpyHostToDevice, stream);

                const int ebs = eagle_kv_->block_size();
                int row_elems = head_.n_kv_heads * head_.head_dim;
                int block_stride = ebs * row_elems;
                int threads = std::min(row_elems, 256);

                write_kv_cache_kernel<<<1, threads, 0, stream>>>(
                    d_k_, d_pos_buf_, d_bt_buf_,
                    static_cast<half*>(eagle_kv_->k_ptr(0, 0)),
                    block_stride, row_elems, ebs, 1, max_blocks_e, 1);

                write_kv_cache_kernel<<<1, threads, 0, stream>>>(
                    d_v_, d_pos_buf_, d_bt_buf_,
                    static_cast<half*>(eagle_kv_->v_ptr(0, 0)),
                    block_stride, row_elems, ebs, 1, max_blocks_e, 1);

                int ctx = position + 1;
                cudaMemcpyAsync(d_cl_buf_, &ctx, sizeof(int),
                                cudaMemcpyHostToDevice, stream);

                float scale = 1.0f / std::sqrt(static_cast<float>(head_.head_dim));

                int64_t q_4d_pa[4] = {1, 1, head_.n_heads, head_.head_dim};
                Tensor q_pa(d_q_, DType::FP16, 4, q_4d_pa, true);

                int total_blocks = eagle_kv_->total_blocks();
                int64_t kv_shape[4] = {total_blocks, ebs,
                                       head_.n_kv_heads, head_.head_dim};
                Tensor k_cache(eagle_kv_->k_ptr(0, 0), DType::FP16, 4, kv_shape, true);
                Tensor v_cache(eagle_kv_->v_ptr(0, 0), DType::FP16, 4, kv_shape, true);

                int64_t o_4d_pa[4] = {1, 1, head_.n_heads, head_.head_dim};
                Tensor o_pa(d_attn_out_, DType::FP16, 4, o_4d_pa, true);

                paged_attention_decode(q_pa, k_cache, v_cache, o_pa,
                                       d_bt_buf_, d_cl_buf_, ebs,
                                       scale, ctx, 0, 0.0f, stream);
            }
        }

        // O projection
        {
            int64_t ao_shape[1] = {qkv_dim};
            int64_t po_shape[1] = {d};
            Tensor ao_t(d_attn_out_, DType::FP16, 1, ao_shape, true);
            Tensor po_t(d_proj_out_, DType::FP16, 1, po_shape, true);
            gemv(head_.wo, ao_t, po_t, stream);
        }
    }

    // 5. Residual: hidden = hidden_states + attn_output
    //    d_fused_ holds hidden_states (FC output or prev EAGLE hidden)
    {
        int threads = 256;
        int blocks = (d + threads - 1) / threads;
        vec_add_kernel<<<blocks, threads, 0, stream>>>(
            d_fused_, d_proj_out_, d_eagle_hidden_, d);
    }

    // 6. Post-attention RMSNorm
    {
        int64_t shape[2] = {1, d};
        Tensor h_t(d_eagle_hidden_, DType::FP16, 2, shape, true);
        Tensor no_t(d_norm_out_, DType::FP16, 2, shape, true);
        rmsnorm(h_t, head_.post_attn_norm, no_t, head_.rms_norm_eps, stream);
    }

    // 7. MLP: gate/up projections
    {
        int64_t in_shape_1d[1] = {d};
        int64_t ff_shape_1d[1] = {head_.d_ff};
        Tensor in_t(d_norm_out_, DType::FP16, 1, in_shape_1d, true);
        Tensor gate_t(d_gate_out_, DType::FP16, 1, ff_shape_1d, true);
        Tensor up_t(d_up_out_, DType::FP16, 1, ff_shape_1d, true);

        gemv(head_.gate_proj, in_t, gate_t, stream);
        gemv(head_.up_proj, in_t, up_t, stream);
    }

    // 8. SwiGLU activation
    {
        int64_t ff_shape[2] = {1, head_.d_ff};
        Tensor gate_t(d_gate_out_, DType::FP16, 2, ff_shape, true);
        Tensor up_t(d_up_out_, DType::FP16, 2, ff_shape, true);
        Tensor swiglu_t(d_swiglu_out_, DType::FP16, 2, ff_shape, true);
        swiglu(gate_t, up_t, swiglu_t, stream);
    }

    // 9. Down projection
    {
        int64_t ff_shape_1d[1] = {head_.d_ff};
        int64_t out_shape_1d[1] = {d};
        Tensor sw_t(d_swiglu_out_, DType::FP16, 1, ff_shape_1d, true);
        Tensor ffn_t(d_ffn_out_, DType::FP16, 1, out_shape_1d, true);
        gemv(head_.down_proj, sw_t, ffn_t, stream);
    }

    // 10. Final residual: eagle_hidden += ffn_out
    {
        int threads = 256;
        int blocks = (d + threads - 1) / threads;
        vec_add_kernel<<<blocks, threads, 0, stream>>>(
            d_eagle_hidden_, d_ffn_out_, d_eagle_hidden_, d);
    }
}

// ---------------------------------------------------------------------------
// Draft token generation using EAGLE head
// ---------------------------------------------------------------------------

std::vector<int32_t> EagleDecoder::draft_tokens(int32_t last_token, int position,
                                                  int seq_id, cudaStream_t stream) {
    std::vector<int32_t> drafts;
    drafts.reserve(config_.spec_k);

    int d = head_.d_model;

    // Ensure EAGLE KV cache has a sequence allocated
    if (eagle_kv_manager_->block_table(0).empty()) {
        (void)eagle_kv_manager_->allocate_blocks(0, 1);
    }

    int32_t cur_token = last_token;
    int cur_pos = 0;  // EAGLE KV cache position (starts fresh each step)

    for (int k = 0; k < config_.spec_k; ++k) {
        // Ensure EAGLE KV cache has enough blocks
        int blocks_needed = (cur_pos + eagle_kv_->block_size()) / eagle_kv_->block_size();
        int blocks_have = static_cast<int>(eagle_kv_manager_->block_table(0).size());
        while (blocks_needed > blocks_have) {
            (void)eagle_kv_manager_->append_block(0);
            blocks_have++;
        }

        // Prepare hidden_states in d_fused_ for eagle_forward:
        // - First iteration: FC(concat(feat_low, feat_mid, feat_high))
        // - Subsequent: d_eagle_hidden_ (previous EAGLE output, no FC)
        if (k == 0) {
            // Concat features → FC → d_fused_
            int total = 3 * d;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            concat_features_kernel<<<blocks, threads, 0, stream>>>(
                d_feat_bufs_[0], d_feat_bufs_[1], d_feat_bufs_[2],
                d_concat_, d);

            // FC: d_fused_ = FC(concat)  (NO embed added — embed handled separately)
            {
                int64_t in_shape[1] = {3 * d};
                int64_t out_shape[1] = {d};
                Tensor in_t(d_concat_, DType::FP16, 1, in_shape, true);
                Tensor out_t(d_fused_, DType::FP16, 1, out_shape, true);
                gemv(head_.fc_weight, in_t, out_t, stream);
            }
            // Add FC bias if present
            if (head_.fc_bias.data) {
                int threads2 = 256;
                int blocks2 = (d + threads2 - 1) / threads2;
                vec_add_kernel<<<blocks2, threads2, 0, stream>>>(
                    d_fused_, static_cast<const half*>(head_.fc_bias.data), d_fused_, d);
            }
        } else {
            // Subsequent iterations: hidden_states = previous EAGLE output
            cudaMemcpyAsync(d_fused_, d_eagle_hidden_, d * sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Run EAGLE head forward (uses d_fused_ as hidden_states)
        eagle_forward(cur_token, cur_pos, stream);

        // LM head: use EAGLE's own norm + lm_head (FP16) → sample from eagle vocab → d2t
        {
            int eagle_vocab = head_.vocab_size;
            int64_t h_shape[2] = {1, d};
            Tensor h_t(d_eagle_hidden_, DType::FP16, 2, h_shape, true);
            Tensor norm_t(d_norm_out_, DType::FP16, 2, h_shape, true);

            // Use EAGLE's own norm (norm.weight) if available, else target's
            if (head_.lm_norm.data) {
                rmsnorm(h_t, head_.lm_norm, norm_t, head_.rms_norm_eps, stream);
            } else {
                rmsnorm(h_t, target_model_->output_norm(), norm_t,
                        target_model_->config().rms_norm_eps, stream,
                        target_model_->config().norm_weight_offset);
            }

            // Use EAGLE's own lm_head (FP16, [eagle_vocab, d_model])
            if (head_.lm_head.data) {
                int64_t norm_1d[1] = {d};
                int64_t logit_1d[1] = {eagle_vocab};
                Tensor norm_1dt(d_norm_out_, DType::FP16, 1, norm_1d, true);
                Tensor logit_t(d_eagle_logits_fp16_, DType::FP16, 1, logit_1d, true);
                gemv(head_.lm_head, norm_1dt, logit_t, stream);
            } else {
                // Fallback: use target's LM head
                eagle_vocab = head_.target_vocab;
                const auto& out_proj = target_model_->output_proj();
                const auto out_qtype = target_model_->out_proj_qtype_;
                if (out_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k(out_proj.data, static_cast<const half*>(norm_t.data),
                             d_eagle_logits_fp16_, eagle_vocab, d, stream);
                } else if (out_qtype == GGMLQuantType::Q8_0) {
                    gemv_q8_0(out_proj.data, static_cast<const half*>(norm_t.data),
                              d_eagle_logits_fp16_, eagle_vocab, d, stream);
                } else {
                    int64_t norm_1d[1] = {d};
                    int64_t logit_1d[1] = {eagle_vocab};
                    Tensor norm_1dt(d_norm_out_, DType::FP16, 1, norm_1d, true);
                    Tensor logit_t(d_eagle_logits_fp16_, DType::FP16, 1, logit_1d, true);
                    gemv(out_proj, norm_1dt, logit_t, stream);
                }
            }

            // Cast FP16 → FP32 for sampling
            {
                int thr = 256;
                int blk = (eagle_vocab + thr - 1) / thr;
                fp16_to_fp32_kernel<<<blk, thr, 0, stream>>>(
                    d_eagle_logits_fp16_, d_eagle_logits_,
                    static_cast<int64_t>(eagle_vocab));
            }
        }

        // Sample from EAGLE vocab (greedy for draft) and map to target via d2t
        {
            int eagle_vocab = head_.lm_head.data ? head_.vocab_size : head_.target_vocab;
            int64_t logit_shape[1] = {eagle_vocab};
            Tensor logit_t(d_eagle_logits_, DType::FP32, 1, logit_shape, true);
            int32_t sampled = sample_greedy(logit_t, d_argmax_scratch_, stream);

            // Map EAGLE token → target token via d2t (host-side lookup, no GPU sync)
            int32_t target_token = sampled;
            if (head_.has_own_vocab && !h_d2t_.empty() &&
                sampled >= 0 && sampled < static_cast<int32_t>(h_d2t_.size())) {
                target_token = h_d2t_[sampled];
            }

            IMP_LOG_DEBUG("EAGLE: k=%d sampled eagle_id=%d → target_id=%d", k, sampled, target_token);
            drafts.push_back(target_token);
            cur_token = target_token;
        }

        // For subsequent iterations, d_eagle_hidden_ is copied to d_fused_ at loop top
        cur_pos++;
    }

    return drafts;
}

// ---------------------------------------------------------------------------
// Verification (reuses target model, similar to SpeculativeDecoder::verify)
// ---------------------------------------------------------------------------

EagleDecoder::VerifyResult
EagleDecoder::verify(const std::vector<int32_t>& draft, int32_t last_token,
                     int position, int seq_id,
                     float temperature, float top_p_val, int top_k_val, int seed,
                     cudaStream_t stream) {
    VerifyResult result;
    const int K = static_cast<int>(draft.size());
    if (K == 0) return result;

    const int n_verify = K + 1;

    // Build token array: [last_token, draft[0], ..., draft[K-1]]
    std::vector<int32_t> h_tokens(n_verify);
    h_tokens[0] = last_token;
    for (int i = 0; i < K; ++i) h_tokens[i + 1] = draft[i];

    // Build position array
    std::vector<int> h_positions(n_verify);
    for (int i = 0; i < n_verify; ++i) h_positions[i] = position + i;

    // Ensure target KV cache has enough blocks
    int final_pos = position + K;
    const int tbs = target_kv_cache_->block_size();
    int needed_blocks = (final_pos + tbs) / tbs;
    int have_blocks = static_cast<int>(target_kv_manager_->block_table(seq_id).size());
    while (needed_blocks > have_blocks) {
        (void)target_kv_manager_->append_block(seq_id);
        have_blocks++;
    }

    const auto& target_blocks = target_kv_manager_->block_table(seq_id);
    int max_blocks = static_cast<int>(target_blocks.size());

    // Verify uses decode mode with K+1 "virtual sequences" sharing the same
    // block table. Each virtual sequence i has context_len = position + i + 1,
    // so its paged attention reads positions 0..position+i from the KV cache.
    // KV write happens before attention (decode path), so each verify token's
    // K/V is in the cache when subsequent tokens attend to it.

    // Build replicated block table: same block table for each virtual sequence
    // Layout: [n_verify, max_blocks] where each row is identical
    std::vector<int> h_block_tables(n_verify * max_blocks);
    for (int s = 0; s < n_verify; ++s) {
        for (int b = 0; b < max_blocks; ++b) {
            h_block_tables[s * max_blocks + b] = target_blocks[b];
        }
    }

    // Build per-sequence context lengths: [position+1, position+2, ..., position+K+1]
    std::vector<int> h_ctx_lens(n_verify);
    for (int i = 0; i < n_verify; ++i) h_ctx_lens[i] = position + i + 1;

    int max_ctx = position + n_verify;

    // Ensure pre-allocated verify buffers are large enough
    int bt_total = n_verify * max_blocks;
    if (n_verify > max_verify_alloc_ || bt_total > max_verify_bt_alloc_) {
        // Grow buffers (tracked by gpu_allocs_ for cleanup)
        if (n_verify > max_verify_alloc_) {
            d_verify_tokens_ = static_cast<int32_t*>(gpu_alloc(n_verify * sizeof(int32_t)));
            d_verify_positions_ = static_cast<int*>(gpu_alloc(n_verify * sizeof(int)));
            d_verify_ctx_lens_ = static_cast<int*>(gpu_alloc(n_verify * sizeof(int)));
            max_verify_alloc_ = n_verify;
        }
        if (bt_total > max_verify_bt_alloc_) {
            d_verify_block_table_ = static_cast<int*>(gpu_alloc(bt_total * sizeof(int)));
            max_verify_bt_alloc_ = bt_total;
        }
    }

    cudaMemcpyAsync(d_verify_tokens_, h_tokens.data(), n_verify * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_positions_, h_positions.data(), n_verify * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_block_table_, h_block_tables.data(),
                    bt_total * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_ctx_lens_, h_ctx_lens.data(), n_verify * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Build InferenceState: decode mode with K+1 virtual sequences.
    // Each virtual sequence has 1 query token, sharing the same physical
    // block table. KV write happens before paged attention (decode path),
    // so each verify token's K/V is visible to subsequent queries.
    InferenceState state;
    state.token_ids = d_verify_tokens_;
    state.positions = d_verify_positions_;
    state.n_tokens = n_verify;
    state.kv_cache = target_kv_cache_;
    state.block_tables = d_verify_block_table_;
    state.context_lens = d_verify_ctx_lens_;
    state.max_context_len = max_ctx;
    state.n_sequences = n_verify;
    state.max_blocks_per_seq = max_blocks;
    state.is_prefill = false;  // decode mode: writes KV then paged attention
    state.skip_nvfp4 = true;  // bypass NVFP4 GEMM (CUTLASS overhead kills M=5), keep FP8
    state.per_row_lm_head = true;  // per-row GEMV LM head for n>1
    state.temperature = temperature;
    state.top_p = top_p_val;
    state.top_k = top_k_val;
    state.seed = seed;

    // Resize workspace for verify pass (K+1 tokens)
    (void)target_executor_->resize_workspace(n_verify, stream);

    // Forward: runs all K+1 tokens through the target model in decode mode.
    // Each layer: GEMM projections → RoPE → KV write → paged attention → MLP.
    // KV entries for positions position..position+K are written to the cache.
    // Feature snapshots capture last row (default behavior).
    Tensor logits;

    target_executor_->forward_logits(state, logits, stream);

    int vocab_size = static_cast<int>(logits.shape[logits.ndim - 1]);

    // Download logits for all K+1 positions.
    // Must sync the non-blocking stream BEFORE cudaMemcpy: the engine stream
    // is created with cudaStreamNonBlocking, so cudaMemcpy (NULL stream)
    // does NOT implicitly synchronize with it.
    size_t logits_bytes = n_verify * vocab_size * sizeof(float);
    std::vector<float> h_logits(n_verify * vocab_size);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_logits.data(), logits.data, logits_bytes, cudaMemcpyDeviceToHost);

    // Greedy verification
    bool greedy = (temperature <= 1e-6f);
    unsigned int rng_state = (seed >= 0) ? static_cast<unsigned int>(seed) : 42u;

    auto softmax_row = [&](int row, std::vector<float>& probs) {
        const float* rl = h_logits.data() + row * vocab_size;
        probs.resize(vocab_size);
        float max_val = rl[0];
        for (int v = 1; v < vocab_size; ++v) max_val = std::max(max_val, rl[v]);
        float inv_temp = greedy ? 1.0f : (1.0f / temperature);
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp((rl[v] - max_val) * inv_temp);
            sum += probs[v];
        }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int v = 0; v < vocab_size; ++v) probs[v] *= inv_sum;
    };

    auto argmax = [&](const std::vector<float>& probs) -> int32_t {
        return static_cast<int32_t>(
            std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    };

    auto sample_from = [&](const std::vector<float>& probs) -> int32_t {
        rng_state = rng_state * 1664525u + 1013904223u;
        float r = static_cast<float>(rng_state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
        float cumsum = 0.0f;
        for (int v = 0; v < static_cast<int>(probs.size()); ++v) {
            cumsum += probs[v];
            if (cumsum >= r) return static_cast<int32_t>(v);
        }
        return static_cast<int32_t>(probs.size() - 1);
    };

    std::vector<float> target_probs;
    int n_accepted = 0;

    for (int i = 0; i < K; ++i) {
        softmax_row(i, target_probs);
        if (greedy) {
            int32_t target_choice = argmax(target_probs);
            if (target_choice == draft[i]) {
                result.accepted.push_back(draft[i]);
                n_accepted++;
            } else {
                result.n_accepted = n_accepted;
                result.next_token = target_choice;
                goto cleanup;
            }
        } else {
            // Simplified stochastic acceptance
            float p_target = target_probs[draft[i]];
            rng_state = rng_state * 1664525u + 1013904223u;
            float r = static_cast<float>(rng_state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
            if (r < p_target) {
                result.accepted.push_back(draft[i]);
                n_accepted++;
            } else {
                result.n_accepted = n_accepted;
                result.next_token = sample_from(target_probs);
                goto cleanup;
            }
        }
    }

    // All accepted — sample token K+1
    {
        softmax_row(K, target_probs);
        result.next_token = greedy ? argmax(target_probs) : sample_from(target_probs);
        result.n_accepted = K;
    }

cleanup:
    return result;
}

// ---------------------------------------------------------------------------
// Feature refresh: run one target decode step to capture correct features.
// After prefill or partial-acceptance verify, the snapshot features may be
// from a stale position.  This single-token decode writes last_token's KV
// at `position` and captures hidden states at the snapshot layers, giving
// the EAGLE head features that encode the prediction AFTER processing
// last_token (i.e., predicting position+1).
// ---------------------------------------------------------------------------

void EagleDecoder::refresh_features(int32_t token, int position, int seq_id,
                                     cudaStream_t stream) {
    // Ensure target KV cache has enough blocks
    int needed_blocks = (position + target_kv_cache_->block_size()) / target_kv_cache_->block_size();
    int have_blocks = static_cast<int>(target_kv_manager_->block_table(seq_id).size());
    while (needed_blocks > have_blocks) {
        (void)target_kv_manager_->append_block(seq_id);
        have_blocks++;
    }

    const auto& blocks = target_kv_manager_->block_table(seq_id);
    int max_blocks = static_cast<int>(blocks.size());
    int ctx_len = position + 1;  // after writing this token

    // Ensure buffers are allocated
    if (!max_verify_alloc_) {
        d_verify_tokens_ = static_cast<int32_t*>(gpu_alloc(sizeof(int32_t)));
        d_verify_positions_ = static_cast<int*>(gpu_alloc(sizeof(int)));
        d_verify_ctx_lens_ = static_cast<int*>(gpu_alloc(sizeof(int)));
        max_verify_alloc_ = 1;
    }
    if (max_blocks > max_verify_bt_alloc_) {
        d_verify_block_table_ = static_cast<int*>(gpu_alloc(max_blocks * sizeof(int)));
        max_verify_bt_alloc_ = max_blocks;
    }

    // Upload single token, position, block table, context length
    cudaMemcpyAsync(d_verify_tokens_, &token, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_positions_, &position, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_block_table_, blocks.data(),
                    max_blocks * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_verify_ctx_lens_, &ctx_len, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    InferenceState state;
    state.token_ids = d_verify_tokens_;
    state.positions = d_verify_positions_;
    state.n_tokens = 1;
    state.kv_cache = target_kv_cache_;
    state.block_tables = d_verify_block_table_;
    state.context_lens = d_verify_ctx_lens_;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = max_blocks;
    state.is_prefill = false;
    state.skip_lm_head = true;  // only need feature snapshots, skip LM head GEMV

    (void)target_executor_->resize_workspace(1, stream);

    Tensor logits;
    target_executor_->forward_logits(state, logits, stream);
    // Logits discarded — we only needed the feature snapshots.
}

// ---------------------------------------------------------------------------
// Full speculative step
// ---------------------------------------------------------------------------

std::vector<int32_t> EagleDecoder::step(int32_t last_token, int position, int seq_id,
                                          float temperature, float top_p, int top_k,
                                          int seed, cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("EAGLE: not initialized");
        return {};
    }

    // Reset EAGLE KV cache for fresh draft
    eagle_kv_manager_->free_sequence(0);

    // Feature refresh: run target model on last_token to capture features.
    // skip_lm_head=true saves the LM head GEMV (~15% of forward cost).
    // Cannot skip: EAGLE needs features from AFTER processing last_token,
    // which verify doesn't provide (it processes the previous step's drafts).
    refresh_features(last_token, position, seq_id, stream);

    // 1. Draft K tokens using EAGLE head
    std::vector<int32_t> draft = draft_tokens(last_token, position, seq_id, stream);
    total_drafted_ += static_cast<int64_t>(draft.size());

    // 2. Verify draft tokens with target model (decode mode, K+1 virtual sequences)
    VerifyResult vr = verify(draft, last_token, position, seq_id,
                             temperature, top_p, top_k, seed, stream);

    // Resize workspace back to 1 token for next decode step
    (void)target_executor_->resize_workspace(1, stream);

    total_accepted_ += vr.n_accepted;

    IMP_LOG_DEBUG("EAGLE: accepted %d/%d (rate=%.1f%%)",
                  vr.n_accepted, static_cast<int>(draft.size()),
                  acceptance_rate() * 100.0f);

    // 3. Combine results: accepted tokens + target-sampled next token
    std::vector<int32_t> output;
    output.reserve(vr.n_accepted + 1);
    for (int i = 0; i < vr.n_accepted; ++i) {
        output.push_back(vr.accepted[i]);
    }
    if (vr.next_token >= 0) {
        output.push_back(vr.next_token);
    }

    return output;
}

} // namespace imp

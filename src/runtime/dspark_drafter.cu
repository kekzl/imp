#include "runtime/dspark_drafter.h"

#include "compute/activation.h"
#include "compute/dspark_attention.h"
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "model/json_util.h"
#include "memory/host_pinned.h"
#include "memory/vram_owned.h"
#include "model/safetensors_raw.h"
#include "quant/fp8_quant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>

namespace imp {

namespace {

constexpr int kHeadDim = 128;  // the norm + RoPE kernel's pair layout (64 threads x 2)

// Per (row, head): RMSNorm over head_dim, then NeoX RoPE at positions[row], in place.
__global__ void dspark_head_norm_rope_kernel(half* __restrict__ x, int row_stride, int heads,
                                             const half* __restrict__ w, const int* __restrict__ pos,
                                             float log2_theta, float eps) {
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    half* v = x + (static_cast<size_t>(row) * row_stride) + (static_cast<size_t>(head) * kHeadDim);
    const int i = threadIdx.x;  // pair (i, i + 64)
    float a = __half2float(v[i]);
    float b = __half2float(v[i + (kHeadDim / 2)]);
    float ss = (a * a) + (b * b);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[2];
    if ((i & 31) == 0)
        part[i >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf(((part[0] + part[1]) / kHeadDim) + eps);
    a *= inv * __half2float(w[i]);
    b *= inv * __half2float(w[i + (kHeadDim / 2)]);
    const float freq = exp2f(-log2_theta * (2.0f * i) / kHeadDim);
    float s = 0.0f;
    float c = 0.0f;
    sincosf(static_cast<float>(pos[row]) * freq, &s, &c);
    v[i] = __float2half((a * c) - (b * s));
    v[i + (kHeadDim / 2)] = __float2half((b * c) + (a * s));
}

__global__ void dspark_iota_kernel(int* __restrict__ out, int base, int n) {
    const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (i < n)
        out[i] = base + i;
}

__global__ void dspark_iota_dev_kernel(int* __restrict__ out, const int* __restrict__ base, int n) {
    const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (i < n)
        out[i] = *base + i;
}

// dst[*row0 * row_elems + i] = src[i]: a block's K / V rows to the cache row the device holds.
__global__ void dspark_store_rows_kernel(half* __restrict__ dst, const half* __restrict__ src,
                                         const int* __restrict__ row0, int row_elems, int n) {
    const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (i < n)
        dst[(static_cast<size_t>(*row0) * row_elems) + i] = src[i];
}

__global__ void dspark_add_kernel(half* __restrict__ h, const half* __restrict__ t, int n) {
    const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (i < n)
        h[i] = __hadd(h[i], t[i]);
}

// One block per row: index of the largest logit (lowest index on ties) to out[row], its softmax probability
// as float bits to out[gridDim.x + row].
__global__ void dspark_argmax_rows_kernel(const half* __restrict__ logits, int vocab,
                                          int32_t* __restrict__ out) {
    const half* r = logits + (static_cast<size_t>(blockIdx.x) * vocab);
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float v = __half2float(r[j]);
        if (v > best) {
            best = v;
            idx = j;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) {
            best = ob;
            idx = oi;
        }
    }
    __shared__ float sb[32];
    __shared__ int si[32];
    if ((threadIdx.x & 31) == 0) {
        sb[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < static_cast<int>(blockDim.x >> 5); ++k)
            if (sb[k] > best || (sb[k] == best && si[k] < idx)) {
                best = sb[k];
                idx = si[k];
            }
        out[blockIdx.x] = idx;
        sb[0] = best;
    }
    __syncthreads();
    const float mx = sb[0];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x)
        sum += __expf(__half2float(r[j]) - mx);
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __syncthreads();  // sb[0] read by every thread before the partial sums overwrite it
    if ((threadIdx.x & 31) == 0)
        sb[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int k = 0; k < static_cast<int>(blockDim.x >> 5); ++k)
            tot += sb[k];
        out[gridDim.x + blockIdx.x] = __float_as_int(1.0f / tot);
    }
}

half bf16_to_half(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return __float2half(f);
}

Tensor t2(void* p, int64_t rows, int64_t cols, QType q = QType::F16) {
    const int64_t s[2] = {rows, cols};
    return Tensor(p, q, 2, s, true);
}

}  // namespace

struct DSparkDrafter::Impl {
    // Per-row E4M3 weight [rows, K] + FP32 row scales; w == nullptr: the FP16 copy serves.
    struct Fp8Mat {
        uint8_t* w = nullptr;
        float* sc = nullptr;
    };
    struct Layer {
        half *in_norm = nullptr, *post_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr,
             *wo = nullptr;
        half *qn = nullptr, *kn = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
        half *k_cache = nullptr, *v_cache = nullptr;  // [cap, n_kv * head_dim]
        Fp8Mat q8, o8, gate8, up8, down8;
    };
    int d = 0, n_layers = 0, nh = 0, nkv = 0, ff = 0, vocab = 0, block = 0, mask_id = 0, cap = 0;
    int row0 = 0;  // first drafting block row: 0 DSpark, 1 DFlash
    float eps = 1e-6f, log2_theta = 0.0f;
    bool fp8 = false;
    bool graph = false;  // capture on the next eager-warmed propose
    bool warm = false;
    cudaGraphExec_t exec = nullptr;
    int replays = 0;
    int* d_params = nullptr;            // [pos, ctx rows] for the propose
    int* h_params = nullptr;            // pin_params
    half *kb = nullptr, *vb = nullptr;  // [block, kv_dim] before the cache row store
    std::vector<int> layer_ids;
    std::vector<Layer> layers;
    half *fc = nullptr, *hidden_norm = nullptr, *norm = nullptr, *lm_head = nullptr, *mask_emb = nullptr;
    Fp8Mat head8;
    // Scratch.
    half *t = nullptr, *tn = nullptr, *h = nullptr, *xn = nullptr, *q = nullptr, *o = nullptr, *res = nullptr;
    half *gate = nullptr, *up = nullptr, *act = nullptr, *logits = nullptr;
    float* s = nullptr;  // dspark_attention split partials
    int* d_pos = nullptr;
    int32_t* d_ids = nullptr;
    PinnedBuffer pin_anchor, pin_ids, pin_params;
    half* h_anchor = nullptr;  // pin_anchor [d]
    int32_t* h_ids = nullptr;  // pin_ids
    RawSafeTensors st;
    std::vector<RawSafeTensors> target_st;  // target shards holding embed_tokens / lm_head (DFlash)
    std::string target_path;
    const uint16_t* embed_bf16 = nullptr;  // host mmap [vocab, d]
    std::vector<int> ctx_pos;              // position of every held context row, ascending
    size_t bytes = 0;
    VRAMAllocator* alloc = nullptr;
    cudaStream_t stream = nullptr;  // load-time uploads
    std::vector<VramOwned<char>> owned;

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() {
        if (exec)
            IMP_CUDA_CHECK_LOG(cudaGraphExecDestroy(exec));
    }

    template <typename T>
    T* dev(size_t n) {
        owned.emplace_back(*alloc, n * sizeof(T), "dspark");
        if (!owned.back())
            return nullptr;
        bytes += n * sizeof(T);
        return reinterpret_cast<T*>(owned.back().get());
    }

    int kv_dim() const { return nkv * kHeadDim; }
    half* upload(const std::map<std::string, const RawTensor*>& by_name, const std::string& name,
                 int64_t numel);
    Fp8Mat upload_fp8(const std::map<std::string, const RawTensor*>& by_name, const std::string& name,
                      int64_t rows, int64_t k, half* stage, int64_t stage_elems);
    // y[b, m] (+)= x[b, k] W^T through the FP8 copy when present, else cuBLAS on the FP16 one.
    void proj(const half* x, int k, const half* w16, const Fp8Mat& w8, half* y, int m, bool accumulate,
              cudaStream_t st_) const;
    std::string borrow_target_tensors(std::map<std::string, const RawTensor*>& by_name);
    std::string upload_weights();
    std::string alloc_scratch(cudaStream_t stream);
    // The whole propose from h_params / h_anchor to h_ids, fixed pointers only: graph-capturable.
    void enqueue_propose(cudaStream_t st_);
    void launch_propose(cudaStream_t st_);

    void head_norm_rope(half* x, int rows, int row_stride, int heads, const half* w, cudaStream_t st_) const {
        if (rows <= 0)
            return;
        dspark_head_norm_rope_kernel<<<rows * heads, kHeadDim / 2, 0, st_>>>(x, row_stride, heads, w, d_pos,
                                                                             log2_theta, eps);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Positions pos0 .. pos0 + n - 1 written on the device: no pinned staging reused across async copies.
    void upload_positions(int pos0, int n, cudaStream_t st_) const {
        dspark_iota_kernel<<<(n + 255) / 256, 256, 0, st_>>>(d_pos, pos0, n);
        IMP_CUDA_CHECK_LAUNCH();
    }
};

DSparkDrafter::DSparkDrafter() = default;
DSparkDrafter::~DSparkDrafter() = default;

int DSparkDrafter::block_size() const { return p_ ? p_->block : 0; }
int DSparkDrafter::draft_len() const { return p_ ? p_->block - p_->row0 : 0; }
const std::vector<int>& DSparkDrafter::target_layer_ids() const { return p_->layer_ids; }
int DSparkDrafter::feature_dim() const { return p_ ? static_cast<int>(p_->layer_ids.size()) * p_->d : 0; }
size_t DSparkDrafter::device_bytes() const { return p_ ? p_->bytes : 0; }
void DSparkDrafter::reset() {
    if (p_)
        p_->ctx_pos.clear();
}
int DSparkDrafter::ctx_rows() const { return p_ ? static_cast<int>(p_->ctx_pos.size()) : 0; }
int DSparkDrafter::next_pos() const { return (p_ && !p_->ctx_pos.empty()) ? p_->ctx_pos.back() + 1 : 0; }

namespace {

bool all_set(std::initializer_list<const void*> ptrs) {
    return std::all_of(ptrs.begin(), ptrs.end(), [](const void* q) { return q != nullptr; });
}

double json_num(const JValue& obj, const char* key, double dflt) {
    const JValue* v = jobj_find(obj, key);
    return (v && v->type == JType::NUMBER) ? v->num_val : dflt;
}

// architectures[0] == "DFlashDraftModel": the z-lab layout (rows 1.. draft, dflash_config sub-object).
bool is_dflash_layout(const JValue& cfg) {
    const JValue* a = jobj_find(cfg, "architectures");
    return a && a->type == JType::ARRAY && !a->arr.empty() && a->arr[0].type == JType::STRING &&
           a->arr[0].str_val == "DFlashDraftModel";
}

// config.json -> dims, block, mask id, theta, target layers. "" or a one-line reason.
std::string parse_dspark_config(const JValue& cfg, int& d, int& n_layers, int& nh, int& nkv, int& ff,
                                int& vocab, int& block, int& mask_id, float& eps, float& log2_theta,
                                std::vector<int>& ids) {
    // DFlash keeps mask_token_id / target_layer_ids in dflash_config; DSpark at the top level.
    const JValue* sub = jobj_find(cfg, "dflash_config");
    const JValue& lay = (sub && sub->type == JType::OBJECT) ? *sub : cfg;
    d = static_cast<int>(json_num(cfg, "hidden_size", 0));
    n_layers = static_cast<int>(json_num(cfg, "num_hidden_layers", 0));
    nh = static_cast<int>(json_num(cfg, "num_attention_heads", 0));
    nkv = static_cast<int>(json_num(cfg, "num_key_value_heads", 0));
    ff = static_cast<int>(json_num(cfg, "intermediate_size", 0));
    vocab = static_cast<int>(json_num(cfg, "vocab_size", 0));
    block = static_cast<int>(json_num(cfg, "block_size", 0));
    mask_id = static_cast<int>(json_num(lay, "mask_token_id", -1));
    eps = static_cast<float>(json_num(cfg, "rms_norm_eps", 1e-6));
    double theta = json_num(cfg, "rope_theta", 1e6);
    if (const JValue* rp = jobj_find(cfg, "rope_parameters"); rp && rp->type == JType::OBJECT)
        theta = json_num(*rp, "rope_theta", theta);
    log2_theta = static_cast<float>(std::log2(theta));
    if (const JValue* v = jobj_find(lay, "target_layer_ids"); v && v->type == JType::ARRAY)
        for (const JValue& e : v->arr)
            ids.push_back(static_cast<int>(e.num_val));
    if (json_num(cfg, "markov_rank", 0) != 0)
        return "markov_rank != 0 (DSpark Markov head) is not supported";
    if (static_cast<int>(json_num(cfg, "head_dim", kHeadDim)) != kHeadDim || nh <= 0 || nkv <= 0 ||
        nh % nkv || nh / nkv > 8)
        return "needs head_dim 128 and n_heads a multiple of n_kv_heads";
    if (block < 2 || block > kDSparkMaxRows || mask_id < 0 || mask_id >= vocab || ids.empty())
        return "block_size / mask_token_id / target_layer_ids missing or out of range";
    return "";
}

}  // namespace

// BF16 tensor `name` with `numel` elements -> FP16 device copy; nullptr when missing, mis-typed or
// mis-shaped.
half* DSparkDrafter::Impl::upload(const std::map<std::string, const RawTensor*>& by_name,
                                  const std::string& name, int64_t numel) {
    auto it = by_name.find(name);
    if (it == by_name.end() || it->second->dtype != "BF16" || it->second->numel() != numel)
        return nullptr;
    std::vector<half> host(static_cast<size_t>(numel));
    const auto* src = static_cast<const uint16_t*>(it->second->data);
    for (int64_t i = 0; i < numel; ++i)
        host[static_cast<size_t>(i)] = bf16_to_half(src[i]);
    half* dst = dev<half>(static_cast<size_t>(numel));
    if (dst && (cudaMemcpyAsync(dst, host.data(), numel * sizeof(half), cudaMemcpyHostToDevice, stream) !=
                    cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess))
        return nullptr;
    return dst;
}

// BF16 [rows, k] -> per-row E4M3 + scales, converted in row chunks through `stage`.
DSparkDrafter::Impl::Fp8Mat DSparkDrafter::Impl::upload_fp8(
    const std::map<std::string, const RawTensor*>& by_name, const std::string& name, int64_t rows, int64_t k,
    half* stage, int64_t stage_elems) {
    auto it = by_name.find(name);
    if (it == by_name.end() || it->second->dtype != "BF16" || it->second->numel() != rows * k)
        return {};
    Fp8Mat m{dev<uint8_t>(static_cast<size_t>(rows * k)), dev<float>(static_cast<size_t>(rows))};
    if (!m.w || !m.sc)
        return {};
    const auto* src = static_cast<const uint16_t*>(it->second->data);
    const int64_t chunk = std::max<int64_t>(1, stage_elems / k);
    std::vector<half> host(static_cast<size_t>(std::min(rows, chunk) * k));
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, rows - r0);
        for (int64_t i = 0; i < nr * k; ++i)
            host[static_cast<size_t>(i)] = bf16_to_half(src[(r0 * k) + i]);
        if (cudaMemcpyAsync(stage, host.data(), nr * k * sizeof(half), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess)
            return {};
        quantize_fp8_rows_async(stage, m.w + (r0 * k), static_cast<int>(nr), static_cast<int>(k), m.sc + r0,
                                stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return {};
    }
    return m;
}

void DSparkDrafter::Impl::proj(const half* x, int k, const half* w16, const Fp8Mat& w8, half* y, int m,
                               bool accumulate, cudaStream_t st_) const {
    if (!w8.w) {
        Tensor yt = t2(y, block, m);
        gemm(t2(const_cast<half*>(x), block, k), t2(const_cast<half*>(w16), m, k), yt, 1.0f,
             accumulate ? 1.0f : 0.0f, st_);
        return;
    }
    half* dst = accumulate ? res : y;
    if (!gemv_fp8_rowscale_f16(w8.w, w8.sc, x, dst, m, k, block, st_)) {
        IMP_LOG_ERROR("dspark: FP8 projection %d x %d refused", m, k);
        return;
    }
    if (accumulate) {
        const int n = block * m;
        dspark_add_kernel<<<(n + 255) / 256, 256, 0, st_>>>(y, res, n);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// embed_tokens / lm_head from the target checkpoint (a .safetensors file or every *.safetensors in a
// directory) into `by_name` under the drafter's names; a tied target serves lm_head from embed_tokens.
std::string DSparkDrafter::Impl::borrow_target_tensors(std::map<std::string, const RawTensor*>& by_name) {
    if (target_path.empty())
        return "embed_tokens / lm_head not in the drafter and no target checkpoint path";
    std::vector<std::string> files;
    std::error_code ec;
    if (std::filesystem::is_directory(target_path, ec)) {
        for (const auto& e : std::filesystem::directory_iterator(target_path, ec))
            if (e.path().extension() == ".safetensors")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(target_path);
    }
    for (const std::string& f : files) {
        RawSafeTensors t;
        if (!t.open(f).empty())
            continue;
        const bool wanted = std::any_of(t.tensors().begin(), t.tensors().end(), [](const RawTensor& rt) {
            return rt.name == "lm_head.weight" || rt.name == "model.embed_tokens.weight";
        });
        if (wanted)
            target_st.push_back(std::move(t));
    }
    for (const RawSafeTensors& t : target_st)
        for (const RawTensor& rt : t.tensors()) {
            if (rt.name == "lm_head.weight")
                by_name.emplace("lm_head.weight", &rt);
            else if (rt.name == "model.embed_tokens.weight")
                by_name.emplace("embed_tokens.weight", &rt);
        }
    if (!by_name.contains("lm_head.weight") && by_name.contains("embed_tokens.weight"))
        by_name.emplace("lm_head.weight", by_name.at("embed_tokens.weight"));
    if (!by_name.contains("lm_head.weight") || !by_name.contains("embed_tokens.weight"))
        return "embed_tokens / lm_head neither in the drafter nor in " + target_path;
    return "";
}

std::string DSparkDrafter::Impl::upload_weights() {
    std::map<std::string, const RawTensor*> by_name;
    for (const RawTensor& rt : st.tensors())
        by_name[rt.name] = &rt;
    if (!by_name.contains("lm_head.weight") || !by_name.contains("embed_tokens.weight"))
        if (std::string err = borrow_target_tensors(by_name); !err.empty())
            return err;
    const int64_t qd = static_cast<int64_t>(nh) * kHeadDim, kvd = kv_dim(), dd = d;
    fc = upload(by_name, "fc.weight", dd * static_cast<int64_t>(layer_ids.size()) * dd);
    hidden_norm = upload(by_name, "hidden_norm.weight", dd);
    norm = upload(by_name, "norm.weight", dd);
    // FP8: propose-only matrices (q, o, gate, up, down, lm_head) keep no FP16 copy; k / v also feed
    // append_ctx's up-to-1024-row GEMMs and stay FP16.
    constexpr int64_t kStageElems = int64_t{32} << 20;  // 64 MiB FP16 staging, freed after load
    VramOwned<char> stage_buf;
    half* stage = nullptr;
    if (fp8) {
        if (dd % 256 || qd % 256 || ff % 256)
            return "FP8 weights need hidden, q and ffn widths divisible by 256";
        stage_buf = VramOwned<char>(*alloc, kStageElems * sizeof(half), "dspark_stage");
        stage = reinterpret_cast<half*>(stage_buf.get());
        if (!stage)
            return "FP8 staging buffer allocation failed";
    }
    auto mat = [&](const std::string& name, int64_t rows, int64_t k, half*& w16, Fp8Mat& w8) {
        if (fp8)
            w8 = upload_fp8(by_name, name, rows, k, stage, kStageElems);
        else
            w16 = upload(by_name, name, rows * k);
        return w16 != nullptr || w8.w != nullptr;
    };
    if (!mat("lm_head.weight", vocab, dd, lm_head, head8) || !all_set({fc, hidden_norm, norm}))
        return "fc / hidden_norm / norm / lm_head missing, not BF16, or mis-shaped";
    layers.resize(static_cast<size_t>(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        auto& L = layers[static_cast<size_t>(l)];
        const std::string pre = "layers." + std::to_string(l) + ".";
        L.in_norm = upload(by_name, pre + "input_layernorm.weight", dd);
        L.post_norm = upload(by_name, pre + "post_attention_layernorm.weight", dd);
        L.wk = upload(by_name, pre + "self_attn.k_proj.weight", kvd * dd);
        L.wv = upload(by_name, pre + "self_attn.v_proj.weight", kvd * dd);
        L.qn = upload(by_name, pre + "self_attn.q_norm.weight", kHeadDim);
        L.kn = upload(by_name, pre + "self_attn.k_norm.weight", kHeadDim);
        const bool mats = mat(pre + "self_attn.q_proj.weight", qd, dd, L.wq, L.q8) &&
                          mat(pre + "self_attn.o_proj.weight", dd, qd, L.wo, L.o8) &&
                          mat(pre + "mlp.gate_proj.weight", ff, dd, L.gate, L.gate8) &&
                          mat(pre + "mlp.up_proj.weight", ff, dd, L.up, L.up8) &&
                          mat(pre + "mlp.down_proj.weight", dd, ff, L.down, L.down8);
        L.k_cache = dev<half>(static_cast<size_t>(cap) * kvd);
        L.v_cache = dev<half>(static_cast<size_t>(cap) * kvd);
        if (!mats || !all_set({L.in_norm, L.post_norm, L.wk, L.wv, L.qn, L.kn, L.k_cache, L.v_cache}))
            return "layer " + std::to_string(l) + ": tensor missing / mis-shaped, or out of VRAM";
    }
    auto eit = by_name.find("embed_tokens.weight");
    if (eit == by_name.end() || eit->second->dtype != "BF16" ||
        eit->second->numel() != static_cast<int64_t>(vocab) * d)
        return "embed_tokens missing or mis-shaped";
    embed_bf16 = static_cast<const uint16_t*>(eit->second->data);
    return "";
}

std::string DSparkDrafter::Impl::alloc_scratch(cudaStream_t stream) {
    const int rows = std::max(kMaxAppendRows, block);
    const size_t qd = static_cast<size_t>(nh) * kHeadDim, b = static_cast<size_t>(block);
    t = dev<half>(static_cast<size_t>(rows) * d);
    tn = dev<half>(static_cast<size_t>(rows) * d);
    h = dev<half>(b * d);
    xn = dev<half>(b * d);
    q = dev<half>(b * qd);
    o = dev<half>(b * qd);
    res = dev<half>(b * d);
    kb = dev<half>(b * kv_dim());
    vb = dev<half>(b * kv_dim());
    d_params = dev<int>(2);
    gate = dev<half>(b * ff);
    up = dev<half>(b * ff);
    act = dev<half>(b * ff);
    logits = dev<half>(b * vocab);
    s = dev<float>(dspark_attention_work_floats(block, nh, cap));
    d_pos = dev<int>(static_cast<size_t>(rows));
    d_ids = dev<int32_t>(2 * b);  // ids, then top-1 probabilities as float bits
    mask_emb = dev<half>(static_cast<size_t>(d));
    if (!all_set(
            {t, tn, h, xn, q, o, res, kb, vb, d_params, gate, up, act, logits, s, d_pos, d_ids, mask_emb}))
        return "scratch allocation failed";
    pin_anchor = PinnedBuffer::acquire(cuda_host_pinned_allocator(), d * sizeof(half));
    pin_ids = PinnedBuffer::acquire(cuda_host_pinned_allocator(), 2 * b * sizeof(int32_t));
    pin_params = PinnedBuffer::acquire(cuda_host_pinned_allocator(), 2 * sizeof(int));
    h_anchor = pin_anchor.as<half>();
    h_ids = pin_ids.as<int32_t>();
    h_params = pin_params.as<int>();
    if (!all_set({h_anchor, h_ids, h_params}))
        return "pinned host scratch allocation failed";
    for (int i = 0; i < d; ++i)
        h_anchor[i] = bf16_to_half(embed_bf16[(static_cast<int64_t>(mask_id) * d) + i]);
    if (cudaMemcpyAsync(mask_emb, h_anchor, d * sizeof(half), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return "mask embedding upload failed";
    return "";
}

std::string DSparkDrafter::load(VRAMAllocator& alloc, const std::string& dir, int target_d_model,
                                int target_layers, int max_ctx, cudaStream_t stream,
                                const DSparkOptions& opt) {
    auto p = std::make_unique<Impl>();
    p->alloc = &alloc;
    p->fp8 = opt.fp8;
    p->graph = opt.graph;
    p->stream = stream;
    p->target_path = opt.target_path;
    std::ifstream cf(dir + "/config.json");
    if (!cf)
        return "no " + dir + "/config.json";
    std::stringstream ss;
    ss << cf.rdbuf();
    const std::string text = ss.str();
    JsonParser parser(text);
    const JValue cfg = parser.parse();
    if (!parser.ok() || cfg.type != JType::OBJECT)
        return "config.json does not parse";
    if (std::string err = parse_dspark_config(cfg, p->d, p->n_layers, p->nh, p->nkv, p->ff, p->vocab,
                                              p->block, p->mask_id, p->eps, p->log2_theta, p->layer_ids);
        !err.empty())
        return err;
    p->row0 = is_dflash_layout(cfg) ? 1 : 0;
    if (p->d != target_d_model)
        return "hidden_size " + std::to_string(p->d) + " != target d_model " + std::to_string(target_d_model);
    for (int id : p->layer_ids)
        if (id < 0 || id >= target_layers)
            return "target_layer_ids entry " + std::to_string(id) + " outside the target's layers";
    if (std::string err = p->st.open(dir + "/model.safetensors"); !err.empty())
        return err;
    p->cap = max_ctx + p->block;
    if (std::string err = p->upload_weights(); !err.empty())
        return err;
    if (std::string err = p->alloc_scratch(stream); !err.empty())
        return err;
    p_ = std::move(p);
    return "";
}

void DSparkDrafter::append_ctx(const half* feat, int ld, int n, int pos0, cudaStream_t stream) {
    Impl& P = *p_;
    // Rollback: rows at positions >= pos0 are superseded.
    P.ctx_pos.erase(std::lower_bound(P.ctx_pos.begin(), P.ctx_pos.end(), pos0), P.ctx_pos.end());
    if (n <= 0)
        return;
    if (n > kMaxAppendRows || ld != feature_dim()) {
        IMP_LOG_ERROR("dspark: append of %d rows (ld %d, want <= %d rows, ld %d) refused", n, ld,
                      kMaxAppendRows, feature_dim());
        return;
    }
    if (static_cast<int>(P.ctx_pos.size()) + n + P.block > P.cap)
        P.ctx_pos.clear();  // past the sized context: restart from these rows
    const int row0 = static_cast<int>(P.ctx_pos.size());
    const int kvd = P.kv_dim();
    P.upload_positions(pos0, n, stream);
    Tensor tt = t2(P.t, n, P.d);
    gemm(t2(const_cast<half*>(feat), n, ld), t2(P.fc, P.d, ld), tt, 1.0f, 0.0f, stream);
    Tensor tn = t2(P.tn, n, P.d);
    rmsnorm(tt, Tensor(P.hidden_norm, QType::F16, 1, std::array<int64_t, 1>{P.d}.data(), true), tn, P.eps,
            stream);
    for (auto& L : P.layers) {
        half* k = L.k_cache + (static_cast<size_t>(row0) * kvd);
        half* v = L.v_cache + (static_cast<size_t>(row0) * kvd);
        Tensor kt = t2(k, n, kvd);
        Tensor vt = t2(v, n, kvd);
        gemm(tn, t2(L.wk, kvd, P.d), kt, 1.0f, 0.0f, stream);
        gemm(tn, t2(L.wv, kvd, P.d), vt, 1.0f, 0.0f, stream);
        P.head_norm_rope(k, n, kvd, P.nkv, L.kn, stream);
    }
    for (int i = 0; i < n; ++i)
        P.ctx_pos.push_back(pos0 + i);
}

void DSparkDrafter::Impl::enqueue_propose(cudaStream_t st_) {
    const int b = block;
    const int qd = nh * kHeadDim;
    const int kvd = kv_dim();
    const int* d_ctx = d_params + 1;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_params, h_params, 2 * sizeof(int), cudaMemcpyHostToDevice, st_));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h, h_anchor, d * sizeof(half), cudaMemcpyHostToDevice, st_));
    for (int r = 1; r < b; ++r)
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h + (static_cast<size_t>(r) * d), mask_emb, d * sizeof(half),
                                           cudaMemcpyDeviceToDevice, st_));
    dspark_iota_dev_kernel<<<1, 32, 0, st_>>>(d_pos, d_params, b);
    IMP_CUDA_CHECK_LAUNCH();
    const int64_t dshape[1] = {d};
    Tensor ht = t2(h, b, d);
    Tensor xt = t2(xn, b, d);
    Tensor gt = t2(gate, b, ff);
    Tensor ut = t2(up, b, ff);
    Tensor at = t2(act, b, ff);
    Tensor kt = t2(kb, b, kvd);
    Tensor vt = t2(vb, b, kvd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const int kv_elems = b * kvd;
    for (auto& L : layers) {
        rmsnorm(ht, Tensor(L.in_norm, QType::F16, 1, dshape, true), xt, eps, st_);
        proj(xn, d, L.wq, L.q8, q, qd, false, st_);
        gemm(xt, t2(L.wk, kvd, d), kt, 1.0f, 0.0f, st_);
        gemm(xt, t2(L.wv, kvd, d), vt, 1.0f, 0.0f, st_);
        head_norm_rope(q, b, qd, nh, L.qn, st_);
        head_norm_rope(kb, b, kvd, nkv, L.kn, st_);
        dspark_store_rows_kernel<<<(kv_elems + 255) / 256, 256, 0, st_>>>(L.k_cache, kb, d_ctx, kvd,
                                                                          kv_elems);
        IMP_CUDA_CHECK_LAUNCH();
        dspark_store_rows_kernel<<<(kv_elems + 255) / 256, 256, 0, st_>>>(L.v_cache, vb, d_ctx, kvd,
                                                                          kv_elems);
        IMP_CUDA_CHECK_LAUNCH();
        IMP_CUDA_CHECK_LOG(
            dspark_attention_dev(q, L.k_cache, L.v_cache, o, s, b, nh, nkv, d_ctx, cap, scale, st_));
        proj(o, qd, L.wo, L.o8, h, d, true, st_);
        rmsnorm(ht, Tensor(L.post_norm, QType::F16, 1, dshape, true), xt, eps, st_);
        proj(xn, d, L.gate, L.gate8, gate, ff, false, st_);
        proj(xn, d, L.up, L.up8, up, ff, false, st_);
        swiglu(gt, ut, at, st_);
        proj(act, ff, L.down, L.down8, h, d, true, st_);
    }
    rmsnorm(ht, Tensor(norm, QType::F16, 1, dshape, true), xt, eps, st_);
    proj(xn, d, lm_head, head8, logits, vocab, false, st_);
    dspark_argmax_rows_kernel<<<b, 1024, 0, st_>>>(logits, vocab, d_ids);
    IMP_CUDA_CHECK_LAUNCH();
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_ids, d_ids, sizeof(int32_t) * 2 * b, cudaMemcpyDeviceToHost, st_));
}

// Eager on the first propose (cuBLAS picks its algorithms outside a capture), captured on the second,
// replayed after; a failed capture stays eager.
void DSparkDrafter::Impl::launch_propose(cudaStream_t st_) {
    if (graph && warm && !exec) {
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            enqueue_propose(st_);
            const cudaError_t end = cudaStreamEndCapture(st_, &g);
            if (end != cudaSuccess || !g || cudaGraphInstantiate(&exec, g, 0) != cudaSuccess)
                exec = nullptr;
            if (g)
                IMP_CUDA_CHECK_LOG(cudaGraphDestroy(g));
        }
        (void)cudaGetLastError();
        if (!exec) {
            IMP_LOG_WARN("dspark: propose graph capture failed, proposing eagerly");
            graph = false;
        }
    }
    warm = true;
    if (exec && cudaGraphLaunch(exec, st_) == cudaSuccess) {
        ++replays;
        return;
    }
    enqueue_propose(st_);
}

std::vector<int32_t> DSparkDrafter::propose(int32_t anchor, int pos, cudaStream_t stream, float min_prob) {
    Impl& P = *p_;
    const int n_ctx = static_cast<int>(P.ctx_pos.size());
    std::vector<int32_t> out;
    if (anchor < 0 || anchor >= P.vocab || n_ctx + P.block > P.cap)
        return out;
    // The previous propose synced: the pinned inputs are free to overwrite.
    for (int i = 0; i < P.d; ++i)
        P.h_anchor[i] = bf16_to_half(P.embed_bf16[(static_cast<int64_t>(anchor) * P.d) + i]);
    P.h_params[0] = pos;
    P.h_params[1] = n_ctx;
    P.launch_propose(stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return out;
    out.assign(P.h_ids + P.row0, P.h_ids + P.block);
    const float* prob = reinterpret_cast<const float*>(P.h_ids + P.block) + P.row0;
    for (size_t i = 1; i < out.size(); ++i)
        if (prob[i] < min_prob) {
            out.resize(i);
            break;
        }
    return out;
}

int DSparkDrafter::graph_replays() const { return p_ ? p_->replays : 0; }

}  // namespace imp

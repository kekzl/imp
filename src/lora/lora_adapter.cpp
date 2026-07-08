#include "lora/lora_adapter.h"
#include "core/logging.h"
#include "model/json_util.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace imp {

namespace {

// Map a PEFT tensor key to (layer, proj, is_A). Returns false for keys that
// are not per-layer LoRA pairs (e.g. embed/lm_head adapters — unsupported v1).
bool parse_key(const std::string& key, int* layer, LoraProj* proj, bool* is_A) {
    size_t lpos = key.find(".layers.");
    if (lpos == std::string::npos)
        return false;
    size_t num_start = lpos + 8;
    size_t num_end = key.find('.', num_start);
    if (num_end == std::string::npos)
        return false;
    *layer = std::atoi(key.substr(num_start, num_end - num_start).c_str());

    struct {
        const char* needle;
        LoraProj p;
    } kMap[] = {
        {".self_attn.q_proj.", LoraProj::Q},   {".self_attn.k_proj.", LoraProj::K},
        {".self_attn.v_proj.", LoraProj::V},   {".self_attn.o_proj.", LoraProj::O},
        {".mlp.gate_proj.", LoraProj::GATE},   {".mlp.up_proj.", LoraProj::UP},
        {".mlp.down_proj.", LoraProj::DOWN},
    };
    bool found = false;
    for (const auto& m : kMap) {
        if (key.find(m.needle) != std::string::npos) {
            *proj = m.p;
            found = true;
            break;
        }
    }
    if (!found)
        return false;

    if (key.find("lora_A") != std::string::npos) {
        *is_A = true;
        return true;
    }
    if (key.find("lora_B") != std::string::npos) {
        *is_A = false;
        return true;
    }
    return false;
}

// Convert one host tensor (F32 / F16 / BF16 wire dtype) to an F16 device
// buffer. Returns nullptr on failure.
void* upload_f16(const uint8_t* src, size_t nbytes, const std::string& dtype, int64_t numel) {
    std::vector<uint16_t> h(numel);
    if (dtype == "F16") {
        if (nbytes != static_cast<size_t>(numel) * 2)
            return nullptr;
        std::memcpy(h.data(), src, nbytes);
    } else if (dtype == "BF16") {
        if (nbytes != static_cast<size_t>(numel) * 2)
            return nullptr;
        const uint16_t* bf = reinterpret_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = static_cast<uint32_t>(bf[i]) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            __half hf = __float2half(f);
            std::memcpy(&h[i], &hf, 2);
        }
    } else if (dtype == "F32") {
        if (nbytes != static_cast<size_t>(numel) * 4)
            return nullptr;
        const float* f = reinterpret_cast<const float*>(src);
        for (int64_t i = 0; i < numel; i++) {
            __half hf = __float2half(f[i]);
            std::memcpy(&h[i], &hf, 2);
        }
    } else {
        return nullptr;
    }
    void* dev = nullptr;
    if (cudaMalloc(&dev, static_cast<size_t>(numel) * 2) != cudaSuccess)
        return nullptr;
    if (cudaMemcpy(dev, h.data(), static_cast<size_t>(numel) * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        return nullptr;
    }
    return dev;
}

}  // namespace

LoraAdapter::~LoraAdapter() {
    for (void* p : device_allocs_)
        if (p)
            cudaFree(p);
}

bool LoraAdapter::load(const std::string& path, int n_layers) {
    namespace fs = std::filesystem;
    std::string st_path = path;
    std::string cfg_path;
    if (fs::is_directory(path)) {
        st_path = path + "/adapter_model.safetensors";
        cfg_path = path + "/adapter_config.json";
    }
    if (!fs::exists(st_path)) {
        IMP_LOG_ERROR("LoRA: %s not found", st_path.c_str());
        return false;
    }

    // ---- adapter_config.json: r, lora_alpha, use_rslora ----
    int cfg_r = 0;
    float alpha = 0.0f;
    bool rslora = false;
    if (!cfg_path.empty() && fs::exists(cfg_path)) {
        std::ifstream f(cfg_path);
        std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        JsonParser p(js.data(), js.size());
        JValue root = p.parse();
        if (p.ok() && root.type == JType::OBJECT) {
            for (auto& [k, v] : root.obj) {
                if (k == "r" && v.type == JType::NUMBER)
                    cfg_r = static_cast<int>(v.num_val);
                else if (k == "lora_alpha" && v.type == JType::NUMBER)
                    alpha = static_cast<float>(v.num_val);
                else if (k == "use_rslora" && v.type == JType::NUMBER)
                    rslora = v.num_val != 0.0;
            }
        }
    }

    // ---- adapter_model.safetensors ----
    std::ifstream f(st_path, std::ios::binary);
    f.seekg(0, std::ios::end);
    uint64_t fsize = static_cast<uint64_t>(f.tellg());
    f.seekg(0);
    if (fsize < 8) {
        IMP_LOG_ERROR("LoRA: %s too small", st_path.c_str());
        return false;
    }
    uint64_t hdr_len = 0;
    f.read(reinterpret_cast<char*>(&hdr_len), 8);
    if (hdr_len == 0 || hdr_len > 64ull * 1024 * 1024 || 8 + hdr_len > fsize) {
        IMP_LOG_ERROR("LoRA: bad safetensors header size %llu", (unsigned long long)hdr_len);
        return false;
    }
    std::string hdr(hdr_len, '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hdr_len));
    std::vector<uint8_t> data(fsize - 8 - hdr_len);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));

    JsonParser p(hdr.data(), hdr.size());
    JValue root = p.parse();
    if (!p.ok() || root.type != JType::OBJECT) {
        IMP_LOG_ERROR("LoRA: safetensors header JSON parse failed");
        return false;
    }

    layers_.assign(static_cast<size_t>(n_layers), {});

    for (auto& [key, tv] : root.obj) {
        if (key == "__metadata__" || tv.type != JType::OBJECT)
            continue;
        int layer = -1;
        LoraProj proj{};
        bool is_A = false;
        if (!parse_key(key, &layer, &proj, &is_A)) {
            if (key.find("lora") != std::string::npos)
                IMP_LOG_WARN("LoRA: unsupported tensor '%s' skipped (v1: per-layer q/k/v/o/gate/up/down)",
                             key.c_str());
            continue;
        }
        if (layer < 0 || layer >= n_layers) {
            IMP_LOG_ERROR("LoRA: tensor '%s' layer out of range (n_layers=%d)", key.c_str(), n_layers);
            return false;
        }

        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t off0 = 0, off1 = 0;
        for (auto& [tk, tvv] : tv.obj) {
            if (tk == "dtype")
                dtype = tvv.str_val;
            else if (tk == "shape" && tvv.type == JType::ARRAY)
                for (auto& d : tvv.arr)
                    shape.push_back(d.as_int());
            else if (tk == "data_offsets" && tvv.type == JType::ARRAY && tvv.arr.size() == 2) {
                off0 = static_cast<uint64_t>(tvv.arr[0].num_val);
                off1 = static_cast<uint64_t>(tvv.arr[1].num_val);
            }
        }
        if (shape.size() != 2 || off1 <= off0 || off1 > data.size()) {
            IMP_LOG_ERROR("LoRA: tensor '%s' bad shape/offsets", key.c_str());
            return false;
        }
        int64_t numel = shape[0] * shape[1];
        void* dev = upload_f16(data.data() + off0, off1 - off0, dtype, numel);
        if (!dev) {
            IMP_LOG_ERROR("LoRA: tensor '%s' upload failed (dtype=%s)", key.c_str(), dtype.c_str());
            return false;
        }
        device_allocs_.push_back(dev);

        LoraWeights& w = layers_[layer].proj[std::to_underlying(proj)];
        if (is_A) {
            w.A = dev;
            w.r = static_cast<int>(shape[0]);
            w.K = static_cast<int>(shape[1]);
        } else {
            w.B = dev;
            w.N = static_cast<int>(shape[0]);
            if (w.r == 0)
                w.r = static_cast<int>(shape[1]);
        }
        n_tensors_++;
    }

    // Validate pairs + collect max rank.
    for (size_t li = 0; li < layers_.size(); li++) {
        for (int pi = 0; pi < std::to_underlying(LoraProj::COUNT); pi++) {
            LoraWeights& w = layers_[li].proj[pi];
            if ((w.A == nullptr) != (w.B == nullptr)) {
                IMP_LOG_ERROR("LoRA: layer %zu proj %d has unpaired A/B", li, pi);
                return false;
            }
            if (w.A)
                max_rank_ = std::max(max_rank_, w.r);
        }
    }
    if (n_tensors_ == 0) {
        IMP_LOG_ERROR("LoRA: no usable lora_A/lora_B pairs in %s", st_path.c_str());
        return false;
    }

    int r_eff = (cfg_r > 0) ? cfg_r : max_rank_;
    if (alpha > 0.0f && r_eff > 0)
        scale_ = rslora ? (alpha / std::sqrt(static_cast<float>(r_eff)))
                        : (alpha / static_cast<float>(r_eff));
    else
        scale_ = 1.0f;

    IMP_LOG_INFO("LoRA: loaded %d tensors from %s (r=%d, alpha=%.1f%s, scale=%.4f, max_rank=%d)",
                 n_tensors_, st_path.c_str(), r_eff, alpha, rslora ? ", rslora" : "", scale_, max_rank_);
    return true;
}

}  // namespace imp

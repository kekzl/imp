// find_sibling_mtp_head: a checkpoint next door completes an incomplete Qwen MTP head.
// nvidia/Qwen3.8-27B-NVFP4-QAD ships 11 of 15 mtp.* tensors; all 11 equal the base export's bytes.
#include "core/logging.h"
#include "model/json_util.h"
#include "model/mtp_head.h"
#include "model/safetensors_loader.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

namespace fs = std::filesystem;

constexpr size_t kMaxSiblingDirs = 256;       // directories scanned next to the model
constexpr size_t kMaxIndexBytes = 64u << 20;  // model.safetensors.index.json read cap

// Every tensor field of the Qwen layout, dense and MoE MLP alike.
constexpr Tensor MtpHead::* kQwenFields[] = {
    &MtpHead::pre_fc_norm_embedding,
    &MtpHead::pre_fc_norm_hidden,
    &MtpHead::fc,
    &MtpHead::input_layernorm,
    &MtpHead::post_attention_layernorm,
    &MtpHead::q_proj,
    &MtpHead::k_proj,
    &MtpHead::v_proj,
    &MtpHead::o_proj,
    &MtpHead::q_norm,
    &MtpHead::k_norm,
    &MtpHead::final_norm,
    &MtpHead::router,
    &MtpHead::experts_gate_up_packed,
    &MtpHead::experts_down_packed,
    &MtpHead::shared_expert_gate_proj,
    &MtpHead::shared_expert_up_proj,
    &MtpHead::shared_expert_down_proj,
    &MtpHead::shared_expert_gate,
};

bool same_tensor(const Tensor& a, const Tensor& b) {
    if (!b.data || a.qtype != b.qtype || a.ndim != b.ndim || a.nbytes() != b.nbytes())
        return false;
    for (int i = 0; i < a.ndim; i++)
        if (a.shape[i] != b.shape[i])
            return false;
    return std::memcmp(a.data, b.data, a.nbytes()) == 0;
}

// Shard of `dir` that holds the fusion projection: the head files by name, else the index entry.
std::string head_file_of(const fs::path& dir) {
    std::error_code ec;
    for (const char* name : {"model_mtp.safetensors", "mtp.safetensors"})
        if (fs::is_regular_file(dir / name, ec))
            return (dir / name).string();
    const fs::path index = dir / "model.safetensors.index.json";
    const auto bytes = fs::file_size(index, ec);
    if (ec || bytes == 0 || bytes > kMaxIndexBytes)
        return {};
    std::ifstream in(index, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    JsonParser parser{std::string_view(text)};
    const JValue root = parser.parse();
    if (!parser.ok() || root.type != JType::OBJECT)
        return {};
    const JValue* map = jobj_find(root, "weight_map");
    if (!map || map->type != JType::OBJECT)
        return {};
    const JValue* shard = jobj_find(*map, kMtpHeadKeyFc);
    if (!shard || shard->type != JType::STRING || !safetensors_shard_name_is_safe(shard->str_val))
        return {};
    return (dir / shard->str_val).string();
}

// True when `file` alone holds a complete Qwen head equal to `partial` on every tensor partial has.
bool completes(const MtpHead& partial, const std::string& file) {
    std::unordered_map<std::string, Tensor> tensors;
    void* base = nullptr;
    size_t size = 0;
    if (!load_raw_safetensors_file(file, tensors, base, size))
        return false;
    bool ok = tensors.contains(kMtpHeadKeyFc);
    if (ok) {
        const MtpHead head = dispatch_mtp_head(tensors, file, size);
        ok = head.loaded && head.layout == MtpLayout::Qwen;
        for (const auto field : kQwenFields)
            if (ok && (partial.*field).data)
                ok = same_tensor(partial.*field, head.*field);
    }
    munmap(base, size);
    return ok;
}

}  // namespace

std::string find_sibling_mtp_head(const MtpHead& partial, const std::string& model_path) {
    if (partial.loaded || partial.layout != MtpLayout::Qwen || model_path.empty())
        return {};
    // No embedded tensor, no evidence that a neighbour's head is this model's.
    const bool has_any = std::ranges::any_of(kQwenFields,
                                             [&](auto f) { return (partial.*f).data != nullptr; });
    if (!has_any)
        return {};
    std::error_code ec;
    fs::path self = fs::weakly_canonical(model_path, ec);
    if (ec)
        return {};
    if (!fs::is_directory(self, ec))
        self = self.parent_path();
    std::vector<fs::path> dirs;
    for (fs::directory_iterator it(self.parent_path(), ec), end; !ec && it != end; it.increment(ec)) {
        if (dirs.size() >= kMaxSiblingDirs)
            break;
        std::error_code e2;
        if (it->is_directory(e2) && fs::weakly_canonical(it->path(), e2) != self)
            dirs.push_back(it->path());
    }
    std::ranges::sort(dirs);
    for (const fs::path& dir : dirs) {
        const std::string file = head_file_of(dir);
        if (!file.empty() && completes(partial, file))
            return file;
    }
    return {};
}

}  // namespace imp

#include "vision/qwen3vl_vision_map.h"

#include <cstdlib>

namespace imp {

namespace {

using Slot = Qwen3VLVisionSlot;

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// Parse "<digits>." at `pos`, advancing past the dot. Returns -1 if there is no
// digit run followed by a dot — a name like `blocks.x.norm1.weight` must fall
// through to Unknown rather than be silently read as block 0.
int take_index(const std::string& s, size_t& pos) {
    size_t start = pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
        ++pos;
    if (pos == start || pos >= s.size() || s[pos] != '.')
        return -1;
    const int v = std::atoi(s.substr(start, pos - start).c_str());
    ++pos;  // skip '.'
    return v;
}

// The block/merger sub-names are identical between the two, so share the tail.
Slot merger_tail(const std::string& tail) {
    if (tail == "norm.weight")
        return Slot::MergerNormWeight;
    if (tail == "norm.bias")
        return Slot::MergerNormBias;
    if (tail == "linear_fc1.weight")
        return Slot::MergerFc1Weight;
    if (tail == "linear_fc1.bias")
        return Slot::MergerFc1Bias;
    if (tail == "linear_fc2.weight")
        return Slot::MergerFc2Weight;
    if (tail == "linear_fc2.bias")
        return Slot::MergerFc2Bias;
    return Slot::Unknown;
}

}  // namespace

Qwen3VLVisionRef qwen3vl_map_vision_tensor(const std::string& name) {
    Qwen3VLVisionRef ref;

    if (name == "patch_embed.proj.weight") {
        ref.slot = Slot::PatchEmbedWeight;
        return ref;
    }
    if (name == "patch_embed.proj.bias") {
        ref.slot = Slot::PatchEmbedBias;
        return ref;
    }
    if (name == "pos_embed.weight") {
        ref.slot = Slot::PosEmbed;
        return ref;
    }

    if (starts_with(name, "blocks.")) {
        size_t pos = 7;  // strlen("blocks.")
        const int idx = take_index(name, pos);
        if (idx < 0)
            return ref;
        const std::string tail = name.substr(pos);
        ref.index = idx;
        if (tail == "norm1.weight")
            ref.slot = Slot::Norm1Weight;
        else if (tail == "norm1.bias")
            ref.slot = Slot::Norm1Bias;
        else if (tail == "attn.qkv.weight")
            ref.slot = Slot::QkvWeight;
        else if (tail == "attn.qkv.bias")
            ref.slot = Slot::QkvBias;
        else if (tail == "attn.proj.weight")
            ref.slot = Slot::ProjWeight;
        else if (tail == "attn.proj.bias")
            ref.slot = Slot::ProjBias;
        else if (tail == "norm2.weight")
            ref.slot = Slot::Norm2Weight;
        else if (tail == "norm2.bias")
            ref.slot = Slot::Norm2Bias;
        else if (tail == "mlp.linear_fc1.weight")
            ref.slot = Slot::Fc1Weight;
        else if (tail == "mlp.linear_fc1.bias")
            ref.slot = Slot::Fc1Bias;
        else if (tail == "mlp.linear_fc2.weight")
            ref.slot = Slot::Fc2Weight;
        else if (tail == "mlp.linear_fc2.bias")
            ref.slot = Slot::Fc2Bias;
        if (ref.slot == Slot::Unknown)
            ref.index = -1;
        return ref;
    }

    // DeepStack mergers must be tested BEFORE the main merger: the main one is
    // `merger.*` and these are `deepstack_merger_list.<i>.*`, so there is no
    // prefix overlap — but keeping the order explicit documents that they are
    // distinct, not variants of one another.
    if (starts_with(name, "deepstack_merger_list.")) {
        size_t pos = 22;  // strlen("deepstack_merger_list.")
        const int idx = take_index(name, pos);
        if (idx < 0)
            return ref;
        const Slot s = merger_tail(name.substr(pos));
        if (s != Slot::Unknown) {
            ref.slot = s;
            ref.index = idx;
        }
        return ref;
    }

    if (starts_with(name, "merger.")) {
        const Slot s = merger_tail(name.substr(7));
        if (s != Slot::Unknown) {
            ref.slot = s;
            ref.index = -1;  // the main merger
        }
        return ref;
    }

    return ref;
}

}  // namespace imp

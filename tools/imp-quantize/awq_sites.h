#pragma once

// Where an AWQ divisor may fold, per architecture/layer, decided from tensor NAMES and the
// architecture table alone (checkable without a GPU or checkpoint): which consumers share an
// activation, which producer absorbs 1/s, whether the norm carries a unit offset. Search
// (awq.cu) and the unlisted-consumer safety scan stay in awq_plan.cpp.
// Architecture question is which convention the LOADER applies to that model_type
// (src/model/weight_upload.cu arch_norm_offset, hf_config_loader.cpp for names); an unknown
// convention is refused.
// qwen3_5 family fold sites (verified against Qwen3.8-27B's tensor index): A
// self_attn.{q,k,v}_proj<-input_layernorm (offset); G linear_attn.in_proj_*<-input_layernorm
// (offset); B mlp.{gate,up}_proj<-post_attention_layernorm (offset); C self_attn.o_proj<-
// self_attn.v_proj rows (no norm); D mlp.down_proj<-mlp.up_proj rows (no norm); E
// linear_attn.out_proj<-linear_attn.norm (PLAIN, no offset, tied across value heads).
// conv1d, mtp.* and visual.* have no absorbing producer and are not sites.

#include "quant/awq_norm_fold.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace imp::awq {

// Which convention the block norms of an architecture follow.
enum class NormConvention { Plain, UnitOffset };

// std::nullopt = an architecture whose norm convention this tool does not know,
// which is refused rather than folded on a guess.
std::optional<NormConvention> arch_norm_convention(const std::string& model_type);

// The per-layer tensor-name prefix a checkpoint actually uses, "" when neither
// form is present. Qwen3.8 nests the text model under model.language_model.
std::string resolve_layer_prefix(const std::set<std::string>& names);

// How a consumer's input channels map onto the producer's channels.
enum class TieMode {
    Identity,    // one producer channel per consumer channel
    GqaValue,    // several query heads read one KV head (o_proj <- v_proj)
    PerHeadDim,  // one channel per head_dim slot, shared by every head
};

// What the producer is.
enum class FoldKind {
    NormVector,  // a 1-D norm weight: divide it (offset-aware)
    MatrixRows,  // the output rows of a 2-D producer: divide them
};

struct Geometry {
    int64_t head_dim = 0;
    int64_t n_rep = 1;  // query heads per KV head
};

struct FoldSite {
    char group = 0;                    // A B C D E G
    std::vector<std::string> members;  // consumers, get the column scale
    std::string producer;              // absorbs 1/s
    std::string producer_bias;         // folded with the producer when present
    FoldKind kind = FoldKind::NormVector;
    NormOffset offset = NormOffset::Plain;  // NormVector only
    TieMode tie = TieMode::Identity;
    std::vector<std::string> calib_keys;   // statistic sources, merged by max
    std::string scan_prefix;               // "" = no unlisted-consumer scan
    std::vector<std::string> scan_exempt;  // suffixes that are not consumers
};

// producer channel for each of the K consumer input channels. Empty when the
// geometry cannot express the tie (head_dim <= 0, K not a multiple of it).
std::vector<int64_t> tie_map(TieMode tie, int64_t K, const Geometry& geo);

// Producer length the tie implies for K consumer channels, 0 when impossible.
int64_t tie_producer_len(TieMode tie, int64_t K, const Geometry& geo);

// Sites of one layer, in search order: row folds first (their producers, v_proj/up_proj, are
// norm-group members and must be searched against the weights they'll quantize). `groups`
// selects by letter; a site whose members aren't all present in `names` is not returned.
std::vector<FoldSite> layer_fold_sites(const std::set<std::string>& names, const std::string& base,
                                       NormConvention conv, const std::string& groups);

}  // namespace imp::awq

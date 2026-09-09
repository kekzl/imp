// The checkpoint's quantization contract, on the CPU lane (#1960).
//
// `config_groups[*].targets = ["Linear"]` plus `quantization_config.ignore` is a
// COMPLETE partition of a compressed-tensors checkpoint's Linears. Measured on
// Qwen3.8-27B-NVFP4-vllm: 496 packed modules, 170 ignore entries, 121 plain 2-D
// weights all covered, 0 Linears left over. imp parsed the list and read it
// nowhere, so a Linear whose weight_scale went missing on the way in was
// indistinguishable from one the author left in BF16 on purpose.
//
// The second half covers the fused-projection scale split: which facts the
// loader may act on, and that the only producer of the provenance flag still
// produces it.

#include "exec/nvfp4_merged_scale_guard.h"
#include "model/model.h"
#include "model/nvfp4_module_policy.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace pol = imp::nvfp4_policy;
// ---- the checkpoint's quantization contract (#1960) --------------------------
//
// `config_groups[*].targets = ["Linear"]` plus `quantization_config.ignore` is a
// COMPLETE partition of a compressed-tensors checkpoint's Linears. Measured on
// Qwen3.8-27B-NVFP4-vllm: 496 packed modules, 170 ignore entries, 121 plain 2-D
// weights of which all 121 are listed, 0 left over. imp parsed the list and read
// it nowhere, so a Linear whose weight_scale went missing on the way in was
// indistinguishable from one the author left in BF16 on purpose.

namespace {

pol::SlotObservation packed(const std::string& module, int64_t k_packed, bool global_scale = true) {
    pol::SlotObservation s;
    s.name = module + ".weight";
    s.ndim = 2;
    s.K = 2 * k_packed;
    s.has_micro_scale = true;
    s.has_global_scale = global_scale;
    return s;
}

pol::SlotObservation plain(const std::string& module, int64_t K, int ndim = 2) {
    pol::SlotObservation s;
    s.name = module + ".weight";
    s.ndim = ndim;
    s.K = K;
    return s;
}

}  // namespace

TEST(NvFP4IgnoreList, IgnoreListIsEnforced) {
    // The three shapes the target checkpoint actually contains.
    const std::vector<pol::SlotObservation> slots = {
        packed("model.layers.0.self_attn.q_proj", 2560), plain("model.lm_head", 5120),
        plain("model.layers.0.mlp.gate_proj", 5120),  // a Linear that lost its scale
    };
    const std::vector<std::string> ignore = {"model.language_model.lm_head"};

    const pol::Inventory inv = pol::classify(slots, ignore);
    EXPECT_EQ(inv.quantized, 1);
    EXPECT_EQ(inv.ignored, 1) << "lm_head is listed, under the on-disk wrapper spelling";
    EXPECT_EQ(inv.unclassified, 1);
    EXPECT_EQ(inv.first_unclassified, "model.layers.0.mlp.gate_proj");

    std::string why;
    EXPECT_TRUE(pol::refuses(inv, /*is_compressed_tensors=*/true, &why));
    EXPECT_NE(why.find("gate_proj"), std::string::npos) << "the refusal names the module";
    // Modelopt's exclude_modules is a hint, not a partition: same inventory, no
    // refusal, or every Modelopt checkpoint would stop loading.
    EXPECT_FALSE(pol::refuses(inv, /*is_compressed_tensors=*/false, nullptr));

    // Listing the module is what makes it servable at source precision.
    const pol::Inventory ok = pol::classify(slots,
                                            {"model.language_model.lm_head", "model.layers.0.mlp.gate_proj"});
    EXPECT_EQ(ok.unclassified, 0);
    EXPECT_FALSE(pol::refuses(ok, /*is_compressed_tensors=*/true, nullptr));
}

TEST(NvFP4IgnoreList, MissingGlobalScaleRefused) {
    // compressed-tensors divides by weight_global_scale. Absent, imp defaulted
    // the tensor scale to 1.0 while the real ones are ~2688/amax (1e3..1e4).
    const std::vector<pol::SlotObservation> slots = {
        packed("model.layers.0.self_attn.k_proj", 2560, /*global_scale=*/true),
        packed("model.layers.0.self_attn.v_proj", 2560, /*global_scale=*/false),
    };
    const pol::Inventory inv = pol::classify(slots, {});
    EXPECT_EQ(inv.quantized, 2);
    EXPECT_EQ(inv.missing_global_scale, 1);
    EXPECT_EQ(inv.first_missing_global_scale, "model.layers.0.self_attn.v_proj");

    std::string why;
    EXPECT_TRUE(pol::refuses(inv, /*is_compressed_tensors=*/true, &why));
    EXPECT_NE(why.find("v_proj"), std::string::npos);
    // Modelopt permits an absent weight_scale_2 (NvFP4PreQuantWeight::valid()
    // only requires weight_scale), so the rule must not reach it.
    EXPECT_FALSE(pol::refuses(inv, /*is_compressed_tensors=*/false, nullptr));
}

TEST(NvFP4IgnoreList, InventoryCountsMatch) {
    // A Qwen3.8-27B-shaped slice: quantized Linears, ignore-listed Linears, and
    // the roles that are not Linear at all and must not enter the partition.
    std::vector<pol::SlotObservation> slots;
    std::vector<std::string> ignore;
    for (int i = 0; i < 4; i++) {
        const std::string L = "model.language_model.layers." + std::to_string(i) + ".";
        slots.push_back(packed("model.layers." + std::to_string(i) + ".self_attn.q_proj", 2560));
        slots.push_back(packed("model.layers." + std::to_string(i) + ".mlp.down_proj", 8704));
        // conv1d is 3-D and listed; it is not a Linear, so it is neither counted
        // as ignored nor as unclassified.
        slots.push_back(plain("model.layers." + std::to_string(i) + ".linear_attn.conv1d", 4, /*ndim=*/3));
        ignore.push_back(L + "linear_attn.conv1d");
        // Norms are 1-D in the real file; assert the role rule too, not just the rank.
        slots.push_back(plain("model.layers." + std::to_string(i) + ".input_layernorm", 5120));
    }
    slots.push_back(plain("model.embed_tokens", 5120));
    slots.push_back(plain("lm_head", 5120));
    ignore.push_back("lm_head");

    const pol::Inventory inv = pol::classify(slots, ignore);
    EXPECT_EQ(inv.quantized, 8);
    EXPECT_EQ(inv.ignored, 1) << "lm_head only: conv1d is 3-D, norms and embeddings are not Linears";
    EXPECT_EQ(inv.unclassified, 0);
    EXPECT_EQ(inv.missing_global_scale, 0);
    EXPECT_FALSE(pol::refuses(inv, /*is_compressed_tensors=*/true, nullptr));
}

TEST(NvFP4IgnoreList, MatcherFormsThatTheTwoProducersWrite) {
    const std::string q = "model.layers.3.self_attn.q_proj";
    // Fully qualified, the form imp-quantize writes (tensor name minus .weight).
    EXPECT_TRUE(pol::module_is_ignored(q, {q}));
    // The on-disk wrapper prefix is normalized on both sides.
    EXPECT_TRUE(pol::module_is_ignored(q, {"model.language_model.layers.3.self_attn.q_proj"}));
    EXPECT_TRUE(pol::module_is_ignored("lm_head", {"language_model.lm_head"}));
    // vLLM's re: form.
    EXPECT_TRUE(pol::module_is_ignored(q, {"re:model\\.layers\\.\\d+\\.self_attn\\.q_proj"}));
    EXPECT_FALSE(pol::module_is_ignored(q, {"re:model\\.layers\\.\\d+\\.mlp\\..*"}));
    // An unparseable pattern matches nothing, never everything.
    EXPECT_FALSE(pol::module_is_ignored(q, {"re:[unterminated"}));
    // Modelopt glob.
    EXPECT_TRUE(pol::module_is_ignored(q, {"*.q_proj"}));
    // Trailing segment, how a short entry resolves.
    EXPECT_TRUE(pol::module_is_ignored("model.lm_head", {"lm_head"}));
    // ...but only at a segment boundary, so `q_proj` does not swallow `xq_proj`.
    EXPECT_FALSE(pol::module_is_ignored("model.layers.3.self_attn.xq_proj", {"q_proj"}));
    EXPECT_FALSE(pol::module_is_ignored(q, {"model.layers.30.self_attn.q_proj"}));
}

TEST(NvFP4IgnoreList, ModuleNameSurvivesEverySuffixSpelling) {
    // The caller may hold either side of the .weight_packed rename.
    for (const char* suf : {".weight", ".weight_packed", ".weight_scale", ".weight_global_scale",
                            ".weight_scale_2", ".input_scale", ".input_global_scale", ".bias"}) {
        EXPECT_EQ(pol::module_of_tensor("model.language_model.layers.1.mlp.up_proj" + std::string(suf)),
                  "model.layers.1.mlp.up_proj")
            << suf;
    }
}

// ---- merged-scale provenance (#1960) ----------------------------------------
//
// The fused-split fix-up used to fire on a predicate that a separate-tensor
// checkpoint also satisfies whenever a sibling failed to promote for an
// unrelated reason. On Qwen3.8-27B that aims w_up's micro-scales 17408 * 320 =
// 5.57 MB past the end of w_gate's plane, logged as a normal split.

namespace {

imp::FusedSplitRequest gate_up_request() {
    imp::FusedSplitRequest r;
    r.provenance = true;
    r.base_rows = 17408;
    r.sib_rows = 17408;
    r.n_sibs = 1;
    r.base_k_packed = 2560;
    r.sib_k_packed = 2560;
    r.plane_rows = 2 * 17408;
    return r;
}

}  // namespace

// The WIRING, not the rule: `fused_split_eligible` below is driven with the flag
// set both ways, but nothing there notices if the only producer of the flag
// stops producing it. That separation is the #1929 shape, so the producer gets
// its own test. Geometry from Phi-4-reasoning-plus-NVFP4 (fused `qkv_proj` and
// `gate_up_proj`), scaled down: q + 2kv rows for the attention tensor, 2 x half
// for the MLP one.
namespace {

struct FusedSplitFixture {
    std::vector<uint16_t> backing;
    imp::Model model;
    std::unordered_map<std::string, imp::Tensor> tensors;

    FusedSplitFixture() : backing(1 << 14, 0) {
        model.config_.arch = imp::ModelArch::LLAMA;
        model.config_.n_layers = 1;
        model.config_.d_model = 64;
        model.config_.n_heads = 8;
        model.config_.n_kv_heads = 2;
        model.config_.head_dim = 8;
        model.layers_.resize(1);
    }

    void add(const std::string& name, int64_t rows, int64_t cols) {
        imp::Tensor t;
        t.data = backing.data();
        t.qtype = imp::QType::F16;
        t.ndim = 2;
        t.shape[0] = rows;
        t.shape[1] = cols;
        tensors[name] = t;
    }

    void apply() {
        imp::WeightMap wm(imp::ModelArch::LLAMA);
        wm.apply_weights(model, tensors);
    }
};

}  // namespace

TEST(NvFP4MergedScaleProvenance, WeightMapRecordsTheFusedSplit) {
    FusedSplitFixture f;
    // q_rows = 8 * 8 = 64, kv_rows = 2 * 8 = 16, so the fused tensor has 96 rows.
    f.add("model.layers.0.self_attn.qkv_proj.weight", 96, 32);
    f.add("model.layers.0.mlp.gate_up_proj.weight", 128, 32);
    f.apply();

    const auto& L = f.model.layers_[0];
    EXPECT_EQ(L.wq.shape[0], 64);
    EXPECT_EQ(L.wk.shape[0], 16);
    EXPECT_EQ(L.w_up.shape[0], 64);
    EXPECT_TRUE(L.qkv_split_from_fused)
        << "the qkv_proj split is the only producer of this flag; without it the loader's scale "
           "fix-up cannot tell a split from an unrelated promotion failure";
    EXPECT_TRUE(L.gate_up_split_from_fused);
}

TEST(NvFP4MergedScaleProvenance, SeparateTensorsRecordNoSplit) {
    FusedSplitFixture f;
    f.add("model.layers.0.self_attn.q_proj.weight", 64, 32);
    f.add("model.layers.0.self_attn.k_proj.weight", 16, 32);
    f.add("model.layers.0.self_attn.v_proj.weight", 16, 32);
    f.add("model.layers.0.mlp.gate_proj.weight", 64, 32);
    f.add("model.layers.0.mlp.up_proj.weight", 64, 32);
    f.apply();

    const auto& L = f.model.layers_[0];
    ASSERT_NE(L.wq.data, nullptr) << "the fixture did not reach the mapper";
    EXPECT_FALSE(L.qkv_split_from_fused);
    EXPECT_FALSE(L.gate_up_split_from_fused);
}

TEST(NvFP4MergedScaleProvenance, FixUpNeedsAnActualSplit) {
    imp::FusedSplitRequest r = gate_up_request();
    std::string why;
    ASSERT_TRUE(imp::fused_split_eligible(r, &why)) << why;

    // The defect: separate tensors, sibling merely unpromoted.
    r.provenance = false;
    EXPECT_FALSE(imp::fused_split_eligible(r, &why));
    EXPECT_NE(why.find("separate tensors"), std::string::npos) << why;
}

TEST(NvFP4MergedScaleProvenance, ShapeBeltStopsAnOutOfPlaneSplit) {
    std::string why;
    // The scale plane was already sliced per sibling, so it covers one half.
    imp::FusedSplitRequest r = gate_up_request();
    r.plane_rows = 17408;
    EXPECT_FALSE(imp::fused_split_eligible(r, &why));
    EXPECT_NE(why.find("already sliced"), std::string::npos) << why;

    // Unknown plane geometry declines rather than guesses.
    r = gate_up_request();
    r.plane_rows = 0;
    EXPECT_FALSE(imp::fused_split_eligible(r, &why));

    // Siblings that disagree about K cannot share one plane.
    r = gate_up_request();
    r.sib_k_packed = 1280;
    EXPECT_FALSE(imp::fused_split_eligible(r, &why));
    EXPECT_NE(why.find("K"), std::string::npos) << why;

    // Three-way qkv geometry: the plane must hold q + 2 * kv.
    imp::FusedSplitRequest qkv;
    qkv.provenance = true;
    qkv.base_rows = 6144;
    qkv.sib_rows = 1024;
    qkv.n_sibs = 2;
    qkv.base_k_packed = qkv.sib_k_packed = 2560;
    EXPECT_EQ(imp::fused_split_needed_rows(qkv), 8192);
    qkv.plane_rows = 8192;
    EXPECT_TRUE(imp::fused_split_eligible(qkv, &why)) << why;
    qkv.plane_rows = 8191;
    EXPECT_FALSE(imp::fused_split_eligible(qkv, &why));
}

namespace {

// Two layouts a fused gate|up group can arrive in.
//
// SLICED is what every producer in the tree makes today and what
// Phi-4-reasoning-plus-NVFP4 loads as: weight_map cuts the `[2*half, K/16]`
// plane in two and weight_upload gives each half its own allocation, so the
// siblings share the global scale and nothing else.
//
// ONE PLANE is what the loader's fix-up arm makes: sibling pointers are offsets
// into the base's plane, and only then is a row offset a meaningful assertion.
// Same geometry as Qwen3.8-27B (17408 rows, 2560 packed cols) scaled down so the
// pointer arithmetic stays inside a real buffer.
constexpr int64_t kRows = 8;
constexpr int64_t kRowBytes = 2560 / 8 / 40;  // 8

imp::MergedScaleGroup sliced_gate_up(const char* plane_gate, const char* plane_up) {
    imp::MergedScaleGroup g{};
    g.layer = 7;
    g.what = "gate|up";
    g.count = 2;
    g.fused = true;
    g.spans_one_plane = false;
    g.scale_row_bytes = kRowBytes;
    g.m[0] = {plane_gate, 0.0125f, kRows, kRows};
    g.m[1] = {plane_up, 0.0125f, kRows, kRows};
    return g;
}

imp::MergedScaleGroup fused_gate_up(const char* plane) {
    imp::MergedScaleGroup g{};
    g.layer = 7;
    g.what = "gate|up";
    g.count = 2;
    g.fused = true;
    g.spans_one_plane = true;
    g.scale_row_bytes = kRowBytes;
    g.m[0] = {plane, 0.0125f, kRows, 2 * kRows};
    g.m[1] = {plane + kRows * kRowBytes, 0.0125f, kRows, 0};
    return g;
}

}  // namespace

TEST(NvFP4MergedScaleProvenance, FusedGroupGeometryIsAsserted) {
    // A stand-in address space: nothing is dereferenced, only compared.
    alignas(64) static char plane[4 * kRows * kRowBytes];
    std::string err;
    ASSERT_TRUE(imp::merged_scale_group_ok(fused_gate_up(plane), &err)) << err;

    // Wrong row offset (the classic off-by-a-plane).
    imp::MergedScaleGroup g = fused_gate_up(plane);
    g.m[1].scales = plane + 2 * kRows * kRowBytes;
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("row offset"), std::string::npos) << err;

    // Siblings of one fused tensor share one weight_global_scale: bit equality,
    // not a tolerance.
    g = fused_gate_up(plane);
    g.m[1].tensor_scale = std::nextafterf(g.m[0].tensor_scale, 1.0f);
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("tensor_scale"), std::string::npos) << err;

    // The split must stay inside the plane it was cut from.
    g = fused_gate_up(plane);
    g.m[0].plane_rows = kRows + 1;
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("rows of a plane"), std::string::npos) << err;

    // A sibling that lost its scales is not a servable split.
    g = fused_gate_up(plane);
    g.m[1].scales = nullptr;
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
}

// The layout Phi-4-reasoning-plus-NVFP4 actually loads as. Asserting a row
// offset here refused it: three independent cudaMallocAsync results are not one
// plane, and `qkv_split_from_fused` never claimed they were.
TEST(NvFP4MergedScaleProvenance, SlicedSiblingsOfAFusedTensorPass) {
    alignas(64) static char plane_gate[kRows * kRowBytes], plane_up[kRows * kRowBytes];
    std::string err;
    EXPECT_TRUE(imp::merged_scale_group_ok(sliced_gate_up(plane_gate, plane_up), &err)) << err;

    // One global scale for the whole fused tensor: a sibling carrying a
    // different one was promoted against a scale the checkpoint never gave it.
    imp::MergedScaleGroup g = sliced_gate_up(plane_gate, plane_up);
    g.m[1].tensor_scale = std::nextafterf(g.m[0].tensor_scale, 1.0f);
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("tensor_scale"), std::string::npos) << err;

    // Each sibling owns a plane of exactly its own rows. Too large means the
    // slice never happened and the sibling reads its neighbour's rows as its own.
    g = sliced_gate_up(plane_gate, plane_up);
    g.m[1].plane_rows = 2 * kRows;
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("scale plane of"), std::string::npos) << err;

    // An unknown plane row count (no scratch entry) is not an accusation.
    g = sliced_gate_up(plane_gate, plane_up);
    g.m[0].plane_rows = 0;
    g.m[1].plane_rows = 0;
    EXPECT_TRUE(imp::merged_scale_group_ok(g, &err)) << err;
}

TEST(NvFP4MergedScaleProvenance, SeparateTensorsAreNotChecked) {
    // Qwen3.8-27B has separate q/k/v. Two independent scale planes may land
    // exactly one plane apart, so offsets must NOT be asserted here.
    alignas(64) static char a[32], b[32];
    imp::MergedScaleGroup g{};
    g.layer = 0;
    g.what = "q|k|v";
    g.count = 3;
    g.fused = false;
    g.scale_row_bytes = 320;
    g.m[0] = {a, 0.01f, 12288};
    g.m[1] = {b, 0.02f, 1024};
    g.m[2] = {b + 16, 0.03f, 1024};
    std::string err;
    EXPECT_TRUE(imp::merged_scale_group_ok(g, &err)) << err;

    // Exact pointer equality cannot happen between two live allocations, so it
    // is the one signature a non-fused group is still held to.
    g.m[2].scales = b;
    EXPECT_FALSE(imp::merged_scale_group_ok(g, &err));
    EXPECT_NE(err.find("same scale pointer"), std::string::npos) << err;
}

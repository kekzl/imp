#pragma once

// The full recipe of one export, written into the checkpoint (#2481): which binary, which flags,
// which calibration data. Same binary + same recipe -> byte-identical files. No timestamps, no
// paths: either would make two identical runs differ.

#include <cstdint>
#include <string>

namespace imp::quantize {

// Bump when a field changes meaning or is removed; adding a field keeps it.
inline constexpr int kRecipeVersion = 1;

struct Recipe {
    std::string imp_version;  // imp_version()
    std::string imp_tree;     // git tree id of the build (scripts/image_tag.sh tree), "unknown" in dev builds
    std::string format;       // modelopt | compressed-tensors
    std::string lm_head;      // fp8 | nvfp4 | source, after the format default
    std::string kv_hint;      // auto | fp8 | none, as passed
    bool kv_cache_fp8 = false;  // what the export declares
    bool keep_attn_gate = false;
    std::string keep_gdn_proj;  // "" = quantized like any Linear
    std::string gdn_proj_format = "source";  // source | mxfp8 (kept GDN projections, #2475)
    // Calibration; the rest of this block is empty when calibrated is false.
    bool calibrated = false;
    std::string calib_sha256;    // of the --calib file
    std::string calib_model_id;  // the file's own provenance string
    uint64_t calib_samples = 0;  // largest row count over entries
    uint64_t calib_entries = 0;
    std::string calib_groups;  // the selector that ran (explicit or the n_rep/hybrid default)
    std::string calib_weight;  // abs | sq
    int64_t n_rep = 0;
    bool hybrid = false;
};

struct Options;

// Everything the command line and the build decide; calib_model_id/samples/entries and the AWQ
// plan fields (groups, n_rep, hybrid) are the caller's, set where the calibration file is read.
void fill_recipe(Recipe& r, const Options& opt, bool kv_cache_fp8);

// IMP_TREE_ID baked in at configure time, "unknown" when the build had none.
const char* build_tree_id();

// The recipe as a JSON object, keys in fixed order, each line prefixed with `indent`
// (the opening brace is not).
std::string recipe_json(const Recipe& r, const std::string& indent);

}  // namespace imp::quantize

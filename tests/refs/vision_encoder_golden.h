// Frozen vision-encoder regression goldens (R9 / issue #583).
//
// THESE ARE A STABILITY LOCK, NOT AN EXTERNAL GROUND TRUTH.
// Unlike the fp64 class-A references in this directory, there is no
// independent oracle for the SigLIP / gemma4v encoder + projector tail.
// The goal here is OUTPUT STABILITY of the GPU encoder+projector: spot
// values of the projector-output image embeddings, frozen from a
// known-good, manually-validated build (PR #489 VL path), asserted at the
// f16 class tolerance (<= 1e-2 rel, small abs floor) on the same machine.
//
// Provenance:
//   commit            : 12aff1ce (branch test/r9-vision-golden, off main)
//   date              : 2026-06-07
//   GPU               : RTX 5090 (sm_120a), CUDA 13.3
//   image fixture     : tests/fixtures/vision_test_64.png (64x64 synthetic
//                       deterministic RGB, stb_image PNG decode)
//   gemma-3 mmproj    : mmproj-F16.gguf  (851251328 bytes)
//   gemma4v mmproj    : mmproj-gemma4-26b-bf16.gguf (1194827840 bytes)
//
// Regeneration (host stays clean; run on the 5090 in the test container):
//   IMP_TEST_MMPROJ=/models/gemma-3-4b-vl/mmproj-F16.gguf \
//   IMP_TEST_MMPROJ_GEMMA4=/models/gemma-3-4b-vl/mmproj-gemma4-26b-bf16.gguf \
//   IMP_VISION_GOLDEN_DUMP=1 imp-tests --gtest_filter='VisionGolden.*'
//   then paste the emitted blocks below (overwrite the matching arch).
// See tests/refs/README.md.
#pragma once

namespace imp_refs {

// Each golden samples the projector-output embedding tensor
// [num_image_tokens, lm_d_model] (FP16, on device, copied to host) at:
//   - spot_idx/spot_val : a strided set of flat indices across the whole
//                         tensor (covers first/last tokens + interior),
//   - tok_l2            : per-token L2 norm for a few sampled tokens,
//   - global_mean       : mean over the entire tensor.
struct VisionGoldenArch {
    const char* name;
    int num_tokens;
    int d_model;
    int n_spot;
    const int* spot_idx;
    const float* spot_val;
    int n_tok_l2;
    const int* tok_l2_idx;  // token indices the L2 norms below correspond to
    const float* tok_l2;
    float global_mean;
};

// ===================== gemma-3 SigLIP =====================
// Frozen 2026-06-07, commit 12aff1ce, RTX 5090 / CUDA 13.3, mmproj-F16.gguf.
#define IMP_VISION_GOLDEN_GEMMA3 1
inline constexpr int gemma3_num_tokens = 256;
inline constexpr int gemma3_d_model = 2560;
inline constexpr int gemma3_n_spot = 64;
inline constexpr int gemma3_n_tok_l2 = 8;
inline constexpr int gemma3_spot_idx[64] = {0,      10402,  20805,  31207,  41610,  52012,  62415,  72817,
                                            83220,  93622,  104025, 114427, 124830, 135232, 145635, 156037,
                                            166440, 176842, 187245, 197647, 208050, 218453, 228855, 239258,
                                            249660, 260063, 270465, 280868, 291270, 301673, 312075, 322478,
                                            332880, 343283, 353685, 364088, 374490, 384893, 395295, 405698,
                                            416100, 426503, 436906, 447308, 457711, 468113, 478516, 488918,
                                            499321, 509723, 520126, 530528, 540931, 551333, 561736, 572138,
                                            582541, 592943, 603346, 613748, 624151, 634553, 644956, 655359};
inline constexpr float gemma3_spot_val[64] = {
    -0.1256104f, -0.6547852f, -0.135498f,  0.09173584f, -0.06488037f, -0.00610733f, -0.4921875f,
    0.8427734f,  -0.8549805f, -1.833984f,  -0.2905273f, 1.306641f,    0.5498047f,   -1.635742f,
    0.1694336f,  1.379883f,   -0.5175781f, 0.380127f,   0.3884277f,   -0.1134033f,  0.1923828f,
    0.815918f,   0.3181152f,  -0.3383789f, 1.819336f,   -0.494873f,   0.07684326f,  -0.04492188f,
    -0.3212891f, -1.026367f,  1.081055f,   -0.3330078f, 1.111328f,    0.2260742f,   2.080078f,
    0.3190918f,  -1.618164f,  -0.784668f,  -5.042969f,  -0.6943359f,  0.2454834f,   0.3046875f,
    -0.4887695f, 0.6430664f,  0.6240234f,  -0.3237305f, 0.2249756f,   0.5053711f,   0.722168f,
    0.6918945f,  -1.027344f,  1.035156f,   0.5097656f,  -0.6977539f,  -0.3017578f,  0.3947754f,
    0.6298828f,  2.025391f,   2.232422f,   0.3017578f,  -0.8164062f,  -1.658203f,   0.6401367f,
    -0.07855225f};
inline constexpr int gemma3_tok_l2_idx[8] = {0, 36, 72, 109, 145, 182, 218, 255};
inline constexpr float gemma3_tok_l2[8] = {67.53173f, 65.16915f, 70.74292f, 62.58034f,
                                           65.39104f, 60.65078f, 67.3276f,  63.26681f};
inline constexpr float gemma3_global_mean = 0.05464977f;

// ===================== gemma4v =====================
// Frozen 2026-06-07, commit 12aff1ce, RTX 5090 / CUDA 13.3,
// mmproj-gemma4-26b-bf16.gguf (encoded standalone, no 26B LM needed).
#define IMP_VISION_GOLDEN_GEMMA4V 1
inline constexpr int gemma4v_num_tokens = 256;
inline constexpr int gemma4v_d_model = 2816;
inline constexpr int gemma4v_n_spot = 64;
inline constexpr int gemma4v_n_tok_l2 = 8;
inline constexpr int gemma4v_spot_idx[64] = {0,      11442,  22885,  34328,  45771,  57213,  68656,  80099,
                                             91542,  102985, 114427, 125870, 137313, 148756, 160198, 171641,
                                             183084, 194527, 205970, 217412, 228855, 240298, 251741, 263183,
                                             274626, 286069, 297512, 308955, 320397, 331840, 343283, 354726,
                                             366168, 377611, 389054, 400497, 411940, 423382, 434825, 446268,
                                             457711, 469153, 480596, 492039, 503482, 514925, 526367, 537810,
                                             549253, 560696, 572138, 583581, 595024, 606467, 617910, 629352,
                                             640795, 652238, 663681, 675123, 686566, 698009, 709452, 720895};
inline constexpr float gemma4v_spot_val[64] = {
    -2.113281f,   2.712891f,  -0.4248047f, 0.8139648f,  -0.7368164f, -3.121094f, 0.8339844f,  1.895508f,
    0.2553711f,   -2.076172f, -1.629883f,  2.951172f,   -0.3237305f, -1.496094f, -0.1594238f, -0.6591797f,
    -0.05264282f, 0.9882812f, 6.585938f,   0.5913086f,  0.1244507f,  0.3364258f, -1.703125f,  0.2626953f,
    1.458008f,    1.019531f,  -1.569336f,  1.010742f,   -4.933594f,  2.269531f,  1.538086f,   0.4328613f,
    3.574219f,    -1.416016f, -0.3601074f, 0.6806641f,  1.418945f,   -1.03125f,  0.2198486f,  0.5498047f,
    -0.6743164f,  0.7802734f, -0.1602783f, 0.6450195f,  0.3984375f,  -1.486328f, 0.7553711f,  1.131836f,
    0.6083984f,   -2.994141f, 0.07037354f, -1.833008f,  -2.808594f,  1.1875f,    -2.970703f,  0.05194092f,
    -0.6455078f,  2.994141f,  -1.288086f,  -0.4248047f, 1.334961f,   -2.722656f, 2.300781f,   -0.005630493f};
inline constexpr int gemma4v_tok_l2_idx[8] = {0, 36, 72, 109, 145, 182, 218, 255};
inline constexpr float gemma4v_tok_l2[8] = {87.7687f, 85.37603f, 84.25941f, 87.10169f,
                                            83.3417f, 93.60386f, 86.6295f,  88.06613f};
inline constexpr float gemma4v_global_mean = -0.02568814f;

// Accessors: return the frozen golden for an arch, or nullptr if not yet
// committed (the test then SKIPs with a regeneration hint). Define
// IMP_VISION_GOLDEN_<ARCH> to 1 once the arrays above are filled in.
inline const VisionGoldenArch* gemma3_golden() {
#ifdef IMP_VISION_GOLDEN_GEMMA3
    static const VisionGoldenArch g{"gemma3",        gemma3_num_tokens, gemma3_d_model,  gemma3_n_spot,
                                    gemma3_spot_idx, gemma3_spot_val,   gemma3_n_tok_l2, gemma3_tok_l2_idx,
                                    gemma3_tok_l2,   gemma3_global_mean};
    return &g;
#else
    return nullptr;
#endif
}

inline const VisionGoldenArch* gemma4v_golden() {
#ifdef IMP_VISION_GOLDEN_GEMMA4V
    static const VisionGoldenArch g{"gemma4v",          gemma4v_num_tokens, gemma4v_d_model,
                                    gemma4v_n_spot,     gemma4v_spot_idx,   gemma4v_spot_val,
                                    gemma4v_n_tok_l2,   gemma4v_tok_l2_idx, gemma4v_tok_l2,
                                    gemma4v_global_mean};
    return &g;
#else
    return nullptr;
#endif
}

}  // namespace imp_refs

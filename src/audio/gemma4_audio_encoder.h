#pragma once
// Gemma-4 audio forward on the GPU: subsampling convs -> 12 Conformer blocks (FFW/2, chunked
// relative-position attention, light conv, FFW/2) -> output_proj -> embedder (RMSNorm without
// scale + projection to the LM width). FP32 residual stream, FP16 GEMM operands with FP32
// accumulation. Scratch comes from the engine arena, sized for max_frames once.

#include "audio/audio_features.h"
#include "audio/gemma4_audio_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace imp {

// Host copies of intermediate states for the reference test (rows x width, FP32).
struct Gemma4AudioTaps {
    std::vector<float> subsample;  // [tokens, hidden] after input_proj_linear
    std::vector<float> layer0;     // [tokens, hidden] after block 0
    std::vector<float> tower;      // [tokens, output_proj_dims] after output_proj
};

class Gemma4AudioEncoder {
public:
    Gemma4AudioEncoder() = default;
    Gemma4AudioEncoder(const Gemma4AudioEncoder&) = delete;
    Gemma4AudioEncoder& operator=(const Gemma4AudioEncoder&) = delete;

    // Frames a max_samples clip produces (front end arithmetic).
    static int max_frames_for(const Gemma4AudioFrontend& fe);
    // Arena bytes init() takes for `max_frames`, shapes only.
    static size_t demand_bytes(const Gemma4AudioConfig& c, int max_frames);
    // `m` must be uploaded (gemma4_audio_upload).
    [[nodiscard]] bool init(const Gemma4AudioModel& m, int max_frames);
    size_t taken_bytes() const { return taken_; }

    // Writes f.n_tokens rows [n_tokens, lm_d_model] FP16 to d_out. Synchronises the stream.
    [[nodiscard]] bool encode(const AudioFeatures& f, half* d_out, cudaStream_t stream,
                              Gemma4AudioTaps* taps = nullptr);

private:
    void linear_(const half* in, const Gemma4AudioLinear& l, half* out, int rows, int n_out, int n_in,
                 cudaStream_t stream);
    void ffw_(const Gemma4AudioFfw& w, int rows, cudaStream_t stream);
    void block_(const Gemma4AudioLayer& L, int rows, cudaStream_t stream);

    const Gemma4AudioModel* m_ = nullptr;
    int max_frames_ = 0;
    size_t taken_ = 0;
    float* d_feat_ = nullptr;     // [F, n_mels]
    float* d_c0_ = nullptr;       // [c0, F1, n_mels/2] conv0 out, then LN+ReLU+mask in place
    float* d_c1_ = nullptr;       // [c1, T, n_mels/4]
    uint8_t* d_mask1_ = nullptr;  // [F1]
    uint8_t* d_valid_ = nullptr;  // [T]
    float* d_x_ = nullptr;        // [T, H] residual
    half* d_a16_ = nullptr;       // [T, max(H, sub width)]
    half* d_tmp16_ = nullptr;     // [T, 4H] clamped linear input
    half* d_wide16_ = nullptr;    // [T, 4H]
    half* d_q_ = nullptr;
    half* d_k_ = nullptr;
    half* d_v_ = nullptr;
    half* d_o_ = nullptr;
    half* d_h16_ = nullptr;       // [T, H]
    half* d_proj16_ = nullptr;    // [T, out_dims]
    half* d_emb_in16_ = nullptr;  // [T, out_dims]
};

}  // namespace imp

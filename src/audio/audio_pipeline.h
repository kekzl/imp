#pragma once
// WAV bytes to LM-ready audio embeddings (Gemma-4). Tower weights live in the Model (same
// checkpoint); this uploads them on init and owns the encoder scratch. preprocess is CPU only
// (any thread); encode_to is serialized on the batch worker, like the vision pipelines.

#include "audio/audio_features.h"
#include "audio/gemma4_audio_encoder.h"
#include "audio/gemma4_audio_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <expected>
#include <span>
#include <string>

namespace imp {

class AudioPipeline {
public:
    // Arena bytes init() takes: tower plus encoder scratch for a max-length clip.
    static size_t demand_bytes(Gemma4AudioModel& tower);
    [[nodiscard]] bool init(Gemma4AudioModel& tower);
    size_t taken_bytes() const { return taken_; }
    bool is_ready() const { return ready_; }

    [[nodiscard]] std::expected<AudioFeatures, std::string> preprocess(std::span<const uint8_t> wav) const;
    // Writes features.n_tokens rows [n_tokens, lm_d()] FP16 to d_out.
    [[nodiscard]] bool encode_to(const AudioFeatures& features, half* d_out, cudaStream_t stream);

    int lm_d() const { return tower_ ? tower_->config.lm_d_model : 0; }
    int32_t audio_token_id() const { return tower_ ? tower_->config.audio_token_id : -1; }
    int32_t boa_token_id() const { return tower_ ? tower_->config.boa_token_id : -1; }
    int32_t eoa_token_id() const { return tower_ ? tower_->config.eoa_token_id : -1; }

private:
    Gemma4AudioModel* tower_ = nullptr;
    Gemma4AudioEncoder encoder_;
    Gemma4AudioFrontend fe_;
    size_t taken_ = 0;
    bool ready_ = false;
};

}  // namespace imp

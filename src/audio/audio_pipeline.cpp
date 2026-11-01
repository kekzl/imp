#include "audio/audio_pipeline.h"

#include "core/logging.h"

namespace imp {

size_t AudioPipeline::demand_bytes(Gemma4AudioModel& tower) {
    const Gemma4AudioFrontend fe;
    return gemma4_audio_tower_device_bytes(tower) +
           Gemma4AudioEncoder::demand_bytes(tower.config, Gemma4AudioEncoder::max_frames_for(fe));
}

bool AudioPipeline::init(Gemma4AudioModel& tower) {
    auto up = gemma4_audio_upload(tower);
    if (!up) {
        IMP_LOG_WARN("Audio tower upload failed: %s", up.error().c_str());
        return false;
    }
    tower_ = &tower;
    if (!encoder_.init(tower, Gemma4AudioEncoder::max_frames_for(fe_)))
        return false;
    taken_ = *up + encoder_.taken_bytes();
    ready_ = true;
    IMP_LOG_INFO("Audio: Gemma-4 encoder ready (%d layers, hidden %d, up to %d s per clip, %.1f MiB)",
                 tower.config.num_layers, tower.config.hidden_size, fe_.max_samples / fe_.sample_rate,
                 static_cast<double>(taken_) / (1024.0 * 1024.0));
    return true;
}

std::expected<AudioFeatures, std::string> AudioPipeline::preprocess(std::span<const uint8_t> wav) const {
    return gemma4_audio_from_wav(wav, fe_);
}

bool AudioPipeline::encode_to(const AudioFeatures& features, half* d_out, cudaStream_t stream) {
    return ready_ && encoder_.encode(features, d_out, stream);
}

}  // namespace imp

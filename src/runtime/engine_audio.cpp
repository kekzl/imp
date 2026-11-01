// Engine side of audio input: features on the handler thread (AudioPipeline::preprocess), encode
// on the batch worker, spliced like mmproj images (vision_emb at the audio soft token).

#include "core/buffer.h"
#include "core/logging.h"
#include "runtime/engine.h"
#include "runtime/request.h"

namespace imp {

bool Engine::encode_audio_for(Request& req) {
    if (req.audios.empty() || !audio_.is_ready() || !req.images.empty())
        return false;
    // One block of rows per clip, prompt order: the k-th audio token of the prompt takes row k.
    size_t total = 0;
    for (const auto& a : req.audios)
        total += static_cast<size_t>(a->n_tokens);
    const size_t row = static_cast<size_t>(audio_.lm_d());
    auto buf = std::make_shared<Buffer>(Buffer::device(total * row * sizeof(half)));
    if (total == 0 || !*buf)
        return false;
    size_t at = 0;
    for (const auto& a : req.audios) {
        if (!audio_.encode_to(*a, buf->as<half>() + (at * row), stream_)) {
            IMP_LOG_ERROR("audio encode failed (clip of %d frames, %d tokens)", a->n_frames, a->n_tokens);
            return false;
        }
        at += static_cast<size_t>(a->n_tokens);
    }
    req.vision_emb = std::move(buf);
    req.vision_token_id = audio_.audio_token_id();
    req.n_vision_tokens = static_cast<int>(total);
    req.audios.clear();
    return true;
}

}  // namespace imp

// Logit lens wire shape (roadmap row 115), one object per layer on a chat logprobs content entry.

#include "utils.h"

json logit_lens_json(const std::vector<imp::LogitLensLayer>& layers) {
    json out = json::array();
    for (const auto& ly : layers) {
        json top = json::array();
        for (const auto& t : ly.top)
            top.push_back({{"token", safe_token_json(t.text)}, {"logprob", t.logprob}});
        out.push_back({{"rank", ly.rank}, {"logprob", ly.logprob}, {"entropy", ly.entropy}, {"top", top}});
    }
    return out;
}

json sae_json(const imp::SaeTokenInfo& s) {
    json feats = json::array();
    for (const auto& [id, act] : s.top)
        feats.push_back({{"id", id}, {"act", act}});
    return {{"l0", s.l0}, {"err", s.rel_err}, {"features", feats}};
}

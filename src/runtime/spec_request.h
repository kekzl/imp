#pragma once

// The per-request half of the speculation contract.
//
// `speculative.mtp_k` is the SERVER's default depth, and a request picks a
// depth within it: anything from 0 (off for this request) up to the armed
// chain. What no request can do is turn the head ON where the process armed
// nothing, because the checkpoint's MTP tensors are ~0.79 GiB uploaded at load
// time and the decision to upload them is made before an engine exists
// (tools/common/mtp_auto.*). Such a request gets an ANSWER SAYING SO, not a
// silent plain decode: the reason travels back in
// `usage.completion_tokens_details.imp_spec_declined`, on all three dialects.
//
// Kept as a pure function so the truth table is testable without a GPU, a
// model or a live Engine, and so the server and the engine cannot disagree
// about what a request asked for: both call this with the same inputs.

namespace imp {

// Wire ceiling for a per-request depth. The device chain buffer
// (kMtpMaxChainK, src/compute/mtp_forward.h) is the real bound; this header
// stays free of the CUDA include chain (Engine forward-declares the workspace
// for the same reason), so the two are tied by a static_assert in
// engine_spec_mtp.cpp rather than by an include.
inline constexpr int kSpecRequestMaxMtpK = 16;

enum class SpecDecline {
    kNone = 0,       // nothing was declined
    kRequestOff,     // the caller switched speculation off
    kNoHead,         // this checkpoint ships no MTP head at all
    kHeadNotLoaded,  // the checkpoint has one, this process did not upload it
    kHeadNotArmed,   // the head is loaded, no depth was armed
    kDepthClamped,   // asked deeper than the armed chain
};

// Stable wire names. They go into a response body, so they are part of the
// API surface: add, never rename.
constexpr const char* spec_decline_name(SpecDecline d) {
    switch (d) {
        case SpecDecline::kNone:
            return "none";
        case SpecDecline::kRequestOff:
            return "request_off";
        case SpecDecline::kNoHead:
            return "no_mtp_head";
        case SpecDecline::kHeadNotLoaded:
            return "mtp_head_not_loaded";
        case SpecDecline::kHeadNotArmed:
            return "mtp_head_not_armed";
        case SpecDecline::kDepthClamped:
            return "mtp_depth_clamped";
    }
    return "none";
}

// One-line explanation per reason, for the response and the log. The
// concurrency decline is the one an operator actually hits: MTP auto stays off
// whenever the server takes concurrent requests, and until now the only trace
// of that decision was a startup INFO line.
constexpr const char* spec_decline_detail(SpecDecline d) {
    switch (d) {
        case SpecDecline::kNone:
            return "";
        case SpecDecline::kRequestOff:
            return "speculation disabled by this request";
        case SpecDecline::kNoHead:
            return "this checkpoint ships no MTP head";
        case SpecDecline::kHeadNotLoaded:
            return "the checkpoint ships an MTP head that this server did not load; "
                   "speculative.mtp_k=auto declines it outside a single-stream run, "
                   "force it with --set speculative.mtp_k=2 --set speculative.ngram=false";
        case SpecDecline::kHeadNotArmed:
            return "the MTP head is loaded but no chain depth is armed (speculative.mtp_k=0)";
        case SpecDecline::kDepthClamped:
            return "requested depth exceeds the armed chain, clamped to it";
    }
    return "";
}

struct MtpRequestState {
    // Request::spec_mtp_k. -1 = the caller said nothing, take the server default.
    int requested_k = -1;
    // Engine::mtp_spec_decode_k(): the depth this process armed. 0 = none.
    int armed_k = 0;
    // The checkpoint carries MTP tensors (loaded or not).
    bool head_present = false;
    // ... and this process uploaded them.
    bool head_loaded = false;
    // `"speculative": false` on the request.
    bool forced_off = false;
};

struct MtpResolution {
    int k = 0;  // chain depth for this request, 0 = no MTP drafting
    SpecDecline reason = SpecDecline::kNone;
};

// The rule. `reason` is kNone whenever the request got what it asked for,
// INCLUDING the ordinary case of a request that asked for nothing on a server
// without a head: silence is only a defect when someone asked.
constexpr MtpResolution mtp_resolve_request(const MtpRequestState& s) {
    if (s.requested_k < 0)  // unset: the server default, whatever it is
        return {s.forced_off ? 0 : s.armed_k, SpecDecline::kNone};
    if (s.forced_off || s.requested_k == 0)
        return {0, SpecDecline::kRequestOff};
    if (!s.head_present)
        return {0, SpecDecline::kNoHead};
    if (!s.head_loaded)
        return {0, SpecDecline::kHeadNotLoaded};
    if (s.armed_k <= 0)
        return {0, SpecDecline::kHeadNotArmed};
    if (s.requested_k > s.armed_k)
        return {s.armed_k, SpecDecline::kDepthClamped};
    return {s.requested_k, SpecDecline::kNone};
}

// Does this decline belong in the response? A caller that switched
// speculation off already knows; every other reason is news.
constexpr bool spec_decline_is_reportable(SpecDecline d) {
    return d != SpecDecline::kNone && d != SpecDecline::kRequestOff;
}

}  // namespace imp

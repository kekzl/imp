// imp chat extensions a load or a request cannot honour (rows 115, 118): checked after the model snapshot.

#include "handlers_internal.h"
#include "runtime/cfg_guidance.h"
#include "exec/sae_token.h"
#include "runtime/phrase_ban.h"

#include <algorithm>
#include <charconv>
#include <cmath>

std::string parse_cfg_fields(const nlohmann::json& body, ChatRequestParams& p) {
    if (const auto s = body.find("cfg_scale"); s != body.end() && !s->is_null()) {
        if (!s->is_number() || s->get<double>() < 1.0 || s->get<double>() > imp::kMaxCfgScale)
            return "\"cfg_scale\" must be a number in [1, " +
                   std::to_string(static_cast<int>(imp::kMaxCfgScale)) + "]";
        p.cfg_scale = s->get<float>();
    }
    if (const auto n = body.find("cfg_negative_prompt"); n != body.end() && !n->is_null()) {
        if (!n->is_string())
            return "\"cfg_negative_prompt\" must be a string";
        p.cfg_negative_prompt = n->get<std::string>();
    }
    return "";
}

std::string parse_sae_fields(const nlohmann::json& body, ChatRequestParams& p, const char** field) {
    if (const auto f = body.find("sae_features"); f != body.end() && !f->is_null()) {
        *field = "sae_features";
        if (!f->is_number_integer() || f->get<int64_t>() < 0 || f->get<int64_t>() > imp::kSaeMaxTopN)
            return "\"sae_features\" must be an integer 0.." + std::to_string(imp::kSaeMaxTopN);
        p.sae_features = static_cast<int>(f->get<int64_t>());
        if (p.sae_features > 0)
            p.req_logprobs = true;  // features ride on the logprobs content entries
    }
    if (const auto c = body.find("sae_clamp"); c != body.end() && !c->is_null()) {
        *field = "sae_clamp";
        if (!c->is_object() || c->empty() || c->size() > static_cast<size_t>(imp::kSaeMaxClamps))
            return "\"sae_clamp\" must be an object of 1.." + std::to_string(imp::kSaeMaxClamps) +
                   " feature ids to numbers";
        for (const auto& [key, val] : c->items()) {
            int64_t id = -1;
            const auto [end, ec] = std::from_chars(key.data(), key.data() + key.size(), id);
            if (ec != std::errc() || end != key.data() + key.size() || id < 0 || !val.is_number() ||
                !std::isfinite(val.get<double>()))
                return "\"sae_clamp\" keys must be feature ids, values finite numbers";
            p.sae_clamp.emplace_back(static_cast<int32_t>(id), val.get<float>());
        }
    }
    return "";
}

void apply_rewind_and_sae(imp::Request& req, const ChatRequestParams& p) {
    if (!p.banned_strings.empty() || !p.banned_regex.empty()) {
        req.phrase = std::make_shared<imp::PhraseBanState>();
        req.phrase->phrases = p.banned_strings;
        req.phrase->regex = p.banned_regex;
    }
    req.sae_top = p.sae_features;
    req.sae_clamp = p.sae_clamp;
}

void cfg_render_negative(ChatRequestContext& ctx, bool force_thinking) {
    if (ctx.params.cfg_scale <= 1.0f)
        return;
    std::vector<imp::ChatMessage> msgs;
    if (!ctx.params.cfg_negative_prompt.empty())
        msgs.push_back({"system", ctx.params.cfg_negative_prompt});
    for (const auto& m : ctx.params.chat_msgs)
        if (m.role != "system")
            msgs.push_back(m);
    if (!ctx.snap.have_template) {
        std::string raw;
        for (const auto& m : msgs)
            raw += m.content + "\n";
        ctx.snap.cfg_tokens = ctx.snap.tok->encode(raw);
        return;
    }
    imp::ChatTemplate tpl = ctx.snap.chat_tpl;  // the request's continue_final mode, on a copy
    tpl.set_continue_final(ctx.params.continue_final);
    ctx.snap.cfg_tokens = tpl.apply(*ctx.snap.tok, msgs, ctx.snap.suppress_thinking, force_thinking,
                                    ctx.snap.reasoning_effort);
}

ExtensionError extension_error(const ChatRequestContext& ctx) {
    const auto& p = ctx.params;
    if (p.logit_lens > 0 && !ctx.snap.logit_lens_ok)
        return {
            "\"logit_lens\" is not available for this model: its residual stream is not the d_model stream "
            "the LM head reads (gated-residual architecture)",
            "logit_lens"};
    if (p.sae_features > 0 || !p.sae_clamp.empty()) {
        const char* field = p.sae_clamp.empty() ? "sae_features" : "sae_clamp";
        if (ctx.snap.sae_width <= 0)
            return {"\"" + std::string(field) + "\" needs an SAE loaded with --sae for this model", field};
        for (const auto& [id, v] : p.sae_clamp)
            if (id >= ctx.snap.sae_width)
                return {"\"sae_clamp\" feature " + std::to_string(id) + " is outside the SAE's " +
                            std::to_string(ctx.snap.sae_width) + " features",
                        "sae_clamp"};
    }
    if (p.cfg_scale > 1.0f) {
        if (!ctx.snap.cfg_ok)
            return {
                "\"cfg_scale\" needs a model that decodes two sequences in one step (hybrids: "
                "runtime.gdn_batched_decode)",
                "cfg_scale"};
        if (!p.images.empty() || !p.videos.empty() || p.has_tools || !p.banned_strings.empty() ||
            !p.banned_regex.empty())
            return {
                "\"cfg_scale\" cannot be combined with images, video, tools, banned_strings or banned_regex",
                "cfg_scale"};
        const bool has_system = std::any_of(p.chat_msgs.begin(), p.chat_msgs.end(),
                                            [](const auto& m) { return m.role == "system"; });
        if (!has_system && p.cfg_negative_prompt.empty())
            return {
                "\"cfg_scale\" contrasts the prompt with itself minus the system message: send a system "
                "message or \"cfg_negative_prompt\"",
                "cfg_scale"};
    }
    if (p.banned_strings.empty() && p.banned_regex.empty())
        return {};
    const char* field = p.banned_strings.empty() ? "banned_regex" : "banned_strings";
    if (!ctx.snap.phrase_ban_ok)
        return {"\"" + std::string(field) +
                    "\" needs a model without the residual KV ring: a rewind cannot restore it",
                field};
    if (p.json_mode || !p.json_schema_str.empty() || !p.regex_pattern.empty() || !p.grammar.empty() ||
        !p.tool_constraint_tools.empty() || p.mirostat != 0)
        return {
            "\"" + std::string(field) +
                "\" cannot be combined with constrained output, enforced tool calls or mirostat: their state "
                "has no rewind",
            field};
    return {};
}

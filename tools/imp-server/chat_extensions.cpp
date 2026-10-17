// imp chat extensions a load or a request cannot honour (rows 115, 118): checked after the model snapshot.

#include "handlers_internal.h"
#include "runtime/cfg_guidance.h"

#include <algorithm>

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
    if (p.cfg_scale > 1.0f) {
        if (!ctx.snap.cfg_ok)
            return {
                "\"cfg_scale\" needs a model that decodes two sequences in one step (hybrids: "
                "runtime.gdn_batched_decode)",
                "cfg_scale"};
        if (!p.images.empty() || !p.videos.empty() || p.has_tools || !p.banned_strings.empty())
            return {"\"cfg_scale\" cannot be combined with images, video, tools or banned_strings",
                    "cfg_scale"};
        const bool has_system = std::any_of(p.chat_msgs.begin(), p.chat_msgs.end(),
                                            [](const auto& m) { return m.role == "system"; });
        if (!has_system && p.cfg_negative_prompt.empty())
            return {
                "\"cfg_scale\" contrasts the prompt with itself minus the system message: send a system "
                "message or \"cfg_negative_prompt\"",
                "cfg_scale"};
    }
    if (p.banned_strings.empty())
        return {};
    if (!ctx.snap.phrase_ban_ok)
        return {
            "\"banned_strings\" needs a model without recurrent state (GDN/SSM) and without the residual KV "
            "ring: a rewind restores KV only",
            "banned_strings"};
    if (p.json_mode || !p.json_schema_str.empty() || !p.regex_pattern.empty() || !p.grammar.empty() ||
        !p.tool_constraint_tools.empty() || p.mirostat != 0)
        return {
            "\"banned_strings\" cannot be combined with constrained output, enforced tool calls or mirostat: "
            "their state has no rewind",
            "banned_strings"};
    return {};
}

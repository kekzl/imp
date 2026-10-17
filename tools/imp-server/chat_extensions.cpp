// imp chat extensions a load or a request cannot honour (rows 115, 118): checked after the model snapshot.

#include "handlers_internal.h"

ExtensionError extension_error(const ChatRequestContext& ctx) {
    const auto& p = ctx.params;
    if (p.logit_lens > 0 && !ctx.snap.logit_lens_ok)
        return {
            "\"logit_lens\" is not available for this model: its residual stream is not the d_model stream "
            "the LM head reads (gated-residual architecture)",
            "logit_lens"};
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

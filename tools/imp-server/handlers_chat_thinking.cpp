// Reply thinking state from the rendered prompt (snapshot_state_and_tokenize_ step).

#include "handlers_internal.h"
#include "output_markers.h"
#include "reasoning_split.h"

#include <algorithm>
#include <string>

void settle_thinking_from_prompt_(ChatRequestContext& ctx) {
    // Thinking-state pipeline: INTENT (explicit request or heuristic) -> RENDER (apply() stamps
    // enable_thinking only when forced/suppressed) -> GROUND TRUTH (below: reconcile against the
    // actual rendered prompt tail, matching vLLM's qwen3_reasoning_parser auto-detection).
    // Detects on decoded text, not token-ID equality: Qwen3.6 ships <think>/</think> as non-special
    // added_tokens, so BPE splits them into 3 pieces instead of one special-token id.
    auto prompt_tail_text = [&](int max_tail_tokens) {
        const int n = static_cast<int>(ctx.snap.tokens.size());
        std::string tail_text;
        for (int i = std::max(0, n - max_tail_tokens); i < n; ++i)
            tail_text += decode_chat_piece(*ctx.snap.tok, ctx.snap.output_aliases, ctx.snap.tokens[i]);
        return tail_text;
    };
    auto prompt_tail_contains = [&](const char* needle, int max_tail_tokens) -> bool {
        return prompt_tail_text(max_tail_tokens).contains(needle);
    };
    // Reconcile: an OPEN <think> prefix (no matching </think> in the tail) turns thinking ON; a
    // pre-closed block (<think>...</think>) turns it OFF even if the heuristic defaulted ON (#934:
    // Qwen3.5-4B mentions enable_thinking but defaults closed, else content is empty).
    // Window 16 (not 8) so both tags of an adjacent closed block land in the same tail scan.
    {
        const bool was_thinking = ctx.snap.enable_thinking;
        if (ctx.params.continue_final) {
            // An assistant prefix decides alone: the reply is reasoning only inside an open <think>
            // (last <think> after the last </think>; a 4096-token tail bounds the decode).
            const std::string text = prompt_tail_text(4096);
            const size_t open = text.rfind("<think>");
            const size_t close = text.rfind("</think>");
            ctx.snap.enable_thinking = open != std::string::npos &&
                                       (close == std::string::npos || open > close);
        } else {
            const bool tail_has_think = prompt_tail_contains("<think>", 16);
            const bool tail_has_close = prompt_tail_contains("</think>", 16);
            ctx.snap.enable_thinking = imp::server::reconcile_thinking_with_prompt_tail(
                ctx.snap.enable_thinking, ctx.params.enable_thinking_set, tail_has_think, tail_has_close);
        }
        // If reconcile flips thinking OFF after the snapshot already removed the provisional <think>
        // stop token (added while ON), restore it - a non-thinking think-model still needs the
        // phantom-"<think>"-turn guard.
        if (was_thinking && !ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
            auto& ids = ctx.snap.stop_token_ids;
            if (std::find(ids.begin(), ids.end(), ctx.snap.think_start_id) == ids.end())
                ids.push_back(ctx.snap.think_start_id);
        }
    }

    // Append <think>\n to trigger reasoning mode (matches llama.cpp behavior).
    // Without this prefix, think-trained models produce degenerate output.
    // Skip if the chat template already added it (Qwen3.x default path).
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0 && !ctx.params.continue_final) {
        if (!prompt_tail_contains("<think>", 8)) {
            ctx.snap.tokens.push_back(ctx.snap.think_start_id);
            // Append newline after <think> - the model expects "\n" before reasoning
            auto nl_ids = ctx.snap.tok->encode("\n");
            ctx.snap.tokens.insert(ctx.snap.tokens.end(), nl_ids.begin(), nl_ids.end());
        }
    }
}

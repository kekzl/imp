#include "prompt_logprobs.h"

#include "utils.h"

#include <algorithm>
#include <string>

using nlohmann::json;

std::string parse_prompt_logprobs(const json& body, bool stream, bool echo, bool req_logprobs,
                                  int top_logprobs, PromptLogprobsRequest& out) {
    out = PromptLogprobsRequest{};
    const std::string range = "[0, " + std::to_string(imp::kMaxPromptLogprobs) + "]";
    if (const auto it = body.find("prompt_logprobs"); it != body.end() && !it->is_null()) {
        if (!it->is_number_integer())
            return "\"prompt_logprobs\" must be an integer in " + range + ", got " + it->type_name();
        const int64_t n = it->get<int64_t>();
        if (n < 0 || n > imp::kMaxPromptLogprobs)
            return "\"prompt_logprobs\" must be an integer in " + range + ", got " + std::to_string(n);
        out.prompt_logprobs = static_cast<int>(n);
    }
    out.echo_logprobs = echo && req_logprobs;
    if (stream && (out.prompt_logprobs >= 0 || out.echo_logprobs))
        return "prompt logprobs (\"prompt_logprobs\", or \"echo\" with \"logprobs\") are not supported "
               "with \"stream\": true";
    out.echo_top = out.echo_logprobs ? std::clamp(top_logprobs, 0, imp::kMaxPromptLogprobs) : -1;
    out.engine_top_n = std::max(out.prompt_logprobs, out.echo_top);
    return "";
}

std::string parse_completions_logprobs(const json& body, bool stream, bool echo, bool& req_logprobs,
                                       int& top_logprobs, PromptLogprobsRequest& plp) {
    // Completions types `logprobs` as an integer (top-N); Chat uses a bool plus `top_logprobs`.
    // Both are accepted so `logprobs: 5` is not a json type error.
    req_logprobs = false;
    top_logprobs = body.value("top_logprobs", 0);
    if (const auto lp = body.find("logprobs"); lp != body.end() && !lp->is_null()) {
        if (lp->is_boolean()) {
            req_logprobs = lp->get<bool>();
        } else if (lp->is_number_integer()) {
            const int n = lp->get<int>();
            if (n > 0) {
                req_logprobs = true;
                top_logprobs = std::max(top_logprobs, n);
            }
        } else {
            return "\"logprobs\" must be an integer (Completions) or boolean";
        }
    }
    top_logprobs = std::clamp(top_logprobs, 0, imp::kMaxPromptLogprobs);
    return parse_prompt_logprobs(body, stream, echo, req_logprobs, top_logprobs, plp);
}

bool prompt_logprobs_complete(const imp::PromptLogprobs& p, size_t n_prompt) {
    if (n_prompt == 0)
        return false;
    const size_t rows = n_prompt - 1;
    const size_t top = static_cast<size_t>(std::max(p.top_n, 0));
    return static_cast<size_t>(p.rows) == rows && p.token_lp.size() == rows && p.rank.size() == rows &&
           p.top_ids.size() == rows * top && p.top_lp.size() == rows * top;
}

json vllm_prompt_logprobs_json(const std::vector<int32_t>& prompt, const imp::PromptLogprobs& p, int limit,
                               const TokenTextFn& text) {
    json arr = json::array();
    if (prompt.empty())
        return arr;
    arr.push_back(nullptr);
    const size_t top = static_cast<size_t>(std::max(p.top_n, 0));
    const size_t use = std::min(top, static_cast<size_t>(std::max(limit, 0)));
    for (size_t pos = 1; pos < prompt.size(); ++pos) {
        const size_t row = pos - 1;  // row p scores prompt[p + 1]
        json entry = json::object();
        for (size_t k = 0; k < use; ++k) {
            const int32_t id = p.top_ids[row * top + k];
            entry[std::to_string(id)] = {{"logprob", p.top_lp[row * top + k]},
                                         {"rank", static_cast<int>(k) + 1},
                                         {"decoded_token", safe_token_json(text(id))}};
        }
        const int32_t tok = prompt[pos];
        entry[std::to_string(tok)] = {{"logprob", p.token_lp[row]},
                                      {"rank", p.rank[row]},
                                      {"decoded_token", safe_token_json(text(tok))}};
        arr.push_back(std::move(entry));
    }
    return arr;
}

namespace {

// Advance `cursor` over `piece` when full_text holds it there; otherwise the offset stays put.
void advance_cursor(size_t& cursor, const std::string& piece, const std::string& full_text) {
    if (!piece.empty() && cursor + piece.size() <= full_text.size() &&
        full_text.compare(cursor, piece.size(), piece) == 0)
        cursor += piece.size();
}

json top_object(const std::vector<std::pair<std::string, float>>& alts) {
    json obj = json::object();
    for (const auto& [t, lp] : alts) {
        const json key = safe_token_json(t);
        if (key.is_string())
            obj[key.get<std::string>()] = lp;
    }
    return obj;
}

}  // namespace

json echo_completions_logprobs_json(const std::vector<int32_t>& prompt, const imp::PromptLogprobs& p,
                                    int limit, const TokenTextFn& text,
                                    const std::vector<imp::TokenLogprobInfo>& out, size_t out_limit,
                                    const std::string& full_text) {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top_logprobs = json::array();
    json text_offset = json::array();
    size_t cursor = 0;
    const size_t top = static_cast<size_t>(std::max(p.top_n, 0));
    const size_t use = std::min(top, static_cast<size_t>(std::max(limit, 0)));

    for (size_t pos = 0; pos < prompt.size(); ++pos) {
        const std::string piece = text(prompt[pos]);
        tokens.push_back(safe_token_json(piece));
        text_offset.push_back(static_cast<int>(cursor));
        if (pos == 0) {
            token_logprobs.push_back(nullptr);
            top_logprobs.push_back(nullptr);
        } else {
            const size_t row = pos - 1;
            token_logprobs.push_back(p.token_lp[row]);
            std::vector<std::pair<std::string, float>> alts;
            for (size_t k = 0; k < use; ++k)
                alts.emplace_back(text(p.top_ids[row * top + k]), p.top_lp[row * top + k]);
            top_logprobs.push_back(top_object(alts));
        }
        advance_cursor(cursor, piece, full_text);
    }
    for (size_t i = 0; i < out.size() && i < out_limit; ++i) {
        const auto& lp = out[i];
        tokens.push_back(safe_token_json(lp.text));
        token_logprobs.push_back(lp.logprob);
        text_offset.push_back(static_cast<int>(cursor));
        std::vector<std::pair<std::string, float>> alts;
        for (const auto& t : lp.top)
            alts.emplace_back(t.text, t.logprob);
        top_logprobs.push_back(top_object(alts));
        advance_cursor(cursor, lp.text, full_text);
    }
    return json{{"tokens", tokens},
                {"token_logprobs", token_logprobs},
                {"top_logprobs", top_logprobs},
                {"text_offset", text_offset}};
}

bool attach_prompt_logprobs(json& choice, const PromptLogprobsRequest& plp, const imp::Request& req,
                            const TokenTextFn& text, size_t out_limit, const std::string& full_text) {
    if (plp.engine_top_n < 0)
        return true;
    if (!prompt_logprobs_complete(req.prompt_lp, req.input_tokens.size()))
        return false;
    if (plp.prompt_logprobs >= 0)
        choice["prompt_logprobs"] = vllm_prompt_logprobs_json(req.input_tokens, req.prompt_lp,
                                                              plp.prompt_logprobs, text);
    if (plp.echo_logprobs)
        choice["logprobs"] = echo_completions_logprobs_json(req.input_tokens, req.prompt_lp, plp.echo_top,
                                                            text, req.output_logprobs, out_limit, full_text);
    return true;
}

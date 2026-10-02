// Parse phase of /v1/completions and POST /infill (#2201): body -> CompletionRequestParams, before
// any model snapshot. Split from handlers_completions.cpp (file-size gate, 800 code LOC).

#include "completions_params.h"

#include "completion_prompt.h"
#include "handlers_internal.h"
#include "utils.h"

#include <algorithm>
#include <string>

bool parse_completions_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                                      bool infill, CompletionRequestParams& out) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return false;
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
        drop_null_fields(body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return false;
    }

    // Validate sampling parameters
    if (!validate_sampling_params(body, res))
        return false;

    // /v1/completions does not implement multi-choice generation. Reject n>1
    // explicitly instead of validating n in [1,4] and then silently returning a
    // single choice (only the chat endpoint honors n, via n_completions).
    if (body.value("n", 1) > 1) {
        send_json_error(res, 400, "invalid_request_error",
                        "n>1 is not supported on /v1/completions; request one completion per call");
        return false;
    }

    // best_of: imp has no candidate-scoring path, so this field is refused (400) rather than
    // silently ignored (#1598) - "best of 8" and "the first one" are different answers.
    if (body.contains("best_of") && !body["best_of"].is_null()) {
        if (!body["best_of"].is_number_integer()) {
            send_json_error(res, 400, "invalid_request_error", "\"best_of\" must be an integer");
            return false;
        }
        if (body["best_of"].get<int>() > 1) {
            send_json_error(res, 400, "invalid_request_error",
                            "best_of>1 is not supported; imp generates no candidate set to choose from");
            return false;
        }
    }

    // Extract prompt
    if (infill) {
        if (const std::string err = parse_infill_fields(body, out.fim); !err.empty()) {
            send_json_error(res, 400, "invalid_request_error", err);
            return false;
        }
        out.prompt = out.fim.input.prefix + out.fim.input.prompt;  // what `echo` returns
    } else {
        if (const std::string err = parse_completion_prompt(body, out.prompt, out.prompt_ids); !err.empty()) {
            send_json_error(res, 400, "invalid_request_error", err, "prompt");
            return false;
        }
        if (const std::string err = parse_completion_suffix(body, out.prompt, !out.prompt_ids.empty(),
                                                            out.fim);
            !err.empty()) {
            send_json_error(res, 400, "invalid_request_error", err, "suffix");
            return false;
        }
    }

    // Extract parameters
    parse_sampling_fields(body, state.default_think_budget, out);
    // max_tokens only: max_completion_tokens is a chat/completions field (parse_max_tokens_field).
    out.max_tokens = body.value("max_tokens", state.default_max_tokens);
    out.echo = body.value("echo", false);

    if (const std::string err = parse_completions_logprobs(body, out.stream, out.echo, out.req_logprobs,
                                                           out.top_logprobs, out.plp);
        !err.empty()) {
        send_json_error(res, 400, "invalid_request_error", err);
        return false;
    }

    // Parse stop sequences (same 16-entry cap as the chat parser).
    if (parse_stop_field(body, 16, out.stop_sequences)) {
        IMP_LOG_WARN("request sent %zu stop sequences; keeping the first 16", body["stop"].size());
    }
    for (const auto& s : out.stop_sequences)
        out.max_stop_len = std::max(out.max_stop_len, s.size());

    if (const std::string err = parse_logit_bias(body, state.max_logit_bias, out.logit_bias); !err.empty()) {
        send_json_error(res, 400, "invalid_request_error", err, "logit_bias");
        return false;
    }

    // Parse stream_options for include_usage
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        out.include_usage = body["stream_options"].value("include_usage", false);
    }

    // Log request received
    out.req_id = make_completion_id(state);
    {
        // Trace join, same contract as chat/completions (see
        // parse_chat_request_params): client id echoed, server id otherwise.
        const std::string cid = sanitize_for_echo(req.get_header_value("X-Request-Id"), 128);
        res.set_header("X-Request-Id", cid.empty() ? out.req_id : cid);
    }
    IMP_LOG_INFO("[%s] completions: prompt_len=%zu stream=%s max_tokens=%d temp=%.2f", out.req_id.c_str(),
                 out.prompt.size(), out.stream ? "true" : "false", out.max_tokens, out.temperature);

    // Validate model field (required per OpenAI spec)
    out.requested_model = body.value("model", "");
    if (out.requested_model.empty() && !infill) {  // llama.cpp /infill clients send no model
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }
    // Same contract as /v1/chat/completions: bool, or {"mtp_k": N}.
    {
        const SpecFieldParse sp = parse_spec_field_(body, state.armed_mtp_k.load(std::memory_order_relaxed));
        if (!sp.ok) {
            send_json_error(res, 400, "invalid_request_error", sp.error);
            return false;
        }
        out.spec_override = sp.spec_override;
        out.spec_mtp_k = sp.mtp_k;
    }
    // Predicted Outputs (string-content form) on the completions route: the
    // prediction only seeds the n-gram draft corpus, output is unchanged.
    if (body.contains("prediction") && body["prediction"].is_object()) {
        const auto& pred = body["prediction"];
        if (pred.value("type", "content") == "content" && pred.contains("content") &&
            pred["content"].is_string()) {
            out.has_prediction = true;
            out.prediction_text = pred["content"].get<std::string>();
        }
    }

    return true;
}

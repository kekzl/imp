#pragma once

// Body-parsed input of /v1/completions and POST /infill: everything read from the request before
// the model snapshot. Mirrors ChatRequestParams / parse_chat_request_params for the chat route.

#include "fim_request.h"
#include "handlers.h"
#include "prompt_logprobs.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct CompletionRequestParams {
    std::string prompt;
    std::vector<int32_t> prompt_ids;
    FimRequest fim;
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f;
    float repetition_penalty = 1.05f, frequency_penalty = 0.0f, presence_penalty = 0.0f;
    float dry_multiplier = 0.0f, dry_base = 1.75f, mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    float think_budget = 0.0f;
    int top_k = 40, max_tokens = 0, seed = -1, priority = 0, repeat_last_n = 0;
    int dry_allowed_length = 2, dry_penalty_last_n = 0, mirostat = 0;
    bool stream = false, echo = false, ignore_eos = false, cache_prompt = false;
    bool req_logprobs = false;
    int top_logprobs = 0;
    PromptLogprobsRequest plp;
    std::vector<std::string> stop_sequences;
    size_t max_stop_len = 0;
    std::vector<std::pair<int32_t, float>> logit_bias;
    bool include_usage = false;
    int spec_override = -1, spec_mtp_k = -1;
    bool has_prediction = false;  // string-content "prediction"; tokenized after the model snapshot
    std::string prediction_text;
    std::string req_id;
    std::string requested_model;
};

// Returns false with the 4xx already on `res`.
bool parse_completions_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                                      bool infill, CompletionRequestParams& out);

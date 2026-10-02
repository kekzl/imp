#pragma once

// Body-parsed input of /v1/completions and POST /infill: everything read from the request before
// the model snapshot. Mirrors ChatRequestParams / parse_chat_request_params for the chat route.

#include "fim_request.h"
#include "handlers.h"
#include "prompt_logprobs.h"
#include "sampling_fields.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct CompletionRequestParams : SamplingFields {
    std::string prompt;
    std::vector<int32_t> prompt_ids;
    FimRequest fim;
    int max_tokens = 0;
    bool echo = false;
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

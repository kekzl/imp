// POST /v1/decide and POST /v1/score (#2198): closed-choice scoring without decoding.
// One prefill per item; softmax over candidate-token logits at the last prompt position.
// Same engine primitive as /v1/rerank (Request::score_token_ids, no sampling).

#include "candidate_tokens.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "runtime/snapshot_boundary.h"
#include "score_waves.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr int kMaxOptions = 16;  // letters A..P
constexpr int kMaxScoreCandidates = 256;
constexpr const char* kDefaultSystem = "Answer only with the letter of the correct option.";

using imp::server::ScoreMode;

struct ScoreJob {
    std::vector<int32_t> tokens;
    std::vector<int32_t> ids;
    int shared_prefix = 0;  // tokens shared with every sibling item (hybrid snapshot hint)
};

struct ScoreOut {
    std::vector<float> logits;
    std::vector<double> probs;
    int argmax = 0;
    int prompt_tokens = 0;
    int cached_tokens = 0;
};

void bad_request(httplib::Response& res, const std::string& msg, const char* param = nullptr) {
    send_json_error(res, 400, "invalid_request_error", msg, param);
}

// auto -> serial.
bool parse_mode(const json& body, httplib::Response& res, ScoreMode& mode) {
    mode = ScoreMode::Serial;
    if (!body.contains("mode"))
        return true;
    if (!body["mode"].is_string()) {
        bad_request(res, "\"mode\" must be one of auto, serial, direct, shared", "mode");
        return false;
    }
    const std::string m = body["mode"].get<std::string>();
    if (m == "auto" || m == "serial")
        return true;
    if (m == "direct" || m == "shared") {
        mode = m == "direct" ? ScoreMode::Direct : ScoreMode::Shared;
        return true;
    }
    bad_request(res, "\"mode\" must be one of auto, serial, direct, shared (got \"" + m + "\")", "mode");
    return false;
}

const char* mode_name(ScoreMode m) {
    if (m == ScoreMode::Direct)
        return "direct";
    return m == ScoreMode::Shared ? "shared" : "serial";
}

// Thinking off for any template that can open a reasoning block: the score position must be
// the first answer token. Harmony: suppress ends the prompt on the final channel.
bool want_suppress_thinking(const ServerState& state) {
    return state.is_think_model || (state.have_template && state.chat_tpl.mentions_thinking()) ||
           (state.have_template && state.chat_tpl.family() == imp::ChatTemplateFamily::HARMONY);
}

std::vector<double> softmax(const std::vector<float>& logits) {
    const float m = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        p[i] = std::exp(static_cast<double>(logits[i]) - static_cast<double>(m));
        sum += p[i];
    }
    for (double& v : p)
        v /= sum;
    return p;
}

std::shared_ptr<ServerRequest> make_score_request(const ScoreJob& job, bool direct) {
    auto r = std::make_shared<imp::Request>();
    r->input_tokens = job.tokens;
    r->max_tokens = 1;
    r->temperature = 0.0f;
    r->stream = false;
    r->score_token_ids = job.ids;
    r->bypass_prefix_cache = direct;
    r->snapshot_hint_tokens = direct ? 0 : job.shared_prefix;
    r->status = imp::RequestStatus::PENDING;
    auto sr = std::make_shared<ServerRequest>();
    sr->request = std::move(r);
    return sr;
}

// One queue insertion per wave: the worker admits the whole wave in one iteration, so the
// ragged batch composition (and its numerics) does not depend on submission timing.
bool submit_all(ServerState& state, const std::vector<std::shared_ptr<ServerRequest>>& srs,
                httplib::Response& res) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);
    if (!state.batching || !state.batching->is_running()) {
        send_json_error(res, 503, "server_error", "Scoring requires the batching worker");
        return false;
    }
    state.batching->submit_all(srs);
    return true;
}

// Waits for the finish event and reads the candidate logits into `out`.
bool collect_one(ServerRequest& sr, size_t n_ids, httplib::Response& res, ScoreOut& out) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    bool finished = false;
    while (!finished) {
        std::unique_lock<std::mutex> ql(sr.token_mutex);
        if (!sr.token_cv.wait_until(ql, deadline, [&] { return !sr.token_queue.empty(); })) {
            sr.cancelled = true;
            send_json_error(res, 503, "server_error", "Scoring request timed out");
            return false;
        }
        while (!sr.token_queue.empty()) {
            if (sr.token_queue.front().is_last)
                finished = true;
            sr.token_queue.pop_front();
        }
    }
    const auto& rr = sr.request;
    if (rr->status != imp::RequestStatus::FINISHED || rr->score_out.size() != n_ids) {
        send_json_error(res, 500, "server_error", "Scoring produced no logits");
        return false;
    }
    out.logits = rr->score_out;
    out.probs = softmax(out.logits);
    out.argmax = static_cast<int>(std::max_element(out.probs.begin(), out.probs.end()) - out.probs.begin());
    out.prompt_tokens = static_cast<int>(rr->input_tokens.size());
    out.cached_tokens = rr->cached_tokens;
    return true;
}

// One wave: every job submitted at once, then all collected (score_waves.h).
bool run_wave(ServerState& state, const std::vector<ScoreJob>& jobs, const std::vector<size_t>& wave,
              ScoreMode mode, httplib::Response& res, std::vector<ScoreOut>& out) {
    std::vector<std::shared_ptr<ServerRequest>> submitted;
    submitted.reserve(wave.size());
    for (const size_t i : wave)
        submitted.push_back(make_score_request(jobs[i], mode == ScoreMode::Direct));
    if (!submit_all(state, submitted, res))
        return false;
    for (size_t k = 0; k < wave.size(); k++) {
        if (!collect_one(*submitted[k], jobs[wave[k]].ids.size(), res, out[wave[k]])) {
            for (size_t j = k + 1; j < submitted.size(); j++)
                submitted[j]->cancelled = true;
            return false;
        }
    }
    return true;
}

bool run_jobs(ServerState& state, const std::vector<ScoreJob>& jobs, ScoreMode mode, httplib::Response& res,
              std::vector<ScoreOut>& out) {
    out.assign(jobs.size(), {});
    for (const auto& wave : imp::server::score_waves(mode, jobs.size()))
        if (!run_wave(state, jobs, wave, mode, res, out))
            return false;
    return true;
}

bool parse_body(const httplib::Request& req, httplib::Response& res, json& body) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return false;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        bad_request(res, std::string("Invalid JSON: ") + e.what());
        return false;
    }
    if (!body.is_object()) {
        bad_request(res, "request body must be a JSON object");
        return false;
    }
    drop_null_fields(body);
    return true;
}

bool check_context(ServerState& state, int n_tokens, httplib::Response& res) {
    if (state.max_input_tokens > 0 && n_tokens > state.max_input_tokens) {
        send_json_error(res, 400, "invalid_request_error",
                        "Prompt exceeds max input tokens (" + std::to_string(n_tokens) + " > " +
                            std::to_string(state.max_input_tokens) + ")",
                        nullptr, "context_length_exceeded");
        return false;
    }
    if (state.max_seq_len > 0 && n_tokens > state.max_seq_len) {
        send_json_error(res, 400, "invalid_request_error",
                        "Prompt exceeds the model context (" + std::to_string(n_tokens) + " tokens, " +
                            std::to_string(state.max_seq_len) + " max)",
                        nullptr, "context_length_exceeded");
        return false;
    }
    return true;
}

void record_metrics(ServerState& state, std::chrono::steady_clock::time_point t0, int prompt_tokens) {
    state.metrics.last_request_duration_ms.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    state.metrics.tokens_prompt_total += prompt_tokens;
}

struct DecideItem {
    json id;
    std::string criterion;
    std::vector<std::string> options;
};

}  // namespace

void handle_decide(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    if (!parse_body(req, res, body))
        return;
    ScoreMode mode;
    if (!parse_mode(body, res, mode))
        return;
    if (!body.contains("evidence") || !body["evidence"].is_string()) {
        bad_request(res, "\"evidence\" (string) is required", "evidence");
        return;
    }
    const std::string evidence = body["evidence"].get<std::string>();
    std::string system = kDefaultSystem;
    if (body.contains("system")) {
        if (!body["system"].is_string()) {
            bad_request(res, "\"system\" must be a string", "system");
            return;
        }
        system = body["system"].get<std::string>();
    }
    if (!body.contains("items") || !body["items"].is_array() || body["items"].empty()) {
        bad_request(res, "\"items\" (non-empty array) is required", "items");
        return;
    }
    if (state.max_batch_items > 0 && static_cast<int>(body["items"].size()) > state.max_batch_items) {
        bad_request(res,
                    "\"items\" has " + std::to_string(body["items"].size()) +
                        " entries, above the server limit of " + std::to_string(state.max_batch_items) +
                        " (--max-batch-items)",
                    "items");
        return;
    }
    std::vector<DecideItem> items;
    for (size_t i = 0; i < body["items"].size(); i++) {
        const json& it = body["items"][i];
        const std::string where = "items[" + std::to_string(i) + "]";
        if (!it.is_object()) {
            bad_request(res, where + " must be an object", "items");
            return;
        }
        DecideItem d;
        d.id = it.contains("id") ? it["id"] : json(static_cast<int>(i));
        if (!it.contains("criterion") || !it["criterion"].is_string()) {
            bad_request(res, where + ".criterion (string) is required", "items");
            return;
        }
        d.criterion = it["criterion"].get<std::string>();
        if (!it.contains("options") || !it["options"].is_array() || it["options"].size() < 2) {
            bad_request(res,
                        where + ".options must be an array of 2 to " + std::to_string(kMaxOptions) +
                            " strings",
                        "items");
            return;
        }
        if (it["options"].size() > static_cast<size_t>(kMaxOptions)) {
            bad_request(res,
                        where + ".options has " + std::to_string(it["options"].size()) +
                            " entries, the maximum is " + std::to_string(kMaxOptions) + " (letters A to P)",
                        "items");
            return;
        }
        for (const auto& o : it["options"]) {
            if (!o.is_string()) {
                bad_request(res, where + ".options must contain only strings", "items");
                return;
            }
            d.options.push_back(o.get<std::string>());
        }
        items.push_back(std::move(d));
    }

    std::string requested_model = body.value("model", std::string());
    std::vector<ScoreJob> jobs;
    std::vector<std::vector<std::string>> letters_per_item;
    int total_prompt_tokens = 0;
    {
        std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
        if (!lock.owns_lock()) {
            send_json_error(res, 503, "server_error",
                            "Server is busy processing another request. Please retry.");
            return;
        }
        if (requested_model.empty())
            requested_model = state.model_name;
        if (!ensure_model_loaded(state, requested_model, res))
            return;
        if (!state.have_template || state.chat_tpl.is_raw() || !state.tok) {
            bad_request(res, "The loaded model (" + state.model_name +
                                 ") has no chat template; /v1/decide renders its prompt through one");
            return;
        }
        const bool suppress = want_suppress_thinking(state);
        for (const auto& d : items) {
            // Key order is the prefix-cache contract: evidence first, shared by every item.
            nlohmann::ordered_json user;
            user["evidence"] = evidence;
            user["criterion"] = d.criterion;
            nlohmann::ordered_json opts = nlohmann::ordered_json::object();
            std::vector<std::string> letters;
            for (size_t k = 0; k < d.options.size(); k++) {
                letters.push_back(std::string(1, static_cast<char>('A' + k)));
                opts[letters.back()] = d.options[k];
            }
            user["options"] = opts;
            const std::vector<imp::ChatMessage> msgs = {{"system", system}, {"user", user.dump()}};
            ScoreJob job;
            job.tokens = state.chat_tpl.apply(*state.tok, msgs, suppress);
            if (job.tokens.empty()) {
                send_json_error(res, 500, "server_error", "Tokenize failed");
                return;
            }
            if (!check_context(state, static_cast<int>(job.tokens.size()), res))
                return;
            auto resolved = imp::server::resolve_candidate_tokens(
                *state.tok, imp::server::prompt_tail_text(*state.tok, job.tokens), letters);
            if (!resolved.error.empty()) {
                bad_request(res, "option letter: " + resolved.error, "items");
                return;
            }
            job.ids = std::move(resolved.ids);
            total_prompt_tokens += static_cast<int>(job.tokens.size());
            jobs.push_back(std::move(job));
            letters_per_item.push_back(std::move(letters));
        }
        state.metrics.requests_total++;
    }
    // Hybrid models restore only at a snapshotted block: without a save at the shared evidence
    // prefix, item 2+ diverge before item 1's prompt-end snapshot and reuse nothing.
    if (mode != ScoreMode::Direct && jobs.size() >= 2) {
        std::vector<std::vector<int32_t>> seqs;
        seqs.reserve(jobs.size());
        for (const auto& j : jobs)
            seqs.push_back(j.tokens);
        const int shared = imp::common_prefix_tokens(seqs);
        for (auto& j : jobs)
            j.shared_prefix = shared;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ScoreOut> outs;
    if (!run_jobs(state, jobs, mode, res, outs))
        return;
    record_metrics(state, t0, total_prompt_tokens);

    json out_items = json::array();
    int total_cached = 0;
    for (size_t i = 0; i < items.size(); i++) {
        const auto& o = outs[i];
        json probs = json::object();
        for (size_t k = 0; k < o.probs.size(); k++)
            probs[letters_per_item[i][k]] = o.probs[k];
        total_cached += o.cached_tokens;
        out_items.push_back({{"id", items[i].id},
                             {"probs", probs},
                             {"argmax", letters_per_item[i][static_cast<size_t>(o.argmax)]},
                             {"argmax_index", o.argmax},
                             {"prompt_tokens", o.prompt_tokens},
                             {"cached_tokens", o.cached_tokens}});
    }
    json response = {{"object", "decide"},
                     {"model", requested_model},
                     {"mode_used", mode_name(mode)},
                     {"items", out_items},
                     {"usage",
                      {{"prompt_tokens", total_prompt_tokens},
                       {"cached_tokens", total_cached},
                       {"total_tokens", total_prompt_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}

void handle_score(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    json body;
    if (!parse_body(req, res, body))
        return;
    ScoreMode mode;
    if (!parse_mode(body, res, mode))
        return;
    const bool has_prompt = body.contains("prompt");
    const bool has_messages = body.contains("messages");
    if (has_prompt == has_messages) {
        bad_request(res, "exactly one of \"prompt\" (string) or \"messages\" (array) is required");
        return;
    }
    if (has_prompt && (!body["prompt"].is_string() || body["prompt"].get<std::string>().empty())) {
        bad_request(res, "\"prompt\" must be a non-empty string", "prompt");
        return;
    }
    std::vector<imp::ChatMessage> msgs;
    if (has_messages) {
        if (!body["messages"].is_array() || body["messages"].empty()) {
            bad_request(res, "\"messages\" must be a non-empty array", "messages");
            return;
        }
        for (const auto& m : body["messages"]) {
            if (!m.is_object() || !m.contains("role") || !m["role"].is_string() || !m.contains("content") ||
                !m["content"].is_string()) {
                bad_request(res, "each message must be {\"role\": string, \"content\": string}", "messages");
                return;
            }
            msgs.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
        }
    }
    if (!body.contains("candidates") || !body["candidates"].is_array() || body["candidates"].size() < 2) {
        bad_request(res, "\"candidates\" must be an array of 2 or more token strings or token ids",
                    "candidates");
        return;
    }
    if (body["candidates"].size() > static_cast<size_t>(kMaxScoreCandidates)) {
        bad_request(res,
                    "\"candidates\" has " + std::to_string(body["candidates"].size()) +
                        " entries, the maximum is " + std::to_string(kMaxScoreCandidates),
                    "candidates");
        return;
    }
    for (const auto& c : body["candidates"]) {
        if (!c.is_string() && !c.is_number_integer()) {
            bad_request(res, "each candidate must be a token string or an integer token id", "candidates");
            return;
        }
        if (c.is_string() && c.get<std::string>().empty()) {
            bad_request(res, "a candidate string must not be empty", "candidates");
            return;
        }
    }
    if (has_prompt && !prompt_within_input_budget(res, body["prompt"].get<std::string>().size(),
                                                  state.max_input_tokens, "prompt"))
        return;

    std::string requested_model = body.value("model", std::string());
    ScoreJob job;
    {
        std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
        if (!lock.owns_lock()) {
            send_json_error(res, 503, "server_error",
                            "Server is busy processing another request. Please retry.");
            return;
        }
        if (requested_model.empty())
            requested_model = state.model_name;
        if (!ensure_model_loaded(state, requested_model, res))
            return;
        if (!state.tok) {
            send_json_error(res, 500, "server_error", "No tokenizer loaded");
            return;
        }
        const imp::Tokenizer& tok = *state.tok;
        if (has_prompt) {
            // Raw prompt, tokenized like /v1/completions (BOS when the tokenizer asks for it).
            job.tokens = tok.encode(body["prompt"].get<std::string>());
            if (tok.add_bos() && tok.bos_id() >= 0 &&
                (job.tokens.empty() || job.tokens.front() != tok.bos_id()))
                job.tokens.insert(job.tokens.begin(), tok.bos_id());
        } else {
            if (!state.have_template || state.chat_tpl.is_raw()) {
                bad_request(res,
                            "The loaded model has no chat template; send \"prompt\" instead of \"messages\"",
                            "messages");
                return;
            }
            job.tokens = state.chat_tpl.apply(tok, msgs, want_suppress_thinking(state));
        }
        if (job.tokens.empty()) {
            send_json_error(res, 500, "server_error", "Tokenize failed");
            return;
        }
        if (!check_context(state, static_cast<int>(job.tokens.size()), res))
            return;
        // Strings go through the boundary guard, integer ids are taken as given (range + distinct).
        const std::string tail = imp::server::prompt_tail_text(tok, job.tokens);
        for (const auto& c : body["candidates"]) {
            if (c.is_string()) {
                auto r = imp::server::resolve_candidate_tokens(tok, tail, {c.get<std::string>()});
                if (!r.error.empty()) {
                    bad_request(res, r.error, "candidates");
                    return;
                }
                job.ids.push_back(r.ids[0]);
            } else {
                const int64_t v = c.get<int64_t>();
                job.ids.push_back(v < 0 || v > INT32_MAX ? -1 : static_cast<int32_t>(v));
            }
        }
        if (const std::string err = imp::server::validate_candidate_ids(tok, job.ids); !err.empty()) {
            bad_request(res, err, "candidates");
            return;
        }
        state.metrics.requests_total++;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ScoreOut> outs;
    if (!run_jobs(state, {job}, mode, res, outs))
        return;
    record_metrics(state, t0, static_cast<int>(job.tokens.size()));

    const auto& o = outs[0];
    json cands = json::array();
    for (size_t k = 0; k < job.ids.size(); k++)
        cands.push_back({{"candidate", body["candidates"][k]},
                         {"token_id", job.ids[k]},
                         {"logit", o.logits[k]},
                         {"prob", o.probs[k]}});
    json response = {{"object", "score"},
                     {"model", requested_model},
                     {"mode_used", mode_name(mode)},
                     {"candidates", cands},
                     {"argmax_index", o.argmax},
                     {"prompt_tokens", o.prompt_tokens},
                     {"cached_tokens", o.cached_tokens},
                     {"usage",
                      {{"prompt_tokens", o.prompt_tokens},
                       {"cached_tokens", o.cached_tokens},
                       {"total_tokens", o.prompt_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}

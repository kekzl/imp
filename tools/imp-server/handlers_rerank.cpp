// =============================================================================
// handle_rerank — POST /v1/rerank (Cohere / Jina / vLLM-compatible shape)
//
// RAG agents use an embedding model to retrieve and a RERANKER to order what
// came back, and imp shipped only the first half (roadmap gap 9). The quality
// bar the gap sets is that query and document must be scored JOINTLY — a score
// recomputed from two independent embeddings is an endpoint with the right name
// and the wrong answer.
//
// WHAT THIS IS NOT: a BERT sequence-classification head. The gap assumed one,
// because that is what a reranker was when it was written. The current
// generation (Qwen3-Reranker, bge-reranker-v2-gemma) are CAUSAL LMs that read
// query and document in one forward and answer a yes/no question, and their
// relevance score is the softmax over those two logits. That is a cross-encoder
// by the definition that matters — one joint forward, no independent
// embeddings — and it runs on imp's existing decoder stack (NVFP4 GEMMs, FA2,
// paged KV, prefix cache) instead of a second architecture family bolted on for
// a 22M-parameter model.
//
// The prompt below is the Qwen3-Reranker format verbatim; the score is
// softmax(logit_yes, logit_no)[yes], which is what the reference implementation
// computes. Deviating from either would make imp's numbers incomparable to
// everyone else's.
//
// The shared "system + instruct + query" prefix is identical across every
// document in one call, so the prefix cache turns an N-document rerank into one
// full prefill plus N document tails.
// =============================================================================

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace {

// Qwen3-Reranker prompt, verbatim from the model card.
constexpr const char* kPrefix =
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
constexpr const char* kDefaultInstruct =
    "Given a web search query, retrieve relevant passages that answer the query";

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// The prompt is model-specific, so serving it from a model that was not trained
// on it would return confident nonsense. Name-gating is crude but honest, and
// it fails loudly instead of silently scoring garbage.
bool looks_like_reranker(const std::string& name) {
    const std::string l = lower(name);
    return l.find("rerank") != std::string::npos;
}

// Tokenize `text` and require it to be exactly one token.
bool single_token(ServerState& state, const char* text, int32_t& out) {
    int32_t buf[8];
    int n = 0;
    if (imp_tokenize(state.model, text, buf, &n, 8) != IMP_SUCCESS || n != 1)
        return false;
    out = buf[0];
    return true;
}

}  // namespace

void handle_rerank(const httplib::Request& req, httplib::Response& res, ServerState& state) {
    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return;
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return;
    }

    if (!body.contains("query") || !body["query"].is_string()) {
        send_json_error(res, 400, "invalid_request_error", "\"query\" (string) is required");
        return;
    }
    const std::string query = body["query"].get<std::string>();

    // Cohere and Jina both spell it "documents"; vLLM also accepts "texts".
    const json* docs_field = nullptr;
    if (body.contains("documents") && body["documents"].is_array())
        docs_field = &body["documents"];
    else if (body.contains("texts") && body["texts"].is_array())
        docs_field = &body["texts"];
    if (!docs_field || docs_field->empty()) {
        send_json_error(res, 400, "invalid_request_error",
                        "\"documents\" (non-empty array of strings) is required");
        return;
    }
    // One request, one rate-limit unit, N scoring passes (#1616).
    if (state.max_batch_items > 0 && static_cast<int>(docs_field->size()) > state.max_batch_items) {
        send_json_error(res, 400, "invalid_request_error",
                        "\"documents\" has " + std::to_string(docs_field->size()) +
                            " entries, above the server limit of " + std::to_string(state.max_batch_items) +
                            " (--max-batch-items)");
        return;
    }
    std::vector<std::string> documents;
    for (const auto& d : *docs_field) {
        if (d.is_string()) {
            documents.push_back(d.get<std::string>());
        } else if (d.is_object() && d.contains("text") && d["text"].is_string()) {
            documents.push_back(d["text"].get<std::string>());  // Cohere's object form
        } else {
            send_json_error(res, 400, "invalid_request_error",
                            "each document must be a string or {\"text\": string}");
            return;
        }
    }
    const std::string instruct = body.value("instruction",
                                            body.value("instruct", std::string(kDefaultInstruct)));
    const bool return_documents = body.value("return_documents", false);
    int top_n = body.value("top_n", static_cast<int>(documents.size()));
    if (top_n <= 0 || top_n > static_cast<int>(documents.size()))
        top_n = static_cast<int>(documents.size());

    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(1));
    if (!lock.owns_lock()) {
        send_json_error(res, 503, "server_error", "Server is busy processing another request. Please retry.");
        return;
    }

    std::string requested_model = body.value("model", std::string());
    if (requested_model.empty())
        requested_model = state.model_name;
    if (!ensure_model_loaded(state, requested_model, res))
        return;

    if (!looks_like_reranker(state.model_name)) {
        send_json_error(res, 400, "invalid_request_error",
                        "The loaded model (" + state.model_name +
                            ") is not a reranker. /v1/rerank scores a query and a document "
                            "jointly with a cross-encoder reranker (e.g. Qwen3-Reranker); "
                            "serving it from a general model would return meaningless scores.");
        return;
    }
    if (!state.batching || !state.batching->is_running()) {
        send_json_error(res, 503, "server_error", "Reranking requires the batching worker");
        return;
    }

    int32_t yes_id = -1, no_id = -1;
    if (!single_token(state, "yes", yes_id) || !single_token(state, "no", no_id)) {
        send_json_error(res, 500, "server_error",
                        "This tokenizer does not spell \"yes\"/\"no\" as single tokens, which the "
                        "reranker score is defined over");
        return;
    }

    state.metrics.requests_total++;
    auto t0 = std::chrono::steady_clock::now();

    // Submit every (query, document) pair while the lock is held, so submission
    // order is queue order and the shared prefix lands in the cache once.
    std::vector<std::shared_ptr<ServerRequest>> submitted;
    int total_prompt_tokens = 0;
    const int tok_cap = std::max(state.max_seq_len, 262144);
    for (const auto& doc : documents) {
        const std::string prompt = std::string(kPrefix) + "<Instruct>: " + instruct + "\n<Query>: " + query +
                                   "\n<Document>: " + doc + kSuffix;
        std::vector<int32_t> tokens(tok_cap);
        int n_tokens = 0;
        if (imp_tokenize(state.model, prompt.c_str(), tokens.data(), &n_tokens, tok_cap) != IMP_SUCCESS ||
            n_tokens == 0) {
            send_json_error(res, 500, "server_error", "Tokenize failed");
            return;
        }
        if (state.max_seq_len > 0 && n_tokens > state.max_seq_len) {
            send_json_error(res, 400, "invalid_request_error",
                            "query + document exceeds the model context (" + std::to_string(n_tokens) +
                                " tokens > " + std::to_string(state.max_seq_len) + " max)");
            return;
        }
        tokens.resize(n_tokens);
        total_prompt_tokens += n_tokens;

        auto r = std::make_shared<imp::Request>();
        r->input_tokens = std::move(tokens);
        r->max_tokens = 1;
        r->temperature = 0.0f;
        r->stream = false;
        r->score_token_ids = {yes_id, no_id};
        r->status = imp::RequestStatus::PENDING;
        auto sr = std::make_shared<ServerRequest>();
        sr->request = r;
        state.batching->submit(sr);
        submitted.push_back(std::move(sr));
    }
    lock.unlock();

    struct Scored {
        int index;
        double score;
    };
    std::vector<Scored> scored;
    scored.reserve(submitted.size());
    for (size_t i = 0; i < submitted.size(); i++) {
        auto& sr = submitted[i];
        bool finished = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
        while (!finished) {
            std::unique_lock<std::mutex> ql(sr->token_mutex);
            if (!sr->token_cv.wait_until(ql, deadline, [&] { return !sr->token_queue.empty(); })) {
                sr->cancelled = true;
                send_json_error(res, 503, "server_error", "Rerank request timed out");
                return;
            }
            while (!sr->token_queue.empty()) {
                if (sr->token_queue.front().is_last)
                    finished = true;
                sr->token_queue.pop_front();
            }
        }
        auto& rr = sr->request;
        if (rr->status != imp::RequestStatus::FINISHED || rr->score_out.size() != 2) {
            send_json_error(res, 500, "server_error", "Rerank scoring produced no logits");
            return;
        }
        // softmax over exactly the two class logits, shifted for stability.
        const double ly = rr->score_out[0], ln = rr->score_out[1];
        const double m = std::max(ly, ln);
        const double ey = std::exp(ly - m), en = std::exp(ln - m);
        scored.push_back({static_cast<int>(i), ey / (ey + en)});
    }

    std::stable_sort(scored.begin(), scored.end(),
                     [](const Scored& a, const Scored& b) { return a.score > b.score; });

    json results = json::array();
    for (int i = 0; i < top_n; i++) {
        json entry = {{"index", scored[static_cast<size_t>(i)].index},
                      {"relevance_score", scored[static_cast<size_t>(i)].score}};
        if (return_documents)
            entry["document"] = {
                {"text", documents[static_cast<size_t>(scored[static_cast<size_t>(i)].index)]}};
        results.push_back(std::move(entry));
    }

    auto t1 = std::chrono::steady_clock::now();
    state.metrics.last_request_duration_ms.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    state.metrics.tokens_prompt_total += total_prompt_tokens;

    json response = {{"object", "rerank"},
                     {"model", requested_model},
                     {"results", results},
                     {"usage",
                      {{"prompt_tokens", total_prompt_tokens}, {"total_tokens", total_prompt_tokens}}}};
    res.set_content(dump_safe(response), "application/json");
}

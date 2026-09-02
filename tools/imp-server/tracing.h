// Per-request tracing for imp-server: W3C trace-context propagation and an
// OpenTelemetry (OTLP/HTTP, JSON encoding) span exporter.
//
// One request becomes one SERVER span (the endpoint) with up to three
// INTERNAL children on the same trace: `queue` (admission wait, from the
// engine's queue_ms), `prefill` (queue end to first token) and `decode`
// (first token to the last). The client's `traceparent` header, when sent,
// supplies the trace id and the parent span, so the hop lands inside the
// agent framework's own trace; otherwise a fresh trace id is minted. Spans
// are exported by a background thread in batches; export failures are
// logged once and dropped - the serving path never waits on the collector.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Parsed W3C traceparent: `00-<32 hex trace id>-<16 hex span id>-<2 hex flags>`.
struct TraceContext {
    std::string trace_id;   // 32 lowercase hex chars
    std::string parent_id;  // 16 lowercase hex chars (the caller's span)
    bool sampled = true;
};

// Returns false (and leaves ctx untouched) on anything but a well-formed
// version-00 header with non-zero ids.
bool parse_traceparent(const std::string& header, TraceContext& ctx);

// What a request hands to the tracer at accounting time. Times are
// system_clock (unix epoch) so the spans line up with other services.
struct RequestSpan {
    std::string endpoint;  // span name: "chat.completions", "messages", ...
    std::string req_id;    // server completion / message id
    std::string client_request_id;
    std::string traceparent;  // raw incoming header, "" = none
    std::string model;
    std::chrono::system_clock::time_point t_start;
    double latency_ms = 0.0;  // t_start -> response complete
    double queue_ms = -1.0;   // admission wait (-1 = unknown)
    double ttft_ms = -1.0;    // t_start -> first token (-1 = unknown / non-stream)
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int cached_tokens = 0;
    std::string finish_reason;
    bool stream = false;
    int http_status = 200;
};

// An exported span (root or child), already assigned ids.
struct OtlpSpan {
    std::string name;
    std::string trace_id, span_id, parent_id;
    int kind = 1;  // OTLP SpanKind: 1 = INTERNAL, 2 = SERVER
    uint64_t start_ns = 0, end_ns = 0;
    std::vector<std::pair<std::string, std::string>> str_attrs;
    std::vector<std::pair<std::string, int64_t>> int_attrs;
    std::vector<std::pair<std::string, bool>> bool_attrs;
};

std::string random_hex_id(int n_bytes);  // 8 -> span id, 16 -> trace id

// The root span plus its queue/prefill/decode children. Uses the request's
// traceparent when it parses, otherwise mints a trace id. Deterministic
// given the ids it draws (tests inject them through `root_span_id`).
std::vector<OtlpSpan> spans_for_request(const RequestSpan& r, const std::string& trace_id_override = "",
                                        const std::string& root_span_id_override = "");

// OTLP/HTTP JSON body (ExportTraceServiceRequest, proto3 JSON mapping: 64-bit
// integers and timestamps as decimal strings, ids as lowercase hex).
std::string otlp_json(const std::vector<OtlpSpan>& spans, const std::string& service_name,
                      const std::string& service_version);

class Tracer {
public:
    ~Tracer();
    // endpoint: full OTLP traces URL, e.g. http://localhost:4318/v1/traces.
    // Empty = disabled. Only http:// is supported (no TLS in this build).
    void init(const std::string& endpoint, const std::string& service_name,
              const std::string& service_version);
    bool enabled() const { return enabled_; }
    // Returns the root span's ids (trace_id, span_id) so the caller can
    // write them next to the request log record.
    std::pair<std::string, std::string> record(const RequestSpan& r);
    void stop();
    // Counters for /metrics and tests.
    uint64_t exported_spans() const { return exported_spans_; }
    uint64_t failed_batches() const { return failed_batches_; }

private:
    void worker_();
    bool post_(const std::string& body);

    bool enabled_ = false;
    std::string host_, path_, service_, version_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<OtlpSpan> pending_;
    bool stop_ = false;
    std::thread thread_;
    std::atomic<uint64_t> exported_spans_{0};
    std::atomic<uint64_t> failed_batches_{0};
    bool warned_ = false;
};

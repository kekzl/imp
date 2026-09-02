#include "tracing.h"

#include "core/logging.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdio>
#include <random>

using json = nlohmann::json;

namespace {

bool is_lower_hex(const std::string& s) {
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

bool all_zero(const std::string& s) {
    for (char c : s)
        if (c != '0')
            return false;
    return true;
}

uint64_t to_ns(std::chrono::system_clock::time_point tp) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

json attr_str(const std::string& k, const std::string& v) {
    return json{{"key", k}, {"value", json{{"stringValue", v}}}};
}
json attr_int(const std::string& k, int64_t v) {
    return json{{"key", k}, {"value", json{{"intValue", std::to_string(v)}}}};
}
json attr_bool(const std::string& k, bool v) { return json{{"key", k}, {"value", json{{"boolValue", v}}}}; }

}  // namespace

bool parse_traceparent(const std::string& header, TraceContext& ctx) {
    // version(2)-traceid(32)-parentid(16)-flags(2) = 55 chars
    if (header.size() != 55 || header[2] != '-' || header[35] != '-' || header[52] != '-')
        return false;
    const std::string ver = header.substr(0, 2), tid = header.substr(3, 32), pid = header.substr(36, 16),
                      flags = header.substr(53, 2);
    if (ver != "00" || !is_lower_hex(tid) || !is_lower_hex(pid) || !is_lower_hex(flags))
        return false;
    if (all_zero(tid) || all_zero(pid))
        return false;
    ctx.trace_id = tid;
    ctx.parent_id = pid;
    ctx.sampled = (std::stoi(flags, nullptr, 16) & 1) != 0;
    return true;
}

std::string random_hex_id(int n_bytes) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(static_cast<size_t>(n_bytes) * 2);
    while (static_cast<int>(out.size()) < n_bytes * 2) {
        uint64_t v = rng();
        for (int i = 0; i < 16 && static_cast<int>(out.size()) < n_bytes * 2; i++) {
            out.push_back(hex[v & 0xF]);
            v >>= 4;
        }
    }
    // A zero id is invalid in W3C / OTLP; the chance is 2^-64, but be exact.
    if (all_zero(out))
        out.back() = '1';
    return out;
}

std::vector<OtlpSpan> spans_for_request(const RequestSpan& r, const std::string& trace_id_override,
                                        const std::string& root_span_id_override) {
    TraceContext parent;
    const bool have_parent = !r.traceparent.empty() && parse_traceparent(r.traceparent, parent);
    std::vector<OtlpSpan> out;
    OtlpSpan root;
    root.name = r.endpoint.empty() ? "request" : r.endpoint;
    root.trace_id = !trace_id_override.empty() ? trace_id_override
                                               : (have_parent ? parent.trace_id : random_hex_id(16));
    root.span_id = !root_span_id_override.empty() ? root_span_id_override : random_hex_id(8);
    root.parent_id = have_parent ? parent.parent_id : "";
    root.kind = 2;  // SERVER
    root.start_ns = to_ns(r.t_start);
    root.end_ns = root.start_ns + static_cast<uint64_t>(r.latency_ms > 0 ? r.latency_ms * 1e6 : 0);
    root.str_attrs.push_back({"imp.request_id", r.req_id});
    if (!r.client_request_id.empty())
        root.str_attrs.push_back({"imp.client_request_id", r.client_request_id});
    if (!r.model.empty())
        root.str_attrs.push_back({"gen_ai.request.model", r.model});
    root.str_attrs.push_back({"gen_ai.system", "imp"});
    root.int_attrs.push_back({"gen_ai.usage.input_tokens", r.prompt_tokens});
    root.int_attrs.push_back({"gen_ai.usage.output_tokens", r.completion_tokens});
    root.int_attrs.push_back({"imp.cached_tokens", r.cached_tokens});
    root.int_attrs.push_back({"http.response.status_code", r.http_status});
    if (!r.finish_reason.empty())
        root.str_attrs.push_back({"gen_ai.response.finish_reasons", r.finish_reason});
    root.bool_attrs.push_back({"imp.stream", r.stream});
    if (r.queue_ms >= 0)
        root.int_attrs.push_back({"imp.queue_ms", static_cast<int64_t>(r.queue_ms)});
    if (r.ttft_ms >= 0)
        root.int_attrs.push_back({"imp.ttft_ms", static_cast<int64_t>(r.ttft_ms)});
    out.push_back(root);

    // Children on the request's timeline: [start, start+queue) queue,
    // [queue end, first token) prefill, [first token, end) decode.
    auto child = [&](const char* name, uint64_t s, uint64_t e) {
        OtlpSpan c;
        c.name = name;
        c.trace_id = root.trace_id;
        c.span_id = random_hex_id(8);
        c.parent_id = root.span_id;
        c.kind = 1;
        c.start_ns = s;
        c.end_ns = e < s ? s : e;
        return c;
    };
    uint64_t t = root.start_ns;
    if (r.queue_ms >= 0) {
        const uint64_t q_end = t + static_cast<uint64_t>(r.queue_ms * 1e6);
        out.push_back(child("queue", t, q_end));
        t = q_end;
    }
    if (r.ttft_ms >= 0 && r.completion_tokens > 0) {
        const uint64_t first = root.start_ns + static_cast<uint64_t>(r.ttft_ms * 1e6);
        out.push_back(child("prefill", t, first));
        OtlpSpan d = child("decode", first, root.end_ns);
        d.int_attrs.push_back({"gen_ai.usage.output_tokens", r.completion_tokens});
        out.push_back(d);
    }
    return out;
}

std::string otlp_json(const std::vector<OtlpSpan>& spans, const std::string& service_name,
                      const std::string& service_version) {
    json jspans = json::array();
    for (const auto& s : spans) {
        json js = {{"traceId", s.trace_id},
                   {"spanId", s.span_id},
                   {"name", s.name},
                   {"kind", s.kind},
                   {"startTimeUnixNano", std::to_string(s.start_ns)},
                   {"endTimeUnixNano", std::to_string(s.end_ns)},
                   {"status", json{{"code", 1}}}};
        if (!s.parent_id.empty())
            js["parentSpanId"] = s.parent_id;
        json attrs = json::array();
        for (const auto& [k, v] : s.str_attrs)
            attrs.push_back(attr_str(k, v));
        for (const auto& [k, v] : s.int_attrs)
            attrs.push_back(attr_int(k, v));
        for (const auto& [k, v] : s.bool_attrs)
            attrs.push_back(attr_bool(k, v));
        js["attributes"] = attrs;
        jspans.push_back(js);
    }
    json body = {
        {"resourceSpans",
         json::array({json{
             {"resource", json{{"attributes", json::array({attr_str("service.name", service_name),
                                                           attr_str("service.version", service_version)})}}},
             {"scopeSpans",
              json::array({json{{"scope", json{{"name", "imp-server"}, {"version", service_version}}},
                                {"spans", jspans}}})}}})}};
    return body.dump();
}

Tracer::~Tracer() { stop(); }

void Tracer::init(const std::string& endpoint, const std::string& service_name,
                  const std::string& service_version) {
    if (endpoint.empty())
        return;
    const auto scheme_end = endpoint.find("://");
    if (scheme_end == std::string::npos || endpoint.compare(0, 7, "http://") != 0) {
        IMP_LOG_WARN("tracing: only http:// OTLP endpoints are supported, got '%s' - tracing off",
                     endpoint.c_str());
        return;
    }
    const auto path_start = endpoint.find('/', scheme_end + 3);
    host_ = path_start == std::string::npos ? endpoint : endpoint.substr(0, path_start);
    path_ = path_start == std::string::npos ? "/v1/traces" : endpoint.substr(path_start);
    service_ = service_name.empty() ? "imp-server" : service_name;
    version_ = service_version;
    enabled_ = true;
    stop_ = false;
    thread_ = std::thread([this] { worker_(); });
    IMP_LOG_INFO("tracing: OTLP/HTTP export to %s%s as service '%s'", host_.c_str(), path_.c_str(),
                 service_.c_str());
}

std::pair<std::string, std::string> Tracer::record(const RequestSpan& r) {
    if (!enabled_)
        return {};
    auto spans = spans_for_request(r);
    const std::pair<std::string, std::string> ids{spans[0].trace_id, spans[0].span_id};
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : spans)
            pending_.push_back(std::move(s));
    }
    cv_.notify_one();
    return ids;
}

bool Tracer::post_(const std::string& body) {
    httplib::Client cli(host_);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);
    cli.set_write_timeout(5, 0);
    auto res = cli.Post(path_, body, "application/json");
    return res && res->status >= 200 && res->status < 300;
}

void Tracer::worker_() {
    for (;;) {
        std::vector<OtlpSpan> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            // Batch what arrives within a second (or 256 spans), then export.
            cv_.wait_for(lk, std::chrono::seconds(1), [this] { return stop_ || pending_.size() >= 256; });
            if (pending_.empty()) {
                if (stop_)
                    return;
                continue;
            }
            batch.swap(pending_);
        }
        const std::string body = otlp_json(batch, service_, version_);
        if (post_(body)) {
            exported_spans_ += batch.size();
        } else {
            failed_batches_++;
            if (!warned_) {
                warned_ = true;
                IMP_LOG_WARN(
                    "tracing: OTLP export to %s%s failed (%zu spans dropped); further failures are "
                    "counted, not logged",
                    host_.c_str(), path_.c_str(), batch.size());
            }
        }
        if (stop_) {
            std::lock_guard<std::mutex> lk(mu_);
            if (pending_.empty())
                return;
        }
    }
}

void Tracer::stop() {
    if (!enabled_)
        return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
    enabled_ = false;
}

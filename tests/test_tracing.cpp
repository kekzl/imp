// imp-server tracing: W3C traceparent parsing, the span set a request
// becomes, and the OTLP/HTTP JSON envelope a collector receives. No GPU, no
// network: the exporter thread is exercised only through its pure pieces.
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <string>

#include "tracing.h"

using json = nlohmann::json;

namespace {

RequestSpan sample_request() {
    RequestSpan r;
    r.endpoint = "chat.completions";
    r.req_id = "chatcmpl-42";
    r.client_request_id = "agent-7";
    r.model = "Qwen3.8-27B-NVFP4-vllm";
    r.t_start = std::chrono::system_clock::time_point(std::chrono::milliseconds(1'700'000'000'000));
    r.latency_ms = 1000.0;
    r.queue_ms = 100.0;
    r.ttft_ms = 400.0;
    r.prompt_tokens = 120;
    r.completion_tokens = 33;
    r.cached_tokens = 64;
    r.finish_reason = "stop";
    r.stream = true;
    return r;
}

}  // namespace

TEST(TracingTest, ParseTraceparentAcceptsOnlyWellFormedVersion00) {
    TraceContext c;
    ASSERT_TRUE(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", c));
    EXPECT_EQ(c.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
    EXPECT_EQ(c.parent_id, "00f067aa0ba902b7");
    EXPECT_TRUE(c.sampled);
    ASSERT_TRUE(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00", c));
    EXPECT_FALSE(c.sampled);
    TraceContext untouched;
    EXPECT_FALSE(parse_traceparent("", untouched));
    EXPECT_FALSE(parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", untouched))
        << "only version 00 is understood";
    EXPECT_FALSE(parse_traceparent("00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01", untouched))
        << "hex must be lowercase per the spec";
    EXPECT_FALSE(parse_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01", untouched))
        << "all-zero trace id is invalid";
    EXPECT_FALSE(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", untouched))
        << "all-zero parent id is invalid";
    EXPECT_FALSE(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7", untouched))
        << "missing flags";
    EXPECT_TRUE(untouched.trace_id.empty());
}

TEST(TracingTest, RandomIdsHaveTheRightWidthAndAreNeverZero) {
    for (int i = 0; i < 64; i++) {
        const std::string t = random_hex_id(16), s = random_hex_id(8);
        EXPECT_EQ(t.size(), 32u);
        EXPECT_EQ(s.size(), 16u);
        EXPECT_NE(t, std::string(32, '0'));
        EXPECT_NE(s, std::string(16, '0'));
        EXPECT_NE(t, random_hex_id(16));
    }
}

TEST(TracingTest, SpansForRequestJoinTheCallersTraceAndSplitTheTimeline) {
    RequestSpan r = sample_request();
    r.traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    auto spans = spans_for_request(r, "", "aaaaaaaaaaaaaaaa");
    ASSERT_EQ(spans.size(), 4u);  // root + queue + prefill + decode
    const auto& root = spans[0];
    EXPECT_EQ(root.name, "chat.completions");
    EXPECT_EQ(root.kind, 2);  // SERVER
    EXPECT_EQ(root.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
    EXPECT_EQ(root.parent_id, "00f067aa0ba902b7");
    EXPECT_EQ(root.span_id, "aaaaaaaaaaaaaaaa");
    const uint64_t t0 = 1'700'000'000'000ull * 1'000'000ull;
    EXPECT_EQ(root.start_ns, t0);
    EXPECT_EQ(root.end_ns, t0 + 1'000'000'000ull);
    EXPECT_EQ(spans[1].name, "queue");
    EXPECT_EQ(spans[1].start_ns, t0);
    EXPECT_EQ(spans[1].end_ns, t0 + 100'000'000ull);
    EXPECT_EQ(spans[2].name, "prefill");
    EXPECT_EQ(spans[2].start_ns, t0 + 100'000'000ull);
    EXPECT_EQ(spans[2].end_ns, t0 + 400'000'000ull);
    EXPECT_EQ(spans[3].name, "decode");
    EXPECT_EQ(spans[3].start_ns, t0 + 400'000'000ull);
    EXPECT_EQ(spans[3].end_ns, root.end_ns);
    for (size_t i = 1; i < spans.size(); i++) {
        EXPECT_EQ(spans[i].trace_id, root.trace_id);
        EXPECT_EQ(spans[i].parent_id, root.span_id);
        EXPECT_EQ(spans[i].kind, 1);  // INTERNAL
        EXPECT_EQ(spans[i].span_id.size(), 16u);
        EXPECT_NE(spans[i].span_id, root.span_id);
    }
    // Attributes a collector query keys on.
    bool saw_model = false, saw_in = false;
    for (const auto& [k, v] : root.str_attrs)
        if (k == "gen_ai.request.model" && v == r.model)
            saw_model = true;
    for (const auto& [k, v] : root.int_attrs)
        if (k == "gen_ai.usage.input_tokens" && v == 120)
            saw_in = true;
    EXPECT_TRUE(saw_model);
    EXPECT_TRUE(saw_in);
}

TEST(TracingTest, NoTraceparentMintsATraceAndNonStreamHasNoPrefillDecodeSplit) {
    RequestSpan r = sample_request();
    r.traceparent = "garbage";
    r.stream = false;
    r.ttft_ms = -1.0;
    auto spans = spans_for_request(r);
    ASSERT_EQ(spans.size(), 2u);  // root + queue only
    EXPECT_EQ(spans[0].trace_id.size(), 32u);
    EXPECT_TRUE(spans[0].parent_id.empty()) << "an unparseable traceparent must not become a parent";
    EXPECT_EQ(spans[1].name, "queue");
    r.queue_ms = -1.0;
    EXPECT_EQ(spans_for_request(r).size(), 1u) << "unknown queue time: root only";
}

TEST(TracingTest, OtlpJsonIsTheProto3MappingACollectorAccepts) {
    RequestSpan r = sample_request();
    r.traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const auto spans = spans_for_request(r, "", "aaaaaaaaaaaaaaaa");
    const json body = json::parse(otlp_json(spans, "imp-server", "0.34.0"));
    ASSERT_EQ(body["resourceSpans"].size(), 1u);
    const auto& rs = body["resourceSpans"][0];
    // service.name is the resource attribute every backend groups on.
    bool saw_service = false;
    for (const auto& a : rs["resource"]["attributes"])
        if (a["key"] == "service.name" && a["value"]["stringValue"] == "imp-server")
            saw_service = true;
    EXPECT_TRUE(saw_service);
    ASSERT_EQ(rs["scopeSpans"].size(), 1u);
    const auto& js = rs["scopeSpans"][0]["spans"];
    ASSERT_EQ(js.size(), 4u);
    const auto& root = js[0];
    EXPECT_EQ(root["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    EXPECT_EQ(root["spanId"], "aaaaaaaaaaaaaaaa");
    EXPECT_EQ(root["parentSpanId"], "00f067aa0ba902b7");
    EXPECT_EQ(root["kind"], 2);
    // 64-bit fields travel as decimal strings in the proto3 JSON mapping.
    EXPECT_TRUE(root["startTimeUnixNano"].is_string());
    EXPECT_EQ(root["startTimeUnixNano"], "1700000000000000000");
    EXPECT_EQ(root["endTimeUnixNano"], "1700000001000000000");
    bool saw_tokens = false;
    for (const auto& a : root["attributes"])
        if (a["key"] == "gen_ai.usage.output_tokens") {
            EXPECT_TRUE(a["value"]["intValue"].is_string());
            EXPECT_EQ(a["value"]["intValue"], "33");
            saw_tokens = true;
        }
    EXPECT_TRUE(saw_tokens);
    EXPECT_EQ(js[1]["parentSpanId"], "aaaaaaaaaaaaaaaa");
    EXPECT_FALSE(js[0].contains("parentSpanIdX"));
}

TEST(TracingTest, TracerDisabledWithoutEndpointAndRefusesHttps) {
    Tracer t;
    t.init("", "imp-server", "0");
    EXPECT_FALSE(t.enabled());
    EXPECT_TRUE(t.record(sample_request()).first.empty());
    Tracer u;
    u.init("https://collector:4318/v1/traces", "imp-server", "0");
    EXPECT_FALSE(u.enabled()) << "no TLS in this build: refuse rather than silently drop";
}

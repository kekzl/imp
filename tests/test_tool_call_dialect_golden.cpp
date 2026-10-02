// Tool-call dialect golden (#2464): every ChatTemplateFamily x corpus through the 9 tool_call.h
// entry points + StreamToolCallFilter (whole, per-byte, 3 seeded random chunkings), compared
// line by line to tests/fixtures/tool_call_dialect_golden.txt. Regenerate: IMP_TOOL_GOLDEN_DUMP=<path>.

#include <gtest/gtest.h>
#include "tool_call.h"
#include "tool_stream_filter.h"
#include "model/chat_template.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using imp::ChatTemplateFamily;
using imp::server::StreamToolCallFilter;

namespace {

constexpr ChatTemplateFamily kFamilies[] = {
    ChatTemplateFamily::RAW,        ChatTemplateFamily::CHATML,      ChatTemplateFamily::LLAMA2,
    ChatTemplateFamily::MISTRAL_V3, ChatTemplateFamily::LLAMA3,      ChatTemplateFamily::NEMOTRON,
    ChatTemplateFamily::GEMMA,      ChatTemplateFamily::DEEPSEEK_R1, ChatTemplateFamily::PHI,
    ChatTemplateFamily::HARMONY,
};

// Model outputs: complete, multiple, malformed, unterminated, no call, every dialect's envelope.
const std::vector<std::string>& corpus() {
    static const std::vector<std::string> c = {
        "",
        "Plain answer with no tool call at all.",
        "Compare a < b and <b>bold</b> or <tool> or <|im_end|> text.",
        "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\", "
        "\"unit\": \"c\"}}\n"
        "</tool_call>",
        "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</"
        "tool_call>\nbetween\n<tool_call>{\"name\":\"b\","
        "\"arguments\":{\"s\":\"q\\\"}\"}}</tool_call>\n\n  after",
        "<tool_call>{\"name\":\"a\",\"arguments\":{}}<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":[1,2]}}<"
        "/tool_call>",
        "<tool_call>{\"name\": \"x\", \"arguments\": {broken</tool_call> tail",
        "pre <tool_call>{\"name\":\"x\",\"arguments\":{\"a\":1}",
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a.txt\n</"
        "parameter>\n<parameter=mode>\n0600\n"
        "</parameter>\n<parameter=n>\n42\n</parameter>\n<parameter=content>\nline1 </function> "
        "x\nline2\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>{\"name\":\"f\",\"city\":\"x\"}</tool_call>",
        "<tool_call>{\"name\":\"say\",\"arguments\":{\"text\":\"h\xC3\xA9llo \xE4\xB8\x96\xE7\x95\x8C "
        "\xF0\x9F\x8E\x89\"}}"
        "</tool_call>",
        "Sure.<function=get_weather>{\"city\": \"Paris\"}</function>",
        "<function=a>{\"x\":1}</function>\n<function=b>not "
        "json</function><function=c>{\"y\":[1,2,{\"z\":\"}\"}]}"
        "</function>",
        "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}",
        "{\"name\": \"launch\", \"parameters\": {}}",
        "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"A\"}}; "
        "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"B\"}}",
        "{\"answer\": 42}",
        "I'll call "
        "it.<|tool_call>call:get_weather{city:<|\"|>Berlin<|\"|>,days:3,flags:[true,null],nested:{a:<|\"|>x,y"
        "<|\"|>}}<tool_call|>",
        "<|tool_call>call:a{x:1}<tool_call|>mid text<|tool_call>call:b{s:<|\"|>q<|\"|>}<tool_call|>",
        "<|tool_call>call:{oops<tool_call|> after",
        "<|tool_call>call:get_weather{\"city\": \"Berlin\"}<tool_call|>",
        "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Berlin\"}<|call|>",
        "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Done.<|end|"
        ">",
        "<|channel|>commentary "
        "to=functions.a<|message|>{\"x\":1}<|call|><|start|>assistant<|channel|>commentary "
        "to=functions.b <|constrain|>json<|message|>{\"y\":2}<|call|>",
        "text then <tool_ca",
        "text then <function=get_wea",
        "text then <|tool_c",
        "text then <|channel|>commentary to=functions.x",
    };
    return c;
}

const std::vector<std::string> kKnownNames = {"get_weather", "say", "a", "b", "c", "f", "write_file", "x"};

std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char ch : s) {
        if (ch == '\n')
            o += "\\n";
        else if (ch == '\r')
            o += "\\r";
        else if (ch == '\t')
            o += "\\t";
        else if (ch == '\\')
            o += "\\\\";
        else if (ch < 0x20 || ch >= 0x7f) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02X", ch);
            o += b;
        } else {
            o += static_cast<char>(ch);
        }
    }
    return o;
}

std::string ser_call(const ParsedToolCall& c) {
    std::string o = "{id=" + esc(c.id) + " name=" + esc(c.name) + " args=" + esc(c.arguments) +
                    " valid=" + (c.valid ? "1" : "0") + " err=" + esc(c.error) + " raw=[";
    for (const auto& [k, v] : c.raw_params)
        o += esc(k) + ":" + esc(v) + ";";
    return o + "]}";
}

std::string ser_segments(const StreamToolCallFilter::Result& r) {
    using K = StreamToolCallFilter::Segment::Kind;
    std::string o;
    for (const auto& s : r) {
        switch (s.kind) {
            case K::TEXT:
                o += "T(" + esc(s.text) + ")";
                break;
            case K::CALL:
                o += "C" + ser_call(s.call);
                break;
            case K::CALL_BEGIN:
                o += "B" + ser_call(s.call);
                break;
            case K::CALL_ARGS_DELTA:
                o += "D(" + esc(s.text) + ")";
                break;
            case K::CALL_END:
                o += "E" + ser_call(s.call);
                break;
        }
    }
    return o;
}

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : s) {
        h ^= ch;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string run_stream(ChatTemplateFamily fam, const std::string& text, const std::vector<size_t>& cuts) {
    StreamToolCallFilter f(fam);
    std::string o;
    size_t pos = 0;
    for (size_t n : cuts) {
        o += "[" + ser_segments(f.feed(text.substr(pos, n))) + "]";
        pos += n;
    }
    o += " mid=" + std::to_string(f.mid_tool()) + " open=" + std::to_string(f.call_open()) +
         " streamed=" + esc(f.streamed_arguments()) + " finish=" + esc(f.finish());
    return o;
}

void dump_parse_and_stream(std::ostringstream& out, ChatTemplateFamily fam, const char* fn) {
    const auto& c = corpus();
    for (size_t i = 0; i < c.size(); ++i) {
        const std::string& t = c[i];
        for (int known = 0; known < 2; ++known) {
            std::atomic<int> ids{0};
            auto [content, calls] = parse_tool_calls(fam, t, ids,
                                                     known ? kKnownNames : std::vector<std::string>{});
            out << "parse|" << fn << "|s" << i << "|known=" << known << "|content=" << esc(content)
                << "|calls=";
            for (const auto& call : calls)
                out << ser_call(call);
            out << "\n";
        }
        out << "stream|" << fn << "|s" << i << "|whole|" << run_stream(fam, t, {t.size()}) << "\n";
        std::vector<size_t> bytes(t.size(), 1);
        const std::string per_byte = run_stream(fam, t, bytes);
        out << "stream|" << fn << "|s" << i << "|bytes|h=" << std::hex << fnv1a(per_byte) << std::dec
            << "|tail=" << per_byte.substr(per_byte.rfind(" mid=")) << "\n";
        for (unsigned seed = 1; seed <= 3; ++seed) {
            std::mt19937 rng(seed * 7919u + static_cast<unsigned>(i));
            std::vector<size_t> cuts;
            for (size_t left = t.size(); left > 0;) {
                size_t n = 1 + rng() % 7;
                n = n < left ? n : left;
                cuts.push_back(n);
                left -= n;
            }
            const std::string s = run_stream(fam, t, cuts);
            out << "stream|" << fn << "|s" << i << "|seed" << seed << "|h=" << std::hex << fnv1a(s)
                << std::dec << "\n";
        }
        for (size_t p = 0; p <= t.size(); ++p) {
            if (p != t.size() && t[p] != '<')
                continue;
            ToolTagScan s = scan_tool_tag(t.substr(p), fam);
            out << "scan|" << fn << "|s" << i << "|@" << p << "|k=" << static_cast<int>(s.kind)
                << " cl=" << s.content_len << " bs=" << s.body_start << " close=" << esc(s.close_tag)
                << " gemma=" << s.gemma_body << " fn=" << esc(s.fn_name) << "\n";
        }
    }
}

json tool_sets() {
    return json::array({
        json::array(),
        json::parse(R"([
          {"type":"function","function":{"name":"get_weather","description":"Weather","strict":true,
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
          {"type":"function","function":{"name":"bash","parameters":{"type":"object",
            "properties":{"cmd":{"type":"string"}}}}}])"),
        json::parse(R"([
          {"type":"function","function":{"name":"get_weather","strict":true,
            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
          {"type":"function","function":{"name":"noargs","strict":true}}])"),
        json::parse(R"([{"type":"function","function":{"name":"bash","parameters":{"type":"object",
            "properties":{"cmd":{"type":"string"}}}}}, {"type":"other"}])"),
        json::parse(R"([{"type":"function","function":{"description":"nameless","strict":true}}])"),
    });
}

json tool_choices() {
    return json::array({nullptr, "auto", "none", "required", "bogus",
                        json::parse(R"({"type":"function","function":{"name":"get_weather"}})"),
                        json::parse(R"({"type":"function","function":{"name":"bash"}})"),
                        json::parse(R"({"type":"function","function":{"name":"missing"}})"),
                        json::parse(R"({"type":"function","function":{}})")});
}

std::string ser_pairs(const std::vector<std::pair<std::string, std::string>>& v) {
    std::string o = "[";
    for (const auto& [a, b] : v)
        o += esc(a) + "=" + esc(b) + ";";
    return o + "]";
}

void dump_prompt_and_constraints(std::ostringstream& out, ChatTemplateFamily fam, const char* fn) {
    const json sets = tool_sets();
    const json choices = tool_choices();
    for (size_t ti = 0; ti < sets.size(); ++ti) {
        for (size_t ci = 0; ci < choices.size(); ++ci) {
            const json& tools = sets[ti];
            const json& tc = choices[ci];
            const std::string k = std::string(fn) + "|t" + std::to_string(ti) + "|c" + std::to_string(ci);
            const std::string p = build_tool_prompt(fam, tools, tc);
            out << "prompt|" << k << "|len=" << p.size() << " h=" << std::hex << fnv1a(p) << std::dec;
            if (ti == 1 && (ci == 0 || ci == 3 || ci == 5))
                out << "|" << esc(p);
            out << "\n";
            for (int native = 0; native < 2; ++native) {
                out << "enforceable|" << k << "|native=" << native << "|"
                    << tool_choice_is_enforceable(fam, tc, native != 0) << "\n";
                ForcedToolEnvelope e = collect_forced_bare_args_tool(fam, tools, tc, native != 0);
                out << "forced|" << k << "|native=" << native << "|name=" << esc(e.name)
                    << " params=" << esc(e.params) << " open=" << esc(e.open) << " close=" << esc(e.close)
                    << "\n";
            }
            out << "constraint|" << k << "|" << ser_pairs(collect_tool_constraint(fam, tools, tc)) << "\n";
            out << "strict|" << k << "|" << ser_pairs(collect_strict_tool_constraint(fam, tools, tc)) << "\n";
        }
    }
}

void dump_reconstruct_and_response(std::ostringstream& out, ChatTemplateFamily fam, const char* fn) {
    const json calls = json::array({
        json::array(),
        json::parse(R"([{"id":"c0","type":"function","function":{"name":"get_weather",
            "arguments":"{\"city\":\"Berlin\",\"n\":3}"}}])"),
        json::parse(
            R"([{"function":{"name":"f","arguments":{"s":"x y","b":true,"z":null,"l":[1,"two",{"k":2.5}],
            "o":{"in":"v"}}}}])"),
        json::parse(R"([{"function":{"name":"g","arguments":"not json"}},{"nofunction":1},
            {"function":{"arguments":"{}"}},{"function":{"name":"h"}}])"),
        json::parse(R"([{"function":{"name":"a","arguments":"{\"x\":1}"}},
            {"function":{"name":"b","arguments":"[1,2]"}}])"),
    });
    const std::vector<std::string> contents = {"", "null", "Some text before."};
    for (size_t ci = 0; ci < calls.size(); ++ci)
        for (size_t ti = 0; ti < contents.size(); ++ti)
            for (int xml = 0; xml < 2; ++xml)
                out << "reconstruct|" << fn << "|k" << ci << "|t" << ti << "|xml=" << xml << "|"
                    << esc(reconstruct_tool_call_output(fam, calls[ci], contents[ti], xml != 0)) << "\n";

    const json msgs = json::array({
        json::parse(R"({"role":"tool","content":"22C"})"),
        json::parse(R"({"role":"tool","name":"get_weather","content":"sunny \"hot\""})"),
        json::parse(R"({"role":"tool","name":"q","content":{"temp":22,"ok":true}})"),
        json::parse(
            R"({"role":"tool","content":[{"type":"text","text":"part1"},{"type":"text","text":"part2"}]})"),
        json::parse(R"({"role":"tool","content":null})"),
        json::parse(R"({"role":"tool"})"),
    });
    for (size_t mi = 0; mi < msgs.size(); ++mi)
        out << "response|" << fn << "|m" << mi << "|" << esc(format_tool_response(fam, msgs[mi])) << "\n";
}

std::string build_dump() {
    std::ostringstream out;
    for (ChatTemplateFamily fam : kFamilies) {
        const char* fn = imp::chat_template_family_name(fam);
        dump_parse_and_stream(out, fam, fn);
        dump_prompt_and_constraints(out, fam, fn);
        dump_reconstruct_and_response(out, fam, fn);
    }
    return out.str();
}

std::vector<std::string> lines_of(const std::string& s) {
    std::vector<std::string> v;
    std::istringstream in(s);
    for (std::string l; std::getline(in, l);)
        v.push_back(l);
    return v;
}

}  // namespace

TEST(ToolCallDialectGolden, AllFamiliesMatchRecordedGolden) {
    const std::string dump = build_dump();
    if (const char* path = std::getenv("IMP_TOOL_GOLDEN_DUMP")) {
        std::ofstream(path) << dump;
        std::printf("golden written: %s (%zu lines)\n", path, lines_of(dump).size());
    }
    std::ifstream in(std::string(IMP_TEST_FIXTURES_DIR) + "/tool_call_dialect_golden.txt");
    ASSERT_TRUE(in.good()) << "missing tests/fixtures/tool_call_dialect_golden.txt";
    std::stringstream buf;
    buf << in.rdbuf();
    const auto want = lines_of(buf.str());
    const auto got = lines_of(dump);
    ASSERT_EQ(got.size(), want.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i] && ++mismatches <= 10)
            ADD_FAILURE() << "line " << i + 1 << "\n want: " << want[i] << "\n  got: " << got[i];
    }
    EXPECT_EQ(mismatches, 0u);
    std::printf("tool-call golden: %zu families x %zu outputs, %zu lines compared\n", std::size(kFamilies),
                corpus().size(), want.size());
}

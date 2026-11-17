// imp's Jinja engine renders byte-equal to the env transformers (and vLLM) render with:
// fixtures from tests/refs/gen_jinja_parity.py (`make jinja-parity`), key order included.

#include "model/jinja.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

namespace {

namespace fs = std::filesystem;
using imp::jinja::Value;
using ojson = nlohmann::ordered_json;

Value to_value(const ojson& j) {
    if (j.is_null())
        return Value();
    if (j.is_boolean())
        return Value(j.get<bool>());
    if (j.is_number_integer())
        return Value(j.get<int64_t>());
    if (j.is_number_float())
        return Value(j.get<double>());
    if (j.is_string())
        return Value(j.get<std::string>());
    if (j.is_array()) {
        Value::Array a;
        for (const auto& e : j)
            a.push_back(to_value(e));
        return Value(std::move(a));
    }
    Value o = Value::make_object();
    for (auto it = j.begin(); it != j.end(); ++it)
        o.set(it.key(), to_value(it.value()));
    return o;
}

imp::jinja::Context to_context(const ojson& j) {
    imp::jinja::Context ctx;
    for (auto it = j.begin(); it != j.end(); ++it)
        ctx[it.key()] = to_value(it.value());
    return ctx;
}

ojson load(const std::string& name) {
    std::ifstream f(fs::path(IMP_TEST_FIXTURES_DIR) / "jinja_parity" / name);
    EXPECT_TRUE(f.good()) << name;
    return ojson::parse(f);
}

// strftime_now: the generator freezes the date (gen_chat_goldens FROZEN_DATE), imp renders today's.
std::string mask_dates(const std::string& s) {
    static const std::regex kDate(R"(\d{4}-\d{2}-\d{2})");
    return std::regex_replace(s, kDate, "YYYY-MM-DD");
}

void check_template(const std::string& file) {
    const ojson fx = load(file);
    imp::jinja::Template tpl;
    ASSERT_TRUE(tpl.parse(fx["template"].get<std::string>())) << fx["source"] << ": " << tpl.error();
    for (const auto& c : fx["cases"])
        EXPECT_EQ(mask_dates(tpl.render(to_context(c["context"]))),
                  mask_dates(c["expected"].get<std::string>()))
            << fx["source"] << " / " << c["name"];
}

}  // namespace

TEST(JinjaParity, Qwen3Coder) { check_template("qwen3_coder.json"); }
TEST(JinjaParity, Nemotron3Nano) { check_template("nemotron3_nano.json"); }
TEST(JinjaParity, Qwen38) { check_template("qwen38.json"); }
TEST(JinjaParity, Glm47Flash) { check_template("glm47_flash.json"); }
TEST(JinjaParity, Flint) { check_template("flint.json"); }
TEST(JinjaParity, KimiLinear) { check_template("kimi_linear.json"); }
TEST(JinjaParity, GptOss) { check_template("gpt_oss.json"); }

TEST(JinjaParity, Probes) {
    const ojson fx = load("probes.json");
    const auto ctx = to_context(fx["context"]);
    for (const auto& p : fx["probes"]) {
        imp::jinja::Template tpl;
        ASSERT_TRUE(tpl.parse(p["template"].get<std::string>())) << p["name"] << ": " << tpl.error();
        EXPECT_EQ(tpl.render(ctx), p["expected"].get<std::string>()) << p["name"];
    }
}

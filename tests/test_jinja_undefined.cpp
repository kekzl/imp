// jinja: `is undefined` / `is not undefined`.
//
// These are standard Jinja2 tests and HF chat templates use them to decide
// whether the CALLER expressed an opinion, which is a different question from
// whether the value is falsy. Qwen3.8's template opens with
//
//     {%- if enable_thinking is undefined or enable_thinking is true %}
//
// to guard its whole reasoning-effort block.
//
// `undefined` was missing from the parser's known-test list, so `x is undefined`
// fell through to the generic "is X = equality" branch and became
// `x == undefined`, with `undefined` evaluating to none. That is accidentally
// right when the variable is unset (none == none) and wrong the moment it is
// set to anything falsy: `false == none` compares true here, so a template that
// asked "did the caller say anything?" was told "no" while the caller was
// explicitly saying false.
//
// Measured consequence before the fix: rendering Qwen3.8 with
// suppress_thinking=true (which stamps enable_thinking=false) still emitted the
// reasoning-effort preamble, because the guard read as undefined.

#include "model/jinja.h"

#include <gtest/gtest.h>
#include <string>

namespace imp {
namespace {

// Render `src` with `enable_thinking` either absent, or set to `val`.
std::string render(const std::string& src, bool stamp, bool val) {
    jinja::Template t;
    EXPECT_TRUE(t.parse(src)) << t.error();
    jinja::Context c;
    if (stamp)
        c["enable_thinking"] = jinja::Value(val);
    return t.render(c);
}

constexpr const char* kIsUndefined =
    "{%- if enable_thinking is undefined %}UNDEF{% else %}DEF{% endif %}";
constexpr const char* kIsNotUndefined =
    "{%- if enable_thinking is not undefined %}DEF{% else %}UNDEF{% endif %}";
// The exact guard Qwen3.8's chat template opens its reasoning block with.
constexpr const char* kQwenGuard =
    "{%- if enable_thinking is undefined or enable_thinking is true %}ON{% else %}OFF{% endif %}";

TEST(JinjaUndefinedTest, UnsetVariableIsUndefined) {
    EXPECT_EQ(render(kIsUndefined, /*stamp=*/false, false), "UNDEF");
}

TEST(JinjaUndefinedTest, VariableSetToFalseIsDefined) {
    // The regression: a stamped `false` is DEFINED. Reading it as undefined is
    // what let a suppressed-thinking render keep the reasoning preamble.
    EXPECT_EQ(render(kIsUndefined, /*stamp=*/true, false), "DEF");
}

TEST(JinjaUndefinedTest, VariableSetToTrueIsDefined) {
    EXPECT_EQ(render(kIsUndefined, /*stamp=*/true, true), "DEF");
}

TEST(JinjaUndefinedTest, IsNotUndefinedIsTheExactInverse) {
    EXPECT_EQ(render(kIsNotUndefined, /*stamp=*/false, false), "UNDEF");
    EXPECT_EQ(render(kIsNotUndefined, /*stamp=*/true, false), "DEF");
    EXPECT_EQ(render(kIsNotUndefined, /*stamp=*/true, true), "DEF");
}

TEST(JinjaUndefinedTest, QwenThinkingGuardHonoursAnExplicitFalse) {
    EXPECT_EQ(render(kQwenGuard, /*stamp=*/false, false), "ON") << "unset: template default applies";
    EXPECT_EQ(render(kQwenGuard, /*stamp=*/true, true), "ON") << "explicit true";
    EXPECT_EQ(render(kQwenGuard, /*stamp=*/true, false), "OFF") << "explicit false must turn it off";
}

// `is defined` is the pre-existing spelling and must keep behaving; the fix adds
// a branch next to it rather than changing it.
TEST(JinjaUndefinedTest, IsDefinedStillWorks) {
    constexpr const char* src = "{%- if enable_thinking is defined %}DEF{% else %}UNDEF{% endif %}";
    EXPECT_EQ(render(src, /*stamp=*/false, false), "UNDEF");
    EXPECT_EQ(render(src, /*stamp=*/true, false), "DEF");
    EXPECT_EQ(render(src, /*stamp=*/true, true), "DEF");
}

}  // namespace
}  // namespace imp

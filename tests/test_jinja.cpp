// Tests for the minimal Jinja2 template engine (src/model/jinja.h/.cpp).
//
// Validates rendering of real-world LLM chat templates from GGUF files.
// Run with: imp-tests --gtest_filter="JinjaTest.*"

#include <gtest/gtest.h>
#include "model/jinja.h"

using namespace imp::jinja;

// ---------------------------------------------------------------------------
// Value type tests
// ---------------------------------------------------------------------------

TEST(JinjaValueTest, None) {
    Value v;
    EXPECT_TRUE(v.is_none());
    EXPECT_FALSE(v.is_truthy());
    // None renders as empty string or "None" depending on implementation
    EXPECT_TRUE(v.to_string().empty() || v.to_string() == "None");
}

TEST(JinjaValueTest, Bool) {
    Value t(true), f(false);
    EXPECT_TRUE(t.is_bool());
    EXPECT_TRUE(t.is_truthy());
    EXPECT_FALSE(f.is_truthy());
}

TEST(JinjaValueTest, Int) {
    Value v(42);
    EXPECT_TRUE(v.is_number());
    EXPECT_EQ(v.as_int(), 42);
    EXPECT_TRUE(v.is_truthy());
    EXPECT_FALSE(Value(0).is_truthy());
}

TEST(JinjaValueTest, String) {
    Value v(std::string("hello"));
    EXPECT_TRUE(v.is_string());
    EXPECT_EQ(v.as_string(), "hello");
    EXPECT_TRUE(v.is_truthy());
    EXPECT_FALSE(Value(std::string("")).is_truthy());
}

TEST(JinjaValueTest, Array) {
    Value arr = Value::array({Value(1), Value(2), Value(3)});
    EXPECT_TRUE(arr.is_array());
    EXPECT_EQ(arr.size(), 3);
    EXPECT_TRUE(arr.is_truthy());
}

TEST(JinjaValueTest, Object) {
    Value obj = Value::object({{"name", Value(std::string("Alice"))}, {"age", Value(30)}});
    EXPECT_TRUE(obj.is_object());
    EXPECT_EQ(obj.get(std::string("name")).as_string(), "Alice");
    EXPECT_EQ(obj.get(std::string("age")).as_int(), 30);
}

// ---------------------------------------------------------------------------
// Basic rendering tests
// ---------------------------------------------------------------------------

TEST(JinjaTest, PlainText) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("Hello, world!"));
    EXPECT_EQ(tpl.render({}), "Hello, world!");
}

TEST(JinjaTest, VariableSubstitution) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("Hello, {{ name }}!"));
    EXPECT_EQ(tpl.render({{"name", Value(std::string("World"))}}), "Hello, World!");
}

TEST(JinjaTest, ExpressionConcat) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ 'a' + 'b' + 'c' }}"));
    EXPECT_EQ(tpl.render({}), "abc");
}

TEST(JinjaTest, TildeConcat) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ 'hello' ~ ' ' ~ 'world' }}"));
    EXPECT_EQ(tpl.render({}), "hello world");
}

TEST(JinjaTest, IntArithmetic) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ 2 + 3 }}"));
    EXPECT_EQ(tpl.render({}), "5");
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

TEST(JinjaTest, IfTrue) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if show %}yes{% endif %}"));
    EXPECT_EQ(tpl.render({{"show", Value(true)}}), "yes");
    EXPECT_EQ(tpl.render({{"show", Value(false)}}), "");
}

TEST(JinjaTest, IfElse) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x %}A{% else %}B{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(true)}}), "A");
    EXPECT_EQ(tpl.render({{"x", Value(false)}}), "B");
}

TEST(JinjaTest, IfElif) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x == 1 %}A{% elif x == 2 %}B{% else %}C{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(1)}}), "A");
    EXPECT_EQ(tpl.render({{"x", Value(2)}}), "B");
    EXPECT_EQ(tpl.render({{"x", Value(3)}}), "C");
}

TEST(JinjaTest, ForLoop) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items %}{{ x }} {% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "a b c ");
}

TEST(JinjaTest, ForLoopVariables) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items %}{{ loop.index }}{% endfor %}"));
    auto items = Value::array({Value(1), Value(2), Value(3)});
    EXPECT_EQ(tpl.render({{"items", items}}), "123");
}

TEST(JinjaTest, ForLoopFirst) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items %}{% if loop.first %}[{% endif %}{{ x }}{% if loop.last %}]{% endif %}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "[ab]");
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

TEST(JinjaTest, TrimFilter) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text | trim }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("  hello  "))}}), "hello");
}

TEST(JinjaTest, LengthFilter) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ items | length }}"));
    auto items = Value::array({Value(1), Value(2)});
    EXPECT_EQ(tpl.render({{"items", items}}), "2");
}

TEST(JinjaTest, DefaultFilter) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ x | default('fallback') }}"));
    EXPECT_EQ(tpl.render({}), "fallback");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("real"))}}), "real");
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

TEST(JinjaTest, Comparisons) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x == 'hello' %}yes{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(std::string("hello"))}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("world"))}}), "");
}

TEST(JinjaTest, NotOperator) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if not x %}yes{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(false)}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(true)}}), "");
}

TEST(JinjaTest, InOperator) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if 'b' in items %}found{% endif %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "found");
}

TEST(JinjaTest, TernaryExpr) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ 'yes' if flag else 'no' }}"));
    EXPECT_EQ(tpl.render({{"flag", Value(true)}}), "yes");
    EXPECT_EQ(tpl.render({{"flag", Value(false)}}), "no");
}

// ---------------------------------------------------------------------------
// Attribute and subscript access
// ---------------------------------------------------------------------------

TEST(JinjaTest, DotAccess) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ msg.role }}"));
    auto msg = Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("hi"))}});
    EXPECT_EQ(tpl.render({{"msg", msg}}), "user");
}

TEST(JinjaTest, BracketAccess) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ msg['content'] }}"));
    auto msg = Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("hi"))}});
    EXPECT_EQ(tpl.render({{"msg", msg}}), "hi");
}

TEST(JinjaTest, NegativeIndex) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ items[-1] }}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "c");
}

// ---------------------------------------------------------------------------
// Whitespace control
// ---------------------------------------------------------------------------

TEST(JinjaTest, WhitespaceStripLeft) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("  {%- if true %} yes {%- endif %}"));
    EXPECT_EQ(tpl.render({}), " yes");
}

// ---------------------------------------------------------------------------
// Set statement
// ---------------------------------------------------------------------------

TEST(JinjaTest, SetVariable) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% set x = 'hello' %}{{ x }}"));
    EXPECT_EQ(tpl.render({}), "hello");
}

TEST(JinjaTest, NamespaceSet) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% set ns = namespace(found=false) %}{% for x in items %}{% if x == 'b' %}{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    auto result = tpl.render({{"items", items}});
    EXPECT_TRUE(result == "True" || result == "true");
}

// ---------------------------------------------------------------------------
// Real chat template tests
// ---------------------------------------------------------------------------

TEST(JinjaChatTest, ChatML) {
    // ChatML format (Qwen3)
    Template tpl;
    ASSERT_TRUE(tpl.parse(
        "{%- for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    ));

    auto msgs = Value::array({
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("Hello"))}}),
    });

    auto result = tpl.render({
        {"messages", msgs},
        {"add_generation_prompt", Value(true)},
    });

    EXPECT_NE(result.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(result.find("Hello"), std::string::npos);
    EXPECT_NE(result.find("<|im_end|>"), std::string::npos);
    EXPECT_NE(result.find("<|im_start|>assistant"), std::string::npos);
}

TEST(JinjaChatTest, Gemma) {
    // Gemma format
    Template tpl;
    ASSERT_TRUE(tpl.parse(
        "{%- for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{ '<start_of_turn>model\\n' }}"
        "{%- endif %}"
    ));

    auto msgs = Value::array({
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("Hi"))}}),
        Value::object({{"role", Value(std::string("model"))}, {"content", Value(std::string("Hello!"))}}),
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("Bye"))}}),
    });

    auto result = tpl.render({
        {"messages", msgs},
        {"add_generation_prompt", Value(true)},
    });

    EXPECT_NE(result.find("<start_of_turn>user\nHi<end_of_turn>"), std::string::npos);
    EXPECT_NE(result.find("<start_of_turn>model\nHello!<end_of_turn>"), std::string::npos);
    EXPECT_NE(result.find("<start_of_turn>model\n"), std::string::npos);  // generation prompt
}

TEST(JinjaChatTest, MultiTurnWithTrim) {
    // Llama3-style with trim filter
    Template tpl;
    ASSERT_TRUE(tpl.parse(
        "{{ bos_token }}"
        "{%- for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{%- endif %}"
    ));

    auto msgs = Value::array({
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("  Hello  "))}}),
    });

    auto result = tpl.render({
        {"messages", msgs},
        {"add_generation_prompt", Value(true)},
        {"bos_token", Value(std::string("<s>"))},
    });

    EXPECT_NE(result.find("<s>"), std::string::npos);
    EXPECT_NE(result.find("Hello"), std::string::npos);
    // Trim should have removed the spaces
    EXPECT_EQ(result.find("  Hello  "), std::string::npos);
}

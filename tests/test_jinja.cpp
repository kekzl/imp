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

// keep_trailing_newline=False (Jinja2/HF/vLLM default): a single trailing
// newline in the template source is stripped. A Qwen3-Coder chat template ends
// "...assistant\n' }}\n{%- endif %}\n"; without this, the generation prompt
// renders "<|im_start|>assistant\n\n" and the model emits an immediate EOS.
TEST(JinjaTest, StripsSingleTrailingNewline) {
    Template a;
    ASSERT_TRUE(a.parse("{{- 'assistant\\n' }}\n"));  // template file ends in a newline
    EXPECT_EQ(a.render({}), "assistant\n");            // trailing template newline stripped

    Template b;
    ASSERT_TRUE(b.parse("{{- 'assistant\\n' }}"));  // no trailing newline in source
    EXPECT_EQ(b.render({}), "assistant\n");

    Template crlf;
    ASSERT_TRUE(crlf.parse("X\r\n"));  // strips \r\n as one newline
    EXPECT_EQ(crlf.render({}), "X");

    Template two;
    ASSERT_TRUE(two.parse("X\n\n"));  // only ONE trailing newline stripped
    EXPECT_EQ(two.render({}), "X\n");
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
    ASSERT_TRUE(
        tpl.parse("{% for x in items %}{% if loop.first %}[{% endif %}{{ x }}{% if loop.last %}]{% endif "
                  "%}{% endfor %}"));
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
    ASSERT_TRUE(
        tpl.parse("{% set ns = namespace(found=false) %}{% for x in items %}{% if x == 'b' %}{% set ns.found "
                  "= true %}{% endif %}{% endfor %}{{ ns.found }}"));
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
    ASSERT_TRUE(
        tpl.parse("{%- for message in messages %}"
                  "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
                  "{%- endfor %}"
                  "{%- if add_generation_prompt %}"
                  "{{ '<|im_start|>assistant\\n' }}"
                  "{%- endif %}"));

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
        "{%- endif %}"));

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
    ASSERT_TRUE(
        tpl.parse("{{ bos_token }}"
                  "{%- for message in messages %}"
                  "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + "
                  "message['content'] | trim + '<|eot_id|>' }}"
                  "{%- endfor %}"
                  "{%- if add_generation_prompt %}"
                  "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                  "{%- endif %}"));

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

// ---------------------------------------------------------------------------
// Slice notation
// ---------------------------------------------------------------------------

TEST(JinjaTest, SliceFromStart) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items[1:] %}{{ x }}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "bc");
}

TEST(JinjaTest, SliceToEnd) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items[:2] %}{{ x }}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "ab");
}

TEST(JinjaTest, SliceStartStop) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items[0:3] %}{{ x }}{% endfor %}"));
    auto items = Value::array(
        {Value(std::string("a")), Value(std::string("b")), Value(std::string("c")), Value(std::string("d"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "abc");
}

TEST(JinjaTest, SliceNegativeIndex) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items[:-1] %}{{ x }}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "ab");
}

TEST(JinjaTest, SliceReverse) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% for x in items[::-1] %}{{ x }}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "cba");
}

TEST(JinjaTest, SliceString) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text[1:4] }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("hello"))}}), "ell");
}

TEST(JinjaTest, SliceInExpr) {
    // Slice as an expression assigned via set
    Template tpl;
    ASSERT_TRUE(tpl.parse("{%- set sub = items[1:] -%}{% for x in sub %}{{ x }}{% endfor %}"));
    auto items = Value::array({Value(std::string("a")), Value(std::string("b")), Value(std::string("c"))});
    EXPECT_EQ(tpl.render({{"items", items}}), "bc");
}

// ---------------------------------------------------------------------------
// Type tests (is string, is iterable, is defined, is none, etc.)
// ---------------------------------------------------------------------------

TEST(JinjaTest, IsString) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x is string %}yes{% else %}no{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(std::string("hello"))}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(42)}}), "no");
    EXPECT_EQ(tpl.render({{"x", Value::array({Value(1)})}}), "no");
}

TEST(JinjaTest, IsNotString) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x is not string %}yes{% else %}no{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(42)}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("hello"))}}), "no");
}

TEST(JinjaTest, IsIterable) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x is iterable %}yes{% else %}no{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value::array({Value(1)})}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("abc"))}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(42)}}), "no");
}

TEST(JinjaTest, IsMapping) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x is mapping %}yes{% else %}no{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value::object({{"a", Value(1)}})}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("abc"))}}), "no");
}

TEST(JinjaTest, IsNumber) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{% if x is number %}yes{% else %}no{% endif %}"));
    EXPECT_EQ(tpl.render({{"x", Value(42)}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(3.14)}}), "yes");
    EXPECT_EQ(tpl.render({{"x", Value(std::string("42"))}}), "no");
}

// ---------------------------------------------------------------------------
// String strip with character arguments
// ---------------------------------------------------------------------------

TEST(JinjaTest, StripWithChars) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text.strip('xy') }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("xyhelloyx"))}}), "hello");
}

TEST(JinjaTest, LstripWithChars) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text.lstrip('ab') }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("aabchello"))}}), "chello");
}

TEST(JinjaTest, RstripWithChars) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text.rstrip('!.') }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("hello!!.."))}}), "hello");
}

// ---------------------------------------------------------------------------
// tojson filter
// ---------------------------------------------------------------------------

TEST(JinjaTest, TojsonString) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ text | tojson }}"));
    EXPECT_EQ(tpl.render({{"text", Value(std::string("hello"))}}), "\"hello\"");
}

TEST(JinjaTest, TojsonNumber) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ x | tojson }}"));
    EXPECT_EQ(tpl.render({{"x", Value(42)}}), "42");
}

TEST(JinjaTest, TojsonArray) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ items | tojson }}"));
    auto items = Value::array({Value(1), Value(std::string("two")), Value(true)});
    EXPECT_EQ(tpl.render({{"items", items}}), "[1, \"two\", true]");
}

TEST(JinjaTest, TojsonNone) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ x | tojson }}"));
    EXPECT_EQ(tpl.render({}), "null");
}

TEST(JinjaTest, TojsonBool) {
    Template tpl;
    ASSERT_TRUE(tpl.parse("{{ x | tojson }}"));
    EXPECT_EQ(tpl.render({{"x", Value(true)}}), "true");
    EXPECT_EQ(tpl.render({{"x", Value(false)}}), "false");
}

// ---------------------------------------------------------------------------
// Real Qwen3 chat template (with is string, system message, slice)
// ---------------------------------------------------------------------------

TEST(JinjaChatTest, Qwen3RealTemplate) {
    Template tpl;
    ASSERT_TRUE(
        tpl.parse("{%- if messages[0].role == 'system' %}"
                  "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
                  "{%- endif %}"
                  "{%- for message in messages %}"
                  "{%- if message.content is string %}"
                  "{%- set content = message.content %}"
                  "{%- else %}"
                  "{%- set content = '' %}"
                  "{%- endif %}"
                  "{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}"
                  "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"
                  "{%- elif message.role == \"assistant\" %}"
                  "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}"
                  "{%- endif %}"
                  "{%- endfor %}"
                  "{%- if add_generation_prompt %}"
                  "{{- '<|im_start|>assistant\\n' }}"
                  "{%- endif %}"));

    auto msgs = Value::array({
        Value::object(
            {{"role", Value(std::string("system"))}, {"content", Value(std::string("You are helpful."))}}),
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("Hello"))}}),
    });

    auto result = tpl.render({
        {"messages", msgs},
        {"add_generation_prompt", Value(true)},
    });

    EXPECT_NE(result.find("<|im_start|>system\nYou are helpful.<|im_end|>"), std::string::npos);
    EXPECT_NE(result.find("<|im_start|>user\nHello<|im_end|>"), std::string::npos);
    EXPECT_NE(result.find("<|im_start|>assistant\n"), std::string::npos);
}

TEST(JinjaChatTest, Qwen3IsStringWithNonStringContent) {
    // Test that "is string" returns false for non-string content
    Template tpl;
    ASSERT_TRUE(
        tpl.parse("{%- for message in messages %}"
                  "{%- if message.content is string %}"
                  "{{- 'STR:' + message.content }}"
                  "{%- else %}"
                  "{{- 'OTHER' }}"
                  "{%- endif %}"
                  "{%- endfor %}"));

    auto msgs = Value::array({
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("hello"))}}),
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(42)}}),
    });

    auto result = tpl.render({{"messages", msgs}});
    EXPECT_EQ(result, "STR:helloOTHER");
}

// ---------------------------------------------------------------------------
// Real Gemma chat template (with slice, is string, trim filter)
// ---------------------------------------------------------------------------

TEST(JinjaChatTest, GemmaRealTemplate) {
    Template tpl;
    ASSERT_TRUE(
        tpl.parse("{%- if messages[0]['role'] == 'system' -%}"
                  "{%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}"
                  "{%- set loop_messages = messages[1:] -%}"
                  "{%- else -%}"
                  "{%- set first_user_prefix = \"\" -%}"
                  "{%- set loop_messages = messages -%}"
                  "{%- endif -%}"
                  "{%- for message in loop_messages -%}"
                  "{%- if (message['role'] == 'assistant') -%}"
                  "{%- set role = \"model\" -%}"
                  "{%- else -%}"
                  "{%- set role = message['role'] -%}"
                  "{%- endif -%}"
                  "{{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}"
                  "{%- if message['content'] is string -%}"
                  "{{ message['content'] | trim }}"
                  "{%- endif -%}"
                  "{{ '<end_of_turn>\\n' }}"
                  "{%- endfor -%}"
                  "{%- if add_generation_prompt -%}"
                  "{{'<start_of_turn>model\\n'}}"
                  "{%- endif -%}"));

    // Test with system message (exercises messages[1:] slice)
    auto msgs_with_sys = Value::array({
        Value::object(
            {{"role", Value(std::string("system"))}, {"content", Value(std::string("Be concise"))}}),
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("  Hello  "))}}),
        Value::object({{"role", Value(std::string("assistant"))}, {"content", Value(std::string("  Hi  "))}}),
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("  Bye  "))}}),
    });

    auto result = tpl.render({
        {"messages", msgs_with_sys},
        {"add_generation_prompt", Value(true)},
    });

    // System message extracted as prefix for first user message
    EXPECT_NE(result.find("<start_of_turn>user\nBe concise\n\nHello<end_of_turn>"), std::string::npos);
    // Assistant mapped to model
    EXPECT_NE(result.find("<start_of_turn>model\nHi<end_of_turn>"), std::string::npos);
    // Second user message without system prefix
    EXPECT_NE(result.find("<start_of_turn>user\nBye<end_of_turn>"), std::string::npos);
    // Generation prompt
    EXPECT_NE(result.find("<start_of_turn>model\n"), std::string::npos);
    // trim filter should have removed spaces
    EXPECT_EQ(result.find("  Hello  "), std::string::npos);
}

TEST(JinjaChatTest, GemmaRealTemplateNoSystem) {
    Template tpl;
    ASSERT_TRUE(
        tpl.parse("{%- if messages[0]['role'] == 'system' -%}"
                  "{%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}"
                  "{%- set loop_messages = messages[1:] -%}"
                  "{%- else -%}"
                  "{%- set first_user_prefix = \"\" -%}"
                  "{%- set loop_messages = messages -%}"
                  "{%- endif -%}"
                  "{%- for message in loop_messages -%}"
                  "{%- if (message['role'] == 'assistant') -%}"
                  "{%- set role = \"model\" -%}"
                  "{%- else -%}"
                  "{%- set role = message['role'] -%}"
                  "{%- endif -%}"
                  "{{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}"
                  "{%- if message['content'] is string -%}"
                  "{{ message['content'] | trim }}"
                  "{%- endif -%}"
                  "{{ '<end_of_turn>\\n' }}"
                  "{%- endfor -%}"
                  "{%- if add_generation_prompt -%}"
                  "{{'<start_of_turn>model\\n'}}"
                  "{%- endif -%}"));

    // Test without system message (loop_messages = messages, no slice needed)
    auto msgs = Value::array({
        Value::object({{"role", Value(std::string("user"))}, {"content", Value(std::string("Hello"))}}),
    });

    auto result = tpl.render({
        {"messages", msgs},
        {"add_generation_prompt", Value(true)},
    });

    EXPECT_NE(result.find("<start_of_turn>user\nHello<end_of_turn>"), std::string::npos);
    EXPECT_NE(result.find("<start_of_turn>model\n"), std::string::npos);
}

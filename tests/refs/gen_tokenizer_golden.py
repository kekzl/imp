#!/usr/bin/env python3
"""Generate the tokenizer parity golden from HuggingFace transformers.

    python3 tests/refs/gen_tokenizer_golden.py <model_dir> <out.h> [namespace] [extended]

`extended` adds EXTENDED_STRINGS plus the rendered chat template and loads with
trust_remote_code (Kimi tiktoken golden, tokenizer files only):

    docker run --rm -e HF_HOME=/tmp/hf -v /mnt/data/scratch/kimi-tok:/tok:ro \\
      -v $PWD:/src <python image with transformers==4.57.1 tiktoken jinja2> \\
      python -I /src/tests/refs/gen_tokenizer_golden.py /tok \\
      /src/tests/refs/tokenizer_golden_kimi.h tokenizer_golden_kimi extended

`namespace` defaults to tokenizer_golden. Pass a distinct one when a second
model's golden has to link alongside the first: two headers both declaring
tokenizer_golden::k_model in one test binary is a redefinition, and the
tokenizers are genuinely different (Qwen3 is 151k, Qwen3.8 is 248k).

The output is a committed C++ header, not a JSON file read at run time
(#1569): the previous generator wrote JSON that nothing committed and the
test parsed with a substring scan, which truncated the one case containing a
'}' inside a string and silently dropped it (#1570). A header has neither
problem - the compiler parses it, and a missing golden is a build error
rather than a skip.

The corpus deliberately over-samples runs of whitespace: that is the defect
class of #657 (gemma-3 SPM USER_DEFINED literal-space pieces decomposing into
N single-space tokens, +37.5% NLL), and it is what a percentage-based
threshold used to let through.

Requires: transformers. The vllm image on this host already carries it:

    docker run --rm --entrypoint python3 \\
      -v $HOME/models:/models:ro -v $PWD:/src vllm/vllm-openai:v0.27.1 \\
      /src/tests/refs/gen_tokenizer_golden.py /models/Qwen3-0.6B \\
      /src/tests/refs/tokenizer_golden_qwen3.h
"""
import json
import sys

from transformers import AutoTokenizer

TEST_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  Leading spaces",
    "Multiple   spaces   between",
    "12345 numbers 67890",
    "Special chars: !@#$%^&*()",
    "Unicode: café résumé naïve",
    "CJK: 你好世界",
    "Emoji: 🎉🚀",
    "Code: def foo(x):\n    return x + 1",
    "Mixed: Hello 你好 🌍 café 123",
    "Empty after trim:   ",
    "Newlines\n\nand\ttabs",
    "Contractions: I'm don't won't can't",
    "URL: https://example.com/path?q=hello&lang=en",
    "JSON: {\"key\": \"value\", \"num\": 42}",
    "",
    " ",
    "a",
    "A" * 100,
    # Whitespace runs, the #657 class.
    "        eight leading spaces",
    "trailing spaces        ",
    "tab\tseparated\tfields",
    "\t\t\tthree leading tabs",
    "mixed \t \t whitespace \t run",
    "line1\r\nline2\r\nline3",
    "para\n\n\n\nbreak",
    "  \n  indent after newline",
    # Chat-control literals appearing as ordinary text: the tokenizer
    # pre-splits on its special pieces, so this is where imp and HF can
    # disagree without any ordinary string showing it.
    "<|im_start|>user",
    "text <|endoftext|> text",
    # Digit grouping and long runs.
    "1234567890123456789012345678901234567890",
    "0.000001 and 1e-9 and -42",
]

# Corpus "extended" (tiktoken / Kimi): scripts, case rules and specials the base corpus misses.
EXTENDED_STRINGS = [
    "Grüße aus München: Äpfel, Öl und Übermut kosten 3,50 € pro Stück.",
    "Straße, Maß, Fußgängerübergänge und Schweißarbeiten",
    "人工智能正在改变世界，我们需要负责任地发展它。",
    "東京は日本の首都です。ひらがな、カタカナ・漢字ー々〆〇を混ぜる。",
    "한국어 문장도 테스트합니다.",
    "Привет, мир! Это тест.",
    "مرحبا بالعالم",
    "abc中文def ABC漢字DEF",
    "I'LL SEE HE'S THERE, We'Re fine, they'VE gone",
    "ǅemal ǈubljana ǋaša",
    "é café ñ",
    "👨‍👩‍👧‍👦 family, 🏳️‍🌈 flag, 👍🏽 thumbs",
    "31415926535897932384626433832795028841971693993751",
    "Price: $1,234,567.89 (approx. 1.2e6), id=000123",
    "fn main() {\n\tlet v: Vec<u8> = vec![1, 2, 3];\n\tprintln!(\"{:?}\", v);\n}\n",
    "    \n\n   \t\n x    nbsp   em \u0085 nel",
    "trailing newline run\n\n\n",
    "<|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>",
    "text [EOS] more [BOS]x[EOT]",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
    "<|reserved_token_163600|> reserved",
    "<|im_end",
]

EXTENDED_CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Wie spät ist es in 東京? Antworte kurz."},
    {"role": "assistant", "content": "Es ist 14:05 Uhr."},
    {"role": "user", "content": "Danke! 👍"},
]


def c_escape(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        c = chr(b)
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif 0x20 <= b < 0x7F:
            out.append(c)
        else:
            out.append("\\x%02x\"\"" % b)  # split so a following hex digit cannot extend the escape
    return "".join(out)


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <out.h>")
        return 1
    model_path, out_path = sys.argv[1], sys.argv[2]
    ns = sys.argv[3] if len(sys.argv) > 3 else "tokenizer_golden"
    extended = len(sys.argv) > 4 and sys.argv[4] == "extended"

    # trust_remote_code: tiktoken checkpoints ship their tokenizer class as Python.
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=extended)
    strings = list(TEST_STRINGS)
    if extended:
        strings += EXTENDED_STRINGS
        strings.append(tok.apply_chat_template(EXTENDED_CHAT, tokenize=False, add_generation_prompt=True))
    cases = []
    differs = 0
    for text in strings:
        ids = tok.encode(text, add_special_tokens=False)
        if extended and tok.encode(text) != ids:  # TikTokenTokenizer.encode without kwargs: raw tiktoken
            differs += 1
        cases.append((text, ids, tok.decode(ids)))
    if differs:
        print(f"{differs} cases: encode() != encode(add_special_tokens=False)", file=sys.stderr)
        return 1

    name = model_path.rstrip("/").split("/")[-1]
    lines = [
        "// GENERATED by tests/refs/gen_tokenizer_golden.py - do not edit.",
        "//",
        f"// Reference: transformers AutoTokenizer.from_pretrained({name!r}),",
        "// encode(add_special_tokens=False) and decode() over the corpus in the",
        "// generator. The test refuses a model whose fingerprint differs rather",
        "// than comparing against the wrong reference (#1569, #1570).",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string>",
        "#include <vector>",
        "",
        f"namespace {ns} {{",
        "",
        "struct Case {",
        "    const char* text;",
        "    std::vector<int32_t> ids;",
        "    const char* decoded;",
        "};",
        "",
        f'inline constexpr const char* k_model = "{name}";',
        f"inline constexpr int k_vocab_size = {tok.vocab_size};",
        f"inline constexpr int k_n_tokens = {len(tok)};",
        "",
        "// Fingerprint: if a model tokenizes this differently it is not the",
        "// tokenizer this golden describes, and the comparison would be noise.",
        f'inline constexpr const char* k_fingerprint_text = "{c_escape(TEST_STRINGS[0])}";',
        "inline const std::vector<int32_t> k_fingerprint_ids = {"
        + ", ".join(str(i) for i in cases[0][1])
        + "};",
        "",
        "// clang-format off: one case per line, as generated",
        "inline const std::vector<Case>& cases() {",
        "    static const std::vector<Case> v = {",
    ]
    for text, ids, decoded in cases:
        lines.append(
            '        {"%s", {%s}, "%s"},'
            % (c_escape(text), ", ".join(str(i) for i in ids), c_escape(decoded))
        )
    lines += [
        "    };",
        "    return v;",
        "}",
        "// clang-format on",
        "",
        f"}}  // namespace {ns}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"{len(cases)} cases -> {out_path}  (vocab_size={tok.vocab_size}, n_tokens={len(tok)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

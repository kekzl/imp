#!/usr/bin/env python3
"""Generate golden tokenizer output for cross-implementation testing.

Usage: python tests/generate_tokenizer_golden.py <model_path> <output_json>

Requires: pip install transformers

Generates a JSON file with test cases:
{
    "model": "meta-llama/Llama-3-8B",
    "cases": [
        {"text": "Hello, world!", "ids": [1, 15043, 29892, 3186, 29991]},
        {"text": "The quick brown fox", "ids": [1, 450, 4996, 17354, 266]},
        ...
    ]
}
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
    "",  # empty string
    " ",  # single space
    "a",  # single char
    "A" * 100,  # repeated char
]

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_path> <output_json>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    tok = AutoTokenizer.from_pretrained(model_path)

    cases = []
    for text in TEST_STRINGS:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": decoded,
        })

    result = {
        "model": model_path,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "cases": cases,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cases)} test cases -> {output_path}")

if __name__ == "__main__":
    main()

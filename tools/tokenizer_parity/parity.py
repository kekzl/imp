#!/usr/bin/env python3
"""HF side of the tokenizer parity harness (run.sh). Runs in python:3.12-slim.

  parity.py corpus <out.bin>                  write the corpus (records separated by 0x1E), print sha256
  parity.py hf <tokenizer.json>               HF ids per record, add_special_tokens=False
  parity.py chat <model_dir>                  transformers apply_chat_template ids per conversation
  parity.py compare <imp.txt> <hf.txt> [tok]  summary line; with tokenizer.json, print the diffs
"""
import hashlib
import random
import sys

SEED = 7

LITERALS = [
    "Hello world", " Hello", "  two leading", "trailing  ", "\n\nnewlines\n", "\t tab\tmix",
    "don't I'll we're THEY'RE", "123456789 1234 12.5e-3 3,141", "a  b   c    d",
    "café naïve Straße Œuvre", "é combining", "Ａｂｃ fullwidth",
    "日本語のテキスト", "中文测试，标点。",
    "한국어 문장", "Привет, мир!",
    "مرحبا بالعالم", "עברית",
    "हिन्दी पाठ", "ภาษาไทย",
    "emoji \U0001F600\U0001F44D\U0001F3FD\U0001F468‍\U0001F469‍\U0001F467 \U0001F1E9\U0001F1EA",
    "math ∑∫√∞ ≤≥", "code: def f(x):\n    return x**2  # sq\n",
    "<think>\nreasoning\n</think>\n\nanswer", "<|im_start|>user\nhi<|im_end|>\n",
    "<|endoftext|>", "<start_of_turn>user\nhi<end_of_turn>",
    "<|start|>assistant<|channel|>final<|message|>ok",
    "{\"key\": [1, 2, {\"n\": null}]}", "http://example.com/path?q=1&b=2#frag",
    "   ", " ", "\n", "\r\n windows \r\n", "x" * 300, "ab" * 200, " nbsp thin​zw",
    "a\u0000b", "� replacement", "\U0001F600" * 5, "  \n  \n\t\t", "!!!???...,,,",
    "Mixed123CASEWithNumbers456", "snake_case_identifier and camelCaseIdentifier",
    "The quick brown fox jumps over the lazy dog. " * 3,
    "’curly’ “quotes” — dash – en",
]

POOL_SMALL = "abcXYZ 019\n\t.,!?'\"-_éüß日本\U0001F600ʼ́ <>|/"
POOL_WIDE = list(POOL_SMALL) + list(
    "  　​，。²١中한각"
    "किאְกิÀÉǅʰΩÅ"
    "̣̀̈̂½Ⅰàè\r\f\v\u0085 "
    "\U0001F44D\U0001F3FD‍️­’“—ABCdefGHI 0123456789")

CONVS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are helpful."},
     {"role": "user", "content": "Hi \U0001F600,\n what's 2+2?"},
     {"role": "assistant", "content": "It is 4."},
     {"role": "user", "content": "  and\n\n5+5? "}],
    [{"role": "user", "content": "Übersetze: naïve café — ‘quotes’"}],
]


def corpus():
    rnd = random.Random(SEED)
    texts = list(LITERALS)
    texts += ["".join(rnd.choice(POOL_SMALL) for _ in range(rnd.randint(1, 40))) for _ in range(150)]
    texts += ["".join(rnd.choice(POOL_WIDE) for _ in range(rnd.randint(1, 60))) for _ in range(1000)]
    return texts


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    mode = sys.argv[1]
    if mode == "corpus":
        blob = "".join(t + "\x1e" for t in corpus()).encode("utf-8")
        with open(sys.argv[2], "wb") as f:
            f.write(blob)
        print(f"corpus: {len(corpus())} records, sha256 {hashlib.sha256(blob).hexdigest()[:16]}")
    elif mode == "hf":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(sys.argv[2])
        for t in corpus():
            print(" ".join(str(i) for i in tok.encode(t, add_special_tokens=False).ids))
    elif mode == "chat":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(sys.argv[2])
        for c in CONVS:
            ids = tok.apply_chat_template(c, add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            print(" ".join(str(i) for i in ids))
    elif mode == "compare":
        imp = [line.split(" ", 1) + [""] for line in read_lines(sys.argv[2])]
        hf = read_lines(sys.argv[3])
        if len(imp) != len(hf):
            print(f"record count differs: imp {len(imp)}, hf {len(hf)}")
            sys.exit(1)
        texts = corpus() if len(hf) == len(corpus()) else None
        bad = [k for k, (a, b) in enumerate(zip(imp, hf)) if a[1].strip() != b.strip()]
        print(f"mismatch {len(bad)}/{len(hf)}")
        if len(sys.argv) > 4 and texts:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(sys.argv[4])
            for k in bad[:20]:
                print(f"  #{k} {texts[k]!r}")
                print("    imp:", [tok.id_to_token(int(x)) for x in imp[k][1].split()])
                print("    hf :", [tok.id_to_token(int(x)) for x in hf[k].split()])


if __name__ == "__main__":
    main()

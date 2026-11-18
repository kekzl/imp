#!/usr/bin/env python3
"""Token ids of a text with a model's tokenizer.json, no BOS, space-separated: the --ids input of
flint_reference.py / flint_quant_eval.py; OUT_FILE.text holds the decoded ids for imp-cli --perplexity.
Usage: flint_tokenize.py MODEL_DIR TEXT_FILE OUT_FILE [MAX_TOKENS]"""
import pathlib
import sys

from tokenizers import Tokenizer

model, text, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else None
tok = Tokenizer.from_file(str(pathlib.Path(model) / "tokenizer.json"))
ids = tok.encode(pathlib.Path(text).read_text(), add_special_tokens=False).ids[:n]
pathlib.Path(out).write_text(" ".join(map(str, ids)))
pathlib.Path(out + ".text").write_text(tok.decode(ids, skip_special_tokens=False))
print(f"{len(ids)} ids -> {out}")

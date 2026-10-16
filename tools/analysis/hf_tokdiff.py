"""imp /tokenize ids vs the HF tokenizer on one text (scripts/check_model.sh step "tokenizer").

Usage (hfref:cpu image, see hf_ppl_cpu.py): python -I hf_tokdiff.py <model_dir> <imp_tokenize.json> <text>
Prints "hf N imp M", then "identical" or the first differing position with both token pieces.
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
imp = json.load(open(sys.argv[2]))["tokens"]
imp = [t["id"] if isinstance(t, dict) else t for t in imp]
text = open(sys.argv[3], encoding="utf-8").read()
hf = tok(text, add_special_tokens=False).input_ids
print(f"hf {len(hf)} imp {len(imp)}")
i = next((k for k in range(min(len(hf), len(imp))) if hf[k] != imp[k]), None)
if i is None and len(hf) == len(imp):
    print("identical")
    sys.exit(0)
i = min(len(hf), len(imp)) if i is None else i
print(f"first diff at {i}")
print("ctx :", repr(tok.decode(hf[max(0, i - 8):i])))
print("hf  :", [tok.decode([t]) for t in hf[i:i + 10]])
print("imp :", [tok.decode([t]) for t in imp[i:i + 10]])
sys.exit(1)

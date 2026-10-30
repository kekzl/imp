"""HF reference embeddings for Qwen/Qwen3-Embedding-0.6B (golden: qwen3_embedding_ref.json).

Model-card recipe: AutoModel fp32 on CPU, last_hidden_state of the LAST token (the tokenizer's
post-processor appends <|endoftext|>), L2-normalised. Consumed by tools/analysis/embed_check.py
(`make test-embed`), which asserts cos >= 0.9999 per text against imp's /v1/embeddings.
Generated 2026-10-30 with the versions written into the JSON (`torch`, `transformers`).

  docker run --rm -v ~/models/Qwen3-Embedding-0.6B:/m:ro -v $PWD/tests/refs:/r python:3.13-slim sh -c \
    'pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q transformers \
     && python -I /r/gen_qwen3_embedding_ref.py /m /r/qwen3_embedding_ref.json'
"""
import json
import sys

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

TEXTS = [
    "The capital of France is Paris.",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: What is the capital of China?",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
    "objects and is responsible for the movement of planets around the sun.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "def add(a, b):\n    return a + b",
]

model_dir, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
rows = []
with torch.no_grad():
    for t in TEXTS:
        b = tok([t], return_tensors="pt")
        e = F.normalize(model(**b).last_hidden_state[:, -1], p=2, dim=1)[0]
        rows.append({"text": t, "n_tokens": int(b["input_ids"].shape[1]),
                     "embedding": [round(float(v), 6) for v in e]})
out = {"model": "Qwen/Qwen3-Embedding-0.6B", "torch": torch.__version__,
       "transformers": transformers.__version__, "rows": rows}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    f.write("\n")
print("wrote", out_path, len(rows))

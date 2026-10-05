"""BF16 reference perplexity on the CPU for checkpoints that do not fit the card (roadmap row 102).

One forward over the whole text, mean NLL of tokens 1..n-1 (the count imp-cli --perplexity uses).
Runs in a throwaway container, never on the host:
  docker build -t hfref:cpu - <<< $'FROM python:3.12-slim\\nRUN pip install --no-cache-dir torch \\
      --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir transformers safetensors'
  docker run --rm -v "$MODELS_DIR:/models:ro" -v "$PWD/tools/analysis:/a:ro" hfref:cpu \\
      python /a/hf_ppl_cpu.py /models/Qwen3-14B /a/ppl_corpus_45k.txt
Qwen3-14B, 13 537 tokens: 196 s on 16 cores, 28 GB host RAM; Qwen3.8-27B (GDN, reference torch kernels): 477 s, 54 GB.
"""
import math
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

model_dir, text_file = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count() or 1)
tok = AutoTokenizer.from_pretrained(model_dir)
with open(text_file, encoding="utf-8") as f:
    text = f.read()
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
print(f"tokens {ids.shape[1]}", flush=True)
# A VL wrapper (Qwen3.8 ships Qwen3_5ForConditionalGeneration) loads as image-text-to-text; the text
# decoder and LM head are read through the generic accessors either way.
arch = (AutoConfig.from_pretrained(model_dir).architectures or [""])[0]
cls = AutoModelForImageTextToText if arch.endswith("ConditionalGeneration") else AutoModelForCausalLM
model = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()
t0 = time.time()
nll_sum, count = 0.0, 0
with torch.inference_mode():
    hidden = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state[0]
    head = model.get_output_embeddings()
    # LM head in 1024-row slices: a full [n, vocab] FP32 logit matrix is 8 GB at 13.5k tokens.
    for s in range(0, hidden.shape[0] - 1, 1024):
        e = min(s + 1024, hidden.shape[0] - 1)
        logits = head(hidden[s:e]).float()
        nll_sum += torch.nn.functional.cross_entropy(logits, ids[0, s + 1 : e + 1], reduction="sum").item()
        count += e - s
print(f"counted {count} mean_nll {nll_sum / count:.6f} PPL {math.exp(nll_sum / count):.4f} "
      f"({time.time() - t0:.0f} s)", flush=True)

#!/usr/bin/env python3
"""Reference logits for tests/test_inkling_quality.cu (InklingQuality.KldAgainstTheReference).

transformers' Inkling2-Loop on the CPU (BF16, experts dequantized) over wikitext-2 test, llama-perplexity
style: consecutive chunks of 512 tokens, rows 256..511 kept. Writes ids.json and chunk_NN_logits.npy
(fp32 [256, vocab], 205 MB each). 20 chunks: 8 s a chunk on 16 cores, 66 GiB peak RSS, reference PPL 7.6538.

  python3 -I inkling_kld_oracle.py <checkpoint dir> <wiki.test.raw> <out dir> <chunks>

Image: python 3.12, torch CPU, transformers main (5.24.0.dev0), HF_HUB_OFFLINE=1. Text:
https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
"""
import sys, time, json, os, resource
import numpy as np, torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
mid, text_path, out, n_chunks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CH, KEEP = 512, 256
os.makedirs(out, exist_ok=True)
proc = AutoProcessor.from_pretrained(mid)
tok = proc.tokenizer
text = open(text_path, encoding='utf-8').read()
ids = tok(text, add_special_tokens=False)['input_ids']
print('tokens', len(ids), flush=True)
chunks = [ids[i * CH:(i + 1) * CH] for i in range(n_chunks)]
json.dump({'chunk': CH, 'keep_from': KEEP, 'chunks': chunks}, open(f'{out}/ids.json', 'w'))
t0 = time.time()
model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print('loaded in', round(time.time() - t0, 1), 's', flush=True)
tot_nll, tot_n = 0.0, 0
for c, ch in enumerate(chunks):
    path = f'{out}/chunk_{c:02d}_logits.npy'
    if os.path.exists(path):
        continue
    t1 = time.time()
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ch])).logits[0].float()
    keep = lg[KEEP:]
    np.save(path + '.tmp.npy', keep.numpy())
    os.replace(path + '.tmp.npy', path)
    lp = torch.log_softmax(keep[:-1].double(), dim=-1)
    tgt = torch.tensor(ch[KEEP + 1:])
    nll = -lp[torch.arange(len(tgt)), tgt].sum().item()
    tot_nll += nll; tot_n += len(tgt)
    print('chunk', c, 'shape', tuple(keep.shape), round(time.time() - t1, 1), 's', 'nll/tok', round(nll / len(tgt), 4), 'running ppl', round(float(np.exp(tot_nll / tot_n)), 4), flush=True)
print('DONE maxrss GiB', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1), flush=True)

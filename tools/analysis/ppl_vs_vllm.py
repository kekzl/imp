#!/usr/bin/env python3
"""imp vs vLLM teacher-forced perplexity on identical token ids and rows (GOAL release bar 1).

Runs in containers only (tools/analysis/ppl_vs_vllm.sh): `vllm` and `hf-tokens` in
vllm/vllm-openai, `parity` and `compare` need the stdlib only.

  ppl_vs_vllm.py hf-tokens <model_dir> <corpus> <out>    HF ids, add_special_tokens=True (vLLM's text path)
  ppl_vs_vllm.py parity <imp_tokens> <hf_tokens>          exit 1 on any id difference
  ppl_vs_vllm.py vllm <model_dir> <tokens> <out_nll> [<tokens> <out_nll>...] [opts]  prompt_logprobs=0
  ppl_vs_vllm.py compare <tokens> <imp_nll> <vllm_nll> [--first F] [--last L] [--json OUT]

Row i is the NLL of token i+1 given tokens 0..i, rows 0..n-2: imp `diagnostics.ppl_dump=<path>`
writes "<row> <nll>", `vllm` writes "<row> <nll> <rank>". Token files: imp's
`diagnostics.dump_tokens` stderr ("TOK <i> <id>", other lines ignored) or one id per line.
"""
import argparse
import json
import math
import sys

BAR_PCT = 0.5  # docs/GOAL.md release bar 1


def read_tokens(path):
    ids, plain = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0] == "TOK":
                if int(p[1]) != len(ids):
                    sys.exit(f"{path}: TOK index {p[1]} out of order at {len(ids)}")
                ids.append(int(p[2]))
            elif len(p) == 1 and p[0].lstrip("-").isdigit():
                plain.append(int(p[0]))
    return ids or plain


def read_nll(path):
    nll, rank = {}, {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[0].isdigit():
                nll[int(p[0])] = float(p[1])
                if len(p) >= 3:
                    rank[int(p[0])] = int(p[2])
    return nll, rank


def cmd_hf_tokens(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    with open(a.corpus, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, add_special_tokens=True)["input_ids"]
    with open(a.out, "w") as f:
        f.write("".join(f"{t}\n" for t in ids))
    print(f"hf_tokens: n={len(ids)} bos={tok.bos_token_id} first={ids[:4]} rt={tok.decode(ids, skip_special_tokens=False) == text}")


def cmd_parity(a):
    x, y = read_tokens(a.imp_tokens), read_tokens(a.hf_tokens)
    diff = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    if diff is None and len(x) == len(y):
        print(f"parity: IDENTICAL n={len(x)}")
        return 0
    if x[1:] == y or y[1:] == x:  # imp-cli prepends BOS when the tokenizer adds one; HF may not
        print(f"parity: BOS-ONLY imp n={len(x)} hf n={len(y)} leading {'imp' if len(x) > len(y) else 'hf'} id "
              f"{(x if len(x) > len(y) else y)[0]}, rest identical")
        return 0
    at = diff if diff is not None else min(len(x), len(y))
    print(f"parity: DIFF imp n={len(x)} hf n={len(y)} first_diff={at}")
    print(f"  imp[{max(0, at - 3)}:{at + 5}] = {x[max(0, at - 3):at + 5]}")
    print(f"  hf [{max(0, at - 3)}:{at + 5}] = {y[max(0, at - 3):at + 5]}")
    return 1


def cmd_vllm(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if len(a.pairs) % 2:
        sys.exit("vllm: expected <tokens> <out_nll> pairs")
    jobs = [(read_tokens(a.pairs[k]), a.pairs[k + 1]) for k in range(0, len(a.pairs), 2)]
    kw = dict(
        model=a.model_dir,
        max_model_len=max(len(ids) for ids, _ in jobs) + 16,
        max_num_seqs=1,
        max_num_batched_tokens=a.chunk,
        enable_prefix_caching=False,
        enforce_eager=a.enforce_eager,
        kv_cache_dtype=a.kv_cache_dtype,
        gpu_memory_utilization=a.gpu_mem,
        seed=0,
    )
    if a.language_model_only:
        kw["language_model_only"] = True
    print(f"vllm_args: {json.dumps(kw)}", flush=True)
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    for ids, out_nll in jobs:  # one request at a time, prefix caching off: every row is computed
        n = len(ids)
        pl = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].prompt_logprobs
        if len(pl) != n:
            sys.exit(f"vllm returned {len(pl)} prompt logprob rows for {n} tokens")
        total, top1 = 0.0, 0
        with open(out_nll, "w") as f:
            for i in range(n - 1):
                lp = pl[i + 1][ids[i + 1]]
                total += -lp.logprob
                top1 += lp.rank == 1
                f.write(f"{i} {-lp.logprob:.6f} {lp.rank}\n")
        print(f"vllm_ppl: {math.exp(total / (n - 1)):.4f} rows={n - 1} top1={top1}/{n - 1} out={out_nll}", flush=True)


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def cmd_compare(a):
    ids = read_tokens(a.tokens)
    n = len(ids)
    ni, _ = read_nll(a.imp_nll)
    nv, rv = read_nll(a.vllm_nll)
    last = n - 2 if a.last < 0 else min(a.last, n - 2)
    rows = range(max(0, a.first), last + 1)
    miss = [i for i in rows if i not in ni or i not in nv]
    if miss:
        sys.exit(f"missing rows: {len(miss)} (first {miss[:5]}); imp has {len(ni)}, vllm {len(nv)}, n={n}")
    xi = [ni[i] for i in rows]
    xv = [nv[i] for i in rows]
    d = [p - q for p, q in zip(xi, xv)]
    c = len(d)
    ppl_i, ppl_v = math.exp(sum(xi) / c), math.exp(sum(xv) / c)
    delta = 100.0 * (ppl_i / ppl_v - 1.0)
    mi, mv, md = sum(xi) / c, sum(xv) / c, sum(d) / c
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, c - 1))
    cov = sum((p - mi) * (q - mv) for p, q in zip(xi, xv))
    r = cov / math.sqrt(sum((p - mi) ** 2 for p in xi) * sum((q - mv) ** 2 for q in xv))
    ad = [abs(x) for x in d]
    res = {
        "n_tokens": n, "first": rows.start, "last": last, "rows": c,
        "ppl_imp": round(ppl_i, 4), "ppl_vllm": round(ppl_v, 4), "delta_pct": round(delta, 3),
        "bar": "PASS" if abs(delta) <= BAR_PCT else "FAIL",
        "mean_d": round(md, 5), "sd_d": round(sd, 5), "pearson_r": round(r, 6),
        "abs_d_p50": round(pct(ad, 0.5), 5), "abs_d_p90": round(pct(ad, 0.9), 5),
        "abs_d_p99": round(pct(ad, 0.99), 5), "abs_d_max": round(max(ad), 5),
        "frac_abs_d_gt_0.1": round(sum(x > 0.1 for x in ad) / c, 5),
        "frac_abs_d_gt_0.5": round(sum(x > 0.5 for x in ad) / c, 5),
        "sum_pos_d": round(sum(x for x in d if x > 0), 4), "sum_neg_d": round(sum(x for x in d if x < 0), 4),
    }
    if rv:
        res["vllm_top1"] = round(sum(rv.get(i) == 1 for i in rows) / c, 5)
    seg = a.segment
    res["segments"] = []
    for s in range(rows.start, last + 1, seg):
        e = min(s + seg, last + 1)
        si, sv = sum(ni[i] for i in range(s, e)), sum(nv[i] for i in range(s, e))
        res["segments"].append({"rows": [s, e - 1], "ppl_imp": round(math.exp(si / (e - s)), 4),
                                "ppl_vllm": round(math.exp(sv / (e - s)), 4)})
    top = sorted(rows, key=lambda i: -abs(ni[i] - nv[i]))[: a.top]
    res["top_abs_d"] = [{"row": i, "next_id": ids[i + 1], "imp": ni[i], "vllm": nv[i]} for i in top]
    print(f"PPL imp={ppl_i:.4f} vllm={ppl_v:.4f} delta={delta:+.3f}% bar(+-{BAR_PCT}%)={res['bar']} "
          f"rows=[{rows.start},{last}] n={c}")
    print(f"per-row d=imp-vllm: mean {md:+.5f} sd {sd:.5f} r {r:.6f} |d| p50 {res['abs_d_p50']} "
          f"p90 {res['abs_d_p90']} p99 {res['abs_d_p99']} max {res['abs_d_max']} "
          f">0.1 {res['frac_abs_d_gt_0.1']} >0.5 {res['frac_abs_d_gt_0.5']}")
    for s in res["segments"]:
        print(f"  seg rows {s['rows'][0]:>6}-{s['rows'][1]:<6} imp {s['ppl_imp']:>9.4f} vllm {s['ppl_vllm']:>9.4f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hf-tokens")
    p.add_argument("model_dir")
    p.add_argument("corpus")
    p.add_argument("out")
    p = sub.add_parser("parity")
    p.add_argument("imp_tokens")
    p.add_argument("hf_tokens")
    p = sub.add_parser("vllm")
    p.add_argument("model_dir")
    p.add_argument("pairs", nargs="+", help="<tokens> <out_nll> [<tokens> <out_nll> ...], one engine")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--chunk", type=int, default=2048, help="max_num_batched_tokens (prompt logprobs are chunk x vocab fp32)")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--language-model-only", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("tokens")
    p.add_argument("imp_nll")
    p.add_argument("vllm_nll")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=-1)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json")
    a = ap.parse_args()
    rc = {"hf-tokens": cmd_hf_tokens, "parity": cmd_parity, "vllm": cmd_vllm, "compare": cmd_compare}[a.cmd](a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()

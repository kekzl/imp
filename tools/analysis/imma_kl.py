"""imp prompt_logprobs vs HF fp32 reference for #2256 (scripts/measure_2256.sh).

ref: HF transformers fp32 on CPU, GGUF dequantized via gguf_file=, top-K + target logprob per position.
stats: KL(HF || imp) on S = HF top-K ids that imp reports, both renormalized over S; stdlib only.
"""
import argparse
import hashlib
import json
import math
import sys
import time


def cmd_ref(a):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = open(a.corpus, "rb").read()
    tok = AutoTokenizer.from_pretrained(a.models, gguf_file=a.gguf)
    ids = tok(raw.decode("utf-8"), add_special_tokens=False)["input_ids"][: a.n]
    assert len(ids) == a.n, (len(ids), a.n)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.models, gguf_file=a.gguf, dtype=torch.float32)
    model.eval()
    t_load = time.time() - t0
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    top_lp, top_id = torch.topk(lp[:-1], a.topk, dim=-1)
    tgt = lp[torch.arange(a.n - 1), torch.tensor(ids[1:])]
    out = {
        "gguf": a.gguf,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "n": a.n,
        "topk": a.topk,
        "vocab": int(lp.shape[1]),
        "ids": ids,
        # row p predicts ids[p + 1]; imp prompt_logprobs entry p + 1
        "target_logprobs": tgt.tolist(),
        "top_ids": top_id.tolist(),
        "top_logprobs": top_lp.tolist(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "load_s": round(t_load, 1),
        "forward_s": round(time.time() - t0 - t_load, 1),
    }
    json.dump(out, open(a.out, "w"))
    print("ref: %d ids, vocab %d, load %.1f s, forward %.1f s, first target logprobs %s"
          % (a.n, out["vocab"], out["load_s"], out["forward_s"], out["target_logprobs"][:3]))


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def kl_on_support(hf_ids, hf_lps, imp):
    """KL(p || q) over S = [i in hf_ids if str(i) in imp], p, q renormalized over S.

    Returns (kl, hf_mass_of_S, |S|). imp: {"<id>": {"logprob": x, ...}}.
    """
    s = [(lp, imp[str(i)]["logprob"]) for i, lp in zip(hf_ids, hf_lps) if str(i) in imp]
    zp = lse([p for p, _ in s])
    zq = lse([q for _, q in s])
    kl = sum(math.exp(p - zp) * ((p - zp) - (q - zq)) for p, q in s)
    return max(kl, 0.0), sum(math.exp(p) for p, _ in s), len(s)


def pct(xs, q):
    """Nearest-rank percentile."""
    s = sorted(xs)
    return s[max(0, math.ceil(q / 100.0 * len(s)) - 1)]


def compare(ref, resp):
    ids = ref["ids"]
    plp = resp["choices"][0]["prompt_logprobs"]
    assert len(plp) == len(ids) and plp[0] is None, (len(plp), len(ids))
    kls, diffs, cov, imp_tgt, top1 = [], [], [], [], 0
    for p in range(len(ids) - 1):
        e = plp[p + 1]
        t = e[str(ids[p + 1])]["logprob"]
        kl, mass, _ = kl_on_support(ref["top_ids"][p], ref["top_logprobs"][p], e)
        kls.append(kl)
        cov.append(mass)
        diffs.append(abs(t - ref["target_logprobs"][p]))
        imp_tgt.append(t)
        imp_argmax = max(e.items(), key=lambda kv: kv[1]["logprob"])[0]
        top1 += imp_argmax == str(ref["top_ids"][p][0])
    n = len(kls)

    def summ(xs):
        return {"mean": sum(xs) / n, "p50": pct(xs, 50), "p99": pct(xs, 99), "max": max(xs)}

    return {
        "n": n,
        "kl": summ(kls),
        "absdiff": summ(diffs),
        "top1_pct": 100.0 * top1 / n,
        "ppl_imp": math.exp(-sum(imp_tgt) / n),
        "ppl_hf": math.exp(-sum(ref["target_logprobs"]) / n),
        "cov_mean": sum(cov) / n,
        "cov_min": min(cov),
    }


def cmd_stats(a):
    ref = json.load(open(a.ref))
    out = {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        out[name] = compare(ref, json.load(open(path)))
    json.dump(out, sys.stdout)
    print()


def cmd_selftest(_):
    # p = softmax(ln .5, ln .3, ln .2) over ids 1,2,3; imp reports 1,2 only (+ id 9 not in HF top)
    # S = {1,2}: p_S = (.625, .375); q = (.4, .4) -> q_S = (.5, .5)
    # KL = .625 ln(.625/.5) + .375 ln(.375/.5) = 0.031584; HF mass of S = 0.8
    hf_ids, hf_lps = [1, 2, 3], [math.log(0.5), math.log(0.3), math.log(0.2)]
    imp = {"1": {"logprob": math.log(0.4)}, "2": {"logprob": math.log(0.4)}, "9": {"logprob": math.log(0.1)}}
    exp_kl = 0.625 * math.log(0.625 / 0.5) + 0.375 * math.log(0.375 / 0.5)
    kl, mass, ns = kl_on_support(hf_ids, hf_lps, imp)
    print("selftest kl: expected %.6f (hand 0.031584) computed %.6f, mass %.6f (0.8), |S| %d (2)"
          % (exp_kl, kl, mass, ns))
    ok = abs(kl - 0.031584) < 1e-6 and abs(mass - 0.8) < 1e-12 and ns == 2
    # compare(): 3 ids -> 2 positions; targets 2 (pos 0), 3 (pos 1)
    # absdiff: |ln .4 - ln .3| = 0.287682, |ln .05 - ln .2| = 1.386294; mean 0.836988
    # top1: pos0 imp argmax id 1 (tie .4/.4 -> first) == HF 1; pos1 imp argmax id 2 != HF 1 -> 50 %
    # ppl_hf = exp(-(ln .3 + ln .2) / 2) = 4.082483; ppl_imp = exp(-(ln .4 + ln .05) / 2) = 7.071068
    ref = {"ids": [0, 2, 3], "target_logprobs": [math.log(0.3), math.log(0.2)],
           "top_ids": [hf_ids, hf_ids], "top_logprobs": [hf_lps, hf_lps]}
    resp = {"choices": [{"prompt_logprobs": [
        None, imp,
        {"1": {"logprob": math.log(0.3)}, "2": {"logprob": math.log(0.6)}, "3": {"logprob": math.log(0.05)}}]}]}
    r = compare(ref, resp)
    print("selftest absdiff mean: expected 0.836988 computed %.6f; max: expected 1.386294 computed %.6f"
          % (r["absdiff"]["mean"], r["absdiff"]["max"]))
    print("selftest top1: expected 50.0 computed %.1f; ppl_hf: expected 4.082483 computed %.6f; "
          "ppl_imp: expected 7.071068 computed %.6f" % (r["top1_pct"], r["ppl_hf"], r["ppl_imp"]))
    ok &= abs(r["absdiff"]["mean"] - 0.836988) < 1e-6 and abs(r["absdiff"]["max"] - 1.386294) < 1e-6
    ok &= r["top1_pct"] == 50.0 and abs(r["ppl_hf"] - 4.082483) < 1e-6 and abs(r["ppl_imp"] - 7.071068) < 1e-6
    print("selftest " + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--models", default="/models")
    r.add_argument("--gguf", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--n", type=int, default=2048)
    r.add_argument("--topk", type=int, default=64)
    r.add_argument("--out", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--ref", required=True)
    s.add_argument("--arm", action="append", required=True, help="NAME=imp_response.json")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"ref": cmd_ref, "stats": cmd_stats, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()

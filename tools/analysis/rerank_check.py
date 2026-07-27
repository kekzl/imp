#!/usr/bin/env python3
"""Rerank gate for imp-server (/v1/rerank, roadmap gap 9).

Asserts the contract a RAG client depends on, against a RUNNING imp-server with
a cross-encoder reranker loaded:

  contract   shape, sorting, top_n, aliases, error codes
  semantics  the relevant document actually wins, and by a margin
  stability  once warm, identical requests score identically, and a
             document's score does not depend on where it sat in the input
             list. The FIRST call after load is a hair different (~1e-3): a
             cold prefill and one reusing cached prefix blocks are not the same
             arithmetic. Ordering is unaffected.

Optionally cross-checks against another OpenAI/Cohere-style reranking server
(llama.cpp `--reranking` serves /rerank with the same shape) on the SAME model
file, which is the only way to tell "our scores are self-consistent" from "our
scores are right".

Stdlib-only, mirrors tools/analysis/degen_suite.py conventions.

Usage:
  python3 tools/analysis/rerank_check.py [--url http://localhost:8080]
                                         [--compare http://localhost:8081]
Exit codes: 0 = clean, 1 = failures, 2 = server unreachable.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

QUERY = "What is the capital of France?"
DOCS = [
    "The mitochondrion is the powerhouse of the cell.",
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "France is a country in Western Europe known for its cuisine.",
]
RELEVANT = 1  # index of the document that answers QUERY

FAILS = []


def record(name, ok, detail=""):
    mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  [{mark}] {name}" + ("" if ok else f": {detail}"))
    if not ok:
        FAILS.append(name)


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except ValueError:
            return e.code, {}


def scores_by_index(payload):
    return {r["index"]: r["relevance_score"] for r in payload["results"]}


def run(url, compare):
    print("== contract ==")
    st, d = post(url, "/v1/rerank", {"query": QUERY, "documents": DOCS})
    if st != 200:
        print(f"rerank returned {st}: {json.dumps(d)[:200]}")
        return 2
    order = [r["index"] for r in d["results"]]
    record("results are sorted by descending score",
           all(d["results"][i]["relevance_score"] >= d["results"][i + 1]["relevance_score"]
               for i in range(len(d["results"]) - 1)), f"order={order}")
    record("every index appears exactly once",
           sorted(order) == list(range(len(DOCS))), f"order={order}")
    record("scores are probabilities in [0,1]",
           all(0.0 <= r["relevance_score"] <= 1.0 for r in d["results"]))
    record("documents are omitted unless asked for", "document" not in d["results"][0])

    st2, d2 = post(url, "/v1/rerank", {"query": QUERY, "documents": DOCS, "return_documents": True,
                                       "top_n": 2})
    record("top_n truncates and return_documents echoes",
           st2 == 200 and len(d2["results"]) == 2 and
           d2["results"][0]["document"]["text"] == DOCS[d2["results"][0]["index"]],
           f"status={st2}")

    st3, d3 = post(url, "/rerank", {"query": QUERY, "documents": DOCS})
    record("unversioned /rerank alias serves the same thing",
           st3 == 200 and [r["index"] for r in d3["results"]] == order, f"status={st3}")

    st4, _ = post(url, "/v1/rerank", {"query": QUERY,
                                      "documents": [{"text": DOCS[RELEVANT]}]})
    record("Cohere object-form documents accepted", st4 == 200, f"status={st4}")

    for name, body in [("missing query", {"documents": DOCS}),
                       ("empty documents", {"query": QUERY, "documents": []}),
                       ("no documents field", {"query": QUERY})]:
        stx, _ = post(url, "/v1/rerank", body)
        record(f"{name} -> 400", stx == 400, f"status={stx}")

    print("== semantics ==")
    s = scores_by_index(d)
    record("the relevant document ranks first", order[0] == RELEVANT,
           f"first={order[0]} ({DOCS[order[0]][:40]!r})")
    margin = s[RELEVANT] - max(v for k, v in s.items() if k != RELEVANT)
    record("it wins by a clear margin (>0.5)", margin > 0.5, f"margin={margin:.4f}")

    print("== stability ==")
    # The FIRST call after load populates the prefix cache, and a cold prefill
    # is numerically a hair different from one that reuses cached blocks (~1e-3
    # on a 0.99 score — the same order as the gap between two engines running
    # the same weights). Ordering is unaffected, so the contract is "stable
    # once warm", not "bit-identical from the first request": measure it that
    # way rather than pretending otherwise.
    cold = s
    _, warm1 = post(url, "/v1/rerank", {"query": QUERY, "documents": DOCS})
    _, warm2 = post(url, "/v1/rerank", {"query": QUERY, "documents": DOCS})
    w1, w2 = scores_by_index(warm1), scores_by_index(warm2)
    record("identical requests score identically once warm", w1 == w2,
           f"{w1} vs {w2}")
    cold_delta = max(abs(cold[i] - w1[i]) for i in cold)
    record("the cold-cache score differs only marginally (<0.01)", cold_delta < 0.01,
           f"max delta={cold_delta:.5f}")
    record("cold and warm agree on the ordering",
           [i for i, _ in sorted(cold.items(), key=lambda kv: -kv[1])] ==
           [i for i, _ in sorted(w1.items(), key=lambda kv: -kv[1])])

    shuffled = [DOCS[2], DOCS[0], DOCS[3], DOCS[1]]
    _, sh = post(url, "/v1/rerank", {"query": QUERY, "documents": shuffled})
    by_text_a = {DOCS[k]: round(v, 4) for k, v in w1.items()}
    by_text_b = {shuffled[k]: round(v, 4) for k, v in scores_by_index(sh).items()}
    record("a document's score does not depend on its position in the list",
           by_text_a == by_text_b, f"{by_text_a} vs {by_text_b}")

    if compare:
        print("== cross-engine (same model file) ==")
        stc, dc = post(compare, "/rerank", {"query": QUERY, "documents": DOCS})
        if stc != 200:
            record("reference server reachable", False, f"status={stc}")
        else:
            ref = scores_by_index(dc)
            ref_order = [i for i, _ in sorted(ref.items(), key=lambda kv: -kv[1])]
            record("top-1 document agrees with the reference engine",
                   ref_order[0] == order[0], f"imp={order[0]} ref={ref_order[0]}")
            deltas = sorted(abs(s[i] - ref[i]) for i in s)
            med = deltas[len(deltas) // 2]
            record("median per-document score delta < 0.05", med < 0.05,
                   f"median={med:.5f} max={deltas[-1]:.4f}")

    print("=" * 60)
    print(f"rerank_check: {len(FAILS)} FAIL")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--compare", default="",
                    help="optional reference reranking server (e.g. llama.cpp --reranking)")
    args = ap.parse_args()
    try:
        urllib.request.urlopen(args.url + "/health", timeout=30).read()
    except Exception as e:  # noqa: BLE001
        print(f"server unreachable at {args.url}: {e}")
        return 2
    return run(args.url, args.compare)


if __name__ == "__main__":
    sys.exit(main())

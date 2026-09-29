"""make bench-serve (#2202): metric math on synthetic timestamps, seeded prompts, brief table.

No server, no GPU. Client under test: tools/analysis/serving_kpi.py (scripts/bench_serve.sh).
"""
import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tools" / "analysis" / "serving_kpi.py"
_SPEC = importlib.util.spec_from_file_location("serving_kpi", _PATH)
kpi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kpi)


def _rec(t0, ttft_ms, gaps_ms):
    """Request sent at t0, first token after ttft_ms, then one token per gap."""
    stamps = [t0 + ttft_ms / 1e3]
    for g in gaps_ms:
        stamps.append(stamps[-1] + g / 1e3)
    return {"ok": True, "t0": t0, "t_first": stamps[0], "t_end": stamps[-1], "stamps": stamps,
            "prompt_tokens": 10, "completion_tokens": len(stamps), "cached_tokens": 0}


def _level():
    # 100 requests, request k: TTFT k ms (1..100), 4 gaps of k ms, so 5 tokens and E2E 5k ms.
    recs = [_rec(100.0 * k, k, [k] * 4) for k in range(1, 101)]
    return kpi.summarize_level(recs, 10.0, 1e9, 1e9)


def test_ttft_percentiles():
    c = _level()
    assert c["ttft_ms"]["p50"] == pytest.approx(50.5)
    assert c["ttft_ms"]["p99"] == pytest.approx(99.01)


def test_itl_percentiles_pool_every_gap():
    # Gaps: each k in 1..100 four times. Sorted index of p99 = 399 * 0.99 = 395.01.
    itl = _level()["itl_ms"]
    assert itl["n"] == 400
    assert itl["p50"] == pytest.approx(50.5)
    assert itl["p99"] == pytest.approx(99.01)


def test_e2e_percentiles_in_seconds():
    # E2E of request k = k ms TTFT + 4 gaps of k ms = 5k ms.
    e2e = _level()["e2e_s"]
    assert e2e["p50"] == pytest.approx(0.2525)
    assert e2e["p99"] == pytest.approx(0.49505)


def test_throughput_and_errors():
    recs = [_rec(0.0, 10, [10] * 4) for _ in range(9)]
    recs.append({"ok": False, "t0": 0.0, "t_first": None, "t_end": 1.0, "stamps": [],
                 "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
    c = kpi.summarize_level(recs, 2.0, 1e9, 1e9)
    assert c["err"] == 1 and c["ok"] == 9
    assert c["output_tok_s"] == pytest.approx(45 / 2.0)
    assert c["req_s"] == pytest.approx(9 / 2.0)


def test_seeded_prompts_are_reproducible_and_seed_dependent():
    a = [kpi.make_prompt("t", 8, i, 200, seed=0) for i in range(32)]
    b = [kpi.make_prompt("t", 8, i, 200, seed=0) for i in range(32)]
    c = [kpi.make_prompt("t", 8, i, 200, seed=1) for i in range(32)]
    assert a == b
    assert a != c
    assert len(set(a)) == 32  # unique per request: the prefix cache stays cold


def test_brief_table_columns_and_rows():
    lvl = _level()
    res = [{"concurrency": 1, "client": lvl}, {"concurrency": 4, "client": lvl}]
    lines = kpi.brief_markdown(res).splitlines()
    assert lines[0] == "| KPI | c=1 | c=4 |"
    labels = [ln.split("|")[1].strip() for ln in lines[2:]]
    assert labels == ["TTFT p50 / p99 ms", "ITL p50 / p99 ms", "E2E p50 / p99 s",
                      "output tok/s", "req/s", "errors"]
    assert lines[3].split("|")[2].strip() == "50.5 / 99.0"  # ITL row

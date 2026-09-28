"""Per-token ITL of tools/analysis/serving_kpi.py against synthetic SSE streams (no server)."""
import importlib.util
import json
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "serving_kpi",
    pathlib.Path(__file__).resolve().parents[2] / "tools" / "analysis" / "serving_kpi.py")
serving_kpi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serving_kpi)


def _chunk(delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


def _stream(token_s):
    """SSE byte lines plus a clock returning token_s[i] at the i-th token event."""
    lines = [b": keep-alive", _chunk({"role": "assistant"}).encode()]
    for i, _ in enumerate(token_s):
        key = "reasoning_content" if i % 2 else "content"
        lines += [_chunk({key: f"t{i}"}).encode(), _chunk({"content": ""}).encode()]
    usage = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": len(token_s)}}
    lines += [b"data: {not json", ("data: " + json.dumps(usage)).encode(), b"data: [DONE]",
              _chunk({"content": "after done"}).encode()]
    it = iter(token_s)
    return lines, lambda: next(it)


def _record(token_s):
    rec = {"ok": True, "t0": token_s[0] - 0.010, "t_first": None, "t_end": token_s[-1],
           "stamps": [], "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    lines, clock = _stream(token_s)
    usage = serving_kpi.consume_sse(lines, rec, clock)
    rec["completion_tokens"] = usage["completion_tokens"]
    return rec


def _times(start, gaps_ms):
    t = [start]
    for g in gaps_ms:
        t.append(t[-1] + g / 1e3)
    return t


def test_only_token_events_are_stamped():
    rec = _record([1.0, 1.003, 1.010])
    assert rec["stamps"] == [1.0, 1.003, 1.010]
    assert rec["t_first"] == 1.0 and rec["completion_tokens"] == 3


def test_itl_percentiles_pool_every_gap_across_streams():
    # Stream A gaps 1..50 ms, stream B gaps 51..100 ms: pooled gaps are exactly 1..100 ms.
    recs = [_record(_times(10.0, range(1, 51))), _record(_times(20.0, range(51, 101)))]
    itl = serving_kpi.summarize_level(recs, 1.0, 500.0, 50.0)["itl_ms"]
    print("itl_ms:", itl)
    assert itl["n"] == 100
    assert itl["p50"] == pytest.approx(50.5, abs=1e-6)
    assert itl["p90"] == pytest.approx(90.1, abs=1e-6)
    assert itl["p95"] == pytest.approx(95.05, abs=1e-6)
    assert itl["p99"] == pytest.approx(99.01, abs=1e-6)
    assert itl["max"] == pytest.approx(100.0, abs=1e-6)
    # Per-request TPOT (25.5, 75.5 ms) caps p99 at 75.0 and hides the 100 ms gap.
    assert serving_kpi.summarize_level(recs, 1.0, 500.0, 50.0)["tpot_ms"]["p99"] == \
        pytest.approx(75.0, abs=1e-6)


def test_no_gaps_reports_nan():
    itl = serving_kpi.summarize_level([_record([1.0])], 1.0, 500.0, 50.0)["itl_ms"]
    assert itl["n"] == 0 and itl["max"] != itl["max"] and itl["p90"] != itl["p90"]

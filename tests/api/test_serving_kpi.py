"""tools/analysis/serving_kpi.py: the KPI arithmetic, without a server.

Percentiles, the Prometheus-style histogram quantile the harness applies to
imp_queue_time_seconds deltas, the /metrics text parser, and the per-level
summary (TTFT / TPOT / ITL / E2E / normalized latency, goodput against the
SLO pair). Lane-agnostic: no request is made.
"""
import importlib.util
import math
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tools" / "analysis" / "serving_kpi.py"
_SPEC = importlib.util.spec_from_file_location("serving_kpi", _PATH)
kpi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kpi)


def test_pct_interpolates_like_numpy():
    xs = [10, 20, 30, 40]
    assert kpi.pct(xs, 50) == 25.0
    assert kpi.pct(xs, 0) == 10
    assert kpi.pct(xs, 100) == 40
    assert kpi.pct(xs, 95) == pytest.approx(38.5)
    assert kpi.pct([7], 99) == 7
    assert math.isnan(kpi.pct([], 50))


def test_hist_quantile_interpolates_inside_the_crossing_bucket():
    # 10 observations: 4 at <=0.1, 4 more at <=0.5, 2 more at <=1.0.
    buckets = {"0.1": 4, "0.5": 8, "1.0": 10, "+Inf": 10}
    assert kpi.hist_quantile(buckets, 0.5) == pytest.approx(0.2)   # rank 5 of 4..8 over 0.1..0.5
    assert kpi.hist_quantile(buckets, 0.9) == pytest.approx(0.75)  # rank 9 of 8..10 over 0.5..1.0
    assert kpi.hist_quantile(buckets, 0.2) == pytest.approx(0.05)  # rank 2 of 0..4 over 0..0.1
    assert math.isnan(kpi.hist_quantile({}, 0.5))
    assert math.isnan(kpi.hist_quantile({"0.1": 0, "+Inf": 0}, 0.5))
    # Everything beyond the last finite bound resolves to that bound.
    assert kpi.hist_quantile({"0.1": 0, "+Inf": 3}, 0.5) == 0.1


def test_parse_metrics_separates_plain_series_and_buckets():
    text = "\n".join([
        "# HELP imp_requests_total Total",
        "# TYPE imp_requests_total counter",
        "imp_requests_total 12",
        'imp_queue_time_seconds_bucket{le="0.005"} 3',
        'imp_queue_time_seconds_bucket{le="+Inf"} 7',
        "imp_queue_time_seconds_sum 0.42",
        'imp_memory_live_bytes{tier="weights"} 100',
    ])
    plain, hists = kpi.parse_metrics(text)
    assert plain["imp_requests_total"] == 12
    assert plain["imp_queue_time_seconds_sum"] == pytest.approx(0.42)
    assert "imp_memory_live_bytes" not in plain  # labeled, not a scalar
    assert hists["imp_queue_time_seconds"] == {"0.005": 3, "+Inf": 7}


def _rec(t0, ttft_s, n_out, tpot_s, prompt=100, ok=True, cached=0):
    stamps = [t0 + ttft_s + k * tpot_s for k in range(n_out)]
    return {"ok": ok, "t0": t0, "t_first": stamps[0] if stamps else None,
            "t_end": stamps[-1] if stamps else t0 + ttft_s, "stamps": stamps,
            "prompt_tokens": prompt, "completion_tokens": n_out, "cached_tokens": cached}


def test_summarize_level_goodput_counts_only_requests_meeting_both_slos():
    recs = [
        _rec(0.0, 0.100, 11, 0.020),   # meets both
        _rec(0.0, 0.900, 11, 0.020),   # TTFT over
        _rec(0.0, 0.100, 11, 0.080),   # TPOT over
        _rec(0.0, 0.200, 11, 0.030, cached=50),  # meets both
        {"ok": False, "t0": 0.0, "t_first": None, "t_end": 1.0, "stamps": [],
         "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
    ]
    s = kpi.summarize_level(recs, wall_s=10.0, slo_ttft_ms=500, slo_tpot_ms=50)
    assert s["ok"] == 4 and s["err"] == 1
    assert s["req_s"] == pytest.approx(0.4)
    assert s["output_tokens"] == 44 and s["input_tokens"] == 400
    assert s["output_tok_s"] == pytest.approx(4.4)
    assert s["total_tok_s"] == pytest.approx(44.4)
    assert s["goodput_req_s"] == pytest.approx(0.2)
    assert s["goodput_tok_s"] == pytest.approx(2.2)
    assert s["slo_attainment_pct"] == pytest.approx(50.0)
    assert s["ttft_ms"]["n"] == 4 and s["tpot_ms"]["n"] == 4
    assert s["itl_ms"]["n"] == 40  # 10 gaps per stream
    # TPOT = (t_last - t_first) / (n_out - 1) = the synthetic gap.
    assert s["tpot_ms"]["p50"] == pytest.approx(25.0)
    assert s["tpot_ms"]["p99"] == pytest.approx(78.5)
    # Normalized latency = E2E / output tokens: E2E 0.3 / 1.1 / 0.9 / 0.5 s
    # over 11 tokens, p50 interpolates between the two middle values.
    assert s["norm_ms_per_tok"]["p50"] == pytest.approx((0.5 + 0.9) / 2 / 11 * 1e3)
    assert s["e2e_s"]["p99"] == pytest.approx(1.1 - 0.2 * 0.03)
    assert s["cached_tokens"] == 50


def test_summarize_level_single_token_request_has_no_tpot():
    s = kpi.summarize_level([_rec(0.0, 0.05, 1, 0.0)], wall_s=1.0, slo_ttft_ms=500, slo_tpot_ms=50)
    assert s["tpot_ms"]["n"] == 0 and math.isnan(s["tpot_ms"]["p50"])
    assert s["itl_ms"]["n"] == 0
    assert s["slo_attainment_pct"] == 100.0  # TTFT alone decides


def test_server_summary_rates_from_deltas():
    m0 = {"imp_decode_batch_steps_total": 100, "imp_decode_batch_rows_total": 1000,
          "imp_tokens_prompt_total": 1000, "imp_tokens_cached_total": 100,
          "imp_spec_drafted_total": 0, "imp_spec_accepted_total": 0,
          "imp_kv_pressure_rejections_total": 1, "imp_streaming_kv_auto_enables_total": 0,
          "imp_prefix_cache_evictions_total": 5}
    m1 = {"imp_decode_batch_steps_total": 200, "imp_decode_batch_rows_total": 4200,
          "imp_tokens_prompt_total": 3000, "imp_tokens_cached_total": 600,
          "imp_spec_drafted_total": 40, "imp_spec_accepted_total": 30,
          "imp_kv_pressure_rejections_total": 3, "imp_streaming_kv_auto_enables_total": 1,
          "imp_prefix_cache_evictions_total": 5}
    h0 = {"imp_queue_time_seconds": {"0.01": 5, "0.1": 5, "+Inf": 5}}
    h1 = {"imp_queue_time_seconds": {"0.01": 5, "0.1": 15, "+Inf": 15}}
    samples = [{"kv_util_pct": 40.0, "active": 8}, {"kv_util_pct": 60.0, "active": 32}]
    s = kpi.server_summary(m0, h0, m1, h1, samples)
    assert s["rows_per_step"] == pytest.approx(32.0)
    assert s["prefix_hit_pct"] == pytest.approx(25.0)
    assert s["spec_accept_pct"] == pytest.approx(75.0)
    assert s["kv_pressure_rejections"] == 2
    assert s["streaming_kv_auto_enables"] == 1
    assert s["prefix_cache_evictions"] == 0
    assert s["kv_util_mean_pct"] == 50.0 and s["kv_util_max_pct"] == 60.0
    assert s["active_seqs_mean"] == 20.0 and s["active_seqs_max"] == 32
    # All 10 new queue observations sit in (0.01, 0.1]: p50 interpolates to 55 ms.
    assert s["queue_ms"]["p50"] == pytest.approx(55.0)
    assert s["queue_ms"]["n"] == 10

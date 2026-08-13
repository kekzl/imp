"""
Performance regression gate.

Measures TTFT and decode throughput against the stored verify.sh baseline.
Fails if performance regresses beyond thresholds.

What it tests:   TTFT p95 regression (optional, additive key), decode tok/s
                 regression vs the canonical baseline.
What it does NOT test: Absolute performance, model quality.
External state:  Running imp-server with real model + GPU.
                 Baseline file: tests/perf_baseline.json

The baseline file is the canonical verify.sh schema (legacy single-model):
    {"metrics": {"decode_tps": {"tg128": ...}, "prefill_tps": {"pp512": ...}},
     "thresholds": {"decode_regression_pct": 8, ...}, "schema_version": ...}
This suite gates decode throughput against metrics.decode_tps.tg128 and reads
its threshold from thresholds.decode_regression_pct so it shares ONE source of
truth with scripts/verify.sh (the audit found the old ["throughput"]/["ttft"]
keys never existed in the baseline, so the gate silently no-op'd — #579).

TTFT has no key in the verify.sh schema; it is an optional additive
"ttft" block this suite owns. When absent the TTFT test skips gracefully.

Run with:  pytest test_perf_regression.py -m perf
Update baselines:  pytest test_perf_regression.py --update-baseline
"""

import json
import os
import time

import httpx
import pytest

import conftest

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "perf_baseline.json")

# Fallback thresholds when the baseline omits a thresholds block. The baseline's
# thresholds.{decode,prefill}_regression_pct take precedence (shared with verify.sh).
TTFT_REGRESSION_PCT = 5         # p95 TTFT must not exceed baseline by more than 5%
THROUGHPUT_REGRESSION_PCT = 3   # decode tok/s must not regress by more than this


def load_baseline() -> dict:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {}


def baseline_decode_tps(baseline: dict):
    """Canonical decode-throughput baseline (verify.sh schema). None if absent."""
    return baseline.get("metrics", {}).get("decode_tps", {}).get("tg128")


def baseline_decode_threshold_pct(baseline: dict) -> float:
    """Decode regression threshold from the baseline, else the module default."""
    return baseline.get("thresholds", {}).get(
        "decode_regression_pct", THROUGHPUT_REGRESSION_PCT
    )


def save_baseline(data: dict):
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nBaseline saved to {BASELINE_PATH}")


def percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@pytest.mark.perf
@pytest.mark.skipif(conftest.USE_MOCK, reason="Performance tests require real model")
class TestTTFTRegression:
    """TTFT regression gate for standard prompt lengths."""

    PROMPT_LENGTHS = [16, 64, 256, 512]
    RUNS_PER_LENGTH = 10

    def _make_prompt(self, n_tokens: int) -> str:
        """Generate a prompt of approximately n_tokens."""
        # ~1 token per word, pad with "word" repeated
        words = ["The"] + ["word"] * (n_tokens - 1)
        return " ".join(words)

    def _measure_ttft(self, model: str, prompt: str) -> float:
        """Measure TTFT via SSE streaming."""
        t0 = time.monotonic()
        with httpx.stream(
            "POST",
            f"{conftest.BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": True,
            },
            timeout=30.0,
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    return time.monotonic() - t0
        return time.monotonic() - t0

    def test_ttft_regression(self, model, request):
        """TTFT p95 must not exceed baseline by more than 5%."""
        update = request.config.getoption("--update-baseline", default=False)
        baseline = load_baseline()

        results = {}
        for n in self.PROMPT_LENGTHS:
            prompt = self._make_prompt(n)
            ttfts = []
            for _ in range(self.RUNS_PER_LENGTH):
                ttfts.append(self._measure_ttft(model, prompt))

            p50 = percentile(ttfts, 50)
            p95 = percentile(ttfts, 95)
            p99 = percentile(ttfts, 99)

            key = f"ttft_pp{n}"
            results[key] = {"p50": p50, "p95": p95, "p99": p99}
            print(f"\n  pp{n}: p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  p99={p99*1000:.0f}ms")

        if update:
            baseline["ttft"] = results
            save_baseline(baseline)
            return

        if "ttft" not in baseline:
            pytest.skip("No TTFT baseline found. Run with --update-baseline first.")

        # Compare against baseline
        for key, measured in results.items():
            if key in baseline["ttft"]:
                base_p95 = baseline["ttft"][key]["p95"]
                threshold = base_p95 * (1 + TTFT_REGRESSION_PCT / 100.0)
                assert measured["p95"] <= threshold, (
                    f"{key}: p95={measured['p95']*1000:.0f}ms exceeds baseline "
                    f"{base_p95*1000:.0f}ms by >{TTFT_REGRESSION_PCT}% "
                    f"(threshold={threshold*1000:.0f}ms)"
                )


@pytest.mark.perf
@pytest.mark.skipif(conftest.USE_MOCK, reason="Performance tests require real model")
class TestThroughputRegression:
    """Decode throughput regression gate."""

    RUNS = 10

    def test_decode_throughput(self, client, model, request):
        """Decode tok/s p50 must not regress vs the canonical verify.sh baseline.

        Gates against metrics.decode_tps.tg128 with thresholds.decode_regression_pct
        (the same keys scripts/verify.sh reads) so the gate has a single source of
        truth. --update-baseline rewrites that key in place, preserving the rest of
        the verify.sh schema.
        """
        update = request.config.getoption("--update-baseline", default=False)
        baseline = load_baseline()

        throughputs = []
        for _ in range(self.RUNS):
            t0 = time.monotonic()
            r = client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 100."}],
                "max_tokens": 128,
                "temperature": 0,
            })
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            tokens = r.json()["usage"]["completion_tokens"]
            if tokens > 0 and elapsed > 0:
                throughputs.append(tokens / elapsed)

        p50 = percentile(throughputs, 50)
        print(f"\n  Decode throughput p50: {p50:.1f} tok/s")

        if update:
            baseline.setdefault("metrics", {}).setdefault("decode_tps", {})["tg128"] = p50
            save_baseline(baseline)
            return

        base = baseline_decode_tps(baseline)
        if base is None:
            pytest.skip(
                "No decode baseline (metrics.decode_tps.tg128) found. "
                "Run scripts/gen_perf_baseline.sh or --update-baseline first."
            )

        thr_pct = baseline_decode_threshold_pct(baseline)
        threshold = base * (1 - thr_pct / 100.0)
        assert p50 >= threshold, (
            f"Decode throughput p50={p50:.1f} tok/s below baseline "
            f"{base:.1f} tok/s by >{thr_pct}% "
            f"(threshold={threshold:.1f} tok/s)"
        )


def pytest_addoption(parser):
    """Add --update-baseline CLI flag to pytest."""
    try:
        parser.addoption("--update-baseline", action="store_true", default=False,
                         help="Update performance baseline file instead of comparing")
    except ValueError:
        pass  # Already added by another test file

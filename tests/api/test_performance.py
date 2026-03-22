"""Performance tests for the imp server API."""

import time
import concurrent.futures

import httpx
import pytest

from conftest import BASE_URL, parse_sse


@pytest.mark.perf
class TestPerformance:
    def test_ttft(self, client, model):
        """Time to first token via SSE should be under 2000ms p95."""
        ttfts = []
        for _ in range(5):
            t0 = time.monotonic()
            with httpx.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": True,
                },
                timeout=30.0,
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                        ttfts.append(time.monotonic() - t0)
                        break

        ttfts.sort()
        p50 = ttfts[len(ttfts) // 2]
        p95 = ttfts[int(len(ttfts) * 0.95)]
        print(f"\nTTFT: p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms")
        assert p95 < 2.0, f"TTFT p95 = {p95*1000:.0f}ms exceeds 2000ms"

    def test_sequential_throughput(self, client, model):
        """Sequential requests should maintain stable throughput."""
        times = []
        for _ in range(3):
            t0 = time.monotonic()
            r = client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 20."}],
                "max_tokens": 64,
                "temperature": 0,
            })
            assert r.status_code == 200
            elapsed = time.monotonic() - t0
            tokens = r.json()["usage"]["completion_tokens"]
            times.append((tokens, elapsed))

        for tokens, elapsed in times:
            tps = tokens / elapsed
            print(f"  {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

    def test_concurrent_requests(self, model):
        """4 concurrent requests should all complete successfully."""
        def make_request(i):
            with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
                r = c.post("/v1/chat/completions", json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"What is {i}+{i}?"}],
                    "max_tokens": 16,
                    "temperature": 0,
                })
                return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(make_request, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert all(s == 200 for s in results), f"Some requests failed: {results}"

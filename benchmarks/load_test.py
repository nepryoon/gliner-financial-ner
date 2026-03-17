#!/usr/bin/env python3
"""Load test for the GLiNER Financial NER API.

Uses the `locust` framework to measure:
  - p50 / p95 / p99 latency
  - Throughput (req/s)
  - Error rate

Usage
-----
    # Headless run (CI / scripts)
    locust -f benchmarks/load_test.py \
        --headless \
        --users 20 \
        --spawn-rate 5 \
        --run-time 60s \
        --host http://localhost:8000 \
        --json \
        --logfile benchmarks/results/locust.log \
        2>&1 | tee benchmarks/results/locust_stdout.txt

    # Interactive UI
    locust -f benchmarks/load_test.py --host http://localhost:8000

Alternatively, run the lightweight httpx-based runner directly:
    python benchmarks/load_test.py --url http://localhost:8000 --concurrency 10 --requests 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ── Locust user class ──────────────────────────────────────────────────────

try:
    from locust import HttpUser, between, task  # type: ignore[import]

    SAMPLE_TEXTS = [
        "Apple Inc. reported quarterly revenue of $90.1 billion, beating Wall Street estimates.",
        "Goldman Sachs CEO David Solomon said interest rates would remain elevated through Q3.",
        "The S&P 500 index dropped 2.4% after the Federal Reserve raised rates by 25 basis points.",
        "Tesla shares (TSLA) fell 8% following disappointing delivery numbers for Q2 2024.",
        "BlackRock managed assets grew to $10.5 trillion, making it the world's largest asset manager.",
        "The 10-year Treasury yield rose to 4.8%, its highest level since the 2008 financial crisis.",
        "JPMorgan Chase posted net income of $13.4 billion in the second quarter.",
        "Warren Buffett's Berkshire Hathaway disclosed a new $2.3 billion stake in Occidental Petroleum.",
    ]

    FINANCIAL_LABELS = [
        "company",
        "person",
        "financial instrument",
        "monetary amount",
        "percentage",
        "date",
        "organization",
        "stock ticker",
        "index",
    ]

    class NERUser(HttpUser):
        """Single-text prediction load test."""

        wait_time = between(0.1, 0.5)
        _idx = 0

        @task(4)
        def predict_single(self) -> None:
            text = SAMPLE_TEXTS[NERUser._idx % len(SAMPLE_TEXTS)]
            NERUser._idx += 1
            self.client.post(
                "/predict",
                json={
                    "text": text,
                    "labels": FINANCIAL_LABELS,
                    "threshold": 0.5,
                },
                name="/predict",
            )

        @task(1)
        def predict_batch(self) -> None:
            texts = SAMPLE_TEXTS[:4]
            self.client.post(
                "/predict/batch",
                json={
                    "texts": texts,
                    "labels": FINANCIAL_LABELS,
                    "threshold": 0.5,
                },
                name="/predict/batch",
            )

        @task(1)
        def health_check(self) -> None:
            self.client.get("/health", name="/health")

except ImportError:
    pass  # locust not installed — fall back to httpx runner below


# ── Lightweight httpx runner ────────────────────────────────────────────────


async def _run_single(
    client,  # type: ignore[type-arg]
    url: str,
    text: str,
    labels: list[str],
    results: list[float],
    errors: list[str],
) -> None:
    import httpx

    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/predict",
            json={"text": text, "labels": labels, "threshold": 0.5},
            timeout=30.0,
        )
        resp.raise_for_status()
        results.append((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        errors.append(str(exc))


async def run_benchmark(
    url: str,
    concurrency: int,
    total_requests: int,
) -> dict[str, Any]:
    import httpx

    sample_texts = [
        "Apple Inc. reported quarterly revenue of $90.1 billion.",
        "Goldman Sachs CEO David Solomon discussed interest rate policy.",
        "The S&P 500 dropped 2.4% after Fed raised rates by 25bps.",
        "Tesla (TSLA) fell 8% after disappointing Q2 delivery numbers.",
        "BlackRock AUM grew to $10.5 trillion in Q3 2024.",
    ]
    labels = [
        "company", "person", "financial instrument",
        "monetary amount", "percentage", "date",
        "organization", "stock ticker",
    ]

    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient() as client:
        # Warmup
        await client.get(f"{url}/health")

        sem = asyncio.Semaphore(concurrency)
        t_start = time.perf_counter()

        async def bounded(text: str) -> None:
            async with sem:
                await _run_single(client, url, text, labels, latencies, errors)

        tasks = [
            bounded(sample_texts[i % len(sample_texts)])
            for i in range(total_requests)
        ]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - t_start

    successful = len(latencies)
    throughput = successful / elapsed if elapsed > 0 else 0

    sorted_lat = sorted(latencies)

    def percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)]

    results = {
        "url": url,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": successful,
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / total_requests * 100, 2),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_rps": round(throughput, 2),
        "latency_ms": {
            "min": round(sorted_lat[0] if sorted_lat else 0, 2),
            "mean": round(statistics.mean(latencies) if latencies else 0, 2),
            "p50": round(percentile(sorted_lat, 50), 2),
            "p95": round(percentile(sorted_lat, 95), 2),
            "p99": round(percentile(sorted_lat, 99), 2),
            "max": round(sorted_lat[-1] if sorted_lat else 0, 2),
        },
    }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the NER API")
    p.add_argument("--url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--concurrency", type=int, default=10, help="Concurrent clients")
    p.add_argument("--requests", type=int, default=200, help="Total requests")
    p.add_argument(
        "--output",
        default="benchmarks/results/benchmark_results.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    print(f"Benchmarking {args.url} with {args.concurrency} concurrent clients …")
    results = asyncio.run(run_benchmark(args.url, args.concurrency, args.requests))

    print("\n=== Benchmark Results ===")
    print(f"  Throughput : {results['throughput_rps']} req/s")
    print(f"  p50 latency: {results['latency_ms']['p50']} ms")
    print(f"  p95 latency: {results['latency_ms']['p95']} ms")
    print(f"  p99 latency: {results['latency_ms']['p99']} ms")
    print(f"  Errors     : {results['errors']} / {results['total_requests']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

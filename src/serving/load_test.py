"""
src/serving/load_test.py

Simulates concurrent requests to the serving endpoint and measures p99
latency under realistic load. Run this after starting serve.py.

Usage:
    python -m src.serving.serve &  # start the server
    python -m src.serving.load_test --concurrency 4 --n-requests 200

Reports p50/p95/p99 latency, throughput (requests/sec), and error rate.
Results are written to experiments/load_test_report.json.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"

SAMPLE_CONTRACTS = [
    (
        "This Agreement shall be governed by and construed in accordance with the laws "
        "of the State of Delaware. Either party may terminate this Agreement for "
        "convenience upon thirty (30) days written notice to the other party. "
        "This Agreement shall automatically renew for successive one-year terms unless "
        "either party provides written notice of non-renewal at least sixty (60) days "
        "prior to the end of the then-current term.",
        ["Governing Law", "Termination For Convenience", "Renewal Term"],
    ),
    (
        "Each party agrees to keep confidential all information received from the other "
        "party. Company shall indemnify and hold harmless Client from any claims arising "
        "from Company's breach of this Agreement. In no event shall either party's total "
        "liability exceed the fees paid in the preceding twelve months.",
        ["Confidentiality Of Agreements", "Indemnification", "Cap On Liability"],
    ),
    (
        "Employee agrees not to compete with Company in any capacity for a period of "
        "two years following termination. This Agreement may not be assigned by either "
        "party without the prior written consent of the other party.",
        ["Non-Compete", "Assignment"],
    ),
]


async def single_request(
    client: httpx.AsyncClient,
    text: str,
    clause_types: list[str],
    base_url: str,
) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/extract",
            json={"text": text, "clause_types": clause_types},
            timeout=30.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return {"status": "ok", "latency_ms": latency_ms}
        else:
            return {"status": "error", "latency_ms": latency_ms, "code": resp.status_code}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"status": "error", "latency_ms": latency_ms, "error": str(e)}


async def run_load_test(
    base_url: str,
    n_requests: int,
    concurrency: int,
) -> dict:
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(i: int):
        text, clause_types = SAMPLE_CONTRACTS[i % len(SAMPLE_CONTRACTS)]
        async with semaphore:
            return await single_request(client, text, clause_types, base_url)

    t_start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    ok = [r for r in results if r["status"] == "ok"]
    latencies = [r["latency_ms"] for r in ok]
    error_rate = (len(results) - len(ok)) / len(results)

    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_time_s": round(total_time, 2),
        "throughput_rps": round(n_requests / total_time, 2),
        "error_rate": round(error_rate, 4),
        "latency": {
            "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
            "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
            "p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else None,
            "mean_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    print(f"Load test: {args.n_requests} requests, concurrency={args.concurrency}")
    report = asyncio.run(run_load_test(args.base_url, args.n_requests, args.concurrency))

    print(f"\n{'='*50}")
    print(f"  LOAD TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Throughput   : {report['throughput_rps']} req/s")
    print(f"  Error rate   : {report['error_rate']:.1%}")
    print(f"  p50 latency  : {report['latency']['p50_ms']}ms")
    print(f"  p95 latency  : {report['latency']['p95_ms']}ms")
    print(f"  p99 latency  : {report['latency']['p99_ms']}ms")
    print(f"{'='*50}")

    out = EXPERIMENTS_DIR / "load_test_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written → {out}")


if __name__ == "__main__":
    main()

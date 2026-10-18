"""Row 11.4.2: smoke and fast-path latency checks against a running terser-cpu container.

Standard library only, so the CD job needs no install. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = (
    "hey so um could you maybe help me write a short python function that checks whether a "
    "string is a palindrome? it should ignore spaces and capital letters, thanks so much!"
)


def _post(url: str, body: dict[str, object]) -> dict[str, object]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        out: dict[str, object] = json.load(r)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--p95-ms", type=float, default=150.0)
    a = ap.parse_args()

    with urllib.request.urlopen(f"{a.url}/healthz", timeout=10) as r:
        health = json.load(r)
    assert health["status"] == "ok", health
    first = _post(f"{a.url}/rewrite", {"prompt": PROMPT, "target": "gpt-4o", "mode": "fast"})
    print("rewrite:", json.dumps(first)[:300])
    with urllib.request.urlopen(f"{a.url}/metrics", timeout=10) as r:
        assert r.status == 200

    lat = []
    for _ in range(a.requests):
        t = time.perf_counter()
        _post(f"{a.url}/rewrite", {"prompt": PROMPT, "target": "gpt-4o", "mode": "fast"})
        lat.append((time.perf_counter() - t) * 1000)
    p95 = statistics.quantiles(lat, n=20)[-1]
    print(f"requests={len(lat)} p50={statistics.median(lat):.1f} ms p95={p95:.1f} ms")
    if p95 > a.p95_ms:
        print(f"FAIL: p95 {p95:.1f} ms > {a.p95_ms} ms")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

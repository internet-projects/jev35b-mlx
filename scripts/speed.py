"""Latency of decide() over a requests file (one request at a time). Run twice in a row: the first pass reads the
experts from disk as prompts route to them; the second shows the warm speed.

  python scripts/speed.py <model> requests.jsonl [n=120]
"""
import json, statistics, sys, time

from jev_mlx import Jev

model, req_file = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
reqs = [json.loads(l) for l in open(req_file) if l.strip()][: n + 3]
jev = Jev(model)
for r in reqs[:3]:
    jev.decide(r["kind"], r["state"], r["question"], r.get("options"))
for p in (1, 2):
    lat = []
    for r in reqs[3:]:
        t0 = time.perf_counter(); jev.decide(r["kind"], r["state"], r["question"], r.get("options"))
        lat.append(time.perf_counter() - t0)
    lat.sort()
    print(f"pass {p}: n={len(lat)} median {statistics.median(lat) * 1000:.0f} ms, p95 {lat[int(len(lat) * 0.95)] * 1000:.0f} ms")

"""Fidelity of an MLX build against a reference build, over a file of /v1/decide requests.

  python scripts/check.py requests.jsonl <reference model> <candidate model> [more candidates...]

For each candidate: how often its top answer matches the reference's, and the mean / median / 95th-percentile
KL divergence KL(reference || candidate) between the two probability distributions. Models are loaded one at a
time; each model's probabilities are cached next to the requests file.
"""
import gc, json, math, os, sys

import mlx.core as mx

from jev_mlx import Jev


def run(model, reqs, cache_dir):
    tag = model.strip("/").replace("/", "__")
    path = os.path.join(cache_dir, f"probs-{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))
    jev = Jev(model); out = {}
    for i, r in enumerate(reqs):
        out[r["id"]] = jev.decide(r["kind"], r["state"], r["question"], r.get("options"))["probabilities"]
        if (i + 1) % 50 == 0:
            print(f"  {model}: {i + 1}/{len(reqs)}", flush=True)
    json.dump(out, open(path, "w"))
    del jev; gc.collect(); mx.clear_cache()
    return out


def main():
    req_file, ref, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    reqs = [json.loads(l) for l in open(req_file) if l.strip()]
    cache = os.path.dirname(os.path.abspath(req_file))
    pr = run(ref, reqs, cache)
    for c in cands:
        pc = run(c, reqs, cache); same = 0; kls = []
        for k, a in pr.items():
            b = pc[k]
            same += max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
            kls.append(sum(p * math.log(max(p, 1e-12) / max(q, 1e-12)) for p, q in zip(a, b)))
        kls.sort(); n = len(kls)
        print(f"{c}: same top answer {same}/{n}, KL mean {sum(kls) / n:.4f} median {kls[n // 2]:.4f} p95 {kls[int(n * 0.95)]:.4f}")


if __name__ == "__main__":
    main()

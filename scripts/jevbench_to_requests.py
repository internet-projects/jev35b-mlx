"""Turn JevBench task files (https://github.com/fstandhartinger/jevbench, datasets/public/*.jsonl) into /v1/decide
requests, one JSON per line. This is the prompt set behind the fidelity numbers in the model cards.

  python scripts/jevbench_to_requests.py easy.jsonl original.jsonl hard.jsonl > requests.jsonl

Mapping: choice -> options "label: criterion"; noul -> the true/false rubric appended to the question;
score levels -> a choice over "i: level".
"""
import json, sys


def to_request(t):
    q = t["question"]; crit = q.get("criteria"); st = t["state"]
    state = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    if q["type"] == "choice":
        return {"kind": "choice", "state": state, "question": q["instructions"],
                "options": [f"{k}: {v}" if v else k for k, v in crit.items()]}
    if q["type"] == "noul":
        crit = crit or {}
        return {"kind": "noul", "state": state,
                "question": q["instructions"] + f"\n(true = {crit.get('true', 'yes')}; false = {crit.get('false', 'no')})"}
    return {"kind": "choice", "state": state, "question": q["instructions"],
            "options": [f"{i}: {c}" for i, c in enumerate(crit)]}


for path in sys.argv[1:]:
    for line in open(path):
        if line.strip():
            t = json.loads(line)
            print(json.dumps({"id": t["id"], **to_request(t)}, ensure_ascii=False))

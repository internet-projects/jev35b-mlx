# jev35b-mlx

[autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx): System 1 decisions with calibrated probabilities, for text and images, in one forward pass.

JEV-35B is AutoTrust AI's decision model, a fine-tune of Qwen3.5-35B-A3B (35 B parameters, about 3 B active per token). Every call is one typed question about a state, and the answer is a probability for every allowed option:

| kind | answers |
|---|---|
| `choice` | one of 2 to 256 options you supply |
| `noul` | yes / no, as probabilities for `["false", "true"]` |
| `score` | 0 to 5 |

It classifies, routes, gates and ranks (sort the options of one `choice` call by probability). It does not chat or generate text.

This package reproduces the read-out of autotrust's own server (`serve_decide.py`): the logits of the answer-slot tokens at the last position, plus the decision-head bias, divided by the per-kind temperature from `calibration.json`. The `/v1/decide` request and response fields are the same.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with AutoTrust AI.

## Builds

| Hugging Face | size | memory in use | same top answer as the original | probability drift (KL mean / p95) | warm median per decision |
|---|---|---|---|---|---|
| [netprojects/JEV-35B-mlx-8bit](https://huggingface.co/netprojects/JEV-35B-mlx-8bit) | 36 GB | 37.7 GB | 228 / 231 | 0.002 / 0.009 | 164 ms |
| [netprojects/JEV-35B-mlx-4bit-mixed](https://huggingface.co/netprojects/JEV-35B-mlx-4bit-mixed) | 21 GB | 21.6 GB | 225 / 231 | 0.004 / 0.021 | 109 ms |
| [netprojects/JEV-35B-mlx-3bit-mixed](https://huggingface.co/netprojects/JEV-35B-mlx-3bit-mixed) | 17 GB | 17.6 GB | 224 / 231 | 0.012 / 0.056 | 99 ms |

All three keep the **vision tower and the router gates in bf16**. The mixed builds put the routed experts (about 32 B of the 35 B parameters) at 4 or 3 bits and keep every other layer, including the decision head in `lm_head`, at 8 bits. Measured on a MacBook Pro M5 Max against the original bf16 weights; see [Verification](#verification).

Pick by memory: the 8-bit build for 48 GB Macs and up, 4-bit mixed for 32 GB, 3-bit mixed for 24 GB.

## Install

```bash
pip install "git+https://github.com/internet-projects/jev35b-mlx"
```

Requires an Apple Silicon Mac, Python 3.10+, `mlx` and `mlx-vlm` 0.7.6 or later (installed with the package).

## Use from Python

```python
import jev_mlx

jev = jev_mlx.load("netprojects/JEV-35B-mlx-4bit-mixed")     # a Hugging Face repo id or a local folder

r = jev.decide("choice",
               state="Customer message: my card was charged twice for the same order.",
               question="Which team should handle this ticket?",
               options=["billing", "shipping", "technical support", "account security"])
print(r["choice"], r["probabilities"])          # billing [0.9997, ...]

r = jev.decide("noul", state=["Photo from the warehouse:", {"image": "shelf.jpg"}],
               question="Is any box on the shelf damaged?")
print(r["probabilities"][1])                     # P(true)
```

`state` is a string, a JSON object, or a list mixing strings and images. An image is `{"image": ...}` with a file path, an `http(s)` URL or a `data:image/...;base64,` URI, or a PIL image.

## Use as a server

```bash
python -m jev_mlx serve netprojects/JEV-35B-mlx-4bit-mixed --port 8000

curl localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "kind": "choice",
  "state": "Customer message: my card was charged twice for the same order.",
  "question": "Which team should handle this ticket?",
  "options": ["billing", "shipping", "technical support", "account security"]}'
```

The response has `options`, `probabilities`, `choice`, `choice_index`, `usage` and `latency_ms`. `GET /v1/decide/info` lists the temperatures. Requests are handled one at a time.

One decision from the command line:

```bash
python -m jev_mlx decide netprojects/JEV-35B-mlx-3bit-mixed --kind noul \
  --state "Order 1182 arrived with a cracked screen." --question "Does the customer report damage?"
```

## Verification

```bash
# 231 decisions from the JevBench public set (https://github.com/fstandhartinger/jevbench)
python scripts/jevbench_to_requests.py jevbench/datasets/public/{easy,original,hard}.jsonl > requests.jsonl
python scripts/convert.py autotrust/JEV-35B ./JEV-35B-bf16 bf16          # reference: the original weights, unquantized
python scripts/check.py requests.jsonl ./JEV-35B-bf16 netprojects/JEV-35B-mlx-8bit netprojects/JEV-35B-mlx-4bit-mixed netprojects/JEV-35B-mlx-3bit-mixed
python scripts/vision_check.py netprojects/JEV-35B-mlx-3bit-mixed        # two generated images, 4 questions
python scripts/speed.py netprojects/JEV-35B-mlx-4bit-mixed requests.jsonl
```

`check.py` reports, for each build, how often its top answer matches the reference and the KL divergence between the two probability distributions. Our numbers are in [`results/results.json`](results/results.json). Every build answers the vision check 4/4.

Speed: one request at a time, prompts of about 94 tokens, measured with other models running on the same Mac. The first requests after loading are slower (a few hundred ms) while the experts are read from disk; the table shows the speed once they are cached.

## Convert it yourself

```bash
python scripts/convert.py autotrust/JEV-35B ./JEV-35B-mlx-4bit-mixed 4e     # recipes: 8, 4e, 3e, N, bf16
```

Convert before serving: loading autotrust's checkpoint straight into mlx-vlm re-lays all experts in one pass, which can exceed the Metal GPU time limit on a Mac.

## More from NetProjects

- [H2O-Lightning-4B on MLX](https://github.com/internet-projects/h2o-lightning-mlx), another decision model
- [NVIDIA Nemotron-Labs Dory 1.0 4B on MLX](https://github.com/internet-projects/dory-mlx)
- [NetProjects on Hugging Face](https://huggingface.co/netprojects)

## Credits and license

Model: [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B) by AutoTrust AI (Apache-2.0), a fine-tune of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (Apache-2.0). Runtime: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Code in this repository: Apache-2.0. Port by Paris Michalas, Partner, NetProjects (Internet Projects Ltd). Not affiliated with AutoTrust AI.

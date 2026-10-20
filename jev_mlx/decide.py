"""autotrust/JEV-35B on MLX: System 1 decisions with calibrated probabilities (text and images).

The read-out is the one in autotrust's serve_decide.py:
  prompt    "[kind] K\n[state] S\n[question] Q\n[options]\nA) ...\n[decision]:"   (raw text, no chat template;
            an image in the state becomes <|vision_start|><|image_pad|><|vision_end|>, expanded by the processor)
  read-out  logits of the answer-slot tokens at the last position + the decision-head bias,
            divided by the per-kind temperature (calibration.json), softmax.
"""
from __future__ import annotations

import base64, io, json, os, re, string, time, urllib.request

import mlx.core as mx

MAX_OPTIONS = 256
IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"
PREFILL_CHUNK = 2048
CACHE_LIMIT = 1 << 30   # cap MLX's buffer cache: freed GPU buffers otherwise pile up across prompt lengths
KINDS = ("choice", "noul", "score")
_DATA_URI = re.compile(r"^data:image/[\w.+-]+;base64,(.*)$", re.S)


def load_image(ref):
    """An image from a data URI, an http(s) URL, a file path, raw base64 or a PIL image."""
    from PIL import Image
    if hasattr(ref, "convert"):
        return ref.convert("RGB")
    if isinstance(ref, dict):
        ref = ref.get("url")
    m = _DATA_URI.match(ref.strip())
    if m:
        raw = base64.b64decode(m.group(1))
    elif ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref, timeout=15) as r:
            raw = r.read(25 * 1024 * 1024)
    elif os.path.exists(os.path.expanduser(ref)):
        raw = open(os.path.expanduser(ref), "rb").read()
    else:
        raw = base64.b64decode(ref)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im.convert("RGB")


def _local_path(path_or_repo: str) -> str:
    if os.path.isdir(os.path.expanduser(path_or_repo)):
        return os.path.expanduser(path_or_repo)
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo)


class Jev:
    """Load once, then call decide(). Not thread-safe: serialise calls (the server does)."""

    def __init__(self, path_or_repo: str, memory_limit_gb: float | None = None):
        from mlx_vlm import load
        mx.set_cache_limit(CACHE_LIMIT)
        if memory_limit_gb:
            mx.set_memory_limit(int(memory_limit_gb * 1e9))
        self.path = _local_path(path_or_repo)
        t0 = time.perf_counter()
        self.model, self.processor = load(self.path)
        self.load_s = time.perf_counter() - t0
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.lm = self.model.language_model
        self.head = json.load(open(os.path.join(self.path, "adapter_vllm", "decision_head.json")))
        self.temps = json.load(open(os.path.join(self.path, "calibration.json")))["per_kind"]
        self.temps.setdefault("score", 1.0)
        out = []
        for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
            t = self.tok.encode(lab, add_special_tokens=False)
            if len(t) == 1 and t[0] in self.tok.encode(f"x\n{lab}) y", add_special_tokens=False):
                out.append((lab, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        self.labels = [l for l, _ in out]
        self.label_ids = [t for _, t in out]
        lo, hi = self.head["slots"]["ranges"]["choice"]
        assert self.label_ids[: hi - lo] == self.head["verbalizer_ids"][lo:hi], "option labels differ from the trained head"

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def _parts(state):
        """-> (state text with image slots, [image refs]). state: str, dict, or a list mixing str and {"image": ...}."""
        if isinstance(state, str):
            return state, []
        if isinstance(state, dict):
            return json.dumps(state, ensure_ascii=False), []
        text, imgs = "", []
        for p in state:
            if isinstance(p, str):
                text += p
            elif hasattr(p, "convert"):
                imgs.append(p); text += IMAGE_SLOT
            elif isinstance(p, dict) and ("image" in p or p.get("type") == "image_url"):
                imgs.append(p.get("image") or p["image_url"]); text += IMAGE_SLOT
            elif isinstance(p, dict) and p.get("type") == "text":
                text += p["text"]
            else:
                text += json.dumps(p, ensure_ascii=False)
        return text, imgs

    def _prompt(self, kind, state_text, question, options=None):
        if kind == "choice":
            lines = [f"{self.labels[i]}) {o}" for i, o in enumerate(options)]
        else:
            lo, hi = self.head["slots"]["ranges"][kind]
            lines = self.head["slots"]["verbalizers"][lo:hi]
        return (f"[kind] {kind}\n[state] {state_text}\n[question] {question}\n[options]\n"
                + "\n".join(lines) + "\n[decision]:")

    # ------------------------------------------------------------------ forward
    def _last_logits(self, text, images):
        if images:
            o = self.processor(text=[text], images=[load_image(u) for u in images], return_tensors="np")
            ids = [int(x) for x in o["input_ids"][0]]
            input_ids = mx.array([ids], dtype=mx.int32)
            feats = self.model.get_input_embeddings(input_ids, pixel_values=mx.array(o["pixel_values"]),
                                                    image_grid_thw=mx.array(o["image_grid_thw"]))
        else:
            ids = self.tok.encode(text, add_special_tokens=False)
            input_ids = mx.array([ids], dtype=mx.int32)
            feats = self.model.get_input_embeddings(input_ids)
        emb, pos = feats.inputs_embeds, feats.position_ids
        if pos is None:
            pos = mx.arange(len(ids), dtype=mx.int32)[None]
        if pos.ndim == 2:
            pos = mx.broadcast_to(pos[None], (3,) + tuple(pos.shape))
        cache, n, h = self.lm.make_cache(), len(ids), None
        for s in range(0, n, PREFILL_CHUNK):
            e = min(n, s + PREFILL_CHUNK)
            h = self.lm.model(input_ids[:, s:e], inputs_embeds=emb[:, s:e], cache=cache, position_ids=pos[..., s:e])
            mx.eval(h)
        logits = self.lm.lm_head(h[:, -1, :]).astype(mx.float32)[0]
        mx.eval(logits)
        return logits, n

    # ------------------------------------------------------------------ public
    def decide(self, kind: str, state, question: str, options=None) -> dict:
        """kind: "choice" (2-256 options), "noul" (yes/no -> ["false", "true"]) or "score" (0-5).
        Returns {"options", "probabilities", "choice", "choice_index", "usage"}."""
        if kind not in KINDS:
            raise ValueError("kind must be choice, noul or score")
        head = self.head
        lo, hi = head["slots"]["ranges"][kind]
        if kind == "choice":
            n = len(options or [])
            if not 2 <= n <= MAX_OPTIONS:
                raise ValueError("choice needs 2-256 options")
            ids = self.label_ids[:n]
            bias = [head["bias"][lo + i] if lo + i < hi else 0.0 for i in range(n)]
            opts = [str(o) for o in options]
        else:
            ids = head["verbalizer_ids"][lo:hi]
            bias = head["bias"][lo:hi]
            opts = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
        text, imgs = self._parts(state)
        logits, n_tok = self._last_logits(self._prompt(kind, text, question, opts if kind == "choice" else None), imgs)
        p = mx.softmax((logits[mx.array(ids)] + mx.array(bias)) / self.temps[kind]).tolist()
        del logits
        mx.clear_cache()                                       # release this prompt's working buffers
        i = max(range(len(p)), key=p.__getitem__)
        return {"options": opts, "probabilities": p, "choice": opts[i], "choice_index": i,
                "usage": {"prompt_tokens": n_tok, "completion_tokens": 0, "images": len(imgs)}}


def load(path_or_repo: str) -> Jev:
    """jev = jev_mlx.load("netprojects/JEV-35B-mlx-4bit-mixed")  (a Hugging Face repo id or a local folder)"""
    return Jev(path_or_repo)

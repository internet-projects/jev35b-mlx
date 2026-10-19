"""python -m jev_mlx serve <model> [--port 8000]
   python -m jev_mlx decide <model> --kind choice --state "..." --question "..." --options a b c"""
import argparse, json


def main():
    ap = argparse.ArgumentParser(prog="python -m jev_mlx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="HTTP server with POST /v1/decide")
    s.add_argument("model", help="Hugging Face repo id or local folder")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    d = sub.add_parser("decide", help="one decision from the command line")
    d.add_argument("model")
    d.add_argument("--kind", default="choice", choices=["choice", "noul", "score"])
    d.add_argument("--state", required=True)
    d.add_argument("--image", action="append", default=[], help="image path or URL (repeatable), added after the state text")
    d.add_argument("--question", required=True)
    d.add_argument("--options", nargs="*")
    a = ap.parse_args()
    if a.cmd == "serve":
        from .server import serve
        serve(a.model, a.host, a.port)
    else:
        from .decide import Jev
        state = [a.state] + [{"image": i} for i in a.image] if a.image else a.state
        print(json.dumps(Jev(a.model).decide(a.kind, state, a.question, a.options), indent=1))


if __name__ == "__main__":
    main()

"""Stock vs mapped weights on one model, through a running oMLX server.

Used for the numbers in jundot/omlx#4541. For each mode it unloads the model,
sets `mapped_weights_enabled`, then sends chat completions with a ~1k and a
~16k-token prompt (two of each, greedy, 256 output tokens) and records the
server's own usage stats plus the server process phys_footprint.

Every prompt starts with a fresh id, so the prefix cache never serves it.

    OMLX_API_KEY=... python3 bench_mapped_weights.py \
        --base http://127.0.0.1:8000 --model <model-id> --modes 0,1,0,1

The model should already be aligned (Model Settings -> Mapped Weights ->
Realign) so the mapped rounds map every tensor. Only the standard library is
used; `footprint` ships with macOS.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import quote

# Neutral filler; only the token count matters.
PARA = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "the quarterly report, the weather stays mild, and the river keeps flowing "
    "past the old mill toward the sea. "
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="0,1,0,1", help="1 = mapped, 0 = stock")
    ap.add_argument("--contexts", default="1000,1000,16000,16000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="bench_mapped_weights.json")
    args = ap.parse_args()

    headers = {
        "Authorization": f"Bearer {os.environ.get('OMLX_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    model_path = quote(args.model, safe="")

    def req(method, path, body=None, timeout=900):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(
            args.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(r, timeout=timeout) as f:
                return f.status, json.loads(f.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def server_pid() -> str:
        port = args.base.rsplit(":", 1)[-1].split("/")[0]
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
        ).stdout
        return out.split()[0]

    def footprint_gb() -> dict:
        out = subprocess.run(
            ["footprint", "-p", server_pid()], capture_output=True, text=True
        ).stdout
        vals = {}
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0] in ("phys_footprint:", "phys_footprint_peak:"):
                n, unit = float(parts[1]), parts[2]
                vals[parts[0][:-1]] = n / 1024 if unit == "MB" else n
        return vals

    def prompt(tokens: int) -> str:
        body = "".join(f"[{i}] {PARA}" for i in range(max(1, tokens // 40)))
        return f"(id {uuid.uuid4()})\n{body}\n\nSummarize the text above in one long paragraph."

    def chat(ctx: int) -> dict:
        st, d = req(
            "POST",
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt(ctx)}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if st != 200 or "usage" not in d:
            return {"ctx": ctx, "error": f"{st} {str(d)[:300]}"}
        u = d["usage"]
        return {
            "ctx": ctx,
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "decode_tok_s": u["generation_tokens_per_second"],
            "prefill_tok_s": u.get("prompt_tokens_per_second"),
            "load_s": u.get("model_load_duration") or 0.0,
        }

    def set_mode(mapped: bool) -> None:
        req("POST", f"/admin/api/models/{model_path}/unload")
        for _ in range(120):
            _, models = req("GET", "/admin/api/models")
            entry = next(
                (m for m in models.get("models", []) if m.get("id") == args.model), {}
            )
            if not entry.get("loaded"):
                break
            time.sleep(1)
        req(
            "PUT",
            f"/admin/api/models/{model_path}/settings",
            {"mapped_weights_enabled": mapped},
        )
        time.sleep(3)

    results = []
    for mode in args.modes.split(","):
        mapped = mode.strip() == "1"
        set_mode(mapped)
        rows = [chat(int(c)) for c in args.contexts.split(",")]
        fp = footprint_gb()
        label = "mapped" if mapped else "stock"
        for r in rows:
            print(label, r, flush=True)
        print(label, "footprint_gb", fp, flush=True)
        results.append({"mode": label, "rows": rows, "footprint_gb": fp})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

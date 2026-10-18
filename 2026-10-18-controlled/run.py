"""Matriz: 2 modelos x {copiado+fusion, copiado sem fusion, mmap sem fusion} x 3 rodadas (quadrado latino)."""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
here = Path(__file__).resolve().parent
BASE, KEY = "http://127.0.0.1:8021", "bench-claude-local"
OUT = here / "results" / "ab.jsonl"

def api(path, body=None, method=None, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
        method=method, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "http": e.code}

def memory():
    raw = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", raw)[1])
    n = lambda k: int(re.search(re.escape(k) + r":\s+(\d+)", raw)[1]) * page / 1e9
    sw = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    return {"wired": n("Pages wired down"), "free": n("Pages free"), "compressed": n("Pages occupied by compressor"),
            "swap": float(re.search(r"used = ([\d.]+)M", sw)[1]) / 1000,
            "pressure": int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True))}

def unload(model):
    api(f"/admin/api/models/{model}/unload", {}, "POST")
    time.sleep(5)
    for _ in range(120):
        row = next(x for x in api("/v1/models/status").get("models", []) if x["id"] == model)
        if not row["loaded"] and not row["is_loading"]:
            time.sleep(3); return
        time.sleep(1)
    raise RuntimeError("unload não terminou")

def ask(model, prompt, tokens, phase, meta):
    before = memory(); t = time.monotonic()
    d = api("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": False}})
    row = {**meta, "phase": phase, "wall": time.monotonic() - t, "mem_before": before, "mem_after": memory(), "resp": d}
    with OUT.open("a") as f: f.write(json.dumps(row) + "\n")
    u = d.get("usage", {})
    print(meta["model"], meta["cond"], meta["round"], phase, "tok/s", u.get("generation_tokens_per_second"),
          "load", u.get("model_load_duration"), "press", row["mem_after"]["pressure"], "err", d.get("error"), flush=True)
    if d.get("http"): raise RuntimeError(d["error"])
    time.sleep(2)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(here / "models" / "Ornith-q4"))
ids = tok.encode("The river runs beside the forest. The city has a library and a train station. " * 10000)
q = "\nExplain in detail how a library catalog helps people find books. Give ten numbered examples."
p1k, p8k = tok.decode(ids[:1024]) + q, tok.decode(ids[:8192]) + q
conds = {"copied_fused": (False, True), "copied_unfused": (False, False), "mapped_unfused": (True, False)}
orders = [["copied_fused", "copied_unfused", "mapped_unfused"], ["mapped_unfused", "copied_fused", "copied_unfused"],
          ["copied_unfused", "mapped_unfused", "copied_fused"]]
models = sys.argv[1:] or ["Ornith-q3", "Ornith-q4"]
for model in models:
    for rnd, order in enumerate(orders, 1):
        for cond in order:
            mapped, fused = conds[cond]
            unload(model)
            (here / "fusion.json").write_text(json.dumps({"fusion": fused}))
            r = api(f"/admin/api/models/{model}/settings", {"mapped_weights_enabled": mapped, "mtp_enabled": False,
                    "vlm_mtp_enabled": False, "turboquant_kv_enabled": False}, "PUT")
            assert not r.get("http"), r
            meta = {"model": model, "cond": cond, "round": rnd, "t": time.time()}
            ask(model, p1k, 128, "load_short", meta)
            for _ in range(3): ask(model, p1k, 128, "warm_short", meta)
            ask(model, p1k, 512, "warm_long", meta)
            ask(model, p8k, 128, "warm_8k", meta)
            unload(model)
print("FIM", flush=True)

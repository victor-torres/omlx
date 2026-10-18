"""Cache frio x quente: primeiro pedido completo (carga + 1k->128), uma carga por caso."""
import json, subprocess, sys, time, glob, os
from pathlib import Path
import run as r
here = Path(__file__).resolve().parent
r.OUT = here / "results" / "cache.jsonl"
PY = sys.executable
BASE = {"mtp_enabled": False, "vlm_mtp_enabled": False, "turboquant_kv_enabled": False,
        "qwen35_ane_prefill_enabled": False, "qwen35_oq_a8_enabled": False}
conds = {"copied_fused": (False, True), "copied_unfused": (False, False), "mapped": (True, False)}
def prepare(model, cache):
    d = os.path.realpath(here / "models" / model)
    if cache == "cold":
        out = subprocess.check_output([PY, str(here / "evict.py"), d], text=True)
    else:
        for f in sorted(glob.glob(d + "/*.safetensors")):
            with open(f, "rb") as fh:
                while fh.read(64 << 20): pass
        out = subprocess.check_output([PY, str(here / "evict.py"), d, "--check"], text=True)
    return out.strip()
for spec in sys.argv[1:]:
    model, cache, cond = spec.split(":")
    mapped, fused = conds[cond]
    r.unload(model)
    (here / "fusion.json").write_text(json.dumps({"fusion": fused}))
    res = r.api(f"/admin/api/models/{model}/settings", {**BASE, "mapped_weights_enabled": mapped}, "PUT")
    assert not res.get("http"), res
    state = prepare(model, cache)
    print("cache", model, cache, cond, state, flush=True)
    r.ask(model, r.p1k, 128, "first", {"model": model, "cond": cond, "cache": cache, "round": 1, "t": time.time(), "cache_state": state})
    r.unload(model)
print("FIM", flush=True)

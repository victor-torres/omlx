"""Só mapeado: cache frio, velocidade e RAM (carregado, gerando, ocioso). Uso: run4.py modelo [mtp]"""
import json, os, subprocess, sys, time
from pathlib import Path
import run as r
here = Path(__file__).resolve().parent
r.OUT = here / "results" / ("dense-" + os.environ.get("TAG", "a") + ".jsonl")
PY = sys.executable
model, extra = sys.argv[1], (sys.argv[2:] or [""])[0]
d = os.path.realpath(here / "models" / model)
def ram(label):
    pid = subprocess.check_output(["lsof", "-tiTCP:8021", "-sTCP:LISTEN"], text=True).split()[0]
    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", pid], text=True)) * 1024
    fp_out = subprocess.check_output(["footprint", "-p", pid], text=True)
    import re as _re
    num, unit = _re.search(r"Footprint: ([\d.]+) (\w+)", fp_out).groups()
    footprint = float(num) * {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}[unit]
    res = subprocess.check_output([PY, str(here / "evict.py"), d, "--check"], text=True).strip()
    m = r.memory()
    row = {"model": model, "extra": extra, "label": label, "t": time.time(), "rss": rss, "footprint": footprint, "file_resident": res, **m}
    with (here / "results" / ("dense-ram-" + os.environ.get("TAG", "a") + ".jsonl")).open("a") as f: f.write(json.dumps(row) + "\n")
    print("RAM", model, extra, label, f"footprint {footprint/1e9:.1f}GB wired {m['wired']:.1f}GB rss {rss/1e9:.1f}GB livre {m['free']:.1f}GB swap {m['swap']:.1f}GB pressão {m['pressure']} | {res}", flush=True)
(here / "fusion.json").write_text(json.dumps({"fusion": True}))
r.unload(model)
ram("antes")
settings = {"mapped_weights_enabled": True, "mtp_enabled": extra == "mtp", "vlm_mtp_enabled": False,
            "turboquant_kv_enabled": False, "qwen35_ane_prefill_enabled": False, "qwen35_oq_a8_enabled": False}
res = r.api(f"/admin/api/models/{model}/settings", settings, "PUT"); assert not res.get("http"), res
print(subprocess.check_output([PY, str(here / "evict.py"), d], text=True).strip(), flush=True)
meta = {"model": model, "cond": "mapped" + ("+mtp" if extra else ""), "round": 1, "t": time.time()}
r.ask(model, r.p1k, 128, "first_cold", meta); ram("carregado")
for _ in range(2): r.ask(model, r.p1k, 512, "warm_long", meta)
r.ask(model, r.p8k, 128, "warm_8k", meta); ram("depois_8k")
time.sleep(90); ram("ocioso_90s")
r.ask(model, r.p1k, 128, "resume_after_idle", meta); ram("retomado")
r.unload(model); ram("descarregado")
print("FIM", flush=True)

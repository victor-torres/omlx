"""Sob pressão: carrega (cache frio), gera, aplica N GB de pressão, mede swap/pressão/páginas do modelo e a retomada."""
import json, os, subprocess, sys, time
from pathlib import Path
import run as r
here = Path(__file__).resolve().parent
PY = sys.executable
model, mapped, gb = sys.argv[1], sys.argv[2] == "mapped", int(sys.argv[3])
d = os.path.realpath(here / "models" / model)
OUT = here / "results" / "pressure.jsonl"
r.OUT = here / "results" / "pressure-req.jsonl"
def footprint():
    pid = subprocess.check_output(["lsof", "-tiTCP:8021", "-sTCP:LISTEN"], text=True).split()[0]
    import re
    num, unit = re.search(r"Footprint: ([\d.]+) (\w+)", subprocess.check_output(["footprint", "-p", pid], text=True)).groups()
    return float(num) * {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}[unit]
def resident():
    out = subprocess.check_output([PY, str(here / "evict.py"), d, "--check"], text=True)
    import re; return float(re.search(r"depois ([\d.]+) GB", out)[1])
(here / "fusion.json").write_text(json.dumps({"fusion": True}))
r.unload(model)
res = r.api(f"/admin/api/models/{model}/settings", {"mapped_weights_enabled": mapped, "mtp_enabled": False}, "PUT"); assert not res.get("http"), res
subprocess.check_output([PY, str(here / "evict.py"), d])
meta = {"model": model, "cond": "mapped" if mapped else "copied", "round": 1, "t": time.time()}
r.ask(model, r.p1k, 128, "first_cold", meta)
before = {"footprint": footprint(), "resident": resident(), **r.memory()}
stress = json.loads(subprocess.check_output([PY, str(here / "stress.py"), str(gb)], text=True, cwd=here).strip().splitlines()[-1])
after = {"footprint": footprint(), "resident": resident(), **r.memory()}
t = time.monotonic(); r.ask(model, r.p1k, 128, "after_pressure", meta); resume = time.monotonic() - t
row = {"model": model, "cond": meta["cond"], "gb": gb, "before": before, "after": after, "stress": stress, "resume_wall": resume}
with OUT.open("a") as f: f.write(json.dumps(row) + "\n")
print("PRESSAO", meta["cond"], f"footprint {before['footprint']/1e9:.1f}->{after['footprint']/1e9:.1f}GB | pesos residentes {before['resident']:.1f}->{after['resident']:.1f}GB | swap {stress['start']['swap']:.1f}->pico {stress['peak']['swap']:.1f}GB | comprimida pico {stress['peak']['compressed']:.1f}GB | pressão pico {stress['peak']['pressure']} | retomada {resume:.1f}s", flush=True)
r.unload(model)
print("FIM", flush=True)

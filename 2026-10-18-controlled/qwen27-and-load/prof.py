"""Carga crua do checkpoint, sem o servidor: copiado (mx.load + eval) x mapeado (_load_mapped).
Uso: prof.py MODELO_DIR copied|mapped cold|warm"""
import glob, json, os, re, subprocess, sys, threading, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskmem import task_mem
from omlx.utils import mapped_weights as mw
d, mode, cache = os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3]
files = sorted(glob.glob(d + "/*.safetensors"))
here = os.path.dirname(os.path.abspath(__file__))
def vm():
    raw = subprocess.check_output(["vm_stat"], text=True); page = int(re.search(r"page size of (\d+)", raw)[1])
    g = lambda k: int(re.search(re.escape(k) + r":\s+(\d+)", raw)[1])
    return {"free": g("Pages free") * page / 1e9, "compressed": g("Pages occupied by compressor") * page / 1e9,
            "pageins": g("Pageins"), "swapouts": g("Swapouts"), "compressions": g("Compressions"), "decompressions": g("Decompressions"), "swapins": g("Swapins"),
            "pressure": int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True))}
if cache == "cold":
    subprocess.check_output([sys.executable, os.path.join(here, "evict.py"), d])
else:
    for f in files:
        with open(f, "rb") as fh:
            while fh.read(64 << 20): pass
samples, stop = [], threading.Event()
phase = ["load"]
timeline = []
def sampler():
    t_start = time.monotonic()
    while not stop.is_set():
        v = vm(); samples.append(v)
        timeline.append({"t": round(time.monotonic() - t_start, 1), "phase": phase[0], "pressure": v["pressure"], "decomp": v["decompressions"], "swapins": v["swapins"], "pageins": v["pageins"], "comp_ops": v["compressions"], **{k: round(x, 2) for k, x in task_mem().items()}})
        stop.wait(0.5)
v0 = vm(); th = threading.Thread(target=sampler, daemon=True); th.start()
t0 = time.monotonic(); arrays = []
for f in files:
    if mode == "mapped":
        try:
            w = mw._load_mapped(f)[0]
        except mw._NativeLoaderError:
            w = mx.load(f)
    else:
        w = mx.load(f)
    arrays.extend(w.values())
t1 = time.monotonic()
phase[0] = "copy"; mx.eval(arrays)                      # materializa (copiado: lê e copia; mapeado: nada a fazer)
t2 = time.monotonic(); phase[0] = "gpu"
mx.eval([mx.sum(a.view(mx.uint8) if a.dtype != mx.bool_ else a) for a in arrays])   # GPU toca todos os bytes
mx.synchronize(); t3 = time.monotonic()
stop.set(); th.join(); v1 = vm()
res = subprocess.check_output([sys.executable, os.path.join(here, "evict.py"), d, "--check"], text=True).strip()
gb = sum(a.nbytes for a in arrays) / 1e9
print(json.dumps({"mode": mode, "cache": cache, "GB": round(gb, 2), "load_lazy_s": round(t1 - t0, 2), "eval_s": round(t2 - t1, 2),
    "gpu_touch_s": round(t3 - t2, 2), "total_s": round(t3 - t0, 2), "peak_pressure": max(s["pressure"] for s in samples + [v1]),
    "peak_compressed": round(max(s["compressed"] for s in samples + [v1]), 2), "compressed_start": round(v0["compressed"], 2),
    "pageins_GB": round((v1["pageins"] - v0["pageins"]) * 16384 / 1e9, 2), "swapouts": v1["swapouts"] - v0["swapouts"],
    "compressions": v1["compressions"] - v0["compressions"], "file_cache_after": res}), flush=True)
open(os.path.join(here, "results", f"timeline-{mode}-{cache}.json"), "w").write(json.dumps(timeline))

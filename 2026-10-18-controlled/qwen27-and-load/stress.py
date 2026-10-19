"""Pressão de memória controlada: aloca e toca N GB (1 GB por vez), segura e solta. Imprime o pico."""
import json, sys, time, numpy as np, subprocess
sys.path.insert(0, sys.argv[2]) if len(sys.argv) > 2 else None
import run as r
gb, blocks, peak = int(sys.argv[1]), [], {"pressure": 0, "swap": 0.0, "compressed": 0.0}
start = r.memory()
for _ in range(gb):
    blocks.append(np.ones(1 << 30, dtype=np.uint8))
    m = r.memory(); peak = {k: max(peak[k], m[k]) for k in peak}
for _ in range(20):
    time.sleep(1); m = r.memory(); peak = {k: max(peak[k], m[k]) for k in peak}
end = r.memory()
del blocks
print(json.dumps({"start": start, "end": end, "peak": peak}), flush=True)

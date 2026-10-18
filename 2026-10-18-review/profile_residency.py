"""Profile final weak-exporter accounting; no model graph or inference."""
import os,time,json,statistics,gc,pathlib
import mlx.core as mx
from omlx.utils.mapped_weights import _load_mapped
from omlx.utils.mapped_residency import resident_mapped_bytes,_views
model=pathlib.Path(os.environ['MODEL_DIR'])
with mx.stream(mx.cpu):
 holders=[_load_mapped(path)[0] for path in sorted(model.glob('*.safetensors'))]
print('Loaded checkpoint views',len(_views),flush=True)
samples=[];resident=[]
for _ in range(100):
 t=time.perf_counter();resident.append(resident_mapped_bytes());samples.append(time.perf_counter()-t)
d={'scope':'final9f372339 standalone checkpoint views, before engine sanitization','samples':len(samples),'live_exporters':len(_views),'median_ms':statistics.median(samples)*1000,'p95_ms':sorted(samples)[94]*1000,'max_ms':max(samples)*1000,'resident_bytes_last':resident[-1]}
holders.clear();mx.synchronize();mx.clear_cache();gc.collect();d['after_release_bytes']=resident_mapped_bytes()
print(json.dumps(d,indent=2),flush=True)
pathlib.Path('work/bench-results/final-residency-profile.json').write_text(json.dumps(d,indent=2)+'\n')

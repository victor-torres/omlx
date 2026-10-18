"""Test-only telemetry for weak exporter residency; not part of oMLX."""
import os,time,json,pathlib,threading
if os.environ.get('OMLX_BENCH_RESIDENCY_TRACE')=='1':
 def trace():
  import mlx.core as mx
  from omlx.utils.mapped_residency import resident_mapped_bytes,_views
  from omlx.utils.proc_memory import get_phys_footprint
  out=pathlib.Path(__file__).resolve().parent/'bench-results/residency-live.jsonl'
  with out.open('a') as f:
   while True:
    t=time.perf_counter();r=resident_mapped_bytes();dt=time.perf_counter()-t
    f.write(json.dumps({'t':time.time(),'resident_bytes':r,'query_s':dt,'live_exporters':len(_views),'active_mlx':mx.get_active_memory(),'footprint':get_phys_footprint()})+'\n');f.flush();time.sleep(1)
 threading.Thread(target=trace,daemon=True).start()
exec(compile(pathlib.Path(__file__).with_name('start_server.py').read_text(),'start_server.py','exec'))

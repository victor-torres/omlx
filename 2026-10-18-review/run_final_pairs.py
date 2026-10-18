"""Three alternating pairs against a clean main server (8012) and PR (8011)."""
import os,time,json,pathlib,threading,subprocess,sys
import bench_client as client
import run_bench as b
from transformers import AutoTokenizer
b.OUT=pathlib.Path(__file__).resolve().parent/'bench-results/verified-pairs'
b.OUT.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(str(pathlib.Path(__file__).resolve().parent/'models/Ornith-35B'))
ids=tok.encode(('The river runs beside the forest. The city has a library and a train station. ')*10000)
prompts={k:tok.decode(ids[:k])+'\nExplain in detail how a library catalog helps people find books. Give ten numbered examples.' for k in [1024,8192]}
thread=threading.Thread(target=b.sampler,daemon=True);thread.start()
try:
 for i,b.mode in enumerate([True,False,False,True,True,False]):
  b.round_id=i//2+1
  client.BASE='http://127.0.0.1:'+('8011' if b.mode else '8012')
  os.environ['OMLX_BENCH_VARIANT']=subprocess.check_output(['git','-C',str(pathlib.Path(__file__).resolve().parent/('omlx' if b.mode else 'omlx-main')),'rev-parse','HEAD'],text=True).strip()
  b.unload('Ornith-35B')
  settings={'mtp_enabled':False,'vlm_mtp_enabled':False,'turboquant_kv_enabled':False}
  if b.mode:settings['mapped_weights_enabled']=True
  result=b.api('/admin/api/models/Ornith-35B/settings',settings,'PUT')
  assert not result.get('http'),result
  b.record('load_short','Ornith-35B',prompts[1024],128)
  b.record('warm_short','Ornith-35B',prompts[1024],128)
  b.record('warm_8k','Ornith-35B',prompts[8192],128)
  if '--full-lifecycle' in sys.argv:
   b.phase='idle_30s';time.sleep(30)
   b.record('resume_30s','Ornith-35B',prompts[1024],128)
  b.unload('Ornith-35B')
finally:
 b.stop.set();thread.join(5)

"""One bounded 64k KV8 needle probe; never raises the configured guard."""
import pathlib,threading,os,json
import run_bench as b
from transformers import AutoTokenizer
b.OUT=pathlib.Path(__file__).resolve().parent/'bench-results/context64k';b.OUT.mkdir(exist_ok=True)
os.environ['OMLX_BENCH_VARIANT']='9f372339-KV8-fixed25GiB'
b.mode=True;b.round_id=1
m='Ornith-35B'
tok=AutoTokenizer.from_pretrained(str(pathlib.Path(__file__).resolve().parent/'models/Ornith-35B'))
ids=tok.encode(('The river runs beside the forest. The city has a library and a train station. ')*10000)
prompt=tok.decode(ids[:32000])+'\nThe library access code is LANTERN-4826. Remember this exact code.\n'+tok.decode(ids[32000:65536])+'\nWhat is the library access code? Output only the code.'
t=threading.Thread(target=b.sampler,daemon=True);t.start()
try:
 b.unload(m)
 result=b.api('/admin/api/models/'+m+'/settings',{'mapped_weights_enabled':True,'mtp_enabled':False,'vlm_mtp_enabled':False,'turboquant_kv_enabled':True,'turboquant_kv_bits':8},'PUT');assert not result.get('http'),result
 try:
  d=b.record('kv8_64k',m,prompt,64)
  text=d['choices'][0]['message'].get('content','')
  print('NEEDLE',text=='LANTERN-4826',repr(text),flush=True)
 except RuntimeError as e:print('BOUNDED_PROBE_REFUSED',str(e),flush=True)
finally:
 b.unload(m)
 b.api('/admin/api/models/'+m+'/settings',{'turboquant_kv_enabled':False},'PUT')
 b.stop.set();t.join(5)

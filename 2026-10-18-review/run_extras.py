"""Final scoped-loader verification, fusion control, proxy and tuning checks."""
import json,pathlib,time,threading,os,sys
os.environ['OMLX_BENCH_VARIANT']='final-scoped-module-proxy'
import run_bench as b
from transformers import AutoTokenizer
b.OUT=pathlib.Path(__file__).resolve().parent/'bench-results/final';b.OUT.mkdir(exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(str(pathlib.Path(__file__).resolve().parent/'models/Ornith-35B'))
ids=tokenizer.encode(('The river runs beside the forest. The city has a library and a train station. ')*10000)
prompt=tokenizer.decode(ids[:1024])+'\nExplain in detail how a library catalog helps people find books. Give ten numbered examples.'
needle=tokenizer.decode(ids[:16000])+'\nThe library access code is LANTERN-4826. Remember this exact code.\n'+tokenizer.decode(ids[16000:32768])+'\nWhat is the library access code? Output only the code.'
thread=threading.Thread(target=b.sampler,daemon=True);thread.start()
configs=[
 ('stock_default','Ornith-35B',False,{}),
 ('mapped_final','Ornith-35B',True,{}),
 ('stock_unfused_control','Ornith-stock-unfused',False,{}),
 ('mapped_proxy','Ornith-proxy',True,{}),
 ('mapped_mtp1','Ornith-35B',True,{'mtp_enabled':True,'mtp_fixed_depth':1}),
 ('mapped_mtp2','Ornith-35B',True,{'mtp_enabled':True,'mtp_fixed_depth':2}),
 ('mapped_kv8','Ornith-35B',True,{'turboquant_kv_enabled':True,'turboquant_kv_bits':8}),
]
try:
 if "--skip-qwen" not in sys.argv:
  # The maintainer's actual Qwen checkpoint, on the final loader hook.
  for b.mode in [False,True]:
   b.unload('Qwen3.5-0.8B')
   b.api('/admin/api/models/Qwen3.5-0.8B/settings',{'mapped_weights_enabled':b.mode},'PUT')
   b.record('final_qwen','Qwen3.5-0.8B','What is the capital of France? Answer briefly.',32)
  b.unload('Qwen3.5-0.8B')
 for n,(name,model,b.mode,extra) in enumerate(configs,1):
  b.round_id=n
  b.unload(model)
  settings={'mapped_weights_enabled':b.mode,'mtp_enabled':False,'mtp_fixed_depth':None,'vlm_mtp_enabled':False,'turboquant_kv_enabled':False,**extra}
  result=b.api('/admin/api/models/'+model+'/settings',settings,'PUT')
  if result.get('http'):
   print(name,'settings error',result,flush=True);continue
  try:
   b.record(name+'_load',model,prompt,128)
   for rep in range(2):b.record(name+'_warm'+str(rep+1),model,prompt,128)
   if name in ['mapped_final','mapped_kv8']:
    d=b.record(name+'_32k',model,needle,64)
    text=d['choices'][0]['message'].get('content','')
    print('NEEDLE',name,'LANTERN-4826' in text,text[:160],flush=True)
   if name=='mapped_final':
    b.phase='final_idle_30s';time.sleep(30)
    b.record(name+'_resume',model,prompt,128)
  except Exception as e:
   print(name,'FAILED',repr(e),flush=True)
  b.unload(model)
finally:
 b.stop.set();thread.join(5)

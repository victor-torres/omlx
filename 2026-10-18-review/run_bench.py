"""Reproducible local HTTP A/B; no GPU-limit or user-config mutations."""
import sys,time,json,subprocess,re,threading,statistics,pathlib,os
from bench_client import api,generate
OUT=pathlib.Path(__file__).resolve().parent/'bench-results'
stop=threading.Event();phase='setup';mode=None;round_id=0

def memory():
 raw=subprocess.check_output(['vm_stat'],text=True)
 page=int(re.search(r'page size of (\d+)',raw)[1])
 def n(name):return int(re.search(re.escape(name)+r':\s+(\d+)',raw)[1])*page/1e9
 sw=subprocess.check_output(['sysctl','-n','vm.swapusage'],text=True)
 return {'wired_gb':n('Pages wired down'),'compressed_gb':n('Pages occupied by compressor'),'file_gb':n('File-backed pages'),'free_gb':n('Pages free'),'swap_gb':float(re.search(r'used = ([\d.]+)M',sw)[1])/1000,'pressure':int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))}

def sampler():
 with (OUT/'memory.jsonl').open('a') as f:
  while not stop.is_set():
   try:f.write(json.dumps({'t':time.time(),'phase':phase,'mapped':mode,'round':round_id,**memory()})+'\n');f.flush()
   except Exception as e:print('sampler',str(e),flush=True)
   stop.wait(2)

def unload(model):
 api('/admin/api/models/'+model+'/unload',{},'POST')
 time.sleep(5)  # response may precede deferred lease/engine teardown
 for _ in range(90):
  status=api('/v1/models/status')
  row=next(x for x in status.get('models',[]) if x['id']==model)
  if not row['loaded'] and not row['is_loading']:
   time.sleep(2);return
  time.sleep(1)
 raise RuntimeError('unload did not settle')

def record(label,model,prompt,tokens):
 global phase
 phase=label
 d=generate(model,prompt,tokens)
 row={'t':time.time(),'round':round_id,'mapped':mode,'phase':label,'model':model,'response':d,'memory_after':memory(),'variant':os.environ.get('OMLX_BENCH_VARIANT','exploratory-core-hook')}
 with (OUT/'ab.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(label,mode,'round',round_id,'wall',round(d['wall_s'],2),'usage',d.get('usage'),'error',d.get('error'),flush=True)
 if d.get('http'):raise RuntimeError(d['error'])
 time.sleep(2)
 return d

if __name__ == "__main__":
 thread=threading.Thread(target=sampler,daemon=True);thread.start()
 try:
  if '--qwen' in sys.argv:
   for mode in [False,True]:
    unload('Qwen3.5-0.8B')
    api('/admin/api/models/Qwen3.5-0.8B/settings',{'mapped_weights_enabled':mode},'PUT')
    record('qwen_correctness','Qwen3.5-0.8B','What is the capital of France? Answer briefly.',32)
   unload('Qwen3.5-0.8B')
  else:
   from transformers import AutoTokenizer
   tokenizer=AutoTokenizer.from_pretrained(str(pathlib.Path(__file__).resolve().parent/'models/Ornith-35B'))
   body=('The river runs beside the forest. The city has a library and a train station. ')*10000
   ids=tokenizer.encode(body)
   prompts={k:tokenizer.decode(ids[:k])+'\nExplain in detail how a library catalog helps people find books. Give ten numbered examples.' for k in [1024,8192,32768]}
   cases=[True,False,False,True,True,False] if '--full' in sys.argv else [True]
   for i,mode in enumerate(cases):
    round_id=i//2+1
    unload('Ornith-35B')
    api('/admin/api/models/Ornith-35B/settings',{'mapped_weights_enabled':mode,'mtp_enabled':False,'vlm_mtp_enabled':False,'turboquant_kv_enabled':False},'PUT')
    record('load_short','Ornith-35B',prompts[1024],128)
    record('warm_short','Ornith-35B',prompts[1024],128)
    record('warm_8k','Ornith-35B',prompts[8192],128)
    phase='idle_30s';time.sleep(30)
    record('resume_30s','Ornith-35B',prompts[1024],128)
    if '--long' in sys.argv:record('long_32k','Ornith-35B',prompts[32768],128)
    if '--hold' not in sys.argv:unload('Ornith-35B')
 finally:
  stop.set();thread.join(5)

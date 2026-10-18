from pathlib import Path
import json,statistics as s
root=Path(__file__).resolve().parent/'bench-results/verified-pairs'
rows=[json.loads(x) for x in (root/'ab.jsonl').read_text().splitlines()]
assert len(rows)==18,len(rows)
out={}
for mode in [False,True]:
 data={}
 for phase in ['load_short','warm_short','warm_8k']:
  r=[x for x in rows if x['mapped']==mode and x['phase']==phase]
  assert len(r)==3,(mode,phase,len(r))
  values={'wall_s':[x['response']['wall_s'] for x in r]}
  for k in ['model_load_duration','time_to_first_token','generation_tokens_per_second','prompt_tokens_per_second']:
   if all(k in x['response']['usage'] for x in r):values[k]=[x['response']['usage'][k] for x in r]
  data[phase]={k:{'median':s.median(v),'min':min(v),'max':max(v),'n':len(v)} for k,v in values.items()}
 memory=[json.loads(x) for x in (root/'memory.jsonl').read_text().splitlines()]
 active=[x for x in memory if x['mapped']==mode and x['wired_gb']>24]
 data['active_system_wired_gb']={'median':s.median(x['wired_gb'] for x in active),'min':min(x['wired_gb'] for x in active),'max':max(x['wired_gb'] for x in active),'n':len(active)}
 out['mapped' if mode else 'main']=data
(root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

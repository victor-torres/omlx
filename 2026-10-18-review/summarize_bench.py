import json,statistics as st
from pathlib import Path
root=Path(__file__).parent/'bench-results'
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
primary=[x for x in read(root/'ab.jsonl') if x['model']=='Ornith-35B' and x['t']>=1792341740]
final=read(root/'final/ab.jsonl')
summary={'primary':{},'final':{}}
for mode in (False,True):
 out={}
 for phase in ['load_short','warm_short','warm_8k','resume_30s']:
  rows=[x for x in primary if x['mapped']==mode and x['phase']==phase]
  out[phase]={'n':len(rows),'wall_s':st.median(x['response']['wall_s'] for x in rows),**{k:st.median(x['response']['usage'].get(k,0) for x in rows) for k in ['model_load_duration','time_to_first_token','generation_tokens_per_second']}}
 summary['primary']['mapped' if mode else 'copied']=out
for x in final:
 summary['final'][x['phase']]={'wall_s':x['response']['wall_s'],**x['response']['usage']}
for name,path in [('primary',root/'memory.jsonl'),('final',root/'final/memory.jsonl')]:
 rows=read(path)
 if name=='primary':rows=[x for x in rows if x['t']>=1792341740]
 summary[name]['system_memory']={}
 for mode in (False,True):
  active=[x for x in rows if x['mapped']==mode and x['wired_gb']>24]
  idle=[x for x in rows if x['mapped']==mode and x['phase'] in ['idle_30s','final_idle_30s']]
  summary[name]['system_memory']['mapped' if mode else 'copied']={'active_wired_gb_median':st.median(x['wired_gb'] for x in active) if active else None,'idle_wired_gb_min':min((x['wired_gb'] for x in idle),default=None)}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['primary'],indent=2))
print('FINAL MEMORY',summary['final']['system_memory'])

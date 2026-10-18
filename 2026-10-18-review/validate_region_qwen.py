import json,time,pathlib
import bench_client as c
root=pathlib.Path(__file__).resolve().parent
rows=[]
for tier in ['safe','balanced','aggressive']:
 r=c.api('/admin/api/global-settings',{'memory_guard_tier':tier},'POST');assert not r.get('http'),r
 c.api('/admin/api/models/Qwen3.5-0.8B/settings',{'mapped_weights_enabled':True},'PUT')
 d=c.generate('Qwen3.5-0.8B','What is the capital of France? Answer briefly.',32)
 print(tier,d.get('usage'),d.get('error'),flush=True);assert not d.get('http'),d
 assert 'Paris' in d['choices'][0]['message']['content']
 rows.append({'tier':tier,'response':d})
 c.api('/admin/api/models/Qwen3.5-0.8B/unload',{},'POST');time.sleep(8)
(root/'bench-results/region-qwen-auto.json').write_text(json.dumps(rows,indent=2))

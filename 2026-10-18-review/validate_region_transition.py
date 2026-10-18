import json,time,pathlib
import bench_client as c
root=pathlib.Path(__file__).resolve().parent
r=c.api('/admin/api/global-settings',{'memory_guard_tier':'custom','memory_guard_custom_ceiling_gb':25},'POST');assert not r.get('http'),r
c.api('/admin/api/models/Ornith-35B/settings',{'model_type_override':None,'mapped_weights_enabled':True,'mtp_enabled':False,'turboquant_kv_enabled':False},'PUT')
rows=[]
d=c.generate('Ornith-35B','Explain how a library catalog works. Give twenty detailed numbered examples.',128);print('BOOTSTRAP',d.get('usage'),d.get('error'),flush=True);assert not d.get('http'),d
rows.append({'phase':'bootstrap_custom','response':d})
r=c.api('/admin/api/global-settings',{'memory_guard_tier':'aggressive'},'POST');assert not r.get('http'),r
for label,prompt,n in [('automatic_active','Explain how a library catalog works. Give twenty detailed numbered examples.',256),('automatic_resume','What is the capital of France? Answer briefly.',32)]:
 if label=='automatic_resume':time.sleep(10)
 d=c.generate('Ornith-35B',prompt,n);print(label,d.get('usage'),d.get('error'),flush=True);rows.append({'phase':label,'response':d})
 if d.get('http'):break
(root/'bench-results/region-transition.json').write_text(json.dumps(rows,indent=2))
c.api('/admin/api/models/Ornith-35B/unload',{},'POST');time.sleep(8)

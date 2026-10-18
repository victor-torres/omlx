"""Standard-library client for the isolated local oMLX benchmark."""
import json,time,urllib.request,urllib.error,uuid
BASE='http://127.0.0.1:8011'
KEY='bench-local-isolated'
def api(path,body=None,method=None,timeout=600):
 r=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,method=method,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(r,timeout=timeout) as res:return json.load(res)
 except urllib.error.HTTPError as e:return {'error':e.read().decode(),'http':e.code}
def generate(model,prompt,tokens=128):
 t=time.monotonic()
 d=api('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':tokens,'chat_template_kwargs':{'enable_thinking':False}})
 return {'wall_s':time.monotonic()-t,**d}
if __name__=='__main__':
 for model in ['Qwen3.5-0.8B','Ornith-35B']:
  for mapped in [False,True]:
   print('UNLOAD',api('/admin/api/models/'+model+'/unload',{},'POST'))
   print('SET',model,mapped,api('/admin/api/models/'+model+'/settings',{'mapped_weights_enabled':mapped,'mtp_enabled':False,'vlm_mtp_enabled':False,'turboquant_kv_enabled':False},'PUT').get('status'))
   print('RESULT',model,mapped,json.dumps(generate(model,'What is the capital of France? Answer briefly.',32)),flush=True)

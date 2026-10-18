"""Offline experiment only: aligned sidecars for affected shards; originals untouched."""
import pathlib,json,struct,time,hashlib,shutil,os
base=pathlib.Path(__file__).resolve().parent
source=(base/'models/Ornith-35B').resolve()
dest=base/'proxy.nosync/Ornith-proxy'
dest.mkdir(parents=True,exist_ok=False)
manifest={'source':str(source),'shards':[],'start':time.time()}; t=time.monotonic()
for path in source.iterdir():
 out=dest/path.name
 if path.suffix!='.safetensors':out.symlink_to(path);continue
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=f.read(n);info=json.loads(header)
  misaligned=any((8+n+v['data_offsets'][0])%8 for k,v in info.items() if k!='__metadata__' and v['data_offsets'][1]>v['data_offsets'][0])
  if not misaligned:out.symlink_to(path);continue
  encoded=header+b' '*((-n)%8)
  # Existing tensor sizes/relative offsets are already multiples of eight.
  assert all((8+len(encoded)+v['data_offsets'][0])%8==0 for k,v in info.items() if k!='__metadata__' and v['data_offsets'][1]>v['data_offsets'][0])
  digest=hashlib.sha256()
  with out.open('xb') as g:
   g.write(struct.pack('<Q',len(encoded)));g.write(encoded)
   while chunk:=f.read(8<<20):digest.update(chunk);g.write(chunk)
  with out.open('rb') as g:
   g.seek(8+len(encoded));other=hashlib.sha256()
   while chunk:=g.read(8<<20):other.update(chunk)
  assert digest.digest()==other.digest()
  manifest['shards'].append({'file':path.name,'bytes':path.stat().st_size,'payload_sha256':digest.hexdigest(),'source_mtime_ns':path.stat().st_mtime_ns})
manifest['build_s']=time.monotonic()-t
manifest['extra_bytes']=sum(x['bytes'] for x in manifest['shards'])
(base/'models/Ornith-proxy').symlink_to(dest,target_is_directory=True)
(base/'bench-results/proxy-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))

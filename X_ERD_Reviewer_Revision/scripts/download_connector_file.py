"""Materialize an already resolved, read-only connector file reference.

The transient URL is supplied as base64 JSON on argv, never persisted in logs.
Source inventory (without tokens) and SHA-256 receipts preserve provenance.
"""
from pathlib import Path
import base64,hashlib,json,sys,urllib.request,os

ROOT=Path(__file__).resolve().parents[1]
p=json.loads(base64.b64decode(sys.argv[1]));name=p['name']
assert Path(name).name==name and name.endswith('.json')
dest=ROOT/'data/zirari_raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
receipt=ROOT/'data/zirari_raw_receipts'/(name+'.receipt.json');receipt.parent.mkdir(parents=True,exist_ok=True)
if not(dest.exists() and dest.stat().st_size==int(p['size']) and receipt.exists()):
    tmp=dest.with_suffix('.partial')
    with urllib.request.urlopen(p['url'],timeout=180) as r, tmp.open('wb') as f:
        while b:=r.read(1024*1024):f.write(b)
    if tmp.stat().st_size!=int(p['size']):raise ValueError('Downloaded size differs from Drive inventory')
    os.replace(tmp,dest)
h=hashlib.sha256()
with dest.open('rb') as f:
    while b:=f.read(1024*1024):h.update(b)
receipt.write_text(json.dumps({'path':dest.relative_to(ROOT).as_posix(),'drive_id':p['id'],'bytes':dest.stat().st_size,'sha256':h.hexdigest()},indent=2))
print(name,'verified',dest.stat().st_size,flush=True)

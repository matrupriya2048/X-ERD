"""Download only inert reference files, enforcing the released byte hashes."""
from pathlib import Path
import json, urllib.request, hashlib, concurrent.futures
ROOT=Path(__file__).resolve().parents[1]

def fetch(row):
    path=ROOT/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']:return path.name
    with urllib.request.urlopen(urllib.request.Request(row['url'],headers={'User-Agent':'XERD-Reproduction/0.2'}),timeout=120) as r:
        data=r.read()
    if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Reference byte drift: '+row['path'])
    path.with_suffix('.part').write_bytes(data);path.with_suffix('.part').replace(path)
    return path.name

if __name__=='__main__':
    rows=json.loads((ROOT/'config/reference_sources.json').read_text())['files']
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for name in pool.map(fetch,rows):print('Verified',name,flush=True)

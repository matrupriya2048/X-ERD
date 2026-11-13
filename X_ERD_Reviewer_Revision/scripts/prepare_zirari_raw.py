"""Acquire the frozen public JSON inventory; never download or execute binaries."""
from pathlib import Path
import argparse,hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');ap.add_argument('--only');a=ap.parse_args()
    inv=json.loads((ROOT/'config/zirari_raw_inventory.json').read_text());expected={}
    saved=ROOT/'config/zirari_raw_sources.json'
    status_file=ROOT/'config/zirari_acquisition_status.json'
    status={r['name']:r for r in json.loads(status_file.read_text())['files']} if status_file.exists() else {}
    if saved.exists():expected={Path(f['path']).name:f for f in json.loads(saved.read_text())['files']}
    for item in inv['files']:
        name=item['title']
        if a.only and name!=a.only:continue
        if status.get(name,{}).get('status')=='unavailable_provider_restriction':
            print(name,'not in frozen acquired cohort: provider download restriction',flush=True);continue
        assert Path(name).name==name and name.endswith('.json') and item['mime_type']=='application/json'
        path=ROOT/'data/zirari_raw'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            if a.verify_only:raise FileNotFoundError(path)
            import gdown
            tmp=path.with_suffix('.gdown-partial')
            downloaded=gdown.download(id=item['id'],output=str(tmp),quiet=False,use_cookies=False)
            if not downloaded:raise RuntimeError('Public Drive download failed: '+name)
            if tmp.stat().st_size!=int(item['size']):raise ValueError('Publisher size mismatch: '+name)
            os.replace(tmp,path)
        if path.stat().st_size!=int(item['size']):raise ValueError('Size mismatch: '+name)
        receipt={'path':path.relative_to(ROOT).as_posix(),'drive_id':item['id'],'bytes':path.stat().st_size,'sha256':sha(path)}
        if name in expected and receipt['sha256']!=expected[name]['sha256']:raise ValueError('Published bytes changed: '+name)
        out=ROOT/'data/zirari_raw_receipts'/(name+'.receipt.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2))
        print(name,'verified',flush=True)

if __name__=='__main__':main()

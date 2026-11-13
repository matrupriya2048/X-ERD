"""Acquire inert published behavioral reports, with per-file SHA256 verification."""
from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'data'/'external_sources'
def request(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'XERD-Research/0.2','Accept':'application/vnd.mendeley-public-dataset.1+json'}),timeout=90)
def hashed(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def download(row):
 p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and hashed(p)==row['sha256']:return row['path']
 for attempt in range(4):
  try:
   with request(row['url']) as r,p.with_suffix('.part').open('wb') as f:
    while b:=r.read(1048576):f.write(b)
   if hashed(p.with_suffix('.part'))!=row['sha256']:raise ValueError('SHA256 mismatch')
   p.with_suffix('.part').replace(p);return row['path']
  except Exception:
   if attempt==3:raise
   time.sleep(2*(attempt+1))
def main():
 manifest=ROOT/'config'/'external_data_sources.json'
 if manifest.exists():rows=json.loads(manifest.read_text())['files']
 else:
  folders=json.loads((S/'mendeley_folders1.json').read_text());rows=[]
  for folder in folders:
   if folder['name'] not in ['benign','lockbit','netwalker','petya','wannacry']:continue
   url='https://data.mendeley.com/public-api/datasets/yh9p6vs53y/files?'+urllib.parse.urlencode({'folder_id':folder['id'],'version':1,'$start':0,'$limit':1000})
   with request(url) as r:files=json.load(r)
   (S/(folder['name']+'_raw_listing.json')).write_text(json.dumps(files,indent=2))
   for item in files:
    assert item['filename'].endswith('.json') and Path(item['filename']).name==item['filename']
    d=item['content_details']
    rows.append({'path':'data/external_reports/'+item['filename'],'family':folder['name'],'label':int(folder['name']!='benign'),'sha256':d['sha256_hash'],'bytes':d['size'],'url':d['download_url'],'source_doi':'10.17632/yh9p6vs53y.1','license':'CC-BY-4.0'})
  manifest.write_text(json.dumps({'dataset':'Iwan et al. API call dataset v1','source':'https://data.mendeley.com/datasets/yh9p6vs53y/1','files':rows},indent=2))
 print('Reports',len(rows),'bytes',sum(x['bytes'] for x in rows),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for i,_ in enumerate(pool.map(download,rows),1):
   if i%5==0:print('Verified',i,'/',len(rows),flush=True)
 print('All external reports verified',flush=True)
if __name__=='__main__':main()

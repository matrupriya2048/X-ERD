"""Persist primary-source availability; never execute downloaded source or samples."""
from pathlib import Path
import json, urllib.request, hashlib, concurrent.futures

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'external_sources'
URLS={
 'mlran_record':'https://zenodo.org/api/records/23403681',
 'sequence_mendeley':'https://data.mendeley.com/public-api/datasets/yh9p6vs53y/versions/1',
 'sequence_files':'https://data.mendeley.com/public-api/datasets/yh9p6vs53y/versions/1/files',
 'rande_tree':'https://api.github.com/repos/rande-code/RanDe-W11/git/trees/main?recursive=1',
 'rande_readme':'https://raw.githubusercontent.com/rande-code/RanDe-W11/main/README.md',
 'mlran_tree':'https://api.github.com/repos/faithfulco/mlran/git/trees/main?recursive=1',
 'isot_page':'https://onlineacademiccommunity.uvic.ca/isot/datasets/ransomware/',
 'ransec_pdf':'https://jastt.org/index.php/jasttpath/article/download/555/125/3254',
 'nunes_record':'https://zenodo.org/api/records/1203289',
}
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'XERD-Research-Reproduction/0.2'})
  with urllib.request.urlopen(req,timeout=45) as r:
   data=r.read();ctype=r.headers.get('Content-Type','')
  suffix='.pdf' if data.startswith(b'%PDF') else '.json' if data[:1] in [b'{',b'['] else '.txt'
  target=OUT/(name+suffix);target.write_bytes(data)
  return {'name':name,'url':url,'file':str(target.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'content_type':ctype}
 except Exception as e:return {'name':name,'url':url,'error':str(e)}
if __name__=='__main__':
 import sys
 if len(sys.argv)>1:URLS=dict(x.split('=',1) for x in sys.argv[1:])
 OUT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(fetch,URLS.items()))
 log=OUT/'availability.json'
 old=json.loads(log.read_text()) if log.exists() else []
 (OUT/'availability.json').write_text(json.dumps(old+rows,indent=2),encoding='utf-8')
 print(json.dumps(rows,indent=2))

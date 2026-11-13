"""Reproduce the pinned inert-report acquisition, without changing cohort selection."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request
import py7zr
HERE=Path(__file__).resolve().parent
URL='https://zenodo.org/api/records/18218588/files/benign.7z/content'
SIZE=83376646

def fetch(bounds):
    a,b=bounds
    request=urllib.request.Request(URL+'?offset='+str(a),headers={'Range':f'bytes={a}-{b}','User-Agent':'XERD-research'})
    with urllib.request.urlopen(request,timeout=60) as response:
        content=response.read()
        if len(content)!=b-a+1:raise ValueError('Range response length mismatch')
    return a,content

if __name__=='__main__':
    root=HERE/'data_en2025';root.mkdir(exist_ok=True);archive=root/'benign.7z'
    if not archive.exists() or archive.stat().st_size!=SIZE:
        ranges=[(a,min(a+8*1024*1024,SIZE)-1) for a in range(0,SIZE,8*1024*1024)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chunks=sorted(executor.map(fetch,ranges))
        archive.write_bytes(b''.join(x[1] for x in chunks))
    if hashlib.md5(archive.read_bytes()).hexdigest()!='c6667140e2eb181dd40a7cfe041328cb':raise ValueError('Publisher MD5 mismatch')
    protocol=json.loads((HERE/'en2025_protocol.json').read_text())
    selected=[x['name'] for x in protocol['files']]
    assert all(not Path(x).is_absolute() and '..' not in Path(x).parts and x.endswith('/report.json') for x in selected)
    with py7zr.SevenZipFile(archive) as source:source.extract(path=root/'raw',targets=selected)
    print('Verified and extracted',len(selected),'inert JSON reports')

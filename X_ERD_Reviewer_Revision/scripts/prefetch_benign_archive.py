"""Optional bounded parallel prefetch for the selected solid-compression blocks.

Cache compressed bytes only. Do not extract or execute archive programs.
Runs safely alongside the ordinary extractor using unique temporary files.
"""
from pathlib import Path
import argparse,json,urllib.request,time,concurrent.futures,hashlib,uuid,threading
import py7zr
from inspect_remote_archive import HTTPRangeFile,bounded_download
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--remediation',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    a=json.loads((ROOT/'config/benign_archive.json').read_text());f=a['file'];url=f['links']['self'];size=f['size']
    selected=set(json.loads((ROOT/'config/raw_remediation_protocol.json').read_text())['holdout_selection']) if args.remediation else set(json.loads((ROOT/'config/benign_report_sources.json').read_text())['selection'])
    remote=HTTPRangeFile(url,size)
    with py7zr.SevenZipFile(remote) as z:
        folders=z.header.main_streams.unpackinfo.folders;pack=z.header.main_streams.packinfo
        sizes=pack.packsizes;indices={next(i for i,o in enumerate(folders) if o is x.folder) for x in z.files if x.filename in selected}
        if len(sizes)!=len(folders):raise ValueError('Prefetch requires one packed stream per folder')
        start=32+pack.packpos;chunks=set();step=1024*1024
        for i,n in enumerate(sizes):
            if i in indices:chunks.update(range(start//step,(start+n+step-1)//step))
            start+=n
    remote.close();cache=remote.disk;done=0;lock=threading.Lock()
    def fetch(block):
        nonlocal done
        start=block*step;end=min(size,start+8*step)-1
        members=list(range(block,(end+1+step-1)//step))
        if all((cache/str(i)).exists() for i in members):return
        marker=cache/(str(block)+'.fetching');marker.write_text(str(time.time()))
        for attempt in range(6):
            try:
                data=bounded_download(url,start,end,size)
                if len(data)!=end-start+1:raise EOFError('Incomplete range')
                for i in members:
                    dest=cache/str(i);part=data[(i-block)*step:(i-block+1)*step]
                    tmp=dest.with_suffix('.'+uuid.uuid4().hex+'.part');tmp.write_bytes(part);tmp.replace(dest)
                marker.unlink(missing_ok=True)
                with lock:
                    done+=len(members)
                    if done%128==0:print('Prefetched MiB',done,flush=True)
                return
            except Exception:
                if attempt==5:raise
                time.sleep(2**attempt)
    todo=sorted({i//8*8 for i in chunks if not (cache/str(i)).exists()})
    print('Prefetch missing eight-MiB ranges',len(todo),flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:list(pool.map(fetch,todo))
    print('Prefetch complete',flush=True)

if __name__=='__main__':main()

"""Inspect a public 7z archive over verified HTTPS ranges without full download."""
import io,urllib.request,urllib.error,json,collections,time,concurrent.futures,hashlib,http.client,uuid
from pathlib import Path

def bounded_download(url,start,end,size):
    """Resume a truncated HTTP response from its exact verified byte offset."""
    data=bytearray();failures=0
    while start+len(data)<=end:
        offset=start+len(data)
        request_end=min(end,offset+512*1024-1)
        req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{request_end}','User-Agent':'XERD-Reproducibility/0.3','Accept-Encoding':'identity'})
        try:
            with urllib.request.urlopen(req,timeout=90) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {offset}-{request_end}/{size}':raise ValueError('Incorrect HTTP byte range')
                try:payload=r.read()
                except http.client.IncompleteRead as ex:payload=ex.partial
            if not payload or len(payload)>request_end-offset+1:raise EOFError('Invalid bounded response length')
            data.extend(payload);failures=0
        except urllib.error.HTTPError as error:
            failures+=1
            if failures>=6:raise
            if error.code==429:
                delay=error.headers.get('Retry-After','120')
                time.sleep(max(120,float(delay)) if delay.isdigit() else 120)
            else:time.sleep(2**failures)
        except Exception:
            failures+=1
            if failures>=6:raise
            time.sleep(2**failures)
    return bytes(data)

class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.pos=0;self.cache=collections.OrderedDict();self.transferred=0
        self.blocksize=1024*1024
        self.disk=Path(__file__).resolve().parents[1]/'data/range_cache'/hashlib.sha256(url.encode()).hexdigest()[:16]
        self.disk.mkdir(parents=True,exist_ok=True)
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending={}
    def fetch(self,block):
        start=block*self.blocksize;end=min(start+self.blocksize,self.size)-1
        dest=self.disk/str(block)
        marker=self.disk/(str(block//8*8)+'.fetching')
        deadline=time.monotonic()+90
        while marker.exists() and time.monotonic()<deadline and not dest.exists():time.sleep(.2)
        if dest.exists() and dest.stat().st_size==end-start+1:return dest
        payload=bounded_download(self.url,start,end,self.size)
        temp=dest.with_suffix('.'+uuid.uuid4().hex+'.part');temp.write_bytes(payload);temp.replace(dest)
        return dest
    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        super().close()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        end=min(self.pos+n,self.size);pieces=[];blocksize=self.blocksize
        while self.pos<end:
            block=self.pos//blocksize;start=block*blocksize
            if block not in self.cache:
                for ahead in range(block,min(block+1,(self.size+blocksize-1)//blocksize)):
                    if ahead not in self.pending:self.pending[ahead]=self.pool.submit(self.fetch,ahead)
                payload=self.pending[block].result().read_bytes()
                self.transferred+=len(payload);self.cache[block]=payload
                if self.transferred//(64*1024*1024)!=(self.transferred-len(payload))//(64*1024*1024):print('Archive transferred MiB',round(self.transferred/1048576),flush=True)
                if len(self.cache)>32:self.cache.popitem(last=False)
            payload=self.cache[block];offset=self.pos-start;take=min(end-self.pos,len(payload)-offset)
            if take<=0:raise EOFError('Truncated range')
            pieces.append(payload[offset:offset+take]);self.pos+=take
        return b''.join(pieces)

if __name__=='__main__':
    import py7zr
    root=Path(__file__).resolve().parents[1];src=root/'data/external_sources'
    record=json.loads((src/'nunes_record.json').read_text());f=next(x for x in record['files'] if x['key']=='Cuckoo.7z')
    remote=HTTPRangeFile(f['links']['self'],f['size'])
    with py7zr.SevenZipFile(remote,'r') as z:
        rows=[{'filename':x.filename,'size':x.uncompressed,'directory':x.is_directory} for x in z.list()]
        folders=[{'unpack_size':x.get_unpack_size(),'solid':x.solid} for x in z.header.main_streams.unpackinfo.folders]
    (src/'nunes_archive_index.json').write_text(json.dumps({'files':rows,'folders':folders,'transferred':remote.transferred},indent=2))
    print('Entries',len(rows),'compression folders',len(folders),'network bytes',remote.transferred)
    print('First entries',rows[:12])
    print('Reports',sum(x['filename'].endswith('report.json') for x in rows))

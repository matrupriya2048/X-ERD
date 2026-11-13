"""Byte integrity when an archive HTTP response is truncated or misranged."""
import http.client
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('archive_acquisition',Path(__file__).resolve().parents[1]/'scripts/inspect_remote_archive.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Response:
    status=206
    def __init__(self,header,payload,partial=None):self.headers={'Content-Range':header};self.payload=payload;self.partial=partial
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):
        if self.partial is not None:raise http.client.IncompleteRead(self.payload[:self.partial],len(self.payload)-self.partial)
        return self.payload

def test_truncated_body_resumes_exact_offset_and_limits_requests(monkeypatch):
    payload=bytes(range(256))*2049;requests=[]
    def fetch(request,timeout):
        a,b=map(int,request.get_header('Range').removeprefix('bytes=').split('-'));requests.append((a,b))
        assert b-a+1<=512*1024
        return Response(f'bytes {a}-{b}/{len(payload)}',payload[a:b+1],200000 if len(requests)==1 else None)
    monkeypatch.setattr(module.urllib.request,'urlopen',fetch)
    assert module.bounded_download('https://example.invalid/archive',0,len(payload)-1,len(payload))==payload
    assert requests[1][0]==200000

def test_wrong_content_range_never_enters_cache(monkeypatch):
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**k:Response('bytes 1-3/4',b'bad'))
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(ValueError,match='Incorrect HTTP byte range'):
        module.bounded_download('https://example.invalid/archive',0,2,4)

"""Seeded raw benign-report sample from Nunes' original public archive.

Only JSON report members are selected. No executables or dropped files are
extracted. HTTP range reads avoid downloading irrelevant archive members.
"""
from pathlib import Path
import py7zr,json,random,hashlib,zlib
from inspect_remote_archive import HTTPRangeFile

def main():
    root=Path(__file__).resolve().parents[1];src=root/'data/external_sources';dest=root/'data/benign_reports'
    archive=json.loads((root/'config/benign_archive.json').read_text());f=archive['file']
    pool=archive['eligible_paths']
    selected=sorted(random.Random(42).sample(pool,200))
    assert all('..' not in Path(p).parts and not Path(p).is_absolute() for p in selected)
    manifest=root/'config/benign_report_sources.json'
    protocol={'source':'https://zenodo.org/records/1203289','doi':'10.5281/zenodo.1203289',
      'archive':f['key'],'archive_bytes':f['size'],'archive_publisher_md5':f['checksum'],
      'sampling':'Python random.Random42 sample200 without replacement from lexicographically sorted original benign report paths; no outcome/size selection',
      'eligible_report_n':len(pool),'selection':selected,
      'verification':'7z header and extracted-member CRCs, plus local per-report SHA256; whole archive MD5 is NOT checked when range extraction avoids whole download',
      'label_basis':'publisher describes these three folders as clean programs; not independently re-adjudicated here; unique executable hashes and source overlap checked before evaluation'}
    manifest.write_text(json.dumps(protocol,indent=2))
    remote=HTTPRangeFile(f['links']['self'],f['size'])
    with py7zr.SevenZipFile(remote,'r') as z:
        lookup={x.filename:x for x in z.files};protocol['selected_crc32']={p:lookup[p].crc32 for p in selected}
        folders={id(lookup[p].folder) for p in selected}
        print('Selected reports',len(selected),'of',len(pool),'compression blocks',len(folders),flush=True)
        todo=[p for p in selected if not (dest/p).exists() or zlib.crc32((dest/p).read_bytes())!=lookup[p].crc32]
        if todo:z.extract(path=dest,targets=todo)
    records=[]
    for p in selected:
        data=(dest/p).read_bytes()
        if zlib.crc32(data)!=protocol['selected_crc32'][p]:raise ValueError('Entry CRC mismatch '+p)
        records.append({'path':str(Path('data/benign_reports')/p).replace('\\','/'),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    protocol['files']=records;protocol['range_bytes_consumed_this_run']=remote.transferred
    protocol['cache_note']='Verified HTTPS range cache may satisfy reads; read count is not a network byte count. Only JSON reports are extracted.'
    remote.close()
    manifest.write_text(json.dumps(protocol,indent=2));print('Verified benign reports',len(records),flush=True)
if __name__=='__main__':main()

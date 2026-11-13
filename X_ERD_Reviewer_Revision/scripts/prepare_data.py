"""Retrieve only published feature data and JSON reports; never binary samples."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]

def checksum(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def fetch(url,path,expected,algorithm='sha256'):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if checksum(path,algorithm)!=expected:raise ValueError(f'Existing file checksum mismatch: {path}; original preserved')
        print('Verified',path.name);return
    temp=path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(url,timeout=120) as src,temp.open('wb') as dst:
        while block:=src.read(1024*1024):dst.write(block)
    if checksum(temp,algorithm)!=expected:raise ValueError(f'Download checksum mismatch: {path.name}')
    temp.replace(path);print('Downloaded and verified',path.name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--strict-temporal',action='store_true');args=p.parse_args()
    spec=json.loads((ROOT/'config/data_sources.json').read_text(encoding='utf-8'))
    for name,source in spec['files'].items():fetch(source['url'],ROOT/'data'/name,source['sha256'])
    if args.strict_temporal:
        source=spec['full_archive'];archive=ROOT/'data/MLRan_full_dataset.zip'
        fetch(source['url'],archive,source['md5'],'md5')
        dest=ROOT/'data/raw_features';dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                filename=Path(member.filename).name
                if member.is_dir() or '__MACOSX' in member.filename:continue
                if any(filename==g+s for g in ['1_api','6_net','7_sys','8_drop','9_sig'] for s in ['_dataset.parquet','_feature_names_dic.json']):
                    # Use the validated basename, never a path from archive content.
                    content=z.read(member);target=dest/filename
                    if target.exists() and target.read_bytes()!=content:raise ValueError(f'Existing raw feature file differs: {filename}')
                    target.write_bytes(content);print('Verified raw modality',filename)

if __name__=='__main__':main()

"""Freeze the files that must be present in the final executable image."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def main():
    files=[]
    for folder in ['src','tests','scripts','config']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.sh','.json'} and p.name!='source_integrity.json':
                files.append(p)
    files += [ROOT/name for name in ['pyproject.toml','requirements.lock','requirements-sequence.lock','requirements-acquisition.lock','requirements-documents.lock']]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'config/source_integrity.json').write_text(json.dumps({'sha256':manifest},indent=2))
    print('Pinned',len(manifest),'source/configuration files')

if __name__=='__main__':main()

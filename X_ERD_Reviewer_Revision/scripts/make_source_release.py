"""Prepare a local reviewable source ZIP; does not upload or publish anything."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=[]
    rootfiles=['README.md','LICENSE','NOTICE','CITATION.cff','pyproject.toml','requirements.lock','requirements-sequence.lock','requirements-acquisition.lock','requirements-documents.lock','Dockerfile','Dockerfile.extended','.dockerignore','.gitignore']
    files.extend(ROOT/p for p in rootfiles)
    for directory in ['src','tests','scripts','config','docs','results']:
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file():continue
            if any(s in p.parts for s in ['__pycache__','.pytest_cache','prior_attempt']):continue
            if p.suffix in ['.pyc','.npz','.joblib','.zip']:continue
            if p.name in ['docker_first_attempt.log']:continue
            files.append(p)
    files.extend(ROOT/'manuscript'/p for p in ['main_scirep_revised.tex','main_scirep_revised.pdf','references_revision.bib','wlscirep.cls','naturemag.bst','fig_class_category_real.png'])
    files.extend(ROOT/'manuscript'/p for p in ['main_scirep_complete.tex','supplementary_information.tex','supplementary_information.pdf'])
    # Include supplied reference source explicitly for reproducible generation,
    # isolated from the revised manuscript and with its original filename clear.
    for p in ['supplied_reference_only.tex','references.bib']:
        files.append(ROOT/'manuscript'/p)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    (ROOT/'release_manifest.json').write_text(json.dumps({'sha256':manifest,'scope':'local source review archive; no uploaded release or DOI yet','excluded':'raw data,model.joblib,shap NPZ,prior attempts,temporary render/build files'},indent=2),encoding='utf-8')
    dest=ROOT/'X_ERD_Reviewer_Revision_Source.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'release_manifest.json','release_manifest.json')
    print('Created local source archive:',dest.name,len(files),'files',dest.stat().st_size,'bytes')

if __name__=='__main__':main()

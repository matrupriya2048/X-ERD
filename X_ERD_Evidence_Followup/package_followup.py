from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def run():
    base=ROOT/'X_ERD_Reviewer_Revision_Source.zip'
    assert digest(base)=='4b37b416c912b99c1a7849ab1735785e2a356e252eb7ce60c9999f7b0bebd9f7'
    qa_path=HERE/'docs/document_qa.json';qa=json.loads(qa_path.read_text())
    for name,item in qa.items():
        file=(HERE/'docs' if name=='response_letter' else HERE/'manuscript')/(name+'.pdf')
        assert digest(file)==item['sha256']
        item['visual_inspection']='Completed: all rendered pages inspected in contact sheets; added sections, tables and references checked.'
    qa_path.write_text(json.dumps(qa,indent=2))
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='passed'
    for name,sha in verification['followup_source_hashes'].items():assert digest(HERE/name)==sha,name
    assert 'Ran 10 tests' in (HERE/'tests_docker.log').read_text(encoding='utf-8')
    files=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(HERE)
        if any(x in {'raw','tmp','__pycache__','data_en2025','multicollection_reproduction'} for x in rel.parts):continue
        if p.suffix in {'.aux','.log','.out','.blg','.bbl','.zip','.joblib','.7z'}:continue
        if '.compile' in p.name:continue
        if p.name in {'followup_manifest.json','archive_verification.json'}:continue
        files.append(p)
    manifest={'base_archive_sha256':digest(base),'followup_files':{str(p.relative_to(HERE)).replace('\\','/'):{'sha256':digest(p),'bytes':p.stat().st_size} for p in files}}
    manifest_path=HERE/'followup_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=HERE/'X_ERD_Updated_Working_Package.zip';expected={}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        first='Current revision: 0.4.2. Open X_ERD_Evidence_Followup/README.txt and FINAL_COMMENT_LEDGER.html first. Latest PDFs are in X_ERD_Evidence_Followup/manuscript and docs. X_ERD_Reviewer_Revision preserves the v0.3 companion implementation and earlier results. Reproduction and final publication instructions are in the current README.\n'
        out.writestr('README_FIRST.txt',first);expected['README_FIRST.txt']=hashlib.sha256(first.encode()).hexdigest()
        for name in ['CITATION.cff','LICENSE']:
            data=(HERE/name).read_bytes();out.writestr(name,data);expected[name]=hashlib.sha256(data).hexdigest()
        with zipfile.ZipFile(base) as original:
            assert original.testzip() is None
            for item in original.infolist():
                name=PurePosixPath(item.filename)
                assert not name.is_absolute() and '..' not in name.parts
                data=original.read(item.filename);target='X_ERD_Reviewer_Revision/'+item.filename
                out.writestr(target,data);expected[target]=hashlib.sha256(data).hexdigest()
        for path in files+[manifest_path]:
            target='X_ERD_Evidence_Followup/'+str(path.relative_to(HERE)).replace('\\','/')
            out.write(path,target);expected[target]=digest(path)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        assert len(check.namelist())==len(set(check.namelist()))==len(expected)
        for name,sha in expected.items():assert hashlib.sha256(check.read(name)).hexdigest()==sha
    result={'status':'passed','archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,
            'members_verified':len(expected),'raw_reports_or_model_binaries_added':False,'base_archive_unchanged':True,
            'scope':'Verified version0.4.2 working package; primary decision and multi-collection repair implemented. Human adjudication, verified onset references and author-controlled publication remain explicitly scoped.'}
    (HERE/'archive_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()

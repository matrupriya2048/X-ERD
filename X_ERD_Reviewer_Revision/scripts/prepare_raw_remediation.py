"""Freeze prospective raw-report holdout and acquire inert JSON members only."""
from pathlib import Path
import argparse,json,random,hashlib,zlib,datetime

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--acquire',action='store_true');args=ap.parse_args()
    archive=json.loads((ROOT/'config/benign_archive.json').read_text())
    old=json.loads((ROOT/'config/benign_report_sources.json').read_text())
    dest=ROOT/'config/raw_remediation_protocol.json'
    pool=sorted(set(archive['eligible_paths'])-set(old['selection']))
    selected=sorted(random.Random(20261113).sample(pool,400))
    if not dest.exists():
        protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'reason':'Post-hoc response to observed external benign FPR; no prospective claim for the original 200 reports',
          'development':'All eligible unique nonoverlapping executables from the previously evaluated 200 reports; publisher clean labels; preserve original negative result',
          'development_manifest_sha256':hashlib.sha256((ROOT/'config/benign_report_sources.json').read_bytes()).hexdigest(),
          'holdout_source':old['source'],'archive':archive['file'],
          'holdout_pool_n':len(pool),'holdout_selection':selected,'holdout_seed':20261113,
          'holdout_sampling':'400 without replacement from sorted remaining publisher-clean report paths; chosen before new model fitting or scoring',
          'eligibility':'Identical original behavior/API/hash rules; exclude duplicates and any overlap with all MLRan and development executables before scoring; account for all selected reports',
          'representation':'Unchanged official 483 features and exact-token parser',
          'base_models':'Original RF100/XGB100/LGB100 settings; same stack and optional previously evaluated passthrough',
          'development_weights':[0.,1.,3.],'meta_C':[.1,1.,10.],
          'threshold':.5,'outer_folds':5,'inner_folds':5,'fold_seed':42,
          'stratification':'source benign, source ransomware, external benign; MD5 groups in both loops',
          'selection':'Among pooled source outer-OOF recall>=0.97 and FPR<=0.05, minimize external-development OOF FPR, then source FPR, then maximize source recall, then declared candidate order. If infeasible maximize source recall then minimize external FPR; report infeasibility.',
          'freeze':'Persist model and configuration SHA-256 before first holdout scoring; all new-holdout comparisons descriptive with no target-driven reselection',
          'evaluation':'Fresh same-collection benign specificity; official benchmark reuse separately assesses ransomware recall. Do not pool unrelated sources or call this a matched external ransomware/benign accuracy test.',
          'limits':'Adaptation to one legacy collection, not zero-shot cross-environment validation; feature selection remains fixed; no human expert or encryption-onset evidence created'}
        dest.write_text(json.dumps(protocol,indent=2));print('Frozen protocol: 400 new report paths, 18 declared configurations',flush=True)
    protocol=json.loads(dest.read_text())
    assert protocol['holdout_selection']==selected and not set(selected)&set(old['selection'])
    if not args.acquire:return
    import py7zr
    from inspect_remote_archive import HTTPRangeFile
    manifest=ROOT/'config/raw_remediation_holdout_sources.json'
    f=archive['file'];remote=HTTPRangeFile(f['links']['self'],f['size']);base=ROOT/'data/remediation_holdout'
    with py7zr.SevenZipFile(remote,'r') as z:
        lookup={x.filename:x for x in z.files};crc={p:lookup[p].crc32 for p in selected}
        for p in selected:
            assert '..' not in Path(p).parts and not Path(p).is_absolute() and p.endswith('/reports/report.json')
        todo=[p for p in selected if not (base/p).exists() or zlib.crc32((base/p).read_bytes())!=crc[p]]
        print('Acquiring',len(todo),'JSON reports; cached archive ranges reused',flush=True)
        if todo:z.extract(path=base,targets=todo)
    remote.close();files=[]
    for p in selected:
        b=(base/p).read_bytes();assert zlib.crc32(b)==crc[p]
        files.append({'path':(base/p).relative_to(ROOT).as_posix(),'archive_member':p,'sha256':hashlib.sha256(b).hexdigest(),'crc32':crc[p],'bytes':len(b)})
    manifest.write_text(json.dumps({'source':old['source'],'protocol_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'files':files,'verification':'Each selected member CRC and local SHA256; no executable members extracted'},indent=2))
    print('Verified new holdout reports',len(files),flush=True)

if __name__=='__main__':main()

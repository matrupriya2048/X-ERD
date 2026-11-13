from pathlib import Path
import collections, json, hashlib
import numpy as np
import pandas as pd
import joblib
from xerd.io import digest,save,load
from xerd.external_raw import interval
from evaluate_mtm import target_hashes

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def run():
    protocol=json.loads((HERE/'protocol.json').read_text());result=json.loads((HERE/'results.json').read_text())
    assert digest(HERE/'protocol.json')==result['protocol_sha256']
    assert digest(HERE/'evaluate_mtm.py')==result['source_script_sha256']
    audit=json.loads((HERE/'cohort_audit.json').read_text());receipts=json.loads((HERE/'acquisition.json').read_text())
    assert len(audit)==len(receipts)==len(protocol['files'])==200
    assert {x['path'] for x in receipts}=={x['path'] for x in protocol['files']}
    for r in receipts:
        assert r['status']=='acquired' and r['git_blob_sha1_verified']
        assert digest(HERE/'raw'/r['path'])==r['sha256']
    assert dict(collections.Counter(r['status'] for r in audit))==result['status_counts']
    eligible=[r for r in audit if r['status']=='evaluated']
    seen=set()
    for r in eligible:
        h=target_hashes({k:r.get('target_'+k) for k in ['md5','sha1','sha256']})
        assert h and not h&seen
        seen.update(h)
    predictions=pd.read_csv(HERE/'predictions.csv');vectors=pd.read_csv(HERE/'vectors.csv');ds=load(ROOT/'data/mlran')
    assert predictions.source_path.tolist()==vectors.source_path.tolist()==[r['path'] for r in eligible]
    assert list(vectors.columns[1:])==ds.columns
    assert np.isin(vectors[ds.columns].to_numpy(),[0,1]).all()
    for label,item in protocol['models'].items():
        assert digest(ROOT/item['path'])==item['sha256']
        replay=joblib.load(ROOT/item['path']).probabilities(vectors[ds.columns].to_numpy())['X-ERD']
        assert np.allclose(replay,predictions[label],rtol=0,atol=1e-12)
        k=int((predictions[label]>=0.5).sum());n=len(predictions)
        assert k==result['models'][label]['alert_n'] and n==result['models'][label]['n']
        assert abs(k/n-result['models'][label]['label_relative_fpr'])<1e-15
        assert np.allclose(interval(k,n),result['models'][label]['binomial_ci95'],rtol=0,atol=1e-15)
    readiness=json.loads((HERE/'annotation_readiness.json').read_text())
    assert digest(HERE/'evidence_packets.py')==readiness['source_sha256']
    assert readiness['completed_human_reviews']==readiness['verified_onset_annotations']==0
    out={'status':'passed','reports_integrity_checked':200,'evaluated_unique_target_files':len(eligible),
         'models_replayed_from_saved_vectors':3,'maximum_allowed_replay_error':1e-12,
         'numeric_antivirus_positive_fields':sum(isinstance(r.get('antivirus_positive_count'),(int,float)) for r in eligible),
         'missing_antivirus_positive_fields':sum(r.get('antivirus_positive_count') is None for r in eligible),
         'target_file_type_counts':dict(collections.Counter(r.get('target_type') for r in eligible)),
         'followup_source_hashes':{p.name:digest(p) for p in sorted(HERE.glob('*.py')) if p.name!='build_documents.py'},
         'scope':'Artifact integrity, cohort accounting and prediction replay; not proof of human participation or clean-label ground truth.'}
    groups=collections.defaultdict(list)
    for row in eligible:
        typ=row.get('target_type','')
        group='PE executable' if typ.startswith('PE32') else 'HTML' if typ.startswith('HTML') else 'PDF' if typ.startswith('PDF') else 'other'
        groups[group].append(row['path'])
    indexed=predictions.set_index('source_path')
    sensitivity={'scope':'Post-hoc descriptive file-type sensitivity; original primary cohort and model retained. Target-file hashes are not necessarily executable hashes.',
                 'groups':{g:{'n':len(ids),'alerts':{m:int((indexed.loc[ids,m]>=.5).sum()) for m in ['original','previous','adapted']}} for g,ids in groups.items()}}
    save(HERE/'file_type_sensitivity.json',sensitivity)
    save(HERE/'verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='target_file_type_counts'},indent=2))

if __name__=='__main__':run()

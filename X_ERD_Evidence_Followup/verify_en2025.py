from pathlib import Path
import json,hashlib,collections,joblib
import numpy as np
import pandas as pd
from primary_pair import probabilities
from xerd.io import digest,save,load
from xerd.external_raw import interval
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'closure_results'
r=json.loads((OUT/'en2025_results.json').read_text());protocol=json.loads((HERE/'en2025_protocol.json').read_text())
assert r['protocol_sha256']==digest(HERE/'en2025_protocol.json')
assert r['script_sha256']==digest(HERE/'evaluate_en2025.py')
assert hashlib.md5((HERE/'data_en2025/benign.7z').read_bytes()).hexdigest()==protocol['source_md5']
audit=json.loads((OUT/'en2025_audit.json').read_text());assert len(audit)==len(protocol['files'])==918
assert dict(collections.Counter(x['status'] for x in audit))==r['status_counts']
rows=[x for x in audit if x['status']=='evaluated'];pred=pd.read_csv(OUT/'en2025_predictions.csv');frame=pd.read_csv(OUT/'en2025_vectors.csv')
assert pred.source_path.tolist()==frame.source_path.tolist()==[x['path'] for x in rows]
assert len({x['target_sha256'] for x in rows})==len(rows)
ds=load(ROOT/'data/mlran');assert list(frame.columns[1:])==ds.columns;X=frame[ds.columns].to_numpy()
assert np.isin(X,[0,1]).all()
models={'primary_RF+LGB':joblib.load(OUT/'primary_pair.joblib'),'original_three':joblib.load(ROOT/'results/model.joblib'),'adapted_three':joblib.load(ROOT/'results/remediation/model.joblib')}
for key,model in models.items():
    p=probabilities(model,X) if key=='primary_RF+LGB' else model.probabilities(X)['X-ERD']
    assert np.allclose(p,pred[key],rtol=0,atol=1e-12)
    n=len(p);k=int((p>=.5).sum());assert r['models'][key]['n']==n and r['models'][key]['alert_n']==k
    assert np.allclose(r['models'][key]['ci95'],interval(k,n),rtol=0,atol=1e-15)
save(OUT/'en2025_verification.json',{'status':'passed','selected_reports_accounted':918,'eligible_unique_pe_targets':len(rows),
     'published_archive_md5_verified':True,'model_replays':3,'label_source':'publisher benign folder, not newly human-adjudicated',
     'recorded_malstatus_counts':dict(collections.Counter(str(x.get('malstatus')) for x in rows)),
     'source_hashes':{p.name:digest(p) for p in [HERE/'evaluate_en2025.py',Path(__file__)]}})
print('Passed EN2025 cohort accounting, archive checksum, independent target hashes and three-model prediction replay.')

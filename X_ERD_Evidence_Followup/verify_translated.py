from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
from primary_pair import probabilities
from xerd.io import load,digest,save
from xerd.external_raw import interval
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'closure_results'
r=json.loads((OUT/'en2025_translated_results.json').read_text());assert r['adapter_sha256']==digest(HERE/'cape_adapter.py');assert r['script_sha256']==digest(HERE/'evaluate_cape_adapter.py')
a=pd.read_csv(OUT/'en2025_translated_vectors.csv');p=pd.read_csv(OUT/'en2025_translated_predictions.csv');raw=pd.read_csv(OUT/'en2025_vectors.csv');ds=load(ROOT/'data/mlran')
assert len(a)==len(p)==917 and a.source_path.tolist()==p.source_path.tolist()==raw.source_path.tolist()
api=[c for c,n in zip(ds.columns,ds.names) if n.startswith('API:')];assert np.array_equal(a[api],raw[api])
X=a[ds.columns].to_numpy();assert np.isin(X,[0,1]).all()
models={'primary_RF+LGB':joblib.load(OUT/'primary_pair.joblib'),'original_three':joblib.load(ROOT/'results/model.joblib'),'adapted_three':joblib.load(ROOT/'results/remediation/model.joblib')}
for name,m in models.items():
    q=probabilities(m,X) if name=='primary_RF+LGB' else m.probabilities(X)['X-ERD'];assert np.allclose(q,p[name],rtol=0,atol=1e-12)
    k=int((q>=.5).sum());assert k==r['models'][name]['alert_n'];assert np.allclose(interval(k,len(q)),r['models'][name]['ci95'],rtol=0,atol=1e-15)
save(OUT/'en2025_translated_verification.json',{'status':'passed','targets':917,'same_cohort':True,'api_presence_unchanged':True,'models_replayed':3,'scope':'Paired post-hoc schema repair; original high false-alarm rates retained.'})
print('Verified all 917 translated vectors, unchanged API presence and three-model predictions.')

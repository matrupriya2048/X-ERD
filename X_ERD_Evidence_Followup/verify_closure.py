from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from primary_pair import probabilities
from xerd.io import load,digest,save
from xerd.metrics import metrics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'closure_results'
ds=load(ROOT/'data/mlran');r=json.loads((OUT/'primary_pair_results.json').read_text())
assert r['source_sha256']==digest(HERE/'close_available_work.py')
assert r['model_sha256']==digest(ROOT/'results/model.joblib')
artifact=joblib.load(OUT/'primary_pair.joblib');assert set(artifact['base'])=={'RF','LGB'}
p=probabilities(artifact,ds.Xt);container=pd.read_csv(OUT/'primary_container_predictions.csv')
assert np.allclose(p,container.probability,rtol=0,atol=1e-12)
assert np.array_equal(p>=.5,container.detected)
for k,v in metrics(ds.yt,p).items():
    expected=r['results']['official']['metrics'][k]
    assert expected==v or (isinstance(v,float) and abs(v-expected)<1e-12),(k,v,expected)
with np.load(OUT/'primary_pair_shap.npz') as z:
    error=float(abs(expit(z['baseline']+z['logit_contributions'].sum(1))-p).max());assert error<1e-7
impact=json.loads((OUT/'impact_proxy.json').read_text());cases=pd.read_csv(OUT/'impact_proxy_per_case.csv')
det=cases[(cases.y_true==1)&cases.detected];assert len(det)==455
assert int(det.proxy_active.sum())==impact['detected_with_active_proxy_n']==19
assert int((det.proxy_masked_probability<.5).sum())==impact['detections_lost_when_proxy_zeroed']==0
assert abs(det.positive_active_logit_mass_fraction.mean()-impact['positive_active_logit_mass_mean'])<1e-12
windows=json.loads((OUT/'sequence_window_controls.json').read_text())
assert len(windows['rows'])==24
assert all(x['benign_n']==20 and x['ransomware_n']==60 for x in windows['rows'])
for row in windows['rows']:
    assert row['metrics']['fp']==row['benign_fp'] and row['metrics']['tp']==row['ransomware_tp']
save(OUT/'verification.json',{'status':'passed','primary_artifact_sha256':digest(OUT/'primary_pair.joblib'),
     'docker_prediction_rows':len(container),'third_base_estimator_present':False,
     'max_pair_reconstruction_error':error,'window_groups_verified':24,'ransomware_reliance_rows_verified':455,
     'source_hashes':{p.name:digest(p) for p in [HERE/'primary_pair.py',HERE/'close_available_work.py',Path(__file__)]}})
print('Passed primary-only Docker inference, 975 prediction identities, pair explanation identity, 455 reliance rows and 24 sequence-control groups.')

from pathlib import Path
import hashlib,json,joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xerd.io import load,save,digest
from xerd.metrics import metrics
from primary_pair import probabilities
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'multicollection'
protocol=json.loads((OUT/'protocol.json').read_text());frozen=json.loads((OUT/'frozen_model.json').read_text());result=json.loads((OUT/'results.json').read_text())
assert digest(OUT/'model.joblib')==frozen['sha256'];assert digest(OUT/'protocol.json')==frozen['protocol_sha256']
assert digest(HERE/'train_multicollection_pair.py')==frozen['script_sha256']
en=pd.read_csv(HERE/'closure_results/en2025_translated_vectors.csv');ds=load(ROOT/'data/mlran')
groups=np.array([hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in en[ds.columns].to_numpy()])
a,b=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=20261113).split(en,groups=groups))
assert en.source_path.iloc[a].tolist()==protocol['train_paths'];assert en.source_path.iloc[b].tolist()==protocol['test_paths']
assert len(a)==634 and len(b)==283 and not set(groups[a])&set(groups[b])
oof=pd.read_csv(OUT/'training_oof.csv');assert len(oof)==4721 and oof[['RF_oof','LGB_oof']].notna().all().all()
assert oof.groupby('group').inner_fold.nunique().max()==1
assert set(oof.inner_fold)==set(range(5));assert sum(oof.domain==2)==634
assert set(oof.loc[oof.domain==2,'group'])=={'CAPEvector:'+x for x in groups[a]}
assert not {'CAPEvector:'+x for x in groups[b]}&set(oof.group)
model=joblib.load(OUT/'model.joblib');assert set(model['base'])=={'RF','LGB'}
inputs={'en_retrospective_holdout':(en.iloc[b],np.zeros(len(b),int)),
        'official_reused_regression':(pd.DataFrame(ds.Xt,columns=ds.columns),ds.yt),
        'prior_nunes_holdout':(pd.read_csv(ROOT/'results/remediation/holdout_vectors.csv'),np.zeros(362,int)),
        'prior_zirari_ransomware':(pd.read_csv(ROOT/'results/remediation/ransomware_vectors.csv'),np.ones(58,int))}
for key,(frame,y) in inputs.items():
    p=probabilities(model,frame[ds.columns].to_numpy());saved=pd.read_csv(OUT/(key+'_predictions.csv'))
    assert np.allclose(p,saved.probability,rtol=0,atol=1e-12)
    for k,v in metrics(y,p).items():
        expected=result['metrics'][key][k]
        assert v==expected or (isinstance(v,float) and abs(v-expected)<1e-12),(key,k)
old=pd.read_csv(HERE/'closure_results/en2025_translated_predictions.csv').set_index('source_path').loc[protocol['test_paths']]
audit={x['path']:x for x in json.loads((HERE/'closure_results/en2025_audit.json').read_text())}
holdstatus=pd.Series([str(audit[x].get('malstatus')) for x in protocol['test_paths']])
holdp=pd.read_csv(OUT/'en_retrospective_holdout_predictions.csv').probability.to_numpy()
dev=pd.read_csv(ROOT/'results/remediation/development_vectors.csv')
fitvectors={bytes(x.astype(np.int8)) for x in np.vstack([ds.X,dev[ds.columns].to_numpy(),en.iloc[a][ds.columns].to_numpy()])}
matches=sum(bytes(x.astype(np.int8)) in fitvectors for x in en.iloc[b][ds.columns].to_numpy())
info={'status':'passed','source_train_n':3905,'prior_external_development_n':182,'en_development_n':634,'en_holdout_n':283,
      'en_train_test_exact_vector_overlap':0,'en_holdout_vector_matches_to_any_fit_row':matches,
      'source_groups_crossing_inner_folds':0,'holdout_in_oof':False,'model_replayed_on_cohorts':list(inputs),
      'same_holdout_old_models':{k:int((old[k]>=.5).sum()) for k in ['primary_RF+LGB','original_three','adapted_three']},
      'publisher_status_sensitivity':{k:{'n':int((holdstatus==k).sum()),'alerts':int(((holdstatus==k).to_numpy()&(holdp>=.5)).sum())} for k in sorted(set(holdstatus))},
      'sha256':digest(OUT/'model.joblib'),'scope':'Retrospective adaptation holdout and reused external/source checks; no prospectively untouched collection or independent class adjudication.'}
reproduction=HERE/'multicollection_reproduction/model.joblib'
if reproduction.exists():
    assert digest(reproduction)==digest(OUT/'model.joblib');info['fresh_retraining_identical_model_sha256']=True
save(OUT/'verification.json',info);print(json.dumps(info,indent=2))

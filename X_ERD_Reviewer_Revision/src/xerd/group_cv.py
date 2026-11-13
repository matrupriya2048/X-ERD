"""Executable-grouped sensitivity analysis for the original fixed architecture."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .models import Ensemble,CONFIGS,SEEDS
from .optimize import base_oof,head
from .metrics import metrics,paired,holm
from .io import save

def run(ds,root,out):
    groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    save(out/'group_cv_protocol.json',{'duplicate_training_records':len(groups)-len(set(groups)),
      'groups':'executable MD5 in both outer and inner folds','outer':'five StratifiedGroupKFold folds per each of ten original seeds',
      'inner':'fresh five-fold StratifiedGroupKFold seed42 OOF per outer fit','base_seed':42,'meta_C':1,
      'scope':'sensitivity analysis; official483 representation remains fixed and upstream feature selection is not repeated',
      'statistical_note':'NB correction1/50+1/4 is approximate; dependent repeated group folds are not independent experiments'})
    rows=[]
    for seed in SEEDS:
        print('Grouped CV seed',seed,flush=True)
        for f,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(ds.X,ds.y,groups)):
            if set(groups[a])&set(groups[b]):raise AssertionError('Outer duplicate executable leakage')
            z,inner,models=base_oof(ds.X[a],ds.y[a],{},groups[a])
            heads={n:head(z[:,ix],ds.y[a],1) for n,ix in CONFIGS.items()};m=Ensemble(models,heads,z,inner,42)
            p=m.probabilities(ds.X[b]);rows.append({'seed':seed,'fold':f,'train_ids':ds.train.sample_id.iloc[a].tolist(),'test_ids':ds.train.sample_id.iloc[b].tolist(),
              'inner_folds':inner.tolist(),'metrics':{n:metrics(ds.y[b],v) for n,v in p.items()}})
        save(out/'group_cv_folds.json',rows)
    result={}
    for endpoint in ['fpr','recall']:
        comp={n:paired([r['metrics']['X-ERD'][endpoint] for r in rows],[r['metrics'][n][endpoint] for r in rows]) for n in ['RF','XGB','LGB','RF+LGB']}
        for n,p in zip(comp,holm([r['nb_p'] for r in comp.values()])):comp[n]['nb_holm_p']=float(p)
        result[endpoint]={'comparisons':comp,'mean_by_model':{n:float(np.mean([r['metrics'][n][endpoint] for r in rows])) for n in rows[0]['metrics']}}
    save(out/'group_cv.json',result)

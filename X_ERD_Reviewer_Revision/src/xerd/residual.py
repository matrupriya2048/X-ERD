"""Bounded exploratory feature-passthrough extension to the existing stack.

Feature passthrough is an established stacking option, not a new algorithm.
All original classifiers/results remain untouched; no test score selects a row.
"""
from dataclasses import dataclass
from pathlib import Path
import json,time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedGroupKFold
from .models import CONFIGS
from .optimize import base_oof,head,candidates
from .metrics import metrics,bootstrap
from .io import save,digest

@dataclass
class ResidualEnsemble:
    base:dict
    meta:dict
    def probabilities(self,X):
        z=np.column_stack([m.predict_proba(X)[:,1] for m in self.base.values()]);result={n:z[:,i] for i,n in enumerate(self.base)}
        result.update({n:m.predict_proba(np.column_stack([z[:,CONFIGS[n]],X]))[:,1] for n,m in self.meta.items()})
        return result

def run(ds,root,out):
    root=Path(root);out=Path(out);start=time.time();selected=json.loads((out/'optimization.json').read_text())['selection']
    configs={'original':candidates()['original'],selected['candidate']:candidates()[selected['candidate']]};cs=[.01,.1,1.,10.]
    groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ds.X,ds.y,groups))
    save(out/'residual_protocol.json',{'source_sha256':digest(Path(__file__)),'base_configurations':configs,'meta_C':cs,
      'head_input':'three nested OOF base probabilities followed by original483 binary features; no output feature scaling',
      'selection':'maximize pooled outer-OOF accuracy among recall>=0.97; ties lower FPR then declared order. If none feasible maximize recall then accuracy.',
      'folds':'five MD5-grouped outer folds,seed42; fresh five MD5-grouped inner OOF folds per outer fit',
      'scope':'Exploratory follow-up after original benchmark and fixed-architecture search. The official test is reused, not a fresh blind trial. Passthrough is established,not claimed as a novel classifier.',
      'controls':'standalone logistic C grid on identical training folds; pairwise passthrough heads reported at selected C; source feature selection held fixed'})
    allp={};logs=[]
    for name,config in configs.items():
        print('Passthrough training search',name,flush=True);pred={c:np.zeros(len(ds.y)) for c in cs}
        for fold,(a,b) in enumerate(folds):
            oof,inner,base=base_oof(ds.X[a],ds.y[a],config,groups[a]);z=np.column_stack([m.predict_proba(ds.X[b])[:,1] for m in base.values()])
            for c in cs:pred[c][b]=head(np.column_stack([oof,ds.X[a]]),ds.y[a],c).predict_proba(np.column_stack([z,ds.X[b]]))[:,1]
            logs.append({'candidate':name,'fold':fold,'train_ids':ds.train.sample_id.iloc[a].tolist(),'validation_ids':ds.train.sample_id.iloc[b].tolist(),'inner_folds':inner.tolist()})
        for c,v in pred.items():allp[(name,c)]=v
        save(out/'residual_fold_log.json',logs)
    controls={}
    for c in cs:
        p=np.zeros(len(ds.y))
        for a,b in folds:p[b]=head(ds.X[a],ds.y[a],c).predict_proba(ds.X[b])[:,1]
        controls[c]=p
    scores=[{'candidate':k,'C':c,**metrics(ds.y,v)} for (k,c),v in allp.items()]
    eligible=[r for r in scores if r['recall']>=.97];win=max(eligible,key=lambda r:(r['accuracy'],-r['fpr'])) if eligible else max(scores,key=lambda r:(r['recall'],r['accuracy']))
    control_scores=[{'C':c,**metrics(ds.y,p)} for c,p in controls.items()];ce=[r for r in control_scores if r['recall']>=.97]
    cw=max(ce,key=lambda r:(r['accuracy'],-r['fpr'])) if ce else max(control_scores,key=lambda r:(r['recall'],r['accuracy']))
    save(out/'residual_training_selection.json',{'selected':win,'scores':scores,'logistic_selected':cw,'logistic_scores':control_scores})
    pd.DataFrame({'sample_id':ds.train.sample_id,'y_true':ds.y,**{k+'__C'+str(c):v for (k,c),v in allp.items()},**{'LR__C'+str(c):v for c,v in controls.items()}}).to_csv(out/'residual_training_predictions.csv',index=False)
    oof,inner,base=base_oof(ds.X,ds.y,configs[win['candidate']],groups)
    meta={n:head(np.column_stack([oof[:,ix],ds.X]),ds.y,win['C']) for n,ix in CONFIGS.items()}
    model=ResidualEnsemble(base,meta);joblib.dump(model,out/'residual_model.joblib')
    p=model.probabilities(ds.Xt);p['Training-selected LR']=head(ds.X,ds.y,cw['C']).predict_proba(ds.Xt)[:,1]
    pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,**p}).to_csv(out/'residual_test_predictions.csv',index=False)
    original=pd.read_csv(root/'results/test_predictions.csv')['X-ERD'].to_numpy();a=(original>=.5)==ds.yt;b=(p['X-ERD']>=.5)==ds.yt
    discord=[int((a&~b).sum()),int((~a&b).sum())];rng=np.random.default_rng(42);delta=[]
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a));delta.append(float(b[ix].mean()-a[ix].mean()))
    save(out/'residual.json',{'selection':win,'logistic_selection':cw,'models':{n:metrics(ds.yt,v) for n,v in p.items()},'full_ci95':bootstrap(ds.yt,p['X-ERD']),
      'accuracy_change_vs_original':float(b.mean()-a.mean()),'paired_accuracy_change_ci95':np.quantile(delta,[.025,.975]),
      'mcnemar_vs_original':{'discordant':discord,'p':float(binomtest(discord[0],sum(discord),.5).pvalue) if sum(discord) else 1.},
      'seconds':time.time()-start})
    print('Passthrough selected',win['candidate'],win['C'],'test',metrics(ds.yt,p['X-ERD']),flush=True)

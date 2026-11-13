"""Training-only configuration search preserving the original three-base stack.

The official test has been reported before this extension, so it is a reused
benchmark, not a newly blinded prospective sample. Its scores never select a row.
"""
from pathlib import Path
import time,json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from .models import bases,Ensemble,CONFIGS
from .metrics import metrics,bootstrap
from scipy.stats import binomtest
from .io import save,digest

def candidates():
    return {
      'original':{},
      'more_trees':{'RF':{'n_estimators':400},'XGB':{'n_estimators':300,'learning_rate':.05},'LGB':{'n_estimators':300,'learning_rate':.05}},
      'shallow_boosting':{'RF':{'n_estimators':300},'XGB':{'n_estimators':300,'learning_rate':.05,'max_depth':3},'LGB':{'n_estimators':300,'learning_rate':.05,'num_leaves':15}},
      'regularized':{'RF':{'n_estimators':300,'min_samples_leaf':2},'XGB':{'n_estimators':300,'learning_rate':.05,'max_depth':4,'reg_lambda':5,'min_child_weight':3},'LGB':{'n_estimators':300,'learning_rate':.05,'num_leaves':15,'min_child_samples':30,'reg_lambda':5}},
      'row_column_sampling':{'RF':{'n_estimators':300,'max_features':.3},'XGB':{'n_estimators':300,'learning_rate':.05,'max_depth':4,'subsample':.8,'colsample_bytree':.8},'LGB':{'n_estimators':300,'learning_rate':.05,'num_leaves':15,'colsample_bytree':.8,'subsample':.8,'subsample_freq':1}},
      'conservative':{'RF':{'n_estimators':300,'max_depth':16,'min_samples_leaf':2},'XGB':{'n_estimators':200,'learning_rate':.05,'max_depth':3,'reg_lambda':10},'LGB':{'n_estimators':200,'learning_rate':.05,'num_leaves':7,'reg_lambda':10}},
      'light_regularization':{'RF':{'n_estimators':300},'XGB':{'n_estimators':200,'learning_rate':.1,'max_depth':4,'reg_lambda':2},'LGB':{'n_estimators':200,'learning_rate':.05,'num_leaves':31,'reg_lambda':2}},
      'strong_random_forest':{'RF':{'n_estimators':500,'max_features':.3},'XGB':{'n_estimators':200,'learning_rate':.1,'max_depth':6},'LGB':{'n_estimators':200,'learning_rate':.05,'num_leaves':31}},
    }

def configured(config):
    models=bases()
    for name,params in config.items():models[name].set_params(**params)
    return models

def base_oof(X,y,config,groups=None):
    oof=np.zeros((len(y),3));folds=np.zeros(len(y),dtype=int)
    splits=(StratifiedKFold(5,shuffle=False).split(X,y) if groups is None else
            StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups))
    for f,(a,b) in enumerate(splits):
        if groups is not None and set(groups[a])&set(groups[b]):raise AssertionError('Inner executable leakage')
        folds[b]=f
        for j,model in enumerate(configured(config).values()):
            model.fit(X[a],y[a]);oof[b,j]=model.predict_proba(X[b])[:,1]
    fitted=configured(config)
    for model in fitted.values():model.fit(X,y)
    return oof,folds,fitted

def head(oof,y,c):
    return LogisticRegression(C=c,solver='lbfgs',max_iter=1000,random_state=42).fit(oof,y)

def run(ds,root,out):
    root=Path(root);out=Path(out);start=time.time();configs=candidates();cs=[.1,1.,10.]
    protocol={'scope':'Additional exploratory optimization; original architecture and all original results preserved',
      'source_sha256':digest(Path(__file__)),'candidates':configs,'meta_C':cs,
      'selection':'maximize pooled training outer-OOF accuracy among recall>=0.97; ties lower FPR then lower candidate index/C index; if none feasible maximize recall then accuracy',
      'validation':'five shuffled stratified GROUP outer folds seed42; groups are executable MD5; each outer-training fit uses fresh five-fold GROUP inner OOF predictions; fixed threshold0.5',
      'feature_scope':'official483 training-selected representation held fixed; outer CV conditional on it, not nested upstream feature selection',
      'test_scope':'official test previously reported in revision; not fresh prospective validation. No test scores participate in configuration selection',
      'seed':42,'official_split':ds.manifest}
    save(out/'optimization_protocol.json',protocol)
    groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ds.X,ds.y,groups));allp={};logs=[]
    for k,config in configs.items():
        print('Training-only search',k,flush=True);pred={c:np.zeros(len(ds.y)) for c in cs}
        for f,(a,b) in enumerate(folds):
            if set(groups[a])&set(groups[b]):raise AssertionError('Outer executable leakage')
            oof,inner,models=base_oof(ds.X[a],ds.y[a],config,groups[a])
            z=np.column_stack([m.predict_proba(ds.X[b])[:,1] for m in models.values()])
            for c in cs:pred[c][b]=head(oof,ds.y[a],c).predict_proba(z)[:,1]
            logs.append({'candidate':k,'fold':f,'train_ids':ds.train.sample_id.iloc[a].tolist(),'validation_ids':ds.train.sample_id.iloc[b].tolist(),'inner_folds':inner.tolist()})
        for c,p in pred.items():allp[(k,c)]=p
        save(out/'optimization_fold_log.json',logs)
    scores=[{'candidate':k,'C':c,**metrics(ds.y,p)} for (k,c),p in allp.items()]
    eligible=[r for r in scores if r['recall']>=.97]
    winner=max(eligible,key=lambda r:(r['accuracy'],-r['fpr'])) if eligible else max(scores,key=lambda r:(r['recall'],r['accuracy']))
    save(out/'optimization_training_selection.json',{'selected':winner,'constraint_feasible':bool(eligible),'scores':scores})
    pd.DataFrame({'sample_id':ds.train.sample_id,'y_true':ds.y,**{k+'__C'+str(c):v for (k,c),v in allp.items()}}).to_csv(out/'optimization_training_predictions.csv',index=False)
    # Configuration is now fixed; test labels are first used below for reporting.
    config=configs[winner['candidate']];oof,inner,models=base_oof(ds.X,ds.y,config,groups)
    heads={n:head(oof[:,ix],ds.y,winner['C']) for n,ix in CONFIGS.items()}
    model=Ensemble(models,heads,oof,inner,42);joblib.dump(model,out/'optimized_model.joblib')
    pred=model.probabilities(ds.Xt);original=pd.read_csv(root/'results/test_predictions.csv')['X-ERD'].to_numpy()
    pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,**pred}).to_csv(out/'optimized_test_predictions.csv',index=False)
    rng=np.random.default_rng(42);delta=[]
    correct=(pred['X-ERD']>=.5)==ds.yt;old_correct=(original>=.5)==ds.yt
    for _ in range(2000):
        ix=rng.integers(0,len(ds.yt),len(ds.yt));delta.append(float(correct[ix].mean()-old_correct[ix].mean()))
    discord=[int((old_correct&~correct).sum()),int((~old_correct&correct).sum())]
    result={'selection':winner,'models':{n:metrics(ds.yt,p) for n,p in pred.items()},'full_ci95':bootstrap(ds.yt,pred['X-ERD']),
      'accuracy_change_vs_original':float(correct.mean()-old_correct.mean()),'paired_accuracy_change_ci95':np.quantile(delta,[.025,.975]),
      'mcnemar_vs_original':{'discordant':discord,'p':float(binomtest(discord[0],sum(discord),.5).pvalue) if sum(discord) else 1.},
      'interpretation':'Selected using training CV only. Reused benchmark and fixed representation limit generalization; superiority is not established just by a higher point estimate.',
      'seconds':time.time()-start}
    save(out/'optimization.json',result);print('Selected',winner['candidate'],winner['C'],'test',result['models']['X-ERD'],flush=True)

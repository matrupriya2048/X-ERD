from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from . import models
from .io import save,environment,digest
from .metrics import metrics,matched,bootstrap,paired,holm,delong

def core(ds,out,jobs=4):
    out=Path(out);m=models.fit(ds.X,ds.y,jobs=jobs);p=m.probabilities(ds.Xt)
    joblib.dump(m,out/'model.joblib')
    result={n:metrics(ds.yt,q) for n,q in p.items()}
    # Strong compatible benchmark baselines, not misattributed literature scores.
    for n,base in {'LogisticRegression':LogisticRegression(C=1,solver='liblinear',max_iter=10000,random_state=42),
                   'ExtraTrees':ExtraTreesClassifier(n_estimators=100,max_depth=20,random_state=42,n_jobs=jobs)}.items():
        base.fit(ds.X,ds.y);q=base.predict_proba(ds.Xt)[:,1];p[n]=q;result[n]=metrics(ds.yt,q)
    pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,**p}).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame({'sample_id':ds.train.sample_id,'y_true':ds.y,'inner_fold':m.inner_folds,**{n:m.oof[:,i] for i,n in enumerate(m.base)}}).to_csv(out/'training_oof.csv',index=False)
    a=p['X-ERD']>=.5;b=p['LGB']>=.5
    discord=[int(((a==ds.yt)&(b!=ds.yt)).sum()),int(((a!=ds.yt)&(b==ds.yt)).sum())]
    r={'metrics':result,'coefficients':m.coefficients(),'bootstrap_stratified_ci95':bootstrap(ds.yt,p['X-ERD']),
       'matched_descriptive':{str(t):{n:matched(ds.yt,q,t) for n,q in p.items()} for t in [.97,.98,455/465]},
       'standardized_partial_auc_005':{n:float(roc_auc_score(ds.yt,q,max_fpr=.05)) for n,q in p.items()},
       'mcnemar':{'discordant':discord,'p':float(stats.binomtest(discord[0],sum(discord),.5).pvalue) if sum(discord) else 1.},
       'delong':delong(ds.yt,p['X-ERD'],p['LGB']), 'data_manifest':ds.manifest,'environment':environment()}
    # Exact covariance identity, no inference about a causal stacking mechanism.
    z=np.column_stack([p[n] for n in ['RF','XGB','LGB']])[ds.yt==0]
    cov=np.cov(z,rowvar=False);r['probability_covariance']={'matrix':cov,'correlation':np.corrcoef(z,rowvar=False),
      'exact_variance_mean':float(cov.sum()/9),'empirical_variance_mean':float(z.mean(1).var(ddof=1)),
      'scope':'algebraic covariance identity for base probability mean, not logistic stack variance or causal explanation'}
    save(out/'core.json',r);return m

def cv(ds,out,jobs=4,repetitions=10):
    out=Path(out);path=out/'cv_folds.json'
    if not 1<=repetitions<=len(models.SEEDS):raise ValueError('repetitions must be 1..10')
    source_hashes={p.name:digest(p) for p in [Path(__file__),Path(models.__file__)]}
    previous=json.loads(path.read_text()) if path.exists() else None
    if previous and (previous['data_manifest']!=ds.manifest or previous.get('source_hashes')!=source_hashes):
        raise ValueError('Resume refused: data or training code differs; use a fresh results directory')
    rows=[r for r in previous['rows'] if r['seed'] in models.SEEDS[:repetitions]] if previous else []
    done={(r['seed'],r['fold']) for r in rows}
    for seed in models.SEEDS[:repetitions]:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(ds.X,ds.y)):
            if (seed,fold) in done:continue
            start=time.time();m=models.fit(ds.X[a],ds.y[a],seed,jobs)
            rows.append({'seed':seed,'fold':fold,'train_ids':ds.train.sample_id.to_numpy()[a],
                'validation_ids':ds.train.sample_id.to_numpy()[b],
                'models':{n:metrics(ds.y[b],q) for n,q in m.probabilities(ds.X[b]).items()}})
            save(path,{'representation_scope':'conditional on official fixed RFE representation; feature selection is not nested',
                       'inner_cv':'fresh 5-fold unshuffled OOF within every outer training fold',
                       'data_manifest':ds.manifest,'source_hashes':source_hashes,'rows':rows})
            print(f'CV seed {seed} fold {fold+1}: {time.time()-start:.1f}s',flush=True)
    summary={}
    for metric in ['fpr','recall']:
        s={n:paired([r['models']['X-ERD'][metric] for r in rows],[r['models'][n][metric] for r in rows])
           for n in ['RF','XGB','LGB','RF+LGB']}
        for (n,v),adj in zip(s.items(),holm([v['nb_p'] for v in s.values()])):v['nb_holm_p']=float(adj)
        summary[metric]={'family':'four competitors, endpoint-specific Holm adjustment','comparisons':s}
    save(out/'cv_summary.json',summary)

def thresholds(ds,m,out,jobs=4):
    a,b=train_test_split(np.arange(len(ds.y)),test_size=.2,stratify=ds.y,random_state=42)
    fit=models.fit(ds.X[a],ds.y[a],jobs=jobs);pv=fit.probabilities(ds.X[b]);pt=m.probabilities(ds.Xt)
    for name,base in {'LogisticRegression':LogisticRegression(C=1,solver='liblinear',max_iter=10000,random_state=42),
                      'ExtraTrees':ExtraTreesClassifier(n_estimators=100,max_depth=20,random_state=42,n_jobs=jobs)}.items():
        base.fit(ds.X[a],ds.y[a]);pv[name]=base.predict_proba(ds.X[b])[:,1]
        base.fit(ds.X,ds.y);pt[name]=base.predict_proba(ds.Xt)[:,1]
    result={};curves=[]
    for n,q in pv.items():
        rows=[metrics(ds.y[b],q,t) for t in np.round(np.arange(.05,.951,.005),3)]
        best=max(rows,key=lambda r:r['f1']);feasible=[r for r in rows if r['recall']>=.97]
        constrained=min(feasible,key=lambda r:(r['fpr'],-r['recall'],r['theta']))
        result[n]={'selection':'validation-only; full-training refit evaluated without test reselection',
            'validation_f1max':best,'validation_constrained':constrained,
            'test_default':metrics(ds.yt,pt[n]),'test_val_f1max':metrics(ds.yt,pt[n],best['theta']),
            'test_val_constrained':metrics(ds.yt,pt[n],constrained['theta'])}
        for r in rows:curves.append({'model':n,'split':'validation',**r})
        for t in np.round(np.arange(.05,.951,.005),3):curves.append({'model':n,'split':'test_descriptive',**metrics(ds.yt,pt[n],t)})
    save(Path(out)/'thresholds.json',result);pd.DataFrame(curves).to_csv(Path(out)/'threshold_curves.csv',index=False)

def corruption(ds,m,out):
    api=[i for i,n in enumerate(ds.names) if n.startswith('API:')]
    sleep=[i for i,n in enumerate(ds.names) if n.startswith('SIGNATURE:') and any(k in n.lower() for k in ['sleep','delay'])]
    vm=[i for i,n in enumerate(ds.names) if n.startswith('SIGNATURE:') and any(k in n.lower() for k in ['antivm','vbox','qemu','vmware'])]
    mimic=np.argsort(ds.X[ds.y==0].mean(0)-ds.X[ds.y==1].mean(0))[-10:]
    rows=[]
    for scenario in ['drop10','drop20','drop30','api10','api20','api30','sleep','antivm','mimicry']:
        for seed in range(100,110):
            rng=np.random.default_rng(seed);X=ds.Xt.copy()
            if scenario.startswith('drop'):X[rng.random(X.shape)<int(scenario[4:])/100]=0
            elif scenario.startswith('api'):X[:,rng.choice(api,int(len(api)*int(scenario[3:])/100),replace=False)]=0
            elif scenario=='sleep':X[:,sleep]=0
            elif scenario=='antivm':X[:,vm]=0
            else:X[np.ix_(np.where(ds.yt==1)[0],mimic)]=1
            for name,p in m.probabilities(X).items():rows.append({'scenario':scenario,'seed':seed,'model':name,**metrics(ds.yt,p)})
    df=pd.DataFrame(rows);df.to_csv(Path(out)/'corruption_runs.csv',index=False)
    records=df.groupby(['scenario','model'])[['recall','fpr']].agg(['mean','std']).reset_index()
    records.columns=['scenario','model','recall_mean','recall_sd','fpr_mean','fpr_sd'];records.to_csv(Path(out)/'corruption_summary.csv',index=False)
    save(Path(out)/'corruption_protocol.json',{'mask':'common per-record dropout / common whole API-column masking per seed',
      'seeds':list(range(100,110)),'api_indices':api,'sleep_indices':sleep,'vm_indices':vm,'mimic_indices':mimic,
      'claim_scope':'feature-space stress only; not validated executable evasion; masks fixed across models'})

def temporal(ds,out,jobs=4):
    df=pd.concat([ds.train,ds.test],ignore_index=True)
    df=df.merge(ds.metadata[['sample_id','first_submission_date_year']],on='sample_id',validate='one_to_one')
    a=df.first_submission_date_year<=2019;b=df.first_submission_date_year>=2020
    X=df[ds.columns].to_numpy();y=(df.type_label>0).to_numpy(dtype=int)
    m=models.fit(X[a],y[a],jobs=jobs);probs=m.probabilities(X[b]);years=df.loc[b,'first_submission_date_year'].to_numpy()
    result={'scope':'fixed official representation; original selection includes future records; NOT strict past-only feature selection',
      'train_ids':df.loc[a,'sample_id'].to_numpy(),'test_ids':df.loc[b,'sample_id'].to_numpy(),
      'metrics':{n:metrics(y[b],p) for n,p in probs.items()},'years':{}}
    for yr in sorted(set(years)):
        k=years==yr;result['years'][str(yr)]={n:metrics(y[b][k],p[k]) for n,p in probs.items()}
    for n in probs:
        for key in ['tn','fp','fn','tp']:
            assert sum(v[n][key] for v in result['years'].values())==result['metrics'][n][key]
    save(Path(out)/'temporal_fixed483.json',result)

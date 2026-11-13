"""Fixed-parameter pair adaptation after disclosed CAPE transport failure."""
from pathlib import Path
import datetime,hashlib,json,time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from xerd.io import load,save,digest,environment
from xerd.models import bases
from xerd.metrics import metrics
from xerd.external_raw import interval
from primary_pair import probabilities
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'multicollection'

def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'model.joblib').exists():raise ValueError('Existing fitted variant must not be overwritten after target evaluation')
    ds=load(ROOT/'data/mlran');en=pd.read_csv(HERE/'closure_results/en2025_translated_vectors.csv')
    audit={r['path']:r for r in json.loads((HERE/'closure_results/en2025_audit.json').read_text())}
    groups=np.array([hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in en[ds.columns].to_numpy()])
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=20261113).split(en,groups=groups))
    assert not set(groups[train])&set(groups[test])
    protocol={'frozen_before_fit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'reason':'The complete EN2025 cohort was previously inspected with old frozen models, revealing severe transport failure. This split is a retrospective internal adaptation holdout, not a new untouched prospective collection.',
      'model':'RF100 + LightGBM100; 5-fold executable/vector-group OOF; logistic C10 with raw-feature passthrough; weights source1, external-benign3, inherited from previous development configuration. No candidate search or threshold selection on EN holdout.',
      'threshold':.5,'source_train_n':len(ds.y),'prior_benign_development_n':182,'en_train_n':len(train),'en_test_n':len(test),
      'split_seed':20261113,'split_group':'exact parsed EN vectors to prevent identical vector leakage; source executable hashes used for OOF grouping',
      'en_vectors_sha256':digest(HERE/'closure_results/en2025_translated_vectors.csv'),
      'train_paths':en.source_path.iloc[train].tolist(),'test_paths':en.source_path.iloc[test].tolist(),
      'raw_ransomware_fit_rows':0,'scope':'Publisher-label specificity, source regression and earlier raw-ransomware sensitivity. No human adjudication, verified onset, or prospective deployment claim.'}
    save(OUT/'protocol.json',protocol)
    dev=pd.read_csv(ROOT/'results/remediation/development_vectors.csv');assert len(dev)==182
    X=np.vstack([ds.X,dev[ds.columns].to_numpy(),en[ds.columns].to_numpy()[train]])
    y=np.r_[ds.y,np.zeros(len(dev)+len(train),int)];domain=np.r_[np.zeros(len(ds.y),int),np.ones(len(dev),int),np.full(len(train),2,int)]
    source_md5=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    g=np.r_[source_md5,'Nunes:'+dev.md5.str.lower().to_numpy(),'CAPEvector:'+groups[train]]
    strata=np.where(domain==0,y,domain+1);weights=np.where(domain==0,1.,3.)
    oof=np.full((len(y),2),np.nan);folds=np.full(len(y),-1)
    for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,strata,g)):
        assert not set(g[a])&set(g[b]);folds[b]=fold
        for j,name in enumerate(['RF','LGB']):
            print('Training OOF',fold,name,'fit',len(a),'validation',len(b),flush=True)
            model=bases(jobs=4)[name];model.fit(X[a],y[a],sample_weight=weights[a]);oof[b,j]=model.predict_proba(X[b])[:,1]
    assert np.isfinite(oof).all() and (folds>=0).all()
    meta=LogisticRegression(C=10.,solver='lbfgs',max_iter=2000,random_state=42).fit(np.column_stack([oof,X]),y,sample_weight=weights)
    final={}
    for name in ['RF','LGB']:
        print('Refitting base',name,flush=True);model=bases(jobs=4)[name];model.fit(X,y,sample_weight=weights);final[name]=model
    artifact={'model':'RF+LGB multi-collection adaptation','base':final,'meta':meta,'passthrough':True,'feature_ids':ds.columns,'threshold':.5}
    joblib.dump(artifact,OUT/'model.joblib')
    pd.DataFrame({'domain':domain,'label':y,'group':g,'inner_fold':folds,'RF_oof':oof[:,0],'LGB_oof':oof[:,1]}).to_csv(OUT/'training_oof.csv',index=False)
    save(OUT/'frozen_model.json',{'frozen_before_scoring_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':digest(OUT/'model.joblib'),'protocol_sha256':digest(OUT/'protocol.json'),'script_sha256':digest(Path(__file__))})
    sets={'en_retrospective_holdout':(en.iloc[test],np.zeros(len(test),int)),
          'official_reused_regression':(pd.DataFrame(ds.Xt,columns=ds.columns),ds.yt),
          'prior_nunes_holdout':(pd.read_csv(ROOT/'results/remediation/holdout_vectors.csv'),np.zeros(362,int)),
          'prior_zirari_ransomware':(pd.read_csv(ROOT/'results/remediation/ransomware_vectors.csv'),np.ones(58,int))}
    results={}
    for key,(frame,labels) in sets.items():
        probability=probabilities(artifact,frame[ds.columns].to_numpy());m=metrics(labels,probability)
        if not np.any(labels):m['fpr_ci95']=interval(m['fp'],len(labels))
        elif np.all(labels):m['recall_ci95']=interval(m['tp'],len(labels))
        results[key]=m;pd.DataFrame({'y_true':labels,'probability':probability}).to_csv(OUT/(key+'_predictions.csv'),index=False)
    save(OUT/'results.json',{'metrics':results,'environment':environment(),'scope':protocol['scope'],'important':protocol['reason']})
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':run()

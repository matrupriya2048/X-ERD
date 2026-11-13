"""Published detector adaptations on the identical hash-novel transfer targets."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from .external_dataset import source_data
from .published import EldeRan
from .metrics import metrics,bootstrap
from .io import save

def run(ds,root,out):
    X,y,names,meta=source_data(root/'data/external_sources');target=pd.read_csv(out/'elderan_transfer_predictions.csv');ix=meta.set_index('ID').index.get_indexer(target.source_id)
    cols=pd.read_csv(out/'elderan_common_apis.csv');train=ds.X[:,cols.mlran_index];test=X[ix][:,cols.elderan_index]
    selected=json.loads((out/'published.json').read_text())['models']
    lr_c=selected['MLRan-LR']['best_parameters']['C'];lam=selected['EldeRan-adapted']['best_parameters']['regularization']
    save(out/'external_comparators_protocol.json',{'scope':'Same56 source APIs and898 deduplicated target executables as original transfer; no target-label tuning',
      'parameters':'RANSEC settings identical to same-split adaptations; LR C and EldeRan lambda transferred from official483 training-only search; not reoptimized on target',
      'LR_C':lr_c,'EldeRan_lambda':lam,'identity_check':'target source IDs exactly match elderan_transfer_predictions.csv'})
    svm=lambda:SVC(kernel='linear',C=1,probability=True,random_state=42)
    xgb=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.3,reg_lambda=1,random_state=42,n_jobs=4,eval_metric='logloss')
    models={
      'MLRan-LR':LogisticRegression(C=lr_c,solver='liblinear',max_iter=1000,random_state=42),
      'EldeRan-adapted':EldeRan(regularization=lam,k=400),
      'RANSEC-stack-adapted':StackingClassifier([('svm',svm()),('xgb',xgb)],final_estimator=LogisticRegression(C=.1,max_iter=1000),cv=5,stack_method='predict_proba',n_jobs=1)}
    predictions={}
    for name,model in models.items():
        print('External comparator',name,flush=True);model.fit(train,ds.y);predictions[name]=model.predict_proba(test)[:,1]
    a,b=train_test_split(np.arange(len(ds.y)),test_size=.2,stratify=ds.y,random_state=42)
    rf=lambda:RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=4)
    weights=np.array([np.mean(make().fit(train[a],ds.y[a]).predict(train[b])==ds.y[b]) for make in [rf,svm]])
    weights/=weights.sum();predictions['RANSEC-vote-adapted']=sum(w*make().fit(train,ds.y).predict_proba(test)[:,1] for w,make in zip(weights,[rf,svm]))
    pd.DataFrame({'source_id':target.source_id,'y_true':y[ix],**predictions}).to_csv(out/'external_comparators_predictions.csv',index=False)
    save(out/'external_comparators.json',{'models':{n:metrics(y[ix],v) for n,v in predictions.items()},'ci95':{n:bootstrap(y[ix],v) for n,v in predictions.items()},'vote_weights':weights})

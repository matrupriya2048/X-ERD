"""Published-method adaptations on the unchanged official MLRan partition.

No scores are imported from other papers. Differences from original procedures
are exported explicitly; these are reimplementations, not author-code replicas.
"""
from pathlib import Path
import time,json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from .additional import binary_mi
from .metrics import metrics,matched
from .io import save,digest

class EldeRan(ClassifierMixin,BaseEstimator):
    def __init__(self,regularization=.002,k=400):
        self.regularization=regularization;self.k=k
    def fit(self,X,y):
        X=np.asarray(X);self.n_features_in_=X.shape[1]
        mi=binary_mi(X,np.asarray(y))
        self.selected_=np.lexsort((np.arange(X.shape[1]),-mi))[:min(self.k,X.shape[1])]
        # sklearn summed loss + ||w||²/(2C) equals mean loss + lambda||w||²/2.
        C=1/(len(y)*self.regularization) if self.regularization else np.inf
        self.model_=LogisticRegression(C=C,solver='lbfgs',tol=1e-7,max_iter=4000).fit(X[:,self.selected_],y)
        self.classes_=self.model_.classes_;return self
    def predict_proba(self,X):return self.model_.predict_proba(np.asarray(X)[:,self.selected_])
    def predict(self,X):return self.classes_[(self.predict_proba(X)[:,1]>=.5).astype(int)]

def benchmark_grids(jobs=4):
    trees={'n_estimators':[50,100,200],'max_depth':[None,10,20],'min_samples_split':[2,5]}
    return {
      'MLRan-LGB':(LGBMClassifier(objective='binary',min_child_samples=10,min_split_gain=0,random_state=42,n_jobs=jobs,verbose=-1),{'num_leaves':[15,31],'n_estimators':[50,100],'learning_rate':[.01]}),
      'MLRan-DT':(DecisionTreeClassifier(random_state=42),{'max_depth':[None,10,20],'min_samples_split':[2,5,10]}),
      'MLRan-LR':(LogisticRegression(max_iter=1000,solver='liblinear',random_state=42),{'C':[.01,.1,1,10]}),
      'MLRan-RF':(RandomForestClassifier(random_state=42,n_jobs=jobs),trees),
      'MLRan-ET':(ExtraTreesClassifier(random_state=42,n_jobs=jobs),trees)}

def run(ds,out,jobs=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    protocol={'official_split':ds.manifest,'seed':42,'test_threshold':.5,
      'MLRan':{'source':'https://github.com/faithfulco/mlran/blob/main/6_experiments/1c_baseline_ML.ipynb',
        'selection':'all five published model grids; 3-fold unshuffled stratified inner CV, ROC-AUC; train only',
        'changes':'seed42 added for determinism; ransomware-positive metrics replace weighted class averages; library versions pinned; official483 representation retained'},
      'EldeRan':{'source':'https://arxiv.org/abs/1609.03020','top_k':400,'lambda_grid':[0,.00002,.0002,.002,.02,.2],
        'selection':'5-fold stratified inner CV; MI reselected inside EVERY fold; ROC-AUC objective',
        'changes':'official483 candidate pool and split instead of original30967 features; convex L2 objective solved by LBFGS instead of batch gradient descent; original whole-dataset hyperparameter search restricted to training data'},
      'RANSEC':{'source':'https://doi.org/10.38094/jastt71555','variants':['SVM+XGB logistic stack','RF+SVM validation-accuracy-weighted soft vote'],
        'specified':'linear SVM, RF100 depth10, logistic head L2 C0.1; min-max normalization unnecessary for binary indicators',
        'ambiguities':'paper describes SVM+XGB in abstract/stacking section but inconsistent diagram prose mentions RF+SVM. We implement the explicit SVM+XGB variant and the separately described RF+SVM vote. No author source code available.',
        'completion_choices':'SVM C1/probability calibration random_state42; XGB100 depth6 eta0.3 lambda1; 5 unshuffled inner OOF folds; original483 binary indicators; same official test',
        'claim_scope':'published architecture adaptations, not exact reproduction of original feature acquisition or CPS deployment'}}
    save(out/'published_protocol.json',protocol) # written before any test evaluation
    results={};pred={'sample_id':ds.test.sample_id,'y_true':ds.yt};tuning=[]
    for name,(est,grid) in benchmark_grids(jobs).items():
        print('Published baseline',name,flush=True)
        search=GridSearchCV(est,grid,cv=3,scoring='roc_auc',n_jobs=1).fit(ds.X,ds.y)
        p=search.predict_proba(ds.Xt)[:,1];pred[name]=p
        results[name]={'metrics':metrics(ds.yt,p),'matched97':matched(ds.yt,p,.97),'matched98':matched(ds.yt,p,.98),'best_parameters':search.best_params_,'inner_auc':search.best_score_}
        for pars,mean,sd in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):tuning.append({'model':name,'parameters':json.dumps(pars),'cv_auc':mean,'cv_sd':sd})
    print('Published baseline EldeRan',flush=True)
    search=GridSearchCV(EldeRan(),{'regularization':protocol['EldeRan']['lambda_grid']},cv=5,scoring='roc_auc',n_jobs=1).fit(ds.X,ds.y)
    p=search.predict_proba(ds.Xt)[:,1];pred['EldeRan-adapted']=p
    results['EldeRan-adapted']={'metrics':metrics(ds.yt,p),'matched97':matched(ds.yt,p,.97),'matched98':matched(ds.yt,p,.98),'best_parameters':search.best_params_,'selected_features':[ds.names[i] for i in search.best_estimator_.selected_]}
    for pars,mean,sd in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):tuning.append({'model':'EldeRan-adapted','parameters':json.dumps(pars),'cv_auc':mean,'cv_sd':sd})
    svm=lambda:SVC(kernel='linear',C=1,probability=True,random_state=42)
    xgb=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.3,reg_lambda=1,random_state=42,n_jobs=jobs,eval_metric='logloss')
    stack=StackingClassifier([('svm',svm()),('xgb',xgb)],final_estimator=LogisticRegression(C=.1,max_iter=1000),cv=5,stack_method='predict_proba',n_jobs=1)
    print('Published baseline RANSEC stack',flush=True);stack.fit(ds.X,ds.y)
    p=stack.predict_proba(ds.Xt)[:,1];pred['RANSEC-stack-adapted']=p
    results['RANSEC-stack-adapted']={'metrics':metrics(ds.yt,p),'matched97':matched(ds.yt,p,.97),'matched98':matched(ds.yt,p,.98),'meta_coefs':stack.final_estimator_.coef_[0]}
    a,b=train_test_split(np.arange(len(ds.y)),test_size=.2,stratify=ds.y,random_state=42)
    rf=lambda:RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=jobs)
    weights=np.array([np.mean(builder().fit(ds.X[a],ds.y[a]).predict(ds.X[b])==ds.y[b]) for builder in [rf,svm]])
    weights=weights/weights.sum();models=[builder().fit(ds.X,ds.y) for builder in [rf,svm]]
    p=sum(w*m.predict_proba(ds.Xt)[:,1] for w,m in zip(weights,models));pred['RANSEC-vote-adapted']=p
    results['RANSEC-vote-adapted']={'metrics':metrics(ds.yt,p),'matched97':matched(ds.yt,p,.97),'matched98':matched(ds.yt,p,.98),'weights_rf_svm':weights}
    pd.DataFrame(pred).to_csv(out/'published_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'published_tuning.csv',index=False)
    save(out/'published.json',{'models':results,'seconds':time.time()-start,'claim_scope':protocol['RANSEC']['claim_scope']})
    return results

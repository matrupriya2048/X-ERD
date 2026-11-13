import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from xerd.io import features,labels,load
from xerd.metrics import metrics,ece,matched,holm,delong,paired
from xerd.parser import vector,events
from xerd.mapping import map_features

def test_schema_rejects_accidental_last_feature_target():
    df=pd.DataFrame({'sample_id':[1,2],'sample_type':[0,1],'family_label':[0,5],'type_label':[0,2],'13':[1,0]})
    cols,X=features(df)
    assert cols==['13'] and X.shape==(2,1)
    assert labels(df).tolist()==[0,1] # The last feature is deliberately the opposite.
    df['sample_type']=[1,0]
    with pytest.raises(ValueError):labels(df)

def test_fpr_is_undefined_for_positive_only_reports():
    assert metrics(np.ones(20),np.ones(20))['fpr'] is None

def test_ece_includes_endpoints():
    assert ece([1,0],[0.,1.])==1
    assert ece([0,1],[0.,1.])==0

def test_exact_recall_target_not_rounded():
    y=np.r_[np.ones(465),np.zeros(510)];p=np.r_[np.linspace(.99,.01,465),np.zeros(510)]
    assert matched(y,p,455/465)['tp']==455
    assert matched(y,p,.9785)['tp']==456

def test_delong_identical_and_different_scores():
    y=np.array([0,0,0,1,1,1]);a=np.array([.1,.2,.4,.3,.8,.9])
    assert delong(y,a,a)['p']==1
    assert delong(y,a,a)['delta_auc']==0

def test_holm_monotone_order_independent():
    assert np.allclose(holm([.04,.001,.03]),[.06,.003,.06])

def test_paired_ties_and_effect_sign():
    r=paired([1,2,3,4],[1,3,4,5]);assert r['ties']==1 and r['paired_rank_biserial']==-1

def test_full_windows_dll_path_and_exact_file_matching():
    rep={'behavior':{'summary':{'dll_loaded':[r'C:\Windows\System32\ole32.dll'],'file_created':[r'C:\a\report.txt']}}}
    names=[r'SYSTEM:DLL_LOADED:c:\windows\system32\ole32.dll','SYSTEM:DLL_LOADED:ole32.dll',r'FILE:CREATED:c:\a\report.txt',r'FILE:CREATED:c:\a\report']
    assert vector(rep,names).tolist()==[1,0,1,0]

def test_raw_timestamp_path_sorted():
    rep={'behavior':{'processes':[{'calls':[{'time':2.,'api':'B'},{'time':1.01,'api':'A'}]}]}}
    assert events(rep)[0]['api']=='A'

def test_string_tokens_do_not_match_substrings_or_description_fields():
    rep={'strings':['AbcDef'],'description':'absent'}
    assert vector(rep,['STRING:abcdef','STRING:abc','STRING:absent','STRING:']).tolist()==[1,0,0,0]

def test_mapping_rwx_is_not_apc():
    schema=json.load(open('config/mapping_schema.json'))
    assert map_features(['SIGNATURE:allocates_rwx'],schema)['SIGNATURE:allocates_rwx']['technique_id'] is None

def test_discrete_training_mi_matches_reference():
    from xerd.additional import binary_mi
    from sklearn.metrics import mutual_info_score
    X=np.random.default_rng(8).integers(0,2,(80,8));y=X[:,0].copy();X[:,7]=0
    assert np.allclose(binary_mi(X,y),[mutual_info_score(y,X[:,j]) for j in range(8)])

def test_raw_parquet_supports_id_column_and_named_index(tmp_path):
    from xerd.additional import raw_frame
    frame=pd.DataFrame({'sample_id':[3,8],'13':[1,0]})
    for name,df in [('column',frame),('index',frame.set_index('sample_id'))]:
        p=tmp_path/(name+'.parquet');df.to_parquet(p)
        assert raw_frame(p,[8,3])['13'].tolist()==[0,1]

def test_shared_oof_matches_independent_stacking_reference():
    from xerd.models import bases,fit
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=140,n_features=10,random_state=71)
    ours=fit(X[:120],y[:120],jobs=1)
    ref=StackingClassifier(estimators=list(bases(jobs=1).items()),
        final_estimator=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=42),cv=5)
    ref.fit(X[:120],y[:120])
    assert np.allclose(ours.probabilities(X[120:])['X-ERD'],ref.predict_proba(X[120:])[:,1],atol=1e-10)
    assert (ours.inner_folds>=0).all() and len(np.unique(ours.inner_folds))==5

@pytest.mark.skipif(not __import__('pathlib').Path('data/mlran/MLRan_X_train_RFE.csv').exists(),reason='benchmark integration data not mounted')
def test_official_split_integration():
    d=load('data/mlran');assert d.X.shape==(3905,483) and d.Xt.shape==(975,483)
    assert d.yt.sum()==465

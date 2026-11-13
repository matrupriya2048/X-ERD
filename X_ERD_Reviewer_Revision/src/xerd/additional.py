"""Additional reviewer checks. Original benchmark representation stays unchanged."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from .models import fit
from .metrics import metrics
from .io import save,digest

RAW_GROUPS=['1_api','6_net','7_sys','8_drop','9_sig']

def raw_frame(path,ids):
    frame=pd.read_parquet(path)
    if 'sample_id' in frame.columns:frame=frame.set_index('sample_id',verify_integrity=True)
    if frame.index.name!='sample_id' or not frame.index.is_unique:raise ValueError('Raw sample ID index missing or duplicated')
    return frame.loc[ids]

def binary_mi(X,y):
    """Empirical discrete MI in nats; counts calculated only on training rows."""
    y=np.asarray(y);n=len(y);total=X.sum(0,dtype=np.float64)
    joint1=X[y==1].sum(0,dtype=np.float64);n1=float(y.sum());n0=n-n1
    score=np.zeros(X.shape[1])
    for joint,py in [(joint1,n1/n),(total-joint1,n0/n)]:
        for count,px in [(joint,total/n),(py*n-joint,1-total/n)]:
            prob=count/n;valid=(prob>0)&(px>0)
            score[valid]+=prob[valid]*np.log(prob[valid]/(py*px[valid]))
    return score

def strict_temporal(ds,raw_features,out,jobs=4):
    """Past-only feature selection in a declared raw behavioral candidate pool.

    File/path/registry/string vocabularies from the original 6.4M-column archive
    are not used. This is a separate temporal stress test, not a claim to repeat
    the complete original 6.4M-feature RFE construction.
    """
    out=Path(out);data=pd.concat([ds.train,ds.test],ignore_index=True)
    data=data.merge(ds.metadata[['sample_id','first_submission_date_year']],on='sample_id',validate='one_to_one')
    data=data.sort_values('sample_id').reset_index(drop=True)
    a=(data.first_submission_date_year<=2019).to_numpy()
    b=(data.first_submission_date_year>=2021).to_numpy()
    ids=data.sample_id.to_numpy();y=(data.type_label>0).to_numpy(dtype=np.int8)
    matrices=[];columns=[];dictionary={};files={}
    for prefix in RAW_GROUPS:
        p=Path(raw_features)/(prefix+'_dataset.parquet');q=Path(raw_features)/(prefix+'_feature_names_dic.json')
        frame=raw_frame(p,ids)
        if not all(str(c).isdigit() for c in frame.columns):raise ValueError('Nonfeature column in raw candidate data')
        X=frame.to_numpy(dtype=np.uint8)
        if not np.isin(X,[0,1]).all():raise ValueError('Nonbinary raw candidate data')
        columns.extend(map(str,frame.columns));matrices.append(X)
        dictionary.update(json.loads(q.read_text(encoding='utf-8')))
        files[p.name]=digest(p);files[q.name]=digest(q)
        print('Loaded raw modality',prefix,X.shape,flush=True)
    X=np.concatenate(matrices,axis=1);del matrices
    if len(columns)!=len(set(columns)):raise ValueError('Duplicate raw feature ID')
    mi=binary_mi(X[a],y[a]);variable=(X[a].sum(0)>0)&(X[a].sum(0)<a.sum())
    candidates=np.where(variable)[0]
    # Deterministic ties by feature ID; all filter decisions use past rows only.
    ranked=candidates[np.lexsort((np.array(columns,dtype=int)[candidates],-mi[candidates]))]
    shortlist=ranked[:2000]
    selector=RFE(RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=jobs),n_features_to_select=483,step=.2)
    selector.fit(X[a][:,shortlist],y[a]);selected=shortlist[selector.support_]
    print('Strict past-only features selected',len(selected),flush=True)
    m=fit(X[a][:,selected],y[a],jobs=jobs);p=m.probabilities(X[b][:,selected])
    selection=pd.DataFrame({'feature_id':np.array(columns)[shortlist],
      'feature_name':[dictionary[str(columns[i])] for i in shortlist],
      'training_mutual_information_nats':mi[shortlist],'rfe_rank':selector.ranking_,'selected':selector.support_})
    selection.to_csv(out/'temporal_selection.csv',index=False)
    derived=out/'derived_data';derived.mkdir(exist_ok=True)
    for name,k in [('train_pre2020',a),('test_2021_2024',b)]:
        frame=pd.DataFrame(X[k][:,selected],columns=np.array(columns)[selected])
        frame.insert(0,'type_label',data.loc[k,'type_label'].to_numpy());frame.insert(0,'sample_id',ids[k])
        frame.to_csv(derived/(name+'.csv'),index=False)
    pd.DataFrame({'sample_id':ids[b],'year':data.loc[b,'first_submission_date_year'],
                  'y_true':y[b],**p}).to_csv(out/'temporal_strict_predictions.csv',index=False)
    result={'scope':'strict past-only selection within API/network/system/dropped-file/signature raw pool; distinct from fixed483 benchmark',
      'excluded_modalities':['registry','file paths','directory paths','strings'],
      'selection':'training-only variable filter, discrete MI top2000, RF100 RFE step0.2 to483; no threshold tuning',
      'candidate_features':len(columns),'training_variable_features':int(variable.sum()),'selected_features':len(selected),
      'train_ids':ids[a],'test_ids':ids[b],'gap_2020_ids':ids[~(a|b)],
      'train_year_max':int(data.loc[a,'first_submission_date_year'].max()),'test_year_min':int(data.loc[b,'first_submission_date_year'].min()),
      'raw_file_sha256':files,'metrics':{n:metrics(y[b],q) for n,q in p.items()},'years':{}}
    years=data.loc[b,'first_submission_date_year'].to_numpy()
    for yr in sorted(set(years)):
        k=years==yr;result['years'][str(yr)]={n:metrics(y[b][k],q[k]) for n,q in p.items()}
    save(out/'temporal_strict.json',result)

def ablation(ds,m,out,jobs=4):
    rows=[]
    for j,name in enumerate(ds.names):
        row={'feature_id':ds.columns[j],'feature_name':name}
        for split,X,y in [('train',ds.X,ds.y),('test',ds.Xt,ds.yt)]:
            for cls in [0,1]:row[f'{split}_class{cls}_prevalence']=float(X[y==cls,j].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(out)/'feature_prevalence.csv',index=False)
    artifacts=np.array([n.startswith('STRING:') or ('DLL_LOADED:' in n and any(t in n.lower() for t in ['uxtheme.dll','ole32.dll'])) for n in ds.names])
    masks={'all483':np.ones(len(ds.names),bool),'no_strings':np.array([not n.startswith('STRING:') for n in ds.names]),
           'no_strings_or_two_dlls':~artifacts}
    result={}
    for name,keep in masks.items():
        print('Ablation',name,int(keep.sum()),flush=True)
        fitted=m if name=='all483' else fit(ds.X[:,keep],ds.y,jobs=jobs)
        probs=fitted.probabilities(ds.Xt[:,keep])
        result[name]={'features':int(keep.sum()),'removed_names':np.array(ds.names)[~keep],
                      'metrics':{n:metrics(ds.yt,q) for n,q in probs.items()}}
    save(Path(out)/'artifact_ablation.json',{'protocol':'same original settings, seeds and unshuffled inner OOF for every representation; fixed official split',
      'limitation':'feature removal is a sensitivity check; does not prove absence of collection artifacts or generalization', 'results':result})

def neural_control(ds,out):
    # A genuine deep feed-forward control on unordered indicators. Calling a
    # tabular CNN/LSTM a temporal sequence detector would invent sequence data.
    rows=[]
    for seed in [0,1,2,3,4]:
        model=MLPClassifier(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',
            alpha=.0001,batch_size=128,learning_rate_init=.001,max_iter=200,
            early_stopping=True,validation_fraction=.2,n_iter_no_change=15,random_state=seed)
        start=time.time();model.fit(ds.X,ds.y);q=model.predict_proba(ds.Xt)[:,1]
        rows.append({'seed':seed,'epochs':model.n_iter_,'fit_seconds':time.time()-start,**metrics(ds.yt,q)})
    save(Path(out)/'neural_control.json',{'architecture':'feed-forward 483-256-128-64-1, ReLU hidden activations',
      'validation':'internal stratified20% training-only early stopping; held-out test never used for tuning',
      'scope':'tabular deep baseline, not a sequence-model benchmark; labeled raw benign/ransomware sequences are unavailable',
      'configuration':'Adam lr0.001, alpha0.0001, batch128, maximum200 epochs, patience15; five fixed seeds', 'runs':rows})

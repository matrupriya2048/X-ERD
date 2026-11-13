"""Disclosed benign-domain adaptation with a locally frozen new holdout.

The previous raw evaluation is development data here. It is never reused as
untouched evidence for this repair. No target labels, thresholds or parser
exceptions are selected from the new holdout.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse,datetime,json,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
from .models import bases,CONFIGS
from .metrics import metrics,bootstrap
from .io import load,save,digest,environment
from .parser import vector
from .external_raw import interval

@dataclass
class RemediatedEnsemble:
    base:dict
    meta:dict
    passthrough:bool
    def probabilities(self,X):
        z=np.column_stack([m.predict_proba(X)[:,1] for m in self.base.values()])
        result={n:z[:,i] for i,n in enumerate(self.base)}
        for n,m in self.meta.items():
            inputs=z[:,CONFIGS[n]]
            if self.passthrough:inputs=np.column_stack([inputs,X])
            result[n]=m.predict_proba(inputs)[:,1]
        return result

def splits(X,y,groups,domain,n=5):
    strata=np.where(domain==1,2,y)
    for a,b in StratifiedGroupKFold(n,shuffle=True,random_state=42).split(X,strata,groups):
        if set(groups[a])&set(groups[b]):raise AssertionError('Executable crosses fit/validation boundary')
        yield a,b

def weighted_bases(X,y,groups,domain,weight):
    weights=np.where(domain==1,weight,1.)
    if (weights<=0).any():raise ValueError('Zero-weight development rows must be excluded before fitting')
    oof=np.full((len(y),3),np.nan);inner=np.full(len(y),-1)
    for f,(a,b) in enumerate(splits(X,y,groups,domain)):
        inner[b]=f
        for j,m in enumerate(bases().values()):
            m.fit(X[a],y[a],sample_weight=weights[a]);oof[b,j]=m.predict_proba(X[b])[:,1]
    assert np.isfinite(oof).all() and (inner>=0).all()
    fitted=bases()
    for m in fitted.values():m.fit(X,y,sample_weight=weights)
    return oof,inner,fitted,weights

def logistic(X,y,c,weights):
    return LogisticRegression(C=c,solver='lbfgs',max_iter=2000,random_state=42).fit(X,y,sample_weight=weights)

def select(scores):
    eligible=[r for r in scores if r['source']['recall']>=.97 and r['source']['fpr']<=.05]
    if eligible:
        return min(eligible,key=lambda r:(r['external_development']['fpr'],r['source']['fpr'],-r['source']['recall'],r['index'])),True
    return min(scores,key=lambda r:(-r['source']['recall'],r['external_development']['fpr'],r['index'])),False

def train(root,out):
    ds=load(root/'data/mlran');protocol_path=root/'config/raw_remediation_protocol.json';protocol=json.loads(protocol_path.read_text())
    if (out/'frozen_model.json').exists():raise ValueError('Frozen model exists; do not overwrite a model after holdout evaluation')
    dev=pd.read_csv(out/'development_vectors.csv');X=np.vstack([ds.X,dev[ds.columns].to_numpy()]);y=np.r_[ds.y,np.zeros(len(dev),dtype=int)]
    domain=np.r_[np.zeros(len(ds.y),dtype=int),np.ones(len(dev),dtype=int)]
    source_groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    groups=np.r_[source_groups,dev.md5.str.lower().to_numpy()]
    assert not set(source_groups)&set(dev.md5.str.lower())
    ids=np.r_[np.array(['MLRan:'+str(s) for s in ds.train.sample_id]),dev.source_path.to_numpy()]
    outer=list(splits(X,y,groups,domain));predictions={};logs=[];score_rows=[];index=0
    for w in protocol['development_weights']:
        print('Training candidate development weight',w,flush=True)
        pred={(pt,c):np.full(len(y),np.nan) for pt in [False,True] for c in protocol['meta_C']}
        for f,(outer_a,b) in enumerate(outer):
            a=outer_a[domain[outer_a]==0] if w==0 else outer_a
            z,inner,base,weights=weighted_bases(X[a],y[a],groups[a],domain[a],w)
            zv=np.column_stack([m.predict_proba(X[b])[:,1] for m in base.values()])
            for (pt,c),p in pred.items():
                tr=np.column_stack([z,X[a]]) if pt else z;va=np.column_stack([zv,X[b]]) if pt else zv
                p[b]=logistic(tr,y[a],c,weights).predict_proba(va)[:,1]
            logs.append({'weight':w,'fold':f,'train_ids':ids[a].tolist(),'validation_ids':ids[b].tolist(),'inner_folds':inner.tolist(),
                         'train_group_sha256':__import__('hashlib').sha256('\n'.join(groups[a]).encode()).hexdigest()})
            save(out/'training_fold_log.json',logs);print('Completed outer fold',f,flush=True)
        for (pt,c),p in pred.items():
            assert np.isfinite(p).all();key=f'w{w}_passthrough{int(pt)}_C{c}';predictions[key]=p
            score_rows.append({'index':index,'key':key,'development_weight':w,'passthrough':pt,'C':c,
              'source':metrics(y[domain==0],p[domain==0]),'external_development':metrics(y[domain==1],p[domain==1])});index+=1
    winner,feasible=select(score_rows)
    pd.DataFrame({'id':ids,'md5':groups,'domain':domain,'y_true':y,**predictions}).to_csv(out/'training_oof_predictions.csv',index=False)
    save(out/'selection.json',{'selected':winner,'source_constraints_feasible':feasible,'candidates':score_rows,
      'scope':'Selection uses only grouped outer OOF development predictions; selection scores are optimistic for the chosen candidate'})
    w=winner['development_weight'];a=np.where(domain==0)[0] if w==0 else np.arange(len(y))
    z,inner,base,weights=weighted_bases(X[a],y[a],groups[a],domain[a],w);pt=winner['passthrough']
    heads={n:logistic(np.column_stack([z[:,ix],X[a]]) if pt else z[:,ix],y[a],winner['C'],weights) for n,ix in CONFIGS.items()}
    m=RemediatedEnsemble(base,heads,pt);joblib.dump(m,out/'model.joblib')
    pd.DataFrame({'id':ids[a],'md5':groups[a],'domain':domain[a],'y_true':y[a],'inner_fold':inner,**{n:z[:,j] for j,n in enumerate(base)}}).to_csv(out/'final_fit_oof.csv',index=False)
    save(out/'frozen_model.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':digest(out/'model.joblib'),
      'protocol_sha256':digest(protocol_path),'source_sha256':digest(Path(__file__)),'development_vectors_sha256':digest(out/'development_vectors.csv'),
      'selection':winner,'threshold':.5,'fit_n':len(a),'holdout_scored':False,'scope':'Local pre-evaluation freeze, not independent preregistration'})
    # The official benchmark was evaluated before this work. It is a reused
    # regression check, never prospective evidence or a selection input here.
    p=m.probabilities(ds.Xt)
    pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,**p}).to_csv(out/'official_test_predictions.csv',index=False)
    save(out/'official_test.json',{'models':{n:metrics(ds.yt,v) for n,v in p.items()},'full_ci95':bootstrap(ds.yt,p['X-ERD']),
      'scope':'Previously used benchmark; reported after configuration freeze; no fresh or external recall claim'})
    print('FROZEN',winner['key'],'official test',metrics(ds.yt,p['X-ERD']),flush=True)

def evaluate(root,out):
    frozen=json.loads((out/'frozen_model.json').read_text());assert digest(out/'model.joblib')==frozen['model_sha256']
    protocol_path=root/'config/raw_remediation_protocol.json';assert digest(protocol_path)==frozen['protocol_sha256']
    manifest=json.loads((root/'config/raw_remediation_holdout_sources.json').read_text())
    assert manifest['protocol_sha256']==frozen['protocol_sha256'] and len(manifest['files'])==400
    ds=load(root/'data/mlran');dev=pd.read_csv(out/'development_vectors.csv')
    excluded={str(v).lower() for frame in [ds.metadata,dev] for col in ['md5','sha1','sha256'] for v in frame[col].dropna()}
    seen=set();audit=[];xs=[];ids=[]
    for item in manifest['files']:
        path=root/item['path'];assert digest(path)==item['sha256'];row={'source_path':item['path'],'report_sha256':item['sha256']}
        try:
            d=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(d,dict) or not isinstance(d.get('behavior'),dict):raise ValueError('Missing behavior object')
            behavior=d['behavior'];calls=sum(len(p.get('calls',[])) for p in behavior.get('processes',[]) if isinstance(p,dict));stats=behavior.get('apistats',{})
            row['api_call_event_count']=calls
            if not calls and not (isinstance(stats,dict) and any(bool(v) for v in stats.values())):raise ValueError('No captured API observations; execution cannot be assessed')
            f=d.get('target',{}).get('file',{});hashes=[str(f[k]).lower() for k in ['sha256','sha1','md5'] if f.get(k)]
            if not hashes:raise ValueError('No executable hash; cannot check independence')
            row.update({'md5':f.get('md5'),'sha256':f.get('sha256'),'file_type':f.get('type'),'filename':f.get('name')})
            if any(h in excluded for h in hashes):row['status']='excluded_source_or_development_overlap'
            elif any(h in seen for h in hashes):row['status']='excluded_duplicate_executable'
            else:
                x=vector(d,ds.names);xs.append(x);ids.append(item['path']);seen.update(hashes);row['status']='evaluated';row['active_features']=int(x.sum())
        except (ValueError,TypeError,KeyError,AttributeError) as e:row.update({'status':'unprocessed','reason':str(e)})
        audit.append(row)
        if len(audit)%25==0:print('Audited new holdout reports',len(audit),flush=True)
    X=np.asarray(xs);y=np.zeros(len(X),dtype=int);assert len(X)>0
    models={'Original full stack':joblib.load(root/'results/model.joblib'),'Previous passthrough':joblib.load(root/'results/extended/residual_model.joblib'),
            'Development-adapted':joblib.load(out/'model.joblib')}
    pred={};scores={}
    for name,m in models.items():
        p=m.probabilities(X);pred.update({name+'__'+n:v for n,v in p.items()})
        scores[name]={n:{**metrics(y,v),'fpr_ci95':interval(int((v>=.5).sum()),len(y))} for n,v in p.items()}
    pd.DataFrame({'source_path':ids,'y_true':y,**pred}).to_csv(out/'holdout_predictions.csv',index=False)
    pd.concat([pd.DataFrame({'source_path':ids}),pd.DataFrame(X,columns=ds.columns)],axis=1).to_csv(out/'holdout_vectors.csv',index=False)
    comparisons={};rng=np.random.default_rng(42);new=(pred['Development-adapted__X-ERD']>=.5).astype(int)
    for label in ['Original full stack','Previous passthrough']:
        old=(pred[label+'__X-ERD']>=.5).astype(int);b=int(((old==1)&(new==0)).sum());c=int(((old==0)&(new==1)).sum())
        delta=new-old;boot=[float(delta[rng.integers(0,len(y),len(y))].mean()) for _ in range(5000)]
        comparisons[label]={'new_minus_old_fpr':float(delta.mean()),'paired_bootstrap_ci95':np.quantile(boot,[.025,.975]),
          'old_only_false_alarms':b,'new_only_false_alarms':c,'exact_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.}
    save(out/'holdout_audit.json',audit)
    save(out/'holdout.json',{'selected_n':400,'evaluated_unique_n':len(y),'status_counts':pd.Series([r['status'] for r in audit]).value_counts().to_dict(),
      'models':scores,'paired_comparisons':comparisons,'model_sha256':frozen['model_sha256'],'source_manifest_sha256':digest(root/'config/raw_remediation_holdout_sources.json'),
      'scope':'Prospective locally frozen same-collection benign specificity after disclosed adaptation; no matched external ransomware recall, no enterprise deployment guarantee',
      'evaluated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('HOLDOUT',len(y),{n:s['X-ERD']['fpr'] for n,s in scores.items()},flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate']);ap.add_argument('--root',type=Path,default=Path('.'));args=ap.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    root=args.root;out=root/'results/remediation';out.mkdir(parents=True,exist_ok=True);start=time.time()
    save(out/(args.stage+'_environment.json'),environment());globals()[args.stage](root,out)
    save(out/(args.stage+'_execution.json'),{'status':'complete','seconds':time.time()-start})

if __name__=='__main__':
    # Persisted estimator classes must use their importable package identity.
    from xerd.remediation import main as canonical_main
    canonical_main()

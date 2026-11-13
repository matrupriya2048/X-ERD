"""Independent result checks for the frozen benign-adapted model.

Published ransomware JSONs are evaluated once, without fitting or reselection.
Attribution uses the unchanged common-background probability/logit identity.
"""
from pathlib import Path
import argparse,datetime,json,time,warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from .io import load,save,digest,environment
from .parser import vector
from .metrics import metrics
from .external_raw import interval

def frozen_model(root,out=None):
    out=out or root/'results/remediation';f=json.loads((out/'frozen_model.json').read_text())
    assert digest(out/'model.joblib')==f['model_sha256']
    return joblib.load(out/'model.joblib'),f

def ransomware(root,out=None):
    out=out or root/'results/remediation';model,frozen=frozen_model(root,out);ds=load(root/'data/mlran')
    inv=json.loads((root/'config/zirari_raw_inventory.json').read_text());assert len(inv['files'])==138
    acquisition={r['name']:r for r in json.loads((root/'config/zirari_acquisition_status.json').read_text())['files']}
    dev=pd.read_csv(out/'development_vectors.csv')
    excluded={str(v).lower() for frame in [ds.metadata,dev] for col in ['md5','sha1','sha256'] for v in frame[col].dropna()}
    benign_audit=json.loads((out/'holdout_audit.json').read_text())
    benign_hashes={str(row[k]).lower() for row in benign_audit for k in ['md5','sha256'] if row.get(k)}
    audit=[];xs=[];ids=[];seen=set();receipts=[]
    for f in sorted(inv['files'],key=lambda x:x['title']):
        availability=acquisition[f['title']]
        if availability['status']!='acquired':
            audit.append({'path':'data/zirari_raw/'+f['title'],'drive_id':f['id'],'publisher_label':'ransomware','status':'unavailable_download','reason':availability['reason']});continue
        receipt=json.loads((root/'data/zirari_raw_receipts'/(f['title']+'.receipt.json')).read_text())
        path=root/receipt['path'];assert digest(path)==receipt['sha256'] and path.stat().st_size==int(f['size'])
        receipts.append(receipt);row={**receipt,'publisher_label':'ransomware'}
        try:
            d=behavior=target=stats=None
            d=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(d,dict) or not isinstance(d.get('behavior'),dict):raise ValueError('Missing behavior object')
            behavior=d['behavior'];calls=sum(len(p.get('calls',[])) for p in behavior.get('processes',[]) if isinstance(p,dict));stats=behavior.get('apistats',{})
            row['api_call_event_count']=calls
            if not calls and not (isinstance(stats,dict) and any(bool(v) for v in stats.values())):raise ValueError('No captured API observations; execution cannot be assessed')
            target=d.get('target',{}).get('file',{});hashes=[str(target[k]).lower() for k in ['sha256','sha1','md5'] if target.get(k)]
            if not hashes:raise ValueError('No executable hash; cannot check independence')
            row.update({k:target.get(k) for k in ['md5','sha1','sha256','name','type']})
            row.update({'cuckoo_version':d.get('info',{}).get('version'),'publisher_score':d.get('info',{}).get('score')})
            if any(h in excluded for h in hashes):row['status']='excluded_source_or_development_overlap'
            elif any(h in benign_hashes for h in hashes):row['status']='excluded_conflicting_external_label'
            elif any(h in seen for h in hashes):row['status']='excluded_duplicate_executable'
            else:
                x=vector(d,ds.names);xs.append(x);ids.append(receipt['path']);seen.update(hashes);row['status']='evaluated';row['active_features']=int(x.sum())
        except (ValueError,TypeError,KeyError,AttributeError) as e:row.update({'status':'unprocessed','reason':str(e)})
        audit.append(row)
        if len(audit)%20==0:print('Audited published ransomware reports',len(audit),flush=True)
    save(root/'config/zirari_raw_sources.json',{'inventory_sha256':digest(root/'config/zirari_raw_inventory.json'),'files':receipts})
    save(out/'ransomware_audit.json',audit)
    if not xs:raise ValueError('No eligible nonoverlapping published ransomware records')
    X=np.asarray(xs);y=np.ones(len(X),dtype=int)
    models={'Original full stack':joblib.load(root/'results/model.joblib'),'Previous passthrough':joblib.load(root/'results/extended/residual_model.joblib'),'Development-adapted':model}
    pred={};scores={}
    for label,m in models.items():
        p=m.probabilities(X);pred.update({label+'__'+n:v for n,v in p.items()})
        scores[label]={n:{**metrics(y,v),'recall_ci95':interval(int((v>=.5).sum()),len(y))} for n,v in p.items()}
    pd.DataFrame({'source_path':ids,'y_true':y,**pred}).to_csv(out/'ransomware_predictions.csv',index=False)
    pd.concat([pd.DataFrame({'source_path':ids}),pd.DataFrame(X,columns=ds.columns)],axis=1).to_csv(out/'ransomware_vectors.csv',index=False)
    new=pred['Development-adapted__X-ERD']>=.5;paired={};rng=np.random.default_rng(42)
    for label in ['Original full stack','Previous passthrough']:
        old=pred[label+'__X-ERD']>=.5;b=int((old&~new).sum());c=int((~old&new).sum());delta=new.astype(int)-old.astype(int)
        boot=[float(delta[rng.integers(0,len(y),len(y))].mean()) for _ in range(5000)]
        paired[label]={'new_minus_old_recall':float(delta.mean()),'paired_bootstrap_ci95':np.quantile(boot,[.025,.975]),'lost_detections':b,'gained_detections':c,'exact_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.}
    save(out/'ransomware.json',{'listed_reports':len(inv['files']),'acquired_reports':len(receipts),'paper_describes_selected_samples':140,'evaluated_unique_n':len(y),'status_counts':pd.Series([r['status'] for r in audit]).value_counts().to_dict(),'models':scores,'paired_comparisons':paired,'model_sha256':frozen['model_sha256'],'source_manifest_sha256':digest(root/'config/zirari_raw_sources.json'),'evaluated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Separate publisher-labeled ransomware sensitivity on accessible raw reports; publisher describes selection of active high-score executions. Acquisition exclusions are reported for every listed file. No local fitting or filtering by model score. Not paired with Nunes benign controls to infer accuracy, precision or deployment performance.'})
    print('RANSOMWARE',len(y),{n:v['X-ERD']['recall'] for n,v in scores.items()},flush=True)

def explanations(root,out=None):
    import shap
    from .explain import positive_values
    out=out or root/'results/remediation';model,frozen=frozen_model(root,out);ds=load(root/'data/mlran')
    dev=pd.read_csv(out/'development_vectors.csv');training=np.vstack([ds.X,dev[ds.columns].to_numpy()]) if frozen['selection']['development_weight']>0 else ds.X
    rng=np.random.default_rng(42);ix=rng.choice(len(training),100,replace=False);background=training[ix]
    X=ds.Xt;vals={};expected={};errors={}
    for n,m in model.base.items():
        print('Adapted probability SHAP',n,flush=True)
        ex=shap.TreeExplainer(m,data=background,model_output='probability',feature_perturbation='interventional');v,b=positive_values(ex,X)
        errors[n]=float(abs(b+v.sum(1)-m.predict_proba(X)[:,1]).max());assert errors[n]<1e-4
        vals[n]=v;expected[n]=b
    head=model.meta['X-ERD'];w=head.coef_[0];phi=sum(w[i]*vals[n] for i,n in enumerate(model.base));baseline=float(head.intercept_[0]+sum(w[i]*expected[n] for i,n in enumerate(model.base)))
    if model.passthrough:
        beta=w[3:];phi+=beta*(X-background.mean(0));baseline+=float(beta@background.mean(0))
    error=float(abs(expit(baseline+phi.sum(1))-model.probabilities(X)['X-ERD']).max());assert error<1e-4
    np.savez_compressed(out/'shap_values.npz',stack_logit=phi,background_indices=ix,baseline=baseline,**vals)
    pd.DataFrame({'feature_id':ds.columns,'feature_name':ds.names,'mean_abs_stack_logit_shap':abs(phi).mean(0)}).sort_values('mean_abs_stack_logit_shap',ascending=False).to_csv(out/'global_shap.csv',index=False)
    save(out/'explanations.json',{'model_sha256':frozen['model_sha256'],'n':len(X),'background_indices_in_fit_data':ix,'background_size':100,'base_expected_values':expected,'max_base_additivity_errors':errors,'maximum_probability_reconstruction_error':error,'meta_coefficients':w,'meta_intercept':head.intercept_[0],'scope':'Refitted base probability SHAP on common training background plus exact raw-feature linear term; explains stack logit, not causal behavior or human-verified ATT&CK.'})
    print('Adapted attribution verified',error,flush=True)

def components(root,out=None):
    from .explain import deletion,targeted
    from .mapping import evaluate as mapping_evaluate
    out=out or root/'results/remediation';model,frozen=frozen_model(root,out);ds=load(root/'data/mlran')
    with np.load(out/'shap_values.npz') as z:
        vals={n:z[n] for n in model.base};phi=z['stack_logit']
    deletion(ds,model,vals,phi,sum(vals.values())/3,out)
    targeted(ds,model,vals['LGB'],phi,out)
    mapping_evaluate(ds,model,out,root/'config/mapping_schema.json')
    save(out/'components.json',{'model_sha256':frozen['model_sha256'],'completed':['deletion','targeted feature-space masking','attribution-weighted mapping coverage'],
      'scope':'Recomputed for the adapted model; sensitivities are not executable-space attacks or causal proof. Mapper rules unchanged; no independent human or execution-level accuracy labels added.'})
    print('Adapted sensitivity and mapping coverage completed',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['ransomware','explanations','components']);ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--output',type=Path);a=ap.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names');start=time.time();out=a.output or a.root/'results/remediation'
    save(out/(a.stage+'_environment.json'),environment());globals()[a.stage](a.root,out)
    save(out/(a.stage+'_execution.json'),{'status':'complete','seconds':time.time()-start})

if __name__=='__main__':main()

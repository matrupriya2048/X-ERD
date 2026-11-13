from pathlib import Path
import json,collections
import numpy as np
import pandas as pd
import joblib
from xerd.io import load,save,digest,environment
from xerd.parser import vector
from xerd.external_raw import interval
from primary_pair import probabilities
from evaluate_mtm import target_hashes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def run():
    protocol=json.loads((HERE/'en2025_protocol.json').read_text());ds=load(ROOT/'data/mlran')
    assert digest(HERE/'closure_results/primary_pair.joblib')==protocol['primary_model_sha256']
    excluded=set()
    for frame in [ds.metadata,pd.read_csv(ROOT/'results/remediation/development_vectors.csv')]:
        for k in ['md5','sha1','sha256']:excluded.update(str(v).lower() for v in frame[k].dropna())
    for path in [ROOT/'results/remediation/holdout_audit.json',ROOT/'results/remediation/ransomware_audit.json',ROOT/'results/extended/external_raw_audit.json']:
        for row in json.loads(path.read_text()):
            excluded.update(target_hashes(row))
            if row.get('executable_hash'):excluded.add(row['executable_hash'].lower())
    for row in json.loads((HERE/'cohort_audit.json').read_text()):excluded.update(target_hashes({k:row.get('target_'+k) for k in ['md5','sha1','sha256']}))
    seen=set();audit=[];xs=[];ids=[]
    fitvectors={bytes(x.astype(np.int8)) for x in ds.X}
    for item in protocol['files']:
        path=HERE/'data_en2025/raw'/item['name'];assert path.stat().st_size==item['size']
        row={'path':item['name'],'report_sha256':digest(path)}
        try:
            d=json.loads(path.read_text(encoding='utf-8'));target=d.get('target',{}).get('file',{});h=target_hashes(target)
            row.update({'target_'+k:target.get(k) for k in ['md5','sha1','sha256','type','name']})
            info=d.get('info') or {};platform=info.get('platform') or (info.get('machine') or {}).get('platform')
            row.update(platform=platform,category=info.get('category'),collector_version=info.get('version'),malscore=d.get('malscore'),malstatus=d.get('malstatus'))
            if not h or target.get('sha256','').lower()!=Path(item['name']).parent.name.lower():raise ValueError('Missing target hash or folder mismatch')
            b=d.get('behavior');calls=sum(len(p.get('calls') or []) for p in b.get('processes',[])) if isinstance(b,dict) else 0
            row['call_n']=calls
            if info.get('category')!='file' or not str(target.get('type','')).startswith('PE32') or (platform and str(platform).lower()!='windows'):row['status']='excluded_non_windows_pe'
            elif not calls:row['status']='unprocessed_no_api_events'
            elif h&excluded:row['status']='excluded_previously_used_target'
            elif h&seen:row['status']='excluded_duplicate_target'
            else:
                x=vector(d,ds.names);seen.update(h);xs.append(x);ids.append(item['name'])
                row.update(status='evaluated',active_features=int(x.sum()),source_fit_vector_match=bytes(x) in fitvectors,
                           strings_available='strings' in d,summary_fields=sorted(b.get('summary',{})))
        except (ValueError,KeyError,TypeError,AttributeError) as e:row.update(status='unprocessed',reason=str(e))
        audit.append(row)
        if len(audit)%100==0:print('EN2025 audited',len(audit),'/',len(protocol['files']),flush=True)
    save(HERE/'closure_results/en2025_audit.json',audit)
    if not xs:raise ValueError('No eligible targets')
    X=np.asarray(xs,dtype=np.int8);primary=joblib.load(HERE/'closure_results/primary_pair.joblib');original=joblib.load(ROOT/'results/model.joblib');adapted=joblib.load(ROOT/'results/remediation/model.joblib')
    pred={'primary_RF+LGB':probabilities(primary,X),'original_three':original.probabilities(X)['X-ERD'],'adapted_three':adapted.probabilities(X)['X-ERD']}
    pd.DataFrame({'source_path':ids,**pred}).to_csv(HERE/'closure_results/en2025_predictions.csv',index=False)
    pd.concat([pd.DataFrame({'source_path':ids}),pd.DataFrame(X,columns=ds.columns)],axis=1).to_csv(HERE/'closure_results/en2025_vectors.csv',index=False)
    rows=[r for r in audit if r['status']=='evaluated'];n=len(X)
    result={'source':protocol['source'],'protocol_sha256':digest(HERE/'en2025_protocol.json'),'script_sha256':digest(Path(__file__)),
            'selected_n':len(protocol['files']),'status_counts':dict(collections.Counter(r['status'] for r in audit)),
            'models':{k:{'n':n,'alert_n':int((p>=.5).sum()),'fpr':float((p>=.5).mean()),'ci95':interval(int((p>=.5).sum()),n)} for k,p in pred.items()},
            'strings_available_n':sum(r['strings_available'] for r in rows),'source_fit_vector_matches':sum(r['source_fit_vector_match'] for r in rows),
            'distinct_vectors':len({bytes(x) for x in X}),'zero_active_vectors':int((X.sum(1)==0).sum()),
            'categories':{g:{'selected_feature_n':sum(s.startswith(g+':') for s in ds.names),'active_values':int(X[:,[s.startswith(g+':') for s in ds.names]].sum())} for g in sorted({s.split(':')[0] for s in ds.names})},
            'environment':environment(),'scope':'Publisher-labeled benign PE executions from a separate CAPE collection. No independent human class adjudication; missing full-run modalities and collector differences limit fidelity. Unknown hashes of 80 previous sequence inputs prevent complete overlap verification. No ransomware in this cohort; no pooled accuracy or early-onset claim.'}
    save(HERE/'closure_results/en2025_results.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()

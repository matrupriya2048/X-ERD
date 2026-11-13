"""Frozen full-representation classifier on externally collected benign reports."""
from pathlib import Path
import json,time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from .parser import vector
from .metrics import metrics
from .io import save,digest

def interval(k,n):
    if not n:return [None,None]
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def run(ds,root,out):
    root=Path(root);out=Path(out);manifest=json.loads((root/'config/benign_report_sources.json').read_text())
    if len(manifest.get('files',[]))!=200:raise ValueError('Complete200-report acquisition/CRC verification required')
    protocol={'source':manifest['source'],'manifest_sha256':digest(root/'config/benign_report_sources.json'),
      'sample_unit':'unique executable hash from a preselected random200 publisher-clean report set; malformed, duplicate or source-overlapping records audited',
      'model':'frozen official MLRan3905 fit, all483 features, original fixed threshold0.5; no target threshold/calibration fit',
      'scope':'external benign false-alert assessment; separate from the20 MLRan ransomware-only conditional parser check. Do not pool these sources into a single accuracy/precision estimate.',
      'labels':'publisher clean-app folder labels; no fresh independent malware adjudication',
      'observability':'reports require a behavior object plus captured API calls or a nonempty API-count dictionary; otherwise counted as unprocessed,not as true negatives',
      'fidelity':'No official483 vectors exist for this external collection; classification is evaluated, but Jaccard fidelity/official-versus-parsed agreement cannot be calculated.',
      'ci':'two-sided Clopper-Pearson binomial95 interval over unique hashes; related applications/environments may still be dependent'}
    save(out/'external_raw_protocol.json',protocol)
    source_hashes={str(v).lower() for col in ['md5','sha1','sha256'] for v in ds.metadata[col].dropna()}
    seen=set();audit=[];xs=[];ids=[];start=time.time()
    for item in sorted(manifest['files'],key=lambda x:x['path']):
        p=root/item['path'];row={'source_path':item['path'],'report_sha256':item['sha256']}
        if digest(p)!=item['sha256']:raise ValueError('Report SHA256 mismatch')
        try:
            d=json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(d,dict) or not isinstance(d.get('behavior'),dict):raise ValueError('Missing behavior object')
            behavior=d['behavior'];calls=sum(len(p.get('calls',[])) for p in behavior.get('processes',[]) if isinstance(p,dict))
            stats=behavior.get('apistats',{});row['api_call_event_count']=calls
            has_stats=isinstance(stats,dict) and any(bool(v) for v in stats.values())
            if not calls and not has_stats:raise ValueError('No captured API observations; execution cannot be assessed')
            file=d.get('target',{}).get('file',{});hashes=[str(file[k]).lower() for k in ['sha256','sha1','md5'] if file.get(k)]
            if not hashes:raise ValueError('No executable hash; cannot check independence')
            row.update({'executable_hash':hashes[0],'file_type':file.get('type',''),'score':d.get('info',{}).get('score')})
            if any(h in source_hashes for h in hashes):row['status']='excluded_source_overlap'
            elif any(h in seen for h in hashes):row['status']='excluded_duplicate_executable'
            else:
                x=vector(d,ds.names);seen.update(hashes);xs.append(x);ids.append(item['path']);row['status']='evaluated';row['active_features']=int(x.sum())
                row['available_fields']={k:k in d for k in ['strings','signatures','dropped']}
                row['behavior_fields']=sorted(d['behavior'])
        except (ValueError,TypeError,KeyError,AttributeError) as e:row.update({'status':'unprocessed','reason':str(e)})
        audit.append(row)
    if not xs:raise ValueError('No valid novel benign reports')
    X=np.array(xs);y=np.zeros(len(X),dtype=int);m=joblib.load(root/'results/model.joblib');p=m.probabilities(X)
    pd.DataFrame({'source_path':ids,'y_true':y,**p}).to_csv(out/'external_raw_predictions.csv',index=False)
    res={n:{**metrics(y,v),'fpr_ci95':interval(int((v>=.5).sum()),len(y))} for n,v in p.items()}
    categories={g:{'feature_n':sum(n.startswith(g+':') for n in ds.names),'active_values':int(X[:,[n.startswith(g+':') for n in ds.names]].sum())} for g in sorted({n.split(':')[0] for n in ds.names})}
    save(out/'external_raw_audit.json',audit)
    candidate_results={};candidate_predictions={}
    for label,file in [('Training-selected fixed stack','optimized_model.joblib'),('Feature passthrough','residual_model.joblib')]:
        candidate=joblib.load(out/file);cp=candidate.probabilities(X)
        candidate_results[label]={n:{**metrics(y,v),'fpr_ci95':interval(int((v>=.5).sum()),len(y))} for n,v in cp.items()}
        candidate_predictions.update({label+'__'+n:v for n,v in cp.items()})
    pd.DataFrame({'source_path':ids,'y_true':y,**candidate_predictions}).to_csv(out/'external_raw_candidates_predictions.csv',index=False)
    save(out/'external_raw_candidates.json',{'models':candidate_results,'scope':'Both additional candidates fixed by MLRan training group CV; no target fitting or threshold selection; all results disclosed'})
    save(out/'external_raw.json',{'selected_reports':len(audit),'evaluated_unique_n':len(X),'status_counts':pd.Series([r['status'] for r in audit]).value_counts().to_dict(),
      'models':res,'active_features':{'mean':float(X.sum(1).mean()),'min':int(X.sum(1).min()),'max':int(X.sum(1).max())},'categories':categories,'seconds':time.time()-start})
    print('Raw benign controls',len(X),'full stack',res['X-ERD'],flush=True)

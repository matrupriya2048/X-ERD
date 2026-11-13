"""Score an untouched collection with three already-frozen classifiers."""
from pathlib import Path
import collections, hashlib, json, math, time, re
import joblib
import numpy as np
import pandas as pd
from xerd.io import load, digest, save, environment
from xerd.parser import vector
from xerd.external_raw import interval

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def target_hashes(target):
    return {str(target[k]).lower() for k,size in [('md5',32),('sha1',40),('sha256',64)]
            if re.fullmatch('[0-9a-fA-F]{'+str(size)+'}',str(target.get(k,'')))}

def eligible(report):
    if not isinstance(report,dict):
        return 'unprocessed_not_object'
    info=report.get('info') or {}
    score=info.get('score')
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=3:
        return 'excluded_score_not_publisher_benign_range'
    platform=info.get('platform') or (info.get('machine') or {}).get('platform')
    if str(platform).lower()!='windows' or info.get('category')!='file':
        return 'excluded_non_windows_file_execution'
    if not target_hashes((report.get('target') or {}).get('file') or {}):
        return 'unprocessed_no_executable_hash'
    behavior=report.get('behavior')
    if not isinstance(behavior,dict):
        return 'unprocessed_no_behavior'
    calls=sum(len(p.get('calls') or []) for p in behavior.get('processes',[]) if isinstance(p,dict))
    stats=behavior.get('apistats')
    if not calls and not (isinstance(stats,dict) and any(bool(v) for v in stats.values())):
        return 'unprocessed_no_api_observations'
    return 'eligible'

def run():
    start=time.time();protocol=json.loads((HERE/'protocol.json').read_text())
    receipt={r['path']:r for r in json.loads((HERE/'acquisition.json').read_text())}
    assert set(receipt)=={f['path'] for f in protocol['files']}
    assert digest(ROOT/'src/xerd/parser.py')==protocol['parser_sha256']
    import xerd.parser
    assert digest(Path(xerd.parser.__file__))==protocol['parser_sha256']
    for item in protocol['models'].values():
        assert digest(ROOT/item['path'])==item['sha256']
    ds=load(ROOT/'data/mlran')
    excluded=target_hashes({})
    for frame in [ds.metadata,pd.read_csv(ROOT/'results/remediation/development_vectors.csv')]:
        for key in ['md5','sha1','sha256']:
            excluded.update(str(x).lower() for x in frame[key].dropna())
    for name in ['results/extended/external_raw_audit.json','results/remediation/holdout_audit.json','results/remediation/ransomware_audit.json']:
        for row in json.loads((ROOT/name).read_text()):
            excluded.update(target_hashes(row))
            if row.get('executable_hash'):
                excluded.add(row['executable_hash'].lower())
    # Earlier API-sequence reports may lack hashes; retain missingness explicitly.
    previous_missing=0
    for path in sorted((ROOT/'data/external_reports').glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8'))
        h=target_hashes((d.get('target') or {}).get('file') or {}) if isinstance(d,dict) else set()
        excluded.update(h);previous_missing+=not bool(h)
    fit_vectors={bytes(x.astype(np.int8)) for x in ds.X}
    dev=pd.read_csv(ROOT/'results/remediation/development_vectors.csv')
    fit_vectors.update(bytes(x.astype(np.int8)) for x in dev[ds.columns].to_numpy())
    seen=set();audit=[];xs=[];ids=[]
    for item in sorted(protocol['files'],key=lambda x:x['path']):
        row=dict(receipt[item['path']]);row['publisher_label']='score-derived benign'
        if row['status']!='acquired':
            audit.append(row);continue
        path=HERE/'raw'/item['path']
        assert digest(path)==row['sha256']
        try:
            d=json.loads(path.read_text(encoding='utf-8'))
            status=eligible(d)
            if status!='eligible':
                row['status']=status;audit.append(row);continue
            target=d['target']['file'];hashes=target_hashes(target)
            row.update({'target_'+k:target.get(k) for k in ['md5','sha1','sha256','name','type']})
            row['publisher_score']=d['info']['score']
            vt=d.get('virustotal') or {}
            row['antivirus_positive_count']=vt.get('positives') if isinstance(vt,dict) else None
            row['antivirus_total_count']=vt.get('total') if isinstance(vt,dict) else None
            if hashes & excluded:
                row['status']='excluded_previously_used_executable'
            elif hashes & seen:
                row['status']='excluded_duplicate_executable'
            else:
                x=vector(d,ds.names);seen.update(hashes);xs.append(x);ids.append(item['path'])
                row.update(status='evaluated',active_features=int(x.sum()),fit_vector_match=bytes(x) in fit_vectors)
                row['available_fields']={k:k in d for k in ['strings','signatures','dropped']}
                row['behavior_fields']=sorted(d['behavior'])
        except (ValueError,KeyError,TypeError,AttributeError) as error:
            row.update(status='unprocessed',reason=str(error))
        audit.append(row)
    save(HERE/'cohort_audit.json',audit)
    if not xs:
        raise ValueError('No eligible reports; no performance estimate')
    X=np.asarray(xs,dtype=np.int8);pred={};scores={}
    matched={r['path'] for r in audit if r.get('fit_vector_match')}
    unique_mask=np.array([p not in matched for p in ids])
    for label,item in protocol['models'].items():
        model=joblib.load(ROOT/item['path'])
        probabilities=model.probabilities(X)['X-ERD']
        assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
        pred[label]=probabilities
        n=len(X);k=int((probabilities>=protocol['threshold']).sum())
        nv=int(unique_mask.sum());kv=int((probabilities[unique_mask]>=protocol['threshold']).sum())
        scores[label]={'n':n,'alert_n':k,'label_relative_fpr':k/n,'binomial_ci95':interval(k,n),
                       'fit_vector_novel_sensitivity':{'n':nv,'alert_n':kv,'label_relative_fpr':kv/nv if nv else None,'binomial_ci95':interval(kv,nv)}}
    pd.DataFrame({'source_path':ids,**pred}).to_csv(HERE/'predictions.csv',index=False)
    pd.concat([pd.DataFrame({'source_path':ids}),pd.DataFrame(X,columns=ds.columns)],axis=1).to_csv(HERE/'vectors.csv',index=False)
    evaluated=[r for r in audit if r['status']=='evaluated']
    vtpositive=sum(isinstance(r.get('antivirus_positive_count'),(int,float)) and r['antivirus_positive_count']>0 for r in evaluated)
    out={'protocol_sha256':digest(HERE/'protocol.json'),'source_script_sha256':digest(Path(__file__)),
         'source_commit':protocol['commit'],'selected_n':len(protocol['files']),
         'status_counts':dict(collections.Counter(r['status'] for r in audit)),
         'models':scores,'environment':environment(),'runtime_seconds':time.time()-start,
         'verified_independent_human_labels':False,'verified_encryption_onset':False,
         'caveat':'Labels derive from a sandbox score, not an independently verified benign ground truth. Rates are label-relative alerts; acquisition platform differs, but broad deployment generalization is not established.',
         'earlier_sequence_reports_without_target_hash':previous_missing,
         'antivirus_positive_evaluated_reports':vtpositive,
         'zero_feature_vectors':int((X.sum(axis=1)==0).sum()),
         'distinct_vectors':len({bytes(x) for x in X}),
         'fit_vector_matches':len(matched),
         'ci_caveat':'Binomial intervals assume independent executions; application-family dependence may make them optimistic.',
         'categories':{g:{'selected_feature_n':sum(n.startswith(g+':') for n in ds.names),'active_values':int(X[:,[n.startswith(g+':') for n in ds.names]].sum())} for g in sorted({n.split(':')[0] for n in ds.names})}}
    save(HERE/'results.json',out)
    print(json.dumps({'status_counts':out['status_counts'],'models':scores,'antivirus_positive_evaluated_reports':vtpositive},indent=2),flush=True)

if __name__=='__main__':
    run()

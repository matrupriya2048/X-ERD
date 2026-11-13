"""Independently reconcile experiment exports and check record isolation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xerd.io import load,digest,save
from xerd.metrics import metrics
from xerd.external_dataset import source_data
from xerd.external_mapping import summarize
from xerd.external_raw import interval

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended'
def read(name):return json.loads((OUT/(name+'.json')).read_text())

def csv(path):
    # Preserve IEEE754 ties when reconstructing rank-based metrics from exports.
    return pd.read_csv(path,float_precision='round_trip')

def main():
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    def reconcile(label,y,p,stored):
        actual=metrics(np.asarray(y),np.asarray(p))
        for key,value in actual.items():
            ok=stored[key] is None if value is None else np.isclose(value,stored[key],atol=1e-12,rtol=1e-10)
            check(label+'/'+key,ok)
    ds=load(ROOT/'data/mlran')
    for stage in ['published','external_dataset','external_mapping','sequences','optimization','group_cv','optimized_validation','residual','residual_validation','external_comparators']:
        check(stage+' completed',read(stage+'_execution')['status']=='complete')
    for name in ['reference_sources','external_data_sources']:
        manifest=json.loads((ROOT/'config'/f'{name}.json').read_text())
        check(name+' SHA256',all(digest(ROOT/r['path'])==r['sha256'] for r in manifest['files']))
    p=csv(OUT/'published_predictions.csv')
    check('published test ID and true labels',np.array_equal(p.sample_id,ds.test.sample_id) and np.array_equal(p.y_true,ds.yt))
    for model,v in read('published')['models'].items():reconcile('published/'+model,p.y_true,p[model],v['metrics'])
    X,y,names,meta=source_data(ROOT/'data/external_sources');spl=csv(OUT/'elderan_split.csv')
    dedup=csv(OUT/'elderan_dedup_audit.csv');keep=meta.sort_values('ID').drop_duplicates('MD5',keep='first').index
    check('EldeRan deterministic hash deduplication',set(dedup.loc[dedup.kept,'source_id'])==set(meta.iloc[keep].ID))
    meta=meta.iloc[keep].reset_index(drop=True);idx=meta.set_index('ID')
    check('EldeRan original IDs unique',not meta.ID.duplicated().any())
    check('EldeRan source executable MD5 unique',not meta.MD5.str.lower().duplicated().any())
    check('EldeRan identical vectors isolated',spl.groupby('vector_group')['split'].nunique().max()==1)
    check('EldeRan complete split',set(spl.source_id)==set(meta.ID))
    for key,name in [('within_collection','elderan_holdout_predictions'),('transfer','elderan_transfer_predictions')]:
        pred=csv(OUT/(name+'.csv'))
        check('EldeRan true target/'+key,np.array_equal(idx.loc[pred.source_id].Ransomware,pred.y_true))
        if key=='within_collection':check('EldeRan test IDs',set(pred.source_id)==set(spl.loc[spl.split=='test','source_id']))
        else:
            old=set(ds.metadata.md5.astype(str).str.lower())
            expected=set(meta.loc[~meta.MD5.str.lower().isin(old),'ID'])
            check('EldeRan transfer complete MD5-disjoint complement',set(pred.source_id)==expected)
        for model,v in read('external_dataset')[key]['models'].items():reconcile('EldeRan/'+key+'/'+model,pred.y_true,pred[model],v)
    a=csv(OUT/'sequence_source_audit.csv');f=csv(OUT/'sequence_folds.csv');p=csv(OUT/'sequence_oof_predictions.csv')
    check('sequence80 original source units',len(a)==80 and not a.source_file.duplicated().any())
    check('sequence folds cover each original file once',len(f)==80 and set(f.source_file)==set(a.source_file) and not f.source_file.duplicated().any())
    check('sequence duplicate grouping',a.merge(f).groupby('first512_sha256').fold.nunique().max()==1)
    for (model,seed),g in p.groupby(['model','seed']):
        check('OOF coverage/'+model+'/'+str(seed),len(g)==80 and not g.source_file.duplicated().any())
        check('OOF labels/'+model+'/'+str(seed),np.array_equal(g.set_index('source_file').loc[a.source_file].y_true,a.y_true))
        reconcile('sequence/'+model+'/'+str(seed),g.y_true,g.probability,read('sequences')['oof_models_by_seed'][model][str(seed)])
    logs=list(OUT.glob('sequence_*_training.json'));check('30 genuine neural fits',len(logs)==30)
    for log in logs:
        v=json.loads(log.read_text());tr,va,final,te=map(set,[v['inner_train_indices'],v['inner_validation_indices'],v['final_train_indices'],v['test_indices']])
        check(log.stem+' isolation',not(tr&va or final&te) and tr|va==final and final|te==set(range(80)))
        check(log.stem+' epoch selected from validation',v['best_epoch'] in [h['epoch'] for h in v['history']] and v['parameter_count']>10000)
    p=csv(OUT/'sequence_transfer_predictions.csv')
    for model,v in read('sequences')['transfer'].items():reconcile('sequence transfer/'+model,p.y_true,p[model],v)
    p=csv(OUT/'sequence_prefix_predictions.csv')
    for v in read('sequences')['prefix_results']:
        g=p[(p.model==v['model'])&(p.seed==v['seed'])&(p.calls.astype(str)==v['calls'])]
        check('prefix coverage/'+v['model']+'/'+str(v['seed'])+'/'+v['calls'],len(g)==80)
        reconcile('prefix/'+v['model']+'/'+str(v['seed'])+'/'+v['calls'],g.y_true,g.probability,v['metrics'])
    m=read('mapping_external');check('mapping has no fabricated human audit',m['human_expert_review'] is False and read('mapping_external_protocol')['human_expert_review'] is False)
    for key,rows in [('cuckoo',m['cuckoo']['reports']),('family_profiles',m['family_profiles']['families'])]:
        check('mapping aggregation/'+key,summarize(rows)==m[key]['summary'])
        for row in rows:
            check('mapping set intersection/'+key+'/'+str(row.get('sample_id',row.get('family'))),len(set(row['predicted'])&set(row['external_annotations']))==row['intersection_n'])
    opt=read('optimization');sel=read('optimization_training_selection');tp=csv(OUT/'optimization_training_predictions.csv')
    check('optimization uses official training IDs only',np.array_equal(tp.sample_id,ds.train.sample_id) and np.array_equal(tp.y_true,ds.y))
    for v in sel['scores']:reconcile('tuning/'+v['candidate']+'/'+str(v['C']),tp.y_true,tp[v['candidate']+'__C'+str(v['C'])],v)
    eligible=[r for r in sel['scores'] if r['recall']>=.97]
    winner=max(eligible,key=lambda r:(r['accuracy'],-r['fpr'])) if eligible else max(sel['scores'],key=lambda r:(r['recall'],r['accuracy']))
    check('tuning winner follows declared train-only rule',winner==sel['selected']==opt['selection'])
    for row in read('optimization_fold_log'):
        a,b=set(row['train_ids']),set(row['validation_ids'])
        check('tuning fresh outer isolation/'+row['candidate']+'/'+str(row['fold']),not(a&b) and a|b==set(ds.train.sample_id) and len(row['inner_folds'])==len(a))
        hashes=ds.metadata.set_index('sample_id').md5.str.lower()
        check('tuning executable isolation/'+row['candidate']+'/'+str(row['fold']),not(set(hashes.loc[list(a)])&set(hashes.loc[list(b)])))
        inner=pd.DataFrame({'hash':hashes.loc[row['train_ids']].to_numpy(),'fold':row['inner_folds']})
        check('tuning inner group isolation/'+row['candidate']+'/'+str(row['fold']),inner.groupby('hash').fold.nunique().max()==1)
    folds=read('group_cv_folds');check('50 grouped sensitivity folds',len(folds)==50)
    for row in folds:
        a,b=set(row['train_ids']),set(row['test_ids']);hashes=ds.metadata.set_index('sample_id').md5.str.lower()
        check('grouped outer isolation/'+str(row['seed'])+'/'+str(row['fold']),not(a&b) and a|b==set(ds.train.sample_id) and not(set(hashes.loc[list(a)])&set(hashes.loc[list(b)])))
        inner=pd.DataFrame({'hash':hashes.loc[row['train_ids']].to_numpy(),'fold':row['inner_folds']})
        check('grouped inner isolation/'+str(row['seed'])+'/'+str(row['fold']),inner.groupby('hash').fold.nunique().max()==1)
    p=csv(OUT/'optimized_test_predictions.csv')
    check('optimized official test unchanged',np.array_equal(p.sample_id,ds.test.sample_id) and np.array_equal(p.y_true,ds.yt))
    for model,v in opt['models'].items():reconcile('optimized/'+model,p.y_true,p[model],v)
    residual=read('residual');sel=read('residual_training_selection');tp=csv(OUT/'residual_training_predictions.csv')
    check('residual training partition',np.array_equal(tp.sample_id,ds.train.sample_id) and np.array_equal(tp.y_true,ds.y))
    for row in sel['scores']:reconcile('residual tuning/'+row['candidate']+'/'+str(row['C']),tp.y_true,tp[row['candidate']+'__C'+str(row['C'])],row)
    eligible=[r for r in sel['scores'] if r['recall']>=.97]
    win=max(eligible,key=lambda r:(r['accuracy'],-r['fpr'])) if eligible else max(sel['scores'],key=lambda r:(r['recall'],r['accuracy']))
    check('residual follows declared selection rule',win==sel['selected']==residual['selection'])
    for row in read('residual_fold_log'):
        a,b=set(row['train_ids']),set(row['validation_ids']);hashes=ds.metadata.set_index('sample_id').md5.str.lower()
        check('residual outer isolation/'+row['candidate']+'/'+str(row['fold']),not(a&b) and a|b==set(ds.train.sample_id) and not(set(hashes.loc[list(a)])&set(hashes.loc[list(b)])))
        frame=pd.DataFrame({'hash':hashes.loc[row['train_ids']].to_numpy(),'fold':row['inner_folds']})
        check('residual inner grouping/'+row['candidate']+'/'+str(row['fold']),frame.groupby('hash').fold.nunique().max()==1)
    p=csv(OUT/'residual_test_predictions.csv')
    check('residual official test unchanged',np.array_equal(p.sample_id,ds.test.sample_id) and np.array_equal(p.y_true,ds.yt))
    for name,v in residual['models'].items():reconcile('residual/'+name,p.y_true,p[name],v)
    from scipy.special import expit
    with np.load(OUT/'residual_validation/shap_values.npz') as z:
        check('residual persisted explanation reconstructs score',np.max(abs(expit(z['baseline']+z['stack_logit'].sum(1))-p['X-ERD']))<1e-4)
        check('residual feature and tree contributions reconcile',np.max(abs(z['stack_logit']-z['raw_component']-z['tree_component']))<1e-10)
    with np.load(OUT/'optimized_validation/shap_values.npz') as z:
        p=csv(OUT/'optimized_test_predictions.csv')
        check('optimized persisted explanation reconstructs score',np.max(abs(expit(z['stack_logit_baseline']+z['stack_logit'].sum(1))-p['X-ERD']))<1e-4)
    ids=set(csv(OUT/'elderan_transfer_predictions.csv').source_id)
    for sub in ['optimized_validation','residual_validation']:
        p=csv(OUT/sub/'transfer_predictions.csv');r=json.loads((OUT/sub/'transfer.json').read_text())
        check(sub+' same transfer targets',set(p.source_id)==ids)
        check(sub+' true target labels',np.array_equal(idx.loc[p.source_id].Ransomware,p.y_true))
        for name,v in r['models'].items():reconcile(sub+'/'+name,p.y_true,p[name],v)
    p=csv(OUT/'external_comparators_predictions.csv');check('published external same target IDs',set(p.source_id)==ids)
    check('published external true target labels',np.array_equal(idx.loc[p.source_id].Ransomware,p.y_true))
    for name,v in read('external_comparators')['models'].items():reconcile('published external/'+name,p.y_true,p[name],v)
    # Validate the full raw acquisition and every exclusion before accepting FPR.
    check('external_raw completed',read('external_raw_execution')['status']=='complete')
    manifest=json.loads((ROOT/'config/benign_report_sources.json').read_text())
    check('200 verified benign report sources',len(manifest['files'])==200 and all(digest(ROOT/r['path'])==r['sha256'] for r in manifest['files']))
    raw=read('external_raw');audit=read('external_raw_audit');p=csv(OUT/'external_raw_predictions.csv');valid=[x for x in audit if x['status']=='evaluated']
    check('raw audit accounts for all200',len(audit)==200 and sum(raw['status_counts'].values())==200)
    check('raw unique executable units',len(valid)==len({x['executable_hash'] for x in valid})==len(p)==raw['evaluated_unique_n'])
    check('raw prediction IDs match accepted reports',set(p.source_path)=={x['source_path'] for x in valid})
    check('raw publisher-clean target labels',set(p.y_true)=={0})
    source_hashes={str(v).lower() for col in ['md5','sha1','sha256'] for v in ds.metadata[col].dropna()}
    accepted_hashes=set()
    for row in valid:
        data=json.loads((ROOT/row['source_path']).read_text())
        file=data['target']['file'];hs={str(file[k]).lower() for k in ['md5','sha1','sha256'] if file.get(k)}
        check('raw executable isolation/'+row['source_path'],bool(hs) and not(hs&source_hashes or hs&accepted_hashes))
        accepted_hashes.update(hs)
    for model,v in raw['models'].items():
        reconcile('raw/'+model,p.y_true,p[model],v)
        check('raw exact interval/'+model,np.allclose(interval(v['fp'],v['n']),v['fpr_ci95']))
    p=csv(OUT/'external_raw_candidates_predictions.csv')
    check('raw candidates identical target IDs',set(p.source_path)=={x['source_path'] for x in valid} and set(p.y_true)=={0})
    for group,rows in read('external_raw_candidates')['models'].items():
        for name,v in rows.items():reconcile('raw candidate/'+group+'/'+name,p.y_true,p[group+'__'+name],v)
    save(OUT/'verification.json',{'status':'passed','checks_count':len(checks),'checks':checks,
      'scope':'Persisted-artifact consistency,actual source integrity and split isolation; not human adjudication or prospective deployment validation'})
    print('PASS',len(checks),'extended checks')

if __name__=='__main__':main()

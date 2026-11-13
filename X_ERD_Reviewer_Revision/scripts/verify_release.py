"""Cross-check persisted artifacts, data isolation and attribution reconstruction."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    stages=['core','cv','thresholds','corruption','temporal','strict_temporal','ablation','neural','explain','parser','mapping','operational','figures']
    for n in stages:check('stage complete:'+n,read(n+'_execution')['status']=='complete')
    train=pd.read_csv(ROOT/'data/mlran/MLRan_X_train_RFE.csv');test=pd.read_csv(ROOT/'data/mlran/MLRan_X_test_RFE.csv')
    p=pd.read_csv(OUT/'test_predictions.csv');core=read('core');truth=(test.type_label>0).to_numpy(dtype=int)
    check('prediction IDs preserve official test order',np.array_equal(p.sample_id,test.sample_id))
    check('predictions use true ransomware target',np.array_equal(p.y_true,truth))
    for n,r in core['metrics'].items():
        a=p[n].to_numpy()>=.5
        cm=[int(((truth==0)&~a).sum()),int(((truth==0)&a).sum()),int(((truth==1)&~a).sum()),int(((truth==1)&a).sum())]
        check('confusion/prediction reconciliation:'+n,cm==[r[k] for k in ['tn','fp','fn','tp']])
    cv=read('cv_folds');seeds=[0,1,2,3,4,5,42,123,256,999]
    check('exact50 distinct declared outer folds',{(r['seed'],r['fold']) for r in cv['rows']}=={(s,f) for s in seeds for f in range(5)} and len(cv['rows'])==50)
    check('CV source hashes are current',all(digest(ROOT/'src/xerd'/name)==value for name,value in cv['source_hashes'].items()))
    for r in cv['rows']:
        a,b=set(r['train_ids']),set(r['validation_ids'])
        check(f'outer ID isolation:{r["seed"]}/{r["fold"]}',not(a&b) and a|b==set(train.sample_id))
    metadata=pd.read_csv(ROOT/'data/mlran/mlran_dataset_metadata.csv').set_index('sample_id');temp=read('temporal_strict')
    a,b,g=map(set,[temp['train_ids'],temp['test_ids'],temp['gap_2020_ids']])
    check('strict temporal ID separation',not(a&b or a&g or b&g) and a|b|g==set(pd.concat([train,test]).sample_id))
    check('no future training record',metadata.loc[list(a),'first_submission_date_year'].max()<=2019)
    check('test begins2021',metadata.loc[list(b),'first_submission_date_year'].min()>=2021)
    check('gap is2020',set(metadata.loc[list(g),'first_submission_date_year'])=={2020})
    for n,r in temp['metrics'].items():
        for key in ['tn','fp','fn','tp']:check('year/aggregate reconciliation:'+n+'/'+key,sum(v[n][key] for v in temp['years'].values())==r[key])
    sel=pd.read_csv(OUT/'temporal_selection.csv');check('strict temporal selection483',int(sel.selected.sum())==483)
    ex=read('explanations')
    with np.load(OUT/'shap_values.npz') as z:
        probability=expit(float(z['stack_logit_baseline'])+z['stack_logit'].sum(1))
        check('persisted stack SHAP reconstructs persisted score',np.max(abs(probability-p['X-ERD'].to_numpy()))<1e-4)
    check('base SHAP errors bounded',max(ex['max_base_additivity_errors'].values())<1e-4)
    parser=read('parser');check('parser report IDs were excluded',parser['evaluation_reports_excluded_from_training'] and parser['training_n']==len(train)-len(parser['sample_ids']))
    check('all evaluation reports are in original training partition',set(parser['sample_ids'])<=set(train.sample_id))
    check('parser reports are ransomware only',parser['benign_n']==0 and parser['ransomware_n']==20)
    check('parser FPR remains undefined',all(r['parsed']['fpr'] is None for r in parser['models']))
    check('all eight feature categories disclosed',set(parser['all_categories'])=={'API','DIRECTORY','DROP','FILE','REG','SIGNATURE','STRING','SYSTEM'})
    thresholds=read('thresholds');check('same threshold policy for every benchmark model',set(thresholds)==set(core['metrics']))
    for name,r in thresholds.items():check('validation recall constraint:'+name,r['validation_constrained']['recall']>=.97)
    mapping=read('mapping');audit=pd.read_csv(OUT/'independent_mapping_audit_blank.csv',keep_default_na=False)
    check('blank audit is not invented expert evidence',len(audit)==30 and (audit.reference_technique=='').all() and (audit.assessor_id=='').all())
    check('mapping technique count is distinct',len(set(x['technique_id'] for x in mapping['all_feature_mappings'].values() if x['technique_id']))==len(mapping['distinct_techniques']))
    check('mapping specificity score available',all('rule_specificity_score' in r for r in mapping['all_feature_mappings'].values()))
    report={'status':'passed','checks_count':len(checks),'checks':checks,
       'scope':'local scientific consistency and reproducibility checks; not independent validation,publication or submission clearance'}
    (OUT/'release_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS:',len(checks),'persisted-artifact checks')

if __name__=='__main__':main()

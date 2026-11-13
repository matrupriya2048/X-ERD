"""Recompute adaptation claims from frozen models, predictions and source receipts."""
from pathlib import Path
import datetime,json
import joblib
import numpy as np
import pandas as pd
from xerd.io import load,save,digest
from xerd.metrics import metrics
from xerd.remediation import select
from xerd.external_raw import interval

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/remediation'

def main():
    checks=[]
    def check(v,label):
        if not v:raise AssertionError(label)
        checks.append(label)
    def near(a,b,label):
        check(a is None and b is None or a is not None and b is not None and abs(float(a)-float(b))<1e-12,label)
    protocol=json.loads((ROOT/'config/raw_remediation_protocol.json').read_text());frozen=json.loads((OUT/'frozen_model.json').read_text())
    check(digest(ROOT/'config/raw_remediation_protocol.json')==frozen['protocol_sha256'],'protocol frozen unchanged')
    check(digest(ROOT/'src/xerd/remediation.py')==frozen['source_sha256'],'training source frozen unchanged')
    check(digest(OUT/'development_vectors.csv')==frozen['development_vectors_sha256'],'development matrix frozen unchanged')
    check(digest(OUT/'model.joblib')==frozen['model_sha256'],'model frozen unchanged')
    old=json.loads((ROOT/'config/benign_report_sources.json').read_text());new=json.loads((ROOT/'config/raw_remediation_holdout_sources.json').read_text())
    check(len(protocol['holdout_selection'])==400 and not set(protocol['holdout_selection'])&set(old['selection']),'400 new paths not used for previous 200-report evaluation')
    check(set(protocol['holdout_selection'])=={f['archive_member'] for f in new['files']},'acquired exactly selected holdout paths')
    ds=load(ROOT/'data/mlran');dev=pd.read_csv(OUT/'development_vectors.csv');oof=pd.read_csv(OUT/'training_oof_predictions.csv',float_precision='round_trip')
    check(len(dev)==182 and len(oof)==4087,'declared development counts')
    selection=json.loads((OUT/'selection.json').read_text());check(len(selection['candidates'])==18,'all 18 candidates disclosed')
    for row in selection['candidates']:
        for domain,label in [(0,'source'),(1,'external_development')]:
            sub=oof[oof.domain==domain];calc=metrics(sub.y_true.to_numpy(),sub[row['key']].to_numpy())
            for key,value in calc.items():near(value,row[label][key],row['key']+' '+label+' '+key)
    winner,feasible=select(selection['candidates']);check(winner==selection['selected'] and feasible==selection['source_constraints_feasible'],'declared selection rule reproduced')
    check(winner==frozen['selection'],'selected model matches freeze')
    logs=json.loads((OUT/'training_fold_log.json').read_text());check(len(logs)==15,'three weights by five outer folds logged')
    lookup=oof.set_index('id');allids=set(oof.id)
    for r in logs:
        train=lookup.loc[r['train_ids']];valid=lookup.loc[r['validation_ids']]
        check(not set(train.md5)&set(valid.md5),'outer group independence '+str((r['weight'],r['fold'])))
        inner=np.asarray(r['inner_folds']);check(set(inner)==set(range(5)) and len(inner)==len(train),'complete inner group split '+str((r['weight'],r['fold'])))
        for k in range(5):check(not set(train.md5.to_numpy()[inner==k])&set(train.md5.to_numpy()[inner!=k]),'inner executable independence '+str((r['weight'],r['fold'],k)))
        expected=allids-set(valid.index)
        if r['weight']==0:expected={i for i in expected if lookup.loc[i,'domain']==0}
        check(set(train.index)==expected,'all permitted training rows included '+str((r['weight'],r['fold'])))
    for w in protocol['development_weights']:
        valid=[i for r in logs if r['weight']==w for i in r['validation_ids']]
        check(len(valid)==len(allids) and set(valid)==allids,'each record validated exactly once '+str(w))
    models={'Original full stack':joblib.load(ROOT/'results/model.joblib'),'Previous passthrough':joblib.load(ROOT/'results/extended/residual_model.joblib'),'Development-adapted':joblib.load(OUT/'model.joblib')}
    official=pd.read_csv(OUT/'official_test_predictions.csv',float_precision='round_trip');summary=json.loads((OUT/'official_test.json').read_text())
    for n,p in models['Development-adapted'].probabilities(ds.Xt).items():
        check(np.allclose(p,official[n],rtol=0,atol=1e-14),'official predictions '+n)
        for k,v in metrics(ds.yt,p).items():near(v,summary['models'][n][k],'official metric '+n+' '+k)
    for name,truth,endpoint in [('holdout',0,'fpr'),('ransomware',1,'recall')]:
        vectors=pd.read_csv(OUT/(name+'_vectors.csv'));pred=pd.read_csv(OUT/(name+'_predictions.csv'),float_precision='round_trip');result=json.loads((OUT/(name+'.json')).read_text());audit=json.loads((OUT/(name+'_audit.json')).read_text())
        check(list(vectors.source_path)==list(pred.source_path),'row alignment '+name)
        check(set(pred.source_path)=={r['source_path'] if 'source_path' in r else r['path'] for r in audit if r['status']=='evaluated'},'accepted audit matches predictions '+name)
        check(result['model_sha256']==frozen['model_sha256'],'same frozen model '+name)
        check(datetime.datetime.fromisoformat(result['evaluated_utc'])>datetime.datetime.fromisoformat(frozen['frozen_utc'])>datetime.datetime.fromisoformat(protocol['created_utc']),'freeze before evaluation '+name)
        check(len(pred)==result['evaluated_unique_n'] and (pred.y_true==truth).all(),'endpoint counts '+name)
        for label,m in models.items():
            for n,p in m.probabilities(vectors[ds.columns].to_numpy()).items():
                check(np.allclose(p,pred[label+'__'+n],rtol=0,atol=1e-14),'predictions '+name+' '+label+' '+n)
                for k,v in metrics(pred.y_true.to_numpy(),p).items():near(v,result['models'][label][n][k],'metric '+name+' '+label+' '+n+' '+k)
                check(np.allclose(interval(int((p>=.5).sum()),len(p)),result['models'][label][n][endpoint+'_ci95'],rtol=0,atol=1e-14),'exact endpoint interval '+name+' '+label+' '+n)
    for filename in ['raw_remediation_holdout_sources','zirari_raw_sources']:
        for f in json.loads((ROOT/'config'/(filename+'.json')).read_text())['files']:
            check(digest(ROOT/f['path'])==f['sha256'],'raw report bytes '+f['path'])
    exp=json.loads((OUT/'explanations.json').read_text());check(exp['n']==975 and exp['maximum_probability_reconstruction_error']<1e-4 and exp['model_sha256']==frozen['model_sha256'],'adapted model exact additive reconstruction')
    save(OUT/'verification.json',{'status':'passed','checks':len(checks),'details':checks,'model_sha256':frozen['model_sha256']});print('Remediation verification passed',len(checks),flush=True)

if __name__=='__main__':main()

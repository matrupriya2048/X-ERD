from pathlib import Path
import concurrent.futures,json,collections
import numpy as np
import pandas as pd
import joblib
from xerd.io import load,save,digest
from xerd.parser import vector
from xerd.external_raw import interval
from primary_pair import probabilities
from cape_adapter import adapt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'closure_results'

def run():
    ds=load(ROOT/'data/mlran');rows=[r for r in json.loads((OUT/'en2025_audit.json').read_text()) if r['status']=='evaluated']
    save(OUT/'cape_adapter_amendment.json',{'stage':'Post-hoc schema compatibility repair after untranslated EN2025 results were inspected.',
         'cohort':'Same previously scored 917 eligible PE targets; no replacements or relabeling.',
         'label_or_score_based_rules':False,'model_fitting':False,'source':'https://github.com/kevoreilly/CAPEv2/blob/master/modules/processing/behavior.py',
         'adapter_sha256':digest(HERE/'cape_adapter.py'),'rules':'Typed enhanced events to matching summary operations; nested static strings to root strings; one layer of path-separator escaping removed. Unknown events remain unsupported.',
         'scope':'Paired input-schema sensitivity, not a new prospective validation or verified official-vector fidelity test.'})
    def process(row):
        path=HERE/'data_en2025/raw'/row['path'];assert digest(path)==row['report_sha256']
        d=json.loads(path.read_text(encoding='utf-8'));canonical,receipt=adapt(d)
        return vector(canonical,ds.names),{'source_path':row['path'],**receipt}
    xs=[];audit=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for x,receipt in executor.map(process,rows):
            xs.append(x);audit.append(receipt)
            if len(xs)%100==0:print('CAPE translation scored',len(xs),'/',len(rows),flush=True)
    X=np.asarray(xs);models={'primary_RF+LGB':joblib.load(OUT/'primary_pair.joblib'),'original_three':joblib.load(ROOT/'results/model.joblib'),'adapted_three':joblib.load(ROOT/'results/remediation/model.joblib')}
    pred={k:probabilities(m,X) if k=='primary_RF+LGB' else m.probabilities(X)['X-ERD'] for k,m in models.items()}
    ids=[r['path'] for r in rows];pd.DataFrame({'source_path':ids,**pred}).to_csv(OUT/'en2025_translated_predictions.csv',index=False)
    pd.concat([pd.DataFrame({'source_path':ids}),pd.DataFrame(X,columns=ds.columns)],axis=1).to_csv(OUT/'en2025_translated_vectors.csv',index=False)
    save(OUT/'en2025_translation_audit.json',audit)
    direct=pd.read_csv(OUT/'en2025_vectors.csv');assert ids==direct.source_path.tolist()
    result={'n':len(X),'models':{k:{'n':len(X),'alert_n':int((p>=.5).sum()),'fpr':float((p>=.5).mean()),'ci95':interval(int((p>=.5).sum()),len(X))} for k,p in pred.items()},
        'changed_vectors':int((X!=direct[ds.columns].to_numpy()).any(1).sum()),'strings_available_n':sum(r['strings_source']!='unavailable' for r in audit),
        'distinct_vectors':len({bytes(x) for x in X}),
        'categories':{g:{'selected_feature_n':sum(s.startswith(g+':') for s in ds.names),'active_values':int(X[:,[s.startswith(g+':') for s in ds.names]].sum())} for g in sorted({s.split(':')[0] for s in ds.names})},
        'adapter_sha256':digest(HERE/'cape_adapter.py'),'script_sha256':digest(Path(__file__)),
        'scope':'Post-hoc schema translation on the same EN2025 cohort; predictions remain from frozen classifiers. Both untranslated and translated results retained. No independent gold vectors or verified onset.'}
    save(OUT/'en2025_translated_results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
